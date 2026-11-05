//! Rigid bodies synchronized with a Hyperon runtime world, solved with
//! Augmented Vertex Block Descent (Giles et al., SIGGRAPH 2025).
//!
//! AVBD solves each body's 6-DOF position as a small block Newton step against
//! every contact touching it, with augmented-Lagrangian contact forces whose
//! penalties ramp up only as far as needed. That makes stacks and large mass
//! ratios stable at a few iterations and no substeps.
//!
//! From the paper: hard constraints (Sec. 3.1), force bounds with stiffness
//! rescaling where needed (3.2, Eq. 14), frictional contacts with static and
//! kinetic friction (3.3, 5), finite stiffness ramped up to its value (3.4,
//! Eq. 16), diagonal geometric stiffness (3.5), warm starting (3.7), bounded
//! duals and step-start Taylor contacts (4). [`Joint`] and [`Spring`] follow
//! the authors' 3D demo, with joint fracture as a torque.
//!
//! From the authors' reference solver rather than the paper: post-stabilization
//! in place of Eq. 18's regularization. The last of the iterations corrects
//! step-start error after velocities are taken, so it adds no energy.
//!
//! Deliberate departures: penalties and β are per kilogram so tuning is
//! unit-free, a cone-limited contact uses the secant of its capped force as
//! its tangential stiffness (as NVIDIA Newton's AVBD does), rotations conserve
//! angular momentum by an explicit midpoint step, and bodies sleep in islands.
//! Contacts are speculative, generated at the pose where a body's predicted path
//! first meets another, so fast bodies stop at surfaces and pass what they miss.
//! Between two free bodies the penalty is limited to their inertial stiffness
//! and the dual updates at every visit, without which momentum is lost; in an
//! impact their velocities come from the contact forces, which are equal and
//! opposite even before the solve converges.
//! `examples/quality.rs` measures these against closed-form results.
//!
//! Bodies sharing a contact are graph-colored; every body of one color is
//! updated in parallel, which is the same structure a GPU port dispatches.
//! The solver runs on CPU threads: game-sized scenes are latency bound, need
//! results the same frame, and on mobile the GPU belongs to the renderer.

mod collide;
mod link;
mod solver;

use glam::{Mat3, Quat, Vec3, Vec3A};
use hyperon_core::{EntityHandle, Transform, World};
use std::collections::HashMap;

/// Solver parameters. The caller supplies the fixed timestep to [`PhysicsWorld::step`].
#[derive(Debug, Clone)]
pub struct PhysicsSettings {
    /// Z is up, as in Hyperon scenes.
    pub gravity: Vec3,
    /// Block descent sweeps per step. All but the last solve the step, and
    /// the last is the stabilization sweep, so at least two are needed for it.
    pub iterations: u32,
    /// Fraction of the starting contact error left uncorrected each step, in
    /// [0, 1). Correction moves positions without adding velocity, so zero
    /// is the default. Higher is smoother.
    pub alpha: f32,
    /// Penalty growth per metre of contact or joint error, per kilogram.
    pub beta: f32,
    /// Penalty growth per radian of joint angle error, per kilogram: angular
    /// rows have other units, so they ramp separately (as in the AVBD demo).
    pub beta_angular: f32,
    /// Per-step decay of warm-started forces and penalties, in (0, 1).
    pub gamma: f32,
    /// Contact damping time k_d in seconds (VBD's Rayleigh damping, applied
    /// to pushing contact normals): stacks stop bouncing and sleep sooner.
    /// Zero disables it.
    pub contact_damping: f32,
}

impl Default for PhysicsSettings {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, 0.0, -9.81),
            // The paper's γ (Table 2), and the reference solver's post-stabilized
            // α, which removes all error from positions. β is per kilogram here, so its
            // value differs from the paper's 10; it and the iteration count
            // were tuned on box pyramids, the slowest case for local solvers.
            iterations: 6,
            alpha: 0.0,
            beta: 100_000.0,
            beta_angular: 100.0,
            gamma: 0.99,
            // Tuned on box pyramids: from 0.0005 to 0.001 s they sleep soonest;
            // more damping slows convergence and they settle later again.
            contact_damping: 0.0007,
        }
    }
}

/// A body in world space. Transform scale is visual only; size the collider itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RigidBody {
    #[default]
    Static,
    /// Moved only by its velocity; pushes dynamic bodies and carries them by friction.
    Kinematic,
    Dynamic,
}

/// One centered collider per body, in metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Collider {
    Cuboid { half_extents: Vec3 },
    Sphere { radius: f32 },
}

impl Collider {
    /// Mass and body-space principal moments of inertia for a density.
    fn mass_properties(self, density: f32) -> (f32, Vec3) {
        match self {
            Self::Cuboid { half_extents: h } => {
                assert!(
                    h.is_finite() && h.min_element() > 0.0,
                    "cuboid half extents must be finite and positive"
                );
                let mass = 8.0 * h.x * h.y * h.z * density;
                let h2 = h * h;
                let moment = Vec3::new(h2.y + h2.z, h2.x + h2.z, h2.x + h2.y) * (mass / 3.0);
                (mass, moment)
            }
            Self::Sphere { radius } => {
                assert!(
                    radius.is_finite() && radius > 0.0,
                    "sphere radius must be finite and positive"
                );
                let mass = 4.0 / 3.0 * std::f32::consts::PI * radius.powi(3) * density;
                (mass, Vec3::splat(0.4 * mass * radius * radius))
            }
        }
    }
}

/// Density in kg/m³ and friction coefficients. Density must be positive and
/// the coefficients nonnegative; all finite. Contacts use geometric means.
/// A contact that held last frame uses `static_friction`, and switches to
/// the kinetic `friction` once it has to slide (AVBD Sec. 3.3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicsMaterial {
    pub density: f32,
    pub friction: f32,
    pub static_friction: f32,
}

impl Default for PhysicsMaterial {
    fn default() -> Self {
        Self {
            density: 1000.0,
            friction: 0.6,
            static_friction: 0.6,
        }
    }
}

/// World-space linear velocity (m/s) and angular velocity (rad/s), updated by
/// every step.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Velocity {
    pub linear: Vec3,
    pub angular: Vec3,
}

/// Persistent force (N) at the center of mass and torque (N·m), in world space.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ExternalForce {
    pub force: Vec3,
    pub torque: Vec3,
}

/// One-shot linear (N·s) and angular (N·m·s) impulses, in world space.
/// Zeroed after the step that applies them.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ExternalImpulse {
    pub linear: Vec3,
    pub angular: Vec3,
}

/// Physics properties associated with one Hyperon entity.
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicsBody {
    pub body_type: RigidBody,
    pub collider: Collider,
    pub material: PhysicsMaterial,
    pub velocity: Velocity,
    pub force: ExternalForce,
    pub impulse: ExternalImpulse,
}

impl PhysicsBody {
    pub fn new(body_type: RigidBody, collider: Collider) -> Self {
        Self {
            body_type,
            collider,
            material: PhysicsMaterial::default(),
            velocity: Velocity::default(),
            force: ExternalForce::default(),
            impulse: ExternalImpulse::default(),
        }
    }
}

/// Pins a point of `body_b` to a point of `body_a`, or to a world point when
/// `body_a` is `None`. Infinite stiffness is a hard constraint, solved exactly
/// with an augmented Lagrangian (AVBD Sec. 3.1); a finite one is a soft
/// constraint of that stiffness (Sec. 3.4). Linked bodies do not collide.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Joint {
    pub body_a: Option<EntityHandle>,
    pub body_b: EntityHandle,
    /// Attachment points in each body's frame; `anchor_a` is a world point
    /// without `body_a`.
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    /// N/m holding the anchors together; infinite by default.
    pub linear_stiffness: f32,
    /// Resistance to turning away from the bodies' relative rotation when
    /// the joint was added: 0 (the default) is a ball joint, infinite welds.
    pub angular_stiffness: f32,
    /// Angular constraint torque (N·m) that breaks the joint; never by default.
    pub fracture: f32,
}

impl Joint {
    /// A rigid ball joint.
    pub fn new(
        body_a: Option<EntityHandle>,
        body_b: EntityHandle,
        anchor_a: Vec3,
        anchor_b: Vec3,
    ) -> Self {
        Self {
            body_a,
            body_b,
            anchor_a,
            anchor_b,
            linear_stiffness: f32::INFINITY,
            angular_stiffness: 0.0,
            fracture: f32::INFINITY,
        }
    }
}

/// A spring of finite `stiffness` (N/m) between two anchors, like [`Joint`].
/// Its penalty ramps up to the stiffness over the iterations (AVBD Eq. 16),
/// so a soft spring next to a stiff one still converges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spring {
    pub body_a: Option<EntityHandle>,
    pub body_b: EntityHandle,
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    pub stiffness: f32,
    /// Negative: the anchors' distance when the spring is added.
    pub rest_length: f32,
}

/// Identifies a joint or spring in its [`PhysicsWorld`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LinkId(u64);

/// An entity is not live in the supplied world.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidEntity;

impl std::fmt::Display for InvalidEntity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("entity is not live in this world")
    }
}
impl std::error::Error for InvalidEntity {}

#[derive(Clone, Copy)]
struct CachedMass {
    collider: Collider,
    density: f32,
    mass: f32,
    moment: Vec3,
}

#[derive(Clone, Copy)]
struct CachedPose {
    rotation_bits: [u32; 4],
    rotation: Quat,
    basis: Mat3,
    inertia: Mat3,
}

struct Slot {
    entity: EntityHandle,
    body: PhysicsBody,
    /// Last step's starting velocity, for AVBD's adaptive gravity warm start.
    previous: Vec3,
    asleep: bool,
    island: u32,
    sleep_time: f32,
    /// The transform the body fell asleep at; moving it wakes the body.
    rest: ([f32; 3], [f32; 4]),
    mass_properties: Option<CachedMass>,
    pose_properties: Option<CachedPose>,
}

impl Slot {
    fn new(entity: EntityHandle, body: PhysicsBody) -> Self {
        Self {
            entity,
            previous: body.velocity.linear,
            body,
            asleep: false,
            island: 0,
            sleep_time: 0.0,
            rest: ([0.0; 3], [0.0; 4]),
            mass_properties: None,
            pose_properties: None,
        }
    }
}

/// Owns entity-associated physics data and solver state. Use each instance
/// with one `World` only: handles are world-local.
#[derive(Default)]
pub struct PhysicsWorld {
    slots: Vec<Slot>,
    index: HashMap<EntityHandle, usize>,
    solver: solver::Solver,
    /// Thread pools by log2 size, built on first use. A step forks once per
    /// color per iteration, and waking more threads than the awake bodies
    /// can feed makes small scenes slower than one thread.
    pools: Vec<Option<rayon::ThreadPool>>,
    /// Joints and springs by id, in step with `solver.links`.
    links: Vec<(LinkId, Option<EntityHandle>, EntityHandle)>,
    next_link: u64,
}

/// Awake dynamic bodies per solver thread worth waking.
const BODIES_PER_THREAD: usize = 32;

/// Solver threads beyond the physical cores share a core with another solver
/// thread and stall every barrier.
// shortcut: assumes two hardware threads per core on machines with 8 or more
// logical CPUs; query the OS topology to support non-SMT many-core machines.
static PHYSICAL_CORES: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
    let logical = std::thread::available_parallelism().map_or(1, |n| n.get());
    if logical >= 8 { logical / 2 } else { logical }
});

impl PhysicsWorld {
    pub fn body_count(&self) -> usize {
        self.slots.len()
    }
    pub fn contains(&self, entity: EntityHandle) -> bool {
        self.index.contains_key(&entity)
    }
    pub fn get(&self, entity: EntityHandle) -> Option<&PhysicsBody> {
        self.index.get(&entity).map(|&i| &self.slots[i].body)
    }
    pub fn get_mut(&mut self, entity: EntityHandle) -> Option<&mut PhysicsBody> {
        self.index.get(&entity).map(|&i| &mut self.slots[i].body)
    }
    pub fn iter(&self) -> impl Iterator<Item = (EntityHandle, &PhysicsBody)> {
        self.slots.iter().map(|slot| (slot.entity, &slot.body))
    }
    /// Contacts touching bodies after the last step, sleeping ones included.
    pub fn contact_count(&self) -> usize {
        self.solver.contact_count()
    }

    /// Whether the body rests asleep: it costs nothing until a moving body
    /// touches its island or its velocity, force, impulse or transform changes.
    pub fn is_sleeping(&self, entity: EntityHandle) -> bool {
        self.index
            .get(&entity)
            .is_some_and(|&i| self.slots[i].asleep)
    }

    /// Wake the islands these slots sleep in or that rest on them: they are
    /// going away or were moved.
    fn wake_islands_of(&mut self, slots: Vec<(bool, u32, u64)>) {
        let mut islands: Vec<u32> = (slots.iter())
            .filter(|&&(asleep, _, _)| asleep)
            .map(|&(_, island, _)| island)
            .collect();
        let keys: Vec<u64> = slots.iter().map(|&(_, _, key)| key).collect();
        islands.extend(self.solver.islands_touching(&keys));
        if islands.is_empty() {
            return;
        }
        for slot in &mut self.slots {
            if slot.asleep && islands.contains(&slot.island) {
                slot.asleep = false;
            }
        }
    }

    /// Associate physics with a live entity, replacing existing properties.
    pub fn insert(
        &mut self,
        world: &World,
        entity: EntityHandle,
        body: PhysicsBody,
    ) -> Result<Option<PhysicsBody>, InvalidEntity> {
        if !world.contains(entity) {
            return Err(InvalidEntity);
        }
        if let Some(&i) = self.index.get(&entity) {
            let slot = &mut self.slots[i];
            // Awake, it wakes whatever it touches by contact.
            slot.asleep = false;
            return Ok(Some(std::mem::replace(&mut slot.body, body)));
        }
        self.index.insert(entity, self.slots.len());
        self.slots.push(Slot::new(entity, body));
        Ok(None)
    }

    /// Add a joint between bodies of live entities with physics. The angular
    /// constraint holds their current relative rotation.
    pub fn add_joint(&mut self, world: &World, joint: Joint) -> Result<LinkId, InvalidEntity> {
        assert!(
            joint.anchor_a.is_finite() && joint.anchor_b.is_finite(),
            "joint anchors must be finite"
        );
        assert!(
            joint.linear_stiffness > 0.0 && joint.angular_stiffness >= 0.0 && joint.fracture > 0.0,
            "joint stiffness must be positive (angular nonnegative) and fracture positive"
        );
        let rotation = |e: Option<EntityHandle>| {
            e.map_or(Quat::IDENTITY, |e| {
                Quat::from_array(world.get(e).unwrap().transform.rotation).normalize()
            })
        };
        self.check_link(world, joint.body_a, joint.body_b)?;
        let rest = rotation(joint.body_a).conjugate() * rotation(Some(joint.body_b));
        let kind = link::Kind::Joint {
            linear: joint.linear_stiffness,
            angular: joint.angular_stiffness,
            fracture: joint.fracture,
            rest,
        };
        Ok(self.push_link(
            joint.body_a,
            joint.body_b,
            kind,
            joint.anchor_a,
            joint.anchor_b,
        ))
    }

    /// Add a spring between bodies of live entities with physics.
    pub fn add_spring(&mut self, world: &World, spring: Spring) -> Result<LinkId, InvalidEntity> {
        assert!(
            spring.anchor_a.is_finite() && spring.anchor_b.is_finite(),
            "spring anchors must be finite"
        );
        assert!(
            spring.stiffness.is_finite() && spring.stiffness > 0.0 && !spring.rest_length.is_nan(),
            "spring stiffness must be finite and positive"
        );
        self.check_link(world, spring.body_a, spring.body_b)?;
        let point = |e: Option<EntityHandle>, anchor: Vec3| match e {
            Some(e) => {
                let t = &world.get(e).unwrap().transform;
                Vec3::from_array(t.translation) + Quat::from_array(t.rotation) * anchor
            }
            None => anchor,
        };
        let rest_length = if spring.rest_length < 0.0 {
            point(spring.body_a, spring.anchor_a)
                .distance(point(Some(spring.body_b), spring.anchor_b))
        } else {
            spring.rest_length
        };
        let kind = link::Kind::Spring {
            stiffness: spring.stiffness,
            rest_length,
        };
        Ok(self.push_link(
            spring.body_a,
            spring.body_b,
            kind,
            spring.anchor_a,
            spring.anchor_b,
        ))
    }

    /// Remove a joint or spring; false if it is not in this world.
    pub fn remove_link(&mut self, id: LinkId) -> bool {
        let Some(i) = self.links.iter().position(|&(l, ..)| l == id) else {
            return false;
        };
        let (_, a, b) = self.links.swap_remove(i);
        self.solver.links.swap_remove(i);
        self.wake_bodies(a.into_iter().chain([b]));
        true
    }

    /// Whether a joint broke under its fracture force; `None` for an unknown id.
    pub fn is_broken(&self, id: LinkId) -> Option<bool> {
        let i = self.links.iter().position(|&(l, ..)| l == id)?;
        Some(self.solver.links[i].broken)
    }

    fn check_link(
        &self,
        world: &World,
        a: Option<EntityHandle>,
        b: EntityHandle,
    ) -> Result<(), InvalidEntity> {
        let ok = |e: EntityHandle| world.contains(e) && self.contains(e);
        if !ok(b) || a.is_some_and(|a| !ok(a) || a == b) {
            return Err(InvalidEntity);
        }
        Ok(())
    }

    fn push_link(
        &mut self,
        a: Option<EntityHandle>,
        b: EntityHandle,
        kind: link::Kind,
        anchor_a: Vec3,
        anchor_b: Vec3,
    ) -> LinkId {
        let id = LinkId(self.next_link);
        self.next_link += 1;
        self.links.push((id, a, b));
        self.solver
            .links
            .push(link::Link::new(kind, anchor_a, anchor_b));
        // A new constraint on sleeping bodies must be solved.
        self.wake_bodies(a.into_iter().chain([b]));
        id
    }

    /// Wake these bodies' islands.
    fn wake_bodies(&mut self, entities: impl Iterator<Item = EntityHandle>) {
        let woken: Vec<_> = entities
            .filter_map(|e| self.index.get(&e).map(|&i| &self.slots[i]))
            .map(|s| (s.asleep, s.island, key(s.entity)))
            .collect();
        self.wake_islands_of(woken);
    }

    /// Drop links whose bodies left this physics world.
    fn prune_links(&mut self) {
        let mut i = 0;
        while i < self.links.len() {
            let (_, a, b) = self.links[i];
            if a.is_some_and(|a| !self.index.contains_key(&a)) || !self.index.contains_key(&b) {
                self.links.swap_remove(i);
                self.solver.links.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    pub fn remove(&mut self, entity: EntityHandle) -> Option<PhysicsBody> {
        let i = self.index.remove(&entity)?;
        let slot = self.slots.swap_remove(i);
        if let Some(moved) = self.slots.get(i) {
            self.index.insert(moved.entity, i);
        }
        self.wake_islands_of(vec![(slot.asleep, slot.island, key(slot.entity))]);
        self.prune_links();
        Some(slot.body)
    }

    /// Advance by one positive fixed timestep in seconds, then write moved
    /// transforms and velocities back. Bodies of despawned entities are dropped.
    pub fn step(&mut self, world: &mut World, settings: &PhysicsSettings, dt: f32) {
        assert!(settings.gravity.is_finite(), "gravity must be finite");
        assert!(settings.iterations > 0, "iterations must be positive");
        assert!(
            (0.0..1.0).contains(&settings.alpha)
                && settings.beta.is_finite()
                && settings.beta >= 0.0
                && settings.beta_angular.is_finite()
                && settings.beta_angular >= 0.0
                && settings.gamma > 0.0
                && settings.gamma < 1.0
                && settings.contact_damping.is_finite()
                && settings.contact_damping >= 0.0,
            "alpha must be in [0, 1), betas finite and nonnegative, gamma in (0, 1)"
        );
        assert!(
            dt.is_finite() && dt > 0.0,
            "physics timestep must be finite and positive"
        );

        if self.slots.iter().any(|slot| !world.contains(slot.entity)) {
            let (gone, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.slots)
                .into_iter()
                .partition(|slot| !world.contains(slot.entity));
            self.slots = kept;
            self.wake_islands_of(
                gone.iter()
                    .map(|s| (s.asleep, s.island, key(s.entity)))
                    .collect(),
            );
            self.index = (self.slots.iter().enumerate())
                .map(|(i, slot)| (slot.entity, i))
                .collect();
            self.prune_links();
        }

        // A static or kinematic body the caller moved wakes what rests on it.
        let mut woken = Vec::new();
        for slot in &mut self.slots {
            if slot.body.body_type != RigidBody::Dynamic {
                let t = &world.get(slot.entity).unwrap().transform;
                if slot.rest != (t.translation, t.rotation) {
                    slot.rest = (t.translation, t.rotation);
                    woken.push((slot.asleep, slot.island, key(slot.entity)));
                }
            }
        }
        if !woken.is_empty() {
            self.wake_islands_of(woken);
        }
        // A fully sleeping world cannot advance until a body, its transform,
        // or an input changes. Cached properties prove that shapes, densities
        // and rotations have already passed validation; all other invalid or
        // one-shot inputs fall through to the normal step.
        let quiescent = self.solver.bodies.len() == self.slots.len()
            && self
                .solver
                .bodies
                .iter()
                .zip(&self.slots)
                .all(|(previous, slot)| {
                    let body = &slot.body;
                    let sleeping = match body.body_type {
                        RigidBody::Dynamic => slot.asleep,
                        RigidBody::Static | RigidBody::Kinematic => !slot.asleep,
                    };
                    if previous.key != key(slot.entity)
                        || previous.kinematic != (body.body_type == RigidBody::Kinematic)
                        || !sleeping
                        || body.velocity != Velocity::default()
                        || body.force != ExternalForce::default()
                        || body.impulse.linear.to_array().map(f32::to_bits) != [0; 3]
                        || body.impulse.angular.to_array().map(f32::to_bits) != [0; 3]
                        || !body.material.friction.is_finite()
                        || body.material.friction < 0.0
                        || !body.material.static_friction.is_finite()
                        || body.material.static_friction < 0.0
                        || !slot.mass_properties.is_some_and(|m| {
                            m.collider == body.collider && m.density == body.material.density
                        })
                    {
                        return false;
                    }
                    let transform = world.get(slot.entity).unwrap().transform;
                    slot.rest == (transform.translation, transform.rotation)
                        && slot.pose_properties.is_some_and(|p| {
                            p.rotation_bits == transform.rotation.map(f32::to_bits)
                        })
                });
        if quiescent {
            return;
        }
        // Bodies are solved in slot order: point links at their slots.
        for (&(_, a, b), link) in self.links.iter().zip(&mut self.solver.links) {
            link.a = a.map(|a| self.index[&a]);
            link.b = self.index[&b];
        }
        let bodies = &mut self.solver.bodies;
        bodies.clear();
        let mut awake = 0;
        for slot in &mut self.slots {
            let transform = world.get(slot.entity).unwrap().transform;
            bodies.push(solver_body(slot, transform, dt));
            awake += usize::from(slot.body.body_type == RigidBody::Dynamic && !slot.asleep);
            slot.body.impulse = ExternalImpulse::default();
        }

        let threads = (awake / BODIES_PER_THREAD)
            .max(1)
            .next_power_of_two()
            .min(rayon::current_num_threads())
            .min(*PHYSICAL_CORES);
        let solver = &mut self.solver;
        if threads == 1 {
            // Avoid dispatching the entire step to a one-worker pool.
            solver.step(settings, dt, false, true);
        } else {
            let size = threads.trailing_zeros() as usize;
            if self.pools.len() <= size {
                self.pools.resize_with(size + 1, || None);
            }
            if self.pools[size].is_none() {
                // A pool that cannot be built leaves the step on the global pool.
                self.pools[size] = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .ok();
            }
            match &self.pools[size] {
                Some(pool) => pool.install(|| solver.step(settings, dt, true, false)),
                None => solver.step(settings, dt, false, false),
            }
        }

        for (slot, body) in self.slots.iter_mut().zip(&self.solver.bodies) {
            if slot.body.body_type == RigidBody::Static {
                continue;
            }
            let was_asleep = std::mem::replace(&mut slot.asleep, body.asleep);
            (slot.island, slot.sleep_time) = (body.island, body.sleep_time);
            slot.previous = body.previous;
            slot.body.velocity = Velocity {
                linear: body.velocity,
                angular: body.angular_velocity,
            };
            if was_asleep && body.asleep {
                continue;
            }
            let item = world.get_mut(slot.entity).unwrap();
            item.transform.translation = body.position.to_array();
            item.transform.rotation = body.rotation.to_array();
            if body.asleep || slot.body.body_type == RigidBody::Kinematic {
                slot.rest = (item.transform.translation, item.transform.rotation);
            }
        }
    }
}

fn key(entity: EntityHandle) -> u64 {
    (u64::from(entity.index()) << 32) | u64::from(entity.generation())
}

/// Validated solver input for one slot, with this step's force and impulse
/// folded into its velocity.
fn solver_body(slot: &mut Slot, transform: &Transform, dt: f32) -> solver::Body {
    let body = &slot.body;
    let position = Vec3::from_array(transform.translation);
    let rotation = Quat::from_array(transform.rotation);
    assert!(
        position.is_finite()
            && rotation.is_finite()
            && (rotation.length_squared() - 1.0).abs() < 0.001,
        "physics position must be finite and rotation must be a unit quaternion"
    );
    let PhysicsMaterial {
        density,
        friction,
        static_friction,
    } = body.material;
    assert!(
        density.is_finite() && density > 0.0,
        "density must be finite and positive"
    );
    assert!(
        friction.is_finite()
            && friction >= 0.0
            && static_friction.is_finite()
            && static_friction >= 0.0,
        "friction coefficients must be finite and nonnegative"
    );
    let (v, f) = (body.velocity, body.force);
    assert!(
        v.linear.is_finite() && v.angular.is_finite(),
        "physics velocities must be finite"
    );
    assert!(
        f.force.is_finite() && f.torque.is_finite(),
        "force and torque must be finite"
    );
    assert!(
        body.impulse.linear.is_finite() && body.impulse.angular.is_finite(),
        "impulses must be finite"
    );

    let (mut mass, moment) = match slot.mass_properties {
        Some(cached) if cached.collider == body.collider && cached.density == density => {
            (cached.mass, cached.moment)
        }
        _ => {
            let (mass, moment) = body.collider.mass_properties(density);
            slot.mass_properties = Some(CachedMass {
                collider: body.collider,
                density,
                mass,
                moment,
            });
            slot.pose_properties = None;
            (mass, moment)
        }
    };
    // Compare the raw quaternion bits so even signed-zero changes are not
    // hidden by float equality. A mass-property change invalidates the tensor.
    let rotation_bits = transform.rotation.map(f32::to_bits);
    let pose = match slot.pose_properties {
        Some(cached) if cached.rotation_bits == rotation_bits => cached,
        _ => {
            let rotation = rotation.normalize();
            let basis = Mat3::from_quat(rotation);
            let inertia = basis * Mat3::from_diagonal(moment) * basis.transpose();
            let cached = CachedPose {
                rotation_bits,
                rotation,
                basis,
                inertia,
            };
            slot.pose_properties = Some(cached);
            cached
        }
    };
    let (rotation, basis, mut inertia) = (pose.rotation, pose.basis, pose.inertia);
    let (mut velocity, mut angular_velocity) = (v.linear, v.angular);
    match body.body_type {
        RigidBody::Dynamic => {
            velocity += (f.force * dt + body.impulse.linear) / mass;
            let angular = f.torque * dt + body.impulse.angular;
            // Most bodies get no torque: skip building the inverse inertia.
            if angular != Vec3::ZERO {
                let inverse = basis * Mat3::from_diagonal(moment.recip()) * basis.transpose();
                angular_velocity += inverse * angular;
            }
        }
        RigidBody::Kinematic => (mass, inertia) = (0.0, Mat3::ZERO),
        RigidBody::Static => {
            (mass, inertia) = (0.0, Mat3::ZERO);
            (velocity, angular_velocity) = (Vec3::ZERO, Vec3::ZERO);
        }
    }
    solver::Body {
        key: key(slot.entity),
        shape: body.collider,
        position,
        rotation,
        initial_position: position,
        initial_rotation: rotation,
        inertial_position: position,
        inertial_rotation: rotation,
        initial_basis: basis,
        velocity,
        angular_velocity,
        previous: slot.previous,
        moved: Vec3A::ZERO,
        turned: Vec3A::ZERO,
        travel: Vec3::ZERO,
        reach: 0.0,
        stiffness: 0.0,
        mass,
        inertia: inertia.into(),
        friction,
        static_friction,
        kinematic: body.body_type == RigidBody::Kinematic,
        asleep: slot.asleep,
        // A caller's change to a sleeping body wakes its island.
        wake: slot.asleep
            && (v != Velocity::default()
                || f != ExternalForce::default()
                || body.impulse != ExternalImpulse::default()
                || slot.rest != (transform.translation, transform.rotation)),
        island: slot.island,
        sleep_time: slot.sleep_time,
    }
}
