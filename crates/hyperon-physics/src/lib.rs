//! Rigid bodies synchronized with a Hyperon runtime world, solved with
//! Augmented Vertex Block Descent (Giles et al., SIGGRAPH 2025).
//!
//! AVBD solves each body's 6-DOF position as a small block Newton step against
//! every contact touching it, with augmented-Lagrangian contact forces whose
//! penalties ramp up only as far as needed. That makes stacks and large mass
//! ratios stable at a few iterations and no substeps.
//!
//! Bodies sharing a contact are graph-colored; every body of one color is
//! updated in parallel, which is the same structure a GPU port dispatches.
//! The solver runs on CPU threads: game-sized scenes are latency bound, need
//! results the same frame, and on mobile the GPU belongs to the renderer.

mod collide;
mod solver;

use glam::{Mat3, Quat, Vec3};
use hyperon_core::{EntityHandle, Transform, World};
use std::collections::HashMap;

/// Solver parameters. The caller supplies the fixed timestep to [`PhysicsWorld::step`].
#[derive(Debug, Clone)]
pub struct PhysicsSettings {
    pub gravity: Vec3,
    /// Block descent sweeps per step.
    pub iterations: u32,
    /// Fraction of the starting contact error left uncorrected each step, in
    /// [0, 1). Higher is smoother, lower is snappier.
    pub alpha: f32,
    /// Penalty growth per metre of contact error.
    pub beta: f32,
    /// Per-step decay of warm-started forces and penalties, in (0, 1).
    pub gamma: f32,
}

impl Default for PhysicsSettings {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            iterations: 10,
            alpha: 0.99,
            beta: 10_000.0,
            gamma: 0.999,
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

/// Density in kg/m³ and friction coefficient. Density must be positive and
/// friction nonnegative; both finite. Contacts use the geometric mean friction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicsMaterial {
    pub density: f32,
    pub friction: f32,
}

impl Default for PhysicsMaterial {
    fn default() -> Self {
        Self {
            density: 1000.0,
            friction: 0.6,
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

/// An entity is not live in the supplied world.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidEntity;

impl std::fmt::Display for InvalidEntity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("entity is not live in this world")
    }
}
impl std::error::Error for InvalidEntity {}

struct Slot {
    entity: EntityHandle,
    body: PhysicsBody,
    /// Last step's starting velocity, for AVBD's adaptive gravity warm start.
    previous: Vec3,
}

/// Owns entity-associated physics data and solver state. Use each instance
/// with one `World` only: handles are world-local.
#[derive(Default)]
pub struct PhysicsWorld {
    slots: Vec<Slot>,
    index: HashMap<EntityHandle, usize>,
    solver: solver::Solver,
}

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
    /// Contacts touching bodies after the last step.
    pub fn contact_count(&self) -> usize {
        self.solver.contact_count()
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
            return Ok(Some(std::mem::replace(&mut self.slots[i].body, body)));
        }
        self.index.insert(entity, self.slots.len());
        self.slots.push(Slot {
            entity,
            previous: body.velocity.linear,
            body,
        });
        Ok(None)
    }

    pub fn remove(&mut self, entity: EntityHandle) -> Option<PhysicsBody> {
        let i = self.index.remove(&entity)?;
        let slot = self.slots.swap_remove(i);
        if let Some(moved) = self.slots.get(i) {
            self.index.insert(moved.entity, i);
        }
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
                && settings.gamma > 0.0
                && settings.gamma < 1.0,
            "alpha must be in [0, 1), beta finite and nonnegative, gamma in (0, 1)"
        );
        assert!(
            dt.is_finite() && dt > 0.0,
            "physics timestep must be finite and positive"
        );

        if self.slots.iter().any(|slot| !world.contains(slot.entity)) {
            self.slots.retain(|slot| world.contains(slot.entity));
            self.index = (self.slots.iter().enumerate())
                .map(|(i, slot)| (slot.entity, i))
                .collect();
        }

        let bodies = &mut self.solver.bodies;
        bodies.clear();
        for slot in &mut self.slots {
            let transform = world.get(slot.entity).unwrap().transform;
            bodies.push(solver_body(slot, transform, dt));
            slot.body.impulse = ExternalImpulse::default();
        }

        self.solver.step(settings, dt);

        for (slot, body) in self.slots.iter_mut().zip(&self.solver.bodies) {
            if slot.body.body_type == RigidBody::Static {
                continue;
            }
            slot.previous = body.previous;
            slot.body.velocity = Velocity {
                linear: body.velocity,
                angular: body.angular_velocity,
            };
            let item = world.get_mut(slot.entity).unwrap();
            item.transform.translation = body.position.to_array();
            item.transform.rotation = body.rotation.to_array();
        }
    }
}

/// Validated solver input for one slot, with this step's force and impulse
/// folded into its velocity.
fn solver_body(slot: &Slot, transform: &Transform, dt: f32) -> solver::Body {
    let body = &slot.body;
    let position = Vec3::from_array(transform.translation);
    let rotation = Quat::from_array(transform.rotation);
    assert!(
        position.is_finite()
            && rotation.is_finite()
            && (rotation.length_squared() - 1.0).abs() < 0.001,
        "physics position must be finite and rotation must be a unit quaternion"
    );
    let PhysicsMaterial { density, friction } = body.material;
    assert!(
        density.is_finite() && density > 0.0,
        "density must be finite and positive"
    );
    assert!(
        friction.is_finite() && friction >= 0.0,
        "friction must be finite and nonnegative"
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

    let rotation = rotation.normalize();
    let (mut mass, moment) = body.collider.mass_properties(density);
    let basis = Mat3::from_quat(rotation);
    let mut inertia = basis * Mat3::from_diagonal(moment) * basis.transpose();
    let (mut velocity, mut angular_velocity) = (v.linear, v.angular);
    match body.body_type {
        RigidBody::Dynamic => {
            let inverse = basis * Mat3::from_diagonal(moment.recip()) * basis.transpose();
            velocity += (f.force * dt + body.impulse.linear) / mass;
            angular_velocity += inverse * (f.torque * dt + body.impulse.angular);
        }
        RigidBody::Kinematic => (mass, inertia) = (0.0, Mat3::ZERO),
        RigidBody::Static => {
            (mass, inertia) = (0.0, Mat3::ZERO);
            (velocity, angular_velocity) = (Vec3::ZERO, Vec3::ZERO);
        }
    }
    solver::Body {
        key: (u64::from(slot.entity.index()) << 32) | u64::from(slot.entity.generation()),
        shape: body.collider,
        position,
        rotation,
        initial_position: position,
        initial_rotation: rotation,
        inertial_position: position,
        inertial_rotation: rotation,
        velocity,
        angular_velocity,
        previous: slot.previous,
        mass,
        inertia,
        friction,
        kinematic: body.body_type == RigidBody::Kinematic,
    }
}
