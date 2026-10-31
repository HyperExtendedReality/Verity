//! The AVBD step: broadphase, persistent manifolds, graph coloring, and the
//! primal (per-body block solve) / dual (per-contact force) iterations.
//! Equation numbers refer to the AVBD paper.

use crate::{Collider, PhysicsSettings, collide};
use glam::{Mat3, Quat, Vec3};
use rayon::prelude::*;
use std::collections::HashMap;

const PENALTY_MIN: f32 = 1.0;
const PENALTY_MAX: f32 = 1.0e10;
/// Contacts hold this much overlap so they don't flicker between steps.
pub(crate) const COLLISION_MARGIN: f32 = 0.01;
/// Tangential drift below which a contact keeps its anchors (static friction).
const STICK_THRESHOLD: f32 = 1.0e-5;
/// Smallest parallel batch; below it rayon's fork cost outweighs the work.
const MIN_BATCH: usize = 64;

pub(crate) struct Body {
    pub key: u64,
    pub shape: Collider,
    pub position: Vec3,
    pub rotation: Quat,
    pub initial_position: Vec3,
    pub initial_rotation: Quat,
    pub inertial_position: Vec3,
    pub inertial_rotation: Quat,
    pub velocity: Vec3,
    pub angular_velocity: Vec3,
    pub previous: Vec3,
    /// Zero for static and kinematic bodies.
    pub mass: f32,
    /// World-space inertia tensor at the start of the step.
    pub inertia: Mat3,
    pub friction: f32,
    pub kinematic: bool,
}

impl Body {
    fn dynamic(&self) -> bool {
        self.mass > 0.0
    }
    fn bound(&self) -> f32 {
        match self.shape {
            Collider::Cuboid { half_extents } => half_extents.length(),
            Collider::Sphere { radius } => radius,
        }
    }
}

/// `q` rotated by the small rotation vector `w`.
pub(crate) fn quat_add(q: Quat, w: Vec3) -> Quat {
    (q + Quat::from_xyzw(w.x, w.y, w.z, 0.0) * q * 0.5).normalize()
}

/// Rotation vector taking `b` to `a`, valid for small differences.
fn quat_sub(a: Quat, b: Quat) -> Vec3 {
    let d = a * b.conjugate();
    if d.w < 0.0 {
        -d.xyz() * 2.0
    } else {
        d.xyz() * 2.0
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct Contact {
    pub feature: u32,
    /// Anchors in each body's local frame.
    pub ra: Vec3,
    pub rb: Vec3,
    /// Constraint error at the start of the step, in the contact basis.
    c0: Vec3,
    penalty: Vec3,
    lambda: Vec3,
    stick: bool,
}

impl Contact {
    pub fn new(feature: u32, ra: Vec3, rb: Vec3) -> Self {
        Self {
            feature,
            ra,
            rb,
            ..Default::default()
        }
    }
}

pub(crate) struct Manifold {
    a: usize,
    b: usize,
    key: (u64, u64),
    /// Normal (B to A) then two tangents.
    basis: [Vec3; 3],
    contacts: [Contact; collide::MAX_CONTACTS],
    len: usize,
    friction: f32,
    /// Heavier body's mass: penalties are per kilogram so tuning is unit-free.
    scale: f32,
}

/// One contact's linearized constraint (Sec. 4) and clamped force.
struct Eval {
    lin: [Vec3; 3],
    ang_a: [Vec3; 3],
    ang_b: [Vec3; 3],
    c: Vec3,
    force: Vec3,
    tangent: f32,
    bound: f32,
}

/// Both bodies' displacement since the step began, shared by a manifold's contacts.
struct Motion {
    rotation_a: Quat,
    rotation_b: Quat,
    linear: Vec3,
    angular_a: Vec3,
    angular_b: Vec3,
}

impl Motion {
    fn new(a: &Body, b: &Body) -> Self {
        Self {
            rotation_a: a.rotation,
            rotation_b: b.rotation,
            linear: (a.position - a.initial_position) - (b.position - b.initial_position),
            angular_a: quat_sub(a.rotation, a.initial_rotation),
            angular_b: quat_sub(b.rotation, b.initial_rotation),
        }
    }
}

impl Manifold {
    fn eval(&self, contact: &Contact, m: &Motion, alpha: f32) -> Eval {
        let (ra, rb) = (m.rotation_a * contact.ra, m.rotation_b * contact.rb);
        let lin = self.basis;
        let ang_a = lin.map(|n| ra.cross(n));
        let ang_b = lin.map(|n| rb.cross(-n));
        let c = Vec3::from_array(std::array::from_fn(|k| {
            contact.c0[k] * (1.0 - alpha)
                + lin[k].dot(m.linear)
                + ang_a[k].dot(m.angular_a)
                + ang_b[k].dot(m.angular_b)
        }));
        let mut force = contact.penalty * c + contact.lambda;
        force.x = force.x.min(0.0);
        let bound = force.x.abs() * self.friction;
        let tangent = force.y.hypot(force.z);
        if tangent > bound && tangent > 0.0 {
            force.y *= bound / tangent;
            force.z *= bound / tangent;
        }
        Eval {
            lin,
            ang_a,
            ang_b,
            c,
            force,
            tangent,
            bound,
        }
    }

    /// Accumulate this manifold's force and Hessian approximation for `body` (Eqs. 5, 6).
    fn stamp(&self, bodies: &[Body], body: usize, alpha: f32, system: &mut System) {
        let motion = Motion::new(&bodies[self.a], &bodies[self.b]);
        let sign = if body == self.a { 1.0 } else { -1.0 };
        for contact in &self.contacts[..self.len] {
            let e = self.eval(contact, &motion, alpha);
            let ang = if body == self.a { e.ang_a } else { e.ang_b };
            for (k, ja) in ang.into_iter().enumerate() {
                let (jl, stiffness) = (e.lin[k] * sign, contact.penalty[k]);
                system.lin += outer(jl * stiffness, jl);
                system.ang += outer(ja * stiffness, ja);
                system.cross += outer(ja * stiffness, jl);
                system.rhs_lin += jl * e.force[k];
                system.rhs_ang += ja * e.force[k];
            }
        }
    }

    /// Store forces and ramp penalties (Eq. 16).
    fn dual(&mut self, bodies: &[Body], alpha: f32, beta: f32) {
        let motion = Motion::new(&bodies[self.a], &bodies[self.b]);
        let (beta, max) = (beta * self.scale, PENALTY_MAX * self.scale);
        for i in 0..self.len {
            let e = self.eval(&self.contacts[i], &motion, alpha);
            let contact = &mut self.contacts[i];
            contact.lambda = e.force;
            if e.force.x < 0.0 {
                contact.penalty.x = (contact.penalty.x + beta * e.c.x.abs()).min(max);
            }
            if e.tangent <= e.bound {
                contact.penalty.y = (contact.penalty.y + beta * e.c.y.abs()).min(max);
                contact.penalty.z = (contact.penalty.z + beta * e.c.z.abs()).min(max);
                contact.stick = e.c.y.hypot(e.c.z) < STICK_THRESHOLD;
            }
        }
    }
}

/// `a bᵀ`.
fn outer(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::from_cols(a * b.x, a * b.y, a * b.z)
}

/// One body's 6x6 SPD system: linear block, angular block, angular-by-linear
/// coupling, and the gradient.
struct System {
    lin: Mat3,
    ang: Mat3,
    cross: Mat3,
    rhs_lin: Vec3,
    rhs_ang: Vec3,
}

impl System {
    /// Solve `A x = -rhs` by LDLᵀ (Eq. 4).
    fn solve(&self) -> (Vec3, Vec3) {
        let at = |i: usize, j: usize| match (i < 3, j < 3) {
            (true, true) => self.lin.col(j)[i],
            (false, false) => self.ang.col(j - 3)[i - 3],
            (false, true) => self.cross.col(j)[i - 3],
            (true, false) => self.cross.col(i)[j - 3],
        };
        let mut l = [[0.0f32; 6]; 6];
        let mut d = [0.0f32; 6];
        for j in 0..6 {
            d[j] = at(j, j) - (0..j).map(|k| l[j][k] * l[j][k] * d[k]).sum::<f32>();
            for i in j + 1..6 {
                l[i][j] = (at(i, j) - (0..j).map(|k| l[i][k] * l[j][k] * d[k]).sum::<f32>()) / d[j];
            }
        }
        let mut x: [f32; 6] = std::array::from_fn(|i| {
            -if i < 3 {
                self.rhs_lin[i]
            } else {
                self.rhs_ang[i - 3]
            }
        });
        for i in 0..6 {
            for k in 0..i {
                x[i] -= l[i][k] * x[k];
            }
        }
        for i in 0..6 {
            x[i] /= d[i];
        }
        for i in (0..6).rev() {
            for k in i + 1..6 {
                x[i] -= l[k][i] * x[k];
            }
        }
        (Vec3::new(x[0], x[1], x[2]), Vec3::new(x[3], x[4], x[5]))
    }
}

#[derive(Default)]
pub(crate) struct Solver {
    pub bodies: Vec<Body>,
    manifolds: Vec<Manifold>,
    /// Manifold index by body-key pair, for warm starting the next step.
    cache: HashMap<(u64, u64), usize>,
    order: Vec<usize>,
    pairs: Vec<(usize, usize)>,
    /// Manifolds touching each body, as CSR offsets into `adjacency`.
    offsets: Vec<u32>,
    adjacency: Vec<u32>,
    colors: Vec<Vec<usize>>,
    updates: Vec<(Vec3, Quat)>,
}

impl Solver {
    pub fn contact_count(&self) -> usize {
        self.manifolds.iter().map(|m| m.len).sum()
    }

    pub fn step(&mut self, settings: &PhysicsSettings, dt: f32) {
        self.broadphase();
        self.narrowphase(settings);
        self.color();

        let gravity = settings.gravity;
        self.bodies
            .par_iter_mut()
            .with_min_len(MIN_BATCH)
            .for_each(|body| {
                body.initial_position = body.position;
                body.initial_rotation = body.rotation;
                if body.dynamic() {
                    // Inertial target (Eq. 2) and adaptive warm start (VBD).
                    body.inertial_position =
                        body.position + body.velocity * dt + gravity * (dt * dt);
                    body.inertial_rotation = quat_add(body.rotation, body.angular_velocity * dt);
                    let accel = (body.velocity - body.previous) / dt;
                    let weight = (accel.dot(gravity) / gravity.length_squared()).clamp(0.0, 1.0);
                    let weight = if weight.is_finite() { weight } else { 0.0 };
                    body.position += body.velocity * dt + gravity * (weight * dt * dt);
                    body.rotation = body.inertial_rotation;
                } else if body.kinematic {
                    body.position += body.velocity * dt;
                    body.rotation = quat_add(body.rotation, body.angular_velocity * dt);
                }
            });

        let (alpha, beta) = (settings.alpha, settings.beta);
        for _ in 0..settings.iterations {
            for color in &self.colors {
                let (bodies, manifolds) = (&self.bodies, &self.manifolds);
                let (offsets, adjacency) = (&self.offsets, &self.adjacency);
                // Bodies of one color share no manifold, so each reads only
                // bodies this pass leaves untouched.
                color
                    .par_iter()
                    .with_min_len(MIN_BATCH)
                    .map(|&i| {
                        let body = &bodies[i];
                        let scale = 1.0 / (dt * dt);
                        let mass = Mat3::from_diagonal(Vec3::splat(body.mass * scale));
                        let inertia = body.inertia * scale;
                        let mut system = System {
                            lin: mass,
                            ang: inertia,
                            cross: Mat3::ZERO,
                            rhs_lin: mass * (body.position - body.inertial_position),
                            rhs_ang: inertia * quat_sub(body.rotation, body.inertial_rotation),
                        };
                        let range = offsets[i] as usize..offsets[i + 1] as usize;
                        for &m in &adjacency[range] {
                            manifolds[m as usize].stamp(bodies, i, alpha, &mut system);
                        }
                        let (dx, dr) = system.solve();
                        (body.position + dx, quat_add(body.rotation, dr))
                    })
                    .collect_into_vec(&mut self.updates);
                for (&i, &(position, rotation)) in color.iter().zip(&self.updates) {
                    self.bodies[i].position = position;
                    self.bodies[i].rotation = rotation;
                }
            }
            let bodies = &self.bodies;
            self.manifolds
                .par_iter_mut()
                .with_min_len(MIN_BATCH)
                .for_each(|m| m.dual(bodies, alpha, beta));
        }

        self.bodies
            .par_iter_mut()
            .with_min_len(MIN_BATCH)
            .for_each(|body| {
                body.previous = body.velocity;
                if body.dynamic() {
                    body.velocity = (body.position - body.initial_position) / dt;
                    body.angular_velocity = quat_sub(body.rotation, body.initial_rotation) / dt;
                }
            });
    }

    /// Sweep and prune bounding-sphere boxes along the axis of widest spread.
    fn broadphase(&mut self) {
        let bodies = &self.bodies;
        self.pairs.clear();
        if bodies.len() < 2 {
            return;
        }
        let (lo, hi) = bodies.iter().fold(
            (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
            |(lo, hi), b| (lo.min(b.position), hi.max(b.position)),
        );
        let spread = hi - lo;
        let axis = if spread.x >= spread.y && spread.x >= spread.z {
            0
        } else if spread.y >= spread.z {
            1
        } else {
            2
        };
        let reach = |b: &Body| b.bound() + COLLISION_MARGIN;
        self.order.clear();
        self.order.extend(0..bodies.len());
        self.order.sort_unstable_by(|&i, &j| {
            let (a, b) = (&bodies[i], &bodies[j]);
            (a.position[axis] - reach(a)).total_cmp(&(b.position[axis] - reach(b)))
        });
        for (n, &i) in self.order.iter().enumerate() {
            let a = &bodies[i];
            let max = a.position[axis] + reach(a);
            for &j in &self.order[n + 1..] {
                let b = &bodies[j];
                if b.position[axis] - reach(b) > max {
                    break;
                }
                let r = reach(a) + reach(b);
                if (a.dynamic() || b.dynamic()) && a.position.distance_squared(b.position) <= r * r
                {
                    self.pairs.push(if a.key < b.key { (i, j) } else { (j, i) });
                }
            }
        }
    }

    /// Collide pairs in parallel, merging last step's contacts and warm
    /// starting their forces and penalties (Eq. 19).
    fn narrowphase(&mut self, settings: &PhysicsSettings) {
        let (bodies, old, cache) = (&self.bodies, &self.manifolds, &self.cache);
        let (alpha, gamma) = (settings.alpha, settings.gamma);
        let manifolds: Vec<Manifold> = self
            .pairs
            .par_iter()
            .with_min_len(MIN_BATCH)
            .filter_map(|&(a, b)| {
                let (ba, bb) = (&bodies[a], &bodies[b]);
                let (basis, mut contacts, len) = collide::collide(ba, bb)?;
                let key = (ba.key, bb.key);
                if let Some(previous) = cache.get(&key).map(|&i| &old[i]) {
                    for contact in &mut contacts[..len] {
                        let matched = previous.contacts[..previous.len]
                            .iter()
                            .find(|p| p.feature == contact.feature);
                        if let Some(p) = matched {
                            let (ra, rb) = (contact.ra, contact.rb);
                            *contact = *p;
                            if !p.stick {
                                (contact.ra, contact.rb) = (ra, rb);
                            }
                        }
                    }
                }
                let scale = ba.mass.max(bb.mass);
                let (min, max) = (
                    Vec3::splat(PENALTY_MIN * scale),
                    Vec3::splat(PENALTY_MAX * scale),
                );
                for contact in &mut contacts[..len] {
                    let d = (ba.position + ba.rotation * contact.ra)
                        - (bb.position + bb.rotation * contact.rb);
                    contact.c0 = Vec3::new(
                        basis[0].dot(d) + COLLISION_MARGIN,
                        basis[1].dot(d),
                        basis[2].dot(d),
                    );
                    contact.lambda *= alpha * gamma;
                    contact.penalty = (contact.penalty * gamma).clamp(min, max);
                }
                Some(Manifold {
                    a,
                    b,
                    key,
                    basis,
                    contacts,
                    len,
                    friction: (ba.friction * bb.friction).sqrt(),
                    scale,
                })
            })
            .collect();
        self.manifolds = manifolds;
        self.cache.clear();
        self.cache
            .extend(self.manifolds.iter().enumerate().map(|(i, m)| (m.key, i)));
    }

    /// Build body-to-manifold adjacency and greedily color dynamic bodies so no
    /// two of one color share a manifold.
    fn color(&mut self) {
        let n = self.bodies.len();
        self.offsets.clear();
        self.offsets.resize(n + 1, 0);
        for m in &self.manifolds {
            self.offsets[m.a + 1] += 1;
            self.offsets[m.b + 1] += 1;
        }
        for i in 0..n {
            self.offsets[i + 1] += self.offsets[i];
        }
        self.adjacency.clear();
        self.adjacency.resize(self.offsets[n] as usize, 0);
        let mut fill: Vec<u32> = self.offsets[..n].to_vec();
        for (index, m) in self.manifolds.iter().enumerate() {
            for body in [m.a, m.b] {
                self.adjacency[fill[body] as usize] = index as u32;
                fill[body] += 1;
            }
        }

        for color in &mut self.colors {
            color.clear();
        }
        let mut color_of = vec![u32::MAX; n];
        // stamp[c] == i marks color c as taken by a neighbor of body i.
        let mut stamp: Vec<usize> = Vec::new();
        for i in (0..n).filter(|&i| self.bodies[i].dynamic()) {
            for &m in &self.adjacency[self.offsets[i] as usize..self.offsets[i + 1] as usize] {
                let m = &self.manifolds[m as usize];
                let other = if m.a == i { m.b } else { m.a };
                if let Some(c) = stamp.get_mut(color_of[other] as usize) {
                    *c = i;
                }
            }
            let c = stamp.iter().position(|&s| s != i).unwrap_or(stamp.len());
            if c == stamp.len() {
                stamp.push(i);
                if self.colors.len() <= c {
                    self.colors.push(Vec::new());
                }
            }
            color_of[i] = c as u32;
            self.colors[c].push(i);
        }
        self.colors.truncate(stamp.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ldlt_solves_spd_system() {
        let m = Mat3::from_cols(
            Vec3::new(4.0, 1.0, 0.5),
            Vec3::new(1.0, 3.0, 0.2),
            Vec3::new(0.5, 0.2, 2.0),
        );
        let system = System {
            lin: m,
            ang: m * 2.0,
            cross: Mat3::from_cols(
                Vec3::new(0.3, 0.0, 0.1),
                Vec3::ZERO,
                Vec3::new(0.0, 0.2, 0.0),
            ),
            rhs_lin: Vec3::new(1.0, -2.0, 3.0),
            rhs_ang: Vec3::new(-1.0, 0.5, 2.0),
        };
        let (x, y) = system.solve();
        // A [x; y] must equal -rhs.
        let lin = system.lin * x + system.cross.transpose() * y;
        let ang = system.cross * x + system.ang * y;
        assert!((lin + system.rhs_lin).length() < 1e-5, "{lin}");
        assert!((ang + system.rhs_ang).length() < 1e-5, "{ang}");
    }

    #[test]
    fn quaternion_difference_inverts_addition() {
        let q = Quat::from_rotation_y(0.7);
        let w = Vec3::new(0.01, -0.02, 0.015);
        assert!((quat_sub(quat_add(q, w), q) - w).length() < 1e-4);
    }
}
