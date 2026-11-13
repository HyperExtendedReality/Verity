//! The AVBD step: broadphase, persistent manifolds, graph coloring, and the
//! primal (per-body block solve) / dual (per-contact force) iterations.
//! Equation numbers refer to the AVBD paper.

use crate::link::Link;
use crate::{Bounds, Collider, Ground, PhysicsSettings, collide};
use glam::{BVec4A, Mat3, Mat3A, Mat4, Quat, Vec3, Vec3A, Vec4};
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;

/// Fast mixing for solver-owned entity pairs. Indices occupy the upper 32
/// bits, so fold the product's high bits into HashMap's low bucket bits.
#[derive(Default)]
struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(u64::from(b));
        }
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    fn finish(&self) -> u64 {
        self.0 ^ (self.0 >> 32)
    }
}

type PairMap = HashMap<(u64, u64), usize, BuildHasherDefault<KeyHasher>>;

/// Compiles a function twice, for AVX2 and for AVX-512: `#[inline(always)]`
/// cannot yet carry target features, and a kernel only uses AVX-512's 32
/// registers if compiled for them.
#[cfg(target_arch = "x86_64")]
macro_rules! avx2_and_avx512 {
    ($(#[$meta:meta])* unsafe fn $avx2:ident | $avx512:ident $($rest:tt)*) => {
        $(#[$meta])*
        #[target_feature(enable = "avx2")]
        unsafe fn $avx2 $($rest)*
        $(#[$meta])*
        #[target_feature(enable = "avx2,avx512f,avx512vl,avx512bw,avx512dq")]
        unsafe fn $avx512 $($rest)*
    };
}

/// Point speed (m/s) below which a body counts as resting.
const SLEEP_SPEED: f32 = 0.05;
/// Cache lines of each manifold to prefetch.
const PREFETCH_LINES: usize = 2;
/// Seconds an island must rest before it sleeps.
const TIME_TO_SLEEP: f32 = 0.5;
/// Closing speed (m/s) of free bodies above which forces set their velocities.
const IMPACT_SPEED: f32 = 1.0;
/// Closing speed (m/s) below which contacts do not bounce, so resting and
/// gently landing bodies settle instead of jittering (Box2D's threshold).
const BOUNCE_SPEED: f32 = 1.0;
/// Velocity passes over bouncing manifolds.
const BOUNCE_ITERATIONS: usize = 4;
/// Rotation per step (rad) above which the gyroscopic predictor is held to
/// the body's energy.
const GYROSCOPIC_TURN: f32 = 0.25;
/// The `iteration` of a stabilization sweep, which updates no duals.
const STABILIZE: u32 = u32::MAX;
/// Marks an adjacency entry as a link, not a manifold.
const LINK: u32 = 1 << 31;
/// Penetration beyond the margin that keeps an island awake: AVBD pushes
/// overlap out slowly, below the sleep speed, so speed alone would freeze it in.
const SLEEP_PENETRATION: f32 = 0.01;
const PENALTY_MIN: f32 = 1.0;
/// Bound on the dual variables' magnitude, per kilogram (Sec. 4, "Bounding
/// the Dual Variables"): conflicting constraints cannot drive them to infinity.
const LAMBDA_MAX: f32 = 1.0e10;
const PENALTY_MAX: f32 = 1.0e10;
/// Shapes this close have contacts, whatever their motion, so contacts do not
/// flicker while bodies rest at zero overlap. Stays below the 1 cm gaps of
/// neighbouring boxes: each such pair would cost a manifold.
pub(crate) const CONTACT_DISTANCE: f32 = 0.004;
/// Half the size of a ground body's box, finite so arithmetic on it stays so.
const UNBOUNDED: f32 = 1.0e30;
/// Relative motion since a manifold's contacts were generated below which it
/// reuses them instead of colliding again: resting contacts barely move.
const REUSE_DISTANCE: f32 = 0.005;
const REUSE_ANGLE: f32 = 0.005;
/// Tangential drift below which a contact keeps its anchors (static friction).
const STICK_THRESHOLD: f32 = 1.0e-5;
/// Manifolds per narrowphase task: a re-collision costs several times a
/// reuse, so small batches keep threads evenly loaded.
const PAIR_BATCH: usize = 16;
/// Smallest batch of a cheap per-body pass: a few nanoseconds each, so
/// splitting one under this size costs more in thread wake-ups than it saves.
const BODY_BATCH: usize = 1024;
/// How far a moving body's broadphase box reaches beyond its shape (m):
/// pairs are swept again only once a body leaves it. Kept small: 0.1 m
/// boxes in a moving pile added more narrowphase pairs than sweeps saved.
const FAT_MARGIN: f32 = 0.02;
/// Boxes per broadphase sweep task.
const SWEEP_CHUNK: usize = 128;
/// Below these many boxes the broadphase sorts, or sweeps, on one thread:
/// splitting smaller arrays costs more in wake-ups than it saves.
const SERIAL_SORT: usize = 512;
const SERIAL_SWEEP: usize = 512;

pub(crate) struct Body {
    pub key: u64,
    pub shape: Collider,
    pub position: Vec3,
    pub rotation: Quat,
    pub initial_position: Vec3,
    pub initial_rotation: Quat,
    pub inertial_position: Vec3,
    pub inertial_rotation: Quat,
    /// Step-start orientation, shared by broadphase and narrowphase pairs.
    pub initial_basis: Mat3,
    pub velocity: Vec3,
    pub angular_velocity: Vec3,
    pub previous: Vec3,
    /// Linear and angular displacement since the step began, kept with the
    /// pose so contacts need no quaternion math per visit.
    pub moved: Vec3A,
    pub turned: Vec3A,
    /// How far this step's velocity should carry the body, and how far any
    /// of its surface may swing: what speculative contacts cover.
    pub travel: Vec3,
    pub reach: f32,
    /// The inertial term of the block system, mass / dt². Zero if unmoved by forces.
    pub stiffness: f32,
    /// Zero for static and kinematic bodies.
    pub mass: f32,
    /// World-space inertia tensor at the start of the step.
    pub inertia: Mat3A,
    pub friction: f32,
    pub static_friction: f32,
    pub restitution: f32,
    pub kinematic: bool,
    /// Frozen until a moving body touches its island or the caller wakes it;
    /// asleep bodies cost no broadphase, narrowphase or solver work.
    pub asleep: bool,
    /// Set by the caller on an asleep body it changed: wakes the island.
    pub wake: bool,
    /// Which sleeping island the body belongs to, while asleep.
    pub island: u32,
    pub sleep_time: f32,
}

impl Body {
    fn place(&mut self, position: Vec3, rotation: Quat) {
        self.position = position;
        self.rotation = rotation;
        self.moved = (position - self.initial_position).into();
        self.turned = quat_step(rotation, self.initial_rotation).into();
    }
    /// Where this step may carry the body, for speculative contacts.
    fn predict(&mut self, dt: f32) {
        self.stiffness = self.mass / (dt * dt);
        (self.travel, self.reach) = if self.active() {
            // Spinning moves only what sticks out past the inscribed sphere.
            let swing = match self.shape {
                Collider::Cuboid { half_extents } => self.bound() - half_extents.min_element(),
                Collider::Capsule { radius, .. } => self.bound() - radius,
                Collider::Sphere { .. } | Collider::Ground => 0.0,
            };
            (
                self.velocity * dt,
                self.angular_velocity.length() * swing * dt,
            )
        } else {
            (Vec3::ZERO, 0.0)
        };
    }
    /// Take velocities from the main sweeps. With `wrench`, the contact forces
    /// on a body in an impact, they come from those forces rather than its
    /// motion: each force acts equally on two bodies, so momentum survives a
    /// solve that has not converged. Motion alone would show the stop of
    /// whichever body moved first, without the push the other has yet to get.
    /// Resting contacts must not use this: their numerical damping keeps
    /// stacks stable, where unconverged forces would feed them energy.
    fn finish(&mut self, dt: f32, wrench: Option<(Vec3A, Vec3A)>) {
        self.previous = self.velocity;
        if self.dynamic() {
            let (moved, turned) = match wrench {
                Some((force, torque)) => {
                    let scale = dt * dt;
                    let start = quat_step(self.inertial_rotation, self.initial_rotation);
                    (
                        Vec3A::from(self.inertial_position - self.initial_position)
                            - force * (scale / self.mass),
                        Vec3A::from(start) - self.inertia.inverse() * torque * scale,
                    )
                }
                None => (self.moved, self.turned),
            };
            self.velocity = (moved / dt).into();
            self.conserve_angular_momentum(turned, dt);
        }
    }
    /// A turned body's world inertia has turned with it: keep its angular
    /// momentum, not its angular velocity. This is the gyroscopic term. A
    /// step's displacement is its average velocity, which pairs with the
    /// inertia `midpoint_spin` used, so a free body's momentum is exact.
    fn conserve_angular_momentum(&mut self, turned: Vec3A, dt: f32) {
        let average = turned / dt;
        self.angular_velocity = if self.gyroscopic() {
            let start = Vec3A::from(self.angular_velocity);
            let momentum = self.inertia_after(start * (0.5 * dt)) * average;
            let spin = self.inertia_after(turned).inverse() * momentum;
            doing_no_work(spin, average, momentum).into()
        } else {
            average.into()
        };
    }
    /// The velocity to cross this step with: the explicit midpoint of a body
    /// conserving momentum, as a whole step at the start velocity gains energy.
    fn midpoint_spin(&self, dt: f32) -> Vec3 {
        if !self.gyroscopic() {
            return self.angular_velocity;
        }
        let spin = Vec3A::from(self.angular_velocity);
        let momentum = self.inertia * spin;
        let midpoint = self.inertia_after(spin * (0.5 * dt)).inverse() * momentum;
        // Slower, a midpoint rightly differs from the start by O(h) in energy.
        if spin.length() * dt > GYROSCOPIC_TURN {
            doing_no_work(midpoint, spin, momentum).into()
        } else {
            midpoint.into()
        }
    }
    /// Equal principal moments have no gyroscopic term.
    fn gyroscopic(&self) -> bool {
        match self.shape {
            Collider::Cuboid { half_extents: h } => h.x != h.y || h.y != h.z,
            Collider::Capsule { .. } => true,
            Collider::Sphere { .. } | Collider::Ground => false,
        }
    }
    /// World inertia after the rotation `w` from the step start.
    fn inertia_after(&self, w: Vec3A) -> Mat3A {
        let turn = Mat3A::from_quat(quat_add(Quat::IDENTITY, w.into()));
        turn * self.inertia * turn.transpose()
    }
    fn dynamic(&self) -> bool {
        self.mass > 0.0 && !self.asleep
    }
    /// Solved this step: dynamic and awake.
    pub(crate) fn solvable(&self) -> bool {
        self.dynamic()
    }
    /// Can move this step, and so touch and wake an asleep body.
    pub(crate) fn active(&self) -> bool {
        self.dynamic()
            || (self.kinematic
                && (self.velocity != Vec3::ZERO || self.angular_velocity != Vec3::ZERO))
    }
    pub(crate) fn bound(&self) -> f32 {
        match self.shape {
            Collider::Cuboid { half_extents } => half_extents.length(),
            Collider::Sphere { radius } => radius,
            Collider::Capsule {
                radius,
                half_height,
            } => radius + half_height,
            Collider::Ground => 0.0,
        }
    }
    /// Rolls over its surface, so contact anchors must not stick.
    fn rolls(&self) -> bool {
        matches!(
            self.shape,
            Collider::Sphere { .. } | Collider::Capsule { .. }
        )
    }
}

/// `spin`, the result of turning `start` at constant `momentum`, with no more
/// energy than `start`. Turning does no work, but these steps are explicit: at
/// a large turn per step they would feed a long thin body energy without bound.
fn doing_no_work(spin: Vec3A, start: Vec3A, momentum: Vec3A) -> Vec3A {
    let (before, after) = (start.dot(momentum), spin.dot(momentum));
    if after > before && before > 0.0 {
        spin * (before / after).sqrt()
    } else {
        spin
    }
}

/// `q` rotated by the small rotation vector `w`.
pub(crate) fn quat_add(q: Quat, w: Vec3) -> Quat {
    (q + Quat::from_xyzw(w.x, w.y, w.z, 0.0) * q * 0.5).normalize()
}

/// The `w` for which `quat_add(b, w)` is `a`. Velocities come from this, so it
/// must be the exact inverse: anything else damps spin in proportion to ω².
pub(crate) fn quat_step(a: Quat, b: Quat) -> Vec3 {
    let d = a * b.conjugate();
    // Near a half turn per step, `w` tends to infinity. Keep it finite.
    let w = if d.w.abs() < 1.0e-3 {
        1.0e-3f32.copysign(d.w)
    } else {
        d.w
    };
    d.xyz() * (2.0 / w)
}

/// Rotation vector taking `b` to `a`, valid for small differences.
pub(crate) fn quat_sub(a: Quat, b: Quat) -> Vec3 {
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
}

impl Contact {
    pub fn new(feature: u32, ra: Vec3, rb: Vec3) -> Self {
        Self { feature, ra, rb }
    }
}

pub(crate) struct Manifold {
    a: usize,
    b: usize,
    /// Normal (B to A) then two tangents as matrix rows: the shared linear
    /// Jacobian.
    rows: Mat3A,
    /// Keep collision metadata out of the repeatedly visited solver array.
    cold: Box<CollisionData>,
    /// Whether A is solved before B each iteration, and so applies the
    /// previous iteration's dual update (see `Solver::step`).
    first_is_a: bool,
    len: usize,
    /// Kinetic and static friction coefficients.
    friction: f32,
    static_friction: f32,
    /// The larger of the bodies' restitutions.
    restitution: f32,
    /// Heavier body's mass: penalties are per kilogram so tuning is unit-free.
    scale: f32,
    /// The largest penalty that helps this pair converge.
    limit: f32,
    /// Two free bodies meeting fast: see `Body::finish`.
    impact: bool,
    /// The contacts' solver state, one SIMD lane per contact.
    lanes: Lanes,
}

struct CollisionData {
    key: (u64, u64),
    /// Basis and relative pose when the contacts were generated.
    local_basis: [Vec3; 3],
    relative: (Vec3, Quat),
    contacts: [Contact; collide::MAX_CONTACTS],
    /// The contacts' anchors in SIMD lanes: each step's `prepare` needs them,
    /// but contacts change far less often.
    anchors: [[Vec4; 3]; 2],
    /// Step-start constraint error, which only stabilization corrects.
    error: [Vec4; 3],
    /// Unregularized step-start penetration exceeds the sleep tolerance.
    deep: bool,
    /// The narrowphase pass that last matched this manifold to a pair.
    seen: u32,
    /// Step-start normal velocity of A from B at each contact, negative when
    /// closing: what restitution reverses. Only kept with restitution.
    approach: Vec4,
}

/// A manifold's contacts in structure-of-arrays form for the solver: each
/// `Vec4` holds one quantity for all four contacts, so a manifold is
/// evaluated and stamped four contacts per instruction. `[r]` indexes the
/// basis row (normal, tangent, tangent) and `[r][m]` an angular component.
/// Unused lanes are inactive and contribute exactly zero.
#[derive(Clone, Copy, Default)]
struct Lanes {
    /// Step-start constraint error with Eq. 18 already applied.
    c0: [Vec4; 3],
    /// World anchors at the start of the step, giving the angular Jacobians
    /// (Sec. 4): linearized once, since re-evaluating at the current rotation
    /// double counts rotation. Storing the Jacobians instead costs more in
    /// cache misses, once a scene outgrows the cache, than crossing them does.
    ra: [Vec4; 3],
    rb: [Vec4; 3],
    penalty: [Vec4; 3],
    lambda: [Vec4; 3],
    /// 1 or 0 per lane: a real contact; static friction applies this step;
    /// the normal row may be stiffness-rescaled.
    active: Vec4,
    sticking: Vec4,
    rescale_normal: Vec4,
    /// One bit per lane: the dual update's stick and held results, retained
    /// for the next step. Bits, because the solver array must fit in cache.
    stick: u8,
    held: u8,
}

/// 1 or 0 in each lane for the bits of the index.
const LANE_BITS: [Vec4; 16] = {
    let mut table = [Vec4::ZERO; 16];
    let mut bits = 0;
    while bits < 16 {
        table[bits] = Vec4::new(
            (bits & 1) as f32,
            (bits >> 1 & 1) as f32,
            (bits >> 2 & 1) as f32,
            (bits >> 3 & 1) as f32,
        );
        bits += 1;
    }
    table
};

#[derive(Default)]
struct Warm {
    penalty: [Vec4; 3],
    lambda: [Vec4; 3],
    stick: u8,
    held: u8,
}

impl Lanes {
    fn set_warm(&mut self, warm: Warm) {
        self.penalty = warm.penalty;
        self.lambda = warm.lambda;
        self.stick = warm.stick;
        self.held = warm.held;
    }
}

/// Eq. 18: step-start error with `alpha` of it left uncorrected. A gap is not
/// an error: speculative contacts must keep it, or they would stop bodies early.
fn regularize(c: [Vec4; 3], alpha: f32) -> [Vec4; 3] {
    let keep = 1.0 - alpha;
    [
        c[0].max(Vec4::ZERO) + c[0].min(Vec4::ZERO) * keep,
        c[1] * keep,
        c[2] * keep,
    ]
}

/// Each contact's anchors as lanes, for `CollisionData::anchors`.
fn anchor_lanes(contacts: &[Contact]) -> [[Vec4; 3]; 2] {
    let (mut ra, mut rb) = ([Vec4::ZERO; 3], [Vec4::ZERO; 3]);
    for (i, c) in contacts.iter().enumerate() {
        for k in 0..3 {
            ra[k][i] = c.ra[k];
            rb[k][i] = c.rb[k];
        }
    }
    [ra, rb]
}

/// Lane-wise angular Hessian (upper triangle: 00 01 02 11 12 22) and
/// gradient summed over a body's manifolds, reduced once per body. Each sum
/// has two halves, one per manifold of a pair, adjacent so that a 256-bit
/// register loads both: paired stamps match unpaired ones bit for bit.
#[derive(Default)]
struct LaneSums {
    ang: [[Vec4; 2]; 6],
    rhs: [[Vec4; 2]; 3],
}

/// The sums of four vectors' lanes, as one vector.
fn sum4(a: Vec4, b: Vec4, c: Vec4, d: Vec4) -> Vec4 {
    let t = Mat4::from_cols(a, b, c, d).transpose();
    t.x_axis + t.y_axis + t.z_axis + t.w_axis
}

/// Lane-wise square root (glam's `Vec4` has none).
fn sqrt4(v: Vec4) -> Vec4 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE is part of the x86-64 baseline, so the CPU has it.
        Vec4::from(unsafe { std::arch::x86_64::_mm_sqrt_ps(v.into()) })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec4::from_array(v.to_array().map(f32::sqrt))
    }
}

/// Rotate four vectors with the orientation already cached for this step.
fn rotate4(m: Mat3, v: [Vec4; 3]) -> [Vec4; 3] {
    [
        v[0] * m.x_axis.x + v[1] * m.y_axis.x + v[2] * m.z_axis.x,
        v[0] * m.x_axis.y + v[1] * m.y_axis.y + v[2] * m.z_axis.y,
        v[0] * m.x_axis.z + v[1] * m.y_axis.z + v[2] * m.z_axis.z,
    ]
}

/// Four vectors, one per lane, crossed with `n`.
#[inline(always)]
fn cross4(a: [Vec4; 3], n: Vec3) -> [Vec4; 3] {
    [
        a[1] * n.z - a[2] * n.y,
        a[2] * n.x - a[0] * n.z,
        a[0] * n.y - a[1] * n.x,
    ]
}

/// Four contacts' linearized constraints (Sec. 4) and clamped forces.
struct Eval {
    c: [Vec4; 3],
    /// The force before its bounds were applied (λ⁺, Eq. 13).
    raw: [Vec4; 3],
    force: [Vec4; 3],
    /// What the Coulomb cone scaled the tangential force by: one inside it.
    cone: Vec4,
    /// Lane masks; persistent friction state is materialized only on update.
    held: BVec4A,
    inside: BVec4A,
}

/// Both bodies' displacement since the step began, shared by a manifold's contacts.
struct Motion {
    linear: Vec3A,
    angular_a: Vec3A,
    angular_b: Vec3A,
}

impl Motion {
    #[inline(always)]
    fn new(a: &Body, b: &Body) -> Self {
        Self {
            linear: a.moved - b.moved,
            angular_a: a.turned,
            angular_b: b.turned,
        }
    }
}

/// `m` with row `k` scaled by `s[k]`.
fn scale_rows(m: Mat3A, s: Vec3A) -> Mat3A {
    Mat3A::from_cols(m.x_axis * s, m.y_axis * s, m.z_axis * s)
}

impl Manifold {
    #[inline(always)]
    fn constraint(&self, m: &Motion) -> [Vec4; 3] {
        let l = &self.lanes;
        // The linear rows are the shared basis: one value per row for all lanes.
        let linear = self.rows * m.linear;
        let (ta, tb) = (m.angular_a, m.angular_b);
        // J θ is n · (θ × r): one cross per endpoint serves all three rows.
        let (ra, rb) = (&l.ra, &l.rb);
        let w = [
            (ra[2] * ta.y - ra[1] * ta.z) - (rb[2] * tb.y - rb[1] * tb.z),
            (ra[0] * ta.z - ra[2] * ta.x) - (rb[0] * tb.z - rb[2] * tb.x),
            (ra[1] * ta.x - ra[0] * ta.y) - (rb[1] * tb.x - rb[0] * tb.y),
        ];
        let rows = &self.rows;
        let c = std::array::from_fn(|r| {
            l.c0[r]
                + Vec4::splat(linear[r])
                + (w[0] * rows.x_axis[r] + w[1] * rows.y_axis[r] + w[2] * rows.z_axis[r])
        });
        // Every lane contributes when the manifold has four contacts.
        if self.len == 4 {
            c
        } else {
            c.map(|v| v * l.active)
        }
    }

    #[inline(always)]
    fn eval_constraint(&self, c: [Vec4; 3]) -> Eval {
        let l = &self.lanes;
        let raw: [Vec4; 3] = std::array::from_fn(|r| l.penalty[r] * c[r] + l.lambda[r]);
        let fx = raw[0].min(Vec4::ZERO);
        let normal = -fx;
        let tangent_squared = raw[1] * raw[1] + raw[2] * raw[2];
        // Test the cone before taking a norm: sticking contacts need neither
        // a square root nor a division. Keep the original overflow behavior
        // when the squared tangential force is infinite.
        let infinity = Vec4::splat(f32::INFINITY);
        let within = |bound: Vec4, squared: Vec4| {
            tangent_squared.cmple(squared)
                & (tangent_squared.cmpne(infinity) | bound.cmpeq(infinity))
        };
        let static_bound = normal * self.static_friction;
        let static_squared = static_bound * static_bound;
        let held = within(static_bound, static_squared);
        // Equal coefficients have the same cone regardless of sticking. Reuse
        // its test and squared bound, including for zero and infinite bounds.
        // Compare bits so differently signed zero coefficients stay distinct.
        let (bound, bound_squared, inside) =
            if self.static_friction.to_bits() == self.friction.to_bits() {
                (static_bound, static_squared, held)
            } else {
                let use_static = l.sticking.cmpgt(Vec4::ZERO) & held;
                let bound = Vec4::select(use_static, static_bound, normal * self.friction);
                let squared = bound * bound;
                (bound, squared, within(bound, squared))
            };
        let over = (tangent_squared.cmpgt(bound_squared)
            | (tangent_squared.cmpeq(infinity) & bound.cmplt(infinity)))
            & tangent_squared.cmpgt(Vec4::ZERO);
        let scale = if over.any() {
            let tangent = sqrt4(tangent_squared);
            Vec4::select(
                over,
                bound / tangent.max(Vec4::splat(f32::MIN_POSITIVE)),
                Vec4::ONE,
            )
        } else {
            Vec4::ONE
        };
        Eval {
            c,
            raw,
            force: [fx, raw[1] * scale, raw[2] * scale],
            cone: scale,
            held,
            inside,
        }
    }

    #[inline(always)]
    fn eval(&self, m: &Motion) -> Eval {
        self.eval_constraint(self.constraint(m))
    }

    /// Accumulate this manifold's force and Hessian approximation for `body`
    /// (Eqs. 5, 6), four contacts at a time. The angular terms stay lane-wise
    /// in `sums` until the body's solve; the linear rows are the shared basis,
    /// so the linear block, coupling and gradient sum once per manifold.
    /// `half` selects which of `sums`' halves this manifold adds to.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn stamp(
        &self,
        a: &Body,
        b: &Body,
        is_a: bool,
        c: Option<[Vec4; 3]>,
        damping: f32,
        system: &mut System,
        sums: &mut LaneSums,
        half: usize,
    ) {
        let c = c.unwrap_or_else(|| self.constraint(&Motion::new(a, b)));
        let e = self.eval_constraint(c);
        let mut force = e.force;
        let l = &self.lanes;
        // A sliding contact's force is capped by the cone, so its stiffness is
        // too: the full penalty would make each Newton step apply only a
        // fraction of the friction. This is the secant of the capped force, as
        // in NVIDIA Newton's AVBD. Rescale the normal row too, except for a
        // contact that began the step penetrating: rescaling there can make even iteration counts
        // alternate between separation and falling back through the contact.
        let mut k = [l.penalty[0], l.penalty[1] * e.cone, l.penalty[2] * e.cone];
        let clamped = e.force[0].cmpne(e.raw[0])
            & e.c[0].cmpne(Vec4::ZERO)
            & l.rescale_normal.cmpgt(Vec4::ZERO);
        if clamped.any() {
            let rescaled = ((e.force[0] - l.lambda[0]) / e.c[0]).clamp(Vec4::ZERO, l.penalty[0]);
            k[0] = Vec4::select(clamped, rescaled, k[0]);
        }
        // rb × (−n) = n × rb, so B's Jacobians cross with the negated basis.
        let (anchor, basis) = if is_a {
            (l.ra, self.rows.transpose())
        } else {
            (l.rb, self.rows.transpose() * -1.0)
        };
        let j = [
            cross4(anchor, basis.x_axis.into()),
            cross4(anchor, basis.y_axis.into()),
            cross4(anchor, basis.z_axis.into()),
        ];
        // Contact damping, VBD's Rayleigh term (Chen et al. 2024): a pushing
        // normal row resists this body's own motion along it this step, with
        // stiffness (k_d / h) k. Corrections otherwise bounce stacks.
        if damping > 0.0 {
            let (body, sign) = if is_a { (a, 1.0) } else { (b, -1.0) };
            // Damping only needs the normal projection, not all three rows.
            let moved = body.moved;
            let along = (self.rows.x_axis.x * moved.x
                + self.rows.y_axis.x * moved.y
                + self.rows.z_axis.x * moved.z)
                * sign;
            let t = body.turned;
            let motion = Vec4::splat(along) + j[0][0] * t.x + j[0][1] * t.y + j[0][2] * t.z;
            let pushing = force[0].cmplt(Vec4::ZERO);
            let kd = Vec4::select(pushing, k[0] * damping, Vec4::ZERO);
            // Damping never pulls a contact closed.
            force[0] = (force[0] + kd * motion).min(Vec4::ZERO);
            k[0] += kd;
        }
        // Stiffness-weighted angular rows K J, per row and component.
        let mut kj = [[Vec4::ZERO; 3]; 3];
        for r in 0..3 {
            let (jr, f) = (&j[r], force[r]);
            kj[r] = [k[r] * jr[0], k[r] * jr[1], k[r] * jr[2]];
            let w = &kj[r];
            sums.ang[0][half] += w[0] * jr[0];
            sums.ang[1][half] += w[0] * jr[1];
            sums.ang[2][half] += w[0] * jr[2];
            sums.ang[3][half] += w[1] * jr[1];
            sums.ang[4][half] += w[1] * jr[2];
            sums.ang[5][half] += w[2] * jr[2];
            sums.rhs[0][half] += jr[0] * f;
            sums.rhs[1][half] += jr[1] * f;
            sums.rhs[2][half] += jr[2] * f;
        }
        self.add_block(
            is_a,
            [
                sum4(k[0], k[1], k[2], force[0]),
                sum4(force[1], force[2], kj[0][0], kj[0][1]),
                sum4(kj[0][2], kj[1][0], kj[1][1], kj[1][2]),
                sum4(kj[2][0], kj[2][1], kj[2][2], Vec4::ZERO),
            ],
            system,
        );
    }

    /// The linear block, coupling and gradient from a manifold's reduced
    /// stiffness `k`, force and K J, packed in that order.
    #[inline(always)]
    fn add_block(&self, is_a: bool, [s1, s2, s3, s4]: [Vec4; 4], system: &mut System) {
        let stiffness = Vec3A::new(s1.x, s1.y, s1.z);
        let force = Vec3A::new(s1.w, s2.x, s2.y);
        // Columns are the contact-basis rows of K J.
        let weighted_transpose = Mat3A::from_cols(
            Vec3A::new(s2.z, s2.w, s3.x),
            Vec3A::new(s3.y, s3.z, s3.w),
            Vec3A::new(s4.x, s4.y, s4.z),
        );
        let sign = if is_a { 1.0 } else { -1.0 };
        let bt = self.rows.transpose();
        system.lin += bt * scale_rows(self.rows, stiffness);
        system.cross += weighted_transpose * self.rows * sign;
        system.rhs_lin += bt * force * sign;
    }

    #[cfg(target_arch = "x86_64")]
    avx2_and_avx512! {
    /// `dual`, where its flag is set, then `stamp`, for two manifolds of body
    /// `i`, one per 256-bit half. Every lane repeats those
    /// functions' operations in the same order, so results match them bit for
    /// bit; do not contract them into FMAs.
    ///
    /// # Safety
    /// The CPU must support the function's features.
    #[allow(clippy::too_many_arguments)]
    unsafe fn stamp_pair_avx2 | stamp_pair_avx512(
        p: &mut Self,
        q: &mut Self,
        [pa, pb]: [&Body; 2],
        [qa, qb]: [&Body; 2],
        i: usize,
        [dual_p, dual_q]: [bool; 2],
        beta: f32,
        damping: f32,
        system: &mut System,
        sums: &mut LaneSums,
    ) {
        use std::arch::x86_64::*;
        let join = |a: Vec4, b: Vec4| _mm256_set_m128(b.into(), a.into());
        let splat2 = |a: f32, b: f32| _mm256_set_m128(_mm_set1_ps(b), _mm_set1_ps(a));
        let mul = |a, b| _mm256_mul_ps(a, b);
        let add = |a, b| _mm256_add_ps(a, b);
        let and = |a, b| _mm256_and_ps(a, b);
        let or = |a, b| _mm256_or_ps(a, b);
        let le = |a, b| _mm256_cmp_ps::<_CMP_LE_OS>(a, b);
        let lt = |a, b| _mm256_cmp_ps::<_CMP_LT_OS>(a, b);
        let gt = |a, b| _mm256_cmp_ps::<_CMP_GT_OS>(a, b);
        let eq = |a, b| _mm256_cmp_ps::<_CMP_EQ_OQ>(a, b);
        let ne = |a, b| _mm256_cmp_ps::<_CMP_NEQ_UQ>(a, b);
        let zero = _mm256_setzero_ps();
        let one = _mm256_set1_ps(1.0);
        let infinity = _mm256_set1_ps(f32::INFINITY);

        let (mp, mq) = (Motion::new(pa, pb), Motion::new(qa, qb));
        let wide = |a: Vec3A, b: Vec3A| _mm256_set_m128(b.into(), a.into());
        // A matrix times `v` in each half, summing as glam does.
        let times = |m: &[__m256; 3], v| {
            let x = mul(m[0], _mm256_permute_ps::<0x00>(v));
            let y = mul(m[1], _mm256_permute_ps::<0x55>(v));
            add(add(x, y), mul(m[2], _mm256_permute_ps::<0xAA>(v)))
        };
        let rows = [
            wide(p.rows.x_axis, q.rows.x_axis),
            wide(p.rows.y_axis, q.rows.y_axis),
            wide(p.rows.z_axis, q.rows.z_axis),
        ];
        let sign = splat2(
            if i == p.a { 1.0 } else { -1.0 },
            if i == q.a { 1.0 } else { -1.0 },
        );
        let linear = times(&rows, wide(mp.linear, mq.linear));
        let linear = [
            _mm256_permute_ps::<0x00>(linear),
            _mm256_permute_ps::<0x55>(linear),
            _mm256_permute_ps::<0xAA>(linear),
        ];
        let (ta, tb) = (
            wide(mp.angular_a, mq.angular_a),
            wide(mp.angular_b, mq.angular_b),
        );
        let (ax, ay, az) = (
            _mm256_permute_ps::<0x00>(ta),
            _mm256_permute_ps::<0x55>(ta),
            _mm256_permute_ps::<0xAA>(ta),
        );
        let (bx, by, bz) = (
            _mm256_permute_ps::<0x00>(tb),
            _mm256_permute_ps::<0x55>(tb),
            _mm256_permute_ps::<0xAA>(tb),
        );
        let (lp, lq) = (&p.lanes, &q.lanes);
        let active = join(lp.active, lq.active);
        let sub = |a, b| _mm256_sub_ps(a, b);
        let ra = [
            join(lp.ra[0], lq.ra[0]),
            join(lp.ra[1], lq.ra[1]),
            join(lp.ra[2], lq.ra[2]),
        ];
        let rb = [
            join(lp.rb[0], lq.rb[0]),
            join(lp.rb[1], lq.rb[1]),
            join(lp.rb[2], lq.rb[2]),
        ];
        let w = [
            sub(
                sub(mul(ra[2], ay), mul(ra[1], az)),
                sub(mul(rb[2], by), mul(rb[1], bz)),
            ),
            sub(
                sub(mul(ra[0], az), mul(ra[2], ax)),
                sub(mul(rb[0], bz), mul(rb[2], bx)),
            ),
            sub(
                sub(mul(ra[1], ax), mul(ra[0], ay)),
                sub(mul(rb[1], bx), mul(rb[0], by)),
            ),
        ];
        // The basis vectors' components, one per half: `n[k][r]` is row r's k.
        let n = [
            [
                _mm256_permute_ps::<0x00>(rows[0]),
                _mm256_permute_ps::<0x55>(rows[0]),
                _mm256_permute_ps::<0xAA>(rows[0]),
            ],
            [
                _mm256_permute_ps::<0x00>(rows[1]),
                _mm256_permute_ps::<0x55>(rows[1]),
                _mm256_permute_ps::<0xAA>(rows[1]),
            ],
            [
                _mm256_permute_ps::<0x00>(rows[2]),
                _mm256_permute_ps::<0x55>(rows[2]),
                _mm256_permute_ps::<0xAA>(rows[2]),
            ],
        ];
        // Plain loops: closures passed to `from_fn` or `map` would leave this
        // function's target features and pass every vector through memory.
        let mut c = [zero; 3];
        let mut penalty = [zero; 3];
        let mut lambda = [zero; 3];
        for r in 0..3 {
            let angular = add(
                add(mul(w[0], n[0][r]), mul(w[1], n[1][r])),
                mul(w[2], n[2][r]),
            );
            let v = add(add(join(lp.c0[r], lq.c0[r]), linear[r]), angular);
            c[r] = mul(v, active);
            penalty[r] = join(lp.penalty[r], lq.penalty[r]);
            lambda[r] = join(lp.lambda[r], lq.lambda[r]);
        }
        let static_friction = splat2(p.static_friction, q.static_friction);
        let friction = splat2(p.friction, q.friction);
        let sticking = join(lp.sticking, lq.sticking);
        let same = |m: &Self| m.static_friction.to_bits() == m.friction.to_bits();
        let same = same(p) && same(q);
        // `eval_constraint`: raw normal, forces, and the held and inside masks.
        let eval = |penalty: &[__m256; 3], lambda: &[__m256; 3]| {
            let raw = [
                add(mul(penalty[0], c[0]), lambda[0]),
                add(mul(penalty[1], c[1]), lambda[1]),
                add(mul(penalty[2], c[2]), lambda[2]),
            ];
            let fx = _mm256_min_ps(raw[0], zero);
            let normal = _mm256_xor_ps(fx, _mm256_set1_ps(-0.0));
            let tangent_squared = add(mul(raw[1], raw[1]), mul(raw[2], raw[2]));
            let within = |bound, squared| {
                and(
                    le(tangent_squared, squared),
                    or(ne(tangent_squared, infinity), eq(bound, infinity)),
                )
            };
            let static_bound = mul(normal, static_friction);
            let static_squared = mul(static_bound, static_bound);
            let held = within(static_bound, static_squared);
            let (bound, bound_squared, inside) = if same {
                (static_bound, static_squared, held)
            } else {
                let use_static = and(gt(sticking, zero), held);
                let bound = _mm256_blendv_ps(mul(normal, friction), static_bound, use_static);
                let squared = mul(bound, bound);
                (bound, squared, within(bound, squared))
            };
            let over = and(
                or(
                    gt(tangent_squared, bound_squared),
                    and(eq(tangent_squared, infinity), lt(bound, infinity)),
                ),
                gt(tangent_squared, zero),
            );
            let scale = if _mm256_movemask_ps(over) != 0 {
                let tangent = _mm256_sqrt_ps(tangent_squared);
                let limit = _mm256_max_ps(tangent, _mm256_set1_ps(f32::MIN_POSITIVE));
                _mm256_blendv_ps(one, _mm256_div_ps(bound, limit), over)
            } else {
                one
            };
            (
                raw[0],
                [fx, mul(raw[1], scale), mul(raw[2], scale)],
                held,
                inside,
                scale,
            )
        };
        let (mut raw0, mut force, held, inside, mut cone) = eval(&penalty, &lambda);
        if dual_p || dual_q {
            let beta = splat2(beta * p.scale, beta * q.scale);
            let max = splat2(p.limit, q.limit);
            let bound = splat2(LAMBDA_MAX * p.scale, LAMBDA_MAX * q.scale);
            let minimum = _mm256_xor_ps(bound, _mm256_set1_ps(-0.0));
            let abs = |v| _mm256_andnot_ps(_mm256_set1_ps(-0.0), v);
            let ramp = |r: usize| {
                let v: __m256 = c[r];
                _mm256_min_ps(add(penalty[r], mul(abs(v), beta)), max)
            };
            let pushing = lt(force[0], zero);
            let slip_squared = add(mul(c[1], c[1]), mul(c[2], c[2]));
            let threshold = _mm256_set1_ps(STICK_THRESHOLD * STICK_THRESHOLD);
            let stuck = _mm256_movemask_ps(lt(slip_squared, threshold)) as u8;
            let (held, inside_bits) = (
                _mm256_movemask_ps(held) as u8,
                _mm256_movemask_ps(inside) as u8,
            );
            // Each flag selects a half: the other manifold's dual waits for
            // its own first visitor.
            let update = _mm256_set_m128(
                _mm_castsi128_ps(_mm_set1_epi32(-i32::from(dual_q))),
                _mm_castsi128_ps(_mm_set1_epi32(-i32::from(dual_p))),
            );
            let mut new_penalty = [zero; 3];
            let mut new_lambda = [zero; 3];
            new_penalty[0] = _mm256_blendv_ps(penalty[0], ramp(0), pushing);
            new_penalty[1] = _mm256_blendv_ps(penalty[1], ramp(1), inside);
            new_penalty[2] = _mm256_blendv_ps(penalty[2], ramp(2), inside);
            for r in 0..3 {
                new_lambda[r] = _mm256_min_ps(_mm256_max_ps(force[r], minimum), bound);
                new_penalty[r] = _mm256_blendv_ps(penalty[r], new_penalty[r], update);
                new_lambda[r] = _mm256_blendv_ps(lambda[r], new_lambda[r], update);
            }
            (penalty, lambda) = (new_penalty, new_lambda);
            let low = |v| Vec4::from(_mm256_castps256_ps128(v));
            let high = |v| Vec4::from(_mm256_extractf128_ps::<1>(v));
            if dual_p {
                let l = &mut p.lanes;
                for r in 0..3 {
                    (l.penalty[r], l.lambda[r]) = (low(penalty[r]), low(lambda[r]));
                }
                let (inside, stuck) = (inside_bits & 15, stuck & 15);
                (l.stick, l.held) = (l.stick & !inside | stuck & inside, held & 15);
            }
            if dual_q {
                let l = &mut q.lanes;
                for r in 0..3 {
                    (l.penalty[r], l.lambda[r]) = (high(penalty[r]), high(lambda[r]));
                }
                let (inside, stuck) = (inside_bits >> 4, stuck >> 4);
                (l.stick, l.held) = (l.stick & !inside | stuck & inside, held >> 4);
            }
            let evaluated = eval(&penalty, &lambda);
            (raw0, force, cone) = (evaluated.0, evaluated.1, evaluated.4);
        }
        let (lp, lq) = (&p.lanes, &q.lanes);
        let mut fx = force[0];

        let mut k = [penalty[0], mul(penalty[1], cone), mul(penalty[2], cone)];
        let clamped = and(
            and(ne(fx, raw0), ne(c[0], zero)),
            gt(join(lp.rescale_normal, lq.rescale_normal), zero),
        );
        if _mm256_movemask_ps(clamped) != 0 {
            let rescaled = _mm256_div_ps(_mm256_sub_ps(fx, lambda[0]), c[0]);
            let rescaled = _mm256_min_ps(_mm256_max_ps(rescaled, zero), penalty[0]);
            k[0] = _mm256_blendv_ps(k[0], rescaled, clamped);
        }
        // `cross4` of this body's anchors with its signed basis, each half.
        let anchor = {
            let (ap, aq) = (
                if i == p.a { &lp.ra } else { &lp.rb },
                if i == q.a { &lq.ra } else { &lq.rb },
            );
            [join(ap[0], aq[0]), join(ap[1], aq[1]), join(ap[2], aq[2])]
        };
        let mut j = [[zero; 3]; 3];
        for r in 0..3 {
            let (x, y, z) = (mul(n[0][r], sign), mul(n[1][r], sign), mul(n[2][r], sign));
            j[r] = [
                sub(mul(anchor[1], z), mul(anchor[2], y)),
                sub(mul(anchor[2], x), mul(anchor[0], z)),
                sub(mul(anchor[0], y), mul(anchor[1], x)),
            ];
        }
        if damping > 0.0 {
            let body = if i == p.a { pa } else { pb };
            let along = times(&rows, wide(body.moved, body.moved));
            let along = mul(_mm256_permute_ps::<0x00>(along), sign);
            let t = body.turned;
            let mut motion = add(along, mul(j[0][0], _mm256_set1_ps(t.x)));
            motion = add(motion, mul(j[0][1], _mm256_set1_ps(t.y)));
            motion = add(motion, mul(j[0][2], _mm256_set1_ps(t.z)));
            let pushing = lt(fx, zero);
            let kd = _mm256_blendv_ps(zero, mul(k[0], _mm256_set1_ps(damping)), pushing);
            fx = _mm256_min_ps(add(fx, mul(kd, motion)), zero);
            k[0] = add(k[0], kd);
        }
        let force = [fx, force[1], force[2]];

        // SAFETY: each sum's two halves are 32 contiguous bytes.
        let load = |v: &[Vec4; 2]| unsafe { _mm256_loadu_ps(v.as_ptr().cast()) };
        let mut ang = [zero; 6];
        for (v, sum) in ang.iter_mut().zip(&sums.ang) {
            *v = load(sum);
        }
        let mut rhs = [zero; 3];
        for (v, sum) in rhs.iter_mut().zip(&sums.rhs) {
            *v = load(sum);
        }
        let mut kj = [[zero; 3]; 3];
        for r in 0..3 {
            let (jr, f) = (&j[r], force[r]);
            kj[r] = [mul(k[r], jr[0]), mul(k[r], jr[1]), mul(k[r], jr[2])];
            let w = &kj[r];
            ang[0] = add(ang[0], mul(w[0], jr[0]));
            ang[1] = add(ang[1], mul(w[0], jr[1]));
            ang[2] = add(ang[2], mul(w[0], jr[2]));
            ang[3] = add(ang[3], mul(w[1], jr[1]));
            ang[4] = add(ang[4], mul(w[1], jr[2]));
            ang[5] = add(ang[5], mul(w[2], jr[2]));
            rhs[0] = add(rhs[0], mul(jr[0], f));
            rhs[1] = add(rhs[1], mul(jr[1], f));
            rhs[2] = add(rhs[2], mul(jr[2], f));
        }
        // SAFETY: as for `load`.
        let store = |to: &mut [Vec4; 2], v| unsafe { _mm256_storeu_ps(to.as_mut_ptr().cast(), v) };
        for (sum, v) in sums.ang.iter_mut().zip(ang) {
            store(sum, v);
        }
        for (sum, v) in sums.rhs.iter_mut().zip(rhs) {
            store(sum, v);
        }

        // `sum4` in each half.
        let sum4 = |a, b, c, d| {
            let t0 = _mm256_shuffle_ps::<0b01_00_01_00>(a, b);
            let t1 = _mm256_shuffle_ps::<0b11_10_11_10>(a, b);
            let t2 = _mm256_shuffle_ps::<0b01_00_01_00>(c, d);
            let t3 = _mm256_shuffle_ps::<0b11_10_11_10>(c, d);
            let x = _mm256_shuffle_ps::<0b10_00_10_00>(t0, t2);
            let y = _mm256_shuffle_ps::<0b11_01_11_01>(t0, t2);
            let z = _mm256_shuffle_ps::<0b10_00_10_00>(t1, t3);
            let w = _mm256_shuffle_ps::<0b11_01_11_01>(t1, t3);
            add(add(add(x, y), z), w)
        };
        let stiffness = sum4(k[0], k[1], k[2], zero);
        let force = sum4(force[0], force[1], force[2], zero);
        // Columns are the contact-basis rows of K J.
        let weighted = [
            sum4(kj[0][0], kj[0][1], kj[0][2], zero),
            sum4(kj[1][0], kj[1][1], kj[1][2], zero),
            sum4(kj[2][0], kj[2][1], kj[2][2], zero),
        ];

        // `add_block` in each half.
        let (t0, t1) = (
            _mm256_unpacklo_ps(rows[0], rows[1]),
            _mm256_unpackhi_ps(rows[0], rows[1]),
        );
        let transposed = [
            _mm256_shuffle_ps::<0b00_00_01_00>(t0, rows[2]),
            _mm256_shuffle_ps::<0b01_01_11_10>(t0, rows[2]),
            _mm256_shuffle_ps::<0b10_10_01_00>(t1, rows[2]),
        ];
        let mut block = [zero; 7];
        for n in 0..3 {
            block[n] = times(&transposed, mul(rows[n], stiffness));
            block[n + 3] = mul(times(&weighted, rows[n]), sign);
        }
        block[6] = mul(times(&transposed, force), sign);
        let low = |v| Vec3A::from(_mm256_castps256_ps128(v));
        let high = |v| Vec3A::from(_mm256_extractf128_ps::<1>(v));
        let both = |to: &mut Vec3A, v| *to = *to + low(v) + high(v);
        both(&mut system.lin.x_axis, block[0]);
        both(&mut system.lin.y_axis, block[1]);
        both(&mut system.lin.z_axis, block[2]);
        both(&mut system.cross.x_axis, block[3]);
        both(&mut system.cross.y_axis, block[4]);
        both(&mut system.cross.z_axis, block[5]);
        both(&mut system.rhs_lin, block[6]);
    }
    }

    /// The force and torque of this manifold's stored multipliers on body A
    /// or B, in the convention of `System`'s gradient.
    fn wrench(&self, is_a: bool) -> (Vec3A, Vec3A) {
        let l = &self.lanes;
        let (anchor, basis, sign) = if is_a {
            (l.ra, self.rows.transpose(), 1.0)
        } else {
            (l.rb, self.rows.transpose() * -1.0, -1.0)
        };
        let total = Vec3A::new(
            l.lambda[0].element_sum(),
            l.lambda[1].element_sum(),
            l.lambda[2].element_sum(),
        );
        let (mut x, mut y, mut z) = (0.0, 0.0, 0.0);
        for (lambda, axis) in l
            .lambda
            .iter()
            .zip([basis.x_axis, basis.y_axis, basis.z_axis])
        {
            let j = cross4(anchor, axis.into());
            x += (j[0] * *lambda).element_sum();
            y += (j[1] * *lambda).element_sum();
            z += (j[2] * *lambda).element_sum();
        }
        (self.rows.transpose() * total * sign, Vec3A::new(x, y, z))
    }

    /// Show the stabilization sweep the error it should correct (Eq. 18).
    fn restore_error(&mut self, alpha: f32) {
        self.lanes.c0 = regularize(self.cold.error, alpha);
    }

    /// Store forces and ramp penalties (Eqs. 11, 12), four contacts at a time.
    #[inline(always)]
    fn dual(&mut self, a: &Body, b: &Body, beta: f32) -> [Vec4; 3] {
        let e = self.eval(&Motion::new(a, b));
        let (beta, max) = (beta * self.scale, Vec4::splat(self.limit));
        let bound = Vec4::splat(LAMBDA_MAX * self.scale);
        let l = &mut self.lanes;
        for r in 0..3 {
            l.lambda[r] = e.force[r].clamp(-bound, bound);
        }
        l.held = e.held.bitmask() as u8;
        let pushing = e.force[0].cmplt(Vec4::ZERO);
        l.penalty[0] = Vec4::select(
            pushing,
            (l.penalty[0] + e.c[0].abs() * beta).min(max),
            l.penalty[0],
        );
        let inside = e.inside;
        for r in 1..3 {
            l.penalty[r] = Vec4::select(
                inside,
                (l.penalty[r] + e.c[r].abs() * beta).min(max),
                l.penalty[r],
            );
        }
        let slip_squared = e.c[1] * e.c[1] + e.c[2] * e.c[2];
        let stuck = slip_squared.cmplt(Vec4::splat(STICK_THRESHOLD * STICK_THRESHOLD));
        let inside = inside.bitmask() as u8;
        l.stick = l.stick & !inside | stuck.bitmask() as u8 & inside;
        e.c
    }

    /// A manifold for a new pair, warm started from `previous` (a sleeping
    /// island's contacts); `None` when the shapes do not touch.
    fn create(
        bodies: &[Body],
        a: usize,
        b: usize,
        previous: Option<&Manifold>,
        prepare: Option<&PhysicsSettings>,
        ground: Option<&dyn Ground>,
    ) -> Option<Self> {
        let (ba, bb) = (&bodies[a], &bodies[b]);
        let (basis, mut contacts, len) = collide::collide(ba, bb, ground)?;
        let warm = if let Some(previous) = previous {
            previous.warm(ba, bb, &mut contacts[..len])
        } else {
            Warm {
                held: (1 << len) - 1,
                ..Warm::default()
            }
        };
        let mut lanes = Lanes::default();
        lanes.set_warm(warm);
        let inverse = ba.rotation.conjugate();
        let mut m = Manifold {
            a,
            b,
            rows: Mat3A::from_cols(basis[0].into(), basis[1].into(), basis[2].into()).transpose(),
            cold: Box::new(CollisionData {
                key: (ba.key, bb.key),
                local_basis: basis.map(|v| inverse * v),
                relative: (inverse * (bb.position - ba.position), inverse * bb.rotation),
                contacts,
                anchors: anchor_lanes(&contacts[..len]),
                error: [Vec4::ZERO; 3],
                deep: false,
                seen: 0,
                approach: Vec4::ZERO,
            }),
            first_is_a: true,
            len,
            friction: 0.0,
            static_friction: 0.0,
            restitution: 0.0,
            scale: 0.0,
            limit: 0.0,
            impact: false,
            lanes,
        };
        if let Some(settings) = prepare {
            m.prepare(ba, bb, basis, settings);
        }
        Some(m)
    }

    /// Bring a kept manifold up to this step: reuse its contacts while B
    /// stays near the pose they were made at, else collide again, warm
    /// starting from the old contacts. False once the shapes separate.
    fn refresh(
        &mut self,
        bodies: &[Body],
        prepare: Option<&PhysicsSettings>,
        ground: Option<&dyn Ground>,
    ) -> bool {
        let (ba, bb) = (&bodies[self.a], &bodies[self.b]);
        let inverse = ba.rotation.conjugate();
        let relative = (inverse * (bb.position - ba.position), inverse * bb.rotation);
        let reusable = self.cold.relative.0.distance_squared(relative.0)
            < REUSE_DISTANCE * REUSE_DISTANCE
            && quat_sub(relative.1, self.cold.relative.1).length_squared()
                < REUSE_ANGLE * REUSE_ANGLE;
        let basis = if reusable {
            self.cold.local_basis.map(|v| ba.rotation * v)
        } else {
            let Some((basis, mut contacts, len)) = collide::collide(ba, bb, ground) else {
                return false;
            };
            if len == self.len
                && contacts[..len]
                    .iter()
                    .zip(&self.cold.contacts[..len])
                    .all(|(new, old)| new.feature == old.feature)
            {
                // Identical feature order already has the correct solver lanes.
                // Keep sticking anchors, except for rolling spheres.
                if !ba.rolls() && !bb.rolls() {
                    for (i, (new, old)) in contacts[..len]
                        .iter_mut()
                        .zip(&self.cold.contacts[..len])
                        .enumerate()
                    {
                        if self.lanes.stick >> i & 1 != 0 {
                            (new.ra, new.rb) = (old.ra, old.rb);
                        }
                    }
                }
            } else {
                let warm = self.warm(ba, bb, &mut contacts[..len]);
                self.lanes.set_warm(warm);
            }
            (self.cold.contacts, self.len) = (contacts, len);
            self.cold.anchors = anchor_lanes(&contacts[..len]);
            self.cold.local_basis = basis.map(|v| inverse * v);
            self.cold.relative = relative;
            basis
        };
        self.rows = Mat3A::from_cols(basis[0].into(), basis[1].into(), basis[2].into()).transpose();
        if let Some(settings) = prepare {
            self.prepare(ba, bb, basis, settings);
        }
        true
    }

    /// Carry matching contacts' forces, penalties and friction state (and,
    /// while static friction held, their anchors) to freshly collided ones.
    fn warm(&self, ba: &Body, bb: &Body, contacts: &mut [Contact]) -> Warm {
        // A rolling sphere's contact point moves over its surface: its
        // anchors never stick, only its forces warm start.
        let rolls = ba.rolls() || bb.rolls();
        let mut warm = Warm::default();
        for (i, contact) in contacts.iter_mut().enumerate() {
            // A new contact starts static and turns kinetic once it slides.
            warm.held |= 1 << i;
            let matched = self.cold.contacts[..self.len]
                .iter()
                .enumerate()
                .find(|(_, p)| p.feature == contact.feature);
            if let Some((j, previous)) = matched {
                let stick = self.lanes.stick >> j & 1;
                if stick != 0 && !rolls {
                    (contact.ra, contact.rb) = (previous.ra, previous.rb);
                }
                for k in 0..3 {
                    warm.penalty[k][i] = self.lanes.penalty[k][j];
                    warm.lambda[k][i] = self.lanes.lambda[k][j];
                }
                warm.stick |= stick << i;
                warm.held = warm.held & !(1 << i) | (self.lanes.held >> j & 1) << i;
            }
        }
        warm
    }

    /// The step's constraint errors, Jacobians, materials and warm-started
    /// penalties and duals (Eq. 19), four contacts at a time in the lanes.
    fn prepare(&mut self, ba: &Body, bb: &Body, basis: [Vec3; 3], settings: &PhysicsSettings) {
        let (alpha, gamma) = (settings.alpha, settings.gamma);
        let scale = ba.mass.max(bb.mass);
        let l = &mut self.lanes;
        let live = (1u8 << self.len) - 1;
        l.active = LANE_BITS[live as usize];
        let [ra, rb] = self.cold.anchors;
        // Dual state already lives in SIMD lanes. Clear unused contacts so
        // their previous values cannot contaminate this step's arithmetic.
        let active = l.active.cmpgt(Vec4::ZERO);
        l.penalty = l.penalty.map(|v| Vec4::select(active, v, Vec4::ZERO));
        l.lambda = l.lambda.map(|v| Vec4::select(active, v, Vec4::ZERO));
        l.stick &= live;
        l.held &= live;
        l.sticking = LANE_BITS[l.held as usize];
        let (ra, rb) = (rotate4(ba.initial_basis, ra), rotate4(bb.initial_basis, rb));
        let gap = ba.position - bb.position;
        let d: [Vec4; 3] =
            std::array::from_fn(|k| (Vec4::splat(gap[k]) + ra[k] - rb[k]) * l.active);
        (l.ra, l.rb) = (ra, rb);
        for (r, n) in basis.iter().enumerate() {
            l.c0[r] = (d[0] * n.x + d[1] * n.y + d[2] * n.z) * l.active;
        }
        // Between two free bodies a much stiffer penalty only locks them: the
        // first to move satisfies the contact alone, the other barely moves,
        // and momentum is lost. Half the heavier body's inertial term is the
        // reduced mass of equal bodies. A static partner has no such limit.
        let (ka, kb) = (ba.stiffness, bb.stiffness);
        let limit = if ka > 0.0 && kb > 0.0 {
            (0.5 * ka.max(kb)).min(PENALTY_MAX * scale)
        } else {
            PENALTY_MAX * scale
        };
        self.limit = limit;
        let closing = (bb.travel - ba.travel).dot(basis[0]);
        self.impact = ka > 0.0 && kb > 0.0 && closing > IMPACT_SPEED * (ba.mass / ka).sqrt();
        // A new contact starts near the body's own inertia: from PENALTY_MIN it
        // spends sweeps ramping up, and slopes creep while it does.
        let start = (0.25 * ka.max(kb)).min(limit);
        let (min, max) = (
            Vec4::splat((PENALTY_MIN * scale).max(start)),
            Vec4::splat(limit),
        );
        for k in 0..3 {
            // With post-stabilization a contact's force carries over whole. Any
            // decay here must be rebuilt every step, and tall stacks cannot.
            l.penalty[k] = (l.penalty[k] * gamma).clamp(min, max) * l.active;
        }
        // Only a touching contact: one far from its surface, on either side,
        // alternates between overshooting and springing back to its target.
        let touching = l.c0[0].cmpge(Vec4::ZERO) & l.c0[0].cmple(Vec4::splat(CONTACT_DISTANCE));
        l.rescale_normal = Vec4::select(touching, l.active, Vec4::ZERO);
        self.cold.deep = l.c0[0].cmplt(Vec4::splat(-SLEEP_PENETRATION)).any();
        self.cold.error = l.c0;
        l.c0 = regularize(l.c0, alpha);
        self.friction = (ba.friction * bb.friction).sqrt();
        self.static_friction = (ba.static_friction * bb.static_friction).sqrt();
        self.restitution = ba.restitution.max(bb.restitution);
        if self.restitution > 0.0 {
            let (va, vb) = (ba.velocity, bb.velocity);
            let (wa, wb) = (ba.angular_velocity, bb.angular_velocity);
            for i in 0..self.len {
                let lane = |r: &[Vec4; 3]| Vec3::new(r[0][i], r[1][i], r[2][i]);
                let relative = va + wa.cross(lane(&l.ra)) - vb - wb.cross(lane(&l.rb));
                self.cold.approach[i] = relative.dot(basis[0]);
            }
        }
        self.scale = scale;
    }
}

/// One body's 6x6 SPD system: linear block, angular block, angular-by-linear
/// coupling, and the gradient.
pub(crate) struct System {
    pub lin: Mat3A,
    pub ang: Mat3A,
    pub cross: Mat3A,
    pub rhs_lin: Vec3A,
    pub rhs_ang: Vec3A,
}

impl System {
    const IDENTITY: Self = Self {
        lin: Mat3A::IDENTITY,
        ang: Mat3A::IDENTITY,
        cross: Mat3A::ZERO,
        rhs_lin: Vec3A::ZERO,
        rhs_ang: Vec3A::ZERO,
    };

    /// Transpose whole spatial columns, avoiding scalar gathers for each
    /// coefficient of the six-by-six block.
    #[inline(always)]
    fn pack_four(s: &[Self]) -> ([[Vec4; 6]; 6], [Vec4; 6]) {
        macro_rules! transpose {
            ($($field:ident).+) => {
                Mat4::from_cols(
                    s[0].$($field).+.extend(0.0),
                    s[1].$($field).+.extend(0.0),
                    s[2].$($field).+.extend(0.0),
                    s[3].$($field).+.extend(0.0),
                ).transpose()
            };
        }
        let (a0, a1, a2) = (
            transpose!(lin.x_axis),
            transpose!(lin.y_axis),
            transpose!(lin.z_axis),
        );
        let (b0, b1, b2) = (
            transpose!(cross.x_axis),
            transpose!(cross.y_axis),
            transpose!(cross.z_axis),
        );
        let (c0, c1, c2) = (
            transpose!(ang.x_axis),
            transpose!(ang.y_axis),
            transpose!(ang.z_axis),
        );
        let (linear, angular) = (transpose!(rhs_lin), transpose!(rhs_ang));
        let z = Vec4::ZERO;
        (
            [
                [a0.x_axis, z, z, z, z, z],
                [a0.y_axis, a1.y_axis, z, z, z, z],
                [a0.z_axis, a1.z_axis, a2.z_axis, z, z, z],
                [b0.x_axis, b1.x_axis, b2.x_axis, c0.x_axis, z, z],
                [b0.y_axis, b1.y_axis, b2.y_axis, c0.y_axis, c1.y_axis, z],
                [
                    b0.z_axis, b1.z_axis, b2.z_axis, c0.z_axis, c1.z_axis, c2.z_axis,
                ],
            ],
            [
                -linear.x_axis,
                -linear.y_axis,
                -linear.z_axis,
                -angular.x_axis,
                -angular.y_axis,
                -angular.z_axis,
            ],
        )
    }

    /// Shared fixed-size factorization, specialized to each SIMD width.
    /// Inlining specializes arithmetic within the caller's instruction-set
    /// boundary, avoiding out-of-line calls to the lane operations.
    #[inline(always)]
    fn ldlt<T: Copy>(
        mut l: [[T; 6]; 6],
        mut x: [T; 6],
        zero: T,
        reciprocal: impl Fn(T) -> T,
        multiply: impl Fn(T, T) -> T,
        subtract: impl Fn(T, T) -> T,
    ) -> [T; 6] {
        let mut inverse_d = [zero; 6];
        macro_rules! eliminate {
            ($j:literal; $($i:literal),*) => {
                inverse_d[$j] = reciprocal(l[$j][$j]);
                $(
                    let v = multiply(l[$i][$j], inverse_d[$j]);
                    x[$i] = subtract(x[$i], multiply(v, x[$j]));
                    for k in $j + 1..=$i {
                        l[$i][k] = subtract(l[$i][k], multiply(v, l[k][$j]));
                    }
                    l[$i][$j] = v;
                )*
            };
        }
        // Literal pivots keep this fixed-size elimination unrolled.
        // Descending rows retain the unnormalized column until its last use.
        eliminate!(0; 5, 4, 3, 2, 1);
        eliminate!(1; 5, 4, 3, 2);
        eliminate!(2; 5, 4, 3);
        eliminate!(3; 5, 4);
        eliminate!(4; 5);
        eliminate!(5;);
        macro_rules! substitute {
            ($i:literal; $($j:literal),*) => {
                x[$i] = multiply(x[$i], inverse_d[$i]);
                $(x[$i] = subtract(x[$i], multiply(l[$j][$i], x[$j]));)*
            };
        }
        substitute!(5;);
        substitute!(4; 5);
        substitute!(3; 4, 5);
        substitute!(2; 3, 4, 5);
        substitute!(1; 2, 3, 4, 5);
        substitute!(0; 1, 2, 3, 4, 5);
        x
    }

    /// Four independent body blocks, one per SIMD lane.
    fn solve_four(s: &[Self; 4]) -> [Vec4; 6] {
        let (l, x) = Self::pack_four(s);
        Self::ldlt(l, x, Vec4::ZERO, |v| v.recip(), |a, b| a * b, |a, b| a - b)
    }

    /// Solve `A x = -rhs` (Eq. 4) by block elimination of the SPD system
    /// `[lin, crossᵀ; cross, ang]`: the linear block, then its Schur
    /// complement, each a 3x3 SIMD inverse.
    fn solve(&self) -> (Vec3, Vec3) {
        // Penalties reach 1e13, so a cofactor inverse would overflow f32:
        // invert each block scaled by its trace, then scale back.
        let inverse = |m: Mat3A| {
            let s = 1.0 / (m.x_axis.x + m.y_axis.y + m.z_axis.z);
            (m * s).inverse() * s
        };
        let lin_inverse = inverse(self.lin);
        let coupling = self.cross * lin_inverse;
        let schur = self.ang - coupling * self.cross.transpose();
        let y = inverse(schur) * (coupling * self.rhs_lin - self.rhs_ang);
        let x = lin_inverse * (-self.rhs_lin - self.cross.transpose() * y);
        (x.into(), y.into())
    }

    #[cfg(target_arch = "x86_64")]
    avx2_and_avx512! {
    /// The same LDLᵀ arithmetic as `solve_four`, eight bodies per register.
    ///
    /// # Safety
    /// The CPU must support the function's features.
    unsafe fn solve_eight | solve_eight_avx512(s: &[Self; 8], rotations: &[Quat; 8]) -> [[f32; 8]; 7] {
        use std::arch::x86_64::*;
        unsafe {
            // Lane `n` of each output comes from row `n` of the eight inputs.
            macro_rules! transpose {
                ($v:expr) => {{
                    let v = $v;
                    let pair =
                        |a: usize| -> __m256 { _mm256_set_m128(v[a + 4].into(), v[a].into()) };
                    let (v0, v1, v2, v3) = (pair(0), pair(1), pair(2), pair(3));
                    let (t0, t1) = (_mm256_unpacklo_ps(v0, v1), _mm256_unpackhi_ps(v0, v1));
                    let (t2, t3) = (_mm256_unpacklo_ps(v2, v3), _mm256_unpackhi_ps(v2, v3));
                    [
                        _mm256_shuffle_ps::<0b01_00_01_00>(t0, t2),
                        _mm256_shuffle_ps::<0b11_10_11_10>(t0, t2),
                        _mm256_shuffle_ps::<0b01_00_01_00>(t1, t3),
                        _mm256_shuffle_ps::<0b11_10_11_10>(t1, t3),
                    ]
                }};
            }
            macro_rules! field {
                ($($field:ident).+) => {
                    transpose!([
                        s[0].$($field).+, s[1].$($field).+, s[2].$($field).+, s[3].$($field).+,
                        s[4].$($field).+, s[5].$($field).+, s[6].$($field).+, s[7].$($field).+,
                    ])
                };
            }
            let (a0, a1, a2) = (field!(lin.x_axis), field!(lin.y_axis), field!(lin.z_axis));
            let (b0, b1, b2) = (
                field!(cross.x_axis),
                field!(cross.y_axis),
                field!(cross.z_axis),
            );
            let (c0, c1, c2) = (field!(ang.x_axis), field!(ang.y_axis), field!(ang.z_axis));
            let (linear, angular) = (field!(rhs_lin), field!(rhs_ang));
            let z = _mm256_setzero_ps();
            let l = [
                [a0[0], z, z, z, z, z],
                [a0[1], a1[1], z, z, z, z],
                [a0[2], a1[2], a2[2], z, z, z],
                [b0[0], b1[0], b2[0], c0[0], z, z],
                [b0[1], b1[1], b2[1], c0[1], c1[1], z],
                [b0[2], b1[2], b2[2], c0[2], c1[2], c2[2]],
            ];
            let negate = |v| _mm256_xor_ps(v, _mm256_set1_ps(-0.0));
            let x = [
                negate(linear[0]),
                negate(linear[1]),
                negate(linear[2]),
                negate(angular[0]),
                negate(angular[1]),
                negate(angular[2]),
            ];
            let one = _mm256_set1_ps(1.0);
            let x = Self::ldlt(
                l,
                x,
                z,
                |v| _mm256_div_ps(one, v),
                |a, b| _mm256_mul_ps(a, b),
                |a, b| _mm256_sub_ps(a, b),
            );
            // Apply and normalize all eight quaternion updates in SIMD.
            // Match glam's quaternion multiplication and dot-product order.
            let [qx, qy, qz, qw] = transpose!(rotations);
            let (rx, ry, rz) = (x[3], x[4], x[5]);
            let half = _mm256_set1_ps(0.5);
            let mul = |a, b| _mm256_mul_ps(a, b);
            let add = |a, b| _mm256_add_ps(a, b);
            let sub = |a, b| _mm256_sub_ps(a, b);
            let nx = add(
                qx,
                mul(half, add(mul(rx, qw), sub(mul(ry, qz), mul(rz, qy)))),
            );
            let ny = add(
                qy,
                mul(
                    half,
                    add(sub(z, mul(rx, qz)), add(mul(ry, qw), mul(rz, qx))),
                ),
            );
            let nz = add(
                qz,
                mul(half, add(mul(rx, qy), sub(mul(rz, qw), mul(ry, qx)))),
            );
            let nw = add(
                qw,
                mul(
                    half,
                    sub(sub(z, mul(rx, qx)), add(mul(ry, qy), mul(rz, qz))),
                ),
            );
            let length = _mm256_sqrt_ps(add(
                add(mul(nx, nx), mul(nz, nz)),
                add(mul(ny, ny), mul(nw, nw)),
            ));
            let normalized = [
                _mm256_div_ps(nx, length),
                _mm256_div_ps(ny, length),
                _mm256_div_ps(nz, length),
                _mm256_div_ps(nw, length),
            ];
            let mut output = [[0.0; 8]; 7];
            for i in 0..3 {
                _mm256_storeu_ps(output[i].as_mut_ptr(), x[i]);
            }
            for i in 0..4 {
                _mm256_storeu_ps(output[i + 3].as_mut_ptr(), normalized[i]);
            }
            output
        }
    }
    }
}

/// Assemble one body's block against its incident manifolds and links.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn assemble_body<'a>(
    body_at: impl Fn(usize) -> &'a Body,
    manifolds: Shared<Manifold>,
    link_at: impl Fn(usize) -> &'a Link,
    adjacency: &[u32],
    i: usize,
    dual: bool,
    visit: bool,
    settings: &PhysicsSettings,
    damping: f32,
    scale: f32,
    #[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))] level: u8,
    system: &mut System,
) {
    let body = body_at(i);
    let mass = body.mass * scale;
    let inertia = body.inertia * scale;
    *system = System {
        lin: Mat3A::from_diagonal(Vec3::splat(mass)),
        ang: inertia,
        cross: Mat3A::ZERO,
        rhs_lin: Vec3A::from(body.position - body.inertial_position) * mass,
        rhs_ang: inertia * Vec3A::from(quat_step(body.rotation, body.inertial_rotation)),
    };
    let mut sums = LaneSums::default();
    // SAFETY (all accesses below): per `solve_batch_with`, only this body
    // uses its manifolds now, and each appears once in its adjacency.
    macro_rules! stamp {
        ($k:expr, $first:expr, $half:expr) => {{
            let m = unsafe { &mut *manifolds.get().add($k) };
            let (a, b) = (body_at(m.a), body_at(m.b));
            // The first visitor applies the previous iteration's dual update:
            // neither body has moved since, so the stamp can reuse the error,
            // though not the force, which depends on the updated duals.
            let c = if $first {
                Some(m.dual(a, b, settings.beta))
            } else {
                None
            };
            m.stamp(a, b, i == m.a, c, damping, system, &mut sums, $half);
        }};
    }
    // Manifolds are stamped two at a time, one per sum half,
    // whether or not the CPU can do so at once: every CPU sums in one order.
    let mut pending = None;
    for &entry in adjacency {
        if entry & LINK != 0 {
            let link = link_at((entry & !LINK) as usize);
            let a = link.a.map(&body_at);
            link.stamp(
                a,
                body_at(link.b),
                link.a == Some(i),
                settings.alpha,
                system,
            );
        } else {
            let k = entry as usize;
            let m = unsafe { &*manifolds.get().add(k) };
            // Every visit updates the dual, not just each iteration's first:
            // the second body must feel the force the first body just used,
            // or a stiff contact stops one body and barely moves the other,
            // and momentum is lost.
            let first = dual || (visit && i != if m.first_is_a { m.a } else { m.b });
            let Some((other, other_first)) = pending.take() else {
                pending = Some((k, first));
                continue;
            };
            #[cfg(target_arch = "x86_64")]
            if level != BASE {
                // SAFETY: `level` comes from runtime CPU detection. `k` and
                // `other` differ, each appearing once in the adjacency.
                unsafe {
                    let p = &mut *manifolds.get().add(other);
                    let q = &mut *manifolds.get().add(k);
                    let (pa, pb, qa, qb) = (body_at(p.a), body_at(p.b), body_at(q.a), body_at(q.b));
                    let stamp = if level == AVX512 {
                        Manifold::stamp_pair_avx512
                    } else {
                        Manifold::stamp_pair_avx2
                    };
                    stamp(
                        p,
                        q,
                        [pa, pb],
                        [qa, qb],
                        i,
                        [other_first, first],
                        settings.beta,
                        damping,
                        system,
                        &mut sums,
                    );
                }
                continue;
            }
            stamp!(other, other_first, 0);
            stamp!(k, first, 1);
        }
    }
    if let Some((k, first)) = pending {
        stamp!(k, first, 0);
    }
    let total = |v: [Vec4; 2]| v[0] + v[1];
    let (ang, rhs) = (sums.ang.map(total), sums.rhs.map(total));
    let h = sum4(ang[0], ang[1], ang[2], ang[3]);
    let g = sum4(ang[4], ang[5], rhs[0], rhs[1]);
    system.ang += Mat3A::from_cols(
        Vec3A::new(h.x, h.y, h.z),
        Vec3A::new(h.y, h.w, g.x),
        Vec3A::new(h.z, g.x, g.y),
    );
    system.rhs_ang += Vec3A::new(g.z, g.w, rhs[2].element_sum());
}

/// The sum of body `i`'s contact forces if it is in an impact; `None` if not,
/// or if a joint or spring also acts on it, whose forces are not among them.
fn contact_wrench(
    manifolds: &[Manifold],
    offsets: &[u32],
    adjacency: &[u32],
    i: usize,
) -> Option<(Vec3A, Vec3A)> {
    let (mut force, mut torque, mut impact) = (Vec3A::ZERO, Vec3A::ZERO, false);
    for &entry in adjacency.get(offsets[i] as usize..offsets[i + 1] as usize)? {
        if entry & LINK != 0 {
            return None;
        }
        let m = &manifolds[entry as usize];
        let (f, t) = m.wrench(i == m.a);
        (force, torque) = (force + f, torque + t);
        impact |= m.impact;
    }
    impact.then_some((force, torque))
}

/// Apply the previous iteration's dual update at a link's first visit.
/// Neither endpoint has moved since the preceding iteration ended.
///
/// # Safety
/// The caller is `i`'s thread in `i`'s color phase: only `i` may write the
/// links it visits first, and no thread writes the bodies they read. No
/// reference to the visited link may remain live across this call.
#[inline(always)]
unsafe fn first_link_dual<'a>(
    body_at: impl Fn(usize) -> &'a Body,
    links: Shared<Link>,
    k: usize,
    i: usize,
    settings: &PhysicsSettings,
) {
    let link = unsafe { &mut *links.get().add(k) };
    let first = if link.first_is_a {
        link.a
    } else {
        Some(link.b)
    };
    if first == Some(i) {
        link.dual(link.a.map(&body_at), body_at(link.b), settings);
    }
}

/// Start loading the manifolds of a body about to be assembled: colors visit
/// them out of memory order. The first lines are enough for the hardware
/// prefetcher to stream the rest.
#[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
#[inline(always)]
fn prefetch_manifolds(manifolds: Shared<Manifold>, adjacency: &[u32]) {
    #[cfg(target_arch = "x86_64")]
    for &entry in adjacency {
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        if entry & LINK == 0 {
            let m = manifolds.get().wrapping_add(entry as usize).cast::<i8>();
            for line in 0..PREFETCH_LINES {
                // SAFETY: prefetching has no effect on memory, valid or not.
                unsafe { _mm_prefetch::<_MM_HINT_T0>(m.wrapping_add(line * 64)) };
            }
        }
    }
}

/// Which compilation of the hot path to run. All give bit-identical results.
const BASE: u8 = 0;
#[cfg(target_arch = "x86_64")]
const AVX2: u8 = 1;
#[cfg(target_arch = "x86_64")]
const AVX512: u8 = 2;

/// The best level this CPU has. The standard library caches the detection.
fn cpu_level() -> u8 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512vl")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512dq")
        {
            return AVX512;
        }
        if std::is_x86_feature_detected!("avx2") {
            return AVX2;
        }
    }
    BASE
}

fn batch_width() -> usize {
    if cpu_level() == BASE { 4 } else { 8 }
}

/// Solve up to eight bodies of one color, writing poses only after assembly.
///
/// # Safety
/// Each body in `indices` is unique and shares no constraint with another
/// body in this color. The caller owns these bodies and their first-visited
/// constraints for the current color phase; other colors cannot run yet.
/// `LEVEL` additionally requires that level's CPU support.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn solve_batch_with<const LEVEL: u8>(
    bodies: Shared<Body>,
    manifolds: Shared<Manifold>,
    links: Shared<Link>,
    offsets: &[u32],
    adjacency: &[u32],
    indices: &[usize],
    settings: &PhysicsSettings,
    scale: f32,
    damping: f32,
    iteration: u32,
) {
    let body_at = |j| unsafe { &*bodies.get().add(j) };
    // A macro, not a closure: closures would not inherit the caller's target
    // features unless inlined, and only functions can force that.
    macro_rules! assemble {
        ($i:expr, $system:expr) => {{
            let i = $i;
            assemble_body(
                body_at,
                manifolds,
                |k| unsafe {
                    if iteration > 0 && iteration != STABILIZE {
                        first_link_dual(body_at, links, k, i, settings);
                    }
                    &*links.get().add(k)
                },
                &adjacency[offsets[i] as usize..offsets[i + 1] as usize],
                i,
                iteration > 0 && iteration != STABILIZE,
                iteration != STABILIZE,
                settings,
                damping,
                scale,
                LEVEL,
                $system,
            )
        }};
    }
    #[cfg(target_arch = "x86_64")]
    if LEVEL != BASE && indices.len() == 8 {
        let mut systems = std::array::from_fn(|_| System::IDENTITY);
        for (n, system) in systems.iter_mut().enumerate() {
            if let Some(&next) = indices.get(n + 1) {
                prefetch_manifolds(
                    manifolds,
                    &adjacency[offsets[next] as usize..offsets[next + 1] as usize],
                );
            }
            assemble!(indices[n], system);
        }
        let rotations = std::array::from_fn(|lane| body_at(indices[lane]).rotation);
        // SAFETY: the level comes from runtime CPU detection.
        let update = unsafe {
            if LEVEL == AVX512 {
                System::solve_eight_avx512(&systems, &rotations)
            } else {
                System::solve_eight(&systems, &rotations)
            }
        };
        for (lane, &i) in indices.iter().enumerate() {
            let dx = Vec3::new(update[0][lane], update[1][lane], update[2][lane]);
            let rotation = Quat::from_xyzw(
                update[3][lane],
                update[4][lane],
                update[5][lane],
                update[6][lane],
            );
            let body = unsafe { &mut *bodies.get().add(i) };
            body.place(body.position + dx, rotation);
        }
        return;
    }
    // Partial batches take the four-body and singleton paths.
    for chunk in indices.chunks(4) {
        if let [i] = *chunk {
            let mut system = System::IDENTITY;
            assemble!(i, &mut system);
            let (dx, dr) = system.solve();
            // No references to these bodies escape assembly.
            let body = unsafe { &mut *bodies.get().add(i) };
            body.place(body.position + dx, quat_add(body.rotation, dr));
            continue;
        }
        let mut systems = std::array::from_fn(|_| System::IDENTITY);
        for (system, &i) in systems.iter_mut().zip(chunk) {
            assemble!(i, system);
        }
        let update = System::solve_four(&systems);
        for (lane, &i) in chunk.iter().enumerate() {
            let dx = Vec3::new(update[0][lane], update[1][lane], update[2][lane]);
            let dr = Vec3::new(update[3][lane], update[4][lane], update[5][lane]);
            // All references used for assembly are dead before the first write.
            let body = unsafe { &mut *bodies.get().add(i) };
            body.place(body.position + dx, quat_add(body.rotation, dr));
        }
    }
}

/// `solve_batch_with` compiled for AVX2, so its inlined arithmetic is too.
///
/// # Safety
/// As for `solve_batch_with::<AVX2>`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn solve_batch_avx2(
    bodies: Shared<Body>,
    manifolds: Shared<Manifold>,
    links: Shared<Link>,
    offsets: &[u32],
    adjacency: &[u32],
    indices: &[usize],
    settings: &PhysicsSettings,
    scale: f32,
    damping: f32,
    iteration: u32,
) {
    unsafe {
        solve_batch_with::<AVX2>(
            bodies, manifolds, links, offsets, adjacency, indices, settings, scale, damping,
            iteration,
        );
    }
}

/// `solve_batch_with` with AVX-512's 32 registers, which spill less. It stays
/// 256 bits wide: PERFORMANCE.md records why 512-bit stamping lost.
///
/// # Safety
/// As for `solve_batch_with::<AVX512>`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512f,avx512vl,avx512bw,avx512dq")]
#[allow(clippy::too_many_arguments)]
unsafe fn solve_batch_avx512(
    bodies: Shared<Body>,
    manifolds: Shared<Manifold>,
    links: Shared<Link>,
    offsets: &[u32],
    adjacency: &[u32],
    indices: &[usize],
    settings: &PhysicsSettings,
    scale: f32,
    damping: f32,
    iteration: u32,
) {
    unsafe {
        solve_batch_with::<AVX512>(
            bodies, manifolds, links, offsets, adjacency, indices, settings, scale, damping,
            iteration,
        );
    }
}

/// `solve_batch_with` for this CPU; its contract applies except for the CPU
/// level, which this detects.
#[allow(clippy::too_many_arguments)]
unsafe fn solve_batch(
    bodies: Shared<Body>,
    manifolds: Shared<Manifold>,
    links: Shared<Link>,
    offsets: &[u32],
    adjacency: &[u32],
    indices: &[usize],
    settings: &PhysicsSettings,
    scale: f32,
    damping: f32,
    iteration: u32,
) {
    #[cfg(target_arch = "x86_64")]
    match cpu_level() {
        // SAFETY: the level comes from runtime CPU detection.
        AVX512 => unsafe {
            solve_batch_avx512(
                bodies, manifolds, links, offsets, adjacency, indices, settings, scale, damping,
                iteration,
            );
            return;
        },
        // SAFETY: as above.
        AVX2 => unsafe {
            solve_batch_avx2(
                bodies, manifolds, links, offsets, adjacency, indices, settings, scale, damping,
                iteration,
            );
            return;
        },
        _ => {}
    }
    unsafe {
        solve_batch_with::<BASE>(
            bodies, manifolds, links, offsets, adjacency, indices, settings, scale, damping,
            iteration,
        );
    }
}

/// A raw pointer the parallel iterations share; see `iterate_parallel`.
struct Shared<T>(*mut T);

// By hand: the derives would require `T: Copy`.
impl<T> Clone for Shared<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Shared<T> {}

impl<T> Shared<T> {
    // A method, not field access, so closures capture the Sync wrapper.
    fn get(self) -> *mut T {
        self.0
    }
}

// SAFETY: `iterate_parallel` partitions every access; see there.
unsafe impl<T> Send for Shared<T> {}
unsafe impl<T> Sync for Shared<T> {}

/// A barrier for the pool's threads. Each phase of an iteration is a few
/// microseconds, so threads spin briefly instead of sleeping on a condvar.
struct SpinBarrier {
    arrived: std::sync::atomic::AtomicUsize,
    generation: std::sync::atomic::AtomicUsize,
    threads: usize,
}

impl SpinBarrier {
    fn new(threads: usize) -> Self {
        Self {
            arrived: Default::default(),
            generation: Default::default(),
            threads,
        }
    }

    fn wait(&self) {
        use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release};
        let generation = self.generation.load(Acquire);
        if self.arrived.fetch_add(1, AcqRel) + 1 == self.threads {
            self.arrived.store(0, Relaxed);
            self.generation.fetch_add(1, Release);
            return;
        }
        let mut spins = 0u32;
        while self.generation.load(Acquire) == generation {
            if spins < 1 << 12 {
                std::hint::spin_loop();
                spins += 1;
            } else {
                // Oversubscribed cores: let the late thread run.
                std::thread::yield_now();
            }
        }
    }
}

/// A body's world bounds for the broadphase sweep.
#[derive(Clone, Copy)]
struct Aabb {
    min: Vec3,
    max: Vec3,
    index: u32,
    /// Dynamic, active and asleep bits.
    flags: u32,
}

/// Four sorted sweep boxes. The candidate loop tests four complete AABBs
/// per SIMD instruction, then visits only the overlapping lanes in order.
struct SweepBlock {
    min: [Vec4; 3],
    max: [Vec4; 3],
    index: [u32; 4],
    flags: [u32; 4],
    valid: u32,
}

impl SweepBlock {
    fn new(boxes: &[Aabb]) -> Self {
        Self {
            min: std::array::from_fn(|axis| {
                Vec4::from_array(std::array::from_fn(|lane| {
                    boxes.get(lane).map_or(f32::INFINITY, |b| b.min[axis])
                }))
            }),
            max: std::array::from_fn(|axis| {
                Vec4::from_array(std::array::from_fn(|lane| {
                    boxes.get(lane).map_or(f32::NEG_INFINITY, |b| b.max[axis])
                }))
            }),
            index: std::array::from_fn(|lane| boxes.get(lane).map_or(0, |b| b.index)),
            flags: std::array::from_fn(|lane| boxes.get(lane).map_or(0, |b| b.flags)),
            valid: (1 << boxes.len()) - 1,
        }
    }

    fn overlaps(&self, min: &[Vec4; 3], max: &[Vec4; 3]) -> u32 {
        let mask = self.min[0].cmple(max[0])
            & self.max[0].cmpge(min[0])
            & self.min[1].cmple(max[1])
            & self.max[1].cmpge(min[1])
            & self.min[2].cmple(max[2])
            & self.max[2].cmpge(min[2]);
        mask.bitmask() & self.valid
    }
}

#[derive(Default)]
pub(crate) struct Solver {
    /// What [`Collider::Ground`] bodies are.
    ground: Option<Arc<dyn Ground>>,
    /// The bodies (key, solved) and links (ends, solved) the coloring was
    /// built for, and whether manifolds changed since: unchanged, the
    /// coloring and adjacency are reused.
    colored: Vec<(u64, bool)>,
    colored_links: Vec<(Option<usize>, usize, bool)>,
    recolor: bool,
    /// Each body's tight box this step, and its enlarged box with the key of
    /// the body it was made for: the broadphase reuses pairs while bodies
    /// stay inside their enlarged boxes.
    tight: Vec<Aabb>,
    fat: Vec<Aabb>,
    fat_keys: Vec<u64>,
    pub bodies: Vec<Body>,
    /// Links that yielded or broke in this step, by index.
    pub joint_changes: Vec<(usize, crate::JointStage)>,
    manifolds: Vec<Manifold>,
    /// Manifold index by body-key pair, for warm starting the next step.
    cache: PairMap,
    /// Manifolds of sleeping islands, tagged by island, kept to warm start
    /// them on waking; `stash_index` finds them by key.
    stash: Vec<(u32, Manifold)>,
    stash_index: PairMap,
    next_island: u32,
    /// Shared temporary storage for coloring and sleeping islands.
    scratch_usize: Vec<usize>,
    rest: Vec<f32>,
    scratch_u32: Vec<u32>,
    /// Joints and springs, kept in step by the physics world.
    pub links: Vec<Link>,
    /// Body-key pairs a link joins: they do not collide.
    linked: PairMap,
    boxes: Vec<Aabb>,
    sweep_blocks: Vec<SweepBlock>,
    swept: Vec<Vec<(usize, usize)>>,
    pairs: Vec<(usize, usize)>,
    /// Manifolds touching each body, as CSR offsets into `adjacency`.
    offsets: Vec<u32>,
    adjacency: Vec<u32>,
    colors: Vec<Vec<usize>>,
    /// Dynamic bodies touching nothing: their solve is the inertial target.
    free: Vec<usize>,
    /// New pairs and their manifolds, reused across narrowphase passes.
    fresh: Vec<(usize, usize)>,
    found: Vec<Option<Manifold>>,
    /// Counts narrowphase passes, to spot manifolds no pair matched.
    pass: u32,
    /// Time in the broadphase and narrowphase during the last step.
    pub broad_time: std::time::Duration,
    pub narrow_time: std::time::Duration,
}

impl Solver {
    pub(crate) fn ground(&self) -> Option<Arc<dyn Ground>> {
        self.ground.clone()
    }

    /// Replace the ground. Contacts inside `changed` come from it afresh; ones
    /// elsewhere stay, as the ground under them is the same.
    pub(crate) fn set_ground(&mut self, ground: Arc<dyn Ground>, changed: Bounds) {
        self.ground = Some(ground);
        let bodies = &self.bodies;
        for manifold in &mut self.manifolds {
            let (a, b) = (&bodies[manifold.a], &bodies[manifold.b]);
            let on_ground = a.shape == Collider::Ground || b.shape == Collider::Ground;
            let body = if a.shape == Collider::Ground { b } else { a };
            let reach = Vec3::splat(body.bound());
            if on_ground
                && (body.position + reach).cmpge(changed.min).all()
                && (body.position - reach).cmple(changed.max).all()
            {
                // Too far from any pose to reuse.
                manifold.cold.relative.0 = Vec3::splat(UNBOUNDED);
            }
        }
    }

    pub fn contact_count(&self) -> usize {
        let stashed = self.stash.iter().map(|(_, m)| m);
        self.manifolds.iter().chain(stashed).map(|m| m.len).sum()
    }

    /// `exclusive`: the step runs alone on its own thread pool, so the
    /// iterations may hold every pool thread at a barrier. On a shared pool
    /// two worlds' barriers could wait on each other's threads forever.
    pub fn step(&mut self, settings: &PhysicsSettings, dt: f32, exclusive: bool, serial: bool) {
        // Post-stabilization, from the AVBD authors' reference solver: the main
        // sweeps leave step-start error alone, velocities are taken from them,
        // and a last sweep then removes the error from positions alone, so
        // correcting penetration adds no energy.
        let stabilize = settings.clone();
        let settings = &PhysicsSettings {
            alpha: 1.0,
            ..settings.clone()
        };
        let sweeps = settings.iterations.saturating_sub(1).max(1);
        let mut has_sleepers = false;
        let requested: Vec<u32> = self
            .bodies
            .iter()
            .filter_map(|b| {
                has_sleepers |= b.asleep;
                (b.asleep && b.wake).then_some(b.island)
            })
            .collect();
        let mut woken = self.wake_islands(&requested);
        self.broad_time = std::time::Duration::ZERO;
        self.narrow_time = std::time::Duration::ZERO;
        self.linked.clear();
        for link in self.links.iter().filter(|l| !l.broken) {
            if let Some(a) = link.a {
                let (ka, kb) = (self.bodies[a].key, self.bodies[link.b].key);
                self.linked.insert((ka.min(kb), ka.max(kb)), 0);
            }
        }
        if !self.bodies.iter().any(Body::active) {
            // Everything rests: nothing to detect or solve.
            self.manifolds.clear();
            self.cache.clear();
            self.recolor = true;
            return;
        }
        for body in &mut self.bodies {
            body.predict(dt);
        }
        let start = std::time::Instant::now();
        self.broadphase(serial);
        let broad = std::time::Instant::now();
        self.broad_time += broad - start;
        // If no island is asleep, collision detection cannot require a second
        // pass, so warm starting can be prepared during the first pass. With
        // sleepers, retain unscaled lanes until both passes finish: preparing
        // during each pass would decay the same warm start more than once.
        let prepare_now = !has_sleepers;
        self.narrowphase(serial, prepare_now.then_some(settings));
        self.narrow_time += broad.elapsed();
        // Real contact with a moving body wakes an asleep island; collide
        // again so the island's own contacts join this step.
        let linked_pairs =
            (self.links.iter().filter(|l| !l.broken)).filter_map(|l| l.a.map(|a| (a, l.b)));
        let touched: Vec<u32> = (self.manifolds.iter().map(|m| (m.a, m.b)))
            .chain(linked_pairs)
            .flat_map(|(a, b)| [(a, b), (b, a)])
            .filter(|&(x, y)| self.bodies[x].asleep && self.bodies[y].active())
            .map(|(x, _)| self.bodies[x].island)
            .collect();
        let also = self.wake_islands(&touched);
        if !also.is_empty() {
            woken.extend(also);
            let start = std::time::Instant::now();
            self.broadphase(serial);
            let broad = std::time::Instant::now();
            self.broad_time += broad - start;
            self.narrowphase(serial, None);
            self.narrow_time += broad.elapsed();
        }
        if !woken.is_empty() {
            self.stash.retain(|(island, _)| !woken.contains(island));
            self.index_stash();
        }
        if !prepare_now {
            let bodies = &self.bodies;
            let prepare = |m: &mut Manifold| {
                let basis = m.rows.transpose();
                let basis = [
                    basis.x_axis.into(),
                    basis.y_axis.into(),
                    basis.z_axis.into(),
                ];
                m.prepare(&bodies[m.a], &bodies[m.b], basis, settings);
            };
            if serial {
                self.manifolds.iter_mut().for_each(prepare);
            } else {
                self.manifolds
                    .par_iter_mut()
                    .with_min_len(PAIR_BATCH)
                    .for_each(prepare);
            }
        }
        for link in &mut self.links {
            if link.solved(&self.bodies) {
                link.initialize(&self.bodies, settings, dt);
            }
        }
        self.color();

        let gravity = settings.gravity;
        let initialize = |body: &mut Body| {
            body.initial_position = body.position;
            body.initial_rotation = body.rotation;
            (body.moved, body.turned) = (Vec3A::ZERO, Vec3A::ZERO);
            if body.dynamic() {
                // Inertial target (Eq. 2) and adaptive warm start (VBD).
                body.inertial_position = body.position + body.velocity * dt + gravity * (dt * dt);
                body.inertial_rotation = quat_add(body.rotation, body.midpoint_spin(dt) * dt);
                let accel = (body.velocity - body.previous) / dt;
                let weight = (accel.dot(gravity) / gravity.length_squared()).clamp(0.0, 1.0);
                let weight = if weight.is_finite() { weight } else { 0.0 };
                let start = body.position + body.velocity * dt + gravity * (weight * dt * dt);
                body.place(start, body.inertial_rotation);
            } else if body.kinematic {
                let rotation = quat_add(body.rotation, body.angular_velocity * dt);
                body.place(body.position + body.velocity * dt, rotation);
            }
        };
        if serial {
            self.bodies.iter_mut().for_each(initialize);
        } else {
            self.bodies
                .par_iter_mut()
                .with_min_len(BODY_BATCH)
                .for_each(initialize);
        }

        for &i in &self.free {
            let body = &mut self.bodies[i];
            body.place(body.inertial_position, body.inertial_rotation);
        }
        let scale = 1.0 / (dt * dt);
        let damping = settings.contact_damping / dt;
        let threads = if serial {
            1
        } else {
            rayon::current_num_threads()
        };
        if threads == 1 || !exclusive {
            let batch_width = batch_width();
            let beta = settings.beta;
            let manifolds = Shared(self.manifolds.as_mut_ptr());
            let links = Shared(self.links.as_mut_ptr());
            let bodies = Shared(self.bodies.as_mut_ptr());
            let sweep = |settings: &PhysicsSettings, iteration: u32, colors: &[Vec<usize>]| {
                for color in colors {
                    for indices in color.chunks(batch_width) {
                        // SAFETY: this thread owns the complete color phase.
                        unsafe {
                            solve_batch(
                                bodies,
                                manifolds,
                                links,
                                &self.offsets,
                                &self.adjacency,
                                indices,
                                settings,
                                scale,
                                damping,
                                iteration,
                            );
                        }
                    }
                }
            };
            for iteration in 0..sweeps {
                sweep(settings, iteration, &self.colors);
            }
            // The last iteration's dual update, for warm starting the next step.
            let solved = &self.bodies;
            for m in &mut self.manifolds {
                m.dual(&solved[m.a], &solved[m.b], beta);
                m.restore_error(stabilize.alpha);
            }
            for link in self.links.iter_mut().filter(|l| l.solved(solved)) {
                link.dual(link.a.map(|j| &solved[j]), &solved[link.b], settings);
            }
            let (manifolds, offsets, adjacency) = (&self.manifolds, &self.offsets, &self.adjacency);
            for (i, body) in self.bodies.iter_mut().enumerate() {
                let wrench = contact_wrench(manifolds, offsets, adjacency, i);
                body.finish(dt, wrench);
            }
            if settings.iterations > 1 {
                sweep(&stabilize, STABILIZE, &self.colors);
            }
        } else {
            self.iterate_parallel(settings, &stabilize, sweeps, dt, scale, damping, threads);
        }
        for (i, link) in self.links.iter_mut().enumerate() {
            if link.solved(&self.bodies) {
                let a = link.a.map(|j| &self.bodies[j]);
                if let Some(stage) = link.settle(a, &self.bodies[link.b], dt) {
                    self.joint_changes.push((i, stage));
                }
            }
        }
        self.bounce();
        self.sleep(dt);
    }

    /// Restitution, as a velocity pass after the solve (as Box2D does): each
    /// pushing contact that closed faster than `BOUNCE_SPEED` leaves at
    /// `restitution` times that speed. Positions are already solved, so this
    /// changes only where bodies go next step.
    fn bounce(&mut self) {
        let bodies = &mut self.bodies;
        let mut springy: Vec<(usize, Vec4)> = (self.manifolds.iter().enumerate())
            .filter(|(_, m)| {
                m.restitution > 0.0
                    && (0..m.len)
                        .any(|i| m.lanes.lambda[0][i] < 0.0 && m.cold.approach[i] < -BOUNCE_SPEED)
            })
            .map(|(i, _)| (i, Vec4::ZERO))
            .collect();
        if springy.is_empty() {
            return;
        }
        let inverse = |b: &Body| {
            if b.dynamic() {
                (1.0 / b.mass, Mat3::from(b.inertia.inverse()))
            } else {
                (0.0, Mat3::ZERO)
            }
        };
        // Several passes share impulses between a manifold's contacts.
        for _ in 0..BOUNCE_ITERATIONS {
            for (index, total) in &mut springy {
                let m = &self.manifolds[*index];
                let normal = Vec3::from(m.rows.row(0));
                let ((ma, ia), (mb, ib)) = (inverse(&bodies[m.a]), inverse(&bodies[m.b]));
                for i in 0..m.len {
                    let approach = m.cold.approach[i];
                    if m.lanes.lambda[0][i] >= 0.0 || approach >= -BOUNCE_SPEED {
                        continue;
                    }
                    let lane = |r: &[Vec4; 3]| Vec3::new(r[0][i], r[1][i], r[2][i]);
                    let (ra, rb) = (lane(&m.lanes.ra), lane(&m.lanes.rb));
                    let (a, b) = (&bodies[m.a], &bodies[m.b]);
                    let speed = (a.velocity + a.angular_velocity.cross(ra)
                        - b.velocity
                        - b.angular_velocity.cross(rb))
                    .dot(normal);
                    let (na, nb) = (ra.cross(normal), rb.cross(normal));
                    let k = ma + mb + na.dot(ia * na) + nb.dot(ib * nb);
                    if k <= 0.0 {
                        continue;
                    }
                    let target = -m.restitution * approach;
                    let accumulated = (total[i] + (target - speed) / k).max(0.0);
                    let impulse = accumulated - total[i];
                    total[i] = accumulated;
                    let a = &mut bodies[m.a];
                    a.velocity += normal * (impulse * ma);
                    a.angular_velocity += ia * na * impulse;
                    let b = &mut bodies[m.b];
                    b.velocity -= normal * (impulse * mb);
                    b.angular_velocity -= ib * nb * impulse;
                }
            }
        }
    }

    /// Sleeping islands with a contact on any of these bodies, which are
    /// going away or were moved by the caller.
    pub fn islands_touching(&self, keys: &[u64]) -> Vec<u32> {
        (self.stash.iter())
            .filter(|(_, m)| keys.contains(&m.cold.key.0) || keys.contains(&m.cold.key.1))
            .map(|&(island, _)| island)
            .collect()
    }

    /// The primal/dual iterations in one parallel region: each pool thread
    /// takes its share of every color, then of the manifolds, meeting at a
    /// barrier between phases. Forking once per color instead cost more than
    /// the work in small and medium scenes. Results match the serial path.
    #[allow(clippy::too_many_arguments)]
    fn iterate_parallel(
        &mut self,
        settings: &PhysicsSettings,
        stabilize: &PhysicsSettings,
        sweeps: u32,
        dt: f32,
        scale: f32,
        damping: f32,
        threads: usize,
    ) {
        let beta = settings.beta;
        let bodies = Shared(self.bodies.as_mut_ptr());
        let manifolds = Shared(self.manifolds.as_mut_ptr());
        let manifold_count = self.manifolds.len();
        let links = Shared(self.links.as_mut_ptr());
        let link_count = self.links.len();
        let body_count = self.bodies.len();
        let (colors, offsets, adjacency) = (&self.colors, &self.offsets, &self.adjacency);
        let batch_width = batch_width();
        let barrier = SpinBarrier::new(threads);
        rayon::broadcast(|ctx| {
            let (t, n) = (ctx.index(), ctx.num_threads());
            let share = |len: usize| len * t / n..len * (t + 1) / n;
            // SAFETY: in a color phase each thread writes only its own bodies
            // of that color, which share no manifold with any body of the
            // color, so no thread reads a body another is writing. A manifold
            // or link is written in a color phase only by its first visitor,
            // and read only by its two bodies, which have different colors.
            // Bodies, manifolds and links are reached one at a time, never as
            // whole slices. In the final dual phase threads write disjoint
            // manifolds and links and only read bodies. The barrier orders
            // every phase after the previous one.
            let body_at = |j: usize| unsafe { &*bodies.get().add(j) };
            let sweep = |settings: &PhysicsSettings, iteration: u32| {
                for color in colors {
                    for group in share(color.len().div_ceil(batch_width)) {
                        let start = group * batch_width;
                        let indices = &color[start..(start + batch_width).min(color.len())];
                        // SAFETY: groups partition the current color; the
                        // barrier excludes all other colors until completion.
                        unsafe {
                            solve_batch(
                                bodies, manifolds, links, offsets, adjacency, indices, settings,
                                scale, damping, iteration,
                            );
                        }
                    }
                    barrier.wait();
                }
            };
            for iteration in 0..sweeps {
                sweep(settings, iteration);
            }
            // The last iteration's dual update, for warm starting the next step.
            for k in share(manifold_count + link_count) {
                if k < manifold_count {
                    let m = unsafe { &mut *manifolds.get().add(k) };
                    m.dual(body_at(m.a), body_at(m.b), beta);
                    m.restore_error(stabilize.alpha);
                } else {
                    let link = unsafe { &mut *links.get().add(k - manifold_count) };
                    // `solved` reads only flags no phase writes.
                    if !link.broken
                        && (link.a.is_some_and(|a| body_at(a).solvable())
                            || body_at(link.b).solvable())
                    {
                        link.dual(link.a.map(body_at), body_at(link.b), settings);
                    }
                }
            }
            // Duals read every body, so velocities wait for all of them.
            barrier.wait();
            for k in share(body_count) {
                // SAFETY: duals are done, so manifolds are only read here, and
                // threads write disjoint bodies and read no others.
                let wrench = unsafe {
                    let all = std::slice::from_raw_parts(manifolds.get(), manifold_count);
                    contact_wrench(all, offsets, adjacency, k)
                };
                unsafe { &mut *bodies.get().add(k) }.finish(dt, wrench);
            }
            barrier.wait();
            if settings.iterations > 1 {
                sweep(stabilize, STABILIZE);
            }
        });
    }

    /// Wake every body of these sleeping islands; returns the islands woken.
    fn wake_islands(&mut self, islands: &[u32]) -> Vec<u32> {
        let mut woken = Vec::new();
        if islands.is_empty() {
            return woken;
        }
        for body in &mut self.bodies {
            body.wake = false;
            if body.asleep && islands.contains(&body.island) {
                body.asleep = false;
                if !woken.contains(&body.island) {
                    woken.push(body.island);
                }
            }
        }
        woken
    }

    fn index_stash(&mut self) {
        self.stash_index.clear();
        (self.stash_index).extend(
            self.stash
                .iter()
                .enumerate()
                .map(|(i, (_, m))| (m.cold.key, i)),
        );
    }

    /// Put to sleep each island (dynamic bodies joined by contacts) whose
    /// bodies have all rested long enough and that no moving kinematic touches.
    fn sleep(&mut self, dt: f32) {
        let n = self.bodies.len();
        for body in self.bodies.iter_mut().filter(|b| b.dynamic()) {
            let speed = body.velocity.length() + body.angular_velocity.length() * body.bound();
            body.sleep_time = if speed < SLEEP_SPEED {
                body.sleep_time + dt
            } else {
                0.0
            };
        }
        // Islands sleep only once every body in them has rested this long.
        let resting = self
            .bodies
            .iter()
            .any(|b| b.dynamic() && b.sleep_time >= TIME_TO_SLEEP);
        if !resting {
            return;
        }
        let mut parent = std::mem::take(&mut self.scratch_usize);
        parent.clear();
        parent.extend(0..n);
        fn find(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            i
        }
        let joined = (self.manifolds.iter().map(|m| (m.a, m.b))).chain(
            self.links
                .iter()
                .filter(|l| !l.broken)
                .filter_map(|l| l.a.map(|a| (a, l.b))),
        );
        for (a, b) in joined.clone() {
            if self.bodies[a].dynamic() && self.bodies[b].dynamic() {
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                parent[ra] = rb;
            }
        }
        // Per root: the shortest rest, or -1 when a moving kinematic touches it.
        let mut rest = std::mem::take(&mut self.rest);
        rest.clear();
        rest.resize(n, f32::INFINITY);
        for i in (0..n).filter(|&i| self.bodies[i].dynamic()) {
            let root = find(&mut parent, i);
            rest[root] = rest[root].min(self.bodies[i].sleep_time);
        }
        for m in &self.manifolds {
            for (x, y) in [(m.a, m.b), (m.b, m.a)] {
                let (bx, by) = (&self.bodies[x], &self.bodies[y]);
                if bx.dynamic() && (m.cold.deep || (by.active() && !by.dynamic())) {
                    let root = find(&mut parent, x);
                    rest[root] = -1.0;
                }
            }
        }
        for (a, b) in joined {
            for (x, y) in [(a, b), (b, a)] {
                let (bx, by) = (&self.bodies[x], &self.bodies[y]);
                if bx.dynamic() && by.active() && !by.dynamic() {
                    let root = find(&mut parent, x);
                    rest[root] = -1.0;
                }
            }
        }
        let mut island = std::mem::take(&mut self.scratch_u32);
        island.clear();
        island.resize(n, u32::MAX);
        for i in 0..n {
            if !self.bodies[i].dynamic() {
                continue;
            }
            let root = find(&mut parent, i);
            if rest[root] < TIME_TO_SLEEP {
                continue;
            }
            if island[root] == u32::MAX {
                island[root] = self.next_island;
                self.next_island = self.next_island.wrapping_add(1);
            }
            let body = &mut self.bodies[i];
            body.asleep = true;
            body.island = island[root];
            body.velocity = Vec3::ZERO;
            body.angular_velocity = Vec3::ZERO;
            body.previous = Vec3::ZERO;
        }
        (self.scratch_usize, self.rest, self.scratch_u32) = (parent, rest, island);
        let bodies = &self.bodies;
        let asleep = |i: usize| bodies[i].asleep;
        if !self.manifolds.iter().any(|m| asleep(m.a) || asleep(m.b)) {
            return;
        }
        let (mut kept, mut slept) = (Vec::new(), Vec::new());
        for m in self.manifolds.drain(..) {
            if asleep(m.a) || asleep(m.b) {
                let owner = if asleep(m.a) { m.a } else { m.b };
                slept.push((bodies[owner].island, m));
            } else {
                kept.push(m);
            }
        }
        self.manifolds = kept;
        self.recolor = true;
        self.stash.extend(slept);
        self.index_stash();
        self.cache.clear();
        (self.cache).extend(
            self.manifolds
                .iter()
                .enumerate()
                .map(|(i, m)| (m.cold.key, i)),
        );
    }

    /// Sweep and prune oriented-box AABBs along the axis where body centers
    /// vary most. The boxes live in one compact array: the sweep's inner loop
    /// then streams through cache instead of chasing whole bodies.
    fn broadphase(&mut self, serial: bool) {
        let bodies = &self.bodies;
        if bodies.len() < 2 {
            self.pairs.clear();
            self.fat.clear();
            return;
        }
        let bounds = |(i, b): (usize, &Body)| {
            let extent = match b.shape {
                Collider::Cuboid { half_extents } => {
                    let m = b.initial_basis;
                    let abs = |row: Vec3| row.abs().dot(half_extents);
                    Vec3::new(abs(m.row(0)), abs(m.row(1)), abs(m.row(2)))
                }
                Collider::Sphere { radius } => Vec3::splat(radius),
                Collider::Capsule {
                    radius,
                    half_height,
                } => b.initial_basis.z_axis.abs() * half_height + radius,
                // Everything near it pairs with it; the shape then says what touches.
                Collider::Ground => Vec3::splat(UNBOUNDED),
            } + (0.5 * CONTACT_DISTANCE + b.reach);
            let flags =
                u32::from(b.dynamic()) | u32::from(b.active()) << 1 | u32::from(b.asleep) << 2;
            Aabb {
                min: b.position - extent + b.travel.min(Vec3::ZERO),
                max: b.position + extent + b.travel.max(Vec3::ZERO),
                index: i as u32,
                flags,
            }
        };
        if serial {
            self.tight.clear();
            self.tight.extend(bodies.iter().enumerate().map(bounds));
        } else {
            (bodies
                .par_iter()
                .enumerate()
                .with_min_len(BODY_BATCH)
                .map(bounds))
            .collect_into_vec(&mut self.tight);
        }
        // Last step's pairs hold while every body stays inside its enlarged
        // box and no body changed: settling bodies skip the sweep entirely.
        let kept = |fat: &[Aabb], keys: &[u64], i: usize, t: &Aabb| {
            keys.get(i) == Some(&bodies[i].key)
                && fat[i].min.cmple(t.min).all()
                && t.max.cmple(fat[i].max).all()
        };
        let fits = self.fat.len() == bodies.len()
            && (self.tight.iter().enumerate()).all(|(i, t)| {
                t.flags == self.fat[i].flags && kept(&self.fat, &self.fat_keys, i, t)
            });
        if fits {
            return;
        }
        self.fat.resize(bodies.len(), self.tight[0]);
        self.fat_keys.resize(bodies.len(), 0);
        for (i, t) in self.tight.iter().enumerate() {
            if !kept(&self.fat, &self.fat_keys, i, t) {
                // Only moving bodies get room to move.
                let margin = if t.flags & 2 != 0 { FAT_MARGIN } else { 0.0 };
                self.fat[i] = Aabb {
                    min: t.min - margin,
                    max: t.max + margin,
                    ..*t
                };
                self.fat_keys[i] = bodies[i].key;
            }
            self.fat[i].flags = t.flags;
            self.fat[i].index = i as u32;
        }
        self.fat.truncate(bodies.len());
        self.fat_keys.truncate(bodies.len());
        self.pairs.clear();
        let n = bodies.len() as f32;
        let mean = bodies.iter().map(|b| b.position).sum::<Vec3>() / n;
        let variance = bodies
            .iter()
            .map(|b| (b.position - mean) * (b.position - mean))
            .sum::<Vec3>();
        let axis = if variance.x >= variance.y && variance.x >= variance.z {
            0
        } else if variance.y >= variance.z {
            1
        } else {
            2
        };
        self.boxes.clear();
        self.boxes.extend_from_slice(&self.fat);
        let order = |a: &Aabb, b: &Aabb| a.min[axis].total_cmp(&b.min[axis]);
        if serial || self.boxes.len() < SERIAL_SORT {
            self.boxes.sort_unstable_by(order);
        } else {
            self.boxes.par_sort_unstable_by(order);
        }
        self.sweep_blocks.clear();
        self.sweep_blocks
            .extend(self.boxes.chunks(4).map(SweepBlock::new));
        const DYNAMIC: u32 = 1;
        const ACTIVE: u32 = 2;
        const ASLEEP: u32 = 4;
        let boxes = &self.boxes;
        let blocks = &self.sweep_blocks;
        // Each chunk sweeps forward from its own boxes; joining the chunks'
        // pairs in order keeps the result independent of thread count.
        let sweep = |(c, chunk): (usize, &[Aabb]), pairs: &mut Vec<(usize, usize)>| {
            for (k, a) in chunk.iter().enumerate() {
                let first = c * SWEEP_CHUNK + k + 1;
                let min = a.min.to_array().map(Vec4::splat);
                let max = a.max.to_array().map(Vec4::splat);
                for (offset, b) in blocks[first / 4..].iter().enumerate() {
                    if b.min[axis].x > a.max[axis] {
                        break;
                    }
                    let mut mask = b.overlaps(&min, &max);
                    if offset == 0 {
                        mask &= 0xf << (first % 4);
                    }
                    while mask != 0 {
                        let lane = mask.trailing_zeros() as usize;
                        mask &= mask - 1;
                        // Pairs an asleep body forms with a moving one only wake it.
                        let flags = b.flags[lane];
                        let relevant = (a.flags | flags) & DYNAMIC != 0
                            || (a.flags & ASLEEP != 0 && flags & ACTIVE != 0)
                            || (flags & ASLEEP != 0 && a.flags & ACTIVE != 0);
                        if relevant {
                            let (i, j) = (a.index as usize, b.index[lane] as usize);
                            let first = bodies[i].key < bodies[j].key;
                            pairs.push(if first { (i, j) } else { (j, i) });
                        }
                    }
                }
            }
        };
        // Below SERIAL_SWEEP boxes one task sweeps them all.
        let min = if boxes.len() < SERIAL_SWEEP {
            usize::MAX
        } else {
            1
        };
        if serial {
            for chunk in boxes.chunks(SWEEP_CHUNK).enumerate() {
                sweep(chunk, &mut self.pairs);
            }
        } else {
            (boxes
                .par_chunks(SWEEP_CHUNK)
                .enumerate()
                .with_min_len(min)
                .map(|chunk| {
                    let mut pairs = Vec::new();
                    sweep(chunk, &mut pairs);
                    pairs
                }))
            .collect_into_vec(&mut self.swept);
            self.pairs.extend(self.swept.drain(..).flatten());
        }
    }

    /// Update the persistent manifolds from this step's pairs: kept pairs
    /// are refreshed in place, new ones collided, and dropped or separated
    /// ones removed. Contacts merge with the previous step's to warm start
    /// their forces and penalties (Eq. 19).
    fn narrowphase(&mut self, serial: bool, prepare: Option<&PhysicsSettings>) {
        self.pass = self.pass.wrapping_add(1);
        let pass = self.pass;
        self.fresh.clear();
        for &(a, b) in &self.pairs {
            let key = (self.bodies[a].key, self.bodies[b].key);
            if self.linked.contains_key(&key) {
                continue;
            }
            match self.cache.get(&key) {
                Some(&i) => {
                    let m = &mut self.manifolds[i];
                    (m.a, m.b, m.cold.seen) = (a, b, pass);
                }
                None => self.fresh.push((a, b)),
            }
        }
        let bodies = &self.bodies;
        let ground = self.ground.as_deref();
        let refresh = |m: &mut Manifold| {
            if m.cold.seen != pass || !m.refresh(bodies, prepare, ground) {
                m.len = 0;
            }
        };
        if serial {
            self.manifolds.iter_mut().for_each(refresh);
        } else {
            self.manifolds
                .par_iter_mut()
                .with_min_len(PAIR_BATCH)
                .for_each(refresh);
        }
        let before = self.manifolds.len();
        self.manifolds.retain(|m| m.len > 0);
        let mut changed = self.manifolds.len() != before;
        let (stash, stash_index) = (&self.stash, &self.stash_index);
        let tight = &self.tight;
        let create = |&(a, b): &(usize, usize)| {
            // Enlarged boxes pair near neighbours that are not touching, and
            // swept ones pair bodies that travel together without closing.
            let (ta, tb, ba, bb) = (&tight[a], &tight[b], &bodies[a], &bodies[b]);
            let relative = ba.travel - bb.travel;
            let (min_a, max_a) = (
                ta.min - ba.travel.min(Vec3::ZERO) + relative.min(Vec3::ZERO),
                ta.max - ba.travel.max(Vec3::ZERO) + relative.max(Vec3::ZERO),
            );
            let (min_b, max_b) = (
                tb.min - bb.travel.min(Vec3::ZERO),
                tb.max - bb.travel.max(Vec3::ZERO),
            );
            if !(min_a.cmple(max_b).all() && min_b.cmple(max_a).all()) {
                return None;
            }
            let key = (bodies[a].key, bodies[b].key);
            let previous = stash_index.get(&key).map(|&i| &stash[i].1);
            Manifold::create(bodies, a, b, previous, prepare, ground)
        };
        if serial || rayon::current_num_threads() == 1 || self.fresh.len() < PAIR_BATCH {
            // A failed narrowphase needs no manifold-sized output slot.
            // Append hits directly when there is no parallel work to merge.
            for pair in &self.fresh {
                if let Some(manifold) = create(pair) {
                    self.manifolds.push(manifold);
                    changed = true;
                }
            }
        } else {
            self.fresh
                .par_iter()
                .with_min_len(PAIR_BATCH)
                .map(create)
                .collect_into_vec(&mut self.found);
            changed |= self.found.iter().any(Option::is_some);
            self.manifolds.extend(self.found.drain(..).flatten());
        }
        if changed {
            self.recolor = true;
            self.cache.clear();
            (self.cache).extend(
                self.manifolds
                    .iter()
                    .enumerate()
                    .map(|(i, m)| (m.cold.key, i)),
            );
        }
    }

    /// Build body-to-manifold adjacency and greedily color dynamic bodies so no
    /// two of one color share a manifold.
    fn color(&mut self) {
        // At rest the graph repeats: same manifolds, links and solved bodies.
        let bodies = &self.bodies;
        let same_bodies = self.colored.len() == bodies.len()
            && (self.colored.iter().zip(bodies))
                .all(|(&(key, solved), b)| key == b.key && solved == b.dynamic());
        let same_links = self.colored_links.len() == self.links.len()
            && (self.colored_links.iter().zip(&self.links))
                .all(|(&(a, b, solved), l)| a == l.a && b == l.b && solved == l.solved(bodies));
        if !self.recolor && same_bodies && same_links {
            return;
        }
        self.recolor = false;
        self.colored.clear();
        self.colored
            .extend(bodies.iter().map(|b| (b.key, b.dynamic())));
        self.colored_links.clear();
        (self.colored_links).extend(self.links.iter().map(|l| (l.a, l.b, l.solved(bodies))));
        let n = self.bodies.len();
        self.offsets.clear();
        self.offsets.resize(n + 1, 0);
        for m in &self.manifolds {
            self.offsets[m.a + 1] += 1;
            self.offsets[m.b + 1] += 1;
        }
        let bodies = &self.bodies;
        let solved = |l: &&Link| l.solved(bodies);
        for link in self.links.iter().filter(solved) {
            for body in link.a.into_iter().chain([link.b]) {
                self.offsets[body + 1] += 1;
            }
        }
        for i in 0..n {
            self.offsets[i + 1] += self.offsets[i];
        }
        self.adjacency.clear();
        self.adjacency.resize(self.offsets[n] as usize, 0);
        self.scratch_u32.clear();
        self.scratch_u32.extend_from_slice(&self.offsets[..n]);
        let fill = &mut self.scratch_u32;
        for (index, m) in self.manifolds.iter().enumerate() {
            for body in [m.a, m.b] {
                self.adjacency[fill[body] as usize] = index as u32;
                fill[body] += 1;
            }
        }
        for (index, link) in self.links.iter().enumerate() {
            if !link.solved(bodies) {
                continue;
            }
            for body in link.a.into_iter().chain([link.b]) {
                self.adjacency[fill[body] as usize] = index as u32 | LINK;
                fill[body] += 1;
            }
        }

        for color in &mut self.colors {
            color.clear();
        }
        self.free.clear();
        fill.fill(u32::MAX);
        let color_of = fill;
        // stamp[c] == i marks color c as taken by a neighbor of body i.
        self.scratch_usize.clear();
        let stamp = &mut self.scratch_usize;
        for i in (0..n).filter(|&i| self.bodies[i].dynamic()) {
            if self.offsets[i] == self.offsets[i + 1] {
                self.free.push(i);
                continue;
            }
            for &entry in &self.adjacency[self.offsets[i] as usize..self.offsets[i + 1] as usize] {
                let other = if entry & LINK != 0 {
                    let link = &self.links[(entry & !LINK) as usize];
                    if link.a == Some(i) {
                        Some(link.b)
                    } else {
                        link.a
                    }
                } else {
                    let m = &self.manifolds[entry as usize];
                    Some(if m.a == i { m.b } else { m.a })
                };
                if let Some(c) = other.and_then(|o| stamp.get_mut(color_of[o] as usize)) {
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
        // Unsolved bodies have no color (MAX), so a solved side always comes first.
        for m in &mut self.manifolds {
            m.first_is_a = color_of[m.a] <= color_of[m.b];
        }
        for link in &mut self.links {
            link.first_is_a = link.a.is_some_and(|a| color_of[a] <= color_of[link.b]);
        }
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
        let cross = Mat3::from_cols(
            Vec3::new(0.3, 0.0, 0.1),
            Vec3::ZERO,
            Vec3::new(0.0, 0.2, 0.0),
        );
        let system = System {
            lin: m.into(),
            ang: (m * 2.0).into(),
            cross: cross.into(),
            rhs_lin: Vec3A::new(1.0, -2.0, 3.0),
            rhs_ang: Vec3A::new(-1.0, 0.5, 2.0),
        };
        let (x, y) = system.solve();
        let (x, y) = (Vec3A::from(x), Vec3A::from(y));
        // A [x; y] must equal -rhs.
        let lin = system.lin * x + system.cross.transpose() * y;
        let ang = system.cross * x + system.ang * y;
        assert!((lin + system.rhs_lin).length() < 1e-5, "{lin}");
        assert!((ang + system.rhs_ang).length() < 1e-5, "{ang}");
    }

    #[test]
    fn batched_solves_keep_lanes_independent_across_mass_scales_and_partial_batches() {
        let scales = [1.0e-12, 1.0e-3, 1.0e9, 1.0e24];
        for width in 1..=4 {
            let expected: [(Vec3A, Vec3A); 4] = std::array::from_fn(|lane| {
                let v = lane as f32 + 1.0;
                (Vec3A::new(v, -0.3 * v, 0.2), Vec3A::new(-0.1, v * 0.2, -v))
            });
            let systems = std::array::from_fn(|lane| {
                if lane >= width {
                    return System::IDENTITY;
                }
                let v = lane as f32;
                let lin = Mat3A::from_cols(
                    Vec3A::new(4.0 + v, 0.4, -0.2),
                    Vec3A::new(0.4, 3.0 + v, 0.3),
                    Vec3A::new(-0.2, 0.3, 5.0 + v),
                ) * scales[lane];
                let ang = Mat3A::from_cols(
                    Vec3A::new(6.0 + v, -0.5, 0.1),
                    Vec3A::new(-0.5, 7.0 + v, -0.4),
                    Vec3A::new(0.1, -0.4, 8.0 + v),
                ) * scales[lane];
                let cross = Mat3A::from_cols(
                    Vec3A::new(0.1, 0.2, -0.3),
                    Vec3A::new(-0.4, 0.1, 0.2),
                    Vec3A::new(0.3, -0.2, 0.4),
                ) * scales[lane];
                let (x, y) = expected[lane];
                System {
                    lin,
                    ang,
                    cross,
                    rhs_lin: -(lin * x + cross.transpose() * y),
                    rhs_ang: -(cross * x + ang * y),
                }
            });
            let update = System::solve_four(&systems);
            for lane in 0..4 {
                let x = Vec3A::new(update[0][lane], update[1][lane], update[2][lane]);
                let y = Vec3A::new(update[3][lane], update[4][lane], update[5][lane]);
                if lane >= width {
                    assert_eq!((x, y), (Vec3A::ZERO, Vec3A::ZERO));
                    continue;
                }
                let s = &systems[lane];
                assert!(
                    (x - expected[lane].0).length() < 1.0e-4,
                    "width {width}, lane {lane}"
                );
                assert!(
                    (y - expected[lane].1).length() < 1.0e-4,
                    "width {width}, lane {lane}"
                );
                let linear = (s.lin * x + s.cross.transpose() * y + s.rhs_lin) / scales[lane];
                let angular = (s.cross * x + s.ang * y + s.rhs_ang) / scales[lane];
                assert!(
                    linear.length() < 1.0e-4,
                    "width {width}, lane {lane}: {linear}"
                );
                assert!(
                    angular.length() < 1.0e-4,
                    "width {width}, lane {lane}: {angular}"
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_solves_keep_eight_body_systems_independent() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let scales = [1.0e-12, 1.0e-3, 1.0e9, 1.0e24];
        let systems = std::array::from_fn(|lane| {
            let v = lane as f32 + 1.0;
            let scale = scales[lane % scales.len()];
            System {
                lin: Mat3A::from_cols(
                    Vec3A::new(4.0 + v, 0.4, -0.2),
                    Vec3A::new(0.4, 3.0 + v, 0.3),
                    Vec3A::new(-0.2, 0.3, 5.0 + v),
                ) * scale,
                ang: Mat3A::from_cols(
                    Vec3A::new(6.0 + v, -0.5, 0.1),
                    Vec3A::new(-0.5, 7.0 + v, -0.4),
                    Vec3A::new(0.1, -0.4, 8.0 + v),
                ) * scale,
                cross: Mat3A::from_cols(
                    Vec3A::new(0.1, 0.2, -0.3),
                    Vec3A::new(-0.4, 0.1, 0.2),
                    Vec3A::new(0.3, -0.2, 0.4),
                ) * scale,
                rhs_lin: Vec3A::new(v, -2.0 * v, 3.0) * scale,
                rhs_ang: Vec3A::new(-v, 0.5 * v, 2.0) * scale,
            }
        });
        // SAFETY: CPU support was checked above.
        let rotations = std::array::from_fn(|lane| {
            Quat::from_euler(glam::EulerRot::XYZ, lane as f32 * 0.1, -0.3, 0.7)
        });
        let update = unsafe { System::solve_eight(&systems, &rotations) };
        for (lane, s) in systems.iter().enumerate() {
            let x = Vec3A::new(update[0][lane], update[1][lane], update[2][lane]);
            let actual = Quat::from_xyzw(
                update[3][lane],
                update[4][lane],
                update[5][lane],
                update[6][lane],
            );
            let relative = actual * rotations[lane].conjugate();
            let y = Vec3A::from(relative.xyz() * (2.0 / relative.w));
            let scale = scales[lane % scales.len()];
            let linear = (s.lin * x + s.cross.transpose() * y + s.rhs_lin) / scale;
            let angular = (s.cross * x + s.ang * y + s.rhs_ang) / scale;
            assert!(linear.length() < 1.0e-4, "lane {lane}: {linear}");
            assert!(angular.length() < 1.0e-4, "lane {lane}: {angular}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn batched_rotation_updates_match_scalar_at_varied_poses() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        for sample in 0..256 {
            let systems = std::array::from_fn(|lane| {
                let t = (sample * 8 + lane) as f32;
                System {
                    rhs_ang: Vec3A::new((t * 0.19).sin(), (t * 0.37).cos(), (t * 0.11).sin())
                        * 0.25,
                    ..System::IDENTITY
                }
            });
            let rotations = std::array::from_fn(|lane| {
                let t = (sample * 8 + lane) as f32;
                let q = Quat::from_euler(glam::EulerRot::XYZ, t * 0.017, t * -0.011, t * 0.031);
                if lane % 2 == 0 { q } else { -q }
            });
            // SAFETY: CPU support was checked above.
            let update = unsafe { System::solve_eight(&systems, &rotations) };
            for lane in 0..8 {
                let dr = -Vec3::from(systems[lane].rhs_ang);
                let expected = quat_add(rotations[lane], dr);
                let actual = Quat::from_xyzw(
                    update[3][lane],
                    update[4][lane],
                    update[5][lane],
                    update[6][lane],
                );
                assert_eq!(
                    actual.to_array().map(f32::to_bits),
                    expected.to_array().map(f32::to_bits),
                    "sample {sample}, lane {lane}"
                );
            }
        }
    }

    /// Pseudo-random value in [-1, 1).
    #[cfg(target_arch = "x86_64")]
    fn random(state: &mut u64) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }

    /// A body with an arbitrary displacement since the step began.
    #[cfg(target_arch = "x86_64")]
    fn moved_body(state: &mut u64) -> Body {
        let mut vector = || Vec3A::new(random(state), random(state), random(state)) * 0.01;
        Body {
            key: 0,
            shape: Collider::Sphere { radius: 1.0 },
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            initial_position: Vec3::ZERO,
            initial_rotation: Quat::IDENTITY,
            inertial_position: Vec3::ZERO,
            inertial_rotation: Quat::IDENTITY,
            initial_basis: Mat3::IDENTITY,
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            previous: Vec3::ZERO,
            moved: vector(),
            turned: vector(),
            travel: Vec3::ZERO,
            reach: 0.0,
            stiffness: 3600.0,
            mass: 1.0,
            inertia: Mat3A::IDENTITY,
            friction: 0.5,
            static_friction: 0.5,
            restitution: 0.0,
            kinematic: false,
            asleep: false,
            wake: false,
            island: 0,
            sleep_time: 0.0,
        }
    }

    /// Solver state varied enough to reach every clamp, in the lanes `len` uses.
    #[cfg(target_arch = "x86_64")]
    fn random_manifold(
        mut state: u64,
        (a, b): (usize, usize),
        len: usize,
        (friction, static_friction): (f32, f32),
    ) -> Manifold {
        let state = &mut state;
        let live = (1u8 << len) - 1;
        let active = LANE_BITS[live as usize];
        let mut vector =
            |scale: f32| Vec4::from_array(std::array::from_fn(|_| random(state))) * scale * active;
        let mut bits = || (vector(8.0).x.abs() as u8) & live;
        let (sticking, rescale_normal, stick, held) = (bits(), bits(), bits(), bits());
        let mut vector =
            |scale: f32| Vec4::from_array(std::array::from_fn(|_| random(state))) * scale * active;
        let lanes = Lanes {
            c0: [vector(0.01), vector(0.01), vector(0.01)],
            ra: [vector(0.5), vector(0.5), vector(0.5)],
            rb: [vector(0.5), vector(0.5), vector(0.5)],
            penalty: [
                vector(1.0e4).abs(),
                vector(1.0e4).abs(),
                vector(1.0e4).abs(),
            ],
            lambda: [-vector(50.0).abs(), vector(30.0), vector(30.0)],
            active,
            sticking: LANE_BITS[sticking as usize],
            rescale_normal: LANE_BITS[rescale_normal as usize],
            stick,
            held,
        };
        let angle = vector(3.0);
        Manifold {
            a,
            b,
            rows: Mat3A::from_quat(Quat::from_euler(
                glam::EulerRot::XYZ,
                angle.x,
                angle.x * 0.7,
                angle.x * 0.3,
            )),
            cold: Box::new(CollisionData {
                key: (0, 0),
                local_basis: [Vec3::ZERO; 3],
                relative: (Vec3::ZERO, Quat::IDENTITY),
                contacts: [Contact::default(); collide::MAX_CONTACTS],
                anchors: [[Vec4::ZERO; 3]; 2],
                error: [Vec4::ZERO; 3],
                deep: false,
                seen: 0,
                approach: Vec4::ZERO,
            }),
            first_is_a: true,
            len,
            friction,
            static_friction,
            restitution: 0.0,
            scale: 1.0,
            limit: PENALTY_MAX,
            impact: false,
            lanes,
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_pair_stamps_match_scalar_bit_for_bit() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let bits = |v: Vec4| v.to_array().map(f32::to_bits);
        let bits3 = |v: Vec3A| v.to_array().map(f32::to_bits);
        let (mut cone, mut clamped) = (0, 0);
        for sample in 0..4096 {
            let bodies: [Body; 3] = std::array::from_fn(|_| moved_body(&mut seed));
            // Body 0 is A of the first manifold and, alternately, B of the second.
            let (first, second) = ((0, 1), if sample % 2 == 0 { (2, 0) } else { (0, 2) });
            let (len_p, len_q) = ([(4, 4), (2, 4), (4, 3), (2, 3)])[sample / 2 % 4];
            let friction = if sample / 8 % 2 == 0 {
                ((0.5, 0.5), (0.5, 0.5))
            } else {
                ((0.3, 0.6), (0.5, 0.5))
            };
            let flags = [sample / 16 % 2 == 0, sample / 32 % 2 == 0];
            let damping = if sample / 64 % 2 == 0 { 0.042 } else { 0.0 };
            // Each path gets its own copy of the same two manifolds.
            let (seed_p, seed_q) = (seed.rotate_left(13), seed.rotate_left(29));
            random(&mut seed);
            let make_p = || random_manifold(seed_p, first, len_p, friction.0);
            let make_q = || random_manifold(seed_q, second, len_q, friction.1);
            let (mut p, mut q, mut wide_p, mut wide_q) = (make_p(), make_q(), make_p(), make_q());
            let beta = 1.0e5;
            // Nonzero, as assembly's inertial term is, so sum order shows.
            let start = || System {
                rhs_lin: Vec3A::new(0.1, -0.2, 0.3),
                ..System::IDENTITY
            };

            let mut scalar = start();
            let mut scalar_sums = LaneSums::default();
            for (half, (m, first)) in [(&mut p, flags[0]), (&mut q, flags[1])]
                .into_iter()
                .enumerate()
            {
                let (a, b) = (&bodies[m.a], &bodies[m.b]);
                let c = first.then(|| m.dual(a, b, beta));
                let e = m.eval(&Motion::new(a, b));
                cone += usize::from(e.force[1].cmpne(e.raw[1]).any());
                clamped += usize::from(e.force[0].cmpne(e.raw[0]).any());
                m.stamp(
                    a,
                    b,
                    m.a == 0,
                    c,
                    damping,
                    &mut scalar,
                    &mut scalar_sums,
                    half,
                );
            }
            let mut wide = start();
            let mut wide_sums = LaneSums::default();
            let (pa, pb) = (&bodies[wide_p.a], &bodies[wide_p.b]);
            let (qa, qb) = (&bodies[wide_q.a], &bodies[wide_q.b]);
            // SAFETY: AVX2 support was checked above, AVX-512's here.
            unsafe {
                let stamp = if sample % 2 == 1 && cpu_level() == AVX512 {
                    Manifold::stamp_pair_avx512
                } else {
                    Manifold::stamp_pair_avx2
                };
                stamp(
                    &mut wide_p,
                    &mut wide_q,
                    [pa, pb],
                    [qa, qb],
                    0,
                    flags,
                    beta,
                    damping,
                    &mut wide,
                    &mut wide_sums,
                );
            }

            for (s, w) in [
                (scalar.lin.x_axis, wide.lin.x_axis),
                (scalar.lin.y_axis, wide.lin.y_axis),
                (scalar.lin.z_axis, wide.lin.z_axis),
                (scalar.cross.x_axis, wide.cross.x_axis),
                (scalar.cross.y_axis, wide.cross.y_axis),
                (scalar.cross.z_axis, wide.cross.z_axis),
                (scalar.rhs_lin, wide.rhs_lin),
            ] {
                assert_eq!(bits3(s), bits3(w), "sample {sample}");
            }
            let (s, w) = (&scalar_sums, &wide_sums);
            for (s, w) in s.ang.iter().chain(&s.rhs).zip(w.ang.iter().chain(&w.rhs)) {
                assert_eq!(s.map(bits), w.map(bits), "sample {sample}");
            }
            for (s, w) in [(&p, &wide_p), (&q, &wide_q)] {
                assert_eq!(s.lanes.penalty.map(bits), w.lanes.penalty.map(bits));
                assert_eq!(s.lanes.lambda.map(bits), w.lanes.lambda.map(bits));
                assert_eq!((s.lanes.stick, s.lanes.held), (w.lanes.stick, w.lanes.held));
            }
        }
        // The samples must reach the friction cone and the normal clamp.
        assert!(
            cone > 100 && clamped > 100,
            "cone {cone}, clamped {clamped}"
        );
    }

    #[test]
    fn lane_rotation_and_cross_match_scalar() {
        let q = Quat::from_euler(glam::EulerRot::XYZ, 0.3, -1.1, 2.0);
        let n = Vec3::new(0.2, -0.7, 0.4).normalize();
        let points = [
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-0.5, 0.25, 4.0),
            Vec3::X,
            Vec3::new(0.0, -3.0, 0.5),
        ];
        let lanes: [Vec4; 3] = std::array::from_fn(|k| Vec4::from_array(points.map(|p| p[k])));
        let (rotated, crossed) = (rotate4(Mat3::from_quat(q), lanes), cross4(lanes, n));
        for (i, p) in points.iter().enumerate() {
            let (r, c) = (q * *p, p.cross(n));
            for k in 0..3 {
                assert!((rotated[k][i] - r[k]).abs() < 1e-5, "rotate lane {i}");
                assert!((crossed[k][i] - c[k]).abs() < 1e-5, "cross lane {i}");
            }
        }
    }

    #[test]
    fn quaternion_difference_inverts_addition() {
        let q = Quat::from_rotation_y(0.7);
        let w = Vec3::new(0.01, -0.02, 0.015);
        assert!((quat_sub(quat_add(q, w), q) - w).length() < 1e-4);
        // Exactly, however fast: velocities must survive being integrated.
        let fast = Vec3::new(0.9, -0.5, 0.7);
        assert!((quat_step(quat_add(q, fast), q) - fast).length() < 1e-5);
        assert!(quat_step(-quat_add(q, fast), q).distance(fast) < 1e-5);
        let half_turn = Quat::from_rotation_z(std::f32::consts::PI);
        assert!(quat_step(half_turn, Quat::IDENTITY).is_finite());
    }

    #[test]
    fn sweep_blocks_preserve_touching_bounds_and_exclude_unused_lanes() {
        for len in 1..=4 {
            let boxes: Vec<_> = (0..len)
                .map(|i| {
                    let center = Vec3::new(i as f32 * 1.5, i as f32 * -0.3, i as f32 * 0.4);
                    Aabb {
                        min: center - Vec3::new(0.5, 0.7, 0.9),
                        max: center + Vec3::new(0.5, 0.7, 0.9),
                        index: i as u32,
                        flags: 1,
                    }
                })
                .collect();
            let block = SweepBlock::new(&boxes);
            let mut queries: Vec<_> = boxes.iter().flat_map(|b| [b.min, b.max]).collect();
            queries.extend((0..100).map(|i| {
                Vec3::new(
                    i as f32 * 0.1 - 2.0,
                    (i % 7) as f32 * 0.4 - 2.0,
                    (i % 11) as f32 * 0.3 - 2.0,
                )
            }));
            for center in queries {
                for half in [Vec3::ZERO, Vec3::splat(0.6), Vec3::splat(100.0)] {
                    let min = center - half;
                    let max = center + half;
                    let expected = boxes.iter().enumerate().fold(0, |mask, (i, b)| {
                        if b.min.cmple(max).all() && b.max.cmpge(min).all() {
                            mask | (1 << i)
                        } else {
                            mask
                        }
                    });
                    assert_eq!(
                        block.overlaps(
                            &min.to_array().map(Vec4::splat),
                            &max.to_array().map(Vec4::splat)
                        ),
                        expected,
                        "len {len}, center {center:?}, half {half:?}",
                    );
                }
            }
        }
    }
}
