//! The AVBD step: broadphase, persistent manifolds, graph coloring, and the
//! primal (per-body block solve) / dual (per-contact force) iterations.
//! Equation numbers refer to the AVBD paper.

use crate::link::Link;
use crate::{Collider, PhysicsSettings, collide};
use glam::{Mat3, Mat3A, Mat4, Quat, Vec3, Vec3A, Vec4};
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Body keys are already unique integers: a multiply mix is enough, and much
/// cheaper than SipHash on the narrowphase's every-step lookups.
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
        self.0
    }
}

type PairMap = HashMap<(u64, u64), usize, BuildHasherDefault<KeyHasher>>;

/// Point speed (m/s) below which a body counts as resting.
const SLEEP_SPEED: f32 = 0.05;
/// Seconds an island must rest before it sleeps.
const TIME_TO_SLEEP: f32 = 0.5;
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
/// Contacts hold this much overlap so they don't flicker between steps. Each
/// layer of a stack sinks this far before its contacts push, so it is kept
/// small (the demo's 1 cm sank a 20-box stack 20 cm, settling it slowly);
/// Box3D keeps resting overlap near 5 mm.
pub(crate) const COLLISION_MARGIN: f32 = 0.003;
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
    pub velocity: Vec3,
    pub angular_velocity: Vec3,
    pub previous: Vec3,
    /// Linear and angular displacement since the step began, kept with the
    /// pose so contacts need no quaternion math per visit.
    pub moved: Vec3,
    pub turned: Vec3,
    /// Zero for static and kinematic bodies.
    pub mass: f32,
    /// World-space inertia tensor at the start of the step.
    pub inertia: Mat3,
    pub friction: f32,
    pub static_friction: f32,
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
        self.moved = position - self.initial_position;
        self.turned = quat_sub(rotation, self.initial_rotation);
    }
    fn dynamic(&self) -> bool {
        self.mass > 0.0 && !self.asleep
    }
    /// Solved this step: dynamic and awake.
    pub(crate) fn solvable(&self) -> bool {
        self.dynamic()
    }
    /// Can move this step, and so touch and wake an asleep body.
    fn active(&self) -> bool {
        self.dynamic()
            || (self.kinematic
                && (self.velocity != Vec3::ZERO || self.angular_velocity != Vec3::ZERO))
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
    penalty: Vec3A,
    lambda: Vec3A,
    stick: bool,
    /// Whether the last dual update's friction fit the static bound: static
    /// friction then applies next step (Sec. 3.3).
    held: bool,
}

impl Contact {
    pub fn new(feature: u32, ra: Vec3, rb: Vec3) -> Self {
        Self {
            feature,
            ra,
            rb,
            // A new contact starts static and turns kinetic once it must slide.
            held: true,
            ..Default::default()
        }
    }
}

pub(crate) struct Manifold {
    a: usize,
    b: usize,
    key: (u64, u64),
    /// Normal (B to A) then two tangents as matrix rows: the shared linear
    /// Jacobian.
    rows: Mat3A,
    /// The basis in A's frame, and B's pose relative to A, when the contacts
    /// were generated: the manifold is reused while B stays near that pose.
    local_basis: [Vec3; 3],
    relative: (Vec3, Quat),
    contacts: [Contact; collide::MAX_CONTACTS],
    /// Whether A is solved before B each iteration, and so applies the
    /// previous iteration's dual update (see `Solver::step`).
    first_is_a: bool,
    /// The narrowphase pass that last matched this manifold to a pair.
    seen: u32,
    len: usize,
    /// Kinetic and static friction coefficients.
    friction: f32,
    static_friction: f32,
    /// Heavier body's mass: penalties are per kilogram so tuning is unit-free.
    scale: f32,
    /// The contacts' solver state, one SIMD lane per contact.
    lanes: Lanes,
}

/// A manifold's contacts in structure-of-arrays form for the solver: each
/// `Vec4` holds one quantity for all four contacts, so a manifold is
/// evaluated and stamped four contacts per instruction. `[r]` indexes the
/// basis row (normal, tangent, tangent) and `[r][m]` an angular component.
/// Unused lanes are inactive and contribute exactly zero.
#[derive(Clone, Copy, Default)]
struct Lanes {
    c0: [Vec4; 3],
    /// Angular Jacobians at the start of the step (Sec. 4): linearized once,
    /// since re-evaluating at the current rotation double counts rotation.
    ja: [[Vec4; 3]; 3],
    jb: [[Vec4; 3]; 3],
    penalty: [Vec4; 3],
    lambda: [Vec4; 3],
    /// 1 or 0 per lane: a real contact; static friction applies this step;
    /// the normal row may be stiffness-rescaled; and the dual update's
    /// stick and held results, copied back to the contacts after the step.
    active: Vec4,
    sticking: Vec4,
    rescale_normal: Vec4,
    stick: Vec4,
    held: Vec4,
}

/// Lane-wise angular Hessian (upper triangle: 00 01 02 11 12 22) and
/// gradient summed over a body's manifolds, reduced once per body.
#[derive(Default)]
struct LaneSums {
    ang: [Vec4; 6],
    rhs: [Vec4; 3],
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

/// Four vectors, one per lane, rotated by `q`.
fn rotate4(q: Quat, v: [Vec4; 3]) -> [Vec4; 3] {
    let (x, y, z, w) = (q.x, q.y, q.z, q.w);
    // v + w t + q × t, with t = 2 q × v.
    let t = [
        (v[2] * y - v[1] * z) * 2.0,
        (v[0] * z - v[2] * x) * 2.0,
        (v[1] * x - v[0] * y) * 2.0,
    ];
    [
        v[0] + t[0] * w + (t[2] * y - t[1] * z),
        v[1] + t[1] * w + (t[0] * z - t[2] * x),
        v[2] + t[2] * w + (t[1] * x - t[0] * y),
    ]
}

/// Four vectors, one per lane, crossed with `n`.
fn cross4(a: [Vec4; 3], n: Vec3) -> [Vec4; 3] {
    [
        a[1] * n.z - a[2] * n.y,
        a[2] * n.x - a[0] * n.z,
        a[0] * n.y - a[1] * n.x,
    ]
}

fn mask(b: bool) -> f32 {
    if b { 1.0 } else { 0.0 }
}

/// Four contacts' linearized constraints (Sec. 4) and clamped forces.
struct Eval {
    c: [Vec4; 3],
    /// The force before its bounds were applied (λ⁺, Eq. 13).
    raw: [Vec4; 3],
    force: [Vec4; 3],
    normal: Vec4,
    tangent: Vec4,
    bound: Vec4,
}

/// Both bodies' displacement since the step began, shared by a manifold's contacts.
struct Motion {
    linear: Vec3A,
    angular_a: Vec3A,
    angular_b: Vec3A,
}

impl Motion {
    fn new(a: &Body, b: &Body) -> Self {
        Self {
            linear: (a.moved - b.moved).into(),
            angular_a: a.turned.into(),
            angular_b: b.turned.into(),
        }
    }
}

/// `m` with row `k` scaled by `s[k]`.
fn scale_rows(m: Mat3A, s: Vec3A) -> Mat3A {
    Mat3A::from_cols(m.x_axis * s, m.y_axis * s, m.z_axis * s)
}

impl Manifold {
    fn eval(&self, m: &Motion, alpha: f32) -> Eval {
        let l = &self.lanes;
        // The linear rows are the shared basis: one value per row for all lanes.
        let linear = self.rows * m.linear;
        let (ta, tb) = (m.angular_a, m.angular_b);
        let c: [Vec4; 3] = std::array::from_fn(|r| {
            let (ja, jb) = (&l.ja[r], &l.jb[r]);
            (l.c0[r] * (1.0 - alpha)
                + Vec4::splat(linear[r])
                + ja[0] * ta.x
                + ja[1] * ta.y
                + ja[2] * ta.z
                + jb[0] * tb.x
                + jb[1] * tb.y
                + jb[2] * tb.z)
                * l.active
        });
        let raw: [Vec4; 3] = std::array::from_fn(|r| l.penalty[r] * c[r] + l.lambda[r]);
        let fx = raw[0].min(Vec4::ZERO);
        let normal = -fx;
        let tangent = sqrt4(raw[1] * raw[1] + raw[2] * raw[2]);
        // Static friction while the contact stuck last frame; kinetic as soon
        // as holding it would take more than the static bound (Sec. 3.3).
        let static_bound = normal * self.static_friction;
        let use_static = l.sticking.cmpgt(Vec4::ZERO) & tangent.cmple(static_bound);
        let bound = Vec4::select(use_static, static_bound, normal * self.friction);
        let over = tangent.cmpgt(bound) & tangent.cmpgt(Vec4::ZERO);
        let scale = Vec4::select(over, bound / tangent.max(Vec4::splat(f32::MIN_POSITIVE)), Vec4::ONE);
        Eval {
            c,
            raw,
            force: [fx, raw[1] * scale, raw[2] * scale],
            normal,
            tangent,
            bound,
        }
    }

    /// Accumulate this manifold's force and Hessian approximation for `body`
    /// (Eqs. 5, 6), four contacts at a time. The angular terms stay lane-wise
    /// in `sums` until the body's solve; the linear rows are the shared basis,
    /// so the linear block, coupling and gradient sum once per manifold.
    #[allow(clippy::too_many_arguments)]
    fn stamp(
        &self,
        a: &Body,
        b: &Body,
        is_a: bool,
        alpha: f32,
        damping: f32,
        system: &mut System,
        sums: &mut LaneSums,
    ) {
        let e = self.eval(&Motion::new(a, b), alpha);
        let mut force = e.force;
        let l = &self.lanes;
        // Stiffness rescaling (Eq. 14): a clamped row contributes the slope
        // that reaches its bound, not the full penalty. Not for the normal
        // row of a contact that began the step penetrating: an iteration that
        // overshoots it apart would leave it no stiffness or force in the
        // next, so the body falls back, and an even iteration count leaves it
        // sinking through.
        let mut k = l.penalty;
        let clamped: [_; 3] = std::array::from_fn(|r| {
            let row = if r == 0 { l.rescale_normal.cmpgt(Vec4::ZERO) } else { l.active.cmpgt(Vec4::ZERO) };
            e.force[r].cmpne(e.raw[r]) & e.c[r].cmpne(Vec4::ZERO) & row
        });
        if (clamped[0] | clamped[1] | clamped[2]).any() {
            for r in 0..3 {
                let rescaled = ((e.force[r] - l.lambda[r]) / e.c[r]).clamp(Vec4::ZERO, l.penalty[r]);
                k[r] = Vec4::select(clamped[r], rescaled, k[r]);
            }
        }
        let j = if is_a { &l.ja } else { &l.jb };
        // Contact damping, VBD's Rayleigh term (Chen et al. 2024): a pushing
        // normal row resists this body's own motion along it this step, with
        // stiffness (k_d / h) k. Corrections otherwise bounce stacks.
        if damping > 0.0 {
            let (body, sign) = if is_a { (a, 1.0) } else { (b, -1.0) };
            let along = (self.rows * Vec3A::from(body.moved)).x * sign;
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
            sums.ang[0] += w[0] * jr[0];
            sums.ang[1] += w[0] * jr[1];
            sums.ang[2] += w[0] * jr[2];
            sums.ang[3] += w[1] * jr[1];
            sums.ang[4] += w[1] * jr[2];
            sums.ang[5] += w[2] * jr[2];
            sums.rhs[0] += jr[0] * f;
            sums.rhs[1] += jr[1] * f;
            sums.rhs[2] += jr[2] * f;
        }
        let s1 = sum4(k[0], k[1], k[2], force[0]);
        let s2 = sum4(force[1], force[2], kj[0][0], kj[0][1]);
        let s3 = sum4(kj[0][2], kj[1][0], kj[1][1], kj[1][2]);
        let s4 = sum4(kj[2][0], kj[2][1], kj[2][2], Vec4::ZERO);
        let stiffness = Vec3A::new(s1.x, s1.y, s1.z);
        let force = Vec3A::new(s1.w, s2.x, s2.y);
        // Rows are K J per basis row.
        let weighted = Mat3A::from_cols(
            Vec3A::new(s2.z, s3.y, s4.x),
            Vec3A::new(s2.w, s3.z, s4.y),
            Vec3A::new(s3.x, s3.w, s4.z),
        );
        let sign = if is_a { 1.0 } else { -1.0 };
        let bt = self.rows.transpose();
        system.lin += bt * scale_rows(self.rows, stiffness);
        system.cross += weighted.transpose() * self.rows * sign;
        system.rhs_lin += bt * force * sign;
    }

    /// Store forces and ramp penalties (Eqs. 11, 12), four contacts at a time.
    fn dual(&mut self, a: &Body, b: &Body, alpha: f32, beta: f32) {
        let e = self.eval(&Motion::new(a, b), alpha);
        let (beta, max) = (beta * self.scale, Vec4::splat(PENALTY_MAX * self.scale));
        let bound = Vec4::splat(LAMBDA_MAX * self.scale);
        let (friction, l) = (self.static_friction, &mut self.lanes);
        for r in 0..3 {
            l.lambda[r] = e.force[r].clamp(-bound, bound);
        }
        l.held = Vec4::select(e.tangent.cmple(e.normal * friction), Vec4::ONE, Vec4::ZERO);
        let pushing = e.force[0].cmplt(Vec4::ZERO);
        l.penalty[0] = Vec4::select(pushing, (l.penalty[0] + e.c[0].abs() * beta).min(max), l.penalty[0]);
        let inside = e.tangent.cmple(e.bound);
        for r in 1..3 {
            l.penalty[r] = Vec4::select(inside, (l.penalty[r] + e.c[r].abs() * beta).min(max), l.penalty[r]);
        }
        let slip = sqrt4(e.c[1] * e.c[1] + e.c[2] * e.c[2]);
        let stuck = Vec4::select(slip.cmplt(Vec4::splat(STICK_THRESHOLD)), Vec4::ONE, Vec4::ZERO);
        l.stick = Vec4::select(inside, stuck, l.stick);
    }

    /// A manifold for a new pair, warm started from `previous` (a sleeping
    /// island's contacts); `None` when the shapes do not touch.
    fn create(
        bodies: &[Body],
        a: usize,
        b: usize,
        previous: Option<&Manifold>,
        settings: &PhysicsSettings,
    ) -> Option<Self> {
        let (ba, bb) = (&bodies[a], &bodies[b]);
        let (basis, mut contacts, len) = collide::collide(ba, bb)?;
        if let Some(previous) = previous {
            previous.warm(ba, bb, &mut contacts[..len]);
        }
        let inverse = ba.rotation.conjugate();
        let mut m = Manifold {
            a,
            b,
            key: (ba.key, bb.key),
            rows: Mat3A::IDENTITY,
            local_basis: basis.map(|v| inverse * v),
            relative: (inverse * (bb.position - ba.position), inverse * bb.rotation),
            contacts,
            first_is_a: true,
            seen: 0,
            len,
            friction: 0.0,
            static_friction: 0.0,
            scale: 0.0,
            lanes: Lanes::default(),
        };
        m.prepare(ba, bb, basis, settings);
        Some(m)
    }

    /// Bring a kept manifold up to this step: reuse its contacts while B
    /// stays near the pose they were made at, else collide again, warm
    /// starting from the old contacts. False once the shapes separate.
    fn refresh(&mut self, bodies: &[Body], settings: &PhysicsSettings) -> bool {
        let (ba, bb) = (&bodies[self.a], &bodies[self.b]);
        let inverse = ba.rotation.conjugate();
        let relative = (inverse * (bb.position - ba.position), inverse * bb.rotation);
        let reusable = self.relative.0.distance_squared(relative.0) < REUSE_DISTANCE * REUSE_DISTANCE
            && quat_sub(relative.1, self.relative.1).length_squared() < REUSE_ANGLE * REUSE_ANGLE;
        let basis = if reusable {
            self.local_basis.map(|v| ba.rotation * v)
        } else {
            let Some((basis, mut contacts, len)) = collide::collide(ba, bb) else {
                return false;
            };
            self.warm(ba, bb, &mut contacts[..len]);
            (self.contacts, self.len) = (contacts, len);
            self.local_basis = basis.map(|v| inverse * v);
            self.relative = relative;
            basis
        };
        self.prepare(ba, bb, basis, settings);
        true
    }

    /// Carry matching contacts' forces, penalties and friction state (and,
    /// while static friction held, their anchors) to freshly collided ones.
    fn warm(&self, ba: &Body, bb: &Body, contacts: &mut [Contact]) {
        // A rolling sphere's contact point moves over its surface: its
        // anchors never stick, only its forces warm start.
        let rolls = matches!(ba.shape, Collider::Sphere { .. })
            || matches!(bb.shape, Collider::Sphere { .. });
        for contact in contacts {
            let matched = self.contacts[..self.len]
                .iter()
                .find(|p| p.feature == contact.feature);
            if let Some(p) = matched {
                let (ra, rb) = (contact.ra, contact.rb);
                *contact = *p;
                if !p.stick || rolls {
                    (contact.ra, contact.rb) = (ra, rb);
                }
            }
        }
    }

    /// The step's constraint errors, Jacobians, materials and warm-started
    /// penalties and duals (Eq. 19), four contacts at a time in the lanes.
    /// The contacts keep last step's state until `store`.
    fn prepare(&mut self, ba: &Body, bb: &Body, basis: [Vec3; 3], settings: &PhysicsSettings) {
        let (alpha, gamma) = (settings.alpha, settings.gamma);
        let scale = ba.mass.max(bb.mass);
        let mut l = Lanes::default();
        let (mut ra, mut rb) = ([Vec4::ZERO; 3], [Vec4::ZERO; 3]);
        for (i, c) in self.contacts[..self.len].iter().enumerate() {
            for k in 0..3 {
                ra[k][i] = c.ra[k];
                rb[k][i] = c.rb[k];
                l.penalty[k][i] = c.penalty[k];
                l.lambda[k][i] = c.lambda[k];
            }
            l.active[i] = 1.0;
            // Static friction while the contact held at the last step's end.
            l.sticking[i] = mask(c.held);
            l.stick[i] = mask(c.stick);
            l.held[i] = mask(c.held);
        }
        let (ra, rb) = (rotate4(ba.rotation, ra), rotate4(bb.rotation, rb));
        let gap = ba.position - bb.position;
        let d: [Vec4; 3] = std::array::from_fn(|k| (Vec4::splat(gap[k]) + ra[k] - rb[k]) * l.active);
        for (r, n) in basis.iter().enumerate() {
            l.ja[r] = cross4(ra, *n);
            // rb × (−n) = n × rb.
            l.jb[r] = cross4(rb, *n).map(|v| -v);
            let margin = if r == 0 { COLLISION_MARGIN } else { 0.0 };
            l.c0[r] = (d[0] * n.x + d[1] * n.y + d[2] * n.z + Vec4::splat(margin)) * l.active;
        }
        let (min, max) = (Vec4::splat(PENALTY_MIN * scale), Vec4::splat(PENALTY_MAX * scale));
        for k in 0..3 {
            l.lambda[k] *= alpha * gamma;
            l.penalty[k] = (l.penalty[k] * gamma).clamp(min, max) * l.active;
        }
        l.rescale_normal = Vec4::select(l.c0[0].cmpge(Vec4::ZERO), l.active, Vec4::ZERO);
        self.lanes = l;
        self.rows = Mat3A::from_cols(basis[0].into(), basis[1].into(), basis[2].into()).transpose();
        self.friction = (ba.friction * bb.friction).sqrt();
        self.static_friction = (ba.static_friction * bb.static_friction).sqrt();
        self.scale = scale;
    }

    /// Copy the lanes' warm-start state back to the contacts, for matching
    /// them to next step's contacts.
    fn store(&mut self) {
        let l = &self.lanes;
        for (i, contact) in self.contacts[..self.len].iter_mut().enumerate() {
            contact.penalty = Vec3A::new(l.penalty[0][i], l.penalty[1][i], l.penalty[2][i]);
            contact.lambda = Vec3A::new(l.lambda[0][i], l.lambda[1][i], l.lambda[2][i]);
            contact.stick = l.stick[i] > 0.5;
            contact.held = l.held[i] > 0.5;
        }
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
}

/// One body's block solve (Eqs. 4-6) against the manifolds and links
/// `adjacency` lists for it; returns its new pose.
#[allow(clippy::too_many_arguments)]
fn solve_body<'a>(
    body_at: impl Fn(usize) -> &'a Body,
    manifold_at: impl Fn(usize) -> &'a Manifold,
    link_at: impl Fn(usize) -> &'a Link,
    adjacency: &[u32],
    i: usize,
    alpha: f32,
    damping: f32,
    dt: f32,
) -> (Vec3, Quat) {
    let body = body_at(i);
    let scale = 1.0 / (dt * dt);
    let mass = body.mass * scale;
    let inertia = Mat3A::from(body.inertia * scale);
    let mut system = System {
        lin: Mat3A::from_diagonal(Vec3::splat(mass)),
        ang: inertia,
        cross: Mat3A::ZERO,
        rhs_lin: Vec3A::from(body.position - body.inertial_position) * mass,
        rhs_ang: inertia * Vec3A::from(quat_sub(body.rotation, body.inertial_rotation)),
    };
    let mut sums = LaneSums::default();
    for &entry in adjacency {
        if entry & LINK != 0 {
            let link = link_at((entry & !LINK) as usize);
            let a = link.a.map(&body_at);
            link.stamp(a, body_at(link.b), link.a == Some(i), alpha, &mut system);
        } else {
            let m = manifold_at(entry as usize);
            m.stamp(body_at(m.a), body_at(m.b), i == m.a, alpha, damping, &mut system, &mut sums);
        }
    }
    let h = sum4(sums.ang[0], sums.ang[1], sums.ang[2], sums.ang[3]);
    let g = sum4(sums.ang[4], sums.ang[5], sums.rhs[0], sums.rhs[1]);
    system.ang += Mat3A::from_cols(
        Vec3A::new(h.x, h.y, h.z),
        Vec3A::new(h.y, h.w, g.x),
        Vec3A::new(h.z, g.x, g.y),
    );
    system.rhs_ang += Vec3A::new(g.z, g.w, sums.rhs[2].element_sum());
    let (dx, dr) = system.solve();
    (body.position + dx, quat_add(body.rotation, dr))
}

/// The previous iteration's dual update (Algorithm 1, lines 26-35) for one
/// manifold or link, applied when its first-solved body reaches it in the
/// next iteration. Neither body has moved since that iteration ended, so this
/// equals a separate dual pass while saving a pass and a barrier.
fn first_dual(
    bodies: &[Body],
    manifolds: &mut [Manifold],
    links: &mut [Link],
    entry: u32,
    i: usize,
    settings: &PhysicsSettings,
) {
    if entry & LINK != 0 {
        let link = &mut links[(entry & !LINK) as usize];
        let first = if link.first_is_a { link.a } else { Some(link.b) };
        if first == Some(i) {
            link.dual(link.a.map(|j| &bodies[j]), &bodies[link.b], settings);
        }
    } else {
        let m = &mut manifolds[entry as usize];
        if i == if m.first_is_a { m.a } else { m.b } {
            m.dual(&bodies[m.a], &bodies[m.b], settings.alpha, settings.beta);
        }
    }
}

/// `first_dual` through the parallel region's shared pointers.
///
/// # Safety
/// The caller is `i`'s thread in `i`'s color phase: only `i` may write the
/// entries it visits first, and no thread writes the bodies they read.
unsafe fn first_dual_raw<'a>(
    body_at: impl Fn(usize) -> &'a Body,
    manifolds: Shared<Manifold>,
    links: Shared<Link>,
    entry: u32,
    i: usize,
    settings: &PhysicsSettings,
) {
    if entry & LINK != 0 {
        let link = unsafe { &mut *links.get().add((entry & !LINK) as usize) };
        let first = if link.first_is_a { link.a } else { Some(link.b) };
        if first == Some(i) {
            link.dual(link.a.map(&body_at), body_at(link.b), settings);
        }
    } else {
        let m = unsafe { &mut *manifolds.get().add(entry as usize) };
        if i == if m.first_is_a { m.a } else { m.b } {
            m.dual(body_at(m.a), body_at(m.b), settings.alpha, settings.beta);
        }
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

#[derive(Default)]
pub(crate) struct Solver {
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
    manifolds: Vec<Manifold>,
    /// Manifold index by body-key pair, for warm starting the next step.
    cache: PairMap,
    /// Manifolds of sleeping islands, tagged by island, kept to warm start
    /// them on waking; `stash_index` finds them by key.
    stash: Vec<(u32, Manifold)>,
    stash_index: PairMap,
    next_island: u32,
    /// The sleep pass's union-find, rest times and island ids, reused.
    parent: Vec<usize>,
    rest: Vec<f32>,
    island_of: Vec<u32>,
    /// Joints and springs, kept in step by the physics world.
    pub links: Vec<Link>,
    /// Body-key pairs a link joins: they do not collide.
    linked: PairMap,
    boxes: Vec<Aabb>,
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
}

impl Solver {
    pub fn contact_count(&self) -> usize {
        let stashed = self.stash.iter().map(|(_, m)| m);
        self.manifolds.iter().chain(stashed).map(|m| m.len).sum()
    }

    /// `exclusive`: the step runs alone on its own thread pool, so the
    /// iterations may hold every pool thread at a barrier. On a shared pool
    /// two worlds' barriers could wait on each other's threads forever.
    pub fn step(&mut self, settings: &PhysicsSettings, dt: f32, exclusive: bool) {
        let requested: Vec<u32> = (self.bodies.iter())
            .filter(|b| b.asleep && b.wake)
            .map(|b| b.island)
            .collect();
        let mut woken = self.wake_islands(&requested);
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
        self.broadphase();
        self.narrowphase(settings);
        // Real contact with a moving body wakes an asleep island; collide
        // again so the island's own contacts join this step.
        let linked_pairs = (self.links.iter().filter(|l| !l.broken))
            .filter_map(|l| l.a.map(|a| (a, l.b)));
        let touched: Vec<u32> = (self.manifolds.iter().map(|m| (m.a, m.b)))
            .chain(linked_pairs)
            .flat_map(|(a, b)| [(a, b), (b, a)])
            .filter(|&(x, y)| self.bodies[x].asleep && self.bodies[y].active())
            .map(|(x, _)| self.bodies[x].island)
            .collect();
        let also = self.wake_islands(&touched);
        if !also.is_empty() {
            woken.extend(also);
            self.broadphase();
            self.narrowphase(settings);
        }
        if !woken.is_empty() {
            self.stash.retain(|(island, _)| !woken.contains(island));
            self.index_stash();
        }
        for link in &mut self.links {
            if link.solved(&self.bodies) {
                link.initialize(&self.bodies, settings);
            }
        }
        self.color();

        let gravity = settings.gravity;
        self.bodies
            .par_iter_mut()
            .with_min_len(BODY_BATCH)
            .for_each(|body| {
                body.initial_position = body.position;
                body.initial_rotation = body.rotation;
                (body.moved, body.turned) = (Vec3::ZERO, Vec3::ZERO);
                if body.dynamic() {
                    // Inertial target (Eq. 2) and adaptive warm start (VBD).
                    body.inertial_position =
                        body.position + body.velocity * dt + gravity * (dt * dt);
                    body.inertial_rotation = quat_add(body.rotation, body.angular_velocity * dt);
                    let accel = (body.velocity - body.previous) / dt;
                    let weight = (accel.dot(gravity) / gravity.length_squared()).clamp(0.0, 1.0);
                    let weight = if weight.is_finite() { weight } else { 0.0 };
                    let start = body.position + body.velocity * dt + gravity * (weight * dt * dt);
                    body.place(start, body.inertial_rotation);
                } else if body.kinematic {
                    let rotation = quat_add(body.rotation, body.angular_velocity * dt);
                    body.place(body.position + body.velocity * dt, rotation);
                }
            });

        for &i in &self.free {
            let body = &mut self.bodies[i];
            body.place(body.inertial_position, body.inertial_rotation);
        }
        let threads = rayon::current_num_threads();
        if threads == 1 || !exclusive {
            let (alpha, beta) = (settings.alpha, settings.beta);
            for iteration in 0..settings.iterations {
                for color in &self.colors {
                    for &i in color {
                        let range = self.offsets[i] as usize..self.offsets[i + 1] as usize;
                        if iteration > 0 {
                            let (manifolds, links) = (&mut self.manifolds, &mut self.links);
                            for &entry in &self.adjacency[range.clone()] {
                                first_dual(&self.bodies, manifolds, links, entry, i, settings);
                            }
                        }
                        let (bodies, adjacency) = (&self.bodies, &self.adjacency[range]);
                        let (manifolds, links) = (&self.manifolds, &self.links);
                        let pose = solve_body(
                            |j| &bodies[j],
                            |k| &manifolds[k],
                            |k| &links[k],
                            adjacency,
                            i,
                            alpha,
                            settings.contact_damping / dt,
                            dt,
                        );
                        self.bodies[i].place(pose.0, pose.1);
                    }
                }
            }
            // The last iteration's dual update, for warm starting the next step.
            let bodies = &self.bodies;
            for m in &mut self.manifolds {
                m.dual(&bodies[m.a], &bodies[m.b], alpha, beta);
                m.store();
            }
            for link in self.links.iter_mut().filter(|l| l.solved(bodies)) {
                link.dual(link.a.map(|j| &bodies[j]), &bodies[link.b], settings);
            }
        } else {
            self.iterate_parallel(settings, dt, threads);
        }

        self.bodies
            .par_iter_mut()
            .with_min_len(BODY_BATCH)
            .for_each(|body| {
                body.previous = body.velocity;
                if body.dynamic() {
                    body.velocity = body.moved / dt;
                    body.angular_velocity = body.turned / dt;
                }
            });
        self.sleep(dt);
    }

    /// Sleeping islands with a contact on any of these bodies, which are
    /// going away or were moved by the caller.
    pub fn islands_touching(&self, keys: &[u64]) -> Vec<u32> {
        (self.stash.iter())
            .filter(|(_, m)| keys.contains(&m.key.0) || keys.contains(&m.key.1))
            .map(|&(island, _)| island)
            .collect()
    }

    /// The primal/dual iterations in one parallel region: each pool thread
    /// takes its share of every color, then of the manifolds, meeting at a
    /// barrier between phases. Forking once per color instead cost more than
    /// the work in small and medium scenes. Results match the serial path.
    fn iterate_parallel(&mut self, settings: &PhysicsSettings, dt: f32, threads: usize) {
        let (alpha, beta) = (settings.alpha, settings.beta);
        let bodies = Shared(self.bodies.as_mut_ptr());
        let manifolds = Shared(self.manifolds.as_mut_ptr());
        let manifold_count = self.manifolds.len();
        let links = Shared(self.links.as_mut_ptr());
        let link_count = self.links.len();
        let (colors, offsets, adjacency) = (&self.colors, &self.offsets, &self.adjacency);
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
            for iteration in 0..settings.iterations {
                for color in colors {
                    for &i in &color[share(color.len())] {
                        let range = offsets[i] as usize..offsets[i + 1] as usize;
                        if iteration > 0 {
                            for &entry in &adjacency[range.clone()] {
                                // SAFETY: only a manifold's first visitor writes
                                // it, in its own color phase.
                                unsafe { first_dual_raw(body_at, manifolds, links, entry, i, settings) };
                            }
                        }
                        let pose = solve_body(
                            body_at,
                            |k| unsafe { &*manifolds.get().add(k) },
                            |k| unsafe { &*links.get().add(k) },
                            &adjacency[range],
                            i,
                            alpha,
                            settings.contact_damping / dt,
                            dt,
                        );
                        unsafe { (*bodies.get().add(i)).place(pose.0, pose.1) };
                    }
                    barrier.wait();
                }
            }
            // The last iteration's dual update, for warm starting the next step.
            for k in share(manifold_count + link_count) {
                if k < manifold_count {
                    let m = unsafe { &mut *manifolds.get().add(k) };
                    m.dual(body_at(m.a), body_at(m.b), alpha, beta);
                    m.store();
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
        (self.stash_index).extend(self.stash.iter().enumerate().map(|(i, (_, m))| (m.key, i)));
    }

    /// Put to sleep each island (dynamic bodies joined by contacts) whose
    /// bodies have all rested long enough and that no moving kinematic touches.
    fn sleep(&mut self, dt: f32) {
        let n = self.bodies.len();
        for body in self.bodies.iter_mut().filter(|b| b.dynamic()) {
            let speed = body.velocity.length() + body.angular_velocity.length() * body.bound();
            body.sleep_time = if speed < SLEEP_SPEED { body.sleep_time + dt } else { 0.0 };
        }
        // Islands sleep only once every body in them has rested this long.
        let resting = self.bodies.iter().any(|b| b.dynamic() && b.sleep_time >= TIME_TO_SLEEP);
        if !resting {
            return;
        }
        let mut parent = std::mem::take(&mut self.parent);
        parent.clear();
        parent.extend(0..n);
        fn find(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            i
        }
        let joined = (self.manifolds.iter().map(|m| (m.a, m.b)))
            .chain(self.links.iter().filter(|l| !l.broken).filter_map(|l| l.a.map(|a| (a, l.b))));
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
            let deep = m.lanes.c0[0].cmplt(Vec4::splat(-SLEEP_PENETRATION)).any();
            for (x, y) in [(m.a, m.b), (m.b, m.a)] {
                let (bx, by) = (&self.bodies[x], &self.bodies[y]);
                if bx.dynamic() && (deep || (by.active() && !by.dynamic())) {
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
        let mut island = std::mem::take(&mut self.island_of);
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
        (self.parent, self.rest, self.island_of) = (parent, rest, island);
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
        (self.cache).extend(self.manifolds.iter().enumerate().map(|(i, m)| (m.key, i)));
    }

    /// Sweep and prune oriented-box AABBs along the axis where body centers
    /// vary most. The boxes live in one compact array: the sweep's inner loop
    /// then streams through cache instead of chasing whole bodies.
    fn broadphase(&mut self) {
        let bodies = &self.bodies;
        if bodies.len() < 2 {
            self.pairs.clear();
            self.fat.clear();
            return;
        }
        let bounds = |(i, b): (usize, &Body)| {
            let extent = match b.shape {
                Collider::Cuboid { half_extents } => {
                    let m = Mat3::from_quat(b.rotation);
                    let abs = |row: Vec3| row.abs().dot(half_extents);
                    Vec3::new(abs(m.row(0)), abs(m.row(1)), abs(m.row(2)))
                }
                Collider::Sphere { radius } => Vec3::splat(radius),
            } + COLLISION_MARGIN;
            let flags = u32::from(b.dynamic())
                | u32::from(b.active()) << 1
                | u32::from(b.asleep) << 2;
            Aabb {
                min: b.position - extent,
                max: b.position + extent,
                index: i as u32,
                flags,
            }
        };
        (bodies.par_iter().enumerate().with_min_len(BODY_BATCH).map(bounds))
            .collect_into_vec(&mut self.tight);
        // Last step's pairs hold while every body stays inside its enlarged
        // box and no body changed: settling bodies skip the sweep entirely.
        let kept = |fat: &[Aabb], keys: &[u64], i: usize, t: &Aabb| {
            keys.get(i) == Some(&bodies[i].key)
                && fat[i].min.cmple(t.min).all()
                && t.max.cmple(fat[i].max).all()
        };
        let fits = self.fat.len() == bodies.len()
            && (self.tight.iter().enumerate())
                .all(|(i, t)| t.flags == self.fat[i].flags && kept(&self.fat, &self.fat_keys, i, t));
        if fits {
            return;
        }
        self.fat.resize(bodies.len(), self.tight[0]);
        self.fat_keys.resize(bodies.len(), 0);
        for (i, t) in self.tight.iter().enumerate() {
            if !kept(&self.fat, &self.fat_keys, i, t) {
                // Only moving bodies get room to move.
                let margin = if t.flags & 2 != 0 { FAT_MARGIN } else { 0.0 };
                self.fat[i] = Aabb { min: t.min - margin, max: t.max + margin, ..*t };
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
        let variance = bodies.iter().map(|b| (b.position - mean) * (b.position - mean)).sum::<Vec3>();
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
        if self.boxes.len() < SERIAL_SORT {
            self.boxes.sort_unstable_by(order);
        } else {
            self.boxes.par_sort_unstable_by(order);
        }
        const DYNAMIC: u32 = 1;
        const ACTIVE: u32 = 2;
        const ASLEEP: u32 = 4;
        let boxes = &self.boxes;
        // Each chunk sweeps forward from its own boxes; joining the chunks'
        // pairs in order keeps the result independent of thread count.
        let sweep = |(c, chunk): (usize, &[Aabb])| {
            let mut pairs = Vec::new();
            for (k, a) in chunk.iter().enumerate() {
                for b in &boxes[c * SWEEP_CHUNK + k + 1..] {
                    if b.min[axis] > a.max[axis] {
                        break;
                    }
                    // Pairs an asleep body forms with a moving one only wake it.
                    let relevant = (a.flags | b.flags) & DYNAMIC != 0
                        || (a.flags & ASLEEP != 0 && b.flags & ACTIVE != 0)
                        || (b.flags & ASLEEP != 0 && a.flags & ACTIVE != 0);
                    if relevant && a.min.cmple(b.max).all() && b.min.cmple(a.max).all() {
                        let (i, j) = (a.index as usize, b.index as usize);
                        let first = bodies[i].key < bodies[j].key;
                        pairs.push(if first { (i, j) } else { (j, i) });
                    }
                }
            }
            pairs
        };
        // Below SERIAL_SWEEP boxes one task sweeps them all.
        let min = if boxes.len() < SERIAL_SWEEP { usize::MAX } else { 1 };
        (boxes.par_chunks(SWEEP_CHUNK).enumerate().with_min_len(min).map(sweep))
            .collect_into_vec(&mut self.swept);
        self.pairs.extend(self.swept.drain(..).flatten());
    }

    /// Update the persistent manifolds from this step's pairs: kept pairs
    /// are refreshed in place, new ones collided, and dropped or separated
    /// ones removed. Contacts merge with the previous step's to warm start
    /// their forces and penalties (Eq. 19).
    fn narrowphase(&mut self, settings: &PhysicsSettings) {
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
                    (m.a, m.b, m.seen) = (a, b, pass);
                }
                None => self.fresh.push((a, b)),
            }
        }
        let bodies = &self.bodies;
        self.manifolds
            .par_iter_mut()
            .with_min_len(PAIR_BATCH)
            .for_each(|m| {
                if m.seen != pass || !m.refresh(bodies, settings) {
                    m.len = 0;
                }
            });
        let (stash, stash_index) = (&self.stash, &self.stash_index);
        self.fresh
            .par_iter()
            .with_min_len(PAIR_BATCH)
            .map(|&(a, b)| {
                let key = (bodies[a].key, bodies[b].key);
                let previous = stash_index.get(&key).map(|&i| &stash[i].1);
                Manifold::create(bodies, a, b, previous, settings)
            })
            .collect_into_vec(&mut self.found);
        let before = self.manifolds.len();
        self.manifolds.retain(|m| m.len > 0);
        let changed = self.manifolds.len() != before || self.found.iter().any(Option::is_some);
        self.manifolds.extend(self.found.drain(..).flatten());
        if changed {
            self.recolor = true;
            self.cache.clear();
            (self.cache).extend(self.manifolds.iter().enumerate().map(|(i, m)| (m.key, i)));
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
        self.colored.extend(bodies.iter().map(|b| (b.key, b.dynamic())));
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
        let mut fill: Vec<u32> = self.offsets[..n].to_vec();
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
        let mut color_of = vec![u32::MAX; n];
        // stamp[c] == i marks color c as taken by a neighbor of body i.
        let mut stamp: Vec<usize> = Vec::new();
        for i in (0..n).filter(|&i| self.bodies[i].dynamic()) {
            if self.offsets[i] == self.offsets[i + 1] {
                self.free.push(i);
                continue;
            }
            for &entry in &self.adjacency[self.offsets[i] as usize..self.offsets[i + 1] as usize] {
                let other = if entry & LINK != 0 {
                    let link = &self.links[(entry & !LINK) as usize];
                    if link.a == Some(i) { Some(link.b) } else { link.a }
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
    fn lane_rotation_and_cross_match_scalar() {
        let q = Quat::from_euler(glam::EulerRot::XYZ, 0.3, -1.1, 2.0);
        let n = Vec3::new(0.2, -0.7, 0.4).normalize();
        let points = [Vec3::new(1.0, 2.0, 3.0), Vec3::new(-0.5, 0.25, 4.0), Vec3::X, Vec3::new(0.0, -3.0, 0.5)];
        let lanes: [Vec4; 3] = std::array::from_fn(|k| Vec4::from_array(points.map(|p| p[k])));
        let (rotated, crossed) = (rotate4(q, lanes), cross4(lanes, n));
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
    }
}
