//! Joints and springs: constraints between two bodies, or a body and a world
//! point. Joints are hard (augmented Lagrangian, Sec. 3.1) or soft with a
//! finite stiffness ramped up to (Eq. 16); springs are soft. Joints evaluate
//! exactly each iteration with the diagonal geometric stiffness of Sec. 3.5,
//! following the AVBD 3D demo, except that hard joints apply their force along
//! the step-start direction (see `Link::stamp`); contacts instead use the
//! step-start Taylor expansion (Sec. 4).

use crate::solver::{Body, System, quat_sub};
use crate::{Collider, JointStage, Muscle, PhysicsSettings, Strength};
use glam::{Mat3A, Quat, Vec3, Vec3A};

const PENALTY_MIN: f32 = 1.0;
const PENALTY_MAX: f32 = 1.0e10;
const LAMBDA_MAX: f32 = 1.0e10;
/// Rotation per step (rad) at which a hard joint's force has fully left the
/// step-start direction for the step-end one; see `Link::stamp`. Measured:
/// chains that whip gain energy from about 0.15, and explode from 0.25.
const START_FORCE_TURN: f32 = 0.07;
/// Seconds over which a joint's load is averaged before it is compared with the
/// joint's strength, so one unconverged step does not snap it.
const LOAD_TIME: f32 = 0.03;
/// A row at this share of its cap counts as having reached it.
const LOAD_MARGIN: f32 = 0.999;
/// Stands for "no limit" on a rotation axis: radians, but finite for the penalty.
const FREE_ANGLE: f32 = 1000.0;
/// The linear and angular rows of `Link::strength` and its state.
const LINEAR: usize = 0;
const ANGULAR: usize = 1;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Kind {
    Joint {
        linear: f32,
        angular: f32,
        /// B's rotation relative to A when the joint was made: the angular
        /// constraint holds it.
        rest: Quat,
    },
    Spring {
        stiffness: f32,
        rest_length: f32,
    },
}

/// A link's per-step solver state. `a` is `None` for a world anchor.
#[derive(Debug, Clone)]
pub(crate) struct Link {
    pub kind: Kind,
    pub a: Option<usize>,
    pub b: usize,
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    pub broken: bool,
    /// Whether A is solved before B each iteration (see `Manifold`).
    pub first_is_a: bool,
    /// Heavier body's mass, so penalties and β are per kilogram like contacts.
    scale: f32,
    /// Squared characteristic length that turns angle error into distance.
    arm: f32,
    c0_linear: Vec3,
    c0_angular: Vec3,
    /// Anchor arms at the step's start, and how much of the force follows them.
    arm0_a: Vec3,
    arm0_b: Vec3,
    start_force: f32,
    penalty_linear: Vec3,
    penalty_angular: Vec3,
    lambda_linear: Vec3,
    lambda_angular: Vec3,
    /// How the linear and angular rows give way, how far each has, and each
    /// one's averaged load in N and N·m.
    pub strength: [Strength; 2],
    strain: [f32; 2],
    pub load: [f32; 2],
    pub stage: JointStage,
    /// Rotation range per axis of the frame `rotation_a * rest`, in radians.
    limit_min: Vec3,
    limit_max: Vec3,
    limited: bool,
    /// Step-start limit violation, which only stabilization corrects.
    limit0: Vec3,
    penalty_limit: Vec3,
    lambda_limit: Vec3,
    /// Stiffness of the stops (N·m/rad); infinite is rigid.
    pub limit_stiffness: f32,
    pub muscle: Option<Muscle>,
    /// The torque the muscle can make this step (N·m).
    muscle_limit: f32,
    /// Stretch (m) that doubles a soft joint's stiffness, and that stiffness now.
    pub stiffening: f32,
    soft_limit: f32,
}

/// The two bodies' current state as the link sees it.
struct Ends {
    point_a: Vec3,
    point_b: Vec3,
    arm_a: Vec3,
    arm_b: Vec3,
    rotation_a: Quat,
    rotation_b: Quat,
}

/// A shape's full widths, for the angular constraint's length scale.
fn size(shape: Collider) -> Vec3 {
    match shape {
        Collider::Cuboid { half_extents } => half_extents * 2.0,
        Collider::Sphere { radius } => Vec3::splat(radius * 2.0),
        Collider::Capsule {
            radius,
            half_height,
        } => Vec3::new(radius, radius, radius + half_height) * 2.0,
        Collider::Ground => Vec3::ONE,
    }
}

/// Column `k` of the second derivative of a ball joint's constraint row `k`
/// with respect to rotation (the demo's `geometricStiffnessBallSocket`).
fn ball_socket(k: usize, v: Vec3) -> Mat3A {
    let mut m = Mat3A::from_diagonal(Vec3::splat(-v[k]));
    let v = Vec3A::from(v);
    match k {
        0 => m.x_axis += v,
        1 => m.y_axis += v,
        _ => m.z_axis += v,
    }
    m
}

/// Sec. 3.5: replace an indefinite geometric stiffness by the norms of its columns.
fn diagonalize(m: Mat3A) -> Mat3A {
    Mat3A::from_diagonal(Vec3::new(
        m.x_axis.length(),
        m.y_axis.length(),
        m.z_axis.length(),
    ))
}

/// `f` and `k` scaled down so `f` is at most `cap`: the secant of a capped force.
fn capped(f: Vec3A, k: Vec3A, cap: f32) -> (Vec3A, Vec3A) {
    let length = f.length();
    if length > cap {
        let s = cap / length;
        (f * s, k * s)
    } else {
        (f, k)
    }
}

/// The rotation vector of `q`, exact at large angles.
fn log(q: Quat) -> Vec3 {
    let q = if q.w < 0.0 { -q } else { q };
    let v = Vec3::new(q.x, q.y, q.z);
    let s = v.length();
    if s < 1.0e-6 {
        v * 2.0
    } else {
        v * (2.0 * s.atan2(q.w) / s)
    }
}

fn skew(v: Vec3) -> Mat3A {
    Mat3A::from_cols(
        Vec3A::new(0.0, v.z, -v.y),
        Vec3A::new(-v.z, 0.0, v.x),
        Vec3A::new(v.y, -v.x, 0.0),
    )
}

impl Link {
    pub fn new(kind: Kind, anchor_a: Vec3, anchor_b: Vec3) -> Self {
        Self {
            kind,
            a: None,
            b: 0,
            anchor_a,
            anchor_b,
            broken: false,
            first_is_a: false,
            scale: 0.0,
            arm: 0.0,
            c0_linear: Vec3::ZERO,
            c0_angular: Vec3::ZERO,
            arm0_a: Vec3::ZERO,
            arm0_b: Vec3::ZERO,
            start_force: 0.0,
            penalty_linear: Vec3::ZERO,
            penalty_angular: Vec3::ZERO,
            lambda_linear: Vec3::ZERO,
            lambda_angular: Vec3::ZERO,
            strength: [Strength::default(); 2],
            strain: [0.0; 2],
            load: [0.0; 2],
            stage: JointStage::Intact,
            limit_min: Vec3::splat(-FREE_ANGLE),
            limit_max: Vec3::splat(FREE_ANGLE),
            limited: false,
            limit0: Vec3::ZERO,
            penalty_limit: Vec3::ZERO,
            lambda_limit: Vec3::ZERO,
            limit_stiffness: f32::INFINITY,
            muscle: None,
            muscle_limit: 0.0,
            stiffening: f32::INFINITY,
            soft_limit: 0.0,
        }
    }

    /// Set the rotation range; infinite bounds leave an axis free.
    pub fn set_limits(&mut self, min: Vec3, max: Vec3) {
        self.limit_min = min.max(Vec3::splat(-FREE_ANGLE));
        self.limit_max = max.min(Vec3::splat(FREE_ANGLE));
        self.limited =
            self.limit_min.max_element() > -FREE_ANGLE || self.limit_max.min_element() < FREE_ANGLE;
    }

    /// The largest force a row can carry now. It rises from the yield load to
    /// the break load as the row strains, a material's hardening.
    fn cap(&self, row: usize) -> f32 {
        let s = self.strength[row];
        if s.yield_load >= s.break_load {
            f32::INFINITY
        } else if s.break_load.is_infinite() {
            s.yield_load
        } else {
            let hardened = (self.strain[row] / s.ductility).min(1.0);
            s.yield_load + (s.break_load - s.yield_load) * hardened
        }
    }

    /// Bodies whose motion this link affects this step.
    pub fn solved(&self, bodies: &[Body]) -> bool {
        !self.broken && (self.a.is_some_and(|a| bodies[a].solvable()) || bodies[self.b].solvable())
    }

    fn ends(&self, a: Option<&Body>, b: &Body) -> Ends {
        let (point_a, arm_a, rotation_a) = match a {
            Some(a) => {
                let arm = a.rotation * self.anchor_a;
                (a.position + arm, arm, a.rotation)
            }
            None => (self.anchor_a, Vec3::ZERO, Quat::IDENTITY),
        };
        let arm_b = b.rotation * self.anchor_b;
        Ends {
            point_a,
            point_b: b.position + arm_b,
            arm_a,
            arm_b,
            rotation_a,
            rotation_b: b.rotation,
        }
    }

    fn angle(&self, e: &Ends, rest: Quat) -> Vec3 {
        quat_sub(e.rotation_a * rest, e.rotation_b) * self.arm
    }

    /// B's rotation from its rest pose, per axis of the frame `rotation_a * rest`.
    fn turned(e: &Ends, rest: Quat) -> Vec3 {
        log((e.rotation_a * rest).conjugate() * e.rotation_b)
    }

    /// Eq. 18's stabilization of the stops, which compliant ones do not need.
    fn limit_shift(&self, alpha: f32) -> Vec3 {
        if self.limit_stiffness.is_infinite() {
            self.limit0 * alpha
        } else {
            Vec3::ZERO
        }
    }

    /// The muscle's torque on B around `axis` and its stiffness, saturating at
    /// what the muscle can make.
    fn muscle_force(&self, muscle: &Muscle, turned: Vec3) -> (f32, f32) {
        let error = turned.dot(muscle.axis) - muscle.target;
        let torque = muscle.stiffness * error;
        if torque.abs() > self.muscle_limit {
            let secant = self.muscle_limit / error.abs();
            (self.muscle_limit.copysign(error), secant)
        } else {
            (torque, muscle.stiffness)
        }
    }

    /// Limit rows' forces, in the frame of `turned`, and their stiffness: an
    /// augmented Lagrangian inequality per bound, like a contact's normal.
    fn limit_force(&self, turned: Vec3, alpha: f32) -> (Vec3A, Vec3A) {
        let shift = self.limit_shift(alpha);
        let k = self.penalty_limit;
        let lambda = self.lambda_limit;
        let over = (turned - self.limit_max - shift) * self.arm;
        let under = (turned - self.limit_min - shift) * self.arm;
        let upper = (k * over + lambda.max(Vec3::ZERO)).max(Vec3::ZERO);
        let lower = (k * under + lambda.min(Vec3::ZERO)).min(Vec3::ZERO);
        let f = upper + lower;
        let active = Vec3::select(f.cmpne(Vec3::ZERO), k, Vec3::ZERO);
        capped(f.into(), active.into(), self.cap(ANGULAR) / self.arm)
    }

    /// Error at the step's start and warm start (Eq. 19), with penalties
    /// clamped to the material stiffness (Eq. 16).
    pub fn initialize(&mut self, bodies: &[Body], settings: &PhysicsSettings, dt: f32) {
        let (a, b) = (self.a.map(|i| &bodies[i]), &bodies[self.b]);
        self.scale = a.map_or(0.0, |a| a.mass).max(b.mass).max(f32::MIN_POSITIVE);
        self.arm = (a.map_or(Vec3::ZERO, |a| size(a.shape)) + size(b.shape)).length_squared();
        let e = self.ends(a, b);
        (self.arm0_a, self.arm0_b) = (e.arm_a, e.arm_b);
        let spin = a.map_or(0.0, |a| a.angular_velocity.length());
        let turn = spin.max(b.angular_velocity.length()) * dt / START_FORCE_TURN;
        self.start_force = (1.0 - turn * turn).max(0.0);
        let gamma = settings.gamma;
        let (min, max) = (PENALTY_MIN * self.scale, PENALTY_MAX * self.scale);
        let warm = |p: Vec3, limit: f32| {
            (p * gamma)
                .clamp(Vec3::splat(min), Vec3::splat(max))
                .min(Vec3::splat(limit))
        };
        match self.kind {
            Kind::Joint {
                linear,
                angular,
                rest,
                ..
            } => {
                self.c0_linear = e.point_a - e.point_b;
                self.c0_angular = self.angle(&e, rest);
                let stretch = self.c0_linear.length() / self.stiffening;
                self.soft_limit = linear * (1.0 + stretch * stretch);
                self.penalty_linear = warm(self.penalty_linear, self.soft_limit);
                self.penalty_angular = warm(self.penalty_angular, angular);
                if self.limited {
                    let turned = Self::turned(&e, rest);
                    self.limit0 = (turned - self.limit_max).max(Vec3::ZERO)
                        + (turned - self.limit_min).min(Vec3::ZERO);
                    self.penalty_limit = if self.limit_stiffness.is_infinite() {
                        warm(self.penalty_limit, f32::INFINITY)
                    } else {
                        Vec3::splat(self.limit_stiffness / (self.arm * self.arm))
                    };
                    self.lambda_limit *= gamma;
                }
                if let Some(muscle) = self.muscle {
                    let turned = Self::turned(&e, rest).dot(muscle.axis);
                    let world = (e.rotation_a * rest) * muscle.axis;
                    let spin = a.map_or(Vec3::ZERO, |a| a.angular_velocity);
                    let speed = (b.angular_velocity - spin).dot(world);
                    // Shortening is moving the way the muscle pulls.
                    let shortening = speed * (muscle.target - turned).signum();
                    self.muscle_limit = muscle.torque_limit(turned, shortening);
                }
                // Unlike contacts, joints keep the decay: without it a swinging
                // chain holds forces from where it just was.
                self.lambda_linear *= gamma;
                self.lambda_angular *= gamma;
            }
            Kind::Spring { stiffness, .. } => {
                self.penalty_linear = warm(self.penalty_linear, stiffness);
            }
        }
    }

    /// Linear joint error, with the stabilization of Eq. 18 for hard joints.
    fn linear_error(&self, e: &Ends, hard: bool, alpha: f32) -> Vec3 {
        let c = e.point_a - e.point_b;
        if hard { c - self.c0_linear * alpha } else { c }
    }

    fn angular_error(&self, e: &Ends, rest: Quat, hard: bool, alpha: f32) -> Vec3 {
        let c = self.angle(e, rest);
        if hard { c - self.c0_angular * alpha } else { c }
    }

    /// Accumulate this link's force and Hessian approximation for one of its bodies.
    pub fn stamp(&self, a: Option<&Body>, b: &Body, is_a: bool, alpha: f32, system: &mut System) {
        let e = self.ends(a, b);
        let sign = if is_a { 1.0 } else { -1.0 };
        match self.kind {
            Kind::Joint {
                linear,
                angular,
                rest,
                ..
            } => {
                if self.penalty_linear != Vec3::ZERO {
                    let k = Vec3A::from(self.penalty_linear);
                    let c = self.linear_error(&e, linear.is_infinite(), alpha);
                    let f = k * Vec3A::from(c) + Vec3A::from(self.lambda_linear);
                    let (f, k) = capped(f, k, self.cap(LINEAR));
                    // Linear Jacobian ±I; angular skew(-r_a) for A, skew(r_b) for B.
                    let j = if is_a { skew(-e.arm_a) } else { skew(e.arm_b) };
                    let (jt, kj) = (j.transpose(), scale_rows(j, k));
                    system.lin += Mat3A::from_diagonal(k.into());
                    system.ang += jt * kj;
                    system.cross += jt * Mat3A::from_diagonal(k.into()) * sign;
                    system.rhs_lin += f * sign;
                    // A hard joint pushes along its step-start direction, as
                    // SHAKE does: that conserves energy, where the step-end
                    // direction is backward Euler and damps a swing by
                    // exp(-ω²ht/2). Unconverged solves gain energy this way
                    // once bodies turn fast, so those fall back and are damped.
                    let w = if linear.is_infinite() {
                        self.start_force
                    } else {
                        0.0
                    };
                    let arm = if is_a {
                        -e.arm_a.lerp(self.arm0_a, w)
                    } else {
                        e.arm_b.lerp(self.arm0_b, w)
                    };
                    system.rhs_ang += skew(arm).transpose() * f;
                    // Diagonal geometric stiffness (Sec. 3.5, Eq. 17).
                    let r = if is_a { e.arm_a } else { -e.arm_b };
                    let g =
                        ball_socket(0, r) * f.x + ball_socket(1, r) * f.y + ball_socket(2, r) * f.z;
                    system.ang += diagonalize(g);
                }
                if self.penalty_angular != Vec3::ZERO {
                    let k = Vec3A::from(self.penalty_angular);
                    let c = self.angular_error(&e, rest, angular.is_infinite(), alpha);
                    let f = k * Vec3A::from(c) + Vec3A::from(self.lambda_angular);
                    let (f, k) = capped(f, k, self.cap(ANGULAR) / self.arm);
                    system.ang += Mat3A::from_diagonal((k * self.arm * self.arm).into());
                    system.rhs_ang += f * (self.arm * sign);
                }
                if self.limited {
                    let (f, k) = self.limit_force(Self::turned(&e, rest), alpha);
                    let axes = Mat3A::from_quat(e.rotation_a * rest);
                    // The rows measure B against A: opposite to `angle`'s sign.
                    system.ang += scale_rows(axes.transpose(), k * self.arm * self.arm).transpose()
                        * axes.transpose();
                    system.rhs_ang += axes * f * (-self.arm * sign);
                }
                if let Some(muscle) = &self.muscle {
                    let (torque, k) = self.muscle_force(muscle, Self::turned(&e, rest));
                    let axis = Vec3A::from((e.rotation_a * rest) * muscle.axis);
                    system.ang += outer(axis, axis) * k;
                    system.rhs_ang += axis * (-torque * sign);
                }
            }
            Kind::Spring { rest_length, .. } => {
                let d = e.point_a - e.point_b;
                let length = d.length();
                if length <= 1.0e-6 {
                    return;
                }
                let n = d / length;
                let k = self.penalty_linear.x;
                let f = k * (length - rest_length);
                let (j_lin, j_ang) = if is_a {
                    (n, e.arm_a.cross(n))
                } else {
                    (-n, -e.arm_b.cross(n))
                };
                let (j_lin, j_ang) = (Vec3A::from(j_lin), Vec3A::from(j_ang));
                system.lin += outer(j_lin, j_lin) * k;
                system.ang += outer(j_ang, j_ang) * k;
                system.cross += outer(j_ang, j_lin) * k;
                system.rhs_lin += j_lin * f;
                system.rhs_ang += j_ang * f;
            }
        }
    }

    /// Update dual variables of hard rows (Eq. 11) and ramp penalties: without
    /// bound for hard rows (Eq. 12), up to the stiffness for soft ones (Eq. 16).
    pub fn dual(&mut self, a: Option<&Body>, b: &Body, settings: &PhysicsSettings) {
        let e = self.ends(a, b);
        let (alpha, max) = (settings.alpha, PENALTY_MAX * self.scale);
        let bound = Vec3::splat(LAMBDA_MAX * self.scale);
        let ramp = |p: Vec3, c: Vec3, beta: f32, limit: f32| {
            (p + c.abs() * (beta * self.scale)).min(Vec3::splat(limit.min(max)))
        };
        match self.kind {
            Kind::Joint {
                linear,
                angular,
                rest,
            } => {
                if self.penalty_linear != Vec3::ZERO {
                    let c = self.linear_error(&e, linear.is_infinite(), alpha);
                    if linear.is_infinite() {
                        let f = self.penalty_linear * c + self.lambda_linear;
                        let f = Vec3::from(capped(f.into(), Vec3A::ZERO, self.cap(LINEAR)).0);
                        self.lambda_linear = f.clamp(-bound, bound);
                    }
                    self.penalty_linear =
                        ramp(self.penalty_linear, c, settings.beta, self.soft_limit);
                }
                if self.penalty_angular != Vec3::ZERO {
                    let c = self.angular_error(&e, rest, angular.is_infinite(), alpha);
                    if angular.is_infinite() {
                        let f = self.penalty_angular * c + self.lambda_angular;
                        let cap = self.cap(ANGULAR) / self.arm;
                        let f = Vec3::from(capped(f.into(), Vec3A::ZERO, cap).0);
                        self.lambda_angular = f.clamp(-bound, bound);
                    }
                    self.penalty_angular =
                        ramp(self.penalty_angular, c, settings.beta_angular, angular);
                }
                if self.limited {
                    let turned = Self::turned(&e, rest);
                    let (f, k) = self.limit_force(turned, alpha);
                    let shift = self.limit_shift(alpha);
                    let over = (turned - self.limit_max - shift).max(Vec3::ZERO)
                        + (turned - self.limit_min - shift).min(Vec3::ZERO);
                    let c = Vec3::select(Vec3::from(k).cmpne(Vec3::ZERO), over, Vec3::ZERO);
                    if self.limit_stiffness.is_infinite() {
                        self.lambda_limit = Vec3::from(f).clamp(-bound, bound);
                        self.penalty_limit = ramp(
                            self.penalty_limit,
                            c * self.arm,
                            settings.beta_angular,
                            f32::INFINITY,
                        );
                    }
                }
            }
            Kind::Spring {
                stiffness,
                rest_length,
            } => {
                let c = (e.point_a - e.point_b).length() - rest_length;
                self.penalty_linear = ramp(
                    self.penalty_linear,
                    Vec3::splat(c),
                    settings.beta,
                    stiffness,
                );
            }
        }
    }
}

impl Link {
    /// Once per step, on the converged forces: average the load, let an
    /// overloaded row flow plastically, and report what changed.
    pub fn settle(&mut self, a: Option<&Body>, b: &Body, dt: f32) -> Option<JointStage> {
        let Kind::Joint {
            linear,
            angular,
            rest,
        } = self.kind
        else {
            return None;
        };
        let e = self.ends(a, b);
        let c = e.point_a - e.point_b;
        let hard = linear.is_infinite();
        let force = if hard {
            self.lambda_linear
        } else {
            self.penalty_linear * c
        };
        let cap = self.cap(LINEAR);
        let flowed = force.length() >= cap * LOAD_MARGIN;
        if flowed {
            // Soft rows keep their elastic stretch: only the excess is permanent.
            let slide = if hard {
                c
            } else {
                c * (1.0 - cap / force.length())
            };
            self.anchor_b += e.rotation_b.conjugate() * slide;
            self.strain[LINEAR] += slide.length();
        }

        let c = self.angle(&e, rest);
        let weld = if angular.is_infinite() {
            self.lambda_angular
        } else {
            self.penalty_angular * c
        };
        let cap = self.cap(ANGULAR);
        let mut torque = weld.length() * self.arm;
        let mut turned_flow = false;
        if self.penalty_angular != Vec3::ZERO && torque >= cap * LOAD_MARGIN {
            let moved = if angular.is_infinite() {
                1.0
            } else {
                1.0 - cap / torque
            };
            let target = e.rotation_a.conjugate() * e.rotation_b;
            if let Kind::Joint { rest, .. } = &mut self.kind {
                *rest = rest.slerp(target, moved).normalize();
            }
            self.strain[ANGULAR] += moved * c.length() / self.arm;
            turned_flow = true;
        }
        if self.limited {
            let stop = if self.limit_stiffness.is_infinite() {
                self.lambda_limit.length()
            } else {
                self.limit_force(Self::turned(&e, rest), 0.0).0.length()
            } * self.arm;
            torque = torque.max(stop);
            if stop >= cap * LOAD_MARGIN {
                let turned = Self::turned(&e, rest);
                let (min, max) = (self.limit_min.min(turned), self.limit_max.max(turned));
                self.strain[ANGULAR] +=
                    (self.limit_min - min).length() + (max - self.limit_max).length();
                (self.limit_min, self.limit_max) = (min, max);
                turned_flow = true;
            }
        }

        let blend = 1.0 - (-dt / LOAD_TIME).exp();
        self.load[LINEAR] += (force.length() - self.load[LINEAR]) * blend;
        self.load[ANGULAR] += (torque - self.load[ANGULAR]) * blend;
        let ruptured = |row: usize| {
            let s = self.strength[row];
            s.break_load.is_finite()
                && (self.load[row] >= s.break_load
                    || (s.yield_load < s.break_load && self.strain[row] >= s.ductility))
        };
        if ruptured(LINEAR) || ruptured(ANGULAR) {
            self.broken = true;
            self.stage = JointStage::Broken;
            self.penalty_linear = Vec3::ZERO;
            self.penalty_angular = Vec3::ZERO;
            self.penalty_limit = Vec3::ZERO;
            self.lambda_linear = Vec3::ZERO;
            self.lambda_angular = Vec3::ZERO;
            self.lambda_limit = Vec3::ZERO;
            self.limited = false;
            return Some(JointStage::Broken);
        }
        if (flowed || turned_flow) && self.stage == JointStage::Intact {
            self.stage = JointStage::Yielded;
            return Some(JointStage::Yielded);
        }
        None
    }
}

/// `m` with row `k` scaled by `s[k]`.
fn scale_rows(m: Mat3A, s: Vec3A) -> Mat3A {
    Mat3A::from_cols(m.x_axis * s, m.y_axis * s, m.z_axis * s)
}

/// `a bᵀ`.
fn outer(a: Vec3A, b: Vec3A) -> Mat3A {
    Mat3A::from_cols(a * b.x, a * b.y, a * b.z)
}
