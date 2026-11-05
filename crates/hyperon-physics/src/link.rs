//! Joints and springs: constraints between two bodies, or a body and a world
//! point. Joints are hard (augmented Lagrangian, Sec. 3.1) or soft with a
//! finite stiffness ramped up to (Eq. 16); springs are soft. Joints evaluate
//! exactly each iteration with the diagonal geometric stiffness of Sec. 3.5,
//! following the AVBD 3D demo; contacts instead use the step-start Taylor
//! expansion (Sec. 4).

use crate::solver::{Body, System, quat_sub};
use crate::{Collider, PhysicsSettings};
use glam::{Mat3A, Quat, Vec3, Vec3A};

const PENALTY_MIN: f32 = 1.0;
const PENALTY_MAX: f32 = 1.0e10;
const LAMBDA_MAX: f32 = 1.0e10;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Kind {
    Joint {
        linear: f32,
        angular: f32,
        fracture: f32,
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
    penalty_linear: Vec3,
    penalty_angular: Vec3,
    lambda_linear: Vec3,
    lambda_angular: Vec3,
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
            penalty_linear: Vec3::ZERO,
            penalty_angular: Vec3::ZERO,
            lambda_linear: Vec3::ZERO,
            lambda_angular: Vec3::ZERO,
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

    /// Error at the step's start and warm start (Eq. 19), with penalties
    /// clamped to the material stiffness (Eq. 16).
    pub fn initialize(&mut self, bodies: &[Body], settings: &PhysicsSettings) {
        let (a, b) = (self.a.map(|i| &bodies[i]), &bodies[self.b]);
        self.scale = a.map_or(0.0, |a| a.mass).max(b.mass).max(f32::MIN_POSITIVE);
        self.arm = (a.map_or(Vec3::ZERO, |a| size(a.shape)) + size(b.shape)).length_squared();
        let e = self.ends(a, b);
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
                self.penalty_linear = warm(self.penalty_linear, linear);
                self.penalty_angular = warm(self.penalty_angular, angular);
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
                    // Linear Jacobian ±I; angular skew(-r_a) for A, skew(r_b) for B.
                    let j = if is_a { skew(-e.arm_a) } else { skew(e.arm_b) };
                    let (jt, kj) = (j.transpose(), scale_rows(j, k));
                    system.lin += Mat3A::from_diagonal(k.into());
                    system.ang += jt * kj;
                    system.cross += jt * Mat3A::from_diagonal(k.into()) * sign;
                    system.rhs_lin += f * sign;
                    system.rhs_ang += jt * f;
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
                    system.ang += Mat3A::from_diagonal((k * self.arm * self.arm).into());
                    system.rhs_ang += f * (self.arm * sign);
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
                fracture,
                rest,
            } => {
                if self.penalty_linear != Vec3::ZERO {
                    let c = self.linear_error(&e, linear.is_infinite(), alpha);
                    if linear.is_infinite() {
                        let f = self.penalty_linear * c + self.lambda_linear;
                        self.lambda_linear = f.clamp(-bound, bound);
                    }
                    self.penalty_linear = ramp(self.penalty_linear, c, settings.beta, linear);
                }
                if self.penalty_angular != Vec3::ZERO {
                    let c = self.angular_error(&e, rest, angular.is_infinite(), alpha);
                    if angular.is_infinite() {
                        let f = self.penalty_angular * c + self.lambda_angular;
                        self.lambda_angular = f.clamp(-bound, bound);
                    }
                    self.penalty_angular =
                        ramp(self.penalty_angular, c, settings.beta_angular, angular);
                }
                // Fracture compares torque: the angular rows' force times their arm.
                if self.lambda_angular.length() * self.arm > fracture {
                    self.broken = true;
                    self.penalty_linear = Vec3::ZERO;
                    self.penalty_angular = Vec3::ZERO;
                    self.lambda_linear = Vec3::ZERO;
                    self.lambda_angular = Vec3::ZERO;
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

/// `m` with row `k` scaled by `s[k]`.
fn scale_rows(m: Mat3A, s: Vec3A) -> Mat3A {
    Mat3A::from_cols(m.x_axis * s, m.y_axis * s, m.z_axis * s)
}

/// `a bᵀ`.
fn outer(a: Vec3A, b: Vec3A) -> Mat3A {
    Mat3A::from_cols(a * b.x, a * b.y, a * b.z)
}
