//! Deformable bodies: cloth and solids.
//!
//! Cloth is a triangle mesh of StVK membrane elements with quadratic bending
//! across its edges (Bergou et al. 2006), stepped implicitly by Newton's method
//! and conjugate gradients. The mesh does not change what the material does. It
//! tears where its largest principal stress passes its strength, across that
//! stress.
//!
//! Solids are lattices solved by shape matching (Müller et al. 2005): each
//! overlapping region of particles is pulled toward the best-fit rotation of its
//! rest shape. That stays stiff in bending, which beams cannot do in a few
//! sweeps, and is stable at any stiffness. Plasticity updates the rest shape,
//! so past its yield strain a solid stays dented. A car's shell is a lattice
//! pinned to a rigid body.
//!
//! Soft bodies step after rigid bodies, seeing them at their new poses, and
//! give them exactly the momentum they took, through
//! [`crate::ExternalImpulse`], one step late.

use crate::{Collider, Strength};
use glam::{Mat3, Quat, Vec2, Vec3};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use verity_core::EntityHandle;

/// Sweeps of every particle per step.
const ITERATIONS: usize = 8;
/// Grid cells from a region's center to its edge: wider is stiffer in bending.
const REGION_REACH: i32 = 2;
/// Grid cells between region centers. Below `REGION_REACH * 2` regions overlap.
const REGION_STRIDE: u32 = 2;
/// Particles that resist a contact together: about one region.
const CONTACT_SPREAD: f32 = 50.0;
/// Share of an overlap with a rigid body resolved each step.
const SEPARATION: f32 = 0.2;
/// Newton steps per time step, and the conjugate gradient work in each.
const NEWTON_STEPS: usize = 1;
const CG_ITERATIONS: usize = 20;
const CG_TOLERANCE: f32 = 1.0e-2;
/// A sheet's contact springs, as a multiple of its stiffness.
const CONTACT_STIFFNESS: f32 = 1.0;
/// One element in this many may tear in a step.
const TEARS_PER_STEP: usize = 100;
/// Least work worth a thread.
const PARALLEL_CHUNK: usize = 512;
/// Seconds of strain rate damped in a sheet's elements.
const ELEMENT_DAMPING: f32 = 0.002;
/// Seconds over which a sheet's stress is averaged, as for joints.
const LOAD_TIME: f32 = 0.03;
/// Most samples along a particle's path through a step.
const MAX_SWEEP_SAMPLES: usize = 16;
/// Particles closer than this many collision radii at rest never collide.
const NEIGHBOUR_REACH: f32 = 1.5;
/// A body this many collision radii across can pass between a sheet's particles.
const SMALL_BODY: f32 = 3.0;
/// A pin's stiffness, as a multiple of its particle's inertial term.
const PIN_STIFFNESS: f32 = 50.0;
/// Speed (m/s) below which a soft body may sleep, and for how long (s).
const SLEEP_SPEED: f32 = 0.02;
const TIME_TO_SLEEP: f32 = 0.5;
/// Speed (m/s) a contact must give a sleeping rigid body to wake it.
const WAKE_SPEED: f32 = 0.05;
const NONE: u32 = u32::MAX;

/// What a soft body is made of. Units follow its dimension: a sheet's are per
/// metre of width (kg/m², N/m), a solid's per square metre (kg/m³, N/m²), so
/// changing the resolution keeps its behaviour roughly the same.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoftMaterial {
    pub density: f32,
    /// Resistance to stretching: Young's modulus, times thickness for a sheet.
    pub stiffness: f32,
    /// A sheet's lateral contraction when stretched: Poisson's ratio.
    pub poisson: f32,
    /// Share of `stiffness` that resists shortening: little for cloth, which
    /// buckles instead.
    pub compression: f32,
    /// A sheet's resistance to folding (N·m).
    pub bending: f32,
    /// How firmly a solid returns to its shape each sweep: 1 is as rigid as it
    /// can be, 0.1 is jelly.
    pub rigidity: f32,
    /// In the units of `stiffness`, with `ductility` as strain: 0.1 is 10%.
    /// A sheet tears at the stress `break_load`. A solid yields at the strain
    /// `yield_load / stiffness` and cracks after `ductility` of plastic strain,
    /// or at `break_load / stiffness` if brittle.
    pub strength: Strength,
    /// Velocity lost per second.
    pub damping: f32,
    pub friction: f32,
    /// Distance kept from what it touches (m).
    pub thickness: f32,
}

impl SoftMaterial {
    /// Canvas-like cloth, which tears at 15 kN per metre of width, near 30% strain.
    pub fn cloth() -> Self {
        Self {
            density: 0.3,
            stiffness: 50_000.0,
            poisson: 0.3,
            compression: 0.05,
            bending: 0.002,
            rigidity: 0.0,
            strength: Strength::brittle(15_000.0),
            damping: 1.0,
            friction: 0.5,
            thickness: 0.02,
        }
    }

    /// Mild steel in a solid. It holds its shape until 3% strain, then dents.
    pub fn sheet_metal() -> Self {
        Self {
            density: 7800.0,
            stiffness: 2.0e9,
            poisson: 0.3,
            compression: 1.0,
            bending: 0.0,
            rigidity: 1.0,
            strength: Strength {
                yield_load: 6.0e7,
                break_load: 2.4e8,
                ductility: 50.0,
            },
            damping: 2.0,
            friction: 0.5,
            thickness: 0.01,
        }
    }

    /// The hollow body of a car, as a solid. Its yield is its crush strength
    /// (Pa): this crushes a 1500 kg car about 0.55 m against a wall at 56 km/h,
    /// within the 0.49 to 0.65 m of published crash tests.
    pub fn crumple_zone() -> Self {
        Self {
            density: 30.0,
            stiffness: 7.0e6,
            poisson: 0.3,
            compression: 1.0,
            bending: 0.0,
            rigidity: 1.0,
            strength: Strength {
                yield_load: 7.0e4,
                break_load: 2.8e5,
                ductility: 200.0,
            },
            damping: 2.0,
            friction: 0.5,
            thickness: 0.01,
        }
    }

    /// Brittle concrete: stiff, and cracks past 0.5% strain without bending.
    pub fn concrete() -> Self {
        Self {
            density: 2400.0,
            stiffness: 2.0e9,
            poisson: 0.2,
            compression: 1.0,
            bending: 0.0,
            rigidity: 1.0,
            strength: Strength::brittle(1.0e7),
            damping: 3.0,
            friction: 0.7,
            thickness: 0.01,
        }
    }

    /// A rubber solid.
    pub fn rubber() -> Self {
        Self {
            density: 1100.0,
            stiffness: 2.0e5,
            poisson: 0.3,
            compression: 1.0,
            bending: 0.0,
            rigidity: 0.25,
            strength: Strength::default(),
            damping: 1.0,
            friction: 0.9,
            thickness: 0.01,
        }
    }
}

/// One triangle of a sheet: a StVK membrane, in its rest shape.
#[derive(Debug, Clone, Copy)]
struct Element {
    /// Each corner's gradient of the deformation, from the rest shape.
    gradient: [Vec2; 3],
    /// Corners at rest, in the triangle's plane.
    rest: [Vec2; 3],
    area: f32,
    /// Largest principal stress, averaged (N/m).
    load: f32,
}

impl Element {
    fn new(points: [Vec3; 3]) -> Self {
        let (e1, e2) = (points[1] - points[0], points[2] - points[0]);
        let u = e1.normalize_or_zero();
        let v = (e2 - u * u.dot(e2)).normalize_or_zero();
        let (p1, p2) = (Vec2::new(e1.length(), 0.0), Vec2::new(u.dot(e2), v.dot(e2)));
        let determinant = p1.x * p2.y - p1.y * p2.x;
        let inverse = 1.0 / determinant;
        // Rows of the inverse of the matrix with columns `p1` and `p2`.
        let g1 = Vec2::new(p2.y, -p2.x) * inverse;
        let g2 = Vec2::new(-p1.y, p1.x) * inverse;
        Self {
            gradient: [-g1 - g2, g1, g2],
            rest: [Vec2::ZERO, p1, p2],
            area: determinant.abs() * 0.5,
            load: 0.0,
        }
    }

    /// The columns of the deformation gradient.
    fn deformation(&self, x: [Vec3; 3]) -> (Vec3, Vec3) {
        let g = &self.gradient;
        (
            x[0] * g[0].x + x[1] * g[1].x + x[2] * g[2].x,
            x[0] * g[0].y + x[1] * g[1].y + x[2] * g[2].y,
        )
    }
}

/// The second Piola-Kirchhoff stress for deformation columns `(a, b)`, as
/// `[xx, yy, xy]`, and how firm it is: shortening counts for `compression` of
/// itself, as cloth buckles, so an element with no tension at all is soft.
fn stress(a: Vec3, b: Vec3, material: &Membrane) -> ([f32; 3], f32) {
    let (exx, eyy, exy) = (
        0.5 * (a.dot(a) - 1.0),
        0.5 * (b.dot(b) - 1.0),
        0.5 * a.dot(b),
    );
    let (mean, half) = (0.5 * (exx + eyy), 0.5 * (exx - eyy));
    let radius = (half * half + exy * exy).sqrt();
    let (high, low) = (mean + radius, mean - radius);
    let soften = |e: f32| if e < 0.0 { e * material.compression } else { e };
    let (high_soft, low_soft) = (soften(high), soften(low));
    // The strain with its negative principal part softened, same axes.
    let (mean, ratio) = (
        0.5 * (high_soft + low_soft),
        if radius > 1.0e-12 {
            0.5 * (high_soft - low_soft) / radius
        } else {
            0.0
        },
    );
    let (exx, eyy, exy) = (mean + half * ratio, mean - half * ratio, exy * ratio);
    let trace = material.lame * (exx + eyy);
    let firm = if high > 0.0 {
        1.0
    } else {
        material.compression
    };
    (
        [
            2.0 * material.shear * exx + trace,
            2.0 * material.shear * eyy + trace,
            2.0 * material.shear * exy,
        ],
        firm,
    )
}

/// What an element's state was at the start of a Newton step.
#[derive(Debug, Clone, Copy, Default)]
struct ElementState {
    a: Vec3,
    b: Vec3,
    stress: [f32; 3],
    /// `stress` without compression, which must not make the step indefinite.
    tension: [f32; 3],
    firm: f32,
}

/// `[xx, yy, xy]` with its negative principal part removed.
fn tension_only(s: [f32; 3]) -> [f32; 3] {
    let (mean, half) = (0.5 * (s[0] + s[1]), 0.5 * (s[0] - s[1]));
    let radius = (half * half + s[2] * s[2]).sqrt();
    let (high, low) = ((mean + radius).max(0.0), (mean - radius).max(0.0));
    let (mean, ratio) = (
        0.5 * (high + low),
        if radius > 1.0e-12 {
            0.5 * (high - low) / radius
        } else {
            0.0
        },
    );
    [mean + half * ratio, mean - half * ratio, s[2] * ratio]
}

/// A sheet's elastic constants (N/m).
#[derive(Debug, Clone, Copy)]
struct Membrane {
    shear: f32,
    lame: f32,
    compression: f32,
}

/// Bending across one edge, by the quadratic model of Bergou et al. 2006: its
/// energy is half the squared length of `Σ weights[i] x[i]`, whatever the pose.
#[derive(Debug, Clone, Copy)]
struct Hinge {
    particles: [u32; 4],
    weights: [f32; 4],
}

/// How a solid was laid out, to tie other points to it.
#[derive(Debug, Clone)]
struct Grid {
    center: Vec3,
    rotation: Quat,
    half_extents: Vec3,
    step: Vec3,
    cells: [u32; 3],
    /// Particle by grid point.
    index: HashMap<u32, u32>,
}

/// A patch of a solid that keeps its shape.
#[derive(Debug, Clone)]
struct Region {
    particles: Vec<u32>,
    /// Rest offsets from the region's center of mass; plasticity changes them.
    rest: Vec<Vec3>,
    /// Inverse of the rest shape's second moment.
    moment: Mat3,
    rotation: Quat,
    mass: f32,
    strain: f32,
    /// Its smallest half extent at rest (m).
    thickness: f32,
}

impl Region {
    fn new(particles: Vec<u32>, positions: &[Vec3], masses: &[f32]) -> Self {
        let mass: f32 = particles.iter().map(|&i| masses[i as usize]).sum();
        let center = particles
            .iter()
            .map(|&i| positions[i as usize] * masses[i as usize])
            .sum::<Vec3>()
            / mass;
        let rest: Vec<Vec3> = (particles.iter())
            .map(|&i| positions[i as usize] - center)
            .collect();
        let reach = rest.iter().fold(Vec3::ZERO, |m, q| m.max(q.abs()));
        let mut region = Self {
            particles,
            rest,
            moment: Mat3::IDENTITY,
            rotation: Quat::IDENTITY,
            mass,
            strain: 0.0,
            thickness: reach.min_element().max(1.0e-4),
        };
        region.update_moment(masses);
        region
    }

    /// A region of rest offsets taken from another, recentred.
    fn from_rest(
        particles: Vec<u32>,
        rest: Vec<Vec3>,
        rotation: Quat,
        strain: f32,
        masses: &[f32],
    ) -> Self {
        let mass: f32 = particles.iter().map(|&i| masses[i as usize]).sum();
        let center = (particles.iter().zip(&rest))
            .map(|(&i, &q)| q * masses[i as usize])
            .sum::<Vec3>()
            / mass;
        let rest: Vec<Vec3> = rest.iter().map(|q| *q - center).collect();
        let reach = rest.iter().fold(Vec3::ZERO, |m, q| m.max(q.abs()));
        let mut region = Self {
            particles,
            rest,
            moment: Mat3::IDENTITY,
            rotation,
            mass,
            strain,
            thickness: reach.min_element().max(1.0e-4),
        };
        region.update_moment(masses);
        region
    }

    fn update_moment(&mut self, masses: &[f32]) {
        let mut moment = Mat3::ZERO;
        for (&i, &q) in self.particles.iter().zip(&self.rest) {
            moment += outer(q, q) * masses[i as usize];
        }
        // A flat region has no extent along one axis.
        let scale = moment.x_axis.x + moment.y_axis.y + moment.z_axis.z;
        self.moment = (moment + Mat3::from_diagonal(Vec3::splat(1.0e-4 * scale))).inverse();
    }

    /// The best linear map from the rest shape to `positions`, about its center.
    fn deformation(&self, positions: &[Vec3], masses: &[f32]) -> (Vec3, Mat3) {
        let center = (self.particles.iter())
            .map(|&i| positions[i as usize] * masses[i as usize])
            .sum::<Vec3>()
            / self.mass;
        let mut moment = Mat3::ZERO;
        for (&i, &q) in self.particles.iter().zip(&self.rest) {
            let i = i as usize;
            moment += outer(positions[i] - center, q) * masses[i];
        }
        (center, moment * self.moment)
    }

    /// Update `rotation` to the rotation of `deformation`, from where it was
    /// (Müller et al. 2016): warm started, a few sweeps are enough.
    fn track_rotation(&mut self, deformation: Mat3) {
        for _ in 0..4 {
            let r = Mat3::from_quat(self.rotation);
            let (a, b, c) = (deformation.x_axis, deformation.y_axis, deformation.z_axis);
            let spin = r.x_axis.cross(a) + r.y_axis.cross(b) + r.z_axis.cross(c);
            let scale = (r.x_axis.dot(a) + r.y_axis.dot(b) + r.z_axis.dot(c)).abs() + 1.0e-9;
            let w = spin / scale;
            let angle = w.length();
            if angle < 1.0e-7 {
                break;
            }
            self.rotation = (Quat::from_axis_angle(w / angle, angle) * self.rotation).normalize();
        }
    }
}

/// Holds a particle to a world point, or a point of a rigid body.
#[derive(Debug, Clone, Copy)]
struct Pin {
    particle: u32,
    body: Option<EntityHandle>,
    point: Vec3,
    /// Force that tears it out (N).
    strength: f32,
    load: f32,
}

/// A contact plane for one step: `normal · x >= offset`.
#[derive(Debug, Clone, Copy)]
struct Contact {
    particle: u32,
    obstacle: u32,
    normal: Vec3,
    offset: f32,
    /// Force on the particle along the normal in the last sweep (N).
    force: f32,
    /// Total distance it pushed the particle out, and how deep it last was.
    pushed: f32,
    depth: f32,
}

/// A rigid body as a soft body sees it.
pub(crate) struct Obstacle {
    pub shape: Collider,
    pub position: Vec3,
    pub rotation: Quat,
    pub velocity: Vec3,
    pub spin: Vec3,
    pub friction: f32,
    pub mass: f32,
    pub awake: bool,
}

impl Obstacle {
    fn radius(&self) -> f32 {
        match self.shape {
            Collider::Cuboid { half_extents } => half_extents.length(),
            Collider::Sphere { radius } => radius,
        }
    }

    /// The nearest point on the surface and the outward normal there, for a
    /// point within `reach` of it.
    fn nearest(&self, point: Vec3, reach: f32) -> Option<(Vec3, Vec3)> {
        let local = self.rotation.conjugate() * (point - self.position);
        let (surface, normal) = match self.shape {
            Collider::Sphere { radius } => {
                let length = local.length();
                if length > radius + reach {
                    return None;
                }
                let normal = if length > 1.0e-6 {
                    local / length
                } else {
                    Vec3::Z
                };
                (normal * radius, normal)
            }
            Collider::Cuboid { half_extents: h } => {
                let clamped = local.clamp(-h, h);
                let outside = local - clamped;
                let distance = outside.length();
                if distance > reach {
                    return None;
                }
                if distance > 1.0e-6 {
                    (clamped, outside / distance)
                } else {
                    // Inside: leave through the nearest face.
                    let gaps = h - local.abs();
                    let axis = if gaps.x < gaps.y && gaps.x < gaps.z {
                        Vec3::X
                    } else if gaps.y < gaps.z {
                        Vec3::Y
                    } else {
                        Vec3::Z
                    };
                    let sign = local.dot(axis).signum();
                    let normal = axis * if sign == 0.0 { 1.0 } else { sign };
                    (local + normal * gaps.dot(axis), normal)
                }
            }
        };
        Some((
            self.position + self.rotation * surface,
            self.rotation * normal,
        ))
    }

    /// Where a moving point padded by `pad` first meets this body, and the
    /// outward normal there.
    fn sweep(&self, from: Vec3, to: Vec3, pad: f32) -> Option<(Vec3, Vec3)> {
        let a = self.rotation.conjugate() * (from - self.position);
        let d = self.rotation.conjugate() * (to - from);
        let (t, normal) = match self.shape {
            Collider::Sphere { radius } => {
                let r = radius + pad;
                let (qa, qb, qc) = (d.dot(d), a.dot(d), a.dot(a) - r * r);
                let discriminant = qb * qb - qa * qc;
                if qa < 1.0e-12 || discriminant < 0.0 {
                    return None;
                }
                let t = ((-qb - discriminant.sqrt()) / qa).max(0.0);
                (t, (a + d * t).normalize_or_zero())
            }
            Collider::Cuboid { half_extents } => {
                let h = half_extents + pad;
                let (mut enter, mut leave, mut normal) = (f32::MIN, f32::MAX, Vec3::ZERO);
                for axis in 0..3 {
                    if d[axis].abs() < 1.0e-9 {
                        if a[axis].abs() > h[axis] {
                            return None;
                        }
                        continue;
                    }
                    let (t0, t1) = (
                        (-h[axis] - a[axis]) / d[axis],
                        (h[axis] - a[axis]) / d[axis],
                    );
                    let (near, far) = (t0.min(t1), t0.max(t1));
                    if near > enter {
                        enter = near;
                        normal = Vec3::ZERO;
                        normal[axis] = -d[axis].signum();
                    }
                    leave = leave.min(far);
                }
                if enter > leave || leave < 0.0 {
                    return None;
                }
                // Starting inside the padding is already touching.
                (enter.max(0.0), normal)
            }
        };
        if !(0.0..=1.0).contains(&t) {
            return None;
        }
        let surface = a + d * t - normal * pad;
        Some((
            self.position + self.rotation * surface,
            self.rotation * normal,
        ))
    }

    /// Its thinnest half extent.
    fn narrowest(&self) -> f32 {
        match self.shape {
            Collider::Cuboid { half_extents } => half_extents.min_element(),
            Collider::Sphere { radius } => radius,
        }
    }

    fn velocity_at(&self, point: Vec3) -> Vec3 {
        self.velocity + self.spin.cross(point - self.position)
    }
}

/// An impulse for a rigid body, from a soft body touching or pinned to it.
#[derive(Clone, Copy)]
pub(crate) struct Reaction {
    pub obstacle: u32,
    pub point: Vec3,
    pub impulse: Vec3,
}

/// A deformable body. Build one with [`SoftBody::cloth`] or
/// [`SoftBody::lattice`], then add it to a [`crate::PhysicsWorld`].
#[derive(Debug, Clone)]
pub struct SoftBody {
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    /// How far a sheet's last solve moved each particle from free flight.
    correction: Vec<Vec3>,
    masses: Vec<f32>,
    /// Each particle's index when built: the same edge keeps its identity
    /// after tearing duplicates its ends.
    origin: Vec<u32>,
    triangles: Vec<[u32; 3]>,
    /// A sheet's elements, one per triangle, and each particle's as
    /// `element * 3 + corner` in `entries[offsets[i]..offsets[i + 1]]`.
    elements: Vec<Element>,
    offsets: Vec<u32>,
    entries: Vec<u32>,
    /// Likewise, as `hinge * 4 + slot`.
    hinges: Vec<Hinge>,
    hinge_offsets: Vec<u32>,
    hinge_entries: Vec<u32>,
    grid: Option<Grid>,
    /// Planes from other particles for this step: particle, normal, offset.
    planes: Vec<(u32, Vec3, f32)>,
    /// Half the particle spacing: how far apart particles keep (m).
    radius: f32,
    /// The membrane's Lamé constants, for `apply_stiffness`.
    shear: f32,
    lame: f32,
    /// Each original particle's position when built.
    rest: Vec<Vec3>,
    regions: Vec<Region>,
    pins: Vec<Pin>,
    /// The pin on each particle, or `NONE`.
    pinned: Vec<u32>,
    material: SoftMaterial,
    /// Sheets rebuild their mesh when an edge breaks.
    sheet: bool,
    /// Original edges that have torn, by `origin` indices.
    cuts: HashSet<(u32, u32)>,
    /// Their ends: where tears can grow from.
    cut_ends: HashSet<u32>,
    tears: usize,
    rest_time: f32,
    asleep: bool,
    contacts: Vec<Contact>,
}

fn edge(a: u32, b: u32) -> (u32, u32) {
    (a.min(b), a.max(b))
}

impl SoftBody {
    fn new(positions: Vec<Vec3>, masses: Vec<f32>, material: SoftMaterial) -> Self {
        let count = positions.len();
        Self {
            velocities: vec![Vec3::ZERO; count],
            correction: vec![Vec3::ZERO; count],
            origin: (0..count as u32).collect(),
            pinned: vec![NONE; count],
            rest: positions.clone(),
            positions,
            masses,
            triangles: Vec::new(),
            elements: Vec::new(),
            offsets: Vec::new(),
            entries: Vec::new(),
            hinges: Vec::new(),
            hinge_offsets: Vec::new(),
            hinge_entries: Vec::new(),
            grid: None,
            planes: Vec::new(),
            radius: 0.0,
            shear: 0.0,
            lame: 0.0,
            regions: Vec::new(),
            pins: Vec::new(),
            material,
            sheet: false,
            cuts: HashSet::new(),
            cut_ends: HashSet::new(),
            tears: 0,
            rest_time: 0.0,
            asleep: false,
            contacts: Vec::new(),
        }
    }

    /// A sheet of `size` metres in its local XY plane, of `cells` squares along
    /// each side, centred on `center`. Particles are slightly jittered, so
    /// tears wander rather than follow a grid line.
    pub fn cloth(
        center: Vec3,
        rotation: Quat,
        size: [f32; 2],
        cells: [u32; 2],
        material: SoftMaterial,
    ) -> Self {
        assert!(
            cells[0] > 0 && cells[1] > 0,
            "a sheet needs at least one cell"
        );
        assert!(
            size[0] > 0.0 && size[1] > 0.0,
            "a sheet's size must be positive"
        );
        let (nx, ny) = (cells[0] + 1, cells[1] + 1);
        let (dx, dy) = (size[0] / cells[0] as f32, size[1] / cells[1] as f32);
        let mut positions = Vec::with_capacity((nx * ny) as usize);
        for y in 0..ny {
            for x in 0..nx {
                let border = x == 0 || y == 0 || x + 1 == nx || y + 1 == ny;
                let jitter = |seed: u32, cell: f32| {
                    let hash = x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663) ^ seed;
                    let hash = hash.wrapping_mul(0x9e37_79b1) >> 16;
                    (hash as f32 / 65_535.0 - 0.5) * 0.3 * cell
                };
                let (jx, jy) = if border {
                    (0.0, 0.0)
                } else {
                    (jitter(1, dx), jitter(2, dy))
                };
                let local = Vec3::new(
                    x as f32 * dx - size[0] / 2.0 + jx,
                    y as f32 * dy - size[1] / 2.0 + jy,
                    0.0,
                );
                positions.push(center + rotation * local);
            }
        }
        let count = (nx * ny) as usize;
        let mut body = Self::new(positions, vec![0.0; count], material);
        for y in 0..cells[1] {
            for x in 0..cells[0] {
                let i = y * nx + x;
                let (a, b, c, d) = (i, i + 1, i + nx, i + nx + 1);
                // Alternating diagonals, so the sheet has no preferred shear.
                if (x + y) % 2 == 0 {
                    body.triangles.extend([[a, b, d], [a, d, c]]);
                } else {
                    body.triangles.extend([[a, b, c], [b, d, c]]);
                }
            }
        }
        body.sheet = true;
        body.radius = 0.45 * dx.min(dy);
        for &[a, b, c] in &body.triangles {
            let element = Element::new([a, b, c].map(|i| body.rest[i as usize]));
            for i in [a, b, c] {
                body.masses[i as usize] += material.density * element.area / 3.0;
            }
            body.elements.push(element);
        }
        body.rebuild();
        body
    }

    /// A solid box of `cells` along each axis: a particle at each grid point,
    /// held in shape by overlapping regions.
    pub fn lattice(
        center: Vec3,
        rotation: Quat,
        half_extents: Vec3,
        cells: [u32; 3],
        material: SoftMaterial,
    ) -> Self {
        Self::voxels(center, rotation, half_extents, cells, material, |_| true)
    }

    /// A solid of any shape: the cells of a box whose centers `inside` accepts,
    /// in the box's frame. Finer cells follow the shape more closely.
    pub fn voxels(
        center: Vec3,
        rotation: Quat,
        half_extents: Vec3,
        cells: [u32; 3],
        material: SoftMaterial,
        inside: impl Fn(Vec3) -> bool,
    ) -> Self {
        assert!(
            cells.iter().all(|&c| c > 0),
            "a lattice needs at least one cell"
        );
        assert!(
            half_extents.min_element() > 0.0,
            "a lattice's size must be positive"
        );
        let size = Vec3::new(cells[0] as f32, cells[1] as f32, cells[2] as f32);
        let step = half_extents * 2.0 / size;
        let filled_at = |x: i64, y: i64, z: i64| {
            let within = x >= 0
                && y >= 0
                && z >= 0
                && x < cells[0] as i64
                && y < cells[1] as i64
                && z < cells[2] as i64;
            within && inside((Vec3::new(x as f32, y as f32, z as f32) + 0.5) * step - half_extents)
        };
        // Particles are the corners of filled cells, each with an eighth of
        // the mass of every cell it is a corner of.
        let [nx, ny] = [cells[0] + 1, cells[1] + 1];
        let grid = |x: u32, y: u32, z: u32| (z * ny + y) * nx + x;
        let mut index = HashMap::new();
        let (mut positions, mut masses) = (Vec::new(), Vec::new());
        let cell_mass = material.density * step.x * step.y * step.z / 8.0;
        for z in 0..cells[2] {
            for y in 0..cells[1] {
                for x in 0..cells[0] {
                    if !filled_at(x as i64, y as i64, z as i64) {
                        continue;
                    }
                    for corner in 0..8u32 {
                        let (cx, cy, cz) =
                            (x + (corner & 1), y + (corner >> 1 & 1), z + (corner >> 2));
                        let i = *index.entry(grid(cx, cy, cz)).or_insert_with(|| {
                            let local =
                                Vec3::new(cx as f32, cy as f32, cz as f32) * step - half_extents;
                            positions.push(center + rotation * local);
                            masses.push(0.0);
                            positions.len() as u32 - 1
                        });
                        masses[i as usize] += cell_mass;
                    }
                }
            }
        }
        assert!(!positions.is_empty(), "a solid needs at least one cell");
        let mut body = Self::new(positions, masses, material);
        body.radius = 0.5 * step.min_element();
        body.grid = Some(Grid {
            center,
            rotation,
            half_extents,
            step,
            cells,
            index: index.clone(),
        });
        // Overlapping regions on a stride, with one at each far edge.
        let centers = |cells: u32| {
            let mut list: Vec<u32> = (0..=cells).step_by(REGION_STRIDE as usize).collect();
            if list.last() != Some(&cells) {
                list.push(cells);
            }
            list
        };
        for &cz in &centers(cells[2]) {
            for &cy in &centers(cells[1]) {
                for &cx in &centers(cells[0]) {
                    let span = |c: u32, n: u32| {
                        let c = c as i32;
                        (c - REGION_REACH).max(0) as u32..=(c + REGION_REACH).min(n as i32) as u32
                    };
                    let mut particles = Vec::new();
                    for z in span(cz, cells[2]) {
                        for y in span(cy, cells[1]) {
                            for x in span(cx, cells[0]) {
                                particles.extend(index.get(&grid(x, y, z)));
                            }
                        }
                    }
                    if particles.len() >= 4 {
                        let region = Region::new(particles, &body.positions, &body.masses);
                        body.regions.push(region);
                    }
                }
            }
        }
        // The surface: every face of a filled cell with no filled neighbour.
        let at = |x: u32, y: u32, z: u32| index[&grid(x, y, z)];
        for z in 0..cells[2] {
            for y in 0..cells[1] {
                for x in 0..cells[0] {
                    let (i, j, k) = (x as i64, y as i64, z as i64);
                    if !filled_at(i, j, k) {
                        continue;
                    }
                    let (a, b, c) = (x + 1, y + 1, z + 1);
                    let mut quad = |p: u32, q: u32, r: u32, s: u32| {
                        body.triangles.extend([[p, q, r], [p, r, s]]);
                    };
                    if !filled_at(i, j, k - 1) {
                        quad(at(x, y, z), at(x, b, z), at(a, b, z), at(a, y, z));
                    }
                    if !filled_at(i, j, k + 1) {
                        quad(at(x, y, c), at(a, y, c), at(a, b, c), at(x, b, c));
                    }
                    if !filled_at(i, j - 1, k) {
                        quad(at(x, y, z), at(a, y, z), at(a, y, c), at(x, y, c));
                    }
                    if !filled_at(i, j + 1, k) {
                        quad(at(x, b, z), at(x, b, c), at(a, b, c), at(a, b, z));
                    }
                    if !filled_at(i - 1, j, k) {
                        quad(at(x, y, z), at(x, y, c), at(x, b, c), at(x, b, z));
                    }
                    if !filled_at(i + 1, j, k) {
                        quad(at(a, y, z), at(a, b, z), at(a, b, c), at(a, y, c));
                    }
                }
            }
        }
        body
    }

    /// Tie points, such as a detailed mesh's vertices, to this solid as built:
    /// [`SoftBody::deformed`] then says where they have gone. Each follows the
    /// corners of its grid cell, or the particle nearest it outside the solid.
    pub fn embed(&self, points: &[Vec3]) -> Vec<[(u32, f32); 8]> {
        let Some(grid) = &self.grid else {
            return Vec::new();
        };
        let [nx, ny] = [grid.cells[0] + 1, grid.cells[1] + 1];
        let limit = Vec3::new(
            grid.cells[0] as f32,
            grid.cells[1] as f32,
            grid.cells[2] as f32,
        );
        (points.iter())
            .map(|&point| {
                let local = grid.rotation.conjugate() * (point - grid.center);
                let cell = ((local + grid.half_extents) / grid.step).clamp(Vec3::ZERO, limit);
                let base = cell.floor().min(limit - 1.0);
                let t = cell - base;
                let mut weights = [(0, 0.0); 8];
                let mut total = 0.0;
                for (corner, weight) in weights.iter_mut().enumerate() {
                    let (dx, dy, dz) = (corner & 1, corner >> 1 & 1, corner >> 2);
                    let (x, y, z) = (
                        base.x as u32 + dx as u32,
                        base.y as u32 + dy as u32,
                        base.z as u32 + dz as u32,
                    );
                    let share = |d: usize, t: f32| if d == 1 { t } else { 1.0 - t };
                    if let Some(&i) = grid.index.get(&((z * ny + y) * nx + x)) {
                        let w = share(dx, t.x) * share(dy, t.y) * share(dz, t.z);
                        *weight = (i, w);
                        total += w;
                    }
                }
                if total > 1.0e-6 {
                    for weight in &mut weights {
                        weight.1 /= total;
                    }
                } else {
                    let nearest = (0..self.rest.len() as u32)
                        .min_by(|&a, &b| {
                            let d = |i: u32| self.rest[i as usize].distance_squared(point);
                            d(a).total_cmp(&d(b))
                        })
                        .unwrap_or(0);
                    weights[0] = (nearest, 1.0);
                }
                weights
            })
            .collect()
    }

    /// Where points tied by [`SoftBody::embed`] are now.
    pub fn deformed(&self, embedding: &[[(u32, f32); 8]]) -> Vec<Vec3> {
        (embedding.iter())
            .map(|weights| {
                (weights.iter()).fold(Vec3::ZERO, |sum, &(i, w)| {
                    sum + self.positions[i as usize] * w
                })
            })
            .collect()
    }

    /// After cuts: drop triangles torn on two edges, duplicate particles the
    /// cuts have separated, and index what is left.
    fn rebuild(&mut self) {
        // A triangle torn on two edges holds nothing: leaving it would hang a
        // sliver from the tear.
        let (origin, cuts) = (&self.origin, &self.cuts);
        let torn = |t: &[u32; 3]| {
            let cut = |a: u32, b: u32| cuts.contains(&edge(origin[a as usize], origin[b as usize]));
            [cut(t[0], t[1]), cut(t[1], t[2]), cut(t[2], t[0])]
                .iter()
                .filter(|&&c| c)
                .count()
        };
        let keep: Vec<bool> = self.triangles.iter().map(|t| torn(t) < 2).collect();
        let mut index = 0;
        self.elements.retain(|_| {
            index += 1;
            keep[index - 1]
        });
        index = 0;
        self.triangles.retain(|_| {
            index += 1;
            keep[index - 1]
        });
        self.split_particles();

        let count = self.positions.len();
        let corners = (self.triangles.iter().enumerate())
            .flat_map(|(t, tri)| (0..3).map(move |c| (tri[c], t as u32 * 3 + c as u32)));
        (self.offsets, self.entries) = index_by_particle(count, corners);

        // Each uncut edge between two triangles is a hinge.
        let mut sides: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        for (t, tri) in self.triangles.iter().enumerate() {
            for corner in 0..3 {
                let (a, b) = (tri[corner], tri[(corner + 1) % 3]);
                sides.entry(edge(a, b)).or_default().push(t as u32);
            }
        }
        let mut edges: Vec<_> = sides.into_iter().filter(|(_, t)| t.len() == 2).collect();
        edges.sort_unstable();
        self.hinges.clear();
        for ((a, b), triangles) in edges {
            let opposite = |t: u32| {
                let tri = self.triangles[t as usize];
                tri.into_iter()
                    .find(|&v| v != a && v != b)
                    .expect("a triangle has three corners")
            };
            let (c, d) = (opposite(triangles[0]), opposite(triangles[1]));
            let from = |i: u32| self.rest[self.origin[i as usize] as usize];
            let area = self.elements[triangles[0] as usize].area
                + self.elements[triangles[1] as usize].area;
            let weights = hinge_weights([from(a), from(b), from(c), from(d)]);
            let scale = (3.0 * self.material.bending / area).sqrt();
            self.hinges.push(Hinge {
                particles: [a, b, c, d],
                weights: weights.map(|w| w * scale),
            });
        }
        let slots = (self.hinges.iter().enumerate()).flat_map(|(h, hinge)| {
            (0..4).map(move |s| (hinge.particles[s], h as u32 * 4 + s as u32))
        });
        (self.hinge_offsets, self.hinge_entries) = index_by_particle(count, slots);
    }

    /// Where cuts have split the triangles around a particle into separate
    /// fans, give each further fan its own copy of the particle.
    fn split_particles(&mut self) {
        let mut fans: Vec<Vec<u32>> = vec![Vec::new(); self.positions.len()];
        for (t, tri) in self.triangles.iter().enumerate() {
            for &v in tri {
                fans[v as usize].push(t as u32);
            }
        }
        for v in 0..fans.len() as u32 {
            let fan = &fans[v as usize];
            if fan.len() < 2 {
                continue;
            }
            // Group the fan's triangles across edges from `v` that are uncut.
            let mut group: Vec<usize> = (0..fan.len()).collect();
            fn find(group: &mut [usize], mut i: usize) -> usize {
                while group[i] != i {
                    group[i] = group[group[i]];
                    i = group[i];
                }
                i
            }
            for i in 0..fan.len() {
                for j in i + 1..fan.len() {
                    let (s, t) = (
                        self.triangles[fan[i] as usize],
                        self.triangles[fan[j] as usize],
                    );
                    let shared = s.iter().find(|&&p| p != v && t.contains(&p));
                    if let Some(&p) = shared {
                        let original = edge(self.origin[v as usize], self.origin[p as usize]);
                        if !self.cuts.contains(&original) {
                            let (a, b) = (find(&mut group, i), find(&mut group, j));
                            group[a] = b;
                        }
                    }
                }
            }
            let roots: Vec<usize> = (0..fan.len()).map(|i| find(&mut group, i)).collect();
            let mut distinct = roots.clone();
            distinct.sort_unstable();
            distinct.dedup();
            if distinct.len() < 2 {
                continue;
            }
            let share = self.masses[v as usize] / distinct.len() as f32;
            self.masses[v as usize] = share;
            for &root in &distinct[1..] {
                let copy = self.positions.len() as u32;
                self.positions.push(self.positions[v as usize]);
                self.velocities.push(self.velocities[v as usize]);
                self.correction.push(self.correction[v as usize]);
                self.masses.push(share);
                self.origin.push(self.origin[v as usize]);
                self.pinned.push(NONE);
                for (i, &t) in fan.iter().enumerate() {
                    if roots[i] == root {
                        for corner in &mut self.triangles[t as usize] {
                            if *corner == v {
                                *corner = copy;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Pin the particles whose starting position satisfies `select` where
    /// they are now, to the world or to `body`, until `strength` newtons tear
    /// them out. Infinite strength never lets go.
    pub fn pin(&mut self, mut select: impl FnMut(Vec3) -> bool, strength: f32) -> usize {
        self.pin_with(&mut select, None, |p| p, strength)
    }

    pub(crate) fn pin_with(
        &mut self,
        select: &mut dyn FnMut(Vec3) -> bool,
        body: Option<EntityHandle>,
        to_body: impl Fn(Vec3) -> Vec3,
        strength: f32,
    ) -> usize {
        let mut count = 0;
        for (i, &position) in self.positions.iter().enumerate() {
            if select(position) && self.pinned[i] == NONE {
                self.pinned[i] = self.pins.len() as u32;
                self.pins.push(Pin {
                    particle: i as u32,
                    body,
                    point: to_body(position),
                    strength,
                    load: 0.0,
                });
                count += 1;
            }
        }
        count
    }

    /// Give every particle this velocity.
    pub fn set_velocity(&mut self, velocity: Vec3) {
        self.velocities.fill(velocity);
    }

    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }
    pub fn velocities(&self) -> &[Vec3] {
        &self.velocities
    }
    /// The surface as triangles of particle indices.
    pub fn triangles(&self) -> &[[u32; 3]] {
        &self.triangles
    }
    /// Edges torn or cracks opened so far.
    pub fn tears(&self) -> usize {
        self.tears
    }
    pub fn pin_count(&self) -> usize {
        self.pins.len()
    }
    /// The largest permanent strain of any region.
    pub fn permanent_strain(&self) -> f32 {
        self.regions.iter().map(|r| r.strain).fold(0.0, f32::max)
    }
    /// Total mass (kg).
    pub fn mass(&self) -> f32 {
        self.masses.iter().sum()
    }
    /// Separate pieces of surface.
    pub fn pieces(&self) -> usize {
        let mut parent: Vec<u32> = (0..self.positions.len() as u32).collect();
        fn find(parent: &mut [u32], mut i: u32) -> u32 {
            while parent[i as usize] != i {
                parent[i as usize] = parent[parent[i as usize] as usize];
                i = parent[i as usize];
            }
            i
        }
        for region in &self.regions {
            for &i in &region.particles {
                let (a, b) = (find(&mut parent, region.particles[0]), find(&mut parent, i));
                parent[a as usize] = b;
            }
        }
        if !self.sheet {
            let mut roots: Vec<u32> = (self.regions.iter())
                .flat_map(|r| r.particles.iter().copied())
                .map(|i| find(&mut parent, i))
                .collect();
            roots.sort_unstable();
            roots.dedup();
            return roots.len();
        }
        for t in &self.triangles {
            for corner in 1..3 {
                let (a, b) = (find(&mut parent, t[0]), find(&mut parent, t[corner]));
                parent[a as usize] = b;
            }
        }
        let mut roots: Vec<u32> = (self.triangles.iter().flatten())
            .map(|&i| find(&mut parent, i))
            .collect();
        roots.sort_unstable();
        roots.dedup();
        roots.len()
    }

    fn bounds(&self) -> (Vec3, Vec3) {
        let first = self.positions.first().copied().unwrap_or(Vec3::ZERO);
        (self.positions.iter()).fold((first, first), |(min, max), &p| (min.min(p), max.max(p)))
    }

    /// Advance by `dt`. `pose` gives a pinned body's world point for a point
    /// on it; `None` drops the pin. Returns impulses for rigid bodies.
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn step(
        &mut self,
        dt: f32,
        gravity: Vec3,
        obstacles: &[Obstacle],
        pin_target: impl Fn(EntityHandle, Vec3) -> Option<(Vec3, u32)>,
        reactions: &mut Vec<Reaction>,
    ) {
        let (min, max) = self.bounds();
        let travel = self
            .velocities
            .iter()
            .fold(0.0f32, |s, v| s.max(v.length_squared()))
            .sqrt()
            * dt;
        let reach = self.material.thickness + travel + gravity.length() * dt * dt;
        let near: Vec<u32> = (obstacles.iter().enumerate())
            .filter(|(_, o)| {
                let r = o.radius() + reach;
                let closest = o.position.clamp(min, max);
                closest.distance_squared(o.position) <= r * r
            })
            .map(|(i, _)| i as u32)
            .collect();
        let carried = self.pins.iter().any(|pin| {
            let held = pin.body.and_then(|body| pin_target(body, pin.point));
            held.is_some_and(|(_, owner)| obstacles[owner as usize].awake)
        });
        if self.asleep {
            if !carried && !near.iter().any(|&i| obstacles[i as usize].awake) {
                return;
            }
            self.asleep = false;
            self.rest_time = 0.0;
        }

        let start = self.positions.clone();
        let momentum_before: Vec<Vec3> = (self.velocities.iter().zip(&self.masses))
            .map(|(v, m)| *v * *m)
            .collect();
        let scale = 1.0 / (dt * dt);
        let inertial: Vec<Vec3> = (self.positions.iter().zip(&self.velocities))
            .map(|(p, v)| *p + *v * dt + gravity * (dt * dt))
            .collect();
        self.positions.copy_from_slice(&inertial);
        // A sheet starts from last step's answer to what held it up: at rest
        // that is nearly this step's too, and the solve has little left to do.
        if self.sheet {
            // Only what opposed gravity is sure to repeat. Anything sideways left
            // by an unfinished solve would feed itself, and the sheet never rest.
            let fall = gravity * (dt * dt);
            let down = fall.length_squared();
            // A particle on a surface needs none: its contact holds it directly,
            // and it must stay free to slide.
            let mut touching = vec![false; self.positions.len()];
            for contact in &self.contacts {
                touching[contact.particle as usize] = true;
            }
            for (i, c) in self.correction.iter().enumerate() {
                if down > 0.0 && !touching[i] {
                    self.positions[i] -= fall * (-c.dot(fall) / down).clamp(0.0, 1.0);
                }
            }
        }

        // Where each pin wants its particle.
        let mut targets = vec![Vec3::ZERO; self.pins.len()];
        let mut owners = vec![NONE; self.pins.len()];
        let mut gone = vec![false; self.pins.len()];
        for (i, pin) in self.pins.iter().enumerate() {
            targets[i] = match pin.body {
                None => pin.point,
                Some(body) => match pin_target(body, pin.point) {
                    Some((point, owner)) => {
                        owners[i] = owner;
                        point
                    }
                    None => {
                        gone[i] = true;
                        pin.point
                    }
                },
            };
        }

        self.contacts.clear();
        let thickness = self.material.thickness;
        for (i, &point) in inertial.iter().enumerate() {
            for &o in near.iter().filter(|o| !owners.contains(o)) {
                let obstacle = &obstacles[o as usize];
                // Swept against the obstacle, so a fast particle meets a thin
                // wall it would otherwise be past by the end of the step.
                let path = point - start[i] - obstacle.velocity * dt;
                let hit = if path.length() > obstacle.narrowest() {
                    obstacle.sweep(point - path, point, thickness)
                } else {
                    obstacle.nearest(point, reach)
                };
                if let Some((surface, normal)) = hit {
                    self.contacts.push(Contact {
                        particle: i as u32,
                        obstacle: o,
                        normal,
                        offset: normal.dot(surface) + thickness,
                        force: 0.0,
                        pushed: 0.0,
                        depth: 0.0,
                    });
                }
            }
        }
        if self.sheet {
            self.touch_triangles(&inertial, obstacles, &near, &owners, dt);
        }
        for &(particle, normal, offset) in &self.planes {
            self.contacts.push(Contact {
                particle,
                obstacle: NONE,
                normal,
                offset,
                force: 0.0,
                pushed: 0.0,
                depth: 0.0,
            });
        }
        let m = self.material;
        let membrane = Membrane {
            shear: m.stiffness / (2.0 * (1.0 + m.poisson)),
            lame: m.stiffness * m.poisson / (1.0 - m.poisson * m.poisson),
            compression: m.compression,
        };
        // A pin holds as firmly as the sheet, but a rigid body feels it a step
        // late: too firm for that body's mass, and the two would ring.
        let pin_stiffness: Vec<f32> = (self.pins.iter().enumerate())
            .map(|(i, pin)| {
                let own = PIN_STIFFNESS * self.masses[pin.particle as usize] * scale;
                let firm = if self.sheet {
                    m.stiffness.max(own)
                } else {
                    own
                };
                if owners[i] == NONE {
                    return firm;
                }
                let sharing = owners.iter().filter(|&&o| o == owners[i]).count() as f32;
                firm.min(0.25 * obstacles[owners[i] as usize].mass * scale / sharing)
                    .max(own)
            })
            .collect();
        let count = self.positions.len();
        let mut pin_pushed = vec![Vec3::ZERO; self.pins.len()];
        let solid = !self.sheet;
        let mut goals = vec![Vec3::ZERO; if solid { count } else { 0 }];
        let mut shares = vec![0u32; goals.len()];
        let settling = gravity.length() * dt * dt;
        for iteration in 0..if solid { ITERATIONS } else { 0 } {
            // Right after contacts first push, before shape sweeps erase it. A
            // shape past yield stops resisting, which limits the force through
            // a solid as a joint's cap does.
            if iteration == 1 {
                self.flow_regions(settling);
            }
            goals.fill(Vec3::ZERO);
            shares.fill(0);
            for region in &mut self.regions {
                let (center, deformation) = region.deformation(&self.positions, &self.masses);
                region.track_rotation(deformation);
                for (&i, &q) in region.particles.iter().zip(&region.rest) {
                    goals[i as usize] += center + region.rotation * q;
                    shares[i as usize] += 1;
                }
            }
            let rigidity = self.material.rigidity;
            for i in 0..count {
                if shares[i] > 0 {
                    let goal = goals[i] / shares[i] as f32;
                    let x = self.positions[i];
                    self.positions[i] = x + (goal - x) * rigidity;
                }
            }
            for (index, pin) in self.pins.iter().enumerate() {
                let i = pin.particle as usize;
                let total = pin_pushed[index] + targets[index] - self.positions[i];
                self.positions[i] += total - pin_pushed[index];
                pin_pushed[index] = total;
            }
            for contact in &mut self.contacts {
                let i = contact.particle as usize;
                let gap = contact.normal.dot(self.positions[i]) - contact.offset;
                contact.depth = (-gap).max(0.0);
                if gap >= 0.0 {
                    continue;
                }
                // A dynamic body lighter than the solid moves it only so far.
                let local = self.masses[i] * CONTACT_SPREAD;
                let share = match obstacles.get(contact.obstacle as usize) {
                    Some(o) if o.mass > 0.0 => o.mass / (o.mass + local),
                    _ => 1.0,
                };
                self.positions[i] -= contact.normal * (gap * share);
                contact.pushed -= gap * share;
            }
        }
        for contact in &mut self.contacts {
            let mass = self.masses[contact.particle as usize];
            if solid {
                contact.force = mass * contact.pushed * scale;
            }
        }
        if !solid {
            self.shear = membrane.shear;
            self.lame = membrane.lame;
            self.solve_sheet(
                &inertial,
                &start,
                &targets,
                &pin_stiffness,
                &membrane,
                obstacles,
                dt,
            );
            for i in 0..count {
                self.correction[i] = self.positions[i] - inertial[i];
            }
        }

        // Friction on a solid, then velocities and what the rigid bodies feel.
        for contact in self.contacts.iter().filter(|c| solid && c.obstacle != NONE) {
            let i = contact.particle as usize;
            let obstacle = &obstacles[contact.obstacle as usize];
            let surface = obstacle.velocity_at(self.positions[i]) * dt;
            let moved = self.positions[i] - start[i] - surface;
            let tangent = moved - contact.normal * contact.normal.dot(moved);
            let length = tangent.length();
            let coefficient = (self.material.friction * obstacle.friction).sqrt();
            let limit = coefficient * contact.force * dt * dt / self.masses[i];
            let slip = if length > limit && length > 0.0 {
                tangent * (limit / length)
            } else {
                tangent
            };
            self.positions[i] -= slip;
        }
        // Damping acts on motion within the body, not on the body as a whole,
        // which would be air resistance. Cloth has only motion within.
        let keep = (-self.material.damping * dt).exp();
        let whole = if solid {
            let sum: Vec3 = (0..count)
                .map(|i| (self.positions[i] - start[i]) * self.masses[i])
                .sum();
            sum / (self.mass() * dt)
        } else {
            Vec3::ZERO
        };
        let mut fastest = 0.0f32;
        for (i, &from) in start.iter().enumerate() {
            let velocity = (self.positions[i] - from) / dt;
            let velocity = whole + (velocity - whole) * keep;
            self.velocities[i] = velocity;
            fastest = fastest.max(velocity.length_squared());
        }

        let blend = 1.0 - (-dt / LOAD_TIME).exp();
        // What the pins and contacts gave the body, from its change of momentum.
        let mut received = -gravity * (self.mass() * dt);
        for (i, &from) in start.iter().enumerate() {
            received += (self.positions[i] - from) / dt * self.masses[i] - momentum_before[i];
        }
        for (i, pin) in self.pins.iter_mut().enumerate() {
            let particle = pin.particle as usize;
            let mass = self.masses[particle];
            let force = if solid {
                -pin_pushed[i] * (mass * scale)
            } else {
                (self.positions[particle] - targets[i]) * pin_stiffness[i]
            };
            pin.load += (force.length() - pin.load) * blend;
            received += force * dt;
            if owners[i] != NONE {
                reactions.push(Reaction {
                    obstacle: owners[i],
                    point: targets[i],
                    impulse: force * dt,
                });
            }
            gone[i] |= pin.load >= pin.strength;
        }
        // A solid carrying a rigid body stops it as a crushing structure does:
        // with its crush strength over the area touching what it hit.
        if let (true, Some(&owner)) = (solid, owners.iter().find(|&&o| o != NONE)) {
            let carried = &obstacles[owner as usize];
            let face = self.grid.as_ref().map_or(0.0, |g| {
                let s = g.step;
                (s.x * s.y).min(s.y * s.z).min(s.x * s.z)
            });
            let crush = self.material.strength.yield_load * face * dt;
            let (mut impulse, mut point, mut touching) = (Vec3::ZERO, Vec3::ZERO, 0.0);
            for contact in self
                .contacts
                .iter()
                .filter(|c| c.pushed > 0.0 && c.obstacle != NONE)
            {
                impulse += contact.normal * crush;
                point += self.positions[contact.particle as usize];
                touching += 1.0;
            }
            if touching > 0.0 {
                let size = impulse.length().max(1.0e-9);
                let closing = -carried.velocity.dot(impulse / size);
                let impulse = impulse * ((closing.max(0.0) * carried.mass).min(size) / size);
                reactions.push(Reaction {
                    obstacle: owner,
                    point: point / touching,
                    impulse,
                });
            }
        }
        // The rest came from contacts: share it by how hard each pushed.
        let total: f32 = self.contacts.iter().map(|c| c.pushed).sum();
        if total > 0.0 {
            let first = reactions.len();
            for contact in self
                .contacts
                .iter()
                .filter(|c| c.pushed > 0.0 && c.obstacle != NONE)
            {
                let obstacle = &obstacles[contact.obstacle as usize];
                if obstacle.mass > 0.0 {
                    reactions.push(Reaction {
                        obstacle: contact.obstacle,
                        point: self.positions[contact.particle as usize],
                        // Along its normal only: pins' share of `received`
                        // must not push a body sideways.
                        impulse: if solid {
                            -contact.normal * contact.pushed
                        } else {
                            -contact.normal
                                * (received.dot(contact.normal).max(0.0) * contact.pushed / total)
                        },
                    });
                }
            }
            // A contact stops bodies closing, and never throws them apart:
            // cap each body's impulse at what cancels its closing speed.
            let (mut speed, mut weight) = (Vec3::ZERO, 0.0);
            for contact in self.contacts.iter().filter(|c| c.pushed > 0.0) {
                speed += self.velocities[contact.particle as usize] * contact.pushed;
                weight += contact.pushed;
            }
            let speed = speed / weight;
            for o in near.iter().copied() {
                let obstacle = &obstacles[o as usize];
                let of_body = || reactions[first..].iter().filter(|r| r.obstacle == o);
                let impulse: Vec3 = of_body().map(|r| r.impulse).sum();
                let size = impulse.length();
                if size == 0.0 {
                    continue;
                }
                // Cloth pressed by a body moves with it, so its speed says nothing
                // of how fast they close: the body's own speed is the limit.
                let surface = if solid { speed } else { Vec3::ZERO };
                let closing = (surface - obstacle.velocity - gravity * dt).dot(impulse / size);
                // Plus a gentle push out of any overlap left, or a body at rest
                // would press into a solid for ever.
                let depth = (self.contacts.iter())
                    .filter(|c| c.obstacle == o)
                    .fold(0.0f32, |d, c| d.max(c.depth));
                let limit = (closing + SEPARATION * depth / dt).max(0.0) * obstacle.mass;
                // A solid meets a body as two masses colliding: the body and
                // the region around the contact. Cloth just passes on momentum.
                let fit = if solid {
                    let local = self.masses[0] * CONTACT_SPREAD;
                    limit * local / (local + obstacle.mass) / size
                } else {
                    (limit / size).min(1.0)
                };
                let wakes = obstacle.awake || size * fit > WAKE_SPEED * obstacle.mass;
                for reaction in reactions[first..].iter_mut().filter(|r| r.obstacle == o) {
                    reaction.impulse *= if wakes { fit } else { 0.0 };
                }
            }
        }
        let released = gone.contains(&true);
        if released {
            let mut index = 0;
            self.pins.retain(|_| {
                index += 1;
                !gone[index - 1]
            });
            self.pinned.fill(NONE);
            for (index, pin) in self.pins.iter().enumerate() {
                self.pinned[pin.particle as usize] = index as u32;
            }
        }
        if self.sheet {
            self.tear(&membrane, blend);
        }

        if fastest.sqrt() < SLEEP_SPEED && !released && !carried {
            self.rest_time += dt;
            self.asleep = self.rest_time >= TIME_TO_SLEEP;
        } else {
            self.rest_time = 0.0;
        }
    }

    /// Move each overstrained region's rest shape toward its current shape.
    /// `settling` is the distance a step's gravity moves anything: pressed on a
    /// support, a body shows that much strain across its thickness at rest.
    fn flow_regions(&mut self, settling: f32) {
        let m = self.material;
        let (yield_strain, break_strain) = (
            m.strength.yield_load / m.stiffness,
            m.strength.break_load / m.stiffness,
        );
        let mut overloaded = None;
        for (index, region) in self.regions.iter_mut().enumerate() {
            let (_, deformation) = region.deformation(&self.positions, &self.masses);
            region.track_rotation(deformation);
            let stretch = Mat3::from_quat(region.rotation).transpose() * deformation;
            let excess = stretch - Mat3::IDENTITY;
            let strain = (excess.x_axis.length_squared()
                + excess.y_axis.length_squared()
                + excess.z_axis.length_squared())
            .sqrt();
            let allowance = 2.0 * settling / region.thickness;
            let threshold = yield_strain + allowance;
            // Past what it can take, a brittle region cracks at once, and a
            // ductile one once it has run out of ductility.
            let brittle = m.strength.yield_load >= m.strength.break_load;
            let spent = if brittle {
                strain > break_strain + allowance
            } else {
                m.strength.break_load.is_finite() && region.strain >= m.strength.ductility
            };
            if spent {
                let worst = overloaded.map_or(0.0, |(_, s, _)| s);
                if strain > worst {
                    overloaded = Some((index, strain, excess));
                }
                continue;
            }
            if !brittle && strain > threshold {
                let flow = (strain - threshold) / strain;
                let change = Mat3::IDENTITY + excess * flow;
                for q in &mut region.rest {
                    *q = change * *q;
                }
                region.update_moment(&self.masses);
                region.strain += strain - threshold;
            }
        }
        if let Some((index, _, excess)) = overloaded {
            self.crack(index, principal_axis(excess));
        }
    }

    /// Split the regions a crack crosses, so nothing holds its two sides
    /// together: particles are coupled only through regions. The crack runs
    /// across the region's largest strain, as far as a region reaches, so a
    /// long crack grows as each new tip overloads.
    fn crack(&mut self, region: usize, direction: Vec3) {
        let (center, _) = self.regions[region].deformation(&self.positions, &self.masses);
        let normal = self.regions[region].rotation * direction;
        let reach = (self.regions[region].rest.iter())
            .fold(0.0f32, |r, q| r.max(q.length_squared()))
            .sqrt();
        let positions = &self.positions;
        let side = |i: u32| normal.dot(positions[i as usize] - center) >= 0.0;
        let mut split = Vec::new();
        self.regions.retain_mut(|r| {
            let members = || r.particles.iter().zip(&r.rest);
            let near = r
                .particles
                .iter()
                .any(|&i| positions[i as usize].distance(center) <= reach);
            let ahead = r.particles.iter().filter(|&&i| side(i)).count();
            if !near || ahead == 0 || ahead == r.particles.len() {
                return true;
            }
            for wanted in [true, false] {
                let (particles, rest): (Vec<u32>, Vec<Vec3>) = members()
                    .filter(|(i, _)| side(**i) == wanted)
                    .map(|(i, q)| (*i, *q))
                    .unzip();
                if particles.len() >= 4 {
                    split.push((particles, rest, r.rotation, r.strain));
                }
            }
            false
        });
        for (particles, rest, rotation, strain) in split {
            self.regions.push(Region::from_rest(
                particles,
                rest,
                rotation,
                strain,
                &self.masses,
            ));
        }
        // The surface opens along the crack.
        self.triangles.retain(|t| {
            let mixed = side(t[0]) != side(t[1]) || side(t[1]) != side(t[2]);
            let far = positions[t[0] as usize].distance(center) > reach;
            !mixed || far
        });
        self.tears += 1;
    }

    /// Contacts for bodies small enough to pass between particles: such a body
    /// presses the triangle it meets, along its path through the step.
    fn touch_triangles(
        &mut self,
        inertial: &[Vec3],
        obstacles: &[Obstacle],
        near: &[u32],
        owners: &[u32],
        dt: f32,
    ) {
        let thickness = self.material.thickness;
        let small = self.radius * SMALL_BODY;
        for &o in near.iter().filter(|o| !owners.contains(o)) {
            let obstacle = &obstacles[o as usize];
            let radius = obstacle.radius();
            if radius > small {
                continue;
            }
            let path = obstacle.velocity * dt;
            let samples = ((path.length() / radius).ceil() as usize).clamp(1, MAX_SWEEP_SAMPLES);
            let hit = (0..=samples).find_map(|s| {
                let center = obstacle.position - path * (1.0 - s as f32 / samples as f32);
                (self.triangles.iter()).find_map(|t| {
                    let corners = t.map(|c| inertial[c as usize]);
                    let closest = closest_on_triangle(center, corners);
                    let d = closest - center;
                    (d.length() < radius + thickness).then_some((*t, closest, d, center))
                })
            });
            if let Some((t, closest, d, center)) = hit {
                // Pressed along its travel, or away from its center if at rest.
                let normal = path
                    .try_normalize()
                    .or(d.try_normalize())
                    .unwrap_or(Vec3::Z);
                let depth = radius + thickness - closest.distance(center);
                for corner in t {
                    let i = corner as usize;
                    self.contacts.push(Contact {
                        particle: corner,
                        obstacle: o,
                        normal,
                        offset: normal.dot(inertial[i]) + depth + path.length(),
                        force: 0.0,
                        pushed: 0.0,
                        depth: 0.0,
                    });
                }
            }
        }
    }

    /// One implicit step of a sheet (Baraff and Witkin 1998): Newton iterations,
    /// each solved by conjugate gradients. Sweeping vertex by vertex cannot do
    /// this: a stiff sheet's large motions barely change in a few sweeps, and
    /// more so the finer its mesh.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn solve_sheet(
        &mut self,
        inertial: &[Vec3],
        start: &[Vec3],
        targets: &[Vec3],
        pin_stiffness: &[f32],
        membrane: &Membrane,
        obstacles: &[Obstacle],
        dt: f32,
    ) {
        let count = self.positions.len();
        let scale = 1.0 / (dt * dt);
        let rate = ELEMENT_DAMPING / dt;
        let contact_stiffness = self.material.stiffness * CONTACT_STIFFNESS;
        let inertia: Vec<f32> = self.masses.iter().map(|m| m * scale).collect();
        let mut pin_of = vec![0.0f32; count];
        for (pin, &k) in self.pins.iter().zip(pin_stiffness) {
            pin_of[pin.particle as usize] = k;
        }
        let mut states = vec![ElementState::default(); self.elements.len()];
        let mut work = vec![[Vec3::ZERO; 3]; self.elements.len()];
        for newton in 0..NEWTON_STEPS {
            let (elements, triangles, positions) =
                (&self.elements, &self.triangles, &self.positions);
            states
                .par_iter_mut()
                .with_min_len(PARALLEL_CHUNK)
                .enumerate()
                .for_each(|(t, state)| {
                    let (a, b) =
                        elements[t].deformation(triangles[t].map(|c| positions[c as usize]));
                    let (stress, firm) = stress(a, b, membrane);
                    *state = ElementState {
                        a,
                        b,
                        stress,
                        tension: tension_only(stress),
                        firm,
                    };
                });
            // Which contacts press, as springs along their normals.
            let mut contact_of = vec![(Vec3::ZERO, 0.0f32, 0.0f32); count];
            let mut pressing = vec![Vec3::ZERO; count];
            let mut deepest = vec![0.0f32; count];
            for contact in &mut self.contacts {
                let i = contact.particle as usize;
                let gap = contact.normal.dot(self.positions[i]) - contact.offset;
                contact.depth = (-gap).max(0.0);
                // The deepest of a particle's contacts, so its force and its
                // stiffness in the solve stay one spring.
                if gap < 0.0 && contact.depth > deepest[i] {
                    deepest[i] = contact.depth;
                    // Friction inside the solve, so the sheet can answer it: a
                    // spring along the surface that slips at Coulomb's limit.
                    let n = contact.normal;
                    let (surface, friction) = match obstacles.get(contact.obstacle as usize) {
                        Some(o) => (o.velocity_at(self.positions[i]), o.friction),
                        None => (Vec3::ZERO, 0.0),
                    };
                    let moved = self.positions[i] - start[i] - surface * dt;
                    let slip = moved - n * n.dot(moved);
                    let limit =
                        (self.material.friction * friction).sqrt() * contact_stiffness * -gap;
                    let along = (limit / slip.length().max(1.0e-9)).min(contact_stiffness);
                    contact_of[i] = (n, contact_stiffness, along);
                    pressing[i] = n * (contact_stiffness * gap) + slip * along;
                }
            }

            // The gradient: inertia, elastic forces, damping, pins, contacts.
            let moved: Vec<Vec3> = (self.positions.iter().zip(start))
                .map(|(x, s)| *x - *s)
                .collect();
            let mut damping = vec![Vec3::ZERO; count];
            self.apply_stiffness(&states, &moved, &mut work, &mut damping);
            let mut forces = vec![[Vec3::ZERO; 3]; self.elements.len()];
            let (elements, positions) = (&self.elements, &self.positions);
            forces
                .par_iter_mut()
                .with_min_len(PARALLEL_CHUNK)
                .enumerate()
                .for_each(|(t, f)| {
                    let (element, state) = (&elements[t], &states[t]);
                    let [sxx, syy, sxy] = state.stress;
                    for c in 0..3 {
                        let g = element.gradient[c];
                        let sg = Vec2::new(sxx * g.x + sxy * g.y, sxy * g.x + syy * g.y);
                        f[c] = (state.a * sg.x + state.b * sg.y) * element.area;
                    }
                });
            let mut gradient = vec![Vec3::ZERO; count];
            for i in 0..count {
                let mut sum =
                    (positions[i] - inertial[i]) * inertia[i] + damping[i] * rate + pressing[i];
                for &entry in &self.entries[self.offsets[i] as usize..self.offsets[i + 1] as usize]
                {
                    sum += forces[(entry / 3) as usize][(entry % 3) as usize];
                }
                let hinges = &self.hinge_entries
                    [self.hinge_offsets[i] as usize..self.hinge_offsets[i + 1] as usize];
                for &entry in hinges {
                    let hinge = &self.hinges[(entry / 4) as usize];
                    let bend = (hinge.particles.iter().zip(&hinge.weights))
                        .fold(Vec3::ZERO, |b, (&p, &w)| b + positions[p as usize] * w);
                    sum += bend * hinge.weights[(entry % 4) as usize];
                }
                gradient[i] = sum;
            }
            for (pin, target) in self.pins.iter().zip(targets) {
                let i = pin.particle as usize;
                gradient[i] += (positions[i] - *target) * pin_of[i];
            }

            // Each particle's own block, to precondition with.
            let mut diagonal = vec![Vec3::ZERO; count];
            for (t, element) in self.elements.iter().enumerate() {
                let state = &states[t];
                let (a, b) = (state.a, state.b);
                let [sxx, syy, sxy] = state.tension;
                for c in 0..3 {
                    let g = element.gradient[c];
                    let sg = Vec2::new(sxx * g.x + sxy * g.y, sxy * g.x + syy * g.y);
                    let along = a * g.x + b * g.y;
                    let spread = membrane.shear * g.length_squared() * state.firm;
                    let sum = Vec3::splat(g.dot(sg))
                        + (a * a + b * b) * spread
                        + along * along * ((membrane.shear + membrane.lame) * state.firm);
                    diagonal[self.triangles[t][c] as usize] += sum * element.area;
                }
            }
            for i in 0..count {
                let (n, k, along) = contact_of[i];
                diagonal[i] = diagonal[i] * (1.0 + rate)
                    + Vec3::splat(inertia[i] + pin_of[i] + along)
                    + n * n * (k - along);
            }

            // Conjugate gradients on (M/h² + (1 + rate)K + pins + contacts) d = -g.
            let dot =
                |a: &[Vec3], b: &[Vec3]| -> f32 { a.iter().zip(b).map(|(x, y)| x.dot(*y)).sum() };
            let mut delta = vec![Vec3::ZERO; count];
            let mut residual: Vec<Vec3> = gradient.iter().map(|g| -*g).collect();
            let precondition = |r: &[Vec3]| -> Vec<Vec3> {
                r.iter()
                    .zip(&diagonal)
                    .map(|(r, d)| *r / d.max(Vec3::splat(1.0e-12)))
                    .collect()
            };
            let mut z = precondition(&residual);
            let mut direction = z.clone();
            let mut rz = dot(&residual, &z);
            let goal = rz * CG_TOLERANCE * CG_TOLERANCE;
            let mut product = vec![Vec3::ZERO; count];
            // Stiff modes across a finer mesh take more iterations.
            let iterations = CG_ITERATIONS + (count as f32).sqrt() as usize / 2;
            for _ in 0..iterations {
                if rz <= goal || rz <= 0.0 {
                    break;
                }
                self.apply_stiffness(&states, &direction, &mut work, &mut product);
                for i in 0..count {
                    let (n, k, along) = contact_of[i];
                    product[i] = product[i] * (1.0 + rate)
                        + direction[i] * (inertia[i] + pin_of[i] + along)
                        + n * ((k - along) * n.dot(direction[i]));
                }
                let curvature = dot(&direction, &product);
                if curvature <= 0.0 {
                    break;
                }
                let alpha = rz / curvature;
                for i in 0..count {
                    delta[i] += direction[i] * alpha;
                    residual[i] -= product[i] * alpha;
                }
                z = precondition(&residual);
                let next = dot(&residual, &z);
                let beta = next / rz;
                rz = next;
                for i in 0..count {
                    direction[i] = z[i] + direction[i] * beta;
                }
            }
            for i in 0..count {
                self.positions[i] += delta[i];
            }
            if newton + 1 == NEWTON_STEPS {
                for contact in &mut self.contacts {
                    let i = contact.particle as usize;
                    let gap = contact.normal.dot(self.positions[i]) - contact.offset;
                    contact.depth = (-gap).max(0.0);
                    contact.force = contact_stiffness * contact.depth;
                    contact.pushed = contact.depth;
                }
            }
        }
    }

    /// `out = K v`: what displacements `v` change the elastic forces by, from
    /// the elements directly, with no matrix built. Each element works alone,
    /// then each particle gathers its corners, so both passes run in parallel
    /// and sum in one order.
    #[allow(clippy::needless_range_loop)]
    fn apply_stiffness(
        &self,
        states: &[ElementState],
        v: &[Vec3],
        work: &mut [[Vec3; 3]],
        out: &mut [Vec3],
    ) {
        let (elements, triangles) = (&self.elements, &self.triangles);
        let (shear, lame) = (self.shear, self.lame);
        work.par_iter_mut()
            .with_min_len(PARALLEL_CHUNK)
            .enumerate()
            .for_each(|(t, corners)| {
                let (element, state) = (&elements[t], &states[t]);
                let (da, db) = element.deformation(triangles[t].map(|c| v[c as usize]));
                let (a, b) = (state.a, state.b);
                let (exx, eyy, exy) = (a.dot(da), b.dot(db), 0.5 * (a.dot(db) + b.dot(da)));
                let trace = lame * (exx + eyy);
                let firm = state.firm;
                let (dxx, dyy, dxy) = (
                    firm * (2.0 * shear * exx + trace),
                    firm * (2.0 * shear * eyy + trace),
                    firm * 2.0 * shear * exy,
                );
                let [sxx, syy, sxy] = state.tension;
                for c in 0..3 {
                    let g = element.gradient[c];
                    let sg = Vec2::new(sxx * g.x + sxy * g.y, sxy * g.x + syy * g.y);
                    let dg = Vec2::new(dxx * g.x + dxy * g.y, dxy * g.x + dyy * g.y);
                    corners[c] = (da * sg.x + db * sg.y + a * dg.x + b * dg.y) * element.area;
                }
            });
        let (offsets, entries) = (&self.offsets, &self.entries);
        let (hinge_offsets, hinge_entries, hinges) =
            (&self.hinge_offsets, &self.hinge_entries, &self.hinges);
        let work = &*work;
        out.par_iter_mut()
            .with_min_len(PARALLEL_CHUNK)
            .enumerate()
            .for_each(|(i, value)| {
                let mut sum = Vec3::ZERO;
                for &entry in &entries[offsets[i] as usize..offsets[i + 1] as usize] {
                    sum += work[(entry / 3) as usize][(entry % 3) as usize];
                }
                for &entry in
                    &hinge_entries[hinge_offsets[i] as usize..hinge_offsets[i + 1] as usize]
                {
                    let hinge = &hinges[(entry / 4) as usize];
                    let bend = (hinge.particles.iter().zip(&hinge.weights))
                        .fold(Vec3::ZERO, |b, (&p, &w)| b + v[p as usize] * w);
                    sum += bend * hinge.weights[(entry % 4) as usize];
                }
                *value = sum;
            });
    }

    /// Average each element's largest principal stress, and tear those past
    /// the material's strength along their edge most nearly across that stress.
    fn tear(&mut self, membrane: &Membrane, blend: f32) {
        let strength = self.material.strength.break_load;
        let mut cuts = Vec::new();
        for (t, element) in self.elements.iter_mut().enumerate() {
            let tri = self.triangles[t];
            let (a, b) = element.deformation(tri.map(|c| self.positions[c as usize]));
            let ([sxx, syy, sxy], _) = stress(a, b, membrane);
            let (mean, half) = (0.5 * (sxx + syy), 0.5 * (sxx - syy));
            let radius = (half * half + sxy * sxy).sqrt();
            element.load += (mean + radius - element.load) * blend;
            if element.load < strength {
                continue;
            }
            let load = element.load;
            let pull = 0.5 * (2.0 * sxy).atan2(sxx - syy);
            let pull = Vec2::new(pull.cos(), pull.sin());
            let across = |corner: usize| {
                let (p, q) = (element.rest[corner], element.rest[(corner + 1) % 3]);
                (q - p).normalize_or_zero().dot(pull).abs()
            };
            let key = |corner: usize| {
                let (p, q) = (tri[corner], tri[(corner + 1) % 3]);
                edge(self.origin[p as usize], self.origin[q as usize])
            };
            let candidates = (0..3).filter(|&c| !self.cuts.contains(&key(c)));
            // Tears grow from their tips: an edge touching a cut is preferred.
            let touches = |c: usize| {
                let (p, q) = key(c);
                self.cut_ends.contains(&p) || self.cut_ends.contains(&q)
            };
            let score = |c: usize| across(c) - if touches(c) { 0.5 } else { 0.0 };
            if let Some(best) = candidates.min_by(|&x, &y| score(x).total_cmp(&score(y))) {
                cuts.push((load, t, key(best)));
            }
        }
        if cuts.is_empty() {
            return;
        }
        // The worst first, and only a few: each cut relieves its neighbours, so
        // a tear runs from where it started instead of shredding the sheet.
        cuts.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        cuts.truncate(1 + self.elements.len() / TEARS_PER_STEP);
        for &(_, t, _) in &cuts {
            self.elements[t].load = 0.0;
        }
        let mut cuts: Vec<_> = cuts.into_iter().map(|(_, _, cut)| cut).collect();
        cuts.sort_unstable();
        cuts.dedup();
        self.tears += cuts.len();
        for &(a, b) in &cuts {
            self.cut_ends.insert(a);
            self.cut_ends.insert(b);
        }
        self.cuts.extend(cuts);
        self.rebuild();
    }
}

/// The point of a triangle nearest `p` (Ericson, Real-Time Collision Detection).
fn closest_on_triangle(p: Vec3, [a, b, c]: [Vec3; 3]) -> Vec3 {
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denominator = 1.0 / (va + vb + vc);
    a + ab * (vb * denominator) + ac * (vc * denominator)
}

/// The axis of a matrix's symmetric part with the largest stretch.
fn principal_axis(m: Mat3) -> Vec3 {
    let symmetric = (m + m.transpose()) * 0.5;
    let mut axis = Vec3::new(0.577, 0.578, 0.577);
    for _ in 0..12 {
        let next = symmetric * axis;
        if next.length_squared() < 1.0e-20 {
            break;
        }
        axis = next.normalize();
    }
    axis
}

/// The offsets and values of `(particle, value)` pairs grouped by particle.
fn index_by_particle(
    count: usize,
    pairs: impl Iterator<Item = (u32, u32)> + Clone,
) -> (Vec<u32>, Vec<u32>) {
    let mut offsets = vec![0u32; count + 1];
    for (particle, _) in pairs.clone() {
        offsets[particle as usize + 1] += 1;
    }
    for i in 1..offsets.len() {
        offsets[i] += offsets[i - 1];
    }
    let mut next = offsets.clone();
    let mut values = vec![0; offsets[count] as usize];
    for (particle, value) in pairs {
        values[next[particle as usize] as usize] = value;
        next[particle as usize] += 1;
    }
    (offsets, values)
}

/// The curvature weights of the hinge on edge `x[0]`, `x[1]`, with opposite
/// corners `x[2]` and `x[3]` (Bergou et al. 2006).
fn hinge_weights(x: [Vec3; 4]) -> [f32; 4] {
    let cotangent = |a: Vec3, b: Vec3| a.dot(b) / a.cross(b).length().max(1.0e-9);
    let (e0, e1, e2) = (x[1] - x[0], x[2] - x[0], x[3] - x[0]);
    let (e3, e4) = (x[2] - x[1], x[3] - x[1]);
    let (c01, c02) = (cotangent(e0, e1), cotangent(e0, e2));
    let (c03, c04) = (cotangent(-e0, e3), cotangent(-e0, e4));
    [c03 + c04, c01 + c02, -c01 - c03, -c02 - c04]
}

fn outer(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::from_cols(a * b.x, a * b.y, a * b.z)
}

/// Find particles of soft bodies about to meet: a sheet and itself, and every
/// body and the others. Each pair becomes a plane for each particle, so the
/// solve can carry the load through the whole sheet: a push after it could not.
/// A sheet's neighbours at rest are left out, as they belong close; so is a
/// solid from itself, which its regions already hold.
pub(crate) fn find_pairs(bodies: &mut [SoftBody], gravity: Vec3, dt: f32) {
    for body in bodies.iter_mut() {
        body.planes.clear();
    }
    if bodies.iter().all(|b| b.asleep) || (bodies.len() < 2 && !bodies.iter().any(|b| b.sheet)) {
        return;
    }
    let widest = bodies.iter().fold(0.0f32, |w, b| w.max(b.radius));
    let fastest = (bodies.iter().flat_map(|b| b.velocities.iter()))
        .fold(0.0f32, |s, v| s.max(v.length_squared()))
        .sqrt();
    let size = 2.0 * widest + 2.0 * (fastest * dt + gravity.length() * dt * dt);
    if size <= 0.0 {
        return;
    }
    let cell = |p: Vec3| (p / size).floor().as_ivec3().to_array();
    let mut grid: HashMap<[i32; 3], Vec<(u32, u32)>> = HashMap::new();
    for (b, body) in bodies.iter().enumerate() {
        for (i, &p) in body.positions.iter().enumerate() {
            grid.entry(cell(p)).or_default().push((b as u32, i as u32));
        }
    }
    let mut cells: Vec<_> = grid.keys().copied().collect();
    cells.sort_unstable();
    for key in cells {
        for &(b, i) in &grid[&key] {
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let other = [key[0] + dx, key[1] + dy, key[2] + dz];
                        let Some(list) = grid.get(&other) else {
                            continue;
                        };
                        for &(c, j) in list {
                            if (c, j) > (b, i) {
                                add_pair(
                                    bodies,
                                    (b as usize, i as usize),
                                    (c as usize, j as usize),
                                    dt,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// If two particles will come closer than their radii this step, keep each on
/// its side of a plane between them, nearer the lighter.
fn add_pair(bodies: &mut [SoftBody], (b, i): (usize, usize), (c, j): (usize, usize), dt: f32) {
    let reach = bodies[b].radius + bodies[c].radius;
    if b == c {
        let body = &bodies[b];
        let (oi, oj) = (body.origin[i] as usize, body.origin[j] as usize);
        if !body.sheet || body.rest[oi].distance(body.rest[oj]) < NEIGHBOUR_REACH * reach {
            return;
        }
    }
    let (pi, pj) = (bodies[b].positions[i], bodies[c].positions[j]);
    let d = pi - pj;
    let distance = d.length();
    if distance < 1.0e-7 {
        return;
    }
    let n = d / distance;
    let closing = n.dot(bodies[b].velocities[i] - bodies[c].velocities[j]) * dt;
    if distance + closing >= reach {
        return;
    }
    let weight = |body: &SoftBody, k: usize| {
        if body.pinned[k] == NONE {
            1.0 / body.masses[k]
        } else {
            0.0
        }
    };
    let (wi, wj) = (weight(&bodies[b], i), weight(&bodies[c], j));
    if wi + wj <= 0.0 {
        return;
    }
    let split = pj + n * (distance * wj / (wi + wj));
    let middle = n.dot(split);
    let (ri, rj) = (reach * wi / (wi + wj), reach * wj / (wi + wj));
    bodies[b]
        .planes
        .push((i as u32, n, middle + ri.min(distance.max(ri * 0.5))));
    bodies[c]
        .planes
        .push((j as u32, -n, -middle + rj.min(distance.max(rj * 0.5))));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cut_across_a_sheet_splits_it_and_no_hinge_bridges_the_tear() {
        let center = Vec3::ZERO;
        let mut cloth = SoftBody::cloth(
            center,
            Quat::IDENTITY,
            [1.0, 1.0],
            [4, 4],
            SoftMaterial::cloth(),
        );
        assert_eq!(cloth.pieces(), 1);
        let mass = cloth.mass();
        // Every edge between the two middle rows of particles.
        let rows = |i: u32| i / 5;
        let mut cuts = Vec::new();
        for t in &cloth.triangles {
            for c in 0..3 {
                let (a, b) = (t[c], t[(c + 1) % 3]);
                if rows(a).min(rows(b)) == 1 && rows(a).max(rows(b)) == 2 {
                    cuts.push(edge(a, b));
                }
            }
        }
        cloth.cuts.extend(cuts);
        cloth.rebuild();
        assert_eq!(cloth.pieces(), 2);
        assert!(
            (cloth.mass() - mass).abs() < 1.0e-5,
            "mass is shared, not made"
        );
        for hinge in &cloth.hinges {
            let origins = hinge.particles.map(|p| cloth.origin[p as usize]);
            assert!(
                origins
                    .iter()
                    .all(|o| origins.iter().filter(|p| *p == o).count() == 1)
            );
        }
    }
}
