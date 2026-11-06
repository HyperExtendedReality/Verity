//! Deformable bodies: cloth and solids.
//!
//! Cloth is a triangle mesh of StVK membrane elements with quadratic bending
//! across its edges (Bergou et al. 2006), stepped implicitly by Newton's method
//! and conjugate gradients. It tears on the plane nearest failing, given its
//! threads, cutting through triangles from a tear's tips. A torn edge holds on as
//! a cohesive bond until it has spent the material's toughness.
//!
//! [`SoftBody::finite_elements`] makes a solid of tetrahedra instead, for flesh and
//! whatever must tear: stable neo-Hookean elements (Smith et al. 2018) in the same
//! implicit solve as cloth, which part along their faces.
//!
//! Solids are lattices solved by shape matching (Müller et al. 2005): each
//! overlapping region of particles is pulled toward the best-fit rotation of its
//! rest shape. That stays stiff in bending, which beams cannot do in a few
//! sweeps, and is stable at any stiffness. Plasticity updates the rest shape,
//! so past its yield strain a solid stays dented. A car's shell is a lattice
//! pinned to a rigid body, and rides it without solving until something comes
//! near. Outside every [`Interest`] a body only records [`Impact`]s, which dent it
//! when it is next seen.
//!
//! Soft bodies step after rigid bodies, seeing them at their new poses, and
//! give them exactly the momentum they took, through
//! [`crate::ExternalImpulse`], one step late.

mod tet;

use crate::{Collider, Ground, Strength};
use glam::{Mat3, Quat, Vec2, Vec3};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tet::{Tet, TetState};
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
/// Energy (J) below which a hit leaves no record: everyday bumps never mark a car.
const MIN_IMPACT_ENERGY: f32 = 100.0;
/// Finished impacts kept for a reader; older ones are dropped.
const MAX_IMPACTS: usize = 64;
/// Metres a body must pass beyond an interest before it is culled: one on the
/// edge does not flicker between states.
const INTEREST_MARGIN: f32 = 5.0;
/// Crush force on a car nobody sees, against one being solved.
/// shortcut: the solve's pins also slow the core, and this stands in for them,
/// measured on one wall crash; derive it from pin forces if unseen cars must
/// stop in the same place for other masses.
const UNSEEN_CRUSH: f32 = 4.0;
/// Metres a body's middle may stray from its anchor. Rounding at this distance
/// is about a micrometre: far below what a solve can feel.
const ANCHOR_REACH: f32 = 16.0;
/// Directions tried across half a turn for an element's weakest plane.
const FAILURE_SAMPLES: usize = 12;
/// Share of a cohesive bond's reach it stays elastic for, before it softens.
const BOND_ELASTIC: f32 = 0.25;
/// Fastest a tear runs (m/s), so that its speed does not depend on the mesh.
const TEAR_SPEED: f32 = 20.0;
/// Nearest an edge's end a tear may split it: closer leaves slivers.
const SPLIT_MARGIN: f32 = 0.2;
/// Share of its strength an element needs next to a tear's tip, where stress
/// gathers faster than a step's average of it can show.
const TIP_LOAD: f32 = 0.5;
/// How much of its old heading a tear keeps, against where the stress points.
const TEAR_MEMORY: f32 = 0.5;
/// Furthest a solid's particle may move in one Newton step, in cells. A crushed
/// element's stiffness says little of what is far from it, and Newton's step
/// there can run to infinity.
const TET_TRUST: f32 = 1.0;
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
    /// A sheet's strength across its warp, as a share of `break_load` along it:
    /// below 1, tears run along the warp, as in a grain.
    pub grain: f32,
    /// Extra strength on the bias, where a tear must break two sets of yarns at
    /// once: 0 for felt, paper or leather, 1 for a plain weave, whose tears run
    /// along its threads.
    pub weave: f32,
    /// Work to open a metre of a sheet's tear (N), or a square metre of a
    /// solid's (J/m²). A torn edge holds on as a softening bond until this is
    /// spent, so tears open gradually, and cost the same however fine the mesh.
    /// 0 parts at once. Keep it in proportion to strength: the zone that holds is
    /// `stiffness * toughness / break_load²` long.
    pub toughness: f32,
    /// Velocity lost per second.
    pub damping: f32,
    pub friction: f32,
    /// Distance kept from what it touches (m).
    pub thickness: f32,
}

impl SoftMaterial {
    /// Strength against tearing on a plane whose normal is at this angle from the
    /// warp. Any plane that crosses yarns must break them, so strength climbs
    /// sharply off a thread's line, not gradually, and on the bias a weave
    /// must break both sets.
    fn strength_at(&self, cos: f32, sin: f32) -> f32 {
        let (cos, sin) = (cos.abs(), sin.abs());
        let yarns = self.grain + (1.0 - self.grain) * cos;
        self.strength.break_load * yarns * (1.0 + self.weave * 2.0 * sin * cos)
    }

    /// Where an element is nearest failing, under stress `[xx, yy, xy]` with
    /// its warp along `warp`: how near, and the normal of the plane it would
    /// part on.
    fn failure(&self, [sxx, syy, sxy]: [f32; 3], warp: Vec2) -> (f32, Vec2) {
        let (mean, half) = (0.5 * (sxx + syy), 0.5 * (sxx - syy));
        let high = mean + (half * half + sxy * sxy).sqrt();
        let angle = 0.5 * (2.0 * sxy).atan2(sxx - syy);
        let principal = Vec2::new(angle.cos(), angle.sin());
        if self.grain == 1.0 && self.weave == 0.0 {
            return (high / self.strength.break_load, principal);
        }
        // Nowhere near even the weakest plane, as most of a sheet is most of the
        // time: this bound is all anything will read.
        let bound = high / (self.strength.break_load * self.grain);
        if bound < 0.5 * TIP_LOAD {
            return (bound, principal);
        }
        let on = |n: Vec2| {
            let traction = n.x * n.x * sxx + n.y * n.y * syy + 2.0 * n.x * n.y * sxy;
            traction / self.strength_at(n.dot(warp), n.perp_dot(warp))
        };
        let mut worst = (on(principal), principal);
        for k in 0..FAILURE_SAMPLES {
            let turn = std::f32::consts::PI * k as f32 / FAILURE_SAMPLES as f32;
            let n = Vec2::from_angle(turn).rotate(warp);
            let ratio = on(n);
            if ratio > worst.0 {
                worst = (ratio, n);
            }
        }
        worst
    }

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
            grain: 1.0,
            weave: 1.0,
            toughness: 150.0,
            damping: 1.0,
            friction: 0.5,
            thickness: 0.02,
        }
    }

    /// Muscle, for [`SoftBody::finite_elements`]: soft, nearly incompressible, and
    /// a third as strong across its fibres, which lie along the warp, so it
    /// tears between them, and stringily.
    pub fn flesh() -> Self {
        Self {
            density: 1060.0,
            stiffness: 1.0e5,
            poisson: 0.45,
            compression: 1.0,
            bending: 0.0,
            rigidity: 0.0,
            // Parts near 80% strain along its fibres, and 25% across them.
            strength: Strength::brittle(8.0e4),
            grain: 0.3,
            weave: 0.0,
            toughness: 300.0,
            damping: 4.0,
            friction: 0.6,
            thickness: 0.005,
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
            grain: 1.0,
            weave: 0.0,
            toughness: 0.0,
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
            grain: 1.0,
            weave: 0.0,
            toughness: 0.0,
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
            grain: 1.0,
            weave: 0.0,
            toughness: 0.0,
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
            grain: 1.0,
            weave: 0.0,
            toughness: 0.0,
            damping: 1.0,
            friction: 0.9,
            thickness: 0.01,
        }
    }
}

/// Where a tear goes next.
#[derive(Debug, Clone, Copy)]
enum Cut {
    /// Along an existing edge, by `origin` indices.
    Edge((u32, u32)),
    /// Through a triangle from its corner, to a share of the way along the
    /// opposite edge.
    Through { corner: usize, share: f32 },
}

/// A tear's next step: how urgent, from which element, and which tip it grows.
#[derive(Debug, Clone, Copy)]
struct Step {
    load: f32,
    element: usize,
    cut: Cut,
    tip: Option<u32>,
}

/// What holds a tear's two sides together while it opens, between two copies
/// of a particle: a spring that softens to nothing as it spends its energy.
#[derive(Debug, Clone, Copy)]
struct Bond {
    particles: [u32; 2],
    /// Greatest force (N), and the opening at which it is gone (m).
    peak: f32,
    reach: f32,
    /// Widest it has been, which it cannot recover from.
    opened: f32,
    /// Force per metre at that opening: the secant, as a softening spring's
    /// true stiffness is negative, which conjugate gradients cannot use.
    stiffness: f32,
}

impl Bond {
    fn new(particles: [u32; 2], peak: f32, reach: f32) -> Self {
        Self {
            particles,
            peak,
            reach,
            opened: 0.0,
            stiffness: peak / (BOND_ELASTIC * reach),
        }
    }

    /// Note an opening. False once it has parted.
    fn open(&mut self, width: f32) -> bool {
        self.opened = self.opened.max(width);
        let elastic = BOND_ELASTIC * self.reach;
        if self.opened > elastic {
            let left = (self.reach - self.opened) / (self.reach - elastic);
            self.stiffness = self.peak * left.max(0.0) / self.opened;
        }
        self.opened < self.reach
    }
}

/// One triangle of a sheet: a StVK membrane, in its rest shape.
#[derive(Debug, Clone, Copy)]
struct Element {
    /// Each corner's gradient of the deformation, from the rest shape.
    gradient: [Vec2; 3],
    area: f32,
    /// The warp, in the triangle's plane.
    warp: Vec2,
    /// How near its weakest plane is to failing, averaged: 1 fails.
    load: f32,
}

impl Element {
    fn new(points: [Vec3; 3], warp: Vec3) -> Self {
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
            area: determinant.abs() * 0.5,
            warp: Vec2::new(u.dot(warp), v.dot(warp)).normalize_or(Vec2::X),
            load: 0.0,
        }
    }

    /// A direction in this element's plane, as a vector of the shape `points`
    /// it was built from.
    fn lift(direction: Vec2, points: [Vec3; 3]) -> Vec3 {
        let (e1, e2) = (points[1] - points[0], points[2] - points[0]);
        let u = e1.normalize_or_zero();
        let v = (e2 - u * u.dot(e2)).normalize_or_zero();
        u * direction.x + v * direction.y
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
    /// The cells that hold matter.
    filled: Vec<[u32; 3]>,
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

/// A plane from another soft body's particle, for one step.
#[derive(Debug, Clone, Copy)]
struct Plane {
    particle: u32,
    normal: Vec3,
    offset: f32,
    /// How fast the two close (m/s), and the other body's mass (kg).
    closing: f32,
    mass: f32,
}

/// A contact plane for one step: `normal · x >= offset`.
#[derive(Debug, Clone, Copy)]
struct Contact {
    particle: u32,
    obstacle: u32,
    normal: Vec3,
    offset: f32,
    /// How fast it closed when it began (m/s), and what it met's mass (kg),
    /// 0 for something immovable.
    closing: f32,
    partner: f32,
    /// Force on the particle along the normal in the last sweep (N).
    force: f32,
    /// Total distance it pushed the particle out, and how deep it last was.
    pushed: f32,
    depth: f32,
}

/// Ground's size as an obstacle: reachable from anywhere, and finite so a
/// product with a zero speed stays zero.
const UNBOUNDED: f32 = 1.0e9;

/// A rigid body as a soft body sees it.
#[derive(Clone)]
pub(crate) struct Obstacle {
    pub shape: Collider,
    /// What [`Collider::Ground`] is.
    pub ground: Option<Arc<dyn Ground>>,
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
            Collider::Ground => UNBOUNDED,
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
            Collider::Ground => {
                let ground = self.ground.as_ref()?;
                let distance = ground.distance(local);
                if distance > reach {
                    return None;
                }
                let normal = ground.normal(local);
                (local - normal * distance, normal)
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
            // `narrowest` keeps paths from sweeping: `nearest` works from inside.
            Collider::Ground => return None,
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
            Collider::Ground => UNBOUNDED,
        }
    }

    fn velocity_at(&self, point: Vec3) -> Vec3 {
        self.velocity + self.spin.cross(point - self.position)
    }
}

/// One collision a soft body took, in its carrier's frame, or the world's if it
/// has none. A few bytes describe a crash: send these rather than vertices, and
/// give them to [`SoftBody::apply_impact`] on another machine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Impact {
    pub point: Vec3,
    /// Into the body.
    pub normal: Vec3,
    /// Energy of the collision (J).
    pub energy: f32,
    /// Extent of the contact (m).
    pub radius: f32,
}

/// A place soft bodies are worth solving: around a player, or in view. A body
/// outside every interest follows its carrier or holds still, and only records
/// what hits it until it is seen again.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interest {
    pub center: Vec3,
    pub radius: f32,
    /// A view's planes `(normal, offset)`, with inside at `normal · x + offset >= 0`.
    pub frustum: Option<[(Vec3, f32); 6]>,
}

impl Interest {
    fn contains(&self, center: Vec3, radius: f32) -> bool {
        let near = center.distance(self.center) <= self.radius + radius;
        near || (self.frustum)
            .is_some_and(|planes| planes.iter().all(|&(n, d)| n.dot(center) + d >= -radius))
    }
}

/// A settled solid carried by one rigid body: it follows that body's pose
/// without solving, until something comes near. A car costs this while driving.
#[derive(Debug, Clone)]
struct Ride {
    owner: EntityHandle,
    /// Each particle in the owner's frame.
    local: Vec<Vec3>,
    /// The owner's velocity last step, for the momentum it must still feel.
    velocity: Vec3,
    /// The owner's rotation when this began and now: regions' rotations are
    /// not tracked while riding, and must catch up.
    start: Quat,
    rotation: Quat,
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
    /// Relative to `anchor`. Solved in world space, rounding far from the origin
    /// acts as a steady force: a lattice 2 km out pushed its car sideways at 5 m/s.
    positions: Vec<Vec3>,
    /// The same in world space, as of the end of the last step.
    world: Vec<Vec3>,
    anchor: Vec3,
    /// `world` has moved on without `positions`, as it does while riding.
    stale: bool,
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
    planes: Vec<Plane>,
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
    /// A sheet's warp, at rest.
    warp: Vec3,
    bonds: Vec<Bond>,
    tear_work: f32,
    /// A finite element solid's elements, and each particle's as
    /// `tet * 4 + corner` in `tet_entries[tet_offsets[i]..tet_offsets[i + 1]]`.
    tets: Vec<Tet>,
    tet_offsets: Vec<u32>,
    tet_entries: Vec<u32>,
    /// Faces between two elements that hold, and those torn, by `origin` indices.
    uncut: HashSet<[u32; 3]>,
    cut_faces: Vec<[u32; 3]>,
    /// Original edges that have torn, by `origin` indices.
    cuts: HashSet<(u32, u32)>,
    /// How many cuts meet at each particle, by `origin`. A tear's tips, where it
    /// grows from, are where only one does.
    cut_ends: HashMap<u32, u32>,
    /// Which way each tip was heading when it got there, at rest.
    headings: HashMap<u32, Vec3>,
    /// Each particle's texture coordinates, for a sheet.
    uvs: Vec<Vec2>,
    tears: usize,
    rest_time: f32,
    asleep: bool,
    ride: Option<Ride>,
    /// Outside every interest; `frozen` once it has stopped there, uncarried.
    culled: bool,
    frozen: bool,
    revision: u64,
    /// The collision under way, finished ones, and those not yet in the shape.
    impact: Option<Impact>,
    impacts: Vec<Impact>,
    pending: Vec<Impact>,
    contacts: Vec<Contact>,
}

/// The root of `i` in a union-find of a particle's surroundings.
fn fan_root(group: &mut [usize], mut i: usize) -> usize {
    while group[i] != i {
        group[i] = group[group[i]];
        i = group[i];
    }
    i
}

fn bounds_of(points: &[Vec3]) -> (Vec3, Vec3) {
    let first = points.first().copied().unwrap_or(Vec3::ZERO);
    (points.iter()).fold((first, first), |(min, max), &p| (min.min(p), max.max(p)))
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
            world: positions.clone(),
            anchor: Vec3::ZERO,
            stale: false,
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
            warp: Vec3::X,
            bonds: Vec::new(),
            tear_work: 0.0,
            tets: Vec::new(),
            tet_offsets: Vec::new(),
            tet_entries: Vec::new(),
            uncut: HashSet::new(),
            cut_faces: Vec::new(),
            cuts: HashSet::new(),
            cut_ends: HashMap::new(),
            headings: HashMap::new(),
            uvs: Vec::new(),
            tears: 0,
            rest_time: 0.0,
            asleep: false,
            ride: None,
            culled: false,
            frozen: false,
            revision: 0,
            impact: None,
            impacts: Vec::new(),
            pending: Vec::new(),
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
        let mut uvs = Vec::with_capacity((nx * ny) as usize);
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
                positions.push(rotation * local);
                uvs.push(Vec2::new(local.x / size[0] + 0.5, local.y / size[1] + 0.5));
            }
        }
        let count = (nx * ny) as usize;
        let mut body = Self::new(positions, vec![0.0; count], material);
        body.uvs = uvs;
        body.settle_at(center);
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
        body.warp = rotation * Vec3::X;
        body.radius = 0.45 * dx.min(dy);
        for &[a, b, c] in &body.triangles {
            let element = Element::new([a, b, c].map(|i| body.rest[i as usize]), body.warp);
            for i in [a, b, c] {
                body.masses[i as usize] += material.density * element.area / 3.0;
            }
            body.elements.push(element);
        }
        body.rebuild();
        body
    }

    /// Start a tear at the edge nearest this world point, as a blade or a bullet
    /// would: under load it runs from there. Does nothing to a solid.
    pub fn nick(&mut self, at: Vec3) {
        if !self.tets.is_empty() {
            return;
        }
        let middle = |&(a, b): &(u32, u32)| {
            0.5 * (self.world[a as usize] + self.world[b as usize]).distance_squared(2.0 * at)
        };
        let nearest = (self.triangles.iter())
            .flat_map(|t| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])])
            .filter(|&(a, b)| {
                !self
                    .cuts
                    .contains(&edge(self.origin[a as usize], self.origin[b as usize]))
            })
            .min_by(|x, y| middle(x).total_cmp(&middle(y)));
        let Some((a, b)) = nearest else {
            return;
        };
        let (a, b) = (self.origin[a as usize], self.origin[b as usize]);
        self.cuts.insert(edge(a, b));
        *self.cut_ends.entry(a).or_default() += 1;
        *self.cut_ends.entry(b).or_default() += 1;
        self.tears += 1;
        self.rebuild();
    }

    /// Lay a sheet's warp along this world direction, as built. It starts along
    /// the sheet's local X.
    pub fn set_warp(&mut self, direction: Vec3) {
        self.warp = direction;
        for (element, triangle) in self.elements.iter_mut().zip(&self.triangles) {
            let corners = triangle.map(|i| self.rest[self.origin[i as usize] as usize]);
            element.warp = Element::new(corners, direction).warp;
        }
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
        let mut filled = Vec::new();
        for z in 0..cells[2] {
            for y in 0..cells[1] {
                for x in 0..cells[0] {
                    if !filled_at(x as i64, y as i64, z as i64) {
                        continue;
                    }
                    filled.push([x, y, z]);
                    for corner in 0..8u32 {
                        let (cx, cy, cz) =
                            (x + (corner & 1), y + (corner >> 1 & 1), z + (corner >> 2));
                        let i = *index.entry(grid(cx, cy, cz)).or_insert_with(|| {
                            let local =
                                Vec3::new(cx as f32, cy as f32, cz as f32) * step - half_extents;
                            positions.push(rotation * local);
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
        body.settle_at(center);
        body.radius = 0.5 * step.min_element();
        body.grid = Some(Grid {
            center,
            rotation,
            half_extents,
            step,
            cells,
            filled,
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

    /// What a contact or pin holds a sheet or solid's surface with (N/m).
    fn surface_stiffness(&self) -> f32 {
        let per_metre = if self.tets.is_empty() {
            1.0
        } else {
            2.0 * self.radius
        };
        self.material.stiffness * per_metre
    }

    /// Tie points, such as a detailed mesh's vertices, to this solid as built:
    /// [`SoftBody::deformed`] then says where they have gone. Each follows the
    /// corners of its grid cell, or the particle nearest it outside the solid.
    /// Corners the solid lacks hold `u32::MAX`, bit 0 of a corner's place being
    /// along the grid's x, 1 its y and 2 its z.
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
                // Past the lattice's edge the nearest cell extrapolates: clamping
                // would flatten whatever sticks out, such as a car's mirrors.
                let cell = (local + grid.half_extents) / grid.step;
                let base = cell.floor().clamp(Vec3::ZERO, limit - 1.0);
                let t = cell - base;
                let mut weights = [(NONE, 0.0); 8];
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
                if total.abs() > 1.0e-6 {
                    for weight in &mut weights {
                        weight.1 /= total;
                    }
                } else {
                    weights = [(NONE, 0.0); 8];
                    let from = point - grid.center;
                    let nearest = (0..self.rest.len() as u32)
                        .min_by(|&a, &b| {
                            let d = |i: u32| self.rest[i as usize].distance_squared(from);
                            d(a).total_cmp(&d(b))
                        })
                        .unwrap_or(0);
                    weights[0] = (nearest, 1.0);
                }
                weights
            })
            .collect()
    }

    /// Takes a world direction as this solid was built to grid cell units, for
    /// transforming what [`SoftBody::embed`] tied to it.
    pub fn cell_basis(&self) -> Option<Mat3> {
        let grid = self.grid.as_ref()?;
        let cells = Mat3::from_diagonal(grid.step.recip());
        Some(cells * Mat3::from_quat(grid.rotation).transpose())
    }

    /// Changes whenever the shape may have, so a renderer can skip the rest:
    /// riding moves a body without changing it.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// How far an impact crushes this body: its energy spent at `yield_load`
    /// over its area, no deeper than it is wide.
    pub fn crush_depth(&self, impact: &Impact) -> f32 {
        let radius = impact.radius;
        let force = self.material.strength.yield_load * std::f32::consts::PI * radius * radius;
        (impact.energy / force).min(radius)
    }

    /// Where points tied by [`SoftBody::embed`] are now.
    pub fn deformed(&self, embedding: &[[(u32, f32); 8]]) -> Vec<Vec3> {
        (embedding.iter())
            .map(|weights| {
                (weights.iter())
                    .filter(|&&(i, _)| i != NONE)
                    .fold(Vec3::ZERO, |sum, &(i, w)| sum + self.world[i as usize] * w)
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
        self.index_corners();

        let count = self.positions.len();
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
        self.index_tets();
    }

    /// Index each particle's triangle corners.
    fn index_corners(&mut self) {
        let corners = (self.triangles.iter().enumerate())
            .flat_map(|(t, tri)| (0..3).map(move |c| (tri[c], t as u32 * 3 + c as u32)));
        (self.offsets, self.entries) = index_by_particle(self.positions.len(), corners);
    }

    /// Add a particle `share` of the way from `p` to `q`, splitting the triangles
    /// either side of that edge. On a linear element a point along an edge has
    /// exactly that strain already, so nothing jumps. Hinges wait for `rebuild`.
    fn split_edge(&mut self, p: u32, q: u32, share: f32) -> u32 {
        let (pi, qi) = (p as usize, q as usize);
        let new = self.positions.len() as u32;
        let blend = |a: Vec3, b: Vec3| a + (b - a) * share;
        let (op, oq) = (self.origin[pi] as usize, self.origin[qi] as usize);
        self.origin.push(self.rest.len() as u32);
        self.rest.push(blend(self.rest[op], self.rest[oq]));
        self.positions
            .push(blend(self.positions[pi], self.positions[qi]));
        self.velocities
            .push(blend(self.velocities[pi], self.velocities[qi]));
        self.correction
            .push(blend(self.correction[pi], self.correction[qi]));
        self.masses.push(0.0);
        self.pinned.push(NONE);
        if !self.uvs.is_empty() {
            let uv = self.uvs[pi].lerp(self.uvs[qi], share);
            self.uvs.push(uv);
        }
        let around: Vec<usize> = (self.entries
            [self.offsets[pi] as usize..self.offsets[pi + 1] as usize])
            .iter()
            .map(|&entry| (entry / 3) as usize)
            .filter(|&t| self.triangles[t].contains(&q))
            .collect();
        for t in around {
            let tri = self.triangles[t];
            let off = (0..3)
                .find(|&c| tri[c] != p && tri[c] != q)
                .expect("a triangle has a corner off any edge");
            // From there the winding runs along the edge.
            let (far, x, y) = (tri[off], tri[(off + 1) % 3], tri[(off + 2) % 3]);
            let load = self.elements[t].load;
            let area = self.elements[t].area * self.material.density;
            // Lumped mass: each child's thirds, which sum to what its parent had.
            let to_p = (1.0 - share) * area / 3.0;
            let to_q = share * area / 3.0;
            self.masses[pi] -= to_p;
            self.masses[qi] -= to_q;
            self.masses[new as usize] += area / 3.0;
            let halves = [[far, x, new], [far, new, y]];
            let elements = halves.map(|half| {
                let corners = half.map(|i| self.rest[self.origin[i as usize] as usize]);
                Element {
                    load,
                    ..Element::new(corners, self.warp)
                }
            });
            self.triangles[t] = halves[0];
            self.elements[t] = elements[0];
            self.triangles.push(halves[1]);
            self.elements.push(elements[1]);
        }
        self.index_corners();
        new
    }

    /// A second particle where `v` is, of this mass, held to it by a bond.
    fn copy_particle(&mut self, v: u32, mass: f32) -> u32 {
        let copy = self.positions.len() as u32;
        self.positions.push(self.positions[v as usize]);
        self.velocities.push(self.velocities[v as usize]);
        self.correction.push(self.correction[v as usize]);
        self.masses.push(mass);
        self.origin.push(self.origin[v as usize]);
        self.pinned.push(NONE);
        if !self.uvs.is_empty() {
            self.uvs.push(self.uvs[v as usize]);
        }
        let m = self.material;
        if m.toughness > 0.0 && m.strength.break_load.is_finite() {
            // Each particle along a tear stands for about a cell of it: of its
            // length in a sheet, of its area in a solid.
            let cell = self.radius / 0.45;
            let area = if self.tets.is_empty() {
                cell
            } else {
                4.0 * self.radius * self.radius
            };
            // Tears run on the weakest plane: a bond any stronger than that would
            // hold shut a crack between fibres.
            let strength = m.strength.break_load * m.grain;
            let reach = 2.0 * m.toughness / strength;
            self.bonds
                .push(Bond::new([v, copy], strength * area, reach));
        }
        copy
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
                            let (a, b) = (fan_root(&mut group, i), fan_root(&mut group, j));
                            group[a] = b;
                        }
                    }
                }
            }
            let roots: Vec<usize> = (0..fan.len()).map(|i| fan_root(&mut group, i)).collect();
            let mut distinct = roots.clone();
            distinct.sort_unstable();
            distinct.dedup();
            if distinct.len() < 2 {
                continue;
            }
            let share = self.masses[v as usize] / distinct.len() as f32;
            self.masses[v as usize] = share;
            for &root in &distinct[1..] {
                let copy = self.copy_particle(v, share);
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
        let anchor = self.anchor;
        self.pin_with(&mut select, None, |p| p + anchor, strength)
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
            if select(self.world[i]) && self.pinned[i] == NONE {
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

    #[allow(clippy::misnamed_getters)] // World space: the field is relative to the anchor.
    pub fn positions(&self) -> &[Vec3] {
        &self.world
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
        let regions = self.regions.iter().map(|r| r.strain);
        regions
            .chain(self.tets.iter().map(|t| t.strain))
            .fold(0.0, f32::max)
    }
    /// Each particle's texture coordinates, for a sheet: tearing adds particles,
    /// and these keep the cloth's pattern in place across them.
    pub fn uvs(&self) -> &[Vec2] {
        &self.uvs
    }

    /// Energy spent opening tears so far (J).
    pub fn tear_work(&self) -> f32 {
        self.tear_work
    }

    /// Each torn edge's ends, around where the body was built: for a solid,
    /// those of each torn face.
    pub fn tear_path(&self) -> Vec<(Vec3, Vec3)> {
        let mut path: Vec<_> = self.cuts.iter().copied().collect();
        path.sort_unstable();
        let faces = self.cut_faces.iter();
        path.extend(faces.flat_map(|&[a, b, c]| [(a, b), (b, c), (c, a)]));
        (path.iter())
            .map(|&(a, b)| (self.rest[a as usize], self.rest[b as usize]))
            .collect()
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
        for bond in &self.bonds {
            let [a, b] = bond.particles.map(|p| find(&mut parent, p));
            parent[a as usize] = b;
        }
        for tet in &self.tets {
            for &p in &tet.particles[1..] {
                let (a, b) = (find(&mut parent, tet.particles[0]), find(&mut parent, p));
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
        bounds_of(&self.positions)
    }

    fn world_bounds(&self) -> (Vec3, Vec3) {
        bounds_of(&self.world)
    }

    /// Anchor a body built around the origin at `center`. Built in world space,
    /// its rest shape would keep the rounding of wherever it was made.
    fn settle_at(&mut self, center: Vec3) {
        self.anchor = center;
        self.publish();
    }

    pub(crate) fn anchor(&self) -> Vec3 {
        self.anchor
    }

    /// Catch `positions` up with `world`, and move the anchor if the body left it.
    fn rebase(&mut self) {
        if std::mem::take(&mut self.stale) {
            for (p, &w) in self.positions.iter_mut().zip(&self.world) {
                *p = w - self.anchor;
            }
        }
        let (min, max) = self.bounds();
        let middle = 0.5 * (min + max);
        if middle.length_squared() > ANCHOR_REACH * ANCHOR_REACH {
            let anchor = self.anchor + middle;
            // What the anchor really moved by, after its own rounding.
            let moved = anchor - self.anchor;
            self.anchor = anchor;
            self.positions.iter_mut().for_each(|p| *p -= moved);
        }
    }

    /// Bring `world` up to date with `positions`, which tearing may have added to.
    fn publish(&mut self) {
        self.world.clear();
        (self.world).extend(self.positions.iter().map(|&p| p + self.anchor));
    }

    /// The rigid bodies as this body's solve sees them: around its anchor.
    fn around_anchor(&self, obstacles: &[Obstacle]) -> Vec<Obstacle> {
        (obstacles.iter())
            .map(|o| Obstacle {
                position: o.position - self.anchor,
                ..o.clone()
            })
            .collect()
    }

    /// Obstacles a body within these bounds could reach this step, and how far
    /// it reaches.
    fn near(
        &self,
        (min, max): (Vec3, Vec3),
        dt: f32,
        gravity: Vec3,
        obstacles: &[Obstacle],
    ) -> (Vec<u32>, f32) {
        let travel = self
            .velocities
            .iter()
            .fold(0.0f32, |s, v| s.max(v.length_squared()))
            .sqrt()
            * dt;
        let reach = self.material.thickness + travel + gravity.length() * dt * dt;
        let near = (obstacles.iter().enumerate())
            .filter(|(_, o)| {
                let r = o.radius() + reach;
                let closest = o.position.clamp(min, max);
                closest.distance_squared(o.position) <= r * r
            })
            .map(|(i, _)| i as u32)
            .collect();
        (near, reach)
    }

    /// Whether an awake rigid body holds any pin.
    fn carried(
        &self,
        obstacles: &[Obstacle],
        pin_target: &impl Fn(EntityHandle, Vec3) -> Option<(Vec3, u32)>,
    ) -> bool {
        self.pins.iter().any(|pin| {
            let held = pin.body.and_then(|body| pin_target(body, pin.point));
            held.is_some_and(|(_, owner)| obstacles[owner as usize].awake)
        })
    }

    /// Advance by `dt`. `pin_target` gives a pinned body's world point for a point
    /// on it; `None` drops the pin. Adds impulses for rigid bodies to `reactions`.
    pub(crate) fn step(
        &mut self,
        dt: f32,
        gravity: Vec3,
        obstacles: &[Obstacle],
        pin_target: impl Fn(EntityHandle, Vec3) -> Option<(Vec3, u32)>,
        reactions: &mut Vec<Reaction>,
    ) {
        if self.culled {
            if self.ride.is_some() {
                self.follow_unseen(dt, gravity, obstacles, &pin_target, reactions);
                return;
            }
            if self.frozen {
                return;
            }
        } else if !self.pending.is_empty() {
            self.rebase();
            let around = self.around_anchor(obstacles);
            let (position, rotation, _) = self.frame(&around, &pin_target);
            for impact in std::mem::take(&mut self.pending) {
                self.dent(impact, position, rotation);
            }
            self.publish();
        }
        if self.ride.is_some() && self.follow(dt, gravity, obstacles, &pin_target, reactions) {
            return;
        }
        if self.asleep {
            let (near, _) = self.near(self.world_bounds(), dt, gravity, obstacles);
            let stirred = near.iter().any(|&i| obstacles[i as usize].awake);
            if !stirred && !self.carried(obstacles, &pin_target) {
                return;
            }
            self.asleep = false;
            self.rest_time = 0.0;
        }
        self.rebase();
        let around = self.around_anchor(obstacles);
        let first = reactions.len();
        self.solve(dt, gravity, &around, pin_target, reactions);
        for reaction in &mut reactions[first..] {
            reaction.point += self.anchor;
        }
        self.publish();
    }

    /// One step's solve, all of it relative to `anchor`: `obstacles` too.
    #[allow(clippy::needless_range_loop)]
    fn solve(
        &mut self,
        dt: f32,
        gravity: Vec3,
        obstacles: &[Obstacle],
        pin_target: impl Fn(EntityHandle, Vec3) -> Option<(Vec3, u32)>,
        reactions: &mut Vec<Reaction>,
    ) {
        let (near, reach) = self.near(self.bounds(), dt, gravity, obstacles);
        let carried = self.carried(obstacles, &pin_target);
        self.revision += 1;
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
                None => pin.point - self.anchor,
                Some(body) => match pin_target(body, pin.point) {
                    Some((_, owner)) => {
                        owners[i] = owner;
                        // From the anchored pose: the world point is already rounded.
                        let holder = &obstacles[owner as usize];
                        holder.position + holder.rotation * pin.point
                    }
                    None => {
                        gone[i] = true;
                        pin.point - self.anchor
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
                    let relative = self.velocities[i] - obstacle.velocity_at(start[i]);
                    self.contacts.push(Contact {
                        particle: i as u32,
                        obstacle: o,
                        normal,
                        offset: normal.dot(surface) + thickness,
                        closing: -relative.dot(normal),
                        partner: obstacle.mass,
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
        for plane in &self.planes {
            self.contacts.push(Contact {
                particle: plane.particle,
                obstacle: NONE,
                normal: plane.normal,
                offset: plane.offset - plane.normal.dot(self.anchor),
                closing: plane.closing,
                partner: plane.mass,
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
                    self.surface_stiffness().max(own)
                } else {
                    own
                };
                // An immovable body has no mass to ring against.
                if owners[i] == NONE || obstacles[owners[i] as usize].mass <= 0.0 {
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
            self.lame = if self.tets.is_empty() {
                membrane.lame
            } else {
                // The energy's own constant: at rest it acts as this less the shear.
                m.stiffness * m.poisson / ((1.0 + m.poisson) * (1.0 - 2.0 * m.poisson))
                    + membrane.shear
            };
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
            let touches: Vec<_> = (self.contacts.iter())
                .filter(|c| c.pushed > 0.0 && c.obstacle != NONE)
                .map(|c| (self.positions[c.particle as usize], c.normal))
                .collect();
            self.crush(
                &touches,
                1.0,
                &obstacles[owner as usize],
                owner,
                dt,
                reactions,
            );
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
            self.tear(&membrane, blend, dt);
        } else {
            let frame = self.frame(obstacles, &pin_target);
            let hits: Vec<_> = (self.contacts.iter())
                .filter(|c| c.pushed > 0.0)
                .map(|c| {
                    let point = self.positions[c.particle as usize];
                    (point, c.normal, c.closing, c.partner)
                })
                .collect();
            self.note_hit(&hits, frame);
        }

        // Carried, a solid is at rest when it stops moving against its carrier.
        let carrier = (self.carrier())
            .filter(|_| solid && !released && self.contacts.is_empty())
            .and_then(|entity| Some((entity, pin_target(entity, Vec3::ZERO)?.1)));
        let speed = carrier.map_or(fastest.sqrt(), |(_, owner)| {
            let body = &obstacles[owner as usize];
            (self.positions.iter().zip(&self.velocities))
                .fold(0.0f32, |s, (&p, &v)| {
                    s.max((v - body.velocity_at(p)).length_squared())
                })
                .sqrt()
        });
        if speed < SLEEP_SPEED && !released && (!carried || carrier.is_some()) {
            self.rest_time += dt;
            if self.rest_time >= TIME_TO_SLEEP {
                match carrier {
                    Some((entity, owner)) => self.start_riding(entity, &obstacles[owner as usize]),
                    None => self.asleep = true,
                }
            }
        } else {
            self.rest_time = 0.0;
        }
        // Just left every interest: stop here, at a pose that matches the step.
        if self.culled && self.ride.is_none() {
            let held = self.carrier().and_then(|entity| {
                let (_, owner) = pin_target(entity, Vec3::ZERO)?;
                Some((entity, owner))
            });
            match held {
                Some((entity, owner)) => self.start_riding(entity, &obstacles[owner as usize]),
                None => {
                    self.frozen = true;
                    self.velocities.fill(Vec3::ZERO);
                }
            }
        }
    }

    /// Slow the carrier by this solid's crush strength over `touches`, each a
    /// point and its normal.
    fn crush(
        &self,
        touches: &[(Vec3, Vec3)],
        scale: f32,
        carried: &Obstacle,
        owner: u32,
        dt: f32,
        reactions: &mut Vec<Reaction>,
    ) {
        if touches.is_empty() {
            return;
        }
        let face = self.grid.as_ref().map_or(0.0, |g| {
            let s = g.step;
            (s.x * s.y).min(s.y * s.z).min(s.x * s.z)
        });
        let crush = self.material.strength.yield_load * face * dt * scale;
        let impulse = touches.iter().map(|t| t.1).sum::<Vec3>() * crush;
        let point = touches.iter().map(|t| t.0).sum::<Vec3>() / touches.len() as f32;
        let size = impulse.length().max(1.0e-9);
        let closing = -carried.velocity.dot(impulse / size);
        reactions.push(Reaction {
            obstacle: owner,
            point,
            impulse: impulse * ((closing.max(0.0) * carried.mass).min(size) / size),
        });
    }

    /// The pose of the carrier this body's impacts are recorded in, and its
    /// mass: the frame is the world's, and the mass 0, without one.
    fn frame(
        &self,
        obstacles: &[Obstacle],
        pin_target: &impl Fn(EntityHandle, Vec3) -> Option<(Vec3, u32)>,
    ) -> (Vec3, Quat, f32) {
        (self.carrier())
            .and_then(|entity| pin_target(entity, Vec3::ZERO))
            .map_or((Vec3::ZERO, Quat::IDENTITY, 0.0), |(_, owner)| {
                let body = &obstacles[owner as usize];
                (body.position, body.rotation, body.mass)
            })
    }

    /// Add this step's touches, `(point, normal, closing speed, partner mass)`,
    /// to the collision under way. One collision spans many steps, in which
    /// closing speed only falls, so it keeps its worst; a step without touches
    /// ends it.
    fn note_hit(&mut self, hits: &[(Vec3, Vec3, f32, f32)], frame: (Vec3, Quat, f32)) {
        let hits: Vec<_> = hits.iter().filter(|h| h.2 > 0.0).collect();
        if hits.is_empty() {
            if let Some(finished) = self.impact.take() {
                if self.culled {
                    self.pending.push(finished);
                }
                if self.impacts.len() == MAX_IMPACTS {
                    self.impacts.remove(0);
                }
                self.impacts.push(finished);
            }
            return;
        }
        let (position, rotation, carried) = frame;
        let count = hits.len() as f32;
        let center = hits.iter().map(|h| h.0).sum::<Vec3>() / count;
        let normal = hits.iter().map(|h| h.1).sum::<Vec3>().normalize_or_zero();
        let closing = hits.iter().fold(0.0f32, |c, h| c.max(h.2));
        let reach = hits.iter().fold(0.0f32, |r, h| r.max(h.0.distance(center)));
        let mass = self.mass() + carried;
        // Hitting something immovable, all of the body's motion is spent.
        let reduced = hits.iter().fold(mass, |m, h| {
            if h.3 > 0.0 {
                m.min(mass * h.3 / (mass + h.3))
            } else {
                m
            }
        });
        let energy = 0.5 * reduced * closing * closing;
        if energy < MIN_IMPACT_ENERGY && self.impact.is_none() {
            return;
        }
        let hit = Impact {
            point: rotation.conjugate() * (center - position),
            normal: rotation.conjugate() * normal,
            energy,
            radius: reach.max(2.0 * self.radius),
        };
        match &mut self.impact {
            Some(impact) if impact.energy >= energy => {
                impact.radius = impact.radius.max(hit.radius)
            }
            Some(impact) => {
                *impact = Impact {
                    radius: impact.radius.max(hit.radius),
                    ..hit
                }
            }
            None => self.impact = Some(hit),
        }
    }

    /// Finished collisions since this was last called, oldest first: what to
    /// send to other machines.
    pub fn take_impacts(&mut self) -> Vec<Impact> {
        std::mem::take(&mut self.impacts)
    }

    /// Dent this body by an impact it did not simulate, such as one from a
    /// server. It takes effect on its next step in view.
    pub fn apply_impact(&mut self, impact: Impact) {
        self.pending.push(impact);
    }

    /// Press an impact into the shape at once, spending its energy crushing at
    /// `yield_load` over its area. Pinned particles hold, as they do in a crash.
    fn dent(&mut self, impact: Impact, position: Vec3, rotation: Quat) {
        let radius = impact.radius;
        let depth = self.crush_depth(&impact);
        if self.sheet || !depth.is_finite() || depth <= 0.0 {
            return;
        }
        self.revision += 1;
        let (point, normal) = (position + rotation * impact.point, rotation * impact.normal);
        let moved: Vec<Vec3> = (self.positions.iter().zip(&self.pinned))
            .map(|(&p, &pin)| {
                let s = p.distance(point) / radius;
                if s < 1.0 && pin == NONE {
                    normal * (depth * (1.0 - s * s).powi(2))
                } else {
                    Vec3::ZERO
                }
            })
            .collect();
        for (p, &u) in self.positions.iter_mut().zip(&moved) {
            *p += u;
        }
        let turn = self.ride.as_mut().map_or(Quat::IDENTITY, |ride| {
            for (q, &u) in ride.local.iter_mut().zip(&moved) {
                *q += ride.rotation.conjugate() * u;
            }
            ride.rotation * ride.start.conjugate()
        });
        for region in &mut self.regions {
            let to_rest = (turn * region.rotation).conjugate();
            let mut worst = 0.0f32;
            for (&i, q) in region.particles.iter().zip(&mut region.rest) {
                *q += to_rest * moved[i as usize];
                worst = worst.max(moved[i as usize].length());
            }
            if worst > 0.0 {
                let shift = (region.particles.iter().zip(&region.rest))
                    .map(|(&i, &q)| q * self.masses[i as usize])
                    .sum::<Vec3>()
                    / region.mass;
                region.rest.iter_mut().for_each(|q| *q -= shift);
                region.update_moment(&self.masses);
                region.strain += worst / radius;
            }
        }
    }

    /// Enter or leave every interest. Coming back in view, a body solves again
    /// and takes the dents it missed.
    pub(crate) fn set_culled(&mut self, culled: bool) {
        if self.culled && !culled {
            self.frozen = false;
            if self.sheet {
                self.stop_riding();
            }
        }
        self.culled = culled;
    }

    /// Whether any interest covers this body, with a margin if it is already in.
    pub(crate) fn in_interest(&self, interests: &[Interest]) -> bool {
        let (min, max) = self.world_bounds();
        let (center, reach) = (0.5 * (min + max), 0.5 * min.distance(max));
        let reach = reach + if self.culled { 0.0 } else { INTEREST_MARGIN };
        interests.iter().any(|i| i.contains(center, reach))
    }

    /// The one rigid body holding every pin, if there is one.
    fn carrier(&self) -> Option<EntityHandle> {
        let first = self.pins.first()?.body?;
        self.pins
            .iter()
            .all(|pin| pin.body == Some(first))
            .then_some(first)
    }

    fn start_riding(&mut self, owner: EntityHandle, body: &Obstacle) {
        let to_local = |&p: &Vec3| body.rotation.conjugate() * (p - body.position);
        self.ride = Some(Ride {
            owner,
            local: self.positions.iter().map(to_local).collect(),
            velocity: body.velocity,
            start: body.rotation,
            rotation: body.rotation,
        });
    }

    fn stop_riding(&mut self) {
        if let Some(ride) = self.ride.take() {
            let turn = ride.rotation * ride.start.conjugate();
            for region in &mut self.regions {
                region.rotation = (turn * region.rotation).normalize();
            }
        }
        self.rest_time = 0.0;
    }

    /// Whether this body is following a carrier without solving.
    pub fn riding(&self) -> bool {
        self.ride.is_some()
    }

    /// Follow the carrier for this step, or stop riding and return false if
    /// anything could touch this body during it.
    fn follow(
        &mut self,
        dt: f32,
        gravity: Vec3,
        obstacles: &[Obstacle],
        pin_target: &impl Fn(EntityHandle, Vec3) -> Option<(Vec3, u32)>,
        reactions: &mut Vec<Reaction>,
    ) -> bool {
        let Some(ride) = &self.ride else {
            return false;
        };
        let Some((_, owner)) = pin_target(ride.owner, Vec3::ZERO) else {
            self.stop_riding();
            return false;
        };
        let (min, max) = self.world_bounds();
        let speed = (self.velocities.iter())
            .fold(0.0f32, |s, v| s.max(v.length_squared()))
            .sqrt();
        let threatened = (obstacles.iter().enumerate())
            .filter(|&(i, _)| i != owner as usize)
            .any(|(_, o)| {
                // Both may close during the step, and the solve must start
                // before they meet: this body is still at last step's pose.
                let travel = speed + o.velocity.length() + o.spin.length() * o.radius();
                let reach = self.material.thickness + 2.0 * travel * dt;
                let r = o.radius() + reach;
                o.position.clamp(min, max).distance_squared(o.position) <= r * r
                    && self.world.iter().any(|&p| o.nearest(p, reach).is_some())
            });
        if threatened {
            self.stop_riding();
            return false;
        }
        self.carry(dt, gravity, &obstacles[owner as usize], owner, reactions);
        true
    }

    /// Ride on unseen, noting what it runs into. Only rigid bodies count: a
    /// car's shell passes through another's until their cores meet.
    fn follow_unseen(
        &mut self,
        dt: f32,
        gravity: Vec3,
        obstacles: &[Obstacle],
        pin_target: &impl Fn(EntityHandle, Vec3) -> Option<(Vec3, u32)>,
        reactions: &mut Vec<Reaction>,
    ) {
        let Some(owner) = (self.ride.as_ref())
            .and_then(|ride| pin_target(ride.owner, Vec3::ZERO))
            .map(|(_, owner)| owner)
        else {
            self.stop_riding();
            self.frozen = true;
            return;
        };
        let (min, max) = self.world_bounds();
        let reach = self.material.thickness;
        let mut hits = Vec::new();
        for (i, o) in obstacles.iter().enumerate() {
            let r = o.radius() + reach;
            let close = o.position.clamp(min, max).distance_squared(o.position) <= r * r;
            if i == owner as usize || !close {
                continue;
            }
            for (&p, &v) in self.world.iter().zip(&self.velocities) {
                if let Some((_, normal)) = o.nearest(p, reach) {
                    hits.push((p, normal, -(v - o.velocity_at(p)).dot(normal), o.mass));
                }
            }
        }
        let body = &obstacles[owner as usize];
        self.note_hit(&hits, (body.position, body.rotation, body.mass));
        // A car stops the same seen or not: only the dent waits.
        let touches: Vec<_> = (hits.iter())
            .filter(|h| h.2 > 0.0)
            .map(|h| (h.0, h.1))
            .collect();
        if !self.sheet {
            self.crush(&touches, UNSEEN_CRUSH, body, owner, dt, reactions);
        }
        self.carry(dt, gravity, body, owner, reactions);
    }

    /// Move rigidly with the carrier, whose body is obstacle `owner`.
    fn carry(
        &mut self,
        dt: f32,
        gravity: Vec3,
        body: &Obstacle,
        owner: u32,
        reactions: &mut Vec<Reaction>,
    ) {
        if !body.awake {
            return;
        }
        let Some(ride) = &mut self.ride else {
            return;
        };
        self.stale = true;
        for ((p, v), &q) in (self.world.iter_mut())
            .zip(&mut self.velocities)
            .zip(&ride.local)
        {
            *p = body.position + body.rotation * q;
            *v = body.velocity_at(*p);
        }
        // The carrier still carries this mass: its weight, and what its own
        // change of speed took. Only the linear part: spin is left to the core.
        let mass = self.masses.iter().sum::<f32>();
        let point = (self.world.iter().zip(&self.masses))
            .map(|(&p, &m)| p * m)
            .sum::<Vec3>()
            / mass;
        reactions.push(Reaction {
            obstacle: owner,
            point,
            impulse: (gravity * dt - (body.velocity - ride.velocity)) * mass,
        });
        ride.velocity = body.velocity;
        ride.rotation = body.rotation;
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
                        closing: 0.0,
                        partner: 0.0,
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
        let contact_stiffness = self.surface_stiffness() * CONTACT_STIFFNESS;
        let inertia: Vec<f32> = self.masses.iter().map(|m| m * scale).collect();
        let mut pin_of = vec![0.0f32; count];
        for (pin, &k) in self.pins.iter().zip(pin_stiffness) {
            pin_of[pin.particle as usize] = k;
        }
        let mut states = vec![ElementState::default(); self.elements.len()];
        let mut work = vec![[Vec3::ZERO; 3]; self.elements.len()];
        let mut tet_states = vec![TetState::default(); self.tets.len()];
        let mut tet_work = vec![[Vec3::ZERO; 4]; self.tets.len()];
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
            let (tets, shear, lame) = (&self.tets, self.shear, self.lame);
            tet_states
                .par_iter_mut()
                .with_min_len(PARALLEL_CHUNK)
                .enumerate()
                .for_each(|(t, state)| {
                    let x = tets[t].particles.map(|p| positions[p as usize]);
                    *state = TetState::new(tets[t].deformation(x), shear, lame);
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
            self.apply_stiffness(
                (&states, &tet_states),
                &moved,
                (&mut work, &mut tet_work),
                &mut damping,
            );
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
            let mut tet_forces = vec![[Vec3::ZERO; 4]; self.tets.len()];
            tet_forces
                .par_iter_mut()
                .with_min_len(PARALLEL_CHUNK)
                .enumerate()
                .for_each(|(t, f)| {
                    let tet = &tets[t];
                    for c in 0..4 {
                        f[c] = tet_states[t].stress * tet.gradient[c] * tet.volume;
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
                for &entry in &self.tet_entries
                    [self.tet_offsets[i] as usize..self.tet_offsets[i + 1] as usize]
                {
                    sum += tet_forces[(entry / 4) as usize][(entry % 4) as usize];
                }
                gradient[i] = sum;
            }
            for (pin, target) in self.pins.iter().zip(targets) {
                let i = pin.particle as usize;
                gradient[i] += (positions[i] - *target) * pin_of[i];
            }
            for bond in &self.bonds {
                let [a, b] = bond.particles.map(|p| p as usize);
                let force = (positions[a] - positions[b]) * bond.stiffness;
                gradient[a] += force;
                gradient[b] -= force;
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
            for (tet, state) in self.tets.iter().zip(&tet_states) {
                for (&p, &g) in tet.particles.iter().zip(&tet.gradient) {
                    let along = state.cofactor * g;
                    let sum = Vec3::splat(2.0 * shear * g.length_squared()) + along * along * lame;
                    diagonal[p as usize] += sum * tet.volume;
                }
            }
            for bond in &self.bonds {
                for p in bond.particles {
                    diagonal[p as usize] += Vec3::splat(bond.stiffness);
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
                self.apply_stiffness(
                    (&states, &tet_states),
                    &direction,
                    (&mut work, &mut tet_work),
                    &mut product,
                );
                for i in 0..count {
                    let (n, k, along) = contact_of[i];
                    product[i] = product[i] * (1.0 + rate)
                        + direction[i] * (inertia[i] + pin_of[i] + along)
                        + n * ((k - along) * n.dot(direction[i]));
                }
                let curvature = dot(&direction, &product);
                if curvature <= 0.0 {
                    // Crushed elements make this indefinite. If that is at once,
                    // down the gradient still beats standing still.
                    if !self.tets.is_empty() && delta.iter().all(|d| *d == Vec3::ZERO) {
                        delta.copy_from_slice(&direction);
                    }
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
            if !self.tets.is_empty() {
                let reach = TET_TRUST * 2.0 * self.radius;
                let worst = delta.iter().fold(0.0f32, |w, d| w.max(d.length_squared()));
                if worst > reach * reach {
                    let scale = reach / worst.sqrt();
                    delta.iter_mut().for_each(|d| *d *= scale);
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
        (states, tet_states): (&[ElementState], &[TetState]),
        v: &[Vec3],
        (work, tet_work): (&mut [[Vec3; 3]], &mut [[Vec3; 4]]),
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
        // All of it: without the change of the cofactors this is half as stiff in
        // shear, and every step overshoots until a stiff body explodes. It can go
        // negative for a badly crushed element, where the solve stops early.
        let tets = &self.tets;
        tet_work
            .par_iter_mut()
            .with_min_len(PARALLEL_CHUNK)
            .enumerate()
            .for_each(|(t, corners)| {
                let (tet, state) = (&tets[t], &tet_states[t]);
                let change = tet.deformation(tet.particles.map(|p| v[p as usize]));
                let cofactor = state.cofactor;
                let volume = cofactor.x_axis.dot(change.x_axis)
                    + cofactor.y_axis.dot(change.y_axis)
                    + cofactor.z_axis.dot(change.z_axis);
                let f = state.deformation;
                let turned = Mat3::from_cols(
                    change.y_axis.cross(f.z_axis) + f.y_axis.cross(change.z_axis),
                    change.z_axis.cross(f.x_axis) + f.z_axis.cross(change.x_axis),
                    change.x_axis.cross(f.y_axis) + f.x_axis.cross(change.y_axis),
                );
                let stress = change * shear + cofactor * (lame * volume) + turned * state.pressure;
                for c in 0..4 {
                    corners[c] = stress * tet.gradient[c] * tet.volume;
                }
            });
        let (offsets, entries) = (&self.offsets, &self.entries);
        let (hinge_offsets, hinge_entries, hinges) =
            (&self.hinge_offsets, &self.hinge_entries, &self.hinges);
        let (tet_offsets, tet_entries) = (&self.tet_offsets, &self.tet_entries);
        let work = &*work;
        let tet_work = &*tet_work;
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
                for &entry in &tet_entries[tet_offsets[i] as usize..tet_offsets[i + 1] as usize] {
                    sum += tet_work[(entry / 4) as usize][(entry % 4) as usize];
                }
                *value = sum;
            });
        for bond in &self.bonds {
            let [a, b] = bond.particles.map(|p| p as usize);
            let pull = (v[a] - v[b]) * bond.stiffness;
            out[a] += pull;
            out[b] -= pull;
        }
    }

    /// Let bonds open, and tear elements near failing along their edge most
    /// nearly on the plane they would part on.
    fn tear(&mut self, membrane: &Membrane, blend: f32, dt: f32) {
        let positions = &self.positions;
        let mut parted = 0.0;
        self.bonds.retain_mut(|bond| {
            let [a, b] = bond.particles.map(|p| positions[p as usize]);
            let holding = bond.open(a.distance(b));
            if !holding {
                parted += 0.5 * bond.peak * bond.reach;
            }
            holding
        });
        self.tear_work += parted;
        if !self.tets.is_empty() {
            self.fail_tets(blend, dt);
            return;
        }
        let material = self.material;
        let mut failing = Vec::new();
        for (t, element) in self.elements.iter_mut().enumerate() {
            let tri = self.triangles[t];
            let (a, b) = element.deformation(tri.map(|c| positions[c as usize]));
            let (stress, _) = stress(a, b, membrane);
            let (ratio, normal) = material.failure(stress, element.warp);
            element.load += (ratio - element.load) * blend;
            if element.load >= TIP_LOAD {
                let corners = tri.map(|i| self.rest[self.origin[i as usize] as usize]);
                failing.push((t, Element::lift(normal, corners)));
            }
        }
        if failing.is_empty() {
            return;
        }
        let cell = self.radius / 0.45;
        let passes = ((TEAR_SPEED * dt / cell).ceil() as usize).max(1);
        let mut started = false;
        let mut changed = false;
        for _ in 0..passes {
            let steps = self.next_steps(&failing, started);
            if steps.is_empty() {
                break;
            }
            started = true;
            changed = true;
            for step in steps {
                self.take_step(step, &mut failing);
            }
        }
        if changed {
            self.rebuild();
        }
    }

    /// The cuts tears make next, given elements that may fail and the normals of
    /// the planes they would part on, at rest.
    fn next_steps(&self, failing: &[(usize, Vec3)], started: bool) -> Vec<Step> {
        let mut steps = Vec::new();
        for &(t, normal) in failing {
            let own = self.elements[t].load;
            let at_tip = (self.triangles[t].iter())
                .any(|&i| self.cut_ends.get(&self.origin[i as usize]) == Some(&1));
            if at_tip && own >= TIP_LOAD {
                let step = self.step_from(t, normal, own);
                steps.extend(step.filter(|step| step.tip.is_some() || own >= 1.0));
                continue;
            }
            if own < 1.0 {
                continue;
            }
            // Over its neighbourhood, not one element: a spike in a single
            // triangle says more of the mesh than of the material.
            let (mut sum, mut area) = (0.0, 0.0);
            for corner in self.triangles[t] {
                let (from, to) = (
                    self.offsets[corner as usize],
                    self.offsets[corner as usize + 1],
                );
                for &entry in &self.entries[from as usize..to as usize] {
                    let near = &self.elements[(entry / 3) as usize];
                    sum += near.load * near.area;
                    area += near.area;
                }
            }
            let load = sum / area;
            if load >= 1.0 {
                steps.extend(self.step_from(t, normal, load));
            }
        }
        // A tear shields the sheet around it, which a step's worth of load cannot
        // yet show: uniformly overloaded, every element fails at once. So while
        // any tear is growing no other starts, and only one starts at a time.
        let growing = steps.iter().any(|step| step.tip.is_some());
        steps.retain(|step| step.tip.is_some() == growing);
        steps.sort_unstable_by(|a, b| b.load.total_cmp(&a.load));
        // One cut per tip, so a tear stays a line instead of branching.
        let mut moved = HashSet::new();
        steps.retain(|step| step.tip.is_none_or(|tip| moved.insert(tip)));
        let limit = 1 + self.elements.len() / TEARS_PER_STEP;
        let allowed = match (growing, started) {
            (true, _) => limit,
            (false, false) => 1,
            (false, true) => 0,
        };
        steps.truncate(allowed);
        steps
    }

    /// Whether a tear at tip `from` may run to `to`: never back the way it came.
    fn onward(&self, from: u32, to: Vec3) -> bool {
        let heading = self.headings.get(&from).copied().unwrap_or(Vec3::ZERO);
        (to - self.rest[from as usize]).dot(heading) >= 0.0
    }

    /// How element `t` would tear, on the plane with this normal at rest.
    fn step_from(&self, t: usize, normal: Vec3, load: f32) -> Option<Step> {
        let tri = self.triangles[t];
        let origins = tri.map(|i| self.origin[i as usize]);
        let at = origins.map(|o| self.rest[o as usize]);
        let is_tip = |o: u32| self.cut_ends.get(&o) == Some(&1);
        let cut_between =
            |a: usize, b: usize| self.cuts.contains(&edge(origins[a % 3], origins[b % 3]));
        // From a tip, straight through the triangle if that beats any edge.
        for corner in (0..3).filter(|&c| is_tip(origins[c])) {
            let (a, p, q) = (at[corner], at[(corner + 1) % 3], at[(corner + 2) % 3]);
            if cut_between(corner + 1, corner + 2) {
                continue;
            }
            let plane = (p - a).cross(q - a);
            let along = plane.cross(normal).normalize_or_zero();
            let sense = along.dot(0.5 * (p + q) - a).signum();
            // A tear keeps some of its heading: right at a tip the stress is
            // the mesh's as much as the material's.
            let heading = self.headings.get(&origins[corner]).copied();
            let heading = heading.unwrap_or(Vec3::ZERO).normalize_or_zero();
            let along = (along * sense + heading * TEAR_MEMORY).normalize_or_zero();
            let share = (a - p).cross(along).dot(plane) / (q - p).cross(along).dot(plane);
            let to = p + (q - p) * share;
            let inside = (SPLIT_MARGIN..=1.0 - SPLIT_MARGIN).contains(&share);
            if inside && self.onward(origins[corner], to) {
                let cut = Cut::Through { corner, share };
                let tip = Some(origins[corner]);
                return Some(Step {
                    load,
                    element: t,
                    cut,
                    tip,
                });
            }
        }
        // Otherwise the uncut edge most nearly on that plane.
        let across = |c: usize| {
            (at[(c + 1) % 3] - at[c])
                .normalize_or_zero()
                .dot(normal)
                .abs()
        };
        let tip_of = |c: usize| {
            let ends = [(c, (c + 1) % 3), ((c + 1) % 3, c)];
            let (from, _) = ends
                .into_iter()
                .find(|&(from, to)| is_tip(origins[from]) && self.onward(origins[from], at[to]))?;
            Some(origins[from])
        };
        let score = |c: usize| across(c) - if tip_of(c).is_some() { 0.5 } else { 0.0 };
        (0..3)
            .filter(|&c| !cut_between(c, c + 1))
            .min_by(|&x, &y| score(x).total_cmp(&score(y)))
            .map(|best| Step {
                load,
                element: t,
                cut: Cut::Edge(edge(origins[best], origins[(best + 1) % 3])),
                tip: tip_of(best),
            })
    }

    /// Make a cut, and record where the tear now ends.
    fn take_step(&mut self, step: Step, failing: &mut Vec<(usize, Vec3)>) {
        let t = step.element;
        let (a, b) = match step.cut {
            Cut::Edge(cut) => {
                if self.cuts.contains(&cut) {
                    return;
                }
                cut
            }
            Cut::Through { corner, share } => {
                let tri = self.triangles[t];
                let from = self.origin[tri[corner] as usize];
                // An earlier cut this pass may have split this triangle.
                if step.tip != Some(from) {
                    return;
                }
                let (p, q) = (tri[(corner + 1) % 3], tri[(corner + 2) % 3]);
                if self
                    .cuts
                    .contains(&edge(self.origin[p as usize], self.origin[q as usize]))
                {
                    return;
                }
                let before = self.elements.len();
                let new = self.split_edge(p, q, share);
                let normal = failing.iter().find(|f| f.0 == t).map(|f| f.1);
                for child in before..self.elements.len() {
                    if let Some(normal) = normal {
                        failing.push((child, normal));
                    }
                }
                (from, self.origin[new as usize])
            }
        };
        self.elements[t].load = 0.0;
        // The far end is the new tip, whichever way round the cut is keyed.
        let from = step.tip.unwrap_or(a);
        let to = if from == a { b } else { a };
        let heading = self.rest[to as usize] - self.rest[from as usize];
        self.headings.insert(to, heading);
        self.tears += 1;
        self.cuts.insert(edge(a, b));
        *self.cut_ends.entry(a).or_default() += 1;
        *self.cut_ends.entry(b).or_default() += 1;
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

/// Stop riding any body another soft body could reach this step, so that
/// [`find_pairs`] can see them meet. Bodies on one carrier move together.
/// shortcut: tests every pair of boxes, which is fine for hundreds of bodies; use
/// the hash grid's cells if thousands ever ride at once.
pub(crate) fn wake_riders(bodies: &mut [SoftBody], dt: f32) {
    if bodies.iter().all(|b| b.ride.is_none()) {
        return;
    }
    let boxes: Vec<(Vec3, Vec3, Option<EntityHandle>)> = (bodies.iter())
        .map(|b| {
            let (min, max) = b.world_bounds();
            let speed = (b.velocities.iter())
                .fold(0.0f32, |s, v| s.max(v.length_squared()))
                .sqrt();
            let pad = b.radius + 2.0 * speed * dt;
            (min - pad, max + pad, b.carrier())
        })
        .collect();
    for (i, body) in bodies.iter_mut().enumerate() {
        let (min, max, owner) = boxes[i];
        let near = |(j, other): (usize, &(Vec3, Vec3, Option<EntityHandle>))| {
            j != i
                && (owner.is_none() || other.2 != owner)
                && min.cmple(other.1).all()
                && other.0.cmple(max).all()
        };
        if body.ride.is_some() && !body.culled && boxes.iter().enumerate().any(near) {
            body.stop_riding();
        }
    }
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
    let idle = |b: &SoftBody| b.asleep || b.frozen || b.ride.is_some();
    if bodies.iter().all(idle) || (bodies.len() < 2 && !bodies.iter().any(|b| b.sheet)) {
        return;
    }
    let widest = bodies.iter().fold(0.0f32, |w, b| w.max(b.radius));
    let moving = bodies.iter().filter(|b| b.ride.is_none());
    let fastest = (moving.flat_map(|b| b.velocities.iter()))
        .fold(0.0f32, |s, v| s.max(v.length_squared()))
        .sqrt();
    let size = 2.0 * widest + 2.0 * (fastest * dt + gravity.length() * dt * dt);
    if size <= 0.0 {
        return;
    }
    let cell = |p: Vec3| (p / size).floor().as_ivec3().to_array();
    let mut grid: HashMap<[i32; 3], Vec<(u32, u32)>> = HashMap::new();
    for (b, body) in bodies.iter().enumerate() {
        if body.ride.is_some() {
            continue;
        }
        for (i, &p) in body.world.iter().enumerate() {
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
        let solid = !body.sheet || !body.tets.is_empty();
        if solid || body.rest[oi].distance(body.rest[oj]) < NEIGHBOUR_REACH * reach {
            return;
        }
    }
    let (pi, pj) = (bodies[b].world[i], bodies[c].world[j]);
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
    let closing = -closing / dt;
    let (mass_b, mass_c) = (bodies[b].mass(), bodies[c].mass());
    bodies[b].planes.push(Plane {
        particle: i as u32,
        normal: n,
        offset: middle + ri.min(distance.max(ri * 0.5)),
        closing,
        mass: mass_c,
    });
    bodies[c].planes.push(Plane {
        particle: j as u32,
        normal: -n,
        offset: -middle + rj.min(distance.max(rj * 0.5)),
        closing,
        mass: mass_b,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The direction a sheet would tear in, degrees from its warp, pulled at
    /// this angle to it.
    fn tear_angle(material: SoftMaterial, pull_degrees: f32) -> f32 {
        let (sin, cos) = pull_degrees.to_radians().sin_cos();
        let stress = [cos * cos, sin * sin, sin * cos].map(|s| s * 2.0e4);
        let (_, normal) = material.failure(stress, Vec2::X);
        let degrees = normal.perp().y.atan2(normal.perp().x).to_degrees();
        (degrees + 180.0) % 180.0
    }

    /// How far apart two directions are, as lines.
    fn apart(a: f32, b: f32) -> f32 {
        let d = (a - b).abs() % 180.0;
        d.min(180.0 - d)
    }

    #[test]
    fn felt_tears_square_to_its_pull_whatever_its_warp() {
        let felt = SoftMaterial {
            weave: 0.0,
            ..SoftMaterial::cloth()
        };
        for pull in [0.0, 30.0, 45.0, 70.0] {
            let angle = tear_angle(felt, pull);
            assert!(apart(angle, pull + 90.0) < 1.0, "pulled at {pull}: {angle}");
        }
    }

    #[test]
    fn a_weave_tears_along_a_thread_even_pulled_off_it() {
        let weave = SoftMaterial::cloth();
        for pull in [20.0, 70.0, 110.0] {
            let angle = tear_angle(weave, pull);
            assert!(
                apart(angle, 0.0).min(apart(angle, 90.0)) < 1.0,
                "pulled at {pull}: {angle}"
            );
        }
    }

    #[test]
    fn a_grain_tears_along_itself_even_pulled_off_square_to_it() {
        let grain = SoftMaterial {
            grain: 0.3,
            weave: 0.0,
            ..SoftMaterial::cloth()
        };
        for pull in [60.0, 90.0, 130.0] {
            let angle = tear_angle(grain, pull);
            assert!(apart(angle, 0.0) < 1.0, "pulled at {pull}: {angle}");
        }
    }

    #[test]
    fn splitting_an_edge_keeps_mass_shape_and_strain() {
        let mut cloth = SoftBody::cloth(
            Vec3::ZERO,
            Quat::IDENTITY,
            [1.0, 1.0],
            [4, 4],
            SoftMaterial::cloth(),
        );
        // Stretched evenly: 10% along X.
        for p in &mut cloth.positions {
            p.x *= 1.1;
        }
        let strain = |c: &SoftBody| {
            (c.elements.iter().zip(&c.triangles))
                .map(|(e, t)| {
                    // Whatever way an element's axes lie, this is 1.1^2 + 1.
                    let (a, b) = e.deformation(t.map(|i| c.positions[i as usize]));
                    a.length_squared() + b.length_squared()
                })
                .fold((f32::MAX, 0.0f32), |(low, high), s| {
                    (low.min(s), high.max(s))
                })
        };
        let mass = cloth.mass();
        let (area, count): (f32, usize) = (cloth.elements.iter().map(|e| e.area).sum(), 32);
        assert_eq!(cloth.elements.len(), count);
        let [p, q, _] = cloth.triangles[8];
        let new = cloth.split_edge(p, q, 0.3);
        cloth.rebuild();
        assert_eq!(
            cloth.elements.len(),
            count + 2,
            "an inner edge has two sides"
        );
        assert!((cloth.mass() - mass).abs() < 1.0e-5 * mass);
        assert!(cloth.masses.iter().all(|&m| m > 0.0));
        let after: f32 = cloth.elements.iter().map(|e| e.area).sum();
        assert!((after - area).abs() < 1.0e-4);
        let (low, high) = strain(&cloth);
        assert!(low > 2.209 && high < 2.211, "strain {low} to {high}");
        let (a, b) = (cloth.uvs[p as usize], cloth.uvs[q as usize]);
        assert!(cloth.uvs[new as usize].distance(a.lerp(b, 0.3)) < 1.0e-6);
    }

    /// A 1 m cube of tetrahedra, 4 cells a side, with no gravity to feel.
    fn cube(material: SoftMaterial) -> SoftBody {
        let half = Vec3::splat(0.5);
        SoftBody::lattice(Vec3::ZERO, Quat::IDENTITY, half, [4, 4, 4], material).finite_elements()
    }

    fn step_alone(body: &mut SoftBody, steps: usize) {
        for _ in 0..steps {
            let mut reactions = Vec::new();
            let no_pin = |_: EntityHandle, _: Vec3| None;
            body.step(1.0 / 60.0, Vec3::ZERO, &[], no_pin, &mut reactions);
        }
    }

    /// Cut every face between the halves of `body` with `side`, as if failed.
    fn cut_where(body: &mut SoftBody, cuts: impl Fn(Vec3) -> bool) {
        for face in body.uncut.clone() {
            if face.iter().all(|&p| cuts(body.rest[p as usize])) {
                body.uncut.remove(&face);
                body.cut_faces.push(face);
                for p in face {
                    *body.cut_ends.entry(p).or_default() += 1;
                }
            }
        }
        body.rebuild_tets();
    }

    /// Whether every edge of a surface is on two triangles, running opposite ways.
    fn closed(triangles: &[[u32; 3]]) -> bool {
        let mut edges: HashMap<(u32, u32), i32> = HashMap::new();
        for t in triangles {
            for c in 0..3 {
                *edges.entry((t[c], t[(c + 1) % 3])).or_default() += 1;
            }
        }
        edges
            .iter()
            .all(|(&(a, b), &n)| n == 1 && edges.get(&(b, a)) == Some(&1))
    }

    #[test]
    fn tetrahedra_fill_their_solid_and_rest_unstressed() {
        let body = cube(SoftMaterial::flesh());
        assert_eq!(body.tets.len(), 4 * 4 * 4 * 6);
        assert!(body.tets.iter().all(|t| t.volume > 0.0));
        let volume: f32 = body.tets.iter().map(|t| t.volume).sum();
        assert!((volume - 1.0).abs() < 1.0e-4, "volume {volume}");
        assert!((body.mass() - 1060.0).abs() < 0.5, "mass {}", body.mass());
        assert!(closed(&body.triangles));
        assert_eq!(body.triangles.len(), 6 * 4 * 4 * 2);
        // Outward: a closed surface's volume by the divergence theorem.
        let signed: f32 = (body.triangles.iter())
            .map(|t| {
                let [a, b, c] = t.map(|i| body.positions[i as usize]);
                a.dot(b.cross(c)) / 6.0
            })
            .sum();
        assert!((signed - 1.0).abs() < 1.0e-3, "enclosed {signed}");
        let state = TetState::new(Mat3::IDENTITY, 3.0e4, 3.0e5);
        assert!(state.stress.x_axis.length() < 1.0e-2);
    }

    #[test]
    fn stress_pushes_back_from_stretch_and_from_inversion() {
        let (shear, lame) = (3.0e4, 3.0e5);
        let stretched = TetState::new(Mat3::from_diagonal(Vec3::new(1.1, 1.0, 1.0)), shear, lame);
        assert!(stretched.stress.x_axis.x > 0.0);
        let inverted = TetState::new(Mat3::from_diagonal(Vec3::new(-0.2, 1.0, 1.0)), shear, lame);
        assert!(
            inverted.stress.x_axis.x < 0.0,
            "pushes it back through zero"
        );
        let flat = TetState::new(Mat3::from_diagonal(Vec3::new(0.0, 1.0, 1.0)), shear, lame);
        assert!(flat.stress.x_axis.x.is_finite() && flat.stress.x_axis.x < 0.0);
    }

    #[test]
    fn past_yield_a_tetrahedral_solid_keeps_some_of_its_stretch() {
        let material = SoftMaterial {
            stiffness: 1.0e6,
            strength: Strength {
                yield_load: 2.0e4,
                break_load: f32::INFINITY,
                ductility: f32::INFINITY,
            },
            ..SoftMaterial::rubber()
        };
        let mut body = cube(material);
        let length = |b: &SoftBody| {
            let (min, max) = b.bounds();
            max.x - min.x
        };
        for p in &mut body.positions {
            p.x *= 1.3;
        }
        // Held there while it flows, as what dents a solid holds it.
        let held = body.positions.clone();
        for _ in 0..10 {
            body.positions.copy_from_slice(&held);
            body.velocities.fill(Vec3::ZERO);
            step_alone(&mut body, 1);
        }
        step_alone(&mut body, 120);
        assert!(body.permanent_strain() > 0.0);
        // Not all the way back, and not nowhere: 2% yield strain recovers.
        let kept = length(&body);
        assert!(kept > 1.15 && kept < 1.3, "{kept} m long");
        let volume: f32 = body.tets.iter().map(|t| t.volume).sum();
        assert!((volume - 1.0).abs() < 1.0e-3);
    }

    #[test]
    fn below_yield_a_tetrahedral_solid_springs_back() {
        let mut body = cube(SoftMaterial::rubber());
        for p in &mut body.positions {
            p.x *= 1.05;
        }
        body.publish();
        step_alone(&mut body, 240);
        let (min, max) = body.bounds();
        assert!(
            (max.x - min.x - 1.0).abs() < 0.01,
            "{} m long",
            max.x - min.x
        );
        assert_eq!(body.permanent_strain(), 0.0);
    }

    #[test]
    fn a_crack_through_a_solid_leaves_two_closed_surfaces() {
        let material = SoftMaterial {
            toughness: 0.0,
            ..SoftMaterial::flesh()
        };
        let mut body = cube(material);
        let (mass, triangles) = (body.mass(), body.triangles.len());
        cut_where(&mut body, |p| p.z.abs() < 1.0e-3);
        assert_eq!(body.pieces(), 2);
        assert!(closed(&body.triangles), "open surface");
        assert_eq!(body.triangles.len(), triangles + 2 * 4 * 4 * 2);
        assert!((body.mass() - mass).abs() < 1.0e-3 * mass);
        assert_eq!(body.positions.len(), body.velocities.len());
        // The solve must take the new particles.
        step_alone(&mut body, 2);
        assert!(body.positions.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn a_part_cracked_solid_stays_closed_and_whole() {
        let mut body = cube(SoftMaterial::flesh());
        cut_where(&mut body, |p| p.z.abs() < 1.0e-3 && p.x <= 0.0);
        assert_eq!(body.pieces(), 1);
        assert!(closed(&body.triangles), "open surface");
        assert!(
            !body.bonds.is_empty(),
            "a crack's faces hold until they open"
        );
    }

    #[test]
    fn bonds_in_a_solid_hold_a_crack_until_it_opens() {
        let mut body = cube(SoftMaterial::flesh());
        cut_where(&mut body, |p| p.z.abs() < 1.0e-3);
        assert_eq!(body.pieces(), 1, "bonds still hold");
        // Each half flies apart with far more energy than the crack can take.
        for tet in body.tets.clone() {
            let rest = tet
                .particles
                .map(|p| body.rest[body.origin[p as usize] as usize]);
            let side = (rest[0] + rest[1] + rest[2] + rest[3]).z.signum();
            for p in tet.particles {
                body.velocities[p as usize] = Vec3::Z * (10.0 * side);
            }
        }
        step_alone(&mut body, 30);
        assert_eq!(body.pieces(), 2, "pulled past their reach");
        assert!(body.tear_work() > 0.0);
    }

    #[test]
    fn flesh_authored_in_a_scene_is_made_of_tetrahedra() {
        let authored = crate::SoftBodyComponent {
            shape: crate::SoftShape::Ball,
            material: crate::SoftPreset::Flesh,
            size: [1.0; 3],
            cells: [6.0; 3],
            ..Default::default()
        };
        let body = authored.to_soft_body(Vec3::ZERO, Quat::IDENTITY).unwrap();
        assert!(!body.tets.is_empty() && body.regions.is_empty());
    }

    #[test]
    fn a_cut_across_a_sheet_splits_it_and_no_hinge_bridges_the_tear() {
        let center = Vec3::ZERO;
        let material = SoftMaterial {
            toughness: 0.0,
            ..SoftMaterial::cloth()
        };
        let mut cloth = SoftBody::cloth(center, Quat::IDENTITY, [1.0, 1.0], [4, 4], material);
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
