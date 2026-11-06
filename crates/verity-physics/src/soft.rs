//! Deformable bodies: cloth and solids.
//!
//! Cloth is a triangle mesh whose edges are beams, solved by vertex block
//! descent (Chen et al. 2024) with each beam's force carried as an AVBD
//! multiplier. It tears along edges that exceed their strength.
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
use glam::{Mat3, Quat, Vec3};
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
/// Beams meeting at a particle for their forces to balance while solving.
const SUPPORTED_BEAMS: usize = 5;
/// Share of an overlap with a rigid body resolved each step.
const SEPARATION: f32 = 0.2;
/// Share of the speed along a beam removed each step.
const BEAM_DAMPING: f32 = 0.3;
/// Share of a beam's force kept into the next step's warm start.
const FORCE_MEMORY: f32 = 0.99;
/// Seconds over which a beam's load is averaged, as for joints.
const LOAD_TIME: f32 = 0.03;
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
    /// Share of `stiffness` that resists shortening: little for cloth, which
    /// buckles instead.
    pub compression: f32,
    /// Share of `stiffness` that resists folding a sheet.
    pub bending: f32,
    /// How firmly a solid returns to its shape each sweep: 1 is as rigid as it
    /// can be, 0.1 is jelly.
    pub rigidity: f32,
    /// In the units of `stiffness`, with `ductility` as strain: 0.1 is 10%.
    /// A solid yields at the strain `yield_load / stiffness`, and hardens no
    /// further than `ductility`; only cloth breaks.
    pub strength: Strength,
    /// Velocity lost per second.
    pub damping: f32,
    pub friction: f32,
    /// Distance kept from what it touches (m).
    pub thickness: f32,
}

impl SoftMaterial {
    /// Woven cloth that tears at 2 kN per metre of width.
    pub fn cloth() -> Self {
        Self {
            density: 0.3,
            stiffness: 20_000.0,
            compression: 0.05,
            bending: 0.002,
            rigidity: 0.0,
            strength: Strength::brittle(2_000.0),
            damping: 1.0,
            friction: 0.5,
            thickness: 0.02,
        }
    }

    /// Mild steel in a solid. It holds its shape until 1% strain, then dents.
    pub fn sheet_metal() -> Self {
        Self {
            density: 7800.0,
            stiffness: 2.0e9,
            compression: 1.0,
            bending: 0.0,
            rigidity: 1.0,
            strength: Strength {
                yield_load: 2.0e7,
                break_load: 8.0e7,
                ductility: 5.0,
            },
            damping: 2.0,
            friction: 0.5,
            thickness: 0.01,
        }
    }

    /// The hollow body of a car, as a solid: light, and crushed by what steel
    /// would shrug off.
    pub fn crumple_zone() -> Self {
        Self {
            density: 150.0,
            stiffness: 2.0e6,
            compression: 1.0,
            bending: 0.0,
            rigidity: 1.0,
            strength: Strength {
                yield_load: 2.0e4,
                break_load: 8.0e4,
                ductility: 20.0,
            },
            damping: 2.0,
            friction: 0.5,
            thickness: 0.01,
        }
    }

    /// A rubber solid.
    pub fn rubber() -> Self {
        Self {
            density: 1100.0,
            stiffness: 2.0e5,
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

#[derive(Debug, Clone, Copy)]
struct Beam {
    a: u32,
    b: u32,
    rest: f32,
    /// Rest length as built, the measure of strain.
    original: f32,
    /// Stiffness (N/m) and its share when shortened.
    k: f32,
    compression: f32,
    /// How this beam fails, in newtons.
    strength: Strength,
    strain: f32,
    load: f32,
    /// The force it carries (N): AVBD's multiplier, which a finite stiffness
    /// relates to stretch by `stretch = force / k` at convergence.
    force: f32,
    /// A mesh edge, not a beam across a fold: only these tear the mesh.
    edge: bool,
}

impl Beam {
    fn cap(&self) -> f32 {
        let s = self.strength;
        if s.yield_load >= s.break_load {
            f32::INFINITY
        } else if s.break_load.is_infinite() {
            s.yield_load
        } else {
            let hardened = (self.strain / s.ductility).min(1.0);
            s.yield_load + (s.break_load - s.yield_load) * hardened
        }
    }

    /// Stiffness for a force of this sign.
    fn stiffness(&self, force: f32) -> f32 {
        if force < 0.0 {
            self.k * self.compression
        } else {
            self.k
        }
    }

    /// The force at this length, by augmented Lagrangian: `stretch - force / k`
    /// is the constraint, whose solution is Hooke's law.
    fn force_at(&self, length: f32, penalty: f32, cap: f32) -> f32 {
        let stretch = length - self.rest;
        let force = penalty * (stretch - self.force / self.stiffness(self.force)) + self.force;
        force.clamp(-cap, cap)
    }
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
    masses: Vec<f32>,
    /// Each particle's index when built: the same edge keeps its identity
    /// after tearing duplicates its ends.
    origin: Vec<u32>,
    triangles: Vec<[u32; 3]>,
    beams: Vec<Beam>,
    regions: Vec<Region>,
    /// Beams by particle: `entries[offsets[i]..offsets[i + 1]]`.
    offsets: Vec<u32>,
    entries: Vec<u32>,
    pins: Vec<Pin>,
    /// The pin on each particle, or `NONE`.
    pinned: Vec<u32>,
    material: SoftMaterial,
    /// Length per cross-section: beam loads are the material's times this.
    section: f32,
    /// Sheets rebuild their mesh when an edge breaks.
    sheet: bool,
    /// Original edges that have broken, by `origin` indices.
    cuts: HashSet<(u32, u32)>,
    tears: usize,
    rest_time: f32,
    asleep: bool,
    contacts: Vec<Contact>,
}

fn edge(a: u32, b: u32) -> (u32, u32) {
    (a.min(b), a.max(b))
}

impl SoftBody {
    fn new(positions: Vec<Vec3>, masses: Vec<f32>, material: SoftMaterial, section: f32) -> Self {
        let count = positions.len();
        Self {
            velocities: vec![Vec3::ZERO; count],
            origin: (0..count as u32).collect(),
            pinned: vec![NONE; count],
            positions,
            masses,
            triangles: Vec::new(),
            beams: Vec::new(),
            regions: Vec::new(),
            offsets: Vec::new(),
            entries: Vec::new(),
            pins: Vec::new(),
            material,
            section,
            sheet: false,
            cuts: HashSet::new(),
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
        let mass = material.density * size[0] * size[1] / (nx * ny) as f32;
        let cell = dx.min(dy);
        let mut body = Self::new(positions, vec![mass; (nx * ny) as usize], material, cell);
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
        body.rebuild(&HashMap::new());
        body
    }

    /// A solid box of `cells` along each axis: a particle at each grid point,
    /// and beams along the edges and diagonals of each cell, so it keeps its
    /// shape with no other help.
    pub fn lattice(
        center: Vec3,
        rotation: Quat,
        half_extents: Vec3,
        cells: [u32; 3],
        material: SoftMaterial,
    ) -> Self {
        assert!(
            cells.iter().all(|&c| c > 0),
            "a lattice needs at least one cell"
        );
        assert!(
            half_extents.min_element() > 0.0,
            "a lattice's size must be positive"
        );
        let [nx, ny, nz] = cells.map(|c| c + 1);
        let at = |x: u32, y: u32, z: u32| (z * ny + y) * nx + x;
        let step =
            half_extents * 2.0 / Vec3::new(cells[0] as f32, cells[1] as f32, cells[2] as f32);
        let mut positions = Vec::with_capacity((nx * ny * nz) as usize);
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let local = Vec3::new(x as f32, y as f32, z as f32) * step - half_extents;
                    positions.push(center + rotation * local);
                }
            }
        }
        let volume = 8.0 * half_extents.x * half_extents.y * half_extents.z;
        let mass = material.density * volume / positions.len() as f32;
        let cell = step.min_element();
        let count = positions.len();
        let mut body = Self::new(positions, vec![mass; count], material, cell * cell);
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
                        (c - REGION_REACH).max(0) as u32
                            ..=(c + REGION_REACH).min(n as i32 - 1) as u32
                    };
                    let mut particles = Vec::new();
                    for z in span(cz, nz) {
                        for y in span(cy, ny) {
                            for x in span(cx, nx) {
                                particles.push(at(x, y, z));
                            }
                        }
                    }
                    let region = Region::new(particles, &body.positions, &body.masses);
                    body.regions.push(region);
                }
            }
        }
        let mut quad = |a: u32, b: u32, c: u32, d: u32| {
            body.triangles.extend([[a, b, c], [a, c, d]]);
        };
        for y in 0..cells[1] {
            for x in 0..cells[0] {
                let top = cells[2];
                quad(
                    at(x, y, 0),
                    at(x, y + 1, 0),
                    at(x + 1, y + 1, 0),
                    at(x + 1, y, 0),
                );
                quad(
                    at(x, y, top),
                    at(x + 1, y, top),
                    at(x + 1, y + 1, top),
                    at(x, y + 1, top),
                );
            }
        }
        for z in 0..cells[2] {
            for x in 0..cells[0] {
                let back = cells[1];
                quad(
                    at(x, 0, z),
                    at(x + 1, 0, z),
                    at(x + 1, 0, z + 1),
                    at(x, 0, z + 1),
                );
                quad(
                    at(x, back, z),
                    at(x, back, z + 1),
                    at(x + 1, back, z + 1),
                    at(x + 1, back, z),
                );
            }
            for y in 0..cells[1] {
                let right = cells[0];
                quad(
                    at(0, y, z),
                    at(0, y, z + 1),
                    at(0, y + 1, z + 1),
                    at(0, y + 1, z),
                );
                quad(
                    at(right, y, z),
                    at(right, y + 1, z),
                    at(right, y + 1, z + 1),
                    at(right, y, z + 1),
                );
            }
        }
        body
    }

    /// Add a beam between two particles at their current distance, keeping
    /// `old`'s state when a mesh is rebuilt.
    fn add_beam(&mut self, a: u32, b: u32, share: f32, edge: bool, old: Option<&Beam>) {
        let beam = if let Some(old) = old {
            Beam { a, b, ..*old }
        } else {
            let rest = self.positions[a as usize].distance(self.positions[b as usize]);
            let m = self.material;
            let scale = |load: f32| load * self.section;
            Beam {
                a,
                b,
                rest,
                original: rest,
                k: m.stiffness * share * self.section / rest.max(1.0e-6),
                compression: m.compression,
                strength: if edge || !self.sheet {
                    Strength {
                        yield_load: scale(m.strength.yield_load),
                        break_load: scale(m.strength.break_load),
                        ductility: m.strength.ductility,
                    }
                } else {
                    Strength::default()
                },
                strain: 0.0,
                load: 0.0,
                force: 0.0,
                edge,
            }
        };
        self.beams.push(beam);
    }

    /// Build each particle's beam list.
    fn index_beams(&mut self) {
        let mut counts = vec![0u32; self.positions.len() + 1];
        for beam in &self.beams {
            counts[beam.a as usize + 1] += 1;
            counts[beam.b as usize + 1] += 1;
        }
        for i in 1..counts.len() {
            counts[i] += counts[i - 1];
        }
        self.offsets = counts.clone();
        self.entries = vec![0; self.beams.len() * 2];
        for (index, beam) in self.beams.iter().enumerate() {
            for end in [beam.a, beam.b] {
                self.entries[counts[end as usize] as usize] = index as u32;
                counts[end as usize] += 1;
            }
        }
    }

    /// Duplicate particles the cuts have separated, then make a beam of every
    /// uncut edge and every fold. `old` holds existing beams by original edge.
    fn rebuild(&mut self, old: &HashMap<((u32, u32), bool), Beam>) {
        // A triangle torn on two edges holds nothing: leaving it would hang a
        // sliver from the tear.
        let (origin, cuts) = (&self.origin, &self.cuts);
        self.triangles.retain(|t| {
            let cut = |a: u32, b: u32| cuts.contains(&edge(origin[a as usize], origin[b as usize]));
            let torn = [cut(t[0], t[1]), cut(t[1], t[2]), cut(t[2], t[0])];
            torn.iter().filter(|&&c| c).count() < 2
        });
        self.split_particles();
        let key = |origin: &[u32], a: u32, b: u32| edge(origin[a as usize], origin[b as usize]);
        // Each edge's triangles, by current particles.
        let mut sides: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        for (t, tri) in self.triangles.iter().enumerate() {
            for corner in 0..3 {
                let (a, b) = (tri[corner], tri[(corner + 1) % 3]);
                sides.entry(edge(a, b)).or_default().push(t as u32);
            }
        }
        let mut edges: Vec<_> = sides.iter().map(|(&e, t)| (e, t.clone())).collect();
        edges.sort_unstable();
        self.beams.clear();
        let bending = self.material.bending;
        for ((a, b), triangles) in edges {
            let original = key(&self.origin, a, b);
            if self.cuts.contains(&original) {
                continue;
            }
            self.add_beam(a, b, 1.0, true, old.get(&(original, true)));
            if bending > 0.0 && triangles.len() == 2 {
                let opposite = |t: u32| {
                    let tri = self.triangles[t as usize];
                    tri.into_iter()
                        .find(|&v| v != a && v != b)
                        .expect("a triangle has three corners")
                };
                let (p, q) = (opposite(triangles[0]), opposite(triangles[1]));
                let fold = key(&self.origin, p, q);
                // Across a tear the two sides can be copies of one particle.
                if fold.0 != fold.1 {
                    self.add_beam(p, q, bending, false, old.get(&(fold, false)));
                }
            }
        }
        self.index_beams();
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
    /// Edges or beams broken so far.
    pub fn tears(&self) -> usize {
        self.tears
    }
    pub fn pin_count(&self) -> usize {
        self.pins.len()
    }
    /// The largest permanent strain of any beam or region.
    pub fn permanent_strain(&self) -> f32 {
        let beams = self.beams.iter().map(|b| b.strain);
        let regions = self.regions.iter().map(|r| r.strain);
        beams.chain(regions).fold(0.0, f32::max)
    }
    /// Total mass (kg).
    pub fn mass(&self) -> f32 {
        self.masses.iter().sum()
    }
    /// Separate pieces of surface, counted by shared beams.
    pub fn pieces(&self) -> usize {
        let mut parent: Vec<u32> = (0..self.positions.len() as u32).collect();
        fn find(parent: &mut [u32], mut i: u32) -> u32 {
            while parent[i as usize] != i {
                parent[i as usize] = parent[parent[i as usize] as usize];
                i = parent[i as usize];
            }
            i
        }
        if !self.sheet {
            return 1;
        }
        for beam in self.beams.iter().filter(|b| b.edge) {
            let (a, b) = (find(&mut parent, beam.a), find(&mut parent, beam.b));
            parent[a as usize] = b;
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
        if self.asleep {
            if !near.iter().any(|&i| obstacles[i as usize].awake) {
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
                if let Some((surface, normal)) = obstacle.nearest(point, reach) {
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
        // Contacts by particle: they are sorted by it.
        let mut first_contact = vec![0u32; self.positions.len() + 1];
        for contact in &self.contacts {
            first_contact[contact.particle as usize + 1] += 1;
        }
        for i in 1..first_contact.len() {
            first_contact[i] += first_contact[i - 1];
        }
        let caps: Vec<f32> = self.beams.iter().map(Beam::cap).collect();
        let penalties: Vec<f32> = (self.beams.iter_mut())
            .map(|beam| {
                beam.force *= FORCE_MEMORY;
                // The softer of its two stiffnesses: the multiplier reaches the
                // stiffer one over a few sweeps, without overshooting.
                (beam.k * beam.compression).max(beam.k * 0.01)
            })
            .collect();

        let count = self.positions.len();
        let mut pin_pushed = vec![Vec3::ZERO; self.pins.len()];
        let solid = !self.regions.is_empty();
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
                let obstacle = &obstacles[contact.obstacle as usize];
                let local = self.masses[i] * CONTACT_SPREAD;
                let share = if obstacle.mass > 0.0 {
                    obstacle.mass / (obstacle.mass + local)
                } else {
                    1.0
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
        for iteration in 0..if solid { 0 } else { ITERATIONS } {
            // Alternating direction carries load through a mesh both ways.
            for n in 0..count {
                let i = if iteration % 2 == 0 { n } else { count - 1 - n };
                let mass = self.masses[i] * scale;
                let x = self.positions[i];
                let mut gradient = (x - inertial[i]) * mass;
                let mut hessian = Mat3::from_diagonal(Vec3::splat(mass));
                let entries = &self.entries[self.offsets[i] as usize..self.offsets[i + 1] as usize];
                for &index in entries {
                    let beam = &self.beams[index as usize];
                    let other = if beam.a as usize == i { beam.b } else { beam.a };
                    let d = x - self.positions[other as usize];
                    let length = d.length();
                    if length < 1.0e-7 {
                        continue;
                    }
                    let n = d / length;
                    let cap = caps[index as usize];
                    let k = penalties[index as usize];
                    let force = beam.force_at(length, k, cap);
                    // A saturated beam gives way: no stiffness holds it. Where few
                    // beams meet, as on a torn fragment, nothing balances a force,
                    // and a step on the penalty alone would overshoot by its ratio
                    // to the true stiffness.
                    let k = if force.abs() >= cap {
                        0.0
                    } else if entries.len() < SUPPORTED_BEAMS {
                        beam.stiffness(force).max(k)
                    } else {
                        k
                    };
                    gradient += n * force;
                    let along = outer(n, n);
                    hessian += along * k;
                    if force > 0.0 {
                        // Tension's geometric stiffness; compression's is not
                        // positive definite, and left out.
                        hessian += (Mat3::IDENTITY - along) * (force / length);
                    }
                }
                let pin = self.pinned[i];
                if pin != NONE {
                    let k = PIN_STIFFNESS * mass;
                    gradient += (x - targets[pin as usize]) * k;
                    hessian += Mat3::from_diagonal(Vec3::splat(k));
                }
                let before = x;
                let mut x = x - solve(hessian, gradient);
                let (from, to) = (first_contact[i] as usize, first_contact[i + 1] as usize);
                for contact in &mut self.contacts[from..to] {
                    // What presses it into the surface is what the surface
                    // holds back: the block's unbalanced force along the normal.
                    let touching = contact.normal.dot(before) - contact.offset < 1.0e-4;
                    contact.force = if touching {
                        contact.normal.dot(gradient).max(0.0)
                    } else {
                        0.0
                    };
                    let gap = contact.normal.dot(x) - contact.offset;
                    contact.depth = (-gap).max(0.0);
                    if gap < 0.0 {
                        x -= contact.normal * gap;
                        contact.pushed -= gap;
                    }
                }
                self.positions[i] = x;
            }
            for (index, beam) in self.beams.iter_mut().enumerate() {
                let (a, b) = (beam.a as usize, beam.b as usize);
                let length = self.positions[a].distance(self.positions[b]);
                beam.force = beam.force_at(length, penalties[index], caps[index]);
            }
        }

        // Friction, then velocities and what the rigid bodies feel.
        for contact in &self.contacts {
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
        // Beams lose speed along their length: cloth rings at high frequency,
        // which the multipliers would otherwise keep alive.
        for beam in self.beams.iter().filter(|_| BEAM_DAMPING > 0.0) {
            let (a, b) = (beam.a as usize, beam.b as usize);
            let d = self.positions[a] - self.positions[b];
            let length = d.length();
            if length < 1.0e-7 {
                continue;
            }
            let n = d / length;
            let closing = n.dot((self.positions[a] - start[a]) - (self.positions[b] - start[b]));
            let (ma, mb) = (self.masses[a], self.masses[b]);
            let shift = closing * BEAM_DAMPING / (ma + mb);
            if self.pinned[a] == NONE {
                self.positions[a] -= n * (shift * mb);
            }
            if self.pinned[b] == NONE {
                self.positions[b] += n * (shift * ma);
            }
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
        let mut broken = false;
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
                (self.positions[particle] - targets[i]) * (PIN_STIFFNESS * mass * scale)
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
        // The rest came from contacts: share it by how hard each pushed.
        let total: f32 = self.contacts.iter().map(|c| c.pushed).sum();
        if total > 0.0 {
            let first = reactions.len();
            for contact in self.contacts.iter().filter(|c| c.pushed > 0.0) {
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
                let closing = (speed - obstacle.velocity - gravity * dt).dot(impulse / size);
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
        for (index, beam) in self.beams.iter_mut().enumerate() {
            let length = self.positions[beam.a as usize].distance(self.positions[beam.b as usize]);
            let cap = caps[index];
            if beam.force.abs() >= cap {
                // Stretch beyond what the force accounts for is now permanent.
                let stretch = length - beam.rest;
                let elastic = beam.force / beam.stiffness(beam.force);
                let flow = stretch - elastic;
                if flow * elastic > 0.0 {
                    beam.rest += flow;
                    beam.strain += flow.abs() / beam.original.max(1.0e-6);
                }
            }
            beam.load += (beam.force.abs() - beam.load) * blend;
            let s = beam.strength;
            if s.break_load.is_finite()
                && (beam.load >= s.break_load
                    || (s.yield_load < s.break_load && beam.strain >= s.ductility))
            {
                beam.k = -1.0;
                broken = true;
            }
        }
        if broken {
            self.break_beams();
        }

        if fastest.sqrt() < SLEEP_SPEED && !released {
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
        let yield_strain = m.strength.yield_load / m.stiffness;
        for region in &mut self.regions {
            if region.strain >= m.strength.ductility {
                continue;
            }
            let (_, deformation) = region.deformation(&self.positions, &self.masses);
            region.track_rotation(deformation);
            let stretch = Mat3::from_quat(region.rotation).transpose() * deformation;
            let excess = stretch - Mat3::IDENTITY;
            let strain = (excess.x_axis.length_squared()
                + excess.y_axis.length_squared()
                + excess.z_axis.length_squared())
            .sqrt();
            let threshold = yield_strain + 2.0 * settling / region.thickness;
            if strain > threshold {
                let flow = (strain - threshold) / strain;
                let change = Mat3::IDENTITY + excess * flow;
                for q in &mut region.rest {
                    *q = change * *q;
                }
                region.update_moment(&self.masses);
                region.strain += strain - threshold;
            }
        }
    }

    /// Remove beams marked broken, and tear a sheet along them.
    fn break_beams(&mut self) {
        let snapped = self.beams.iter().filter(|b| b.k < 0.0).count();
        self.tears += snapped;
        if !self.sheet {
            self.beams.retain(|b| b.k >= 0.0);
            self.index_beams();
            return;
        }
        let origin = &self.origin;
        let key = |b: &Beam| edge(origin[b.a as usize], origin[b.b as usize]);
        let cuts: Vec<_> = (self.beams.iter()).filter(|b| b.k < 0.0).map(key).collect();
        let old: HashMap<_, _> = (self.beams.iter())
            .filter(|b| b.k >= 0.0)
            .map(|b| ((key(b), b.edge), *b))
            .collect();
        self.cuts.extend(cuts);
        self.rebuild(&old);
    }
}

fn outer(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::from_cols(a * b.x, a * b.y, a * b.z)
}

/// `H⁻¹ g`, for the symmetric positive definite block of one particle.
fn solve(hessian: Mat3, gradient: Vec3) -> Vec3 {
    let determinant = hessian.determinant();
    if determinant.abs() > 1.0e-20 {
        hessian.inverse() * gradient
    } else {
        Vec3::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cut_across_a_sheet_splits_it_and_no_beam_bridges_the_tear() {
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
        let cuts: Vec<_> = (cloth.beams.iter())
            .filter(|b| b.edge && rows(b.a).min(rows(b.b)) == 1 && rows(b.a).max(rows(b.b)) == 2)
            .map(|b| edge(b.a, b.b))
            .collect();
        cloth.cuts.extend(cuts);
        cloth.rebuild(&HashMap::new());
        assert_eq!(cloth.pieces(), 2);
        assert!(
            (cloth.mass() - mass).abs() < 1.0e-5,
            "mass is shared, not made"
        );
        for beam in &cloth.beams {
            let (a, b) = (cloth.origin[beam.a as usize], cloth.origin[beam.b as usize]);
            assert_ne!(a, b, "a particle is tied to its own copy");
            assert!(!(rows(a).min(rows(b)) == 1 && rows(a).max(rows(b)) == 2 && beam.edge));
        }
    }
}
