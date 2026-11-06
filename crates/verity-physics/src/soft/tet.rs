//! Finite element solids: stable neo-Hookean tetrahedra (Smith et al. 2018) in the
//! same implicit solve as cloth. Plasticity moves each element's rest shape, and
//! fracture parts elements along their faces.

use super::{SoftBody, TEAR_SPEED, TEARS_PER_STEP, TIP_LOAD, fan_root, index_by_particle, outer};
use glam::{Mat3, Vec3};
use std::collections::HashMap;

/// One tetrahedron of a solid, its corners in positive order.
#[derive(Debug, Clone, Copy)]
pub(super) struct Tet {
    pub(super) particles: [u32; 4],
    /// Each corner's gradient of the deformation, from the rest shape. Each also
    /// points in from the face opposite that corner. Plasticity changes these.
    pub(super) gradient: [Vec3; 4],
    pub(super) volume: f32,
    /// How near failing, averaged: 1 fails.
    pub(super) load: f32,
    /// Plastic strain so far.
    pub(super) strain: f32,
}

/// Each face, opposite the corner of the same place, wound outward.
const TET_FACES: [[usize; 3]; 4] = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];

impl Tet {
    pub(super) fn new(mut particles: [u32; 4], at: impl Fn(u32) -> Vec3) -> Self {
        let [a, b, c, d] = particles.map(&at);
        if Mat3::from_cols(b - a, c - a, d - a).determinant() < 0.0 {
            particles.swap(2, 3);
        }
        let [a, b, c, d] = particles.map(&at);
        let shape = Mat3::from_cols(b - a, c - a, d - a);
        let rows = shape.inverse().transpose();
        let (x, y, z) = (rows.x_axis, rows.y_axis, rows.z_axis);
        Self {
            particles,
            gradient: [-(x + y + z), x, y, z],
            volume: shape.determinant() / 6.0,
            load: 0.0,
            strain: 0.0,
        }
    }

    pub(super) fn deformation(&self, x: [Vec3; 4]) -> Mat3 {
        (x.iter().zip(&self.gradient)).fold(Mat3::ZERO, |sum, (&x, &g)| sum + outer(x, g))
    }
}

/// A tetrahedron's deformation, its first Piola-Kirchhoff stress, and the
/// cofactors of its deformation: how its volume changes with it.
#[derive(Debug, Clone, Copy)]
pub(super) struct TetState {
    pub(super) deformation: Mat3,
    /// Stress along the cofactors: their share of the stress.
    pub(super) pressure: f32,
    pub(super) stress: Mat3,
    pub(super) cofactor: Mat3,
}

impl TetState {
    /// Stable neo-Hookean, without the log term: it needs no decomposition,
    /// and pushes an inverted element back out.
    pub(super) fn new(deformation: Mat3, shear: f32, lame: f32) -> Self {
        let (a, b, c) = (deformation.x_axis, deformation.y_axis, deformation.z_axis);
        let cofactor = Mat3::from_cols(b.cross(c), c.cross(a), a.cross(b));
        let volume = a.dot(cofactor.x_axis);
        let pressure = lame * (volume - 1.0) - shear;
        Self {
            deformation,
            pressure,
            stress: deformation * shear + cofactor * pressure,
            cofactor,
        }
    }
}

impl Default for TetState {
    fn default() -> Self {
        Self::new(Mat3::IDENTITY, 0.0, 0.0)
    }
}

impl SoftBody {
    /// Make a solid of tetrahedra instead of shape matching: slower, but with a
    /// real material, whose `stiffness` and `poisson` mean what they say, and
    /// which tears along real faces. Fibres lie along its local X, or
    /// [`SoftBody::set_warp`]. Call this before pinning. Does nothing to a sheet.
    pub fn finite_elements(mut self) -> Self {
        let Some(grid) = self.grid.clone() else {
            return self;
        };
        let [nx, ny] = [grid.cells[0] + 1, grid.cells[1] + 1];
        let node = |[x, y, z]: [u32; 3]| grid.index[&((z * ny + y) * nx + x)];
        self.regions.clear();
        self.masses.fill(0.0);
        for &[x, y, z] in &grid.filled {
            // Six around the cell's diagonal, one for each order of its axes,
            // so that neighbouring cells' faces always match.
            for axes in [
                [0, 1, 2],
                [0, 2, 1],
                [1, 0, 2],
                [1, 2, 0],
                [2, 0, 1],
                [2, 1, 0],
            ] {
                let mut at = [x, y, z];
                let mut particles = [node(at); 4];
                for (corner, axis) in axes.into_iter().enumerate() {
                    at[axis] += 1;
                    particles[corner + 1] = node(at);
                }
                let tet = Tet::new(particles, |p| self.rest[p as usize]);
                for p in tet.particles {
                    self.masses[p as usize] += self.material.density * tet.volume / 4.0;
                }
                self.tets.push(tet);
            }
        }
        let mut sides: HashMap<[u32; 3], u32> = HashMap::new();
        for tet in &self.tets {
            for face in 0..4 {
                *sides.entry(self.face_key(tet, face)).or_default() += 1;
            }
        }
        self.uncut = (sides.into_iter())
            .filter(|&(_, count)| count == 2)
            .map(|(face, _)| face)
            .collect();
        self.sheet = true;
        self.warp = grid.rotation * Vec3::X;
        self.rebuild_tets();
        self
    }

    /// A face of a tetrahedron, whoever's copies of its particles it holds.
    fn face_key(&self, tet: &Tet, face: usize) -> [u32; 3] {
        let mut key = TET_FACES[face].map(|c| self.origin[tet.particles[c] as usize]);
        key.sort_unstable();
        key
    }

    /// After cuts: give each group of elements around a particle that no longer
    /// holds together its own copy, then find the surface, torn faces included.
    pub(super) fn rebuild_tets(&mut self) {
        let mut around: Vec<Vec<u32>> = vec![Vec::new(); self.positions.len()];
        for (t, tet) in self.tets.iter().enumerate() {
            for p in tet.particles {
                around[p as usize].push(t as u32);
            }
        }
        for v in 0..around.len() as u32 {
            let fan = &around[v as usize];
            if !self.cut_ends.contains_key(&self.origin[v as usize]) {
                continue;
            }
            let mut group: Vec<usize> = (0..fan.len()).collect();
            for i in 0..fan.len() {
                for j in i + 1..fan.len() {
                    let (s, t) = (&self.tets[fan[i] as usize], &self.tets[fan[j] as usize]);
                    let holds = (0..4).any(|face| {
                        let corners = TET_FACES[face].map(|c| s.particles[c]);
                        corners.contains(&v)
                            && corners.iter().all(|p| t.particles.contains(p))
                            && self.uncut.contains(&self.face_key(s, face))
                    });
                    if holds {
                        let (a, b) = (fan_root(&mut group, i), fan_root(&mut group, j));
                        group[a] = b;
                    }
                }
            }
            let roots: Vec<usize> = (0..fan.len()).map(|i| fan_root(&mut group, i)).collect();
            let mut distinct = roots.clone();
            distinct.sort_unstable();
            distinct.dedup();
            let share = self.masses[v as usize] / distinct.len() as f32;
            for &root in distinct.iter().skip(1) {
                let copy = self.copy_particle(v, share);
                for (i, &t) in fan.iter().enumerate() {
                    if roots[i] == root {
                        for corner in &mut self.tets[t as usize].particles {
                            if *corner == v {
                                *corner = copy;
                            }
                        }
                    }
                }
            }
            self.masses[v as usize] = share;
        }
        // A face only one element holds is the surface.
        let mut faces: HashMap<[u32; 3], (u32, [u32; 3])> = HashMap::new();
        for tet in &self.tets {
            for corners in TET_FACES {
                let wound = corners.map(|c| tet.particles[c]);
                let mut key = wound;
                key.sort_unstable();
                faces.entry(key).or_insert((0, wound)).0 += 1;
            }
        }
        let mut surface: Vec<_> = (faces.into_iter())
            .filter(|(_, (count, _))| *count == 1)
            .collect();
        surface.sort_unstable_by_key(|(key, _)| *key);
        self.triangles = surface.into_iter().map(|(_, (_, wound))| wound).collect();

        let count = self.positions.len();
        (self.offsets, self.entries) = index_by_particle(count, std::iter::empty());
        (self.hinge_offsets, self.hinge_entries) = index_by_particle(count, std::iter::empty());
        self.index_tets();
    }

    pub(super) fn index_tets(&mut self) {
        let corners = (self.tets.iter().enumerate())
            .flat_map(|(t, tet)| (0..4).map(move |c| (tet.particles[c], t as u32 * 4 + c as u32)));
        (self.tet_offsets, self.tet_entries) = index_by_particle(self.positions.len(), corners);
    }

    /// Let elements past yield flow, then tear those near failing along the
    /// face nearest failing: the only planes a mesh of them can part on.
    pub(super) fn fail_tets(&mut self, blend: f32, dt: f32) {
        let m = self.material;
        let (shear, lame) = (self.shear, self.lame);
        let yield_strain = m.strength.yield_load / m.stiffness;
        let ductile = m.strength.yield_load < m.strength.break_load;
        let mut failing = Vec::new();
        for t in 0..self.tets.len() {
            let tet = &self.tets[t];
            let mut f = tet.deformation(tet.particles.map(|p| self.positions[p as usize]));
            if ductile {
                let strain = (f.transpose() * f - Mat3::IDENTITY) * 0.5;
                let mean = (strain.x_axis.x + strain.y_axis.y + strain.z_axis.z) / 3.0;
                let shape = strain - Mat3::IDENTITY * mean;
                let size = (shape.x_axis.length_squared()
                    + shape.y_axis.length_squared()
                    + shape.z_axis.length_squared())
                .sqrt();
                if size > yield_strain {
                    // Move the rest shape toward this one, keeping its volume.
                    let flow = Mat3::IDENTITY + shape * ((size - yield_strain) / size);
                    let undo = flow.inverse() * flow.determinant().cbrt();
                    let tet = &mut self.tets[t];
                    tet.gradient = tet.gradient.map(|g| undo * g);
                    tet.strain += size - yield_strain;
                    f *= undo;
                }
            }
            let tet = &self.tets[t];
            let stress = TetState::new(f, shear, lame).stress;
            let mut worst = (0.0, None);
            for face in 0..4 {
                let normal = -tet.gradient[face].normalize_or_zero();
                let spatial = f * normal;
                let traction = (stress * normal).dot(spatial) / spatial.length().max(1.0e-9);
                let along = normal.dot(self.warp);
                let across = (1.0 - along * along).max(0.0).sqrt();
                let ratio = if ductile {
                    tet.strain / m.strength.ductility
                } else {
                    traction / m.strength_at(along, across)
                };
                let key = self.face_key(tet, face);
                if ratio > worst.0 && ratio >= TIP_LOAD * 0.5 && self.uncut.contains(&key) {
                    worst = (ratio, Some(key));
                }
            }
            let tet = &mut self.tets[t];
            tet.load += (worst.0 - tet.load) * blend;
            if let (true, Some(face)) = (tet.load >= TIP_LOAD, worst.1) {
                failing.push((t, face));
            }
        }
        let passes = ((TEAR_SPEED * dt / (2.0 * self.radius)).ceil() as usize).max(1);
        let limit = 1 + self.tets.len() / TEARS_PER_STEP;
        let mut changed = false;
        for pass in 0..passes {
            // A face along a tear's edge shares two corners with it.
            let at_front = |face: &[u32; 3]| {
                face.iter()
                    .filter(|p| self.cut_ends.contains_key(p))
                    .count()
                    >= 2
            };
            let mut steps: Vec<_> = (failing.iter())
                .filter(|(t, face)| {
                    let load = self.tets[*t].load;
                    self.uncut.contains(face)
                        && (load >= 1.0 || (at_front(face) && load >= TIP_LOAD))
                })
                .map(|&(t, face)| (self.tets[t].load, t, face, at_front(&face)))
                .collect();
            // As in a sheet: while a tear grows no other starts.
            let growing = steps.iter().any(|step| step.3);
            steps.retain(|step| step.3 == growing);
            steps.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.2.cmp(&b.2)));
            steps.truncate(match (growing, pass) {
                (true, _) => limit,
                (false, 0) => 1,
                (false, _) => 0,
            });
            if steps.is_empty() {
                break;
            }
            changed = true;
            for (_, t, face, _) in steps {
                if self.uncut.remove(&face) {
                    self.cut_faces.push(face);
                    self.tears += 1;
                    for p in face {
                        *self.cut_ends.entry(p).or_default() += 1;
                    }
                }
                self.tets[t].load = 0.0;
            }
        }
        if changed {
            self.rebuild_tets();
        }
    }
}
