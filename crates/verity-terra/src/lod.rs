//! Level of detail for heightfield surfaces: a quadtree of patches per face,
//! whose leaves split near the eye and merge away from it. Patches mesh on
//! worker threads and swap in a group at a time (a parent for its children,
//! or the children for their parent), so the surface never shows a hole or
//! two levels on top of each other. [`crate::planet::Planet`] and [`Far`]
//! are the surfaces.

use crate::{Landscape, MeshData};
use glam::{DVec2, DVec3, Vec3};
use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc,
        mpsc::{Receiver, Sender, channel},
    },
};

/// Vertices along a patch side.
pub const GRID: usize = 33;

/// A node of one face's quadtree: `x` and `y` count patches across the face at `level`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Patch {
    pub face: u8,
    pub level: u8,
    pub x: u32,
    pub y: u32,
}

impl Patch {
    pub fn root(face: u8) -> Self {
        Self {
            face,
            level: 0,
            x: 0,
            y: 0,
        }
    }

    /// A unique key under 2⁵⁶, for retained meshes.
    pub fn key(self) -> u64 {
        self.face as u64 | (self.level as u64) << 3 | (self.x as u64) << 8 | (self.y as u64) << 32
    }

    pub fn children(self) -> [Patch; 4] {
        std::array::from_fn(|i| Patch {
            face: self.face,
            level: self.level + 1,
            x: self.x * 2 + (i as u32 & 1),
            y: self.y * 2 + (i as u32 >> 1),
        })
    }

    fn parent(self) -> Option<Patch> {
        (self.level > 0).then(|| Patch {
            face: self.face,
            level: self.level - 1,
            x: self.x / 2,
            y: self.y / 2,
        })
    }

    /// Whether `other` is this patch or lies inside it.
    pub fn contains(self, other: Patch) -> bool {
        let Some(down) = other.level.checked_sub(self.level) else {
            return false;
        };
        other.face == self.face && other.x >> down == self.x && other.y >> down == self.y
    }

    /// `(s, t)` in [0, 1]² across this patch, as a point in [0, 1]² across its face.
    pub fn on_face(self, s: f64, t: f64) -> DVec2 {
        (DVec2::new(self.x as f64 + s, self.y as f64 + t)) * 0.5f64.powi(self.level as i32)
    }
}

/// A surface drawn as quadtree patches.
pub trait Surface: Send + Sync + 'static {
    /// The patches to draw from `eye`: a set covering the surface once.
    fn leaves(&self, eye: Vec3) -> Vec<Patch>;
    fn mesh(&self, patch: Patch) -> MeshData;
    /// Where the patch is, to mesh the nearest first.
    fn center(&self, patch: Patch) -> Vec3;
}

/// Patch meshes to add and patches to remove, swapped in together so the surface never has holes.
#[derive(Default)]
pub struct LodUpdate {
    pub added: Vec<(Patch, MeshData)>,
    pub removed: Vec<Patch>,
}

/// Keeps a surface's patches matched to the eye.
pub struct PatchLod<S> {
    pub surface: Arc<S>,
    /// Patches being meshed at once, at most.
    pub budget: usize,
    shown: HashSet<Patch>,
    ready: HashMap<Patch, MeshData>,
    running: HashSet<Patch>,
    sender: Sender<(Patch, MeshData)>,
    receiver: Receiver<(Patch, MeshData)>,
}

impl<S: Surface> PatchLod<S> {
    pub fn new(surface: S) -> Self {
        let (sender, receiver) = channel();
        Self {
            surface: Arc::new(surface),
            budget: 48,
            shown: HashSet::new(),
            ready: HashMap::new(),
            running: HashSet::new(),
            sender,
            receiver,
        }
    }

    pub fn shown(&self) -> usize {
        self.shown.len()
    }

    /// Start meshing the nearest missing patches on workers, and swap in every
    /// group whose meshes are all ready. `settled` in the result means the
    /// patches on show are exactly those the eye wants.
    pub fn update(&mut self, eye: Vec3) -> (LodUpdate, bool) {
        let wanted: HashSet<Patch> = self.surface.leaves(eye).into_iter().collect();
        while let Ok((patch, mesh)) = self.receiver.try_recv() {
            self.running.remove(&patch);
            if wanted.contains(&patch) && !self.shown.contains(&patch) {
                self.ready.insert(patch, mesh);
            }
        }
        self.ready.retain(|patch, _| wanted.contains(patch));
        let mut missing: Vec<Patch> = (wanted.iter())
            .filter(|p| {
                !self.shown.contains(p) && !self.ready.contains_key(p) && !self.running.contains(p)
            })
            .copied()
            .collect();
        let distance = |p: &Patch| self.surface.center(*p).distance_squared(eye);
        missing.sort_by(|a, b| distance(a).total_cmp(&distance(b)));
        missing.truncate(self.budget.saturating_sub(self.running.len()));
        for patch in missing {
            self.running.insert(patch);
            let (surface, sender) = (self.surface.clone(), self.sender.clone());
            rayon::spawn(move || {
                // Only fails once the LOD is gone and nobody wants the result.
                let _ = sender.send((patch, surface.mesh(patch)));
            });
        }
        let mut update = LodUpdate::default();
        // Splits: a patch the eye wants finer gives way once every patch inside it is ready.
        let coarse: Vec<Patch> = (self.shown.iter())
            .filter(|s| !wanted.contains(s))
            .copied()
            .collect();
        for patch in coarse {
            let inside: Vec<Patch> = (wanted.iter())
                .filter(|w| patch.contains(**w))
                .copied()
                .collect();
            if !inside.is_empty() && inside.iter().all(|w| self.ready.contains_key(w)) {
                self.shown.remove(&patch);
                update.removed.push(patch);
                for w in inside {
                    let mesh = self.ready.remove(&w).expect("checked ready");
                    self.shown.insert(w);
                    update.added.push((w, mesh));
                }
            }
        }
        // Merges and new ground: a ready patch under no shown one replaces any shown inside it.
        let ready: Vec<Patch> = self.ready.keys().copied().collect();
        for patch in ready {
            let covered = std::iter::successors(patch.parent(), |p| p.parent())
                .any(|p| self.shown.contains(&p));
            if covered {
                continue;
            }
            let inside: Vec<Patch> = (self.shown.iter())
                .filter(|s| patch.contains(**s))
                .copied()
                .collect();
            for s in inside {
                self.shown.remove(&s);
                update.removed.push(s);
            }
            let mesh = self.ready.remove(&patch).expect("listed ready");
            self.shown.insert(patch);
            update.added.push((patch, mesh));
        }
        let settled =
            self.shown.len() == wanted.len() && wanted.iter().all(|w| self.shown.contains(w));
        (update, settled)
    }
}

/// A patch's mesh: [`GRID`]² vertices from `sample(s, t)` over [0, 1]², which
/// gives a vertex's position, its up direction and its terrain uv (the ground
/// height and wetness), with a skirt `skirt` metres deep hanging below its
/// edges to hide cracks between levels. `color(up, height, cos slope)` gives
/// vertex colours, if any. Triangles wind like the planet's patches.
pub(crate) fn grid_mesh(
    sample: impl Fn(f64, f64) -> (DVec3, DVec3, [f32; 2]),
    color: Option<&dyn Fn(DVec3, f64, f64) -> [f32; 3]>,
    skirt: f64,
) -> MeshData {
    const SIDE: usize = GRID + 2;
    let step = 1.0 / (GRID - 1) as f64;
    // One ring beyond the patch, for normals that match the neighbours'.
    let samples: Vec<_> = (0..SIDE * SIDE)
        .map(|k| {
            let (i, j) = (k % SIDE, k / SIDE);
            sample((i as f64 - 1.0) * step, (j as f64 - 1.0) * step)
        })
        .collect();
    let at = |i: usize, j: usize| (j + 1) * SIDE + i + 1;
    let vertices = GRID * GRID + 4 * GRID;
    let mut mesh = MeshData {
        positions: Vec::with_capacity(vertices),
        normals: Vec::with_capacity(vertices),
        terrain_uv: Vec::with_capacity(vertices),
        indices: Vec::with_capacity((GRID - 1) * (GRID - 1) * 6 + 4 * (GRID - 1) * 6),
        colors: Vec::with_capacity(if color.is_some() { vertices } else { 0 }),
        bounds: (Vec3::INFINITY, Vec3::NEG_INFINITY),
    };
    let push = |mesh: &mut MeshData, p: DVec3, normal: [f32; 3], uv: [f32; 2]| {
        let p = p.as_vec3();
        mesh.bounds = (mesh.bounds.0.min(p), mesh.bounds.1.max(p));
        mesh.positions.push(p.to_array());
        mesh.normals.push(normal);
        mesh.terrain_uv.push(uv);
    };
    for j in 0..GRID {
        for i in 0..GRID {
            let k = at(i, j);
            let (point, up, uv) = samples[k];
            let du = samples[k + 1].0 - samples[k - 1].0;
            let dv = samples[k + SIDE].0 - samples[k - SIDE].0;
            let normal = du.cross(dv).normalize_or(up);
            push(&mut mesh, point, normal.as_vec3().to_array(), uv);
            if let Some(color) = color {
                mesh.colors.push(color(up, uv[0] as f64, normal.dot(up)));
            }
        }
    }
    for j in 0..GRID - 1 {
        for i in 0..GRID - 1 {
            let a = (j * GRID + i) as u32;
            let (b, c, d) = (a + 1, a + GRID as u32, a + GRID as u32 + 1);
            mesh.indices.extend([a, d, b, a, c, d]);
        }
    }
    // The boundary counterclockwise seen from outside, each edge hung with a skirt.
    let ring: Vec<usize> = (0..GRID - 1)
        .chain((0..GRID - 1).map(|j| j * GRID + GRID - 1))
        .chain((1..GRID).rev().map(|i| (GRID - 1) * GRID + i))
        .chain((1..GRID).rev().map(|j| j * GRID))
        .collect();
    let first = mesh.positions.len() as u32;
    for &v in &ring {
        let (point, up, _) = samples[at(v % GRID, v / GRID)];
        let (normal, uv) = (mesh.normals[v], mesh.terrain_uv[v]);
        push(&mut mesh, point - up * skirt, normal, uv);
        if color.is_some() {
            mesh.colors.push(mesh.colors[v]);
        }
    }
    for e in 0..ring.len() {
        let (a, b) = (ring[e] as u32, ring[(e + 1) % ring.len()] as u32);
        let (a2, b2) = (first + e as u32, first + ((e + 1) % ring.len()) as u32);
        mesh.indices.extend([a, b, a2, b, b2, a2]);
    }
    mesh
}

/// How far below the ground [`Far`] patches sit, in metres, so the voxel chunks drawn over them win.
const SINK: f64 = 1.0;

/// A landscape beyond its voxel chunks, as heightfield patches, one face
/// over the whole map. They sit [`SINK`] below the ground and carry no
/// colours, so the terrain material shades them like the chunks.
/// shortcut: a hole dug deeper than `SINK` where far patches reach under the voxels shows them;
/// cut the patches out under the streaming radius if that shows.
pub struct Far {
    pub landscape: Arc<Landscape>,
    /// Patches split while closer than this many of their widths.
    pub detail: f32,
    max_level: u8,
}

impl Far {
    pub fn new(landscape: Arc<Landscape>) -> Self {
        // Splitting past one vertex per height sample adds nothing.
        let spacing = 2.0 * landscape.half() as f64 / (GRID - 1) as f64;
        let max_level = (spacing / landscape.cell as f64)
            .log2()
            .ceil()
            .clamp(0.0, 20.0) as u8;
        Self {
            landscape,
            detail: 2.0,
            max_level,
        }
    }

    fn width(&self, patch: Patch) -> f64 {
        2.0 * self.landscape.half() as f64 * 0.5f64.powi(patch.level as i32)
    }

    fn point(&self, patch: Patch, s: f64, t: f64) -> DVec2 {
        (patch.on_face(s, t) * 2.0 - 1.0) * self.landscape.half() as f64
    }
}

impl Surface for Far {
    fn leaves(&self, eye: Vec3) -> Vec<Patch> {
        let mut stack = vec![Patch::root(0)];
        let mut leaves = Vec::new();
        while let Some(patch) = stack.pop() {
            let distance = self.center(patch).distance(eye) as f64;
            if patch.level < self.max_level
                && distance < self.width(patch) * self.detail.max(0.5) as f64
            {
                stack.extend(patch.children());
            } else {
                leaves.push(patch);
            }
        }
        leaves
    }

    fn mesh(&self, patch: Patch) -> MeshData {
        let land = &self.landscape;
        grid_mesh(
            |s, t| {
                let p = self.point(patch, s, t).as_vec2();
                let h = land.height(p.x, p.y);
                let point = DVec3::new(p.x as f64, p.y as f64, h as f64 - SINK);
                (point, DVec3::Z, [h, land.wetness(p.x, p.y)])
            },
            None,
            self.width(patch) * 0.04 + 1.0,
        )
    }

    fn center(&self, patch: Patch) -> Vec3 {
        let p = self.point(patch, 0.5, 0.5).as_vec2();
        p.extend(self.landscape.height(p.x, p.y))
    }
}
