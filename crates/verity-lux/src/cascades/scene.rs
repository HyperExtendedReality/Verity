//! The GI scene: a two-level BVH over a loaded glTF scene, packed into the
//! word buffers `shaders/traversal.slang` reads. Every word is a `UVec4`.
//!
//! Geometry words: a header `(1, vertex base, VERTEX_WORDS, vertex count)`,
//! one `(a, b, c, 0)` index word per triangle, padding up to the vertex base,
//! then [`VERTEX_WORDS`] words per vertex: position (w: flags), normal,
//! tangent, texture coordinates and color.
//!
//! Scene words: a prefix `(end of instance words, instance base, instance
//! count, INSTANCE_WORDS)`, the TLAS nodes, the BLAS nodes, then
//! [`INSTANCE_WORDS`] words per instance. Nodes take two words, `(min,
//! first)` and `(max, count)`: a leaf (count > 0) spans `count` elements from
//! `first`; an inner node's children are nodes `first` and `first + 1`, in
//! the combined TLAS + BLAS node array, the TLAS with room for its largest
//! tree so the BLASes stay put when it is rebuilt. After the instances comes one
//! render-layer mask word: Verity has a single layer (bit 0), and lux's
//! traversal only hits instances whose mask meets the view's (see `LAYER_MASK`).

use super::surface::{self, Charts};
use glam::{Affine3A, Mat4, UVec2, UVec3, UVec4, Vec3};
use rayon::prelude::*;
use std::{collections::BTreeMap, mem, ops::Range};
use verity_render::{
    RetainedChange, RetainedMesh, Vertex,
    asset::{AlphaMode, SceneData},
};

pub const VERTEX_WORDS: u32 = 5;
/// The surface atlas's texel budget; past it texels grow.
const MAX_SURFACE_TEXELS: usize = 1 << 21;
pub const INSTANCE_WORDS: usize = 9;
/// Instance flags, as in the traversal shader: bits 0-1 are the cull mode.
pub const FLAG_CULL_BACK: u32 = 1;
pub const FLAG_MIRRORED: u32 = 1 << 2;
pub const FLAG_DOUBLE_SIDED: u32 = 1 << 3;
/// The one render layer every instance and view is on.
pub const LAYER_MASK: UVec4 = UVec4::new(1, 0, 0, 0);

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BvhNode {
    pub min: Vec3,
    /// Leaf: first element. Inner: index of the left child; the right follows.
    pub first: u32,
    pub max: Vec3,
    /// Elements in a leaf; zero for an inner node.
    pub count: u32,
}

#[derive(Clone, Copy)]
struct Triangle {
    /// Global vertex indices.
    indices: UVec3,
    min: Vec3,
    max: Vec3,
}

#[derive(Clone, Copy)]
struct Instance {
    world_from_local: Affine3A,
    local_from_world: Affine3A,
    min: Vec3,
    max: Vec3,
    /// BLAS root, as an index into the BLAS node array.
    root: u32,
    first: u32,
    count: u32,
    flags: u32,
    material: u32,
    /// The material's emitted radiance, before its texture.
    emission: Vec3,
    /// Index into [`GiScene::charts`], or `u32::MAX` when the surface cache
    /// leaves the instance out.
    chart: u32,
    /// Instance word 7's `y` (island table base + 1, 0 = uncharted) and `z`
    /// (texels per local metre), set by [`GiScene::pack`].
    surface: UVec2,
}

/// One retained mesh's BLAS, with its indices and nodes relative to itself, so
/// it can be replaced without touching the others.
struct RetainedBlas {
    triangles: Vec<Triangle>,
    nodes: Vec<BvhNode>,
    vertex_words: Vec<UVec4>,
    slot: Option<Slot>,
}

/// Where a retained mesh sits in the packed words, with room to change: edits
/// then rewrite only its own words, not tens of megabytes of scene. A mesh that
/// outgrows its slot moves to the [`Tail`].
#[derive(Clone, Copy)]
struct Slot {
    triangles: u32,
    triangle_room: u32,
    vertices: u32,
    vertex_room: u32,
    nodes: u32,
    node_room: u32,
}

/// Free space after the slots: next free element and the end, per section.
#[derive(Clone, Copy, Default)]
struct Tail {
    triangles: [u32; 2],
    vertices: [u32; 2],
    nodes: [u32; 2],
}

impl Tail {
    /// Room for a mesh and for it to double, if there is any left.
    fn place(&mut self, blas: &RetainedBlas) -> Option<Slot> {
        let take = |section: &[u32; 2], used: usize| {
            let room = 2 * used as u32 + 8;
            (section[0] + room <= section[1]).then_some(room)
        };
        let triangle_room = take(&self.triangles, blas.triangles.len())?;
        let vertex_room = take(
            &self.vertices,
            blas.vertex_words.len() / VERTEX_WORDS as usize,
        )?;
        let node_room = take(&self.nodes, blas.nodes.len())?;
        let slot = Slot {
            triangles: self.triangles[0],
            triangle_room,
            vertices: self.vertices[0],
            vertex_room,
            nodes: self.nodes[0],
            node_room,
        };
        self.triangles[0] += triangle_room;
        self.vertices[0] += vertex_room;
        self.nodes[0] += node_room;
        Some(slot)
    }
}

impl Slot {
    fn holds(&self, blas: &RetainedBlas) -> bool {
        blas.triangles.len() <= self.triangle_room as usize
            && blas.vertex_words.len() <= (self.vertex_room * VERTEX_WORDS) as usize
            && blas.nodes.len() <= self.node_room as usize
    }
}

impl RetainedBlas {
    fn new(mesh: &RetainedMesh) -> Self {
        let position = |i: u32| {
            let [x, y, z, _] = mesh.vertices[i as usize].position;
            Vec3::new(x, y, z)
        };
        let mut triangles: Vec<Triangle> = (mesh.indices.as_chunks::<3>().0.iter())
            .filter_map(|&[a, b, c]| {
                let [pa, pb, pc] = [a, b, c].map(position);
                let cross = (pb - pa).cross(pc - pa);
                // Degenerate triangles can never be hit.
                (cross.is_finite() && cross != Vec3::ZERO).then(|| Triangle {
                    indices: UVec3::new(a, b, c),
                    min: pa.min(pb).min(pc),
                    max: pa.max(pb).max(pc),
                })
            })
            .collect();
        let nodes = build_tree_parallel(&mut triangles, |t| (t.min, t.max));
        let mut vertex_words = Vec::with_capacity(mesh.vertices.len() * VERTEX_WORDS as usize);
        mesh.vertices
            .iter()
            .for_each(|v| push_vertex(&mut vertex_words, v));
        Self {
            triangles,
            nodes,
            vertex_words,
            slot: None,
        }
    }
}

/// A growing GI scene: [`GiScene::add`] appends a loaded scene's BLASes,
/// triangles and vertices, [`GiScene::set_retained`] follows meshes that change,
/// and [`GiScene::pack`] rebuilds the TLAS over every instance and returns the
/// word buffers.
#[derive(Default)]
pub struct GiScene {
    triangles: Vec<Triangle>,
    blas: Vec<BvhNode>,
    vertex_words: Vec<UVec4>,
    instances: Vec<Instance>,
    retained: BTreeMap<u64, RetainedBlas>,
    retained_material: u32,
    /// Each charted BLAS's first global triangle and islands.
    charts: Vec<(u32, Charts)>,
    /// Surface cache texel size in metres; zero leaves every instance out.
    surface_texel: f32,
    /// Instances the words can hold, and nodes reserved for a TLAS over them, so
    /// BLAS and instance addresses survive it being rebuilt.
    instance_room: u32,
    tlas_room: u32,
    tail: Tail,
    packed: GiWords,
    dirty: Vec<u64>,
}

/// Word ranges of [`GiWords`] that changed.
#[derive(Debug, Default)]
pub struct GiRanges {
    pub geometry: Vec<Range<usize>>,
    pub scene: Vec<Range<usize>>,
}

impl GiRanges {
    fn merge(&mut self, other: Self) {
        self.geometry.extend(other.geometry);
        self.scene.extend(other.scene);
    }
}

/// The packed buffers and what they hold.
#[derive(Default)]
pub struct GiWords {
    pub geometry_words: Vec<UVec4>,
    pub scene_words: Vec<UVec4>,
    pub triangles: u32,
    pub instances: u32,
    pub surface: SurfaceWords,
    /// Emissive triangles in world space, for next-event estimation.
    pub emitters: Vec<Emitter>,
}

/// Emissive triangles the GI samples at most; past it the brightest are kept.
pub const MAX_EMITTERS: usize = 4096;

/// One emissive triangle, as the light sampling in `cascades.slang` reads it.
#[derive(Clone, Copy, Debug)]
pub struct Emitter {
    pub a: Vec3,
    pub edge1: Vec3,
    pub edge2: Vec3,
    /// Unit normal of the emitting side.
    pub normal: Vec3,
    pub area: f32,
    /// Area times the luminance of the material's emission: what sampling is proportional to.
    pub power: f32,
    pub double_sided: bool,
    /// Instance as traversal numbers them, and global triangle.
    pub instance: u32,
    pub triangle: u32,
}

/// The surface cache's words, laid out by [`GiScene::pack`].
#[derive(Default)]
pub struct SurfaceWords {
    /// One word per global triangle: its three corners' chart coordinates as
    /// half-float pairs, then island + 1 (0 = uncharted).
    pub triangles: Vec<UVec4>,
    /// Per instance island: its atlas rect `(x, y, w, h)`, `w` = 0 when unplaced.
    pub islands: Vec<UVec4>,
    /// Texel records, row-major over an atlas [`surface::ATLAS_WIDTH`] wide;
    /// see [`surface::rasterize`].
    pub texels: Vec<UVec4>,
    /// The texel size used, in metres: larger than asked when the atlas was full.
    pub texel: f32,
}

impl GiScene {
    /// Size surface cache texels in metres from the next [`GiScene::pack`]; zero
    /// turns the cache off.
    pub fn set_surface_texel(&mut self, texel: f32) {
        self.surface_texel = texel;
    }

    /// Build one scene's words in one step.
    pub fn build(scene: &SceneData, material_base: u32) -> GiWords {
        let mut gi = Self::default();
        gi.add(scene, material_base);
        gi.pack();
        gi.packed
    }

    /// Append one BLAS per primitive and an instance per draw. Instance
    /// material words hold `material_base` plus the primitive's material, an
    /// index into the renderer's material buffer.
    pub fn add(&mut self, scene: &SceneData, material_base: u32) {
        let vertex_base = (self.vertex_words.len() / VERTEX_WORDS as usize) as u32;
        // Per primitive: (root node, first triangle, triangle count).
        let mut spans = Vec::with_capacity(scene.primitives.len());
        // Each primitive's tree is independent, so build them in parallel and
        // append them in order.
        let built: Vec<_> = scene
            .primitives
            .par_iter()
            .map(|primitive| {
                let start = primitive.first_index as usize;
                let indices = &scene.indices[start..start + primitive.index_count as usize];
                let mut local: Vec<Triangle> = indices
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .filter_map(|tri| {
                        let local = UVec3::new(tri[0], tri[1], tri[2]) + primitive.base_vertex;
                        let [a, b, c] = local.to_array().map(|i| {
                            let p = scene.vertices[i as usize].position;
                            Vec3::new(p[0], p[1], p[2])
                        });
                        let cross = (b - a).cross(c - a);
                        // Degenerate triangles can never be hit.
                        (cross.is_finite() && cross != Vec3::ZERO).then(|| Triangle {
                            indices: local + vertex_base,
                            min: a.min(b).min(c),
                            max: a.max(b).max(c),
                        })
                    })
                    .collect();
                let nodes = build_tree_parallel(&mut local, |t| (t.min, t.max));
                // Blended and double-sided surfaces (foliage, glass) have no
                // single outward side to cache; their hits read world probes.
                let material = &scene.materials[primitive.material as usize];
                let charted = !material.double_sided && material.alpha_mode != AlphaMode::Blend;
                let charts = charted.then(|| {
                    let corners: Vec<[Vec3; 3]> = local
                        .iter()
                        .map(|t| {
                            t.indices.to_array().map(|i| {
                                let p = scene.vertices[(i - vertex_base) as usize].position;
                                Vec3::new(p[0], p[1], p[2])
                            })
                        })
                        .collect();
                    surface::chart(&corners)
                });
                (local, nodes, charts)
            })
            .collect();
        let mut chart_of = Vec::with_capacity(built.len());
        for (local, nodes, charts) in built {
            let first = self.triangles.len() as u32;
            chart_of.push(charts.map_or(u32::MAX, |charts| {
                self.charts.push((first, charts));
                self.charts.len() as u32 - 1
            }));
            let root = self.blas.len() as u32;
            self.blas.extend(nodes.iter().map(|node| BvhNode {
                first: node.first + if node.count == 0 { root } else { first },
                ..*node
            }));
            spans.push((root, first, local.len() as u32));
            self.triangles.extend(local);
        }

        let blas = &self.blas;
        self.instances.extend(scene.draws.iter().filter_map(|draw| {
            let (root, first, count) = spans[draw.primitive as usize];
            if count == 0 {
                return None;
            }
            let primitive = scene.primitives[draw.primitive as usize];
            let material = &scene.materials[primitive.material as usize];
            let world_from_local = Affine3A::from_mat4(draw.transform);
            let determinant = world_from_local.matrix3.determinant();
            if !determinant.is_finite() || determinant == 0.0 {
                return None;
            }
            let chart = chart_of[draw.primitive as usize];
            let node = blas[root as usize];
            let (min, max) = transformed_bounds(draw.transform, node.min, node.max);
            let cull = if material.double_sided || material.alpha_mode == AlphaMode::Blend {
                FLAG_DOUBLE_SIDED
            } else {
                FLAG_CULL_BACK
            };
            let mirrored = if determinant < 0.0 { FLAG_MIRRORED } else { 0 };
            Some(Instance {
                world_from_local,
                local_from_world: world_from_local.inverse(),
                min,
                max,
                root,
                first,
                count,
                flags: cull | mirrored,
                material: material_base + primitive.material,
                emission: Vec3::from_array(material.emission),
                chart,
                surface: UVec2::ZERO,
            })
        }));

        for vertex in &scene.vertices {
            push_vertex(&mut self.vertex_words, vertex);
        }
    }

    /// Follow changed retained meshes, rebuilding only their BLASes. Call
    /// [`GiScene::refresh`] after.
    pub fn set_retained(&mut self, changes: &[RetainedChange], material: u32) {
        let built: Vec<_> = changes
            .par_iter()
            .map(|change| (change.key, change.mesh.map(RetainedBlas::new)))
            .collect();
        for (key, blas) in built {
            let old = self.retained.remove(&key);
            if let Some(mut blas) = blas.filter(|blas| !blas.triangles.is_empty()) {
                blas.slot = old
                    .and_then(|old| old.slot)
                    .filter(|slot| slot.holds(&blas));
                self.retained.insert(key, blas);
                self.dirty.push(key);
            }
        }
        self.retained_material = material;
    }

    /// Bring the packed words up to date: in place, returning the ranges to
    /// upload, unless the free space ran out and all of it changed.
    pub fn refresh(&mut self) -> Option<GiRanges> {
        let count = self.instances.len() + self.retained.len();
        if self.packed.scene_words.is_empty() || count > self.instance_room as usize {
            self.pack();
            return None;
        }
        for key in &self.dirty {
            let Some(blas) = self.retained.get_mut(key) else {
                continue;
            };
            if blas.slot.is_none() {
                blas.slot = self.tail.place(blas);
                if blas.slot.is_none() {
                    self.pack();
                    return None;
                }
            }
        }
        let mut ranges = GiRanges::default();
        for key in mem::take(&mut self.dirty) {
            if let Some(blas) = self.retained.get(&key) {
                let slot = blas.slot.expect("placed above");
                ranges.merge(write_blas(&mut self.packed, self.tlas_room, blas, slot));
            }
        }
        ranges.scene.extend(self.write_top());
        self.packed.triangles = self.triangle_count();
        Some(ranges)
    }

    /// The words as of the last [`GiScene::pack`] or [`GiScene::refresh`].
    pub fn words(&self) -> &GiWords {
        &self.packed
    }

    fn triangle_count(&self) -> u32 {
        let retained = self.retained.values().map(|b| b.triangles.len());
        (self.triangles.len() + retained.sum::<usize>()) as u32
    }

    /// Lay out every word, leaving each retained mesh room to change.
    pub fn pack(&mut self) -> &GiWords {
        let (mut triangles, mut nodes) = (self.triangles.len() as u32, self.blas.len() as u32);
        let mut vertices = (self.vertex_words.len() / VERTEX_WORDS as usize) as u32;
        for blas in self.retained.values_mut() {
            let room = |used: usize| used as u32 + used as u32 / 4 + 8;
            let slot = Slot {
                triangles,
                triangle_room: room(blas.triangles.len()),
                vertices,
                vertex_room: room(blas.vertex_words.len() / VERTEX_WORDS as usize),
                nodes,
                node_room: room(blas.nodes.len()),
            };
            triangles += slot.triangle_room;
            vertices += slot.vertex_room;
            nodes += slot.node_room;
            blas.slot = Some(slot);
        }
        // Free space for meshes that grow or appear, in proportion to the scene.
        let free = |used: u32| used / 4 + 8192;
        self.tail = Tail {
            triangles: [triangles, triangles + free(triangles)],
            vertices: [vertices, vertices + free(vertices)],
            nodes: [nodes, nodes + free(nodes)],
        };
        (triangles, vertices, nodes) = (
            self.tail.triangles[1],
            self.tail.vertices[1],
            self.tail.nodes[1],
        );
        let count = self.instances.len() + self.retained.len();
        self.instance_room = count as u32 + count as u32 / 4 + 64;
        self.tlas_room = 2 * self.instance_room;
        let base = 1 + 2 * (self.tlas_room + nodes);
        let mask = base + self.instance_room * INSTANCE_WORDS as u32;
        // After the mask, each glTF instance's place in the TLAS order, four a
        // word, for surface texels that name instances by their load order.
        let slots = self.instances.len().div_ceil(4).max(1);
        let mut scene_words = vec![UVec4::ZERO; mask as usize + 1 + slots];
        scene_words[0] = UVec4::new(mask, base, 0, INSTANCE_WORDS as u32);
        let blas_words = node_words(&self.blas, self.tlas_room);
        for (word, value) in (1 + 2 * self.tlas_room as usize..).zip(blas_words) {
            scene_words[word] = value;
        }
        scene_words[mask as usize] = LAYER_MASK;

        let vertex_base = (1 + triangles).next_multiple_of(VERTEX_WORDS);
        let mut geometry_words =
            vec![UVec4::ZERO; (vertex_base + vertices * VERTEX_WORDS) as usize];
        geometry_words[0] = UVec4::new(1, vertex_base, VERTEX_WORDS, vertices);
        for (word, triangle) in geometry_words[1..].iter_mut().zip(&self.triangles) {
            *word = triangle.indices.extend(0);
        }
        let first_vertex = vertex_base as usize;
        geometry_words[first_vertex..first_vertex + self.vertex_words.len()]
            .copy_from_slice(&self.vertex_words);

        let surface = self.pack_surface(triangles);
        self.packed = GiWords {
            geometry_words,
            scene_words,
            triangles: 0,
            instances: count as u32,
            surface,
            // `write_top` lists them.
            emitters: Vec::new(),
        };
        for blas in self.retained.values() {
            let slot = blas.slot.expect("laid out above");
            write_blas(&mut self.packed, self.tlas_room, blas, slot);
        }
        self.write_top();
        self.packed.triangles = self.triangle_count();
        self.dirty.clear();
        &self.packed
    }

    /// Charts every glTF instance into the surface atlas, growing the texel
    /// until the atlas fits [`MAX_SURFACE_TEXELS`] or islands are one texel. `triangle_room` is the
    /// geometry's triangle capacity. Sets each instance's `surface`.
    fn pack_surface(&mut self, triangle_room: u32) -> SurfaceWords {
        for instance in &mut self.instances {
            instance.surface = UVec2::ZERO;
        }
        let mut words = SurfaceWords {
            triangles: vec![UVec4::ZERO; triangle_room as usize],
            ..SurfaceWords::default()
        };
        if self.surface_texel.is_nan() || self.surface_texel <= 0.0 || self.charts.is_empty() {
            return words;
        }
        for (first, charts) in &self.charts {
            for (j, (uv, &island)) in charts.uv.iter().zip(&charts.island).enumerate() {
                let pair = |v: glam::Vec2| surface::half(v.x) | (surface::half(v.y) << 16);
                words.triangles[*first as usize + j] =
                    UVec4::new(pair(uv[0]), pair(uv[1]), pair(uv[2]), island + 1);
            }
        }
        // Texels per local metre: the largest axis scale, so no axis undersamples.
        let scale = |i: &Instance| {
            let m = i.world_from_local.matrix3;
            m.x_axis.length().max(m.y_axis.length()).max(m.z_axis.length())
        };
        let mut texel = self.surface_texel;
        let (owners, rects, height) = loop {
            let mut sizes = Vec::new();
            let mut owners = Vec::new();
            for (index, instance) in self.instances.iter().enumerate() {
                let Some((_, charts)) = self.charts.get(instance.chart as usize) else {
                    continue;
                };
                let density = scale(instance) / texel;
                for (island, &extent) in charts.extents.iter().enumerate() {
                    if let Some(size) = surface::island_size(extent, density) {
                        sizes.push(size);
                        owners.push((index, island));
                    }
                }
            }
            let (rects, height) = surface::pack(&sizes);
            // Every island takes at least a texel, so past a point growing texels saves nothing.
            let fits = height as usize * surface::ATLAS_WIDTH as usize <= MAX_SURFACE_TEXELS;
            if fits || texel > 16.0 * self.surface_texel {
                break (owners, rects, height);
            }
            texel *= 1.25;
        };
        words.texel = texel;
        // Island tables, one block per charted instance, in instance order.
        let mut tables: Vec<Vec<Option<surface::Rect>>> = vec![Vec::new(); self.instances.len()];
        for (index, instance) in self.instances.iter().enumerate() {
            if let Some((_, charts)) = self.charts.get(instance.chart as usize) {
                tables[index] = vec![None; charts.extents.len()];
            }
        }
        for (&(index, island), rect) in owners.iter().zip(rects) {
            tables[index][island] = Some(rect);
        }
        for (index, table) in tables.iter().enumerate() {
            if table.is_empty() {
                continue;
            }
            let density = scale(&self.instances[index]) / texel;
            self.instances[index].surface =
                UVec2::new(words.islands.len() as u32 + 1, density.to_bits());
            words.islands.extend(
                table.iter().map(|r| r.map_or(UVec4::ZERO, |r| UVec4::new(r.x, r.y, r.w, r.h))),
            );
        }
        let blocks: Vec<_> = (self.instances.par_iter().enumerate())
            .filter_map(|(index, instance)| {
                let (first, charts) = self.charts.get(instance.chart as usize)?;
                let density = scale(instance) / texel;
                Some(surface::rasterize(charts, *first, index as u32, density, &tables[index]))
            })
            .collect();
        let width = surface::ATLAS_WIDTH as usize;
        words.texels = vec![UVec4::ZERO; width * height as usize];
        for (rect, texels) in blocks.into_iter().flatten() {
            for (row, line) in texels.chunks_exact(rect.w as usize).enumerate() {
                let at = (rect.y as usize + row) * width + rect.x as usize;
                words.texels[at..at + line.len()].copy_from_slice(line);
            }
        }
        words
    }

    /// The TLAS and every instance, which are small enough to rewrite whole.
    /// Returns the ranges of `scene_words` written.
    fn write_top(&mut self) -> [Range<usize>; 3] {
        // Each glTF instance keeps its load order through the sort.
        let mut instances: Vec<(u32, Instance)> = (0u32..).zip(self.instances.iter().copied()).collect();
        instances.extend(self.retained.values().map(|blas| {
            let slot = blas.slot.expect("laid out by pack");
            let top = blas.nodes[0];
            // Retained vertices are in world space.
            (u32::MAX, Instance {
                world_from_local: Affine3A::IDENTITY,
                local_from_world: Affine3A::IDENTITY,
                min: top.min,
                max: top.max,
                root: slot.nodes,
                first: slot.triangles,
                count: blas.triangles.len() as u32,
                flags: FLAG_CULL_BACK,
                material: self.retained_material,
                // Retained meshes (terrain) do not emit.
                emission: Vec3::ZERO,
                chart: u32::MAX,
                surface: UVec2::ZERO,
            })
        }));
        let tlas = build_tree(&mut instances, |(_, i)| (i.min, i.max));
        self.packed.instances = instances.len() as u32;
        self.packed.emitters = self.emitters(&instances);
        let words = &mut self.packed.scene_words;
        words[0].z = instances.len() as u32;
        let [mask, base, _, _] = words[0].to_array();
        let base = base as usize;
        for (word, value) in (1..).zip(node_words(&tlas, 0)) {
            words[word] = value;
        }
        // Every instance shares the layer-mask word after the instances.
        let mut word = base;
        let mut push = |value: UVec4| {
            words[word] = value;
            word += 1;
        };
        for (index, (_, instance)) in instances.iter().enumerate() {
            for transform in [instance.local_from_world, instance.world_from_local] {
                let rows = Mat4::from(transform).transpose();
                for row in [rows.x_axis, rows.y_axis, rows.z_axis] {
                    push(UVec4::from_array(row.to_array().map(f32::to_bits)));
                }
            }
            // No hardware-opaque hint yet.
            push(UVec4::new(mask, 1, 0, instance.flags));
            push(UVec4::new(index as u32, instance.surface.x, instance.surface.y, instance.material));
            push(UVec4::new(
                self.tlas_room + instance.root,
                instance.first,
                instance.count,
                0,
            ));
        }
        let end = base + instances.len() * INSTANCE_WORDS;
        let slots = mask as usize + 1;
        let count = self.instances.len().div_ceil(4);
        for word in &mut words[slots..slots + count] {
            *word = UVec4::ZERO;
        }
        for (index, &(order, _)) in instances.iter().enumerate() {
            if order != u32::MAX {
                words[slots + order as usize / 4][order as usize % 4] = index as u32;
            }
        }
        [0..1 + 2 * tlas.len(), base..end, slots..slots + count]
    }

    /// The emissive triangles of `instances`, numbered as traversal numbers them.
    fn emitters(&self, instances: &[(u32, Instance)]) -> Vec<Emitter> {
        let position = |vertex: u32| {
            let word = self.vertex_words[(vertex * VERTEX_WORDS) as usize];
            Vec3::new(f32::from_bits(word.x), f32::from_bits(word.y), f32::from_bits(word.z))
        };
        let mut emitters = Vec::new();
        for (index, (_, instance)) in instances.iter().enumerate() {
            let luminance = instance.emission.dot(Vec3::new(0.2126, 0.7152, 0.0722));
            if luminance <= 0.0 || luminance.is_nan() {
                continue;
            }
            // A mirroring transform turns the front face's winding around.
            let side = if instance.flags & FLAG_MIRRORED != 0 { -1.0 } else { 1.0 };
            for triangle in instance.first..instance.first + instance.count {
                let [a, b, c] = self.triangles[triangle as usize]
                    .indices
                    .to_array()
                    .map(|v| instance.world_from_local.transform_point3(position(v)));
                let cross = (b - a).cross(c - a) * side;
                let area = 0.5 * cross.length();
                if area <= 0.0 || !area.is_finite() {
                    continue;
                }
                emitters.push(Emitter {
                    a,
                    edge1: b - a,
                    edge2: c - a,
                    normal: cross.normalize(),
                    area,
                    power: area * luminance,
                    double_sided: instance.flags & FLAG_DOUBLE_SIDED != 0,
                    instance: index as u32,
                    triangle,
                });
            }
        }
        // shortcut: dropped emitters still light what sees them, but not through
        // next-event estimation; sample a light tree if scenes need more.
        if emitters.len() > MAX_EMITTERS {
            emitters.sort_by(|x, y| y.power.total_cmp(&x.power));
            emitters.truncate(MAX_EMITTERS);
        }
        emitters
    }
}

/// One retained mesh's triangles, vertices and nodes, into its slot. Returns the
/// ranges written.
fn write_blas(packed: &mut GiWords, tlas_room: u32, blas: &RetainedBlas, slot: Slot) -> GiRanges {
    let geometry = &mut packed.geometry_words;
    let vertex_base = geometry[0].y as usize;
    let triangles = 1 + slot.triangles as usize;
    for (word, triangle) in geometry[triangles..].iter_mut().zip(&blas.triangles) {
        *word = (triangle.indices + slot.vertices).extend(0);
    }
    let vertices = vertex_base + (slot.vertices * VERTEX_WORDS) as usize;
    geometry[vertices..vertices + blas.vertex_words.len()].copy_from_slice(&blas.vertex_words);
    let nodes = 1 + 2 * (tlas_room + slot.nodes) as usize;
    let relocated: Vec<_> = (blas.nodes.iter())
        .map(|node| BvhNode {
            first: node.first
                + if node.count == 0 {
                    slot.nodes
                } else {
                    slot.triangles
                },
            ..*node
        })
        .collect();
    for (word, value) in (nodes..).zip(node_words(&relocated, tlas_room)) {
        packed.scene_words[word] = value;
    }
    GiRanges {
        geometry: vec![
            triangles..triangles + blas.triangles.len(),
            vertices..vertices + blas.vertex_words.len(),
        ],
        scene: std::iter::once(nodes..nodes + 2 * blas.nodes.len()).collect(),
    }
}

fn push_vertex(words: &mut Vec<UVec4>, vertex: &Vertex) {
    let bits = |v: [f32; 4]| UVec4::from_array(v.map(f32::to_bits));
    let [x, y, z, u] = vertex.position;
    let [nx, ny, nz, v] = vertex.normal;
    words.extend([
        bits([x, y, z, 0.0]),
        bits([nx, ny, nz, 0.0]),
        bits(vertex.tangent),
        bits([u, v, 0.0, 0.0]),
        bits(vertex.color),
    ]);
}

fn transformed_bounds(transform: Mat4, min: Vec3, max: Vec3) -> (Vec3, Vec3) {
    (0..8).fold((Vec3::INFINITY, Vec3::NEG_INFINITY), |(lo, hi), corner| {
        let pick = |bit: u32, a: f32, b: f32| if corner & bit != 0 { b } else { a };
        let p = transform.transform_point3(Vec3::new(
            pick(1, min.x, max.x),
            pick(2, min.y, max.y),
            pick(4, min.z, max.z),
        ));
        (lo.min(p), hi.max(p))
    })
}

fn vec_word(value: Vec3, metadata: u32) -> UVec4 {
    UVec4::new(
        value.x.to_bits(),
        value.y.to_bits(),
        value.z.to_bits(),
        metadata,
    )
}

fn node_words(nodes: &[BvhNode], child_offset: u32) -> impl Iterator<Item = UVec4> + '_ {
    nodes.iter().flat_map(move |node| {
        let first = node.first + if node.count == 0 { child_offset } else { 0 };
        [vec_word(node.min, first), vec_word(node.max, node.count)]
    })
}

fn bounds_area(min: Vec3, max: Vec3) -> f64 {
    let size = max.as_dvec3() - min.as_dvec3();
    2.0 * (size.x * size.y + size.y * size.z + size.z * size.x)
}

/// Binned SAH, with a median fallback for coincident centroids. Depth is
/// capped at 48, bounding traversal's depth-first stacks to 49 pending nodes.
/// Reorders `elements` so leaves cover contiguous ranges.
pub fn build_tree<T>(elements: &mut [T], bounds: impl Fn(&T) -> (Vec3, Vec3)) -> Vec<BvhNode> {
    build_tree_with(elements, bounds, serial_children)
}

fn build_tree_parallel<T: Send>(
    elements: &mut [T],
    bounds: impl Fn(&T) -> (Vec3, Vec3) + Sync,
) -> Vec<BvhNode> {
    build_tree_with(elements, bounds, parallel_children)
}

type Children<T, F> = fn(&mut Vec<BvhNode>, usize, &mut [T], &mut [T], usize, u32, &F);

fn build_tree_with<T, F: Fn(&T) -> (Vec3, Vec3)>(
    elements: &mut [T],
    bounds: F,
    children: Children<T, F>,
) -> Vec<BvhNode> {
    if elements.is_empty() {
        return Vec::new();
    }
    let mut nodes = vec![BvhNode::default()];
    build(&mut nodes, 0, elements, 0, 0, &bounds, children);
    nodes
}

fn build<T, F: Fn(&T) -> (Vec3, Vec3)>(
    nodes: &mut Vec<BvhNode>,
    index: usize,
    elements: &mut [T],
    first: usize,
    depth: u32,
    bounds: &F,
    descend: Children<T, F>,
) {
    let (mut min, mut max) = (Vec3::INFINITY, Vec3::NEG_INFINITY);
    let (mut center_min, mut center_max) = (Vec3::INFINITY, Vec3::NEG_INFINITY);
    for element in elements.iter() {
        let (a, b) = bounds(element);
        min = min.min(a);
        max = max.max(b);
        let center = a * 0.5 + b * 0.5;
        center_min = center_min.min(center);
        center_max = center_max.max(center);
    }
    nodes[index] = BvhNode {
        min,
        max,
        first: first as u32,
        count: elements.len() as u32,
    };
    if elements.len() <= 4 || depth == 48 {
        return;
    }
    const BINS: usize = 12;
    #[derive(Clone, Copy)]
    struct Bin {
        min: Vec3,
        max: Vec3,
        count: usize,
    }
    const EMPTY: Bin = Bin {
        min: Vec3::INFINITY,
        max: Vec3::NEG_INFINITY,
        count: 0,
    };
    let join = |a: Bin, b: Bin| Bin {
        min: a.min.min(b.min),
        max: a.max.max(b.max),
        count: a.count + b.count,
    };
    let cost = |b: Bin| {
        if b.count == 0 {
            0.0
        } else {
            bounds_area(b.min, b.max) * b.count as f64
        }
    };
    let bin_of = |center: f32, axis: usize, span: f64| {
        (((f64::from(center) - f64::from(center_min[axis])) / span * BINS as f64) as usize)
            .min(BINS - 1)
    };
    let mut best = (f64::INFINITY, None);
    for axis in 0..3 {
        let span = f64::from(center_max[axis]) - f64::from(center_min[axis]);
        if span <= 0.0 {
            continue;
        }
        let mut bins = [EMPTY; BINS];
        for element in elements.iter() {
            let (a, b) = bounds(element);
            let bin = bin_of(a[axis] * 0.5 + b[axis] * 0.5, axis, span);
            bins[bin] = join(
                bins[bin],
                Bin {
                    min: a,
                    max: b,
                    count: 1,
                },
            );
        }
        let mut right = [EMPTY; BINS];
        let mut sum = EMPTY;
        for i in (0..BINS).rev() {
            sum = join(sum, bins[i]);
            right[i] = sum;
        }
        let mut left = EMPTY;
        for i in 0..BINS - 1 {
            left = join(left, bins[i]);
            let total = cost(left) + cost(right[i + 1]);
            if left.count > 0 && right[i + 1].count > 0 && total < best.0 {
                best = (total, Some((axis, i, span)));
            }
        }
    }
    let midpoint = if let (_, Some((axis, last_bin, span))) = best {
        let (mut left, mut right) = (0, elements.len());
        while left < right {
            let (a, b) = bounds(&elements[left]);
            if bin_of(a[axis] * 0.5 + b[axis] * 0.5, axis, span) <= last_bin {
                left += 1;
            } else {
                right -= 1;
                elements.swap(left, right);
            }
        }
        left
    } else {
        let size = center_max - center_min;
        let axis = if size.x >= size.y && size.x >= size.z {
            0
        } else if size.y >= size.z {
            1
        } else {
            2
        };
        let midpoint = elements.len() / 2;
        elements.select_nth_unstable_by(midpoint, |a, b| {
            let (a, b) = (bounds(a), bounds(b));
            (a.0[axis] + a.1[axis]).total_cmp(&(b.0[axis] + b.1[axis]))
        });
        midpoint
    };
    let children = nodes.len();
    nodes.extend([BvhNode::default(); 2]);
    nodes[index].first = children as u32;
    nodes[index].count = 0;
    let (left, right) = elements.split_at_mut(midpoint);
    descend(nodes, children, left, right, first, depth, bounds);
}

fn parallel_children<T: Send, F: Fn(&T) -> (Vec3, Vec3) + Sync>(
    nodes: &mut Vec<BvhNode>,
    children: usize,
    left: &mut [T],
    right: &mut [T],
    first: usize,
    depth: u32,
    bounds: &F,
) {
    if depth < 3 && left.len() + right.len() >= 250_000 {
        let subtree = |elements: &mut [T], first| {
            let mut tree = vec![BvhNode::default()];
            build(
                &mut tree,
                0,
                elements,
                first,
                depth + 1,
                bounds,
                parallel_children,
            );
            tree
        };
        let midpoint = left.len();
        let (left, right) =
            rayon::join(|| subtree(left, first), || subtree(right, first + midpoint));
        for (root, tree) in [(children, left), (children + 1, right)] {
            let offset = nodes.len() as u32 - 1;
            for (i, mut node) in tree.into_iter().enumerate() {
                if node.count == 0 {
                    node.first += offset;
                }
                if i == 0 {
                    nodes[root] = node;
                } else {
                    nodes.push(node);
                }
            }
        }
    } else {
        serial_children(nodes, children, left, right, first, depth, bounds);
    }
}

fn serial_children<T, F: Fn(&T) -> (Vec3, Vec3)>(
    nodes: &mut Vec<BvhNode>,
    children: usize,
    left: &mut [T],
    right: &mut [T],
    first: usize,
    depth: u32,
    bounds: &F,
) {
    let midpoint = left.len();
    build(
        nodes,
        children,
        left,
        first,
        depth + 1,
        bounds,
        serial_children,
    );
    build(
        nodes,
        children + 1,
        right,
        first + midpoint,
        depth + 1,
        bounds,
        serial_children,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use verity_render::{
        Vertex,
        asset::{Draw, MaterialData, Primitive},
    };

    /// Closest hit `(t, instance, triangle)`, reading only the packed words
    /// the way the traversal shader does.
    fn trace_words(scene: &GiWords, origin: Vec3, direction: Vec3) -> Option<(f32, u32, u32)> {
        let words = &scene.scene_words;
        let geometry = &scene.geometry_words;
        let float = |w: UVec4| {
            Vec3::new(
                f32::from_bits(w.x),
                f32::from_bits(w.y),
                f32::from_bits(w.z),
            )
        };
        let node = |i: u32| {
            (
                float(words[1 + 2 * i as usize]),
                words[1 + 2 * i as usize].w,
                float(words[2 + 2 * i as usize]),
                words[2 + 2 * i as usize].w,
            )
        };
        let [_, base, _, stride] = words[0].to_array();
        let vertex_base = geometry[0].y;
        let position = |v: u32| float(geometry[(vertex_base + v * VERTEX_WORDS) as usize]);
        let mut best: Option<(f32, u32, u32)> = None;
        let mut tlas_stack = vec![0u32];
        while let Some(index) = tlas_stack.pop() {
            let (min, first, max, count) = node(index);
            if !slab(
                origin,
                direction,
                min,
                max,
                best.map_or(f32::INFINITY, |b| b.0),
            ) {
                continue;
            }
            if count == 0 {
                tlas_stack.extend([first, first + 1]);
                continue;
            }
            for instance in first..first + count {
                let at = (base + instance * stride) as usize;
                let row = |r: usize| words[at + r].to_array().map(f32::from_bits);
                let local = |p: Vec3, w: f32| {
                    Vec3::from_array([0, 1, 2].map(|r| {
                        let [a, b, c, d] = row(r);
                        a * p.x + b * p.y + c * p.z + d * w
                    }))
                };
                // The local ray keeps the world `t` because the direction is not renormalized.
                let (o, d) = (local(origin, 1.0), local(direction, 0.0));
                let mut blas_stack = vec![words[at + 8].x];
                while let Some(index) = blas_stack.pop() {
                    let (min, first, max, count) = node(index);
                    if !slab(o, d, min, max, best.map_or(f32::INFINITY, |b| b.0)) {
                        continue;
                    }
                    if count == 0 {
                        blas_stack.extend([first, first + 1]);
                        continue;
                    }
                    for triangle in first..first + count {
                        let [a, b, c, _] = geometry[1 + triangle as usize].to_array();
                        if let Some(t) = hit(o, d, position(a), position(b), position(c))
                            && best.is_none_or(|b| t < b.0)
                        {
                            best = Some((t, instance, triangle));
                        }
                    }
                }
            }
        }
        best
    }

    fn slab(origin: Vec3, direction: Vec3, min: Vec3, max: Vec3, limit: f32) -> bool {
        let inverse = direction.recip();
        let (a, b) = ((min - origin) * inverse, (max - origin) * inverse);
        let near = a.min(b).max_element().max(0.0);
        let far = a.max(b).min_element().min(limit);
        near <= far
    }

    fn hit(origin: Vec3, direction: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
        let (e1, e2) = (b - a, c - a);
        let p = direction.cross(e2);
        let det = e1.dot(p);
        if det.abs() < 1e-12 {
            return None;
        }
        let s = origin - a;
        let u = s.dot(p) / det;
        let q = s.cross(e1);
        let v = direction.dot(q) / det;
        let t = e2.dot(q) / det;
        (u >= 0.0 && v >= 0.0 && u + v <= 1.0 && t > 0.0).then_some(t)
    }

    /// Deterministic pseudo-random numbers in [0, 1).
    fn random(state: &mut u32) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        (*state >> 8) as f32 / (1 << 24) as f32
    }

    /// Two meshes of random triangles drawn several times, one draw mirrored.
    fn soup() -> SceneData {
        let mut seed = 0x9e37_79b9;
        let mut scene = SceneData::default();
        for mesh in 0..2u32 {
            let first_index = scene.indices.len() as u32;
            let base_vertex = scene.vertices.len() as u32;
            for i in 0..300u32 {
                let center =
                    Vec3::new(random(&mut seed), random(&mut seed), random(&mut seed)) * 4.0;
                for _ in 0..3 {
                    let p = center
                        + Vec3::new(random(&mut seed), random(&mut seed), random(&mut seed)) * 0.4;
                    scene.vertices.push(Vertex {
                        position: [p.x, p.y, p.z, 0.0],
                        normal: [0.0, 1.0, 0.0, 0.0],
                        tangent: [0.0; 4],
                        color: [1.0; 4],
                    });
                }
                scene.indices.extend([0, 1, 2].map(|k| i * 3 + k));
            }
            // One degenerate triangle, which the builder must drop.
            scene.indices.extend([0, 0, 0]);
            scene.primitives.push(Primitive {
                first_index,
                index_count: scene.indices.len() as u32 - first_index,
                base_vertex,
                material: mesh,
            });
            scene.materials.push(MaterialData::default());
        }
        let transforms = [
            Mat4::IDENTITY,
            Mat4::from_translation(Vec3::new(6.0, 0.0, 0.0)) * Mat4::from_rotation_y(0.7),
            Mat4::from_translation(Vec3::new(0.0, 0.0, 7.0))
                * Mat4::from_scale(Vec3::new(-1.0, 2.0, 0.5)),
            Mat4::from_translation(Vec3::new(-5.0, 1.0, 1.0)),
        ];
        for (i, transform) in transforms.into_iter().enumerate() {
            scene.draws.push(Draw {
                transform,
                primitive: (i % 2) as u32,
            });
        }
        scene
    }

    #[test]
    fn packed_traversal_matches_brute_force() {
        let scene = soup();
        let gi = GiScene::build(&scene, 10);
        let scene = &scene;
        assert_eq!((gi.triangles, gi.instances), (600, 4));
        let mut seed = 7;
        let mut hits = 0;
        for _ in 0..3000 {
            let origin = Vec3::new(
                random(&mut seed) * 20.0 - 8.0,
                random(&mut seed) * 10.0 - 3.0,
                -6.0,
            );
            let target = Vec3::new(
                random(&mut seed) * 14.0 - 6.0,
                random(&mut seed) * 8.0,
                random(&mut seed) * 10.0,
            );
            let direction = (target - origin).normalize();
            let expected = scene
                .draws
                .iter()
                .flat_map(|draw| {
                    let primitive = scene.primitives[draw.primitive as usize];
                    let range = primitive.first_index as usize
                        ..(primitive.first_index + primitive.index_count) as usize;
                    scene.indices[range]
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .filter_map(move |tri| {
                            let p = [tri[0], tri[1], tri[2]].map(|i| {
                                let v =
                                    scene.vertices[(primitive.base_vertex + i) as usize].position;
                                draw.transform.transform_point3(Vec3::new(v[0], v[1], v[2]))
                            });
                            hit(origin, direction, p[0], p[1], p[2])
                        })
                })
                .min_by(f32::total_cmp);
            let found = trace_words(&gi, origin, direction).map(|h| h.0);
            match (expected, found) {
                (Some(e), Some(f)) => {
                    assert!((e - f).abs() < 1e-3 * e.max(1.0), "{e} vs {f}");
                    hits += 1;
                }
                (None, None) => {}
                other => panic!("brute force and packed traversal disagree: {other:?}"),
            }
        }
        assert!(hits > 200, "too few hits ({hits}) to exercise traversal");
    }

    #[test]
    fn appended_scenes_trace_like_brute_force() {
        let first = soup();
        let mut second = soup();
        for draw in &mut second.draws {
            draw.transform = Mat4::from_translation(Vec3::new(0.0, 0.0, 30.0)) * draw.transform;
        }
        let mut gi = GiScene::default();
        gi.add(&first, 0);
        gi.add(&second, 2);
        let words = gi.pack();
        assert_eq!((words.triangles, words.instances), (1200, 8));
        let mut seed = 11;
        let mut hits = 0;
        for _ in 0..2000 {
            let origin = Vec3::new(
                random(&mut seed) * 14.0 - 6.0,
                random(&mut seed) * 8.0,
                -10.0,
            );
            let target = Vec3::new(
                random(&mut seed) * 14.0 - 6.0,
                random(&mut seed) * 8.0,
                random(&mut seed) * 45.0,
            );
            let direction = (target - origin).normalize();
            let expected = [&first, &second]
                .into_iter()
                .flat_map(|scene| {
                    scene.draws.iter().flat_map(move |draw| {
                        let primitive = scene.primitives[draw.primitive as usize];
                        let range = primitive.first_index as usize
                            ..(primitive.first_index + primitive.index_count) as usize;
                        scene.indices[range]
                            .as_chunks::<3>()
                            .0
                            .iter()
                            .filter_map(move |tri| {
                                let p = tri.map(|i| {
                                    let v = scene.vertices[(primitive.base_vertex + i) as usize]
                                        .position;
                                    draw.transform.transform_point3(Vec3::new(v[0], v[1], v[2]))
                                });
                                hit(origin, direction, p[0], p[1], p[2])
                            })
                    })
                })
                .min_by(f32::total_cmp);
            match (expected, trace_words(words, origin, direction).map(|h| h.0)) {
                (Some(e), Some(f)) => {
                    assert!((e - f).abs() < 1e-3 * e.max(1.0), "{e} vs {f}");
                    hits += 1;
                }
                (None, None) => {}
                other => panic!("appended scene disagrees with brute force: {other:?}"),
            }
        }
        assert!(hits > 100, "too few hits ({hits})");
    }

    #[test]
    fn instance_words_carry_flags_material_and_blas() {
        let mut scene = soup();
        scene.materials[1].double_sided = true;
        let gi = GiScene::build(&scene, 10);
        let [_, base, count, stride] = gi.scene_words[0].to_array();
        assert_eq!((count, stride as usize), (4, INSTANCE_WORDS));
        let mut mirrored = 0;
        for i in 0..count {
            let at = (base + i * stride) as usize;
            let flags = gi.scene_words[at + 6].w;
            let material = gi.scene_words[at + 7].w;
            assert!(material == 10 || material == 11);
            let expected = if material == 11 {
                FLAG_DOUBLE_SIDED
            } else {
                FLAG_CULL_BACK
            };
            assert_eq!(flags & (3 | FLAG_DOUBLE_SIDED), expected);
            mirrored += (flags & FLAG_MIRRORED != 0) as u32;
            assert_eq!(gi.scene_words[at + 8].z, 300);
        }
        assert_eq!(mirrored, 1);
        // Every instance must reference the layer mask, or traversal skips it.
        let mask = gi.scene_words[0].x;
        assert_eq!(gi.scene_words[mask as usize], LAYER_MASK);
        for i in 0..count {
            let word = gi.scene_words[(base + i * stride) as usize + 6];
            assert_eq!((word.x, word.y), (mask, 1));
        }
        // After the mask, every instance's TLAS slot by load order, each slot once.
        let slots = &gi.scene_words[mask as usize + 1..];
        assert_eq!(slots.len(), (count as usize).div_ceil(4));
        let mut seen: Vec<u32> = (0..count as usize).map(|i| slots[i / 4][i % 4]).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..count).collect::<Vec<_>>());
    }

    #[test]
    fn emitters_come_from_emissive_instances_facing_their_front() {
        let mut scene = soup();
        scene.materials[1].emission = [2.0, 1.0, 0.5];
        let gi = GiScene::build(&scene, 10);
        let [_, base, count, stride] = gi.scene_words[0].to_array();
        let words = |i: u32| (base + i * stride) as usize;
        let emissive = (0..count).filter(|&i| gi.scene_words[words(i) + 7].w == 11).count();
        assert!(emissive > 0);
        assert_eq!(gi.emitters.len(), emissive * 300);
        for e in &gi.emitters {
            let at = words(e.instance);
            assert_eq!(gi.scene_words[at + 7].w, 11);
            // Triangles are numbered from the instance's first.
            let first = gi.scene_words[at + 8].y;
            assert!((first..first + 300).contains(&e.triangle));
            let cross = e.edge1.cross(e.edge2);
            assert!((e.area - 0.5 * cross.length()).abs() <= 1e-4 * e.area.max(1.0));
            // Mirrored instances emit from the other side of their world winding.
            let mirrored = gi.scene_words[at + 6].w & FLAG_MIRRORED != 0;
            assert_eq!(e.normal.dot(cross) < 0.0, mirrored);
            assert!(e.power > 0.0);
        }
    }

    #[test]
    fn tree_leaves_cover_every_element_once() {
        let mut seed = 3;
        let mut points: Vec<Vec3> = (0..5000)
            .map(|_| Vec3::new(random(&mut seed), random(&mut seed), random(&mut seed)))
            .collect();
        points.extend([Vec3::ONE; 64]); // coincident centroids force the median split
        let nodes = build_tree(&mut points, |&p| (p, p));
        let mut covered = vec![0u8; points.len()];
        for node in nodes.iter().filter(|n| n.count > 0) {
            assert!(node.count <= 4 || nodes.len() > 1000);
            for i in node.first..node.first + node.count {
                covered[i as usize] += 1;
                let p = points[i as usize];
                assert!(p.cmpge(node.min).all() && p.cmple(node.max).all());
            }
        }
        assert!(covered.iter().all(|&c| c == 1));
    }

    #[test]
    fn tree_accepts_thread_local_data() {
        let value = std::rc::Rc::new(Vec3::ONE);
        let mut points = [value];
        let nodes = build_tree(&mut points, |p| (**p, **p));
        assert_eq!(nodes[0].min, Vec3::ONE);
    }

    #[test]
    fn large_tree_reaches_every_element() {
        let mut points: Vec<Vec3> = (0..1_000_000)
            .map(|i| Vec3::new((i % 1000) as f32, (i / 1000) as f32, 0.0))
            .collect();
        let nodes = build_tree_parallel(&mut points, |&p| (p, p));
        let mut covered = vec![false; points.len()];
        let mut visited = vec![false; nodes.len()];
        let mut pending = vec![0usize];
        while let Some(i) = pending.pop() {
            assert!(!std::mem::replace(&mut visited[i], true));
            let node = nodes[i];
            if node.count == 0 {
                let first = node.first as usize;
                for child in [first, first + 1] {
                    assert!(nodes[child].min.cmpge(node.min).all());
                    assert!(nodes[child].max.cmple(node.max).all());
                    pending.push(child);
                }
            } else {
                for j in node.first as usize..(node.first + node.count) as usize {
                    assert!(!std::mem::replace(&mut covered[j], true));
                    assert!(points[j].cmpge(node.min).all() && points[j].cmple(node.max).all());
                }
            }
        }
        assert!(visited.into_iter().all(|seen| seen));
        assert!(covered.into_iter().all(|seen| seen));
    }
}
