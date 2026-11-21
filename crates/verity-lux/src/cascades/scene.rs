//! The GI scene: a two-level BVH over a loaded glTF scene, packed into the
//! word buffers `shaders/traversal.slang` reads. Every word is a `UVec4`.
//!
//! Geometry words: a header `(1, vertex base, VERTEX_WORDS, vertex count)`,
//! one `(a, b, c, 0)` index word per triangle, padding up to the vertex base,
//! then [`VERTEX_WORDS`] words per vertex: position (w: flags), normal,
//! tangent, texture coordinates and color.
//!
//! Scene words: a prefix `(end of instance words, instance base, instance
//! count, INSTANCE_WORDS)`, the binary TLAS and retained BLAS nodes, then
//! [`INSTANCE_WORDS`] words per instance. Binary nodes take two words,
//! `(min, first)` and `(max, count)`: a leaf (count > 0) spans `count` elements
//! from `first`; otherwise the children are `first` and `first + 1`. The
//! TLAS has reserved room so rebuilding it leaves the BLAS addresses stable.
//! After the instances come the shared layer mask, the glTF instance reverse
//! map, and the immutable compressed wide BLAS arena. Instance word 8 is
//! `(binary root, first triangle, triangle count, wide root word or zero)`.
//! `VERITY_LUX_ORIENTED_BLAS=1` enables selective rotated child bounds;
//! `VERITY_LUX_SPATIAL_BLAS=1` enables profitable static spatial splits. Both
//! preserve geometry word ordering and leave retained BLASes binary, which
//! measured faster than wide for small terrain chunks ([`RetainedBlas`]).

use super::{
    bvh,
    surface::{self, Charts},
};
use crate::tier::SoftwareLayout;
use glam::{Affine3A, Mat4, UVec2, UVec3, UVec4, Vec3};
use rayon::prelude::*;
use std::{collections::BTreeMap, mem, ops::Range};
use verity_render::{
    RetainedChange, RetainedMesh, Vertex,
    asset::{AlphaMode, SceneData},
};

mod oriented;
mod spatial;
mod wide;

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
enum BlasRoot {
    Binary(u32),
    Wide(u32),
}

#[derive(Clone, Copy)]
struct Instance {
    world_from_local: Affine3A,
    local_from_world: Affine3A,
    min: Vec3,
    max: Vec3,
    root: BlasRoot,
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
///
/// Its nodes stay binary and uncompressed. Packing terrain chunks (about 1,200
/// triangles each) in the four-wide layout static meshes use made software GI
/// slower on both an RTX 4070 and an RDNA 3 iGPU (docs/validation.md).
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
        let nodes = bvh::build_mesh(&mut triangles, |t| (t.min, t.max));
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
    wide_blas: Vec<UVec4>,
    vertex_words: Vec<UVec4>,
    instances: Vec<Instance>,
    retained: BTreeMap<u64, RetainedBlas>,
    retained_material: u32,
    /// Each charted BLAS's first global triangle and islands.
    charts: Vec<(u32, Charts)>,
    /// Surface cache texel size in metres; zero leaves every instance out.
    surface_texel: f32,
    /// How static meshes' software trees are laid out.
    software_layout: SoftwareLayout,
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

/// Stable instance identity, independent of TLAS ordering and retained slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum InstanceIdentity {
    Static(u32),
    Retained(u64),
}

/// The packed buffers and what they hold.
#[derive(Default)]
pub struct GiWords {
    pub geometry_words: Vec<UVec4>,
    pub scene_words: Vec<UVec4>,
    pub triangles: u32,
    pub instances: u32,
    pub(crate) instance_ids: Vec<InstanceIdentity>,
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
        let SoftwareLayout { oriented, spatial } = self.software_layout;
        self.add_with_static_bounds(scene, material_base, StaticBounds { oriented, spatial });
    }

    /// Lay out the software BVH of static meshes added from now on; see
    /// [`crate::tier::HardwareProfile`].
    pub fn set_software_layout(&mut self, layout: SoftwareLayout) {
        self.software_layout = layout;
    }

    fn add_with_static_bounds(
        &mut self,
        scene: &SceneData,
        material_base: u32,
        options: StaticBounds,
    ) {
        let vertex_base = (self.vertex_words.len() / VERTEX_WORDS as usize) as u32;
        // Per primitive: static BLAS address, triangle span, and local bounds.
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
                let nodes = bvh::build_mesh(&mut local, |t| (t.min, t.max));
                let root = nodes.first().map(|n| (n.min, n.max));
                // Keep the object tree's source order even if traversal uses the
                // spatial tree: hardware, surface charts and emitters share IDs.
                // Blended and double-sided surfaces (foliage, glass) have no
                // single outward side to cache; their hits read world probes.
                let material = &scene.materials[primitive.material as usize];
                let charted = !material.double_sided && material.alpha_mode != AlphaMode::Blend;
                let corners: Vec<[Vec3; 3]> = if charted || options.oriented || options.spatial {
                    local
                        .iter()
                        .map(|t| {
                            t.indices.to_array().map(|i| {
                                let p = scene.vertices[(i - vertex_base) as usize].position;
                                Vec3::new(p[0], p[1], p[2])
                            })
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                let oriented = options.oriented.then_some(corners.as_slice());
                let spatial = options
                    .spatial
                    .then(|| spatial::build(&corners, &nodes))
                    .flatten();
                let words = if let Some(spatial) = spatial {
                    wide::pack_static(&spatial.nodes, oriented, &spatial.references)
                } else if oriented.is_some() {
                    wide::pack_static(&nodes, oriented, &[])
                } else {
                    wide::pack(&nodes)
                };
                drop(nodes);
                let charts = charted.then(|| surface::chart(&corners));
                (local, words, root, charts)
            })
            .collect();
        let mut chart_of = Vec::with_capacity(built.len());
        for (local, words, bounds, charts) in built {
            let first = self.triangles.len() as u32;
            chart_of.push(charts.map_or(u32::MAX, |charts| {
                self.charts.push((first, charts));
                self.charts.len() as u32 - 1
            }));
            let root = self.wide_blas.len() as u32;
            self.wide_blas.extend(words);
            spans.push(bounds.map(|(min, max)| (root, first, local.len() as u32, min, max)));
            self.triangles.extend(local);
        }

        self.instances.extend(scene.draws.iter().filter_map(|draw| {
            let (root, first, count, min, max) = spans[draw.primitive as usize]?;
            let primitive = scene.primitives[draw.primitive as usize];
            let material = &scene.materials[primitive.material as usize];
            let world_from_local = Affine3A::from_mat4(draw.transform);
            let determinant = world_from_local.matrix3.determinant();
            if !determinant.is_finite() || determinant == 0.0 {
                return None;
            }
            let chart = chart_of[draw.primitive as usize];
            let (min, max) = transformed_bounds(draw.transform, min, max);
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
                root: BlasRoot::Wide(root),
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
        let packed = &self.packed.scene_words;
        let layout_changed = packed
            .first()
            .is_none_or(|header| self.wide_base(header.x) + self.wide_blas.len() != packed.len());
        if layout_changed || count > self.instance_room as usize {
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

    fn wide_base(&self, mask: u32) -> usize {
        // The reverse map starts after the mask, four load-order entries per
        // word. Keep its reserved word even with no static instances.
        mask as usize + 1 + self.instances.len().div_ceil(4).max(1)
    }

    /// Lay out every word, leaving each retained mesh room to change.
    pub fn pack(&mut self) -> &GiWords {
        let mut triangles = self.triangles.len() as u32;
        let mut nodes = 0;
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
        let wide_base = self.wide_base(mask);
        let mut scene_words = Vec::with_capacity(wide_base + self.wide_blas.len());
        scene_words.resize(wide_base, UVec4::ZERO);
        scene_words[0] = UVec4::new(mask, base, 0, INSTANCE_WORDS as u32);
        scene_words.extend_from_slice(&self.wide_blas);
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
            instance_ids: Vec::new(),
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
            m.x_axis
                .length()
                .max(m.y_axis.length())
                .max(m.z_axis.length())
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
                table
                    .iter()
                    .map(|r| r.map_or(UVec4::ZERO, |r| UVec4::new(r.x, r.y, r.w, r.h))),
            );
        }
        let blocks: Vec<_> = (self.instances.par_iter().enumerate())
            .filter_map(|(index, instance)| {
                let (first, charts) = self.charts.get(instance.chart as usize)?;
                let density = scale(instance) / texel;
                Some(surface::rasterize(
                    charts,
                    *first,
                    index as u32,
                    density,
                    &tables[index],
                ))
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
        let mut instances: Vec<(InstanceIdentity, Instance)> = (0u32..)
            .map(InstanceIdentity::Static)
            .zip(self.instances.iter().copied())
            .collect();
        instances.extend(self.retained.iter().map(|(&key, blas)| {
            let slot = blas.slot.expect("laid out by pack");
            let top = blas.nodes[0];
            // Retained vertices are in world space.
            (
                InstanceIdentity::Retained(key),
                Instance {
                    world_from_local: Affine3A::IDENTITY,
                    local_from_world: Affine3A::IDENTITY,
                    min: top.min,
                    max: top.max,
                    root: BlasRoot::Binary(slot.nodes),
                    first: slot.triangles,
                    count: blas.triangles.len() as u32,
                    flags: FLAG_CULL_BACK,
                    material: self.retained_material,
                    // Retained meshes (terrain) do not emit.
                    emission: Vec3::ZERO,
                    chart: u32::MAX,
                    surface: UVec2::ZERO,
                },
            )
        }));
        let tlas = build_tree(&mut instances, |(_, i)| (i.min, i.max));
        self.packed.instances = instances.len() as u32;
        self.packed.instance_ids.clear();
        self.packed
            .instance_ids
            .extend(instances.iter().map(|&(id, _)| id));
        self.packed.emitters = self.emitters(&instances);
        let wide_base = self.wide_base(self.packed.scene_words[0].x) as u32;
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
            push(UVec4::new(
                index as u32,
                instance.surface.x,
                instance.surface.y,
                instance.material,
            ));
            let (binary, wide) = match instance.root {
                BlasRoot::Binary(root) => (self.tlas_room + root, 0),
                BlasRoot::Wide(root) => (0, wide_base + root),
            };
            push(UVec4::new(binary, instance.first, instance.count, wide));
        }
        let end = base + instances.len() * INSTANCE_WORDS;
        let slots = mask as usize + 1;
        let count = self.instances.len().div_ceil(4);
        for word in &mut words[slots..slots + count] {
            *word = UVec4::ZERO;
        }
        for (index, &(identity, _)) in instances.iter().enumerate() {
            if let InstanceIdentity::Static(order) = identity {
                words[slots + order as usize / 4][order as usize % 4] = index as u32;
            }
        }
        [0..1 + 2 * tlas.len(), base..end, slots..slots + count]
    }

    /// The emissive triangles of `instances`, numbered as traversal numbers them.
    fn emitters(&self, instances: &[(InstanceIdentity, Instance)]) -> Vec<Emitter> {
        let position = |vertex: u32| {
            let word = self.vertex_words[(vertex * VERTEX_WORDS) as usize];
            Vec3::new(
                f32::from_bits(word.x),
                f32::from_bits(word.y),
                f32::from_bits(word.z),
            )
        };
        let mut emitters = Vec::new();
        for (index, (_, instance)) in instances.iter().enumerate() {
            let luminance = instance.emission.dot(Vec3::new(0.2126, 0.7152, 0.0722));
            if luminance <= 0.0 || luminance.is_nan() {
                continue;
            }
            // A mirroring transform turns the front face's winding around.
            let side = if instance.flags & FLAG_MIRRORED != 0 {
                -1.0
            } else {
                1.0
            };
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

#[derive(Clone, Copy)]
struct StaticBounds {
    oriented: bool,
    spatial: bool,
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

/// Build a binary BVH over finite, ordered AABBs, reordering elements so
/// each leaf covers a contiguous range. Sibling nodes are adjacent.
pub fn build_tree<T>(elements: &mut [T], bounds: impl Fn(&T) -> (Vec3, Vec3)) -> Vec<BvhNode> {
    bvh::build(elements, bounds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use verity_render::{
        Vertex,
        asset::{Draw, MaterialData, Primitive},
    };
    use wide::packed::{self, Reference};

    /// Closest hit `(t, instance, triangle)`, reading only the packed words
    /// the way the traversal shader does.
    fn trace_words(scene: &GiWords, origin: Vec3, direction: Vec3) -> Option<(f32, u32, u32)> {
        trace_words_with_limit(scene, origin, direction, f32::INFINITY, false)
    }

    fn trace_words_with_limit(
        scene: &GiWords,
        origin: Vec3,
        direction: Vec3,
        limit: f32,
        any_hit: bool,
    ) -> Option<(f32, u32, u32)> {
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
        let [_, base, instances, stride] = words[0].to_array();
        if instances == 0 {
            return None;
        }
        let vertex_base = geometry[0].y;
        let position = |v: u32| float(geometry[(vertex_base + v * VERTEX_WORDS) as usize]);
        let mut best: Option<(f32, u32, u32)> = None;
        let mut tlas_stack = vec![0u32];
        while let Some(index) = tlas_stack.pop() {
            let (min, first, max, count) = node(index);
            if slab(origin, direction, min, max, best.map_or(limit, |b| b.0)).is_none() {
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
                let [binary, mesh_first, _, wide] = words[at + 8].to_array();
                let mut blas_stack = Vec::new();
                let records = if wide == 0 {
                    blas_stack.push((binary, 0.0));
                    None
                } else {
                    let (min, max, root, records) = packed::root(words, wide as usize);
                    let Some(entry) = slab(o, d, min, max, best.map_or(limit, |b| b.0)) else {
                        continue;
                    };
                    blas_stack.push((root, entry));
                    Some((records, (min, max)))
                };
                while let Some((index, entry)) = blas_stack.pop() {
                    let cutoff = best.map_or(limit, |b| b.0);
                    if entry > cutoff {
                        continue;
                    }
                    let mut indirect = None;
                    let (first, count) = if let Some((records, root)) = records {
                        match packed::reference(records, index) {
                            Reference::Branch(at) => {
                                blas_stack.extend(packed::children(records, at).filter_map(
                                    |(min, max, link)| {
                                        slab(o, d, min, max, cutoff).map(|entry| (link, entry))
                                    },
                                ));
                                continue;
                            }
                            Reference::Oriented(at, frame) => {
                                blas_stack.extend(packed::children(records, at).filter_map(
                                    |(min, max, link)| {
                                        oriented::packed_entry(
                                            frame,
                                            (min, max),
                                            root,
                                            o,
                                            d,
                                            cutoff,
                                        )
                                        .map(|entry| (link, entry))
                                    },
                                ));
                                continue;
                            }
                            Reference::Indirect(at, count) => {
                                indirect = Some((records, at));
                                (0, count)
                            }
                            Reference::Span(first, count) => (mesh_first + first, count),
                        }
                    } else {
                        let (min, first, max, count) = node(index);
                        if slab(o, d, min, max, cutoff).is_none() {
                            continue;
                        }
                        if count == 0 {
                            blas_stack.extend([(first, 0.0), (first + 1, 0.0)]);
                            continue;
                        }
                        (first, count)
                    };
                    for item in first..first + count {
                        let triangle = indirect.map_or(item, |(records, at)| {
                            mesh_first + packed::source(records, at, item)
                        });
                        let [a, b, c, _] = geometry[1 + triangle as usize].to_array();
                        if let Some(t) = hit(o, d, position(a), position(b), position(c))
                            && t < best.map_or(limit, |b| b.0)
                        {
                            if any_hit {
                                return Some((t, instance, triangle));
                            }
                            best = Some((t, instance, triangle));
                        }
                    }
                }
            }
        }
        best
    }

    fn slab(origin: Vec3, direction: Vec3, min: Vec3, max: Vec3, limit: f32) -> Option<f32> {
        let (mut near, mut far) = (0.0_f32, limit);
        for axis in 0..3 {
            if direction[axis] == 0.0 {
                if origin[axis] < min[axis] || origin[axis] > max[axis] {
                    return None;
                }
                continue;
            }
            let inverse = direction[axis].recip();
            let (a, b) = (
                (min[axis] - origin[axis]) * inverse,
                (max[axis] - origin[axis]) * inverse,
            );
            let exit = a.max(b);
            near = near.max(a.min(b));
            far = far.min(exit + exit.abs() * 3.576_279_1e-7);
        }
        (near <= far).then_some(near)
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

    fn static_fixture(corners: &[[Vec3; 3]]) -> SceneData {
        let mut scene = SceneData::default();
        for &p in corners.iter().flatten() {
            scene.indices.push(scene.vertices.len() as u32);
            scene.vertices.push(Vertex {
                position: p.extend(0.0).to_array(),
                normal: [0.0, 0.0, 1.0, 0.0],
                tangent: [0.0; 4],
                color: [1.0; 4],
            });
        }
        scene.primitives.push(Primitive {
            first_index: 0,
            index_count: scene.indices.len() as u32,
            base_vertex: 0,
            material: 0,
        });
        scene.materials.push(MaterialData::default());
        scene.draws.push(Draw {
            transform: Mat4::IDENTITY,
            primitive: 0,
        });
        scene
    }

    fn static_words(scene: &SceneData, oriented: bool, spatial: bool) -> GiWords {
        let mut gi = GiScene::default();
        gi.add_with_static_bounds(scene, 0, StaticBounds { oriented, spatial });
        gi.pack();
        gi.packed
    }

    fn static_root(words: &GiWords) -> (u32, &[UVec4]) {
        let base = words.scene_words[0].y as usize;
        let wide = words.scene_words[base + 8].w as usize;
        let (_, _, root, records) = packed::root(&words.scene_words, wide);
        (root, records)
    }

    #[test]
    fn rotated_static_bounds_preserve_source_ids_and_tight_hits() {
        for offset in [Vec3::ZERO, Vec3::new(-262144.0, 262144.0, 4096.0)] {
            let corners: Vec<_> = (0..16)
                .map(|layer| {
                    let base = offset + Vec3::Z * (4.0 * layer as f32);
                    [
                        Vec3::ZERO,
                        Vec3::new(128.0, 128.0, 0.0),
                        Vec3::new(64.5, 63.5, 2.0),
                    ]
                    .map(|p| p + base)
                })
                .collect();
            let scene = static_fixture(&corners);
            let baseline = static_words(&scene, false, false);
            let rotated = static_words(&scene, true, false);
            let (root, records) = static_root(&rotated);
            assert!(matches!(
                packed::reference(records, root),
                Reference::Oriented(_, 0)
            ));
            assert_eq!(baseline.geometry_words, rotated.geometry_words);
            for layer in [0, 3, 15] {
                let origin = offset + Vec3::new(64.125, 63.875, 4.0 * layer as f32 + 3.0);
                let expected = trace_words(&baseline, origin, Vec3::NEG_Z);
                assert!(expected.is_some());
                assert_eq!(trace_words(&rotated, origin, Vec3::NEG_Z), expected);
                assert!(
                    trace_words_with_limit(&rotated, origin, Vec3::NEG_Z, 2.49, true).is_none()
                );
                assert!(
                    trace_words_with_limit(&rotated, origin, Vec3::NEG_Z, 2.51, true).is_some()
                );
            }
        }
    }

    #[test]
    fn spatial_static_leaves_index_original_geometry_in_both_modes() {
        let straddler = [
            Vec3::new(-8.0, -8.0, 0.0),
            Vec3::new(8.0, 8.0, 0.0),
            Vec3::new(8.0, 7.5, 0.0),
        ];
        let mut corners = vec![straddler];
        for i in 0..8 {
            for (x, y) in [(-7.5, -7.75), (7.5, 6.875)] {
                let a = Vec3::new(x, y + 0.125 * i as f32, 0.0);
                corners.push([a, a + Vec3::X * 0.5, a + Vec3::Y * 0.0625]);
            }
        }
        let scene = static_fixture(&corners);
        let baseline = static_words(&scene, false, false);
        let original = baseline.geometry_words[1..=corners.len()]
            .iter()
            .position(|word| word.x == 0)
            .expect("straddler was packed") as u32;
        for oriented in [false, true] {
            let spatial = static_words(&scene, oriented, true);
            assert_eq!(baseline.geometry_words, spatial.geometry_words);
            let (root, records) = static_root(&spatial);
            let mut pending = vec![root];
            let mut duplicates = 0;
            while let Some(link) = pending.pop() {
                match packed::reference(records, link) {
                    Reference::Branch(at) | Reference::Oriented(at, _) => {
                        pending.extend(packed::children(records, at).map(|(_, _, link)| link));
                    }
                    Reference::Indirect(at, count) => {
                        duplicates += (0..count)
                            .filter(|&i| packed::source(records, at, i) == original)
                            .count();
                    }
                    Reference::Span(..) => {
                        panic!("spatial leaf must preserve an indirect source ID")
                    }
                }
            }
            assert!(duplicates > 1, "expected more than one fragment reference");
            for (x, y, hits) in [(-6.0, -6.02, true), (6.0, 5.75, true), (0.0, 1.0, false)] {
                let origin = Vec3::new(x, y, 10.0);
                let expected = trace_words(&baseline, origin, Vec3::NEG_Z);
                assert_eq!(expected.is_some(), hits);
                assert_eq!(trace_words(&spatial, origin, Vec3::NEG_Z), expected);
                assert_eq!(
                    trace_words_with_limit(&spatial, origin, Vec3::NEG_Z, 11.0, true).is_some(),
                    hits
                );
            }
        }
    }

    #[test]
    fn large_static_blas_preserves_packed_triangle_hits() {
        // Shuffled disjoint triangles: a bounds/index permutation mismatch
        // can hide a hit even when every index still occurs once.
        const COUNT: u32 = 1 << 16;
        let corner =
            |cell: u32| Vec3::new((cell % 256) as f32, (cell / 256) as f32, (cell % 7) as f32);
        let mut scene = SceneData::default();
        for i in 0..COUNT {
            let cell = i.wrapping_mul(17_431) & (COUNT - 1);
            let base = corner(cell);
            for p in [base, base + Vec3::X * 0.5, base + Vec3::Y * 0.5] {
                scene.vertices.push(Vertex {
                    position: p.extend(0.0).to_array(),
                    normal: [0.0, 0.0, 1.0, 0.0],
                    tangent: [0.0; 4],
                    color: [1.0; 4],
                });
            }
            scene.indices.extend([3 * i, 3 * i + 1, 3 * i + 2]);
        }
        scene.primitives.push(Primitive {
            first_index: 0,
            index_count: COUNT * 3,
            base_vertex: 0,
            material: 0,
        });
        scene.materials.push(MaterialData::default());
        scene.draws.push(Draw {
            transform: Mat4::IDENTITY,
            primitive: 0,
        });

        let words = GiScene::build(&scene, 0);
        assert_eq!(words.triangles, COUNT);
        for i in (0..COUNT).step_by(97) {
            let cell = i.wrapping_mul(17_431) & (COUNT - 1);
            let origin = corner(cell) + Vec3::new(0.125, 0.125, 2.0);
            let (t, _, triangle) = trace_words(&words, origin, Vec3::NEG_Z)
                .unwrap_or_else(|| panic!("missing triangle {i} in cell {cell}"));
            assert_eq!(t, 2.0);
            assert_eq!(words.geometry_words[1 + triangle as usize].x, 3 * i);
        }
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

    /// A terrain-like retained grid of `n` × `n` cells, its heights from `seed`.
    fn heightfield(n: u32, seed: u32, offset: Vec3) -> RetainedMesh {
        let mut state = seed;
        let mut vertices = Vec::new();
        for z in 0..=n {
            for x in 0..=n {
                let p = offset + Vec3::new(x as f32, random(&mut state) * 3.0, z as f32);
                vertices.push(Vertex {
                    position: p.extend(0.0).to_array(),
                    normal: [0.0, 1.0, 0.0, 0.0],
                    tangent: [0.0; 4],
                    color: [1.0; 4],
                });
            }
        }
        let at = |x: u32, z: u32| z * (n + 1) + x;
        let mut indices = Vec::new();
        for z in 0..n {
            for x in 0..n {
                let [a, b, c, d] = [at(x, z), at(x + 1, z), at(x, z + 1), at(x + 1, z + 1)];
                indices.extend([a, c, b, b, c, d]);
            }
        }
        let (min, max) = vertices.iter().fold(
            (Vec3::INFINITY, Vec3::NEG_INFINITY),
            |(lo, hi), v: &Vertex| {
                let p = Vec3::from_slice(&v.position[..3]);
                (lo.min(p), hi.max(p))
            },
        );
        RetainedMesh {
            vertices,
            indices,
            bounds: (min, max),
            plain: false,
        }
    }

    fn brute_force(meshes: &[&RetainedMesh], origin: Vec3, direction: Vec3) -> Option<f32> {
        meshes
            .iter()
            .flat_map(|mesh| {
                mesh.indices
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .filter_map(move |tri| {
                        let p =
                            tri.map(|i| Vec3::from_slice(&mesh.vertices[i as usize].position[..3]));
                        hit(origin, direction, p[0], p[1], p[2])
                    })
            })
            .min_by(f32::total_cmp)
    }

    #[test]
    fn retained_blases_trace_like_brute_force_through_edits() {
        let mut gi = GiScene::default();
        gi.add(&soup(), 0);
        let near = heightfield(24, 3, Vec3::new(-12.0, -6.0, -12.0));
        let far = heightfield(16, 5, Vec3::new(14.0, -6.0, -8.0));
        gi.set_retained(
            &[
                RetainedChange {
                    key: 1,
                    mesh: Some(&near),
                },
                RetainedChange {
                    key: 2,
                    mesh: Some(&far),
                },
            ],
            7,
        );
        gi.pack();
        let check = |gi: &GiScene, meshes: &[&RetainedMesh], seed: u32| {
            let words = gi.words();
            let [_, base, count, stride] = words.scene_words[0].to_array();
            for i in 0..count {
                let mesh = words.scene_words[(base + i * stride) as usize + 8];
                assert!(
                    (mesh.x == 0) != (mesh.w == 0),
                    "each BLAS has exactly one root"
                );
            }
            let mut seed = seed;
            let mut hits = 0;
            for _ in 0..1500 {
                let origin = Vec3::new(
                    random(&mut seed) * 50.0 - 15.0,
                    10.0,
                    random(&mut seed) * 30.0 - 14.0,
                );
                let direction =
                    Vec3::new(random(&mut seed) - 0.5, -1.0, random(&mut seed) - 0.5).normalize();
                // Only the retained meshes lie below y = -2; stop rays above
                // them hitting the static soup first.
                let start = origin + direction * ((origin.y + 2.0) / -direction.y);
                let expected = brute_force(meshes, start, direction);
                let found = trace_words(words, start, direction).map(|h| h.0);
                match (expected, found) {
                    (Some(e), Some(f)) => {
                        assert!((e - f).abs() < 1e-3 * e.max(1.0), "{e} vs {f}");
                        hits += 1;
                    }
                    (None, None) => {}
                    other => panic!("retained BLAS disagrees with brute force: {other:?}"),
                }
            }
            assert!(hits > 500, "too few hits ({hits})");
        };
        check(&gi, &[&near, &far], 13);

        // A same-sized edit rewrites its slot in place.
        let edited = heightfield(24, 17, Vec3::new(-12.0, -6.0, -12.0));
        gi.set_retained(
            &[RetainedChange {
                key: 1,
                mesh: Some(&edited),
            }],
            7,
        );
        let ranges = gi.refresh().expect("same-sized edit fits its slot");
        assert!(!ranges.scene.is_empty());
        check(&gi, &[&edited, &far], 19);

        // Growing past the slot moves the mesh to the tail.
        let grown = heightfield(40, 23, Vec3::new(-12.0, -6.0, -12.0));
        gi.set_retained(
            &[RetainedChange {
                key: 1,
                mesh: Some(&grown),
            }],
            7,
        );
        gi.refresh();
        check(&gi, &[&grown, &far], 29);
    }

    #[test]
    fn packed_wide_decode_finds_nearest_triangles_and_blockers() {
        let mut scene = SceneData::default();
        let vertex = |offset: Vec3, corner: Vec3| Vertex {
            position: (offset + corner).extend(0.0).to_array(),
            normal: [0.0, 0.0, 1.0, 0.0],
            tangent: [0.0; 4],
            color: [1.0; 4],
        };
        // A one-triangle BLAS precedes the wide tree, so the wide leaves must
        // add a nonzero mesh triangle base rather than using absolute IDs.
        let corners = [Vec3::ZERO, Vec3::X, Vec3::Y];
        scene
            .vertices
            .extend(corners.map(|corner| vertex(Vec3::NEG_X * 4.0, corner)));
        scene.indices.extend([0, 1, 2]);
        for i in 0..64u32 {
            let z = (i * 29 % 64) as f32;
            scene
                .vertices
                .extend(corners.map(|corner| vertex(Vec3::Z * z, corner)));
            scene.indices.extend([0, 1, 2].map(|corner| i * 3 + corner));
        }
        scene.primitives.extend([
            Primitive {
                first_index: 0,
                index_count: 3,
                base_vertex: 0,
                material: 0,
            },
            Primitive {
                first_index: 3,
                index_count: 64 * 3,
                base_vertex: 3,
                material: 0,
            },
        ]);
        scene.materials.push(MaterialData::default());
        scene.draws.extend([0, 1].map(|primitive| Draw {
            transform: Mat4::IDENTITY,
            primitive,
        }));

        let mut gi = GiScene::default();
        gi.add(&scene, 0);
        let words = gi.pack();
        let [mask, base, _, stride] = words.scene_words[0].to_array();
        let index = words.scene_words[mask as usize + 1][1];
        let mesh = words.scene_words[(base + index * stride) as usize + 8];
        assert_eq!((mesh.x, mesh.y, mesh.z), (0, 1, 64));
        assert!(mesh.w >= mask + 2);
        let (min, max, root, records) = packed::root(&words.scene_words, mesh.w as usize);
        assert_eq!((min, max), (Vec3::ZERO, Vec3::new(1.0, 1.0, 63.0)));
        let Reference::Branch(at) = packed::reference(records, root) else {
            panic!("expected a packed wide root");
        };
        assert_eq!(packed::children(records, at).count(), 4);

        for z in [0, 15, 31, 47, 63] {
            let origin = Vec3::new(0.25, 0.25, z as f32 + 0.75);
            let (distance, instance, triangle) = trace_words(words, origin, Vec3::NEG_Z)
                .expect("each ray reaches a triangle in the wide BLAS");
            let input = (0..64u32)
                .find(|i| i * 29 % 64 == z)
                .expect("shuffled layer exists");
            assert_eq!(
                (
                    distance,
                    instance,
                    words.geometry_words[1 + triangle as usize].x
                ),
                (0.75, index, 3 + input * 3),
            );
            assert!(trace_words_with_limit(words, origin, Vec3::NEG_Z, 0.75, true).is_none());
            assert!(trace_words_with_limit(words, origin, Vec3::NEG_Z, 0.8, true).is_some());
        }
        let miss = Vec3::new(0.9, 0.9, 64.0);
        assert!(trace_words_with_limit(words, miss, Vec3::NEG_Z, 100.0, true).is_none());

        // Retained updates keep their binary route and never dirty the static
        // arena, even when a nearer retained triangle changes the answer.
        let arena_start = mask as usize + 2;
        let arena = words.scene_words[arena_start..].to_vec();
        let retained = RetainedMesh {
            vertices: corners
                .map(|corner| vertex(Vec3::Z * 63.5, corner))
                .to_vec(),
            indices: vec![0, 1, 2],
            bounds: (Vec3::Z * 63.5, Vec3::new(1.0, 1.0, 63.5)),
            plain: false,
        };
        gi.set_retained(
            &[RetainedChange {
                key: 1,
                mesh: Some(&retained),
            }],
            1,
        );
        let ranges = gi
            .refresh()
            .expect("retained triangle fits the reserved slots");
        assert!(ranges.scene.iter().all(|range| range.end <= arena_start));
        let words = gi.words();
        assert_eq!(&words.scene_words[arena_start..], arena.as_slice());
        let ray = Vec3::new(0.25, 0.25, 64.0);
        let (distance, index, _) =
            trace_words(words, ray, Vec3::NEG_Z).expect("retained triangle hits");
        assert_eq!(distance, 0.5);
        assert_eq!(words.scene_words[(base + index * stride) as usize + 8].w, 0);
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
        let begin = mask as usize + 1;
        let end = begin + (count as usize).div_ceil(4);
        let slots = &gi.scene_words[begin..end];
        assert!(
            (0..count).all(|i| gi.scene_words[(base + i * stride) as usize + 8].w as usize >= end)
        );
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
        let emissive = (0..count)
            .filter(|&i| gi.scene_words[words(i) + 7].w == 11)
            .count();
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
            assert!(node.count <= 4);
            for i in node.first..node.first + node.count {
                covered[i as usize] += 1;
                let p = points[i as usize];
                assert!(p.cmpge(node.min).all() && p.cmple(node.max).all());
            }
        }
        assert!(covered.iter().all(|&c| c == 1));
    }

    #[test]
    fn collinear_bounds_stay_shallow_at_f32_extremes() {
        let mut points: Vec<_> = (-1024..1024)
            .map(|i| Vec3::X * (f32::MAX * (i as f32 / 1024.0)))
            .collect();
        let nodes = build_tree(&mut points, |&p| (p, p));
        let mut pending = vec![(0, 0)];
        while let Some((index, depth)) = pending.pop() {
            assert!(depth <= 10, "collinear tree reached depth {depth}");
            let node = nodes[index];
            let first = node.first as usize;
            if node.count == 0 {
                pending.extend([(first, depth + 1), (first + 1, depth + 1)]);
            } else {
                assert!(node.count <= 4);
                for p in &points[first..first + node.count as usize] {
                    assert!(p.cmpge(node.min).all() && p.cmple(node.max).all());
                }
            }
        }
    }

    #[test]
    fn skewed_tree_fits_shader_stack_without_oversized_leaves() {
        // Each scale is sixteen times larger than its predecessor, so SAH
        // prefers to peel off one cluster at a time near the top of the tree.
        let mut points: Vec<_> = (-104..=96)
            .step_by(4)
            .flat_map(|exponent| [Vec3::splat(2.0_f32.powi(exponent)); 8])
            .collect();
        let nodes = bvh::build_mesh(&mut points, |&p| (p, p));
        let mut pending = vec![(0, 0)];
        let mut leaves = 0;
        while let Some((index, depth)) = pending.pop() {
            assert!(
                depth <= 48,
                "packed traversal stack cannot fit depth {depth}"
            );
            let node = nodes[index];
            let first = node.first as usize;
            if node.count == 0 {
                pending.extend([(first, depth + 1), (first + 1, depth + 1)]);
            } else {
                assert!(node.count <= 8);
                leaves += node.count as usize;
                for p in &points[first..first + node.count as usize] {
                    assert!(p.cmpge(node.min).all() && p.cmple(node.max).all());
                }
            }
        }
        assert_eq!(leaves, points.len());
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
        let nodes = bvh::build_mesh(&mut points, |&p| (p, p));
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
                assert!(node.count <= 8);
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
