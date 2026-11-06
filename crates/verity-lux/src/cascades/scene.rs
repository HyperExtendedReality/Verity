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

use glam::{Affine3A, Mat4, UVec3, UVec4, Vec3};
use rayon::prelude::*;
use std::{collections::BTreeMap, mem, ops::Range};
use verity_render::{
    RetainedChange, RetainedMesh, Vertex,
    asset::{AlphaMode, SceneData},
};

pub const VERTEX_WORDS: u32 = 5;
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
        let nodes = build_tree(&mut triangles, |t| (t.min, t.max));
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
}

impl GiScene {
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
        for primitive in &scene.primitives {
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
            let first = self.triangles.len() as u32;
            let root = self.blas.len() as u32;
            let nodes = build_tree(&mut local, |t| (t.min, t.max));
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
        let mut scene_words = vec![UVec4::ZERO; mask as usize + 1];
        scene_words[0] = UVec4::new(mask, base, 0, INSTANCE_WORDS as u32);
        let blas_words = node_words(&self.blas, self.tlas_room);
        for (word, value) in (1 + 2 * self.tlas_room as usize..).zip(blas_words) {
            scene_words[word] = value;
        }
        *scene_words.last_mut().expect("holds the layer mask") = LAYER_MASK;

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

        self.packed = GiWords {
            geometry_words,
            scene_words,
            triangles: 0,
            instances: count as u32,
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

    /// The TLAS and every instance, which are small enough to rewrite whole.
    /// Returns the ranges of `scene_words` written.
    fn write_top(&mut self) -> [Range<usize>; 2] {
        let mut instances = self.instances.clone();
        instances.extend(self.retained.values().map(|blas| {
            let slot = blas.slot.expect("laid out by pack");
            let top = blas.nodes[0];
            // Retained vertices are in world space.
            Instance {
                world_from_local: Affine3A::IDENTITY,
                local_from_world: Affine3A::IDENTITY,
                min: top.min,
                max: top.max,
                root: slot.nodes,
                first: slot.triangles,
                count: blas.triangles.len() as u32,
                flags: FLAG_CULL_BACK,
                material: self.retained_material,
            }
        }));
        let tlas = build_tree(&mut instances, |i| (i.min, i.max));
        self.packed.instances = instances.len() as u32;
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
        for (index, instance) in instances.iter().enumerate() {
            for transform in [instance.local_from_world, instance.world_from_local] {
                let rows = Mat4::from(transform).transpose();
                for row in [rows.x_axis, rows.y_axis, rows.z_axis] {
                    push(UVec4::from_array(row.to_array().map(f32::to_bits)));
                }
            }
            // No hardware-opaque hint yet.
            push(UVec4::new(mask, 1, 0, instance.flags));
            push(UVec4::new(index as u32, 0, 0, instance.material));
            push(UVec4::new(
                self.tlas_room + instance.root,
                instance.first,
                instance.count,
                0,
            ));
        }
        let end = base + instances.len() * INSTANCE_WORDS;
        [0..1 + 2 * tlas.len(), base..end]
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
    if elements.is_empty() {
        return Vec::new();
    }
    fn build<T>(
        nodes: &mut Vec<BvhNode>,
        index: usize,
        elements: &mut [T],
        first: usize,
        depth: u32,
        bounds: &impl Fn(&T) -> (Vec3, Vec3),
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
        build(nodes, children, left, first, depth + 1, bounds);
        build(
            nodes,
            children + 1,
            right,
            first + midpoint,
            depth + 1,
            bounds,
        );
    }
    let mut nodes = vec![BvhNode::default()];
    build(&mut nodes, 0, elements, 0, 0, &bounds);
    nodes
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
        assert_eq!(mask as usize + 1, gi.scene_words.len());
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
}
