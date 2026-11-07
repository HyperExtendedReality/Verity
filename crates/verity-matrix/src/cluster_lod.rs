//! Continuous LOD: a cluster DAG whose coarse clusters may be voxel bricks.
//!
//! [`ClusterLod::cook`] builds meshoptimizer's cluster DAG for one static
//! material primitive. Simplifying a disconnected aggregate such as a canopy
//! thins it, because whole leaves vanish at little geometric error. So a mesh
//! whose coarsest cut has lost most of its area counts as an aggregate, and any
//! of its clusters that thinned is rebuilt from the original triangles as up to
//! [`MAX_CLUSTER_BRICKS`] bricks of 4x4x4 voxels, as Nanite Foliage does.
//!
//! Each simplified cluster inherits a share of the original triangles, disjoint
//! from its siblings, so every level voxelizes source geometry rather than the
//! previous approximation and a DAG cut covers the source exactly once.
//!
//! A voxel is solid or empty, and is kept with probability equal to the largest
//! fraction of its face its contents cover along an axis. Choosing by a stable
//! hash keeps mean density without the per-pixel noise of stochastic opacity,
//! which this renderer has no temporal filter to remove.
//!
//! A cluster should be drawn when both hold for threshold `t`:
//! - its group's simplified error, projected, is above `t`;
//! - it has no refined group, or that group's simplified error is at or below `t`.

use std::collections::HashMap;

use rayon::prelude::*;

use meshopt::{
    ClodBounds, ClodCluster, ClodGroup, ClodMesh, VertexDataAdapter, clod_build,
    clod_build_hierarchy, clod_default_config, typed_to_bytes,
};

/// Triangles per cluster; the visibility ID gives a triangle seven bits.
pub const MAX_CLUSTER_TRIANGLES: usize = 128;
/// Voxels along each side of a brick.
pub const BRICK_SIZE: i32 = 4;
/// Bricks in one voxel cluster.
pub const MAX_CLUSTER_BRICKS: usize = 128;
/// No refined group: original geometry.
pub const NO_GROUP: u32 = u32::MAX;
/// Bump when cooking changes, so older cached results are not reused.
pub const COOK_VERSION: u32 = 1;

const NODE_WIDTH: usize = 8;
/// A mesh is an aggregate if its coarsest cut keeps less than this of its area.
/// Measured: foliage keeps 0.2-0.3%, New Sponza's solid meshes 7-100%.
const AGGREGATE_AREA: f64 = 0.05;
/// A simplified cluster keeping less than this of its source's area has thinned.
const MIN_RETAINED_AREA: f64 = 0.9;
/// Voxels across a cluster's source at most; finer ones cannot pay for themselves.
const MAX_VOXELS_ACROSS: f32 = 64.0;
/// Least growth per retry, doubling the size in three steps.
const VOXEL_GROWTH: f32 = 1.26;
/// Source triangles a brick must replace. A brick draws 12 triangles and marches,
/// so finer voxels than this cost more than the triangles they stand for; coarser
/// ones raise the error instead, keeping the exact triangles until voxels are
/// near pixel size.
const TRIANGLES_PER_BRICK: usize = 16;
/// Triangles a cluster samples at most. A cluster holds 8192 voxels, so far
/// fewer than a whole tree's triangles decide their contents.
const MAX_SAMPLED_TRIANGLES: usize = 1 << 17;
/// A voxel's normals agree when their mean is at least this long.
const MIN_MEAN_NORMAL: f32 = 0.9;
/// Surface samples per triangle at most.
const MAX_TRIANGLE_SAMPLES: u32 = 1024;

/// Vertex data of one primitive. Slices are indexed by `indices`.
#[derive(Clone, Copy)]
pub struct Source<'a> {
    pub positions: &'a [[f32; 3]],
    pub normals: &'a [[f32; 3]],
    pub uvs: &'a [[f32; 2]],
    pub indices: &'a [u32],
}

/// What a point on a triangle looks like, for voxel baking.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceSample {
    /// Linear base color.
    pub color: [f32; 3],
    /// Fraction of the point that exists: 0 where an alpha test removes it.
    pub coverage: f32,
}

/// Sphere bounds with a simplification error, in object space.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Bounds {
    pub center: [f32; 3],
    pub radius: f32,
    pub error: f32,
}

impl From<ClodBounds> for Bounds {
    fn from(bounds: ClodBounds) -> Self {
        Self {
            center: bounds.center,
            radius: bounds.radius,
            error: bounds.error,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Group {
    /// Bounds and error of what simplifying this group produced;
    /// the error is `f32::MAX` for terminal groups.
    pub simplified: Bounds,
    pub first_cluster: u32,
    pub cluster_count: u32,
    pub depth: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Cluster {
    pub bounds: Bounds,
    /// The group whose simplification produced this cluster, or [`NO_GROUP`].
    pub refined: u32,
    pub first_index: u32,
    pub triangle_count: u32,
    pub first_brick: u32,
    pub brick_count: u32,
    pub depth: u32,
}

/// 4x4x4 voxels. Bit `(z * 4 + y) * 4 + x` of `mask` marks a solid voxel, whose
/// attributes are at `first_voxel` plus the count of lower set bits.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Brick {
    pub origin: [f32; 3],
    pub voxel_size: f32,
    pub mask: [u32; 2],
    pub first_voxel: u32,
    pub padding: u32,
}

/// One solid voxel.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Voxel {
    /// sRGB base color in the low three bytes; high byte: the spread of the
    /// normals inside the voxel, 0 when they agree and 255 when they cancel.
    pub color: u32,
    /// Octahedral mean normal, 16 bits per axis.
    pub normal: u32,
}

/// Spatial tree over groups; `group` is [`NO_GROUP`] for inner nodes.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Node {
    pub bounds: Bounds,
    pub group: u32,
    pub first_child: u32,
    pub child_count: u32,
}

#[derive(Clone, Debug, Default)]
pub struct ClusterLod {
    pub groups: Vec<Group>,
    pub clusters: Vec<Cluster>,
    /// Triangle lists of `clusters`, relative to the source vertices.
    pub indices: Vec<u32>,
    pub bricks: Vec<Brick>,
    pub voxels: Vec<Voxel>,
    /// A forest: the first `root_count` nodes are roots, one per DAG level.
    pub nodes: Vec<Node>,
    pub root_count: u32,
    /// Levels below a root, at most.
    pub tree_depth: u32,
    /// Bounds of the whole primitive, in object space.
    pub center: [f32; 3],
    pub radius: f32,
}

type Triangle = [u32; 3];

impl ClusterLod {
    /// Build the DAG, or `None` for empty or invalid input.
    ///
    /// `surface` receives a triangle's vertices and barycentrics. It is only
    /// called for aggregates.
    pub fn cook(
        source: Source<'_>,
        surface: impl Fn(Triangle, [f32; 3]) -> SurfaceSample + Sync,
    ) -> Option<Self> {
        Self::build(source, true, surface)
    }

    fn build(
        source: Source<'_>,
        voxels: bool,
        surface: impl Fn(Triangle, [f32; 3]) -> SurfaceSample + Sync,
    ) -> Option<Self> {
        let vertex_count = source.positions.len();
        if source.indices.is_empty()
            || !source.indices.len().is_multiple_of(3)
            || source.normals.len() != vertex_count
            || source.uvs.len() != vertex_count
            || source.indices.iter().any(|&i| i as usize >= vertex_count)
            || source.positions.iter().flatten().any(|v| !v.is_finite())
        {
            return None;
        }
        let vertices = VertexDataAdapter::new(typed_to_bytes(source.positions), 12, 0).ok()?;
        // Normal and UV discontinuities are seams permissive simplification keeps.
        let attributes: Vec<[f32; 5]> = source
            .normals
            .iter()
            .zip(source.uvs)
            .map(|(n, uv)| [n[0], n[1], n[2], uv[0], uv[1]].map(finite_or_zero))
            .collect();
        let mesh = ClodMesh {
            vertex_attributes: bytemuck::cast_slice(&attributes),
            vertex_attributes_stride: 20,
            attribute_weights: &[0.5, 0.5, 0.5, 0.5, 0.5],
            attribute_protect_mask: 0b11111,
            ..ClodMesh::new(source.indices, &vertices)
        };
        let mut groups: Vec<ClodGroup> = Vec::new();
        let mut members: Vec<Vec<Member>> = Vec::new();
        clod_build(
            clod_default_config(MAX_CLUSTER_TRIANGLES),
            &mesh,
            |group: ClodGroup, clusters: &[ClodCluster<'_>]| {
                groups.push(group);
                members.push(clusters.iter().map(Member::new).collect());
                groups.len() as i32 - 1
            },
        );
        if voxels && is_aggregate(source, &groups, &members) {
            voxelize_clusters(source, &mut groups, &mut members, &surface);
        }
        // Raised errors must stay monotonic for a cut to exist; refined groups
        // always precede the groups they feed.
        for id in 0..groups.len() {
            let inherited = members[id]
                .iter()
                .filter(|member| member.refined >= 0)
                .map(|member| groups[member.refined as usize].simplified.error)
                .fold(0.0, f32::max);
            let group = &mut groups[id];
            group.simplified.error = group.simplified.error.max(inherited);
        }

        let mut lod = Self::default();
        for (group, members) in groups.iter().zip(members) {
            let first_cluster = lod.clusters.len() as u32;
            let cluster_count = members.len() as u32;
            for member in members {
                let first_brick = lod.bricks.len() as u32;
                let first_voxel = lod.voxels.len() as u32;
                let (triangles, bricks, voxels) = match member.voxels {
                    Some(voxels) => (Vec::new(), voxels.bricks, voxels.voxels),
                    None => (member.triangles, Vec::new(), Vec::new()),
                };
                lod.clusters.push(Cluster {
                    bounds: member.bounds,
                    refined: u32::try_from(member.refined).unwrap_or(NO_GROUP),
                    first_index: lod.indices.len() as u32,
                    triangle_count: triangles.len() as u32,
                    first_brick,
                    brick_count: bricks.len() as u32,
                    depth: group.depth as u32,
                });
                lod.indices.extend(triangles.iter().flatten());
                lod.bricks.extend(bricks.into_iter().map(|brick| Brick {
                    first_voxel: brick.first_voxel + first_voxel,
                    ..brick
                }));
                lod.voxels.extend(voxels);
            }
            lod.groups.push(Group {
                simplified: group.simplified.into(),
                first_cluster,
                cluster_count,
                depth: group.depth as u32,
            });
        }
        let levels = groups.iter().map(|g| g.depth as usize + 1).max()?;
        lod.nodes = clod_build_hierarchy(&groups, NODE_WIDTH, levels)
            .iter()
            .map(|node| Node {
                bounds: node.bounds.into(),
                group: u32::try_from(node.group).unwrap_or(NO_GROUP),
                first_child: node.child_offset,
                child_count: node.child_count,
            })
            .collect();
        lod.root_count = levels as u32;
        lod.tree_depth = (0..levels).map(|root| lod.height(root)).max()?;
        (lod.center, lod.radius) = enclosing_sphere(source);
        Some(lod)
    }

    fn height(&self, node: usize) -> u32 {
        let node = self.nodes[node];
        let children = node.first_child as usize..(node.first_child + node.child_count) as usize;
        if node.group != NO_GROUP {
            return 0;
        }
        children
            .map(|child| 1 + self.height(child))
            .max()
            .unwrap_or(0)
    }

    /// Whether any cluster is drawn as voxels.
    pub fn has_voxels(&self) -> bool {
        !self.bricks.is_empty()
    }

    /// The whole LOD as bytes, for [`ClusterLod::from_bytes`] in the same build.
    pub fn to_bytes(&self) -> Vec<u8> {
        let header = [
            COOK_VERSION,
            self.groups.len() as u32,
            self.clusters.len() as u32,
            self.indices.len() as u32,
            self.bricks.len() as u32,
            self.voxels.len() as u32,
            self.nodes.len() as u32,
            self.root_count,
            self.tree_depth,
            self.center[0].to_bits(),
            self.center[1].to_bits(),
            self.center[2].to_bits(),
            self.radius.to_bits(),
        ];
        let mut bytes = bytemuck::cast_slice(&header).to_vec();
        bytes.extend_from_slice(bytemuck::cast_slice(&self.groups));
        bytes.extend_from_slice(bytemuck::cast_slice(&self.clusters));
        bytes.extend_from_slice(bytemuck::cast_slice(&self.indices));
        bytes.extend_from_slice(bytemuck::cast_slice(&self.bricks));
        bytes.extend_from_slice(bytemuck::cast_slice(&self.voxels));
        bytes.extend_from_slice(bytemuck::cast_slice(&self.nodes));
        bytes
    }

    /// Read [`ClusterLod::to_bytes`] for geometry of `vertex_count` vertices.
    /// `None` unless every index and range is in bounds, because shaders index
    /// with these values unchecked.
    pub fn from_bytes(bytes: &[u8], vertex_count: usize) -> Option<Self> {
        fn take<T: bytemuck::Pod>(bytes: &mut &[u8], count: usize) -> Option<Vec<T>> {
            let size = count.checked_mul(size_of::<T>())?;
            let (head, tail) = bytes.split_at_checked(size)?;
            *bytes = tail;
            Some(bytemuck::pod_collect_to_vec(head))
        }
        let mut rest = bytes;
        let header: Vec<u32> = take(&mut rest, 13)?;
        if header[0] != COOK_VERSION {
            return None;
        }
        let count = |index: usize| header[index] as usize;
        let lod = Self {
            groups: take(&mut rest, count(1))?,
            clusters: take(&mut rest, count(2))?,
            indices: take(&mut rest, count(3))?,
            bricks: take(&mut rest, count(4))?,
            voxels: take(&mut rest, count(5))?,
            nodes: take(&mut rest, count(6))?,
            root_count: header[7],
            tree_depth: header[8],
            center: [header[9], header[10], header[11]].map(f32::from_bits),
            radius: f32::from_bits(header[12]),
        };
        (rest.is_empty() && lod.is_valid(vertex_count)).then_some(lod)
    }

    fn is_valid(&self, vertex_count: usize) -> bool {
        let within = |first: u32, count: u32, length: usize| {
            first
                .checked_add(count)
                .is_some_and(|end| end as usize <= length)
        };
        let group = |id: u32| id == NO_GROUP || (id as usize) < self.groups.len();
        let brick_voxels = |b: &Brick| b.mask[0].count_ones() + b.mask[1].count_ones();
        self.indices.iter().all(|&i| (i as usize) < vertex_count)
            && self
                .groups
                .iter()
                .all(|g| within(g.first_cluster, g.cluster_count, self.clusters.len()))
            && self.clusters.iter().all(|c| {
                group(c.refined)
                    && c.triangle_count as usize <= MAX_CLUSTER_TRIANGLES
                    && c.first_index
                        .checked_add(c.triangle_count * 3)
                        .is_some_and(|end| end as usize <= self.indices.len())
                    && within(c.first_brick, c.brick_count, self.bricks.len())
            })
            && self
                .bricks
                .iter()
                .all(|b| within(b.first_voxel, brick_voxels(b), self.voxels.len()))
            && self
                .nodes
                .iter()
                .all(|n| group(n.group) && within(n.first_child, n.child_count, self.nodes.len()))
            && self.root_count as usize <= self.nodes.len()
            && self.root_count > 0
    }
}

/// Whether simplifying all the way removed most of the surface.
fn is_aggregate(source: Source<'_>, groups: &[ClodGroup], members: &[Vec<Member>]) -> bool {
    let area = |select: &dyn Fn(usize, &Member) -> bool| -> f64 {
        members
            .iter()
            .enumerate()
            .flat_map(|(group, members)| members.iter().map(move |member| (group, member)))
            .filter(|&(group, member)| select(group, member))
            .map(|(_, member)| surface_area(source.positions, &member.triangles))
            .sum()
    };
    let original = area(&|_, member| member.refined < 0);
    let coarsest = area(&|group, _| groups[group].simplified.error == f32::MAX);
    coarsest < original * AGGREGATE_AREA
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

struct Member {
    refined: i32,
    bounds: Bounds,
    triangles: Vec<Triangle>,
    voxels: Option<VoxelCluster>,
}

impl Member {
    fn new(cluster: &ClodCluster<'_>) -> Self {
        Self {
            refined: cluster.refined,
            bounds: cluster.bounds.into(),
            triangles: cluster.indices.as_chunks::<3>().0.to_vec(),
            voxels: None,
        }
    }
}

struct VoxelCluster {
    bricks: Vec<Brick>,
    voxels: Vec<Voxel>,
}

fn voxelize_clusters(
    source: Source<'_>,
    groups: &mut [ClodGroup],
    members: &mut [Vec<Member>],
    surface: &(impl Fn(Triangle, [f32; 3]) -> SurfaceSample + Sync),
) {
    // Who each group's simplification produced: (group, member).
    let mut produced: Vec<Vec<(usize, usize)>> = vec![Vec::new(); groups.len()];
    for (group, members) in members.iter().enumerate() {
        for (index, member) in members.iter().enumerate() {
            if let Ok(refined) = usize::try_from(member.refined) {
                produced[refined].push((group, index));
            }
        }
    }
    // The original triangles behind each cluster, consumed as groups merge.
    let mut shares: Vec<Vec<Vec<Triangle>>> = members
        .iter()
        .map(|members| {
            members
                .iter()
                .map(|member| {
                    if member.refined < 0 {
                        member.triangles.clone()
                    } else {
                        Vec::new()
                    }
                })
                .collect()
        })
        .collect();
    let mut voxelized = vec![false; groups.len()];
    for group in 0..groups.len() {
        // A group's clusters are independent; one big mesh gives few other chances.
        let built: Vec<_> = members[group]
            .par_iter()
            .zip(&shares[group])
            .map(|(member, share)| {
                let refined = usize::try_from(member.refined).ok()?;
                let error = groups[refined].simplified.error;
                let area = surface_area(source.positions, share);
                let kept = surface_area(source.positions, &member.triangles);
                let thinned = kept < area * MIN_RETAINED_AREA;
                // Everything coarser than voxels must be voxels, or density returns.
                if !(voxelized[refined] || thinned) || share.is_empty() {
                    return None;
                }
                let (cluster, size) = voxelize(source, share, error, surface);
                Some((refined, cluster, error.max(size * 0.5)))
            })
            .collect();
        for (member, built) in members[group].iter_mut().zip(built) {
            if let Some((refined, cluster, error)) = built {
                member.voxels = Some(cluster);
                let simplified = &mut groups[refined].simplified;
                simplified.error = simplified.error.max(error);
                voxelized[group] = true;
            }
        }
        let merged: Vec<Triangle> = shares[group].iter_mut().flat_map(std::mem::take).collect();
        let outputs = &produced[group];
        if outputs.is_empty() {
            continue;
        }
        let centers: Vec<[f32; 3]> = outputs
            .iter()
            .map(|&(g, i)| members[g][i].bounds.center)
            .collect();
        for triangle in merged {
            let centroid = centroid(source.positions, triangle);
            let nearest = (0..centers.len())
                .min_by(|&a, &b| {
                    distance2(centroid, centers[a]).total_cmp(&distance2(centroid, centers[b]))
                })
                .expect("outputs is not empty");
            let (g, i) = outputs[nearest];
            shares[g][i].push(triangle);
        }
    }
}

/// Voxelize at the size matching `error`, or larger if it would not fit.
/// Returns the voxels and the size used.
fn voxelize(
    source: Source<'_>,
    triangles: &[Triangle],
    error: f32,
    surface: &(impl Fn(Triangle, [f32; 3]) -> SurfaceSample + Sync),
) -> (VoxelCluster, f32) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for &vertex in triangles.iter().flatten() {
        let position = source.positions[vertex as usize];
        for axis in 0..3 {
            min[axis] = min[axis].min(position[axis]);
            max[axis] = max[axis].max(position[axis]);
        }
    }
    let extent = (0..3).map(|a| max[a] - min[a]).fold(0.0, f32::max);
    let brick_budget = (triangles.len() / TRIANGLES_PER_BRICK).clamp(1, MAX_CLUSTER_BRICKS);
    let mut size = (error * 2.0)
        .max(extent / MAX_VOXELS_ACROSS)
        .max(f32::MIN_POSITIVE);
    loop {
        let cluster = voxelize_at(source, triangles, size, surface);
        if cluster.bricks.len() <= brick_budget {
            return (cluster, size);
        }
        // Surfaces fill bricks with the square of the resolution.
        let excess = cluster.bricks.len() as f32 / brick_budget as f32;
        size *= excess.sqrt().max(VOXEL_GROWTH);
    }
}

#[derive(Default)]
struct Accumulator {
    /// Area projected along each axis.
    projected: [f64; 3],
    area: f64,
    normal: [f64; 3],
    color: [f64; 3],
    samples: i32,
    /// One sample's normal, each sample equally likely by area.
    picked: [f32; 3],
}

fn voxelize_at(
    source: Source<'_>,
    triangles: &[Triangle],
    size: f32,
    surface: &impl Fn(Triangle, [f32; 3]) -> SurfaceSample,
) -> VoxelCluster {
    let mut cells = HashMap::<[i32; 3], Accumulator>::new();
    let spacing = size * 0.5;
    let stride = triangles.len().div_ceil(MAX_SAMPLED_TRIANGLES);
    for &triangle in triangles.iter().step_by(stride) {
        let [a, b, c] = triangle.map(|v| source.positions[v as usize]);
        let area = triangle_area(a, b, c);
        if area <= 0.0 {
            continue;
        }
        let side = ((area / f64::from(spacing * spacing)).sqrt().ceil() as u32)
            .clamp(1, MAX_TRIANGLE_SAMPLES.isqrt());
        let weight = area * stride as f64 / f64::from(side * side);
        let [na, nb, nc] = triangle.map(|v| source.normals[v as usize]);
        let face = cross(sub(b, a), sub(c, a));
        for cell in 0..side * side {
            let bary = stratified(cell, side);
            let sample = surface(triangle, bary);
            let coverage = f64::from(sample.coverage);
            if coverage <= 0.0 {
                continue;
            }
            let position = blend(a, b, c, bary);
            let mut normal = blend(na, nb, nc, bary);
            if dot(normal, normal) < 1.0e-12 {
                normal = face;
            }
            let normal = normalize(normal);
            let key = position.map(|v| (v / size).floor() as i32);
            let cell = cells.entry(key).or_default();
            let amount = weight * coverage;
            // The geometric normal measures coverage: shading normals can lie.
            let unit_face = normalize(face);
            for axis in 0..3 {
                cell.projected[axis] += amount * f64::from(unit_face[axis].abs());
                cell.normal[axis] += amount * f64::from(normal[axis]);
                cell.color[axis] += amount * f64::from(sample.color[axis]);
            }
            cell.area += amount;
            cell.samples += 1;
            if f64::from(hash([key[0] ^ cell.samples, key[1], key[2]])) * cell.area < amount {
                cell.picked = normal;
            }
        }
    }
    let face_area = f64::from(size) * f64::from(size);
    let mut bricks = HashMap::<[i32; 3], Vec<(u32, Voxel)>>::new();
    for (key, cell) in cells {
        let coverage = cell.projected.into_iter().fold(0.0, f64::max) / face_area;
        if f64::from(hash(key)) >= coverage {
            continue;
        }
        let brick = key.map(|v| v.div_euclid(BRICK_SIZE));
        let local = key.map(|v| v.rem_euclid(BRICK_SIZE) as u32);
        let bit = (local[2] * 4 + local[1]) * 4 + local[0];
        bricks.entry(brick).or_default().push((bit, pack(&cell)));
    }
    let mut keys: Vec<_> = bricks.keys().copied().collect();
    keys.sort_unstable();
    let mut cluster = VoxelCluster {
        bricks: Vec::with_capacity(keys.len()),
        voxels: Vec::new(),
    };
    for key in keys {
        let mut voxels = bricks.remove(&key).expect("key came from this map");
        voxels.sort_unstable_by_key(|&(bit, _)| bit);
        let mut mask = [0; 2];
        for &(bit, _) in &voxels {
            mask[bit as usize / 32] |= 1 << (bit % 32);
        }
        cluster.bricks.push(Brick {
            origin: key.map(|v| (v * BRICK_SIZE) as f32 * size),
            voxel_size: size,
            mask,
            first_voxel: cluster.voxels.len() as u32,
            padding: 0,
        });
        cluster.voxels.extend(voxels.into_iter().map(|(_, v)| v));
    }
    cluster
}

fn pack(cell: &Accumulator) -> Voxel {
    let color = cell.color.map(|c| (c / cell.area) as f32);
    let srgb = color.map(|c| {
        let c = c.clamp(0.0, 1.0);
        let encoded = if c <= 0.003_130_8 {
            c * 12.92
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
        (encoded * 255.0 + 0.5) as u32
    });
    let mean = cell.normal.map(|n| (n / cell.area) as f32);
    let length = dot(mean, mean).sqrt();
    let spread = ((1.0 - length.min(1.0)) * 255.0 + 0.5) as u32;
    // Shading a mean normal is not the mean of the shading, so a voxel of
    // scattered leaves shades as one of them. Nanite picks per pixel; picking
    // per voxel keeps the same distribution without noise to filter.
    let unit = if length > MIN_MEAN_NORMAL {
        mean.map(|n| n / length)
    } else {
        cell.picked
    };
    Voxel {
        color: srgb[0] | srgb[1] << 8 | srgb[2] << 16 | spread << 24,
        normal: octahedral(unit),
    }
}

/// Unit vector to 16-bit-per-axis octahedral coordinates.
fn octahedral(n: [f32; 3]) -> u32 {
    let scale = 1.0 / (n[0].abs() + n[1].abs() + n[2].abs());
    let (mut x, mut y) = (n[0] * scale, n[1] * scale);
    if n[2] < 0.0 {
        (x, y) = ((1.0 - y.abs()) * x.signum(), (1.0 - x.abs()) * y.signum());
    }
    let quantize = |v: f32| ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 65535.0 + 0.5) as u32;
    quantize(x) | quantize(y) << 16
}

/// Barycentrics of the centroid of one cell of a triangle split `side` ways.
fn stratified(cell: u32, side: u32) -> [f32; 3] {
    let (row, column) = (cell / side, cell % side);
    let (mut u, mut v) = (
        (column as f32 + 0.5) / side as f32,
        (row as f32 + 0.5) / side as f32,
    );
    if u + v > 1.0 {
        (u, v) = (1.0 - u, 1.0 - v);
    }
    [1.0 - u - v, u, v]
}

/// A stable value in `0..1` for a voxel coordinate.
fn hash(key: [i32; 3]) -> f32 {
    let mut h = 0x9e37_79b9_u32;
    for value in key {
        h = (h ^ value as u32).wrapping_mul(0x85eb_ca6b);
        h ^= h >> 13;
        h = h.wrapping_mul(0xc2b2_ae35);
        h ^= h >> 16;
    }
    (h >> 8) as f32 / (1 << 24) as f32
}

/// Center of the bounding box and the radius reaching every used vertex.
fn enclosing_sphere(source: Source<'_>) -> ([f32; 3], f32) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for &vertex in source.indices {
        let position = source.positions[vertex as usize];
        for axis in 0..3 {
            min[axis] = min[axis].min(position[axis]);
            max[axis] = max[axis].max(position[axis]);
        }
    }
    let center = [0, 1, 2].map(|a| (min[a] + max[a]) * 0.5);
    let radius = source
        .indices
        .iter()
        .map(|&v| distance2(center, source.positions[v as usize]))
        .fold(0.0, f32::max)
        .sqrt();
    (center, radius)
}

fn surface_area(positions: &[[f32; 3]], triangles: &[Triangle]) -> f64 {
    triangles
        .iter()
        .map(|t| {
            let [a, b, c] = t.map(|v| positions[v as usize]);
            triangle_area(a, b, c)
        })
        .sum()
}

fn triangle_area(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f64 {
    let n = cross(sub(b, a), sub(c, a));
    f64::from(dot(n, n)).sqrt() * 0.5
}

fn centroid(positions: &[[f32; 3]], triangle: Triangle) -> [f32; 3] {
    let [a, b, c] = triangle.map(|v| positions[v as usize]);
    [0, 1, 2].map(|i| (a[i] + b[i] + c[i]) / 3.0)
}

fn blend(a: [f32; 3], b: [f32; 3], c: [f32; 3], w: [f32; 3]) -> [f32; 3] {
    [0, 1, 2].map(|i| a[i] * w[0] + b[i] * w[1] + c[i] * w[2])
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let length = dot(v, v).sqrt();
    if length > 0.0 {
        v.map(|c| c / length)
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn distance2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = sub(a, b);
    dot(d, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Mesh {
        positions: Vec<[f32; 3]>,
        normals: Vec<[f32; 3]>,
        uvs: Vec<[f32; 2]>,
        indices: Vec<u32>,
    }

    impl Mesh {
        fn source(&self) -> Source<'_> {
            Source {
                positions: &self.positions,
                normals: &self.normals,
                uvs: &self.uvs,
                indices: &self.indices,
            }
        }
    }

    fn solid(_: Triangle, _: [f32; 3]) -> SurfaceSample {
        SurfaceSample {
            color: [0.2, 0.5, 0.1],
            coverage: 1.0,
        }
    }

    fn grid(n: u32) -> Mesh {
        let mut mesh = Mesh {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
        };
        for y in 0..=n {
            for x in 0..=n {
                let (u, v) = (x as f32 / n as f32, y as f32 / n as f32);
                mesh.positions
                    .push([u * 10.0, v * 10.0, (u * 7.0).sin() * (v * 5.0).cos()]);
                mesh.normals.push([0.0, 0.0, 1.0]);
                mesh.uvs.push([u, v]);
            }
        }
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                mesh.indices
                    .extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
            }
        }
        mesh
    }

    /// Random leaf quads filling a ball, with a small random-number generator.
    fn canopy(leaves: u32, radius: f32, leaf: f32) -> Mesh {
        let mut state = 0x2545_f491_u32;
        let mut random = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as f32 / u32::MAX as f32
        };
        let mut mesh = Mesh {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new(),
        };
        for _ in 0..leaves {
            let center = loop {
                let p = [0, 1, 2].map(|_| random() * 2.0 - 1.0);
                if dot(p, p) <= 1.0 {
                    break p.map(|v| v * radius);
                }
            };
            let normal = normalize([0, 1, 2].map(|_| random() * 2.0 - 1.0));
            let u = normalize(cross(normal, [0.3, 0.5, 0.8]));
            let v = cross(normal, u);
            let base = mesh.positions.len() as u32;
            for (a, b) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                mesh.positions
                    .push([0, 1, 2].map(|i| center[i] + (u[i] * a + v[i] * b) * leaf * 0.5));
                mesh.normals.push(normal);
                mesh.uvs.push([a * 0.5 + 0.5, b * 0.5 + 0.5]);
            }
            mesh.indices
                .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        mesh
    }

    /// Clusters of the cut at `threshold`, ignoring distance.
    fn cut(lod: &ClusterLod, threshold: f32) -> Vec<Cluster> {
        let mut selected = Vec::new();
        for group in &lod.groups {
            if group.simplified.error <= threshold {
                continue;
            }
            let first = group.first_cluster as usize;
            for cluster in &lod.clusters[first..first + group.cluster_count as usize] {
                if cluster.refined == NO_GROUP
                    || lod.groups[cluster.refined as usize].simplified.error <= threshold
                {
                    selected.push(*cluster);
                }
            }
        }
        selected
    }

    /// The same cut found through the tree, as the GPU finds it.
    fn cut_through_tree(lod: &ClusterLod, threshold: f32) -> usize {
        fn visit(lod: &ClusterLod, node: usize, threshold: f32) -> usize {
            let node = lod.nodes[node];
            if node.bounds.error <= threshold {
                return 0;
            }
            if node.group == NO_GROUP {
                return (node.first_child..node.first_child + node.child_count)
                    .map(|child| visit(lod, child as usize, threshold))
                    .sum();
            }
            let group = lod.groups[node.group as usize];
            let first = group.first_cluster as usize;
            lod.clusters[first..first + group.cluster_count as usize]
                .iter()
                .filter(|c| {
                    c.refined == NO_GROUP
                        || lod.groups[c.refined as usize].simplified.error <= threshold
                })
                .count()
        }
        (0..lod.root_count as usize)
            .map(|root| visit(lod, root, threshold))
            .sum()
    }

    fn area(mesh: &Mesh, lod: &ClusterLod, clusters: &[Cluster]) -> f64 {
        clusters
            .iter()
            .map(|c| {
                let first = c.first_index as usize;
                lod.indices[first..first + c.triangle_count as usize * 3]
                    .chunks_exact(3)
                    .map(|t| {
                        triangle_area(
                            mesh.positions[t[0] as usize],
                            mesh.positions[t[1] as usize],
                            mesh.positions[t[2] as usize],
                        )
                    })
                    .sum::<f64>()
            })
            .sum()
    }

    /// Fraction of a square of rays down Z blocked by the source triangles.
    fn source_opacity(mesh: &Mesh, resolution: usize, extent: f32) -> f64 {
        let mut hit = vec![false; resolution * resolution];
        let cell = 2.0 * extent / resolution as f32;
        // Dense sampling of each leaf stands in for rasterizing it.
        for t in mesh.indices.chunks_exact(3) {
            let [a, b, c] = [t[0], t[1], t[2]].map(|v| mesh.positions[v as usize]);
            for i in 0..32 {
                for j in 0..32 - i {
                    let w = [i as f32 / 32.0, j as f32 / 32.0];
                    let p = blend(a, b, c, [1.0 - w[0] - w[1], w[0], w[1]]);
                    let (x, y) = (
                        ((p[0] + extent) / cell) as usize,
                        ((p[1] + extent) / cell) as usize,
                    );
                    if x < resolution && y < resolution {
                        hit[y * resolution + x] = true;
                    }
                }
            }
        }
        hit.iter().filter(|&&h| h).count() as f64 / hit.len() as f64
    }

    /// The same for the bricks of a cut.
    fn voxel_opacity(lod: &ClusterLod, cut: &[Cluster], resolution: usize, extent: f32) -> f64 {
        let mut hit = vec![false; resolution * resolution];
        let cell = 2.0 * extent / resolution as f32;
        for cluster in cut {
            let first = cluster.first_brick as usize;
            for brick in &lod.bricks[first..first + cluster.brick_count as usize] {
                for bit in 0..64 {
                    if brick.mask[bit / 32] & (1 << (bit % 32)) == 0 {
                        continue;
                    }
                    let (x, y) = ((bit % 4) as f32, (bit / 4 % 4) as f32);
                    let low = [
                        brick.origin[0] + x * brick.voxel_size,
                        brick.origin[1] + y * brick.voxel_size,
                    ];
                    let span = |v: f32| ((v + extent) / cell) as usize;
                    let (x0, y0) = (span(low[0]), span(low[1]));
                    let (x1, y1) = (
                        span(low[0] + brick.voxel_size),
                        span(low[1] + brick.voxel_size),
                    );
                    for y in y0..=y1.min(resolution - 1) {
                        for x in x0..=x1.min(resolution - 1) {
                            hit[y * resolution + x] = true;
                        }
                    }
                }
            }
        }
        hit.iter().filter(|&&h| h).count() as f64 / hit.len() as f64
    }

    #[test]
    fn surface_cuts_cover_the_mesh_once_at_every_threshold() {
        let mesh = grid(64);
        let lod = ClusterLod::cook(mesh.source(), solid).unwrap();
        assert!(lod.groups.len() > 8);
        assert!(!lod.has_voxels());
        let full = area(&mesh, &lod, &cut(&lod, 0.0));
        for threshold in [0.0, 0.001, 0.01, 0.05, 0.2, 1.0, 10.0] {
            let selected = cut(&lod, threshold);
            assert!(!selected.is_empty());
            assert_eq!(selected.len(), cut_through_tree(&lod, threshold));
            let covered = area(&mesh, &lod, &selected);
            assert!((covered / full - 1.0).abs() < 0.1, "{covered} of {full}");
        }
        assert!(cut(&lod, 10.0).len() < cut(&lod, 0.0).len());
        assert!(
            lod.clusters
                .iter()
                .all(|c| c.triangle_count as usize <= MAX_CLUSTER_TRIANGLES)
        );
    }

    #[test]
    fn connected_leaves_still_count_as_an_aggregate() {
        // Each leaf is a fan of many triangles, as real foliage is: part size
        // says nothing, what matters is that simplification removes them.
        let mut mesh = canopy(0, 5.0, 0.2);
        let leaves = canopy(800, 5.0, 0.2);
        for quad in leaves.positions.chunks_exact(4) {
            let base = mesh.positions.len() as u32;
            for y in 0..=6 {
                for x in 0..=6 {
                    let (u, v) = (x as f32 / 6.0, y as f32 / 6.0);
                    mesh.positions.push([0, 1, 2].map(|i| {
                        let near = quad[0][i] + (quad[1][i] - quad[0][i]) * u;
                        let far = quad[3][i] + (quad[2][i] - quad[3][i]) * u;
                        near + (far - near) * v
                    }));
                    mesh.normals.push([0.0, 0.0, 1.0]);
                    mesh.uvs.push([u, v]);
                }
            }
            for y in 0..6 {
                for x in 0..6 {
                    let a = base + y * 7 + x;
                    mesh.indices.extend([a, a + 1, a + 8, a, a + 8, a + 7]);
                }
            }
        }
        assert!(ClusterLod::cook(mesh.source(), solid).unwrap().has_voxels());
    }

    #[test]
    fn errors_grow_from_refined_groups() {
        let mesh = canopy(4000, 5.0, 0.15);
        let lod = ClusterLod::cook(mesh.source(), solid).unwrap();
        for group in &lod.groups {
            let first = group.first_cluster as usize;
            for cluster in &lod.clusters[first..first + group.cluster_count as usize] {
                if cluster.refined != NO_GROUP {
                    let refined = lod.groups[cluster.refined as usize];
                    assert!(refined.simplified.error <= group.simplified.error);
                }
            }
        }
    }

    #[test]
    fn aggregates_become_valid_bricks_and_solids_do_not() {
        let mesh = canopy(4000, 5.0, 0.15);
        let lod = ClusterLod::cook(mesh.source(), solid).unwrap();
        assert!(lod.has_voxels());
        let surface = grid(32);
        assert!(
            !ClusterLod::cook(surface.source(), solid)
                .unwrap()
                .has_voxels()
        );
        for cluster in &lod.clusters {
            assert!(cluster.brick_count as usize <= MAX_CLUSTER_BRICKS);
            assert!(cluster.brick_count == 0 || cluster.triangle_count == 0);
        }
        let mut next = 0;
        for brick in &lod.bricks {
            assert_eq!(brick.first_voxel, next);
            next += brick.mask[0].count_ones() + brick.mask[1].count_ones();
        }
        assert_eq!(next as usize, lod.voxels.len());
        // Below a voxel cluster's cut, the source is still triangles.
        assert!(cut(&lod, 0.0).iter().all(|c| c.brick_count == 0));
        // A coarse cut must contain no triangles that thinning has reached.
        let coarse = cut(&lod, f32::MAX / 2.0);
        assert!(coarse.iter().all(|c| c.triangle_count == 0));
    }

    #[test]
    fn voxels_keep_the_density_that_simplification_loses() {
        let mesh = canopy(6000, 5.0, 0.2);
        let extent = 5.5;
        let truth = source_opacity(&mesh, 64, extent);
        let thin = ClusterLod::build(mesh.source(), false, solid).unwrap();
        let lod = ClusterLod::cook(mesh.source(), solid).unwrap();
        // Compare at the error where a voxel is about one opacity cell.
        let threshold = 2.0 * extent / 64.0;
        let triangles = area(&mesh, &thin, &cut(&thin, threshold));
        let source_area = area(&mesh, &thin, &cut(&thin, 0.0));
        let voxels = cut(&lod, threshold);
        assert!(voxels.iter().any(|c| c.brick_count > 0));
        let opacity = voxel_opacity(&lod, &voxels, 64, extent);
        println!(
            "truth {truth:.3}, voxels {opacity:.3}, simplified area {:.3}",
            triangles / source_area
        );
        assert!(triangles / source_area < 0.9, "simplification did not thin");
        assert!(
            (opacity - truth).abs() < 0.12,
            "voxels {opacity} against source {truth}"
        );
    }

    #[test]
    fn alpha_coverage_removes_voxels() {
        let mesh = canopy(4000, 5.0, 0.15);
        let lod = ClusterLod::cook(mesh.source(), solid).unwrap();
        let cut_out = ClusterLod::cook(mesh.source(), |t, b| SurfaceSample {
            coverage: 0.0,
            ..solid(t, b)
        })
        .unwrap();
        assert!(cut_out.voxels.len() * 10 < lod.voxels.len().max(10));
    }

    #[test]
    fn bytes_round_trip_and_bad_data_is_refused() {
        let mesh = canopy(2000, 5.0, 0.15);
        let vertices = mesh.positions.len();
        let lod = ClusterLod::cook(mesh.source(), solid).unwrap();
        assert!(lod.has_voxels());
        let bytes = lod.to_bytes();
        let read = ClusterLod::from_bytes(&bytes, vertices).unwrap();
        assert_eq!(read.to_bytes(), bytes);

        assert!(ClusterLod::from_bytes(&bytes[..bytes.len() - 1], vertices).is_none());
        assert!(ClusterLod::from_bytes(&[], vertices).is_none());
        assert!(
            ClusterLod::from_bytes(&bytes, 3).is_none(),
            "other vertices"
        );

        let mut stale = bytes.clone();
        stale[0] ^= 1;
        assert!(ClusterLod::from_bytes(&stale, vertices).is_none());

        // A cluster pointing past the index data must not reach a shader.
        let mut bad = lod.clone();
        bad.clusters[0].first_index = bad.indices.len() as u32;
        bad.clusters[0].triangle_count = 1;
        assert!(ClusterLod::from_bytes(&bad.to_bytes(), vertices).is_none());
        let mut bad = lod.clone();
        bad.nodes[0].first_child = u32::MAX;
        assert!(ClusterLod::from_bytes(&bad.to_bytes(), vertices).is_none());
    }

    #[test]
    fn rejects_invalid_input() {
        let mesh = grid(2);
        let source = mesh.source();
        let empty = Source {
            indices: &[],
            ..source
        };
        assert!(ClusterLod::cook(empty, solid).is_none());
        let partial = Source {
            indices: &[0, 1],
            ..source
        };
        assert!(ClusterLod::cook(partial, solid).is_none());
        let outside = Source {
            indices: &[0, 1, 99],
            ..source
        };
        assert!(ClusterLod::cook(outside, solid).is_none());
    }

    #[test]
    fn octahedral_round_trips_every_octant() {
        for n in [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.6, -0.48, 0.64],
            [-0.6, 0.48, -0.64],
        ] {
            let packed = octahedral(n);
            let unpack = |v: u32| v as f32 / 65535.0 * 2.0 - 1.0;
            let (x, y) = (unpack(packed & 0xffff), unpack(packed >> 16));
            let z = 1.0 - x.abs() - y.abs();
            let shift = (-z).max(0.0);
            let decoded = normalize([x - shift * x.signum(), y - shift * y.signum(), z]);
            assert!(dot(decoded, n) > 0.999, "{n:?} became {decoded:?}");
        }
    }
}
