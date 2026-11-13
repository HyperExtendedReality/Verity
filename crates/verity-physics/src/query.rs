//! Ray queries against what the simulation holds: rigid colliders, soft body
//! surfaces and static triangle meshes.
//!
//! [`RayScene`] follows a [`PhysicsWorld`] between steps and answers rays on
//! the CPU. Its trees are flat, so a GPU tracer can walk the same nodes; PXTGI
//! in verity-lux does. It also reports where things moved, which is exactly
//! where light changed: a renderer alone could only guess that from pixels.
//!
//! Static meshes answer rays only. They make no contacts, so bodies fall
//! through them; give level geometry colliders too.

use crate::{Collider, PhysicsWorld, SoftBody, SoftId};
use glam::{Affine3A, Mat3A, Quat, Vec3, Vec3A};
use std::{collections::HashMap, sync::Arc};
use verity_core::{EntityHandle, Transform, World};

/// Items a leaf holds at most, unless they cannot be told apart.
const LEAF_ITEMS: usize = 4;
const BINS: usize = 12;
/// Past this depth a tree splits at the median, which bounds it for a GPU's
/// fixed traversal stack however badly the items are placed.
const SAH_DEPTH: u32 = 24;
/// Refits between rebuilds: each one loosens the tree a little.
const REFITS_PER_BUILD: u32 = 15;

/// An axis-aligned box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    pub min: Vec3,
    pub max: Vec3,
}

impl Bounds {
    pub const EMPTY: Self = Self {
        min: Vec3::splat(f32::INFINITY),
        max: Vec3::splat(f32::NEG_INFINITY),
    };

    pub fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    fn with(self, point: Vec3) -> Self {
        Self {
            min: self.min.min(point),
            max: self.max.max(point),
        }
    }

    pub fn center(self) -> Vec3 {
        0.5 * (self.min + self.max)
    }

    /// Grown by `margin` on every side.
    pub fn expanded(self, margin: f32) -> Self {
        Self {
            min: self.min - margin,
            max: self.max + margin,
        }
    }

    /// Where a ray enters, if before `best`; `inverse` is 1 / its direction.
    fn entry(self, origin: Vec3, inverse: Vec3, best: f32) -> Option<f32> {
        let (t0, t1) = ((self.min - origin) * inverse, (self.max - origin) * inverse);
        let enter = t0.min(t1).max_element().max(0.0);
        let leave = t0.max(t1).min_element().min(best);
        (enter <= leave).then_some(enter)
    }
}

/// Index of the largest component.
fn widest(v: Vec3) -> usize {
    if v.x >= v.y && v.x >= v.z {
        0
    } else if v.y >= v.z {
        1
    } else {
        2
    }
}

/// `1 / d` with zero components kept finite, so a box test stays NaN-free.
fn safe_inverse(d: Vec3) -> Vec3 {
    let guard = |x: f32| if x.abs() < 1.0e-20 { 1.0e-20 } else { x };
    Vec3::new(guard(d.x), guard(d.y), guard(d.z)).recip()
}

/// A tree node as two rows: min with the right child (or first item of a
/// leaf) in w, and max with the item count (zero for an inner node) in w, both
/// as `i32` bits. The left child follows its parent.
pub type BvhNode = [[f32; 4]; 2];

/// A bounding volume tree over caller-indexed boxes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Bvh {
    nodes: Vec<BvhNode>,
    order: Vec<u32>,
}

impl Bvh {
    /// Built by binned surface area heuristic.
    pub fn build(bounds: &[Bounds]) -> Self {
        let mut order: Vec<u32> = (0..bounds.len() as u32).collect();
        let mut nodes = Vec::with_capacity(bounds.len());
        if !bounds.is_empty() {
            // SIMD boxes and centers, made once: the builder is mostly their math.
            let boxes: Vec<Box3> = bounds
                .iter()
                .map(|b| Box3 { min: b.min.into(), max: b.max.into() })
                .collect();
            let centers: Vec<Vec3A> = boxes.iter().map(|b| (b.min + b.max) * 0.5).collect();
            nodes = subtree(&boxes, &centers, &mut order, 0, 0);
        }
        Self { nodes, order }
    }

    pub fn nodes(&self) -> &[BvhNode] {
        &self.nodes
    }

    /// The items by leaf: a leaf's `first` and count index this.
    pub fn order(&self) -> &[u32] {
        &self.order
    }

    /// New boxes for the same items, keeping the shape of the tree.
    pub fn refit(&mut self, bounds: &[Bounds]) {
        // Children come after their parents.
        for i in (0..self.nodes.len()).rev() {
            let [lo, hi] = self.nodes[i];
            let (a, count) = (lo[3].to_bits() as usize, hi[3].to_bits() as usize);
            let fitted = if count == 0 {
                node_bounds(&self.nodes[i + 1]).union(node_bounds(&self.nodes[a]))
            } else {
                self.order[a..a + count]
                    .iter()
                    .fold(Bounds::EMPTY, |all, &item| all.union(bounds[item as usize]))
            };
            self.nodes[i] = node(fitted, a as u32, count as u32);
        }
    }

    /// Calls `test(slot in order, nearest so far)` for leaves the ray reaches,
    /// near child first, and returns the nearest distance it reported.
    fn ray(
        &self,
        origin: Vec3,
        direction: Vec3,
        mut best: f32,
        mut test: impl FnMut(usize, f32) -> Option<f32>,
    ) -> f32 {
        let inverse = safe_inverse(direction);
        let entry = |at: usize, best| node_bounds(&self.nodes[at]).entry(origin, inverse, best);
        if self.nodes.is_empty() || entry(0, best).is_none() {
            return best;
        }
        let mut stack = vec![(0, 0.0)];
        while let Some((at, enter)) = stack.pop() {
            if enter > best {
                continue;
            }
            let [lo, hi] = self.nodes[at];
            let (a, count) = (lo[3].to_bits() as usize, hi[3].to_bits() as usize);
            if count > 0 {
                for slot in a..a + count {
                    if let Some(t) = test(slot, best) {
                        best = t;
                    }
                }
                continue;
            }
            let (left, right) = (at + 1, a);
            let (near, far) = match (entry(left, best), entry(right, best)) {
                (Some(l), Some(r)) if l <= r => (Some((left, l)), Some((right, r))),
                (Some(l), Some(r)) => (Some((right, r)), Some((left, l))),
                (Some(l), None) => (Some((left, l)), None),
                (None, r) => (r.map(|r| (right, r)), None),
            };
            stack.extend(far);
            stack.extend(near);
        }
        best
    }
}

fn node(bounds: Bounds, a: u32, count: u32) -> BvhNode {
    let (lo, hi) = (bounds.min, bounds.max);
    [
        [lo.x, lo.y, lo.z, f32::from_bits(a)],
        [hi.x, hi.y, hi.z, f32::from_bits(count)],
    ]
}

fn node_bounds(node: &BvhNode) -> Bounds {
    let [lo, hi] = node;
    Bounds {
        min: Vec3::new(lo[0], lo[1], lo[2]),
        max: Vec3::new(hi[0], hi[1], hi[2]),
    }
}

/// A box in SIMD lanes, for building.
#[derive(Clone, Copy)]
struct Box3 {
    min: Vec3A,
    max: Vec3A,
}

impl Box3 {
    const EMPTY: Self = Self {
        min: Vec3A::splat(f32::INFINITY),
        max: Vec3A::splat(f32::NEG_INFINITY),
    };

    fn union(self, other: Self) -> Self {
        Self { min: self.min.min(other.min), max: self.max.max(other.max) }
    }

    fn with(self, p: Vec3A) -> Self {
        Self { min: self.min.min(p), max: self.max.max(p) }
    }

    /// Half the surface area, which is all the heuristic compares.
    fn area(self) -> f32 {
        let d = (self.max - self.min).max(Vec3A::ZERO);
        d.x * d.y + d.y * d.z + d.z * d.x
    }

    fn bounds(self) -> Bounds {
        Bounds { min: self.min.into(), max: self.max.into() }
    }
}

/// Items past which the two halves of a node build on separate threads.
const PARALLEL_ITEMS: usize = 4096;
/// Items below which a node bins only its widest axis: small nodes are most of
/// a tree, and searching all three costs them more than it finds.
const ALL_AXES_ITEMS: usize = 64;

/// A tree over `items` whose right-child links count from its own root, for
/// [`subdivide`] to splice in.
fn subtree(boxes: &[Box3], centers: &[Vec3A], items: &mut [u32], first: u32, depth: u32) -> Vec<BvhNode> {
    let mut nodes = Vec::with_capacity(items.len());
    subdivide(boxes, centers, items, first, depth, &mut nodes);
    nodes
}

/// Append a [`subtree`], moving its right-child links to where it lands.
fn splice(nodes: &mut Vec<BvhNode>, subtree: Vec<BvhNode>) {
    let base = nodes.len() as u32;
    nodes.extend(subtree.into_iter().map(|mut n| {
        if n[1][3].to_bits() == 0 {
            n[0][3] = f32::from_bits(n[0][3].to_bits() + base);
        }
        n
    }));
}

fn subdivide(
    boxes: &[Box3],
    centers: &[Vec3A],
    items: &mut [u32],
    first: u32,
    depth: u32,
    nodes: &mut Vec<BvhNode>,
) {
    let at = nodes.len();
    let all = items.iter().fold(Box3::EMPTY, |all, &i| all.union(boxes[i as usize])).bounds();
    nodes.push(node(all, first, items.len() as u32));
    if items.len() <= LEAF_ITEMS {
        return;
    }
    let middle = (depth < SAH_DEPTH)
        .then(|| sah_split(boxes, centers, items))
        .flatten()
        .unwrap_or_else(|| median_split(centers, items));
    let parallel = items.len() >= PARALLEL_ITEMS;
    let (left, right) = items.split_at_mut(middle);
    let right_first = first + middle as u32;
    let right_at = if parallel {
        // The same tree as built in order, whichever half finishes first.
        let (l, r) = rayon::join(
            || subtree(boxes, centers, left, first, depth + 1),
            || subtree(boxes, centers, right, right_first, depth + 1),
        );
        splice(nodes, l);
        let right_at = nodes.len() as u32;
        splice(nodes, r);
        right_at
    } else {
        subdivide(boxes, centers, left, first, depth + 1, nodes);
        let right_at = nodes.len() as u32;
        subdivide(boxes, centers, right, right_first, depth + 1, nodes);
        right_at
    };
    nodes[at] = node(all, right_at, 0);
}

/// Partition `items` at the cheapest of the binned planes along any axis of
/// their centers; `None` when the centers coincide or nothing beats a leaf of a
/// size a leaf may have. All three axes, not only the widest: a long thin
/// cluster often splits best across its length, and on Sponza the trees took
/// PXTGI's software rays 28% less time. One pass bins all three.
fn sah_split(boxes: &[Box3], centers: &[Vec3A], items: &mut [u32]) -> Option<usize> {
    let span = items.iter().fold(Box3::EMPTY, |all, &i| all.with(centers[i as usize]));
    let extent = span.max - span.min;
    // No more bins than items: small nodes, most of a tree, sweep fewer planes.
    let count = items.len().clamp(4, BINS);
    let scale = Vec3A::select(extent.cmpgt(Vec3A::ZERO), count as f32 / extent, Vec3A::ZERO);
    let top = Vec3A::splat((count - 1) as f32);
    let bin = |c: Vec3A| ((c - span.min) * scale).min(top).as_uvec3();
    let widest = widest(extent.into());
    let searched = |axis: usize| extent[axis] > 0.0 && (items.len() >= ALL_AXES_ITEMS || axis == widest);
    let mut bins = [[(Box3::EMPTY, 0u32); BINS]; 3];
    for &i in items.iter() {
        let at = bin(centers[i as usize]);
        let item = boxes[i as usize];
        for (axis, bins) in bins.iter_mut().enumerate() {
            if !searched(axis) {
                continue;
            }
            let b = &mut bins[at[axis] as usize];
            *b = (b.0.union(item), b.1 + 1);
        }
    }
    let mut best: Option<(usize, u32)> = None;
    let mut cost = f32::INFINITY;
    for (axis, bins) in bins.iter().enumerate() {
        if !searched(axis) {
            continue;
        }
        let mut right_area = [0.0; BINS];
        let (mut area, mut right_count) = (Box3::EMPTY, 0);
        for plane in (1..count).rev() {
            area = area.union(bins[plane].0);
            right_count += bins[plane].1;
            right_area[plane] = area.area() * right_count as f32;
        }
        let (mut left, mut left_count) = (Box3::EMPTY, 0);
        for plane in 1..count {
            left = left.union(bins[plane - 1].0);
            left_count += bins[plane - 1].1;
            let total = left.area() * left_count as f32 + right_area[plane];
            if left_count > 0 && (left_count as usize) < items.len() && total < cost {
                (best, cost) = (Some((axis, plane as u32)), total);
            }
        }
    }
    let (axis, plane) = best?;
    let (mut low, mut high) = (0, items.len());
    while low < high {
        if bin(centers[items[low] as usize])[axis] < plane {
            low += 1;
        } else {
            high -= 1;
            items.swap(low, high);
        }
    }
    Some(low)
}

fn median_split(centers: &[Vec3A], items: &mut [u32]) -> usize {
    let spread = items.iter().fold(Box3::EMPTY, |all, &i| all.with(centers[i as usize]));
    let axis = widest((spread.max - spread.min).into());
    let middle = items.len() / 2;
    items.select_nth_unstable_by(middle, |&a, &b| {
        let center = |i: u32| centers[i as usize][axis];
        center(a).total_cmp(&center(b))
    });
    middle
}

/// Triangles under a tree, stored in leaf order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TriMesh {
    tree: Bvh,
    triangles: Vec<[Vec3; 3]>,
}

impl TriMesh {
    pub fn new(positions: &[Vec3], indices: &[[u32; 3]]) -> Self {
        let corners = |t: &[u32; 3]| t.map(|i| positions[i as usize]);
        let bounds: Vec<_> = indices.iter().map(|t| corner_bounds(corners(t))).collect();
        let tree = Bvh::build(&bounds);
        let triangles = tree
            .order()
            .iter()
            .map(|&i| corners(&indices[i as usize]))
            .collect();
        Self { tree, triangles }
    }

    pub fn nodes(&self) -> &[BvhNode] {
        self.tree.nodes()
    }

    /// In leaf order: a leaf's `first` and count index this.
    pub fn triangles(&self) -> &[[Vec3; 3]] {
        &self.triangles
    }

    pub fn bounds(&self) -> Bounds {
        self.tree.nodes().first().map_or(Bounds::EMPTY, node_bounds)
    }

    /// The nearest hit on either side, and the normal facing the ray.
    pub fn ray(&self, origin: Vec3, direction: Vec3, max: f32) -> Option<(f32, Vec3)> {
        let mut normal = None;
        let t = self.tree.ray(origin, direction, max, |slot, best| {
            let [a, b, c] = self.triangles[slot];
            let (e1, e2) = (b - a, c - a);
            let p = direction.cross(e2);
            let inverse = 1.0 / e1.dot(p);
            if !inverse.is_finite() {
                return None;
            }
            let s = origin - a;
            let u = s.dot(p) * inverse;
            let q = s.cross(e1);
            let v = direction.dot(q) * inverse;
            let t = e2.dot(q) * inverse;
            let inside = u >= 0.0 && v >= 0.0 && u + v <= 1.0;
            (inside && t > 0.0 && t < best).then(|| {
                normal = Some(e1.cross(e2));
                t
            })
        });
        normal.map(|n: Vec3| {
            let n = n.normalize_or_zero();
            (t, if n.dot(direction) > 0.0 { -n } else { n })
        })
    }

    /// New corners for the same triangles; `false` if none moved.
    fn update(&mut self, positions: &[Vec3], indices: &[[u32; 3]]) -> bool {
        let order = self.tree.order();
        let changed = order
            .iter()
            .zip(&self.triangles)
            .any(|(&i, old)| indices[i as usize].map(|v| positions[v as usize]) != *old);
        if changed {
            for (&i, old) in order.iter().zip(&mut self.triangles) {
                *old = indices[i as usize].map(|v| positions[v as usize]);
            }
            let bounds: Vec<_> = indices
                .iter()
                .map(|t| corner_bounds(t.map(|i| positions[i as usize])))
                .collect();
            self.tree.refit(&bounds);
        }
        changed
    }
}

fn corner_bounds(corners: [Vec3; 3]) -> Bounds {
    corners.into_iter().fold(Bounds::EMPTY, Bounds::with)
}

/// Where `origin + direction t` enters a Z-aligned capsule at the origin,
/// and the outward normal there: a negative `t` when `origin` is inside.
pub(crate) fn capsule_ray(
    origin: Vec3,
    direction: Vec3,
    radius: f32,
    half_height: f32,
) -> Option<(f32, Vec3)> {
    let sphere = |center: Vec3| {
        let o = origin - center;
        let (a, b, c) = (
            direction.length_squared(),
            o.dot(direction),
            o.length_squared(),
        );
        let h = b * b - a * (c - radius * radius);
        (h >= 0.0 && a > 0.0).then(|| {
            let t = (-b - h.sqrt()) / a;
            (t, (o + direction * t) / radius)
        })
    };
    // The side, where it lies between the caps.
    let (o, d) = (origin.truncate(), direction.truncate());
    let a = d.length_squared();
    if a > 1.0e-12 {
        let b = o.dot(d);
        let h = b * b - a * (o.length_squared() - radius * radius);
        if h < 0.0 {
            return None;
        }
        let t = (-b - h.sqrt()) / a;
        let p = origin + direction * t;
        if p.z.abs() <= half_height {
            return Some((t, Vec3::new(p.x, p.y, 0.0) / radius));
        }
    }
    let caps = [half_height, -half_height].map(|z| sphere(Vec3::new(0.0, 0.0, z)));
    caps.into_iter()
        .flatten()
        .min_by(|x, y| x.0.total_cmp(&y.0))
}

impl Collider {
    /// The nearest entry of a ray from outside, in the collider's own frame,
    /// and the outward normal there. `direction` need not be unit length: `t`
    /// is in units of it.
    pub fn ray(self, origin: Vec3, direction: Vec3, max: f32) -> Option<(f32, Vec3)> {
        match self {
            Self::Cuboid { half_extents } => {
                let inverse = safe_inverse(direction);
                let (t0, t1) = (
                    (-half_extents - origin) * inverse,
                    (half_extents - origin) * inverse,
                );
                let near = t0.min(t1);
                let t = near.max_element();
                let leave = t0.max(t1).min_element();
                if t > leave || t <= 0.0 || t >= max {
                    return None;
                }
                let axis = widest(near);
                let mut normal = Vec3::ZERO;
                normal[axis] = -direction[axis].signum();
                Some((t, normal))
            }
            Self::Sphere { radius } => {
                let a = direction.length_squared();
                let b = origin.dot(direction);
                let c = origin.length_squared() - radius * radius;
                let h = b * b - a * c;
                if h < 0.0 {
                    return None;
                }
                let t = (-b - h.sqrt()) / a;
                (t > 0.0 && t < max).then(|| (t, (origin + direction * t) / radius))
            }
            Self::Capsule {
                radius,
                half_height,
            } => capsule_ray(origin, direction, radius, half_height)
                .filter(|&(t, _)| t > 0.0 && t < max),
            // Rays see ground as the mesh drawn for it, not through its body.
            Self::Ground => None,
        }
    }

    fn world_bounds(self, pose: &Affine3A) -> Bounds {
        let center = Vec3::from(pose.translation);
        let reach = match self {
            Self::Sphere { radius } => Vec3::splat(radius),
            Self::Capsule {
                radius,
                half_height,
            } => Vec3::from(pose.matrix3.z_axis.abs()) * half_height + radius,
            Self::Ground => Vec3::ZERO,
            Self::Cuboid { half_extents } => {
                let m = Mat3A::from_cols(
                    pose.matrix3.x_axis.abs(),
                    pose.matrix3.y_axis.abs(),
                    pose.matrix3.z_axis.abs(),
                );
                Vec3::from(m * Vec3A::from(half_extents))
            }
        };
        Bounds {
            min: center - reach,
            max: center + reach,
        }
    }
}

/// What a [`RayScene`] item shows to rays.
#[derive(Debug, Clone, PartialEq)]
pub enum RayShape {
    Collider(Collider),
    Mesh(Arc<TriMesh>),
}

/// Where a [`RayScene`] item comes from. A static item's number is the caller's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RaySource {
    Body(EntityHandle),
    Soft(SoftId),
    Static(u32),
}

/// One thing rays can hit.
#[derive(Debug, Clone, PartialEq)]
pub struct RayItem {
    pub shape: RayShape,
    pub source: RaySource,
    /// Its frame's inverse pose; rays are carried into it for testing.
    pub local_from_world: Affine3A,
    pub bounds: Bounds,
}

impl RayItem {
    fn new(shape: RayShape, source: RaySource, pose: Affine3A) -> Self {
        let bounds = match &shape {
            RayShape::Collider(collider) => collider.world_bounds(&pose),
            RayShape::Mesh(mesh) => {
                let Bounds { min, max } = mesh.bounds();
                (0..8).fold(Bounds::EMPTY, |all, i| {
                    let corner = Vec3::select(
                        glam::BVec3::new(i & 1 != 0, i & 2 != 0, i & 4 != 0),
                        max,
                        min,
                    );
                    all.with(pose.transform_point3(corner))
                })
            }
        };
        Self {
            shape,
            source,
            local_from_world: pose.inverse(),
            bounds,
        }
    }

    fn ray(&self, origin: Vec3, direction: Vec3, max: f32) -> Option<(f32, Vec3)> {
        let m = &self.local_from_world;
        let (o, d) = (m.transform_point3(origin), m.transform_vector3(direction));
        let (t, n) = match &self.shape {
            RayShape::Collider(collider) => collider.ray(o, d, max)?,
            RayShape::Mesh(mesh) => mesh.ray(o, d, max)?,
        };
        // Normals transform by the inverse transpose of the pose.
        let normal = Mat3A::from(m.matrix3).transpose() * Vec3A::from(n);
        Some((t, Vec3::from(normal).normalize_or_zero()))
    }
}

/// The nearest thing along a ray.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    /// Distance along the ray, in units of its direction.
    pub t: f32,
    pub normal: Vec3,
    /// Index into [`RayScene::items`].
    pub item: usize,
}

/// What [`RayScene::sync`] found.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RaySync {
    /// Anything moved, appeared or went: items or the tree differ from before.
    pub changed: bool,
    /// Items were added or removed, so their indices may differ.
    pub restructured: bool,
}

/// Everything rays can hit, under one tree. Sync it after each physics step:
/// resting scenes cost one comparison per body and change nothing.
#[derive(Debug, Default)]
pub struct RayScene {
    /// Count entities without a body as static cuboids of their transform's
    /// scale, which is how the renderer draws them. Entities of no size stay out.
    pub entity_boxes: bool,
    items: Vec<RayItem>,
    tree: Bvh,
    index: HashMap<RaySource, usize>,
    moved: Vec<Bounds>,
    refits: u32,
}

fn pose_of(transform: &Transform) -> Affine3A {
    Affine3A::from_rotation_translation(
        Quat::from_array(transform.rotation).normalize(),
        Vec3::from_array(transform.translation),
    )
}

/// The cuboid a body-less entity is drawn as, unless it has no size.
fn entity_box(transform: &Transform) -> Option<Collider> {
    let half_extents = Vec3::from_array(transform.scale).abs() * 0.5;
    (half_extents.min_element() > 0.0).then_some(Collider::Cuboid { half_extents })
}

/// A soft body's surface, when it has one to show.
fn soft_surface(body: &SoftBody) -> Option<(&[Vec3], &[[u32; 3]])> {
    (!body.triangles().is_empty()).then(|| (body.positions(), body.triangles()))
}

impl RayScene {
    pub fn items(&self) -> &[RayItem] {
        &self.items
    }

    /// The tree over [`Self::items`].
    pub fn tree(&self) -> &Bvh {
        &self.tree
    }

    /// Bounds of what changed in the last sync, old and new place together.
    pub fn moved(&self) -> &[Bounds] {
        &self.moved
    }

    /// Add meshes that stay put, with their poses and numbers. Meshes may be
    /// shared between items. Numbers must be unique; they come back in
    /// [`RaySource::Static`].
    pub fn add_statics(&mut self, meshes: impl IntoIterator<Item = (Arc<TriMesh>, Affine3A, u32)>) {
        for (mesh, pose, number) in meshes {
            let source = RaySource::Static(number);
            self.index.insert(source, self.items.len());
            self.items
                .push(RayItem::new(RayShape::Mesh(mesh), source, pose));
        }
        self.rebuild();
    }

    /// Add, replace or, with `None`, remove static items by number, under one
    /// tree rebuild. [`Self::moved`] then holds what changed.
    pub fn update_statics(
        &mut self,
        changes: impl IntoIterator<Item = (u32, Option<(Arc<TriMesh>, Affine3A)>)>,
    ) {
        self.moved.clear();
        for (number, change) in changes {
            let source = RaySource::Static(number);
            let item = change.map(|(mesh, pose)| RayItem::new(RayShape::Mesh(mesh), source, pose));
            match (self.index.get(&source).copied(), item) {
                (Some(i), Some(item)) => {
                    self.moved.push(self.items[i].bounds.union(item.bounds));
                    self.items[i] = item;
                }
                (None, Some(item)) => {
                    self.moved.push(item.bounds);
                    self.index.insert(source, self.items.len());
                    self.items.push(item);
                }
                (Some(i), None) => {
                    self.moved.push(self.items.swap_remove(i).bounds);
                    self.index.remove(&source);
                    if let Some(swapped) = self.items.get(i) {
                        self.index.insert(swapped.source, i);
                    }
                }
                (None, None) => {}
            }
        }
        self.rebuild();
    }

    fn rebuild(&mut self) {
        let bounds: Vec<_> = self.items.iter().map(|item| item.bounds).collect();
        self.tree = Bvh::build(&bounds);
        self.refits = 0;
    }

    /// Follow the physics world: bodies and soft bodies that appeared, went or
    /// moved. Sleeping bodies do not move, so they cost nothing downstream.
    pub fn sync(&mut self, physics: &PhysicsWorld, world: &World) -> RaySync {
        self.moved.clear();
        let mut restructured = false;
        let boxes = self.entity_boxes;
        let gone = |item: &RayItem| match item.source {
            RaySource::Body(entity) => {
                let boxed = |e| {
                    world
                        .get(e)
                        .is_some_and(|e| entity_box(e.transform).is_some())
                };
                !world.contains(entity) || !(physics.contains(entity) || (boxes && boxed(entity)))
            }
            RaySource::Soft(id) => physics
                .soft_body(id)
                .is_none_or(|b| soft_surface(b).is_none()),
            RaySource::Static(_) => false,
        };
        if self.items.iter().any(gone) {
            self.moved
                .extend(self.items.iter().filter(|i| gone(i)).map(|i| i.bounds));
            self.items.retain(|item| !gone(item));
            restructured = true;
        }
        if restructured {
            self.reindex();
        }

        let mut added = false;
        for (entity, body) in physics.iter() {
            let Some(found) = world.get(entity) else {
                continue;
            };
            if body.collider != Collider::Ground {
                added |= self.place(entity, body.collider, found.transform);
            }
        }
        if boxes {
            for found in world.iter() {
                if let (false, Some(collider)) =
                    (physics.contains(found.handle), entity_box(found.transform))
                {
                    added |= self.place(found.handle, collider, found.transform);
                }
            }
        }
        for (id, body) in physics.soft_bodies() {
            let Some((positions, triangles)) = soft_surface(body) else {
                continue;
            };
            let source = RaySource::Soft(id);
            let fresh = |this: &mut Vec<RayItem>, at: Option<usize>| {
                let mesh = Arc::new(TriMesh::new(positions, triangles));
                let item = RayItem::new(RayShape::Mesh(mesh), source, Affine3A::IDENTITY);
                let bounds = item.bounds;
                match at {
                    Some(i) => this[i] = item,
                    None => this.push(item),
                }
                bounds
            };
            let Some(&i) = self.index.get(&source) else {
                self.index.insert(source, self.items.len());
                self.moved.push(fresh(&mut self.items, None));
                added = true;
                continue;
            };
            let item = &mut self.items[i];
            let RayShape::Mesh(mesh) = &mut item.shape else {
                continue;
            };
            let old = item.bounds;
            if mesh.triangles.len() != triangles.len() {
                // Torn: new topology, so the same item gets a new tree.
                let new = fresh(&mut self.items, Some(i));
                self.moved.push(old.union(new));
            } else if Arc::make_mut(mesh).update(positions, triangles) {
                item.bounds = mesh.bounds();
                self.moved.push(old.union(item.bounds));
            }
        }

        restructured |= added;
        let changed = restructured || !self.moved.is_empty();
        if restructured || (changed && self.refits >= REFITS_PER_BUILD) {
            self.rebuild();
        } else if changed {
            let bounds: Vec<_> = self.items.iter().map(|item| item.bounds).collect();
            self.tree.refit(&bounds);
            self.refits += 1;
        }
        RaySync {
            changed,
            restructured,
        }
    }

    /// Add or update an entity's item; `true` when it is new.
    fn place(&mut self, entity: EntityHandle, collider: Collider, transform: &Transform) -> bool {
        let source = RaySource::Body(entity);
        let shape = RayShape::Collider(collider);
        let pose = pose_of(transform);
        if let Some(&i) = self.index.get(&source) {
            let item = &mut self.items[i];
            if item.shape != shape || item.local_from_world != pose.inverse() {
                let old = item.bounds;
                *item = RayItem::new(shape, source, pose);
                self.moved.push(old.union(item.bounds));
            }
            return false;
        }
        self.index.insert(source, self.items.len());
        let item = RayItem::new(shape, source, pose);
        self.moved.push(item.bounds);
        self.items.push(item);
        true
    }

    fn reindex(&mut self) {
        self.index = self
            .items
            .iter()
            .enumerate()
            .map(|(i, item)| (item.source, i))
            .collect();
    }

    /// The nearest hit along a ray within `max` units of `direction`.
    pub fn ray(&self, origin: Vec3, direction: Vec3, max: f32) -> Option<RayHit> {
        let mut hit = None;
        self.tree.ray(origin, direction, max, |slot, best| {
            let item = self.tree.order()[slot] as usize;
            let (t, normal) = self.items[item].ray(origin, direction, best)?;
            hit = Some(RayHit { t, normal, item });
            Some(t)
        });
        hit
    }
}
