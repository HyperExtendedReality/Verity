//! SAH spatial splits for immutable meshes. Leaves index source triangles through a reference array.

use glam::{DVec3, Vec3};
use std::cmp::Ordering::{self, Equal, Greater, Less};

use super::{BvhNode, bvh};

const MAX_LEAF: usize = 8;
const MAX_DEPTH: u32 = 48;
const TRAVERSAL_COST: f64 = 3.0;
const OBJECT_BINS: usize = 16;
const SPATIAL_BINS: usize = 8;
const PLANES: usize = SPATIAL_BINS - 1;
// Avoid scoring clips for negligible overlap, or taking a marginal gain in exchange for indirection.
const MIN_OVERLAP: f64 = 0.001;
const PROFIT_RATIO: f64 = 0.98;
const DUPLICATION_DIVISOR: usize = 4;

/// A binary BLAS with adjacent siblings and indirect, mesh-relative leaf spans.
#[derive(Debug)]
pub(super) struct SpatialTree {
    pub(super) nodes: Vec<BvhNode>,
    pub(super) references: Vec<u32>,
}

/// Builds a spatial BLAS only when duplication improves the object tree's estimated ray cost.
pub(super) fn build(corners: &[[Vec3; 3]]) -> Option<SpatialTree> {
    let count = u32::try_from(corners.len()).ok()?;
    let limit = corners.len().saturating_add(corners.len() / DUPLICATION_DIVISOR)
        .min(u32::MAX as usize);
    let duplicates_left = limit - corners.len();
    if duplicates_left == 0 {
        return None;
    }
    let mut pending = Vec::with_capacity(corners.len());
    let mut bounds = Bounds::EMPTY;
    for (source, triangle) in (0..count).zip(corners) {
        if triangle.iter().any(|p| !p.is_finite()) {
            return None;
        }
        let fragment = Bounds::triangle(triangle);
        bounds.extend(fragment);
        pending.push(Reference { source, bounds: fragment });
    }
    if bounds.area() == 0.0 {
        return None;
    }

    let mut builder = Builder {
        corners,
        pending,
        nodes: vec![BvhNode::default()],
        references: Vec::with_capacity(corners.len()),
        duplicates_left,
    };
    builder.descend(0, 0, 0, bounds)?;
    if builder.duplicates_left == duplicates_left {
        return None;
    }

    // A better immediate partition can still make later partitions worse.
    // Compare the full trees only once the spatial build has actually used clipping.
    let mut sources: Vec<u32> = (0..count).collect();
    let object = bvh::build_mesh(&mut sources, |&source| {
        let bounds = Bounds::triangle(&corners[source as usize]);
        (bounds.min, bounds.max)
    });
    (tree_cost(&builder.nodes) < tree_cost(&object) * PROFIT_RATIO).then_some(SpatialTree {
        nodes: builder.nodes,
        references: builder.references,
    })
}

fn tree_cost(nodes: &[BvhNode]) -> f64 {
    nodes.iter().map(|node| {
        let bounds = Bounds { min: node.min, max: node.max };
        let cost = if node.count == 0 { TRAVERSAL_COST } else { f64::from(node.count) };
        bounds.area() * cost
    }).sum()
}

#[derive(Clone, Copy)]
struct Bounds {
    min: Vec3,
    max: Vec3,
}

impl Bounds {
    const EMPTY: Self = Self { min: Vec3::splat(f32::INFINITY), max: Vec3::splat(f32::NEG_INFINITY) };

    fn triangle(&[a, b, c]: &[Vec3; 3]) -> Self {
        Self { min: a.min(b).min(c), max: a.max(b).max(c) }
    }

    fn extend(&mut self, other: Self) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    fn area(self) -> f64 {
        let extent = self.max.as_dvec3() - self.min.as_dvec3();
        extent.x * extent.y + extent.y * extent.z + extent.z * extent.x
    }

    fn intersection(self, other: Self) -> Self {
        Self { min: self.min.max(other.min), max: self.max.min(other.max) }
    }

    fn overlaps(self, other: Self, min_area: f64) -> bool {
        let overlap = self.intersection(other);
        overlap.min.cmple(overlap.max).all() && overlap.area() > min_area
    }

    fn center(self) -> DVec3 {
        (self.min.as_dvec3() + self.max.as_dvec3()) * 0.5
    }

    fn node(self, first: u32, count: u32) -> BvhNode {
        BvhNode { min: self.min, max: self.max, first, count }
    }
}

#[derive(Clone, Copy)]
struct Reference {
    source: u32,
    bounds: Bounds,
}

#[derive(Clone, Copy)]
struct Summary {
    bounds: Bounds,
    count: usize,
}

impl Summary {
    const EMPTY: Self = Self { bounds: Bounds::EMPTY, count: 0 };

    fn add(&mut self, bounds: Bounds) {
        self.bounds.extend(bounds);
        self.count += 1;
    }

    fn merge(&mut self, other: Self) {
        self.bounds.extend(other.bounds);
        self.count += other.count;
    }

    fn cost(self) -> f64 {
        if self.count == 0 { 0.0 } else { self.bounds.area() * self.count as f64 }
    }
}

#[derive(Clone, Copy)]
struct Grid {
    axis: usize,
    origin: f64,
    scale: f64,
}

impl Grid {
    fn bin(self, bounds: Bounds) -> usize {
        (((bounds.center()[self.axis] - self.origin) * self.scale) as usize).min(OBJECT_BINS - 1)
    }
}

#[derive(Clone, Copy)]
enum Partition {
    Object { grid: Grid, boundary: usize },
    Median,
    Spatial { axis: usize, plane: f32 },
}

#[derive(Clone, Copy)]
struct Split {
    partition: Partition,
    left: Summary,
    right: Summary,
}

impl Split {
    fn cost(self) -> f64 {
        self.left.cost() + self.right.cost()
    }
}

struct Builder<'mesh> {
    corners: &'mesh [[Vec3; 3]],
    // Each subtree owns the suffix it processes. Descending into the right child first lets
    // it remove its suffix before the left child runs, with no per-node reference allocations.
    pending: Vec<Reference>,
    nodes: Vec<BvhNode>,
    references: Vec<u32>,
    duplicates_left: usize,
}

impl Builder<'_> {
    fn descend(&mut self, at: usize, start: usize, depth: u32, bounds: Bounds) -> Option<()> {
        let count = self.pending.len() - start;
        let split = if depth < MAX_DEPTH && count > TRAVERSAL_COST as usize {
            self.choose_split(start, depth, bounds)
        } else {
            None
        };
        let Some(split) = split else {
            if count > MAX_LEAF {
                return None;
            }
            let first = u32::try_from(self.references.len()).ok()?;
            self.nodes[at] = bounds.node(first, u32::try_from(count).ok()?);
            self.references.extend(self.pending.drain(start..).map(|reference| reference.source));
            return Some(());
        };

        let mid = start + split.left.count;
        match split.partition {
            Partition::Object { grid, boundary } => {
                partition(&mut self.pending[start..], |r| grid.bin(r.bounds) < boundary);
            }
            Partition::Median => {}
            Partition::Spatial { axis, plane } => {
                let extra = split.left.count + split.right.count - count;
                if self.references.capacity() == self.corners.len() {
                    // Reserve the entire bounded growth at the first accepted split, not once per node.
                    self.references.reserve_exact(self.corners.len() + self.duplicates_left - self.references.len());
                    self.pending.reserve_exact(self.corners.len() + self.duplicates_left - self.pending.len());
                }
                self.duplicates_left -= extra;
                for i in start..self.pending.len() {
                    let reference = &mut self.pending[i];
                    if reference.bounds.min[axis] < plane && plane < reference.bounds.max[axis] {
                        let triangle = &self.corners[reference.source as usize];
                        let [left, right] = Polygon::bounded(triangle, reference.bounds)?
                            .split(reference.bounds, axis, plane)?;
                        reference.bounds = left;
                        let source = reference.source;
                        self.pending.push(Reference { source, bounds: right });
                    }
                }
                partition(&mut self.pending[start..], |r| r.bounds.max[axis] <= plane);
            }
        }

        let children = self.nodes.len();
        u32::try_from(children.checked_add(1)?).ok()?;
        self.nodes[at] = bounds.node(children as u32, 0);
        self.nodes.extend([BvhNode::default(); 2]);
        self.descend(children + 1, mid, depth + 1, split.right.bounds)?;
        self.descend(children, start, depth + 1, split.left.bounds)
    }

    fn choose_split(&mut self, start: usize, depth: u32, bounds: Bounds) -> Option<Split> {
        let references = &mut self.pending[start..];
        let count = references.len();
        let fits_leaf = count <= MAX_LEAF;
        let area = bounds.area();
        // Don't give either child a span too large for the levels it has left.
        let levels = MAX_DEPTH - depth - 1;
        let child_limit = MAX_LEAF.saturating_mul(1usize.checked_shl(levels).unwrap_or(usize::MAX));
        let mut best = object_split(references, child_limit);
        if best.is_none() && !fits_leaf {
            let mid = count / 2;
            let extent = bounds.max.as_dvec3() - bounds.min.as_dvec3();
            let axis = if extent.x >= extent.y && extent.x >= extent.z {
                0
            } else if extent.y >= extent.z {
                1
            } else {
                2
            };
            references.select_nth_unstable_by(mid, |a, b| {
                a.bounds.center()[axis].total_cmp(&b.bounds.center()[axis])
            });
            let summarize = |refs: &[Reference]| {
                let mut summary = Summary::EMPTY;
                for reference in refs {
                    summary.add(reference.bounds);
                }
                summary
            };
            let (left, right) = references.split_at(mid);
            best = Some(Split { partition: Partition::Median, left: summarize(left), right: summarize(right) });
        }

        let leaf_cost = area * (count as f64 - TRAVERSAL_COST);
        if self.duplicates_left != 0 && area > 0.0 && best.is_none_or(|split| {
            split.left.bounds.overlaps(split.right.bounds, area * MIN_OVERLAP)
        }) {
            let cost_limit = best.map_or(f64::INFINITY, Split::cost)
                .min(if fits_leaf { leaf_cost } else { f64::INFINITY });
            if let Some(split) = spatial_split(self.corners, references, bounds, child_limit,
                self.duplicates_left, cost_limit * PROFIT_RATIO)
            {
                best = Some(split);
            }
        }
        best.filter(|split| !fits_leaf || split.cost() < leaf_cost)
    }
}

fn partition(references: &mut [Reference], goes_left: impl Fn(&Reference) -> bool) {
    let mut left = 0;
    for i in 0..references.len() {
        if goes_left(&references[i]) {
            if i != left {
                references.swap(left, i);
            }
            left += 1;
        }
    }
}

fn object_split(references: &[Reference], child_limit: usize) -> Option<Split> {
    let mut lo = DVec3::splat(f64::INFINITY);
    let mut hi = DVec3::splat(f64::NEG_INFINITY);
    for reference in references {
        let center = reference.bounds.center();
        lo = lo.min(center);
        hi = hi.max(center);
    }
    let mut best = None::<(f64, usize, Split)>;
    for axis in 0..3 {
        let span = hi[axis] - lo[axis];
        if span == 0.0 {
            continue;
        }
        let grid = Grid { axis, origin: lo[axis], scale: OBJECT_BINS as f64 / span };
        let mut bins = [Summary::EMPTY; OBJECT_BINS];
        for reference in references {
            bins[grid.bin(reference.bounds)].add(reference.bounds);
        }
        let mut suffix = [Summary::EMPTY; OBJECT_BINS];
        let mut right = Summary::EMPTY;
        for i in (1..OBJECT_BINS).rev() {
            right.merge(bins[i]);
            suffix[i] = right;
        }
        let mut left = Summary::EMPTY;
        for boundary in 1..OBJECT_BINS {
            let bin = bins[boundary - 1];
            if bin.count == 0 {
                continue;
            }
            left.merge(bin);
            let right = suffix[boundary];
            if right.count == 0 || left.count > child_limit || right.count > child_limit {
                continue;
            }
            let split = Split { partition: Partition::Object { grid, boundary }, left, right };
            let cost = split.cost();
            let imbalance = left.count.abs_diff(right.count);
            if best.is_none_or(|(old, diff, _)| cost < old || (cost == old && imbalance < diff)) {
                best = Some((cost, imbalance, split));
            }
        }
    }
    best.map(|(_, _, split)| split)
}

fn spatial_split(
    corners: &[[Vec3; 3]],
    references: &[Reference],
    bounds: Bounds,
    child_limit: usize,
    budget: usize,
    mut cost_limit: f64,
) -> Option<Split> {
    let mut best = None;
    for axis in 0..3 {
        let (lo, hi) = (bounds.min[axis], bounds.max[axis]);
        let step = (f64::from(hi) - f64::from(lo)) / SPATIAL_BINS as f64;
        let planes: [f32; PLANES] = std::array::from_fn(|i| (f64::from(lo) + (i + 1) as f64 * step) as f32);
        let mut bins = std::array::from_fn::<_, PLANES, _>(|i| {
            (lo < planes[i] && planes[i] < hi && (i == 0 || planes[i - 1] != planes[i]))
                .then_some([Summary::EMPTY; 2])
        });
        if bins.iter().all(Option::is_none) {
            continue;
        }
        let mut left_starts = [Summary::EMPTY; SPATIAL_BINS];
        let mut right_ends = [Summary::EMPTY; SPATIAL_BINS];
        for reference in references {
            let before = planes.partition_point(|&p| p <= reference.bounds.min[axis]);
            let after = planes.partition_point(|&p| p < reference.bounds.max[axis]);
            // Whole references enter prefix/suffix unions once. Only strict crossings
            // need clipping; a reference on the plane belongs to the left child.
            left_starts[after].add(reference.bounds);
            right_ends[before.min(after)].add(reference.bounds);
            let mut polygon = None;
            for i in before.min(after)..after {
                let Some([left, right]) = &mut bins[i] else { continue };
                if left.count == budget {
                    bins[i] = None;
                    continue;
                }
                let fragment = polygon.get_or_insert_with(|| {
                    Polygon::bounded(&corners[reference.source as usize], reference.bounds)
                });
                if let Some([l, r]) = fragment.as_ref().and_then(|p| p.split(reference.bounds, axis, planes[i])) {
                    left.add(l);
                    right.add(r);
                } else {
                    // Don't select a plane if roundoff leaves any crossing indeterminate.
                    bins[i] = None;
                }
            }
        }
        let mut right = Summary::EMPTY;
        for i in (0..PLANES).rev() {
            right.merge(right_ends[i + 1]);
            if let Some([_, clipped]) = &mut bins[i] {
                clipped.merge(right);
            }
        }
        let mut left = Summary::EMPTY;
        for i in 0..PLANES {
            left.merge(left_starts[i]);
            let Some([mut clipped, right]) = bins[i] else { continue };
            if clipped.count == 0 {
                continue;
            }
            clipped.merge(left);
            if clipped.count > child_limit || right.count > child_limit {
                continue;
            }
            let split = Split {
                partition: Partition::Spatial { axis, plane: planes[i] },
                left: clipped,
                right,
            };
            let cost = split.cost();
            if cost < cost_limit {
                cost_limit = cost;
                best = Some(split);
            }
        }
    }
    best
}

// A clipped polygon is temporary. A reference's box encloses its fragment and lies
// inside every plane used to create it. Re-intersecting the source with that box
// therefore reconstructs the same fragment without storing extra triangle geometry.
const POLYGON_CAPACITY: usize = 3 + 6;
type Vertex = [Interval; 3];

struct Polygon {
    vertices: [Vertex; POLYGON_CAPACITY],
    count: usize,
}

impl Polygon {
    fn empty() -> Self {
        Self { vertices: [[Interval::exact(0.0); 3]; POLYGON_CAPACITY], count: 0 }
    }

    fn bounded(triangle: &[Vec3; 3], bounds: Bounds) -> Option<Self> {
        let mut polygon = Self::empty();
        for point in triangle {
            polygon.push(point.to_array().map(|v| Interval::exact(f64::from(v))))?;
        }
        let source = Bounds::triangle(triangle);
        for axis in 0..3 {
            if bounds.min[axis] > source.min[axis] {
                polygon = polygon.clip(axis, f64::from(bounds.min[axis]), Greater)?;
            }
            if bounds.max[axis] < source.max[axis] {
                polygon = polygon.clip(axis, f64::from(bounds.max[axis]), Less)?;
            }
        }
        Some(polygon)
    }

    fn push(&mut self, vertex: Vertex) -> Option<()> {
        *self.vertices.get_mut(self.count)? = vertex;
        self.count += 1;
        Some(())
    }

    fn visit(
        &self,
        axis: usize,
        plane: f64,
        mut vertex: impl FnMut(&Vertex, Ordering) -> Option<()>,
    ) -> Option<()> {
        let vertices = &self.vertices[..self.count];
        let mut prev = vertices.last()?;
        let mut prev_side = prev[axis].side(plane)?;
        for next in vertices {
            let side = next[axis].side(plane)?;
            if prev_side != side && prev_side != Equal && side != Equal {
                let (low, high) = if prev_side == Less { (prev, next) } else { (next, prev) };
                vertex(&edge_intersection(low, high, axis, plane)?, Equal)?;
            }
            vertex(next, side)?;
            prev = next;
            prev_side = side;
        }
        Some(())
    }

    fn clip(&self, axis: usize, plane: f64, keep: Ordering) -> Option<Self> {
        let mut result = Self::empty();
        self.visit(axis, plane, |point, side| {
            if side == keep || side == Equal {
                result.push(*point)?;
            }
            Some(())
        })?;
        (result.count != 0).then_some(result)
    }

    fn split(&self, bounds: Bounds, axis: usize, plane: f32) -> Option<[Bounds; 2]> {
        let mut left = Extents::EMPTY;
        let mut right = Extents::EMPTY;
        self.visit(axis, f64::from(plane), |point, side| {
            if side != Greater {
                left.add(point);
            }
            if side != Less {
                right.add(point);
            }
            Some(())
        })?;
        let mut left = left.to_bounds().intersection(bounds);
        let mut right = right.to_bounds().intersection(bounds);
        // Outward rounding cannot cross a representable split plane. Clip to it
        // explicitly, and decline numerically ambiguous or merely touching cases.
        left.max[axis] = plane;
        right.min[axis] = plane;
        (left.min[axis] < plane && plane < right.max[axis]).then_some([left, right])
    }
}

#[derive(Clone, Copy)]
struct Interval {
    lo: f64,
    hi: f64,
}

impl Interval {
    const fn exact(v: f64) -> Self {
        Self { lo: v, hi: v }
    }

    fn side(self, plane: f64) -> Option<Ordering> {
        if self.hi < plane {
            Some(Less)
        } else if self.lo > plane {
            Some(Greater)
        } else if self.lo == plane && self.hi == plane {
            Some(Equal)
        } else {
            None
        }
    }

    // Bounds on a nonnegative distance from either endpoint to the plane.
    fn gap(lo: f64, hi: f64) -> Self {
        Self { lo: lo.next_down().max(0.0), hi: hi.next_up() }
    }

    fn ratio(self, denom: Self) -> Self {
        Self { lo: (self.lo / denom.hi).next_down().max(0.0), hi: (self.hi / denom.lo).next_up().min(1.0) }
    }

    fn weighted(self, weight: Self) -> Self {
        let lo = self.lo * if self.lo < 0.0 { weight.hi } else { weight.lo };
        let hi = self.hi * if self.hi < 0.0 { weight.lo } else { weight.hi };
        Self { lo: lo.next_down(), hi: hi.next_up() }
    }
}

fn edge_intersection(low: &Vertex, high: &Vertex, axis: usize, plane: f64) -> Option<Vertex> {
    // Round each arithmetic endpoint outward; converting just the final f64 value
    // outward is not sufficient after cancellation or repeated intersections.
    let left = Interval::gap(plane - low[axis].hi, plane - low[axis].lo);
    let right = Interval::gap(high[axis].lo - plane, high[axis].hi - plane);
    let width = Interval::gap(left.lo + right.lo, left.hi + right.hi);
    if width.lo == 0.0 {
        return None;
    }
    let low_weight = right.ratio(width);
    let high_weight = left.ratio(width);
    Some(std::array::from_fn(|i| {
        if i == axis {
            return Interval::exact(plane);
        }
        let (a, b) = (low[i], high[i]);
        let hull = Interval { lo: a.lo.min(b.lo), hi: a.hi.max(b.hi) };
        if hull.lo == hull.hi {
            return hull;
        }
        let (a, b) = (a.weighted(low_weight), b.weighted(high_weight));
        Interval { lo: (a.lo + b.lo).next_down().max(hull.lo), hi: (a.hi + b.hi).next_up().min(hull.hi) }
    }))
}

struct Extents {
    lo: DVec3,
    hi: DVec3,
}

impl Extents {
    const EMPTY: Self = Self { lo: DVec3::splat(f64::INFINITY), hi: DVec3::splat(f64::NEG_INFINITY) };

    fn add(&mut self, point: &Vertex) {
        self.lo = self.lo.min(DVec3::from_array(point.map(|v| v.lo)));
        self.hi = self.hi.max(DVec3::from_array(point.map(|v| v.hi)));
    }

    fn to_bounds(&self) -> Bounds {
        let min = self.lo.to_array().map(|v| {
            let f = v as f32;
            if f64::from(f) > v { f.next_down() } else { f }
        });
        let max = self.hi.to_array().map(|v| {
            let f = v as f32;
            if f64::from(f) < v { f.next_up() } else { f }
        });
        Bounds { min: Vec3::from_array(min), max: Vec3::from_array(max) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clipping_tightens_other_axes_and_keeps_shared_boundary() {
        let triangle = [Vec3::ZERO, Vec3::new(4.0, 4.0, 0.0), Vec3::new(4.0, 0.0, 0.0)];
        let bounds = Bounds::triangle(&triangle);
        let [left, right] = Polygon::bounded(&triangle, bounds).expect("source polygon")
            .split(bounds, 0, 1.0).expect("triangle straddles x = 1");
        assert_eq!(left.max.x, 1.0);
        assert_eq!(right.min.x, 1.0);
        assert!((1.0..=1.0_f32.next_up()).contains(&left.max.y));
        assert_eq!(right.max.y, 4.0);
        let [_, top] = Polygon::bounded(&triangle, left).expect("reconstructed left fragment")
            .split(left, 1, 0.5).expect("fragment straddles y = 0.5");
        assert!(top.min.x <= 0.5 && top.min.x >= 0.5_f32.next_down());
        assert_eq!(top.min.y, 0.5);
        assert_eq!(top.max.x, 1.0);
        assert_eq!(top.max.z, 0.0);
    }

    fn clusters() -> Vec<[Vec3; 3]> {
        let mut triangles = Vec::new();
        for i in 0..8 {
            for (x, y) in [(-8.5, -8.0), (8.0, 8.0)] {
                let a = Vec3::new(x, y + 0.125 * i as f32, 1.0);
                triangles.push([a, a + Vec3::X * 0.25, a + Vec3::Y * 0.0625]);
            }
        }
        triangles
    }

    #[test]
    fn chooses_spatial_sah_and_preserves_actual_source_hits() {
        let mut triangles = clusters();
        let straddler = 3;
        triangles.insert(straddler, [Vec3::new(-8.0, -8.0, 0.0), Vec3::new(8.0, 8.0, 0.0), Vec3::new(8.0, 7.5, 0.0)]);
        let tree = build(&triangles).expect("diagonal straddler causes profitable spatial splits");
        assert!(tree.references.iter().filter(|&&id| id == straddler as u32).count() > 1);
        assert!(tree.references.len() <= triangles.len() + triangles.len() / DUPLICATION_DIVISOR);
        let mut leaves = Vec::new();
        let mut stack = vec![(0, 0)];
        while let Some((at, depth)) = stack.pop() {
            let node = tree.nodes[at];
            assert!(depth <= MAX_DEPTH);
            if node.count == 0 {
                stack.push((node.first as usize, depth + 1));
                stack.push((node.first as usize + 1, depth + 1));
            } else {
                assert!(node.count as usize <= MAX_LEAF);
                leaves.extend(node.first..node.first + node.count);
            }
        }
        leaves.sort_unstable();
        assert!(leaves.into_iter().eq(0..tree.references.len() as u32));
        for (x, y, expected) in [
            (-6.0, -6.02, Some(straddler as u32)),
            (6.0, 5.75, Some(straddler as u32)),
            (-8.45, -7.98, Some(0)),
            (8.05, 8.02, Some(1)),
            (0.0, 1.0, None), // Inside the diagonal's source AABB, outside the triangle.
        ] {
            let origin = DVec3::new(x, y, 10.0);
            let brute = closest(&triangles, 0..triangles.len() as u32, origin);
            assert_eq!(brute.map(|hit| hit.1), expected, "source ray at {origin:?}");
            let mut candidates = Vec::new();
            let mut stack = vec![0];
            while let Some(at) = stack.pop() {
                let node = tree.nodes[at];
                if x < f64::from(node.min.x) || x > f64::from(node.max.x)
                    || y < f64::from(node.min.y) || y > f64::from(node.max.y)
                    || origin.z < f64::from(node.min.z)
                {
                    continue;
                }
                let first = node.first as usize;
                if node.count == 0 {
                    stack.extend([first, first + 1]);
                } else {
                    candidates.extend_from_slice(&tree.references[first..first + node.count as usize]);
                }
            }
            assert_eq!(closest(&triangles, candidates, origin), brute, "spatial ray at {origin:?}");
        }
    }

    fn closest(
        triangles: &[[Vec3; 3]],
        references: impl IntoIterator<Item = u32>,
        origin: DVec3,
    ) -> Option<(f64, u32)> {
        references.into_iter().filter_map(|id| {
            let [a, b, c] = triangles[id as usize].map(|p| p.as_dvec3());
            let (ab, ac) = (b - a, c - a);
            let direction = -DVec3::Z;
            let p = direction.cross(ac);
            let det = ab.dot(p);
            if det == 0.0 {
                return None;
            }
            let offset = origin - a;
            let q = offset.cross(ab);
            let u = offset.dot(p) / det;
            let v = direction.dot(q) / det;
            let t = ac.dot(q) / det;
            (u >= 0.0 && v >= 0.0 && u + v <= 1.0 && t >= 0.0).then_some((t, id))
        }).min_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)))
    }

    #[test]
    fn disjoint_geometry_and_invalid_input_use_object_path() {
        assert!(build(&clusters()).is_none());
        assert!(build(&[[Vec3::NAN; 3]; 8]).is_none());
        assert!(build(&[]).is_none());
    }
}
