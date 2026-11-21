//! Object-split BVH construction for Lux's packed binary traversal.

use glam::{DVec3, Vec3, Vec3A};

use super::scene::BvhNode;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod simd;

const BINS: usize = 16;
// `traversal.slang` reserves 49 pending entries. Reserve enough levels for
// balanced splits before budget exhaustion, rather than emitting giant leaves.
const MAX_DEPTH: u32 = 48;
const PARALLEL_THRESHOLD: usize = 16_384;

#[derive(Clone, Copy)]
struct Bounds {
    min: Vec3A,
    max: Vec3A,
}

impl Bounds {
    const EMPTY: Self = Self {
        min: Vec3A::splat(f32::INFINITY),
        max: Vec3A::splat(f32::NEG_INFINITY),
    };

    #[inline]
    fn extend(&mut self, other: Self) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    #[inline]
    fn center(self) -> Vec3A {
        self.min * 0.5 + self.max * 0.5
    }

    #[inline]
    fn area(self) -> f64 {
        // Finite f32 bounds can still overflow f32 extents or areas.
        let d = self.max.as_dvec3() - self.min.as_dvec3();
        d.x * d.y + d.y * d.z + d.z * d.x
    }
}

#[derive(Clone, Copy)]
struct Summary {
    bounds: Bounds,
    centers: Bounds,
}

impl Summary {
    const EMPTY: Self = Self {
        bounds: Bounds::EMPTY,
        centers: Bounds::EMPTY,
    };

    #[inline]
    fn add(&mut self, bounds: Bounds, center: Vec3A) {
        self.bounds.extend(bounds);
        self.centers.min = self.centers.min.min(center);
        self.centers.max = self.centers.max.max(center);
    }
}

#[derive(Clone, Copy)]
struct Bin {
    bounds: Bounds,
    count: usize,
}

impl Bin {
    const EMPTY: Self = Self {
        bounds: Bounds::EMPTY,
        count: 0,
    };

    #[inline]
    fn add(&mut self, bounds: Bounds) {
        self.bounds.extend(bounds);
        self.count += 1;
    }

    fn merge(&mut self, other: Self) {
        self.bounds.extend(other.bounds);
        self.count += other.count;
    }

    fn cost(self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.bounds.area() * self.count as f64
        }
    }
}

#[derive(Clone, Copy)]
struct Layout {
    max_leaf: usize,
    traversal_cost: f64,
}

// A triangle leaf can amortize node loads over several triangle tests. An
// instance test can launch another tree walk, so use smaller TLAS leaves.
const BLAS: Layout = Layout {
    max_leaf: 8,
    traversal_cost: 3.0,
};
const TLAS: Layout = Layout {
    max_leaf: 4,
    traversal_cost: 1.0,
};

#[derive(Clone, Copy)]
enum Backend {
    Portable,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2(simd::Avx2),
}

impl Backend {
    fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Some(avx2) = simd::Avx2::detect() {
            return Self::Avx2(avx2);
        }
        Self::Portable
    }

    fn partition<T>(
        self,
        elements: &mut [T],
        bounds: &impl Fn(&T) -> Bounds,
        sah: Sah,
        origin: Vec3A,
        scale: Vec3A,
    ) -> Option<Partition> {
        match self {
            Self::Portable => partition_sah(elements, bounds, sah, |point| {
                ((point - origin) * scale).as_uvec3().to_array()
            }),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Self::Avx2(avx2) => avx2.partition(elements, bounds, sah, origin, scale),
        }
    }
}

#[derive(Clone, Copy)]
struct BuildConfig {
    layout: Layout,
    backend: Backend,
}

impl BuildConfig {
    fn new(layout: Layout) -> Self {
        Self {
            layout,
            backend: Backend::detect(),
        }
    }
}

struct Subtree<'a, T> {
    elements: &'a mut [T],
    summary: Summary,
    first: usize,
    depth: u32,
}

impl<T> Subtree<'_, T> {
    fn leaf(&self) -> BvhNode {
        BvhNode {
            min: self.summary.bounds.min.into(),
            max: self.summary.bounds.max.into(),
            first: self.first as u32,
            count: self.elements.len() as u32,
        }
    }
}

#[inline]
fn summarize<T>(elements: &[T], bounds: &impl Fn(&T) -> Bounds) -> Summary {
    let mut summary = Summary::EMPTY;
    for element in elements {
        let item = bounds(element);
        summary.add(item, item.center());
    }
    summary
}

pub(super) fn build<T>(elements: &mut [T], bounds: impl Fn(&T) -> (Vec3, Vec3)) -> Vec<BvhNode> {
    build_with(elements, to_bounds(bounds), BuildConfig::new(TLAS), serial)
}

pub(super) fn build_mesh<T: Send>(
    elements: &mut [T],
    bounds: impl Fn(&T) -> (Vec3, Vec3) + Sync,
) -> Vec<BvhNode> {
    build_with(
        elements,
        to_bounds(bounds),
        BuildConfig::new(BLAS),
        parallel,
    )
}

fn to_bounds<T, F: Fn(&T) -> (Vec3, Vec3)>(bounds: F) -> impl Fn(&T) -> Bounds {
    move |element| {
        let (min, max) = bounds(element);
        Bounds {
            min: min.into(),
            max: max.into(),
        }
    }
}

type Descend<T, F> = fn(&mut Vec<BvhNode>, usize, Subtree<'_, T>, &F, BuildConfig);

fn build_with<T, F: Fn(&T) -> Bounds>(
    elements: &mut [T],
    bounds: F,
    config: BuildConfig,
    descend: Descend<T, F>,
) -> Vec<BvhNode> {
    if elements.is_empty() {
        return Vec::new();
    }
    let root = Subtree {
        summary: summarize(elements, &bounds),
        elements,
        first: 0,
        depth: 0,
    };
    let mut nodes = vec![root.leaf()];
    descend(&mut nodes, 0, root, &bounds, config);
    nodes
}

fn serial<T, F: Fn(&T) -> Bounds>(
    nodes: &mut Vec<BvhNode>,
    index: usize,
    tree: Subtree<'_, T>,
    bounds: &F,
    config: BuildConfig,
) {
    let Some((left, right)) = split(tree, bounds, config) else {
        return;
    };
    let children = append_children(nodes, index, &left, &right);
    serial(nodes, children, left, bounds, config);
    serial(nodes, children + 1, right, bounds, config);
}

fn parallel<T: Send, F: Fn(&T) -> Bounds + Sync>(
    nodes: &mut Vec<BvhNode>,
    index: usize,
    tree: Subtree<'_, T>,
    bounds: &F,
    config: BuildConfig,
) {
    if tree.elements.len() < PARALLEL_THRESHOLD {
        serial(nodes, index, tree, bounds, config);
        return;
    }
    let Some((left, right)) = split(tree, bounds, config) else {
        return;
    };
    let children = append_children(nodes, index, &left, &right);
    if left.elements.len().min(right.elements.len()) < PARALLEL_THRESHOLD / 2 {
        parallel(nodes, children, left, bounds, config);
        parallel(nodes, children + 1, right, bounds, config);
        return;
    }
    // Keep one subtree in its final buffer. Only the stolen work needs an
    // independent node vector and an index fixup after the join.
    let ((), sibling) = rayon::join(
        || parallel(nodes, children, left, bounds, config),
        || {
            let mut sibling = vec![right.leaf()];
            parallel(&mut sibling, 0, right, bounds, config);
            sibling
        },
    );
    let end = nodes.len() as u32 - 1;
    let mut sibling = sibling.into_iter().map(|mut node| {
        if node.count == 0 {
            node.first += end;
        }
        node
    });
    if let Some(root) = sibling.next() {
        nodes[children + 1] = root;
    }
    nodes.extend(sibling);
}

#[inline]
fn append_children<T>(
    nodes: &mut Vec<BvhNode>,
    index: usize,
    left: &Subtree<'_, T>,
    right: &Subtree<'_, T>,
) -> usize {
    let children = nodes.len();
    nodes[index].first = children as u32;
    nodes[index].count = 0;
    nodes.extend([left.leaf(), right.leaf()]);
    children
}

fn split<'a, T>(
    tree: Subtree<'a, T>,
    bounds: &impl Fn(&T) -> Bounds,
    config: BuildConfig,
) -> Option<(Subtree<'a, T>, Subtree<'a, T>)> {
    let Subtree {
        elements,
        summary,
        first,
        depth,
    } = tree;
    let BuildConfig { layout, backend } = config;
    let count = elements.len();
    if count as f64 <= layout.traversal_cost.max(2.0) {
        return None;
    }
    let fits_leaf = count <= layout.max_leaf;
    let median_levels = count
        .div_ceil(layout.max_leaf)
        .next_power_of_two()
        .trailing_zeros();
    let centers = summary.centers;
    let active = centers.min.cmplt(centers.max);
    let split = if MAX_DEPTH - depth > median_levels && active.any() {
        // Scale the bin count to the number of available centroids.
        let bins = if count <= 8 {
            4
        } else if count <= 32 {
            8
        } else {
            BINS
        };
        let sah = Sah {
            bins,
            active: active.bitmask(),
            summary,
            layout,
        };
        let scale = Vec3A::splat(bins as f32) / (centers.max - centers.min);
        let scale = Vec3A::select(active, scale, Vec3A::ZERO);
        if scale.is_finite() && (scale.cmpgt(Vec3A::ZERO) | !active).all() {
            backend.partition(elements, bounds, sah, centers.min, scale)
        } else {
            let origin = centers.min.as_dvec3();
            let span = centers.max.as_dvec3() - origin;
            let scale = DVec3::select(span.cmpgt(DVec3::ZERO), bins as f64 / span, DVec3::ZERO);
            partition_sah(elements, bounds, sah, |point| {
                ((point.as_dvec3() - origin) * scale).as_uvec3().to_array()
            })
        }
    } else {
        None
    };
    let (mid, left, right) = if let Some(parts) = split {
        parts
    } else if fits_leaf {
        return None;
    } else {
        let mid = count / 2;
        if active.any() {
            let extent = centers.max.as_dvec3() - centers.min.as_dvec3();
            let axis = if extent.x >= extent.y && extent.x >= extent.z {
                0
            } else if extent.y >= extent.z {
                1
            } else {
                2
            };
            elements.select_nth_unstable_by(mid, |a, b| {
                bounds(a).center()[axis].total_cmp(&bounds(b).center()[axis])
            });
        }
        let (left, right) = elements.split_at(mid);
        (mid, summarize(left, bounds), summarize(right, bounds))
    };
    let (l, r) = elements.split_at_mut(mid);
    Some((
        Subtree {
            elements: l,
            summary: left,
            first,
            depth: depth + 1,
        },
        Subtree {
            elements: r,
            summary: right,
            first: first + mid,
            depth: depth + 1,
        },
    ))
}

type Partition = (usize, Summary, Summary);

struct Sah {
    bins: usize,
    active: u32,
    summary: Summary,
    layout: Layout,
}

impl Sah {
    // Keep the packed bucket reader in the AVX2 caller rather than crossing target features.
    #[inline(always)]
    fn best_split(
        &self,
        count: usize,
        bucket: impl Fn(usize, usize) -> Bin,
    ) -> Option<(usize, usize)> {
        let mut best = None::<(f64, usize, usize, usize)>;
        for axis in 0..3 {
            if self.active & (1 << axis) == 0 {
                continue;
            }
            let mut right_cost = [0.0; BINS];
            let mut right = Bin::EMPTY;
            for i in (1..self.bins).rev() {
                right.merge(bucket(axis, i));
                right_cost[i] = right.cost();
            }
            let mut left = Bin::EMPTY;
            for (plane, &suffix_cost) in right_cost[..self.bins].iter().enumerate().skip(1) {
                let bin = bucket(axis, plane - 1);
                if bin.count == 0 {
                    continue;
                }
                left.merge(bin);
                let right_count = count - left.count;
                if right_count == 0 {
                    break;
                }
                let cost = left.cost() + suffix_cost;
                let imbalance = left.count.abs_diff(right_count);
                if best
                    .is_none_or(|(old, diff, _, _)| cost < old || (cost == old && imbalance < diff))
                {
                    best = Some((cost, imbalance, axis, plane));
                }
            }
        }
        let (cost, _, axis, plane) = best?;
        if count <= self.layout.max_leaf
            && cost >= self.summary.bounds.area() * (count as f64 - self.layout.traversal_cost)
        {
            return None;
        }
        Some((axis, plane))
    }
}

fn partition_sah<T>(
    elements: &mut [T],
    bounds: &impl Fn(&T) -> Bounds,
    sah: Sah,
    coordinate: impl Fn(Vec3A) -> [u32; 3],
) -> Option<Partition> {
    let mut buckets = [[Bin::EMPTY; BINS]; 3];
    let bin = |i: u32| (i as usize).min(sah.bins - 1);
    for element in elements.iter() {
        let item = bounds(element);
        let [x, y, z] = coordinate(item.center()).map(bin);
        if sah.active & 1 != 0 {
            buckets[0][x].add(item);
        }
        if sah.active & 2 != 0 {
            buckets[1][y].add(item);
        }
        if sah.active & 4 != 0 {
            buckets[2][z].add(item);
        }
    }
    let (axis, plane) = sah.best_split(elements.len(), |axis, i| buckets[axis][i])?;
    Some(partition_by(elements, bounds, |center| {
        bin(coordinate(center)[axis]) < plane
    }))
}

// Keep the SIMD predicate in the AVX2 caller, not in a separate baseline loop.
#[inline(always)]
fn partition_by<T>(
    elements: &mut [T],
    bounds: &impl Fn(&T) -> Bounds,
    goes_left: impl Fn(Vec3A) -> bool,
) -> Partition {
    let (mut i, mut j) = (0, elements.len());
    let (mut left, mut right) = (Summary::EMPTY, Summary::EMPTY);
    while i < j {
        let item = bounds(&elements[i]);
        let center = item.center();
        if goes_left(center) {
            left.add(item, center);
            i += 1;
        } else {
            right.add(item, center);
            j -= 1;
            elements.swap(i, j);
        }
    }
    (i, left, right)
}

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
mod tests {
    use super::*;

    #[test]
    fn avx2_matches_portable_topology_and_leaf_order() {
        let Backend::Avx2(avx2) = Backend::detect() else {
            return;
        };
        for (case, count) in [(0, 7), (0, 8), (0, 9), (0, 32), (0, 33), (1, 257), (2, 129)] {
            let mut portable: Vec<_> = (0..count)
                .map(|id| {
                    let rank = (id * 53) % count;
                    let (center, half) = match case {
                        // Bin edges, duplicated centroids, and inactive Z.
                        0 => (
                            Vec3::new((rank % 9) as f32, (rank / 3 % 5) as f32, 0.0),
                            Vec3::splat(0.125 * (id % 3 + 1) as f32),
                        ),
                        // Equal axis costs and exact bin edges, including the last bin.
                        1 => (Vec3::splat((rank % 17) as f32), Vec3::splat(0.25)),
                        // The root centroid span overflows f32, but the bounds are finite.
                        _ => (
                            Vec3::new((rank as f32 / 64.0 - 1.0) * 3.0e38, 0.0, 0.0),
                            Vec3::new(0.0, 1.0, 1.0),
                        ),
                    };
                    (id, center - half, center + half)
                })
                .collect();
            let mut simd = portable.clone();
            let bounds = |&(_, min, max): &(usize, Vec3, Vec3)| Bounds {
                min: min.into(),
                max: max.into(),
            };
            let config = |backend| BuildConfig {
                layout: BLAS,
                backend,
            };
            let expected = build_with(&mut portable, bounds, config(Backend::Portable), serial);
            let actual = build_with(&mut simd, bounds, config(Backend::Avx2(avx2)), serial);
            assert_eq!(
                simd, portable,
                "primitive order in case {case}, count {count}",
            );
            assert_eq!(
                actual.len(),
                expected.len(),
                "node count in case {case}, count {count}",
            );
            for (index, (a, b)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(
                    (a.min, a.max, a.first, a.count),
                    (b.min, b.max, b.first, b.count),
                    "node {index} in case {case}, count {count}",
                );
            }
        }
    }
}
