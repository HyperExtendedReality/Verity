//! CPU ray queries against the world-space triangles of a static scene.
//!
//! This is a geometric alternative to the renderer's acceleration structures,
//! not a material-aware tracer: triangles are two-sided and opaque, regardless
//! of the draw's material, alpha, or render-layer settings.

use glam::{Affine3A, DVec3, Vec3, Vec4};
use std::{error::Error, fmt, ops::RangeInclusive};
use verity_render::asset::SceneData;

const LEAF_SIZE: usize = 4;
const MAX_DEPTH: usize = 32;
const SAH_PLANES: usize = 15;

/// The source indices of a triangle instance, independent of kd-tree layout.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub struct TriangleId {
    /// Index in [`SceneData::draws`].
    pub instance_index: usize,
    /// Index in [`SceneData::primitives`].
    pub primitive_index: u32,
    /// Index of the source index triplet within that primitive, before filtering.
    pub triangle_index: usize,
}

/// The intersection of a ray with a source triangle instance.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Hit {
    /// Parameter in `origin + direction * t`; a distance only for unit directions.
    pub t: f64,
    /// Original draw, primitive, and triangle indices.
    pub id: TriangleId,
}

/// Invalid source geometry for a draw in [`KdTree::from_scene`].
#[derive(Debug)]
pub struct BuildError {
    instance_index: usize,
    reason: &'static str,
}

impl BuildError {
    /// Index of the invalid draw in [`SceneData::draws`].
    pub fn instance_index(&self) -> usize {
        self.instance_index
    }
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "draw {}: {}", self.instance_index, self.reason)
    }
}

impl Error for BuildError {}

/// An owned spatial kd-tree over the drawn triangles in one scene snapshot.
#[derive(Debug, Default)]
pub struct KdTree {
    bounds: Option<Bounds>,
    triangles: Vec<Triangle>,
    nodes: Vec<Node>,
    references: Vec<usize>,
}

impl KdTree {
    /// Builds from the scene's triangle-list draws using their affine world transforms.
    ///
    /// Indices are relative to each primitive's base vertex. Only complete index
    /// triplets are read, and triangles with zero world-space area are omitted.
    /// The tree owns its geometry; changes to the scene require a rebuild.
    ///
    /// # Errors
    ///
    /// Returns an error for a bad primitive, index span, or vertex reference, a
    /// non-finite world vertex, or a non-finite or non-affine draw transform.
    pub fn from_scene(scene: &SceneData) -> Result<Self, BuildError> {
        let mut triangles = Vec::new();
        let mut triangle_bounds = Vec::new();
        let mut root_bounds: Option<Bounds> = None;
        for (instance_index, draw) in scene.draws.iter().enumerate() {
            let invalid = |reason| BuildError {
                instance_index,
                reason,
            };
            let primitive_index = draw.primitive;
            let primitive = scene
                .primitives
                .get(primitive_index as usize)
                .ok_or_else(|| invalid("primitive index out of bounds"))?;
            let start = primitive.first_index as usize;
            let count = primitive.index_count as usize;
            let indices = scene
                .indices
                .get(start..)
                .and_then(|tail| tail.get(..count))
                .ok_or_else(|| invalid("primitive index span out of bounds"))?;
            if !draw.transform.is_finite() || draw.transform.row(3) != Vec4::W {
                return Err(invalid("world transform is not finite and affine"));
            }
            let transform = Affine3A::from_mat4(draw.transform);
            for (triangle_index, indices) in indices.as_chunks::<3>().0.iter().enumerate() {
                let mut points = [Vec3::ZERO; 3];
                for (&index, point) in indices.iter().zip(&mut points) {
                    let vertex = primitive
                        .base_vertex
                        .checked_add(index)
                        .and_then(|index| scene.vertices.get(index as usize))
                        .ok_or_else(|| invalid("triangle vertex index out of bounds"))?;
                    let [x, y, z, _] = vertex.position;
                    *point = transform.transform_point3(Vec3::new(x, y, z));
                    if !point.is_finite() {
                        return Err(invalid("world vertex is not finite"));
                    }
                }
                let [a, b, c] = points.map(|point| point.as_dvec3());
                if (b - a).cross(c - a) == DVec3::ZERO {
                    continue;
                }
                let bounds = Bounds {
                    min: a.min(b).min(c),
                    max: a.max(b).max(c),
                };
                root_bounds = Some(root_bounds.map_or(bounds, |root| root.union(bounds)));
                triangle_bounds.push(bounds);
                triangles.push(Triangle {
                    points,
                    id: TriangleId {
                        instance_index,
                        primitive_index,
                        triangle_index,
                    },
                });
            }
        }
        let Some(bounds) = root_bounds else {
            return Ok(Self::default());
        };
        let count = triangles.len();
        let mut builder = Builder {
            triangle_bounds: &triangle_bounds,
            nodes: Vec::new(),
            references: Vec::with_capacity(count),
            // Every extra reference is charged once, capping total leaf references at 2N.
            duplicates_left: count,
        };
        builder.build((0..count).collect(), bounds, 0);
        Ok(Self {
            bounds: Some(bounds),
            triangles,
            nodes: builder.nodes,
            references: builder.references,
        })
    }

    /// Finds the nearest hit within the inclusive, finite parameter range.
    ///
    /// Direction need not be normalized. Non-finite inputs, a zero direction,
    /// or a range not satisfying `0 <= start <= end` yield no hit. Coplanar
    /// rays do not hit a triangle. Equal hit parameters prefer the lowest
    /// [`TriangleId`].
    #[must_use]
    pub fn nearest_hit(
        &self,
        origin: Vec3,
        direction: Vec3,
        range: RangeInclusive<f64>,
    ) -> Option<Hit> {
        self.trace::<false>(origin, direction, range)
    }

    /// Tests whether any triangle intersects the segment, with the same input
    /// and boundary rules as [`Self::nearest_hit`].
    #[must_use]
    pub fn is_occluded(&self, origin: Vec3, direction: Vec3, range: RangeInclusive<f64>) -> bool {
        self.trace::<true>(origin, direction, range).is_some()
    }

    fn trace<const ANY: bool>(
        &self,
        origin: Vec3,
        direction: Vec3,
        range: RangeInclusive<f64>,
    ) -> Option<Hit> {
        let bounds = self.bounds?;
        let ray = Ray::new(origin, direction, range)?;
        let span = ray.clip(bounds)?;
        let mut query = Query { ray, hit: None };
        self.visit::<ANY>(self.nodes.len() - 1, span, &mut query);
        query.hit
    }

    fn visit<const ANY: bool>(&self, node: usize, span: Span, query: &mut Query) -> bool {
        if span.start > span.end.min(query.ray.segment.end) {
            return false;
        }
        match self.nodes[node] {
            Node::Leaf { first, end } => {
                for &reference in &self.references[first..end] {
                    let triangle = &self.triangles[reference];
                    // Test the query segment, not the cell segment: a straddler
                    // can yield a valid hit outside the cell that found it.
                    let Some(t) = query.ray.intersect(&triangle.points) else {
                        continue;
                    };
                    if query
                        .hit
                        .is_none_or(|hit| t < hit.t || (t == hit.t && triangle.id < hit.id))
                    {
                        query.ray.segment.end = t;
                        query.hit = Some(Hit { t, id: triangle.id });
                    }
                    if ANY {
                        return true;
                    }
                }
                false
            }
            Node::Split {
                axis,
                plane,
                lower,
                upper,
            } => {
                let inverse = query.ray.inverse[axis];
                let offset = plane - query.ray.origin[axis];
                if inverse.is_infinite() {
                    // For a ray on the plane, both closed cells can contribute.
                    return (offset >= 0.0 && self.visit::<ANY>(lower, span, query))
                        || (offset <= 0.0 && self.visit::<ANY>(upper, span, query));
                }
                let (near, far) = if inverse > 0.0 {
                    (lower, upper)
                } else {
                    (upper, lower)
                };
                let (down, up) = conservative_parameter(offset * inverse);
                let near_span = Span {
                    end: span.end.min(up),
                    ..span
                };
                let far_span = Span {
                    start: span.start.max(down),
                    ..span
                };
                self.visit::<ANY>(near, near_span, query) || self.visit::<ANY>(far, far_span, query)
            }
        }
    }
}

#[derive(Debug)]
struct Triangle {
    points: [Vec3; 3],
    id: TriangleId,
}

#[derive(Clone, Copy, Debug)]
struct Bounds {
    min: DVec3,
    max: DVec3,
}

impl Bounds {
    fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
}

#[derive(Debug)]
enum Node {
    Leaf {
        first: usize,
        end: usize,
    },
    Split {
        axis: usize,
        plane: f64,
        lower: usize,
        upper: usize,
    },
}

struct Builder<'a> {
    triangle_bounds: &'a [Bounds],
    nodes: Vec<Node>,
    references: Vec<usize>,
    duplicates_left: usize,
}

struct Split {
    axis: usize,
    plane: f64,
    upper_count: usize,
    duplicates: usize,
}

impl Builder<'_> {
    fn build(&mut self, mut references: Vec<usize>, cell: Bounds, depth: usize) -> usize {
        let split = if references.len() > LEAF_SIZE && depth < MAX_DEPTH {
            self.best_split(&references, cell)
        } else {
            None
        };
        let node = if let Some(Split {
            axis,
            plane,
            upper_count,
            duplicates,
        }) = split
        {
            self.duplicates_left -= duplicates;
            let mut upper_refs = Vec::with_capacity(upper_count);
            references.retain(|&reference| {
                let bounds = self.triangle_bounds[reference];
                if bounds.max[axis] >= plane {
                    upper_refs.push(reference);
                }
                bounds.min[axis] <= plane
            });
            let (mut lower_cell, mut upper_cell) = (cell, cell);
            lower_cell.max[axis] = plane;
            upper_cell.min[axis] = plane;
            let lower = self.build(references, lower_cell, depth + 1);
            let upper = self.build(upper_refs, upper_cell, depth + 1);
            Node::Split {
                axis,
                plane,
                lower,
                upper,
            }
        } else {
            let first = self.references.len();
            self.references.extend(references);
            Node::Leaf {
                first,
                end: self.references.len(),
            }
        };
        // Append in postorder so the root is always the last node.
        let index = self.nodes.len();
        self.nodes.push(node);
        index
    }

    fn best_split(&self, references: &[usize], cell: Bounds) -> Option<Split> {
        let extent = cell.max - cell.min;
        let parent_area = extent.x * extent.y + extent.y * extent.z + extent.z * extent.x;
        let count = references.len();
        let mut best_cost = parent_area * count as f64;
        let mut best = None;
        for axis in 0..3 {
            if extent[axis] == 0.0 {
                continue;
            }
            let (lo, hi) = (cell.min[axis], cell.max[axis]);
            let step = extent[axis] / (SAH_PLANES + 1) as f64;
            let planes: [f64; SAH_PLANES] = std::array::from_fn(|i| lo + step * (i + 1) as f64);
            let mut lower_starts = [0; SAH_PLANES + 1];
            let mut upper_ends = [0; SAH_PLANES + 1];
            for &reference in references {
                let bounds = &self.triangle_bounds[reference];
                // Each reference contributes to a suffix of the lower counts and
                // a prefix of the upper counts. A straddler contributes to both.
                lower_starts[planes.partition_point(|&p| p < bounds.min[axis])] += 1;
                upper_ends[planes.partition_point(|&p| p <= bounds.max[axis])] += 1;
            }
            let (a, b) = (extent[(axis + 1) % 3], extent[(axis + 2) % 3]);
            let (face, perimeter) = (a * b, a + b);
            let (mut lower_count, mut upper_count) = (0, count);
            for (i, plane) in planes.into_iter().enumerate() {
                lower_count += lower_starts[i];
                upper_count -= upper_ends[i];
                if plane <= lo
                    || plane >= hi
                    || lower_count == 0
                    || upper_count == 0
                    || lower_count == count
                    || upper_count == count
                {
                    continue;
                }
                let duplicates = lower_count - (count - upper_count);
                if duplicates > self.duplicates_left {
                    continue;
                }
                let lower_area = face + perimeter * (plane - lo);
                let upper_area = face + perimeter * (hi - plane);
                // SAH, multiplied by the parent area: one traversal and the
                // expected number of triangle tests in each spatial child.
                let cost =
                    parent_area + lower_area * lower_count as f64 + upper_area * upper_count as f64;
                if cost < best_cost {
                    best_cost = cost;
                    best = Some(Split {
                        axis,
                        plane,
                        upper_count,
                        duplicates,
                    });
                }
            }
        }
        best
    }
}

#[derive(Clone, Copy)]
struct Span {
    start: f64,
    end: f64,
}

struct Ray {
    origin: DVec3,
    inverse: DVec3,
    axes: [usize; 3],
    dominant: f64,
    shear: [f64; 2],
    segment: Span,
}

struct Query {
    ray: Ray,
    hit: Option<Hit>,
}

impl Ray {
    fn new(origin: Vec3, direction: Vec3, range: RangeInclusive<f64>) -> Option<Self> {
        let (start, end) = range.into_inner();
        if !origin.is_finite()
            || !direction.is_finite()
            || direction == Vec3::ZERO
            || !start.is_finite()
            || !end.is_finite()
            || start < 0.0
            || end < start
        {
            return None;
        }
        let direction = direction.as_dvec3();
        let absolute = direction.abs();
        let mut z = if absolute.y > absolute.x { 1 } else { 0 };
        if absolute.z > absolute[z] {
            z = 2;
        }
        let (x, y) = ((z + 1) % 3, (z + 2) % 3);
        let inverse = direction.recip();
        Some(Self {
            origin: origin.as_dvec3(),
            inverse,
            axes: [x, y, z],
            dominant: direction[z],
            shear: [direction[x] / direction[z], direction[y] / direction[z]],
            segment: Span { start, end },
        })
    }

    fn clip(&self, bounds: Bounds) -> Option<Span> {
        let mut span = self.segment;
        for axis in 0..3 {
            let origin = self.origin[axis];
            let (min, max) = (bounds.min[axis], bounds.max[axis]);
            let inverse = self.inverse[axis];
            if inverse.is_infinite() {
                if origin < min || origin > max {
                    return None;
                }
                continue;
            }
            let (near, far) = if inverse > 0.0 {
                (min, max)
            } else {
                (max, min)
            };
            span.start = span
                .start
                .max(conservative_parameter((near - origin) * inverse).0);
            span.end = span
                .end
                .min(conservative_parameter((far - origin) * inverse).1);
            if span.start > span.end {
                return None;
            }
        }
        Some(span)
    }

    fn intersect(&self, points: &[Vec3; 3]) -> Option<f64> {
        let [x, y, z] = self.axes;
        let [sx, sy] = self.shear;
        // In ray-aligned space, shared edges are tested with the same projected
        // endpoints, and an edge (or vertex) hit is accepted by either triangle.
        let [a, b, c] = points.each_ref().map(|point| {
            let p = point.as_dvec3() - self.origin;
            DVec3::new(p[x] - sx * p[z], p[y] - sy * p[z], p[z])
        });
        let u = b.x * c.y - b.y * c.x;
        let v = c.x * a.y - c.y * a.x;
        let w = a.x * b.y - a.y * b.x;
        if (u < 0.0 || v < 0.0 || w < 0.0) && (u > 0.0 || v > 0.0 || w > 0.0) {
            return None;
        }
        let determinant = u + v + w;
        if determinant == 0.0 {
            return None;
        }
        // The weights have one sign, so interpolation cannot leave this depth
        // interval. Clamping keeps flat triangles at their exact plane depth.
        let depth = ((u * a.z + v * b.z + w * c.z) / determinant)
            .clamp(a.z.min(b.z).min(c.z), a.z.max(b.z).max(c.z));
        let t = depth / self.dominant;
        (self.segment.start <= t && t <= self.segment.end).then_some(t)
    }
}

fn conservative_parameter(t: f64) -> (f64, f64) {
    // Bound rounding in the subtraction, reciprocal and multiplication for
    // cell traversal only. Triangle hits still use the original query limits.
    (
        t.next_down().next_down().next_down(),
        t.next_up().next_up().next_up(),
    )
}
