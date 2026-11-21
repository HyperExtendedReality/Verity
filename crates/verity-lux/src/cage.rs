//! CPU ray queries for meshes deformed by an authored tetrahedral cage.
//!
//! Preprocessing clips the input triangles to each rest tetrahedron and builds
//! one immutable mini-BLAS per occupied cell. A pose only updates the cage
//! bounds and inverse affine maps; queries move the ray, not the triangles.
//! This is piecewise-affine geometry for two-sided, opaque ray queries, not a
//! raster skinning path or a hardware ray-tracing backend.

use crate::cascades::scene::{BvhNode, build_tree};
use glam::{DVec2, DVec3, DVec4, Vec3};
use std::{error::Error, fmt, ops::RangeInclusive, sync::Arc};

/// One authored mesh triangle, in rest coordinates and original vertex order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CageTriangle {
    /// The caller's original triangle identifier, retained by all its fragments.
    pub primitive_id: u32,
    /// Positions in the same coordinate system as the rest cage.
    pub vertices: [Vec3; 3],
}

/// Whether posed tetrahedra may reverse their orientation relative to rest.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum InversionPolicy {
    /// Reject a pose if any cell has changed its orientation.
    #[default]
    Reject,
    /// Allow orientation reversal, but still reject singular cells.
    Allow,
}

/// A ray hit identified in terms of the original (unclipped) mesh triangle.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct CageHit {
    /// World-ray parameter in `origin + direction * t`, not necessarily a distance.
    pub t: f64,
    /// The identifier supplied in [`CageTriangle::primitive_id`].
    pub primitive_id: u32,
    /// Weights in the original triangle's vertex order, not the clipped fragment's.
    pub barycentrics: DVec3,
    /// The index of the corresponding tetrahedron in the input cell list.
    pub cell_index: usize,
}

/// Malformed cage geometry, source triangles, or pose positions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CageError(CageErrorKind);

#[derive(Clone, Debug, Eq, PartialEq)]
enum CageErrorKind {
    EmptyCage,
    VertexCount { expected: usize, actual: usize },
    NonFiniteVertex { vertex: usize },
    InvalidIndex { cell: usize, vertex: u32 },
    SingularCell { cell: usize },
    InvertedCell { cell: usize },
    NonFiniteTriangle { primitive_id: u32 },
    UnstableClip { cell: usize, primitive_id: u32 },
    TooManyFragments { cell: usize },
}

impl fmt::Display for CageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            CageErrorKind::EmptyCage => f.write_str("cage has no tetrahedra"),
            CageErrorKind::VertexCount { expected, actual } => {
                write!(f, "pose has {actual} vertices, expected {expected}")
            }
            CageErrorKind::NonFiniteVertex { vertex } => {
                write!(f, "cage vertex {vertex} is not finite")
            }
            CageErrorKind::InvalidIndex { cell, vertex } => {
                write!(f, "cage cell {cell} references missing vertex {vertex}")
            }
            CageErrorKind::SingularCell { cell } => write!(f, "cage cell {cell} is degenerate"),
            CageErrorKind::InvertedCell { cell } => write!(f, "pose inverts cage cell {cell}"),
            CageErrorKind::NonFiniteTriangle { primitive_id } => {
                write!(f, "mesh triangle {primitive_id} has a non-finite vertex")
            }
            CageErrorKind::UnstableClip { cell, primitive_id } => {
                write!(
                    f,
                    "clipping mesh triangle {primitive_id} to cell {cell} lost precision"
                )
            }
            CageErrorKind::TooManyFragments { cell } => {
                write!(
                    f,
                    "cage cell {cell} has too many clipped triangles for its BVH"
                )
            }
        }
    }
}

impl Error for CageError {}

/// Shared rest geometry and static mini-BLASes for independently posed cages.
///
/// Cloning the asset shares its immutable data rather than copying the geometry.
#[derive(Clone, Debug)]
pub struct CageAsset(Arc<CageData>);

#[derive(Debug)]
struct CageData {
    vertex_count: usize,
    cells: Vec<Cell>,
}

#[derive(Debug)]
struct Cell {
    indices: [u32; 4],
    reversed: bool,
    fragments: Vec<Fragment>,
    nodes: Vec<BvhNode>,
}

impl CageAsset {
    /// Clips authored triangles against indexed tetrahedra and builds their static trees.
    ///
    /// The caller supplies a conforming cage with disjoint cell interiors;
    /// topology and coverage are not checked. Triangle portions outside the
    /// cage are omitted. A triangle on a shared face can live in both closed
    /// cells. Zero-area input triangles do not participate in ray queries.
    /// Either winding is accepted for each tetrahedron and becomes its rest orientation.
    ///
    /// # Errors
    /// Rejects empty cages, missing indices, non-finite points, numerically
    /// singular tetrahedra, poorly scaled triangle/cell pairs that cannot be
    /// clipped reliably, and a cell exceeding the BVH's index space.
    pub fn build(
        vertices: &[Vec3],
        tetrahedra: &[[u32; 4]],
        triangles: impl IntoIterator<Item = CageTriangle>,
    ) -> Result<Self, CageError> {
        if tetrahedra.is_empty() {
            return Err(CageError(CageErrorKind::EmptyCage));
        }
        validate_vertices(vertices)?;
        let mut cells = Vec::with_capacity(tetrahedra.len());
        let mut rest = Vec::with_capacity(tetrahedra.len());
        for (cell, &indices) in tetrahedra.iter().enumerate() {
            if let Some(&vertex) = indices
                .iter()
                .find(|&&vertex| vertex as usize >= vertices.len())
            {
                return Err(CageError(CageErrorKind::InvalidIndex { cell, vertex }));
            }
            let points = cell_points(vertices, indices);
            let (frame, reversed) =
                Frame::new(points).ok_or(CageError(CageErrorKind::SingularCell { cell }))?;
            rest.push(PlacedCell {
                frame,
                bounds: Bounds::of(points),
            });
            cells.push(Cell {
                indices,
                reversed,
                fragments: Vec::new(),
                nodes: Vec::new(),
            });
        }
        for triangle in triangles {
            if !triangle.vertices.iter().all(|vertex| vertex.is_finite()) {
                return Err(CageError(CageErrorKind::NonFiniteTriangle {
                    primitive_id: triangle.primitive_id,
                }));
            }
            let points = triangle.vertices.map(|point| point.as_dvec3());
            if !has_area(points) {
                continue;
            }
            let bounds = Bounds::of(triangle.vertices);
            for (cell, (geometry, placed)) in cells.iter_mut().zip(&rest).enumerate() {
                if bounds.overlaps(placed.bounds) {
                    clip_triangle(
                        &mut geometry.fragments,
                        &points,
                        triangle.primitive_id,
                        &placed.frame,
                        cell,
                    )?;
                    // A binary tree can have up to 2n - 1 nodes for n fragments.
                    if geometry.fragments.len() > (u32::MAX / 2 + 1) as usize {
                        return Err(CageError(CageErrorKind::TooManyFragments { cell }));
                    }
                }
            }
        }
        for cell in &mut cells {
            cell.nodes = build_tree(&mut cell.fragments, Fragment::bounds);
        }
        Ok(Self(Arc::new(CageData {
            vertex_count: vertices.len(),
            cells,
        })))
    }

    /// Creates an independent pose from cage vertex positions in world space.
    ///
    /// Positions use the same order as in [`Self::build`]. This validates each
    /// cell locally but does not detect intersections between posed cells.
    ///
    /// # Errors
    /// Rejects mismatched or non-finite positions, numerically singular cells,
    /// and inverted cells when using [`InversionPolicy::Reject`].
    pub fn pose(
        &self,
        positions: &[Vec3],
        inversion: InversionPolicy,
    ) -> Result<CagePose, CageError> {
        let mut cells = Vec::new();
        self.0.place(positions, inversion, &mut cells)?;
        Ok(CagePose {
            asset: self.clone(),
            cells,
            pending: Vec::new(),
            inversion,
        })
    }
}

impl CageData {
    fn place(
        &self,
        positions: &[Vec3],
        inversion: InversionPolicy,
        dst: &mut Vec<PlacedCell>,
    ) -> Result<(), CageError> {
        if positions.len() != self.vertex_count {
            return Err(CageError(CageErrorKind::VertexCount {
                expected: self.vertex_count,
                actual: positions.len(),
            }));
        }
        validate_vertices(positions)?;
        dst.clear();
        dst.reserve(self.cells.len());
        for (cell, geometry) in self.cells.iter().enumerate() {
            let points = cell_points(positions, geometry.indices);
            let (frame, reversed) =
                Frame::new(points).ok_or(CageError(CageErrorKind::SingularCell { cell }))?;
            if inversion == InversionPolicy::Reject && reversed != geometry.reversed {
                return Err(CageError(CageErrorKind::InvertedCell { cell }));
            }
            dst.push(PlacedCell {
                frame,
                bounds: Bounds::of(points),
            });
        }
        Ok(())
    }
}

fn validate_vertices(vertices: &[Vec3]) -> Result<(), CageError> {
    if let Some(vertex) = vertices.iter().position(|point| !point.is_finite()) {
        return Err(CageError(CageErrorKind::NonFiniteVertex { vertex }));
    }
    Ok(())
}

fn cell_points(vertices: &[Vec3], indices: [u32; 4]) -> [Vec3; 4] {
    indices.map(|vertex| vertices[vertex as usize])
}

/// An independently updateable world-space cage sharing an immutable asset.
#[derive(Debug)]
pub struct CagePose {
    asset: CageAsset,
    cells: Vec<PlacedCell>,
    pending: Vec<PlacedCell>,
    inversion: InversionPolicy,
}

impl CagePose {
    /// Changes the world-space cage positions without rebuilding the mini-BLASes.
    ///
    /// # Errors
    /// Has the same validation as [`CageAsset::pose`] and leaves the previous
    /// pose queryable on failure.
    pub fn update(&mut self, positions: &[Vec3]) -> Result<(), CageError> {
        self.asset
            .0
            .place(positions, self.inversion, &mut self.pending)?;
        std::mem::swap(&mut self.cells, &mut self.pending);
        Ok(())
    }

    /// Finds the nearest hit within an inclusive world-ray parameter range.
    ///
    /// The direction need not be normalized. Non-finite inputs, zero direction,
    /// and ranges not satisfying `0 <= start <= end` yield no hit. Coplanar
    /// rays do not hit a triangle. Equal parameters prefer the lowest primitive
    /// identifier, then the lowest cell index.
    #[must_use]
    pub fn nearest_hit(
        &self,
        origin: Vec3,
        direction: Vec3,
        range: RangeInclusive<f64>,
    ) -> Option<CageHit> {
        let hit = self.trace::<false>(origin, direction, range)?;
        let [a, b, c] = hit.fragment.source;
        let [u, v, w] = hit.weights.to_array();
        let uv = (a * u + b * v + c * w) / (u + v + w);
        Some(CageHit {
            t: hit.t,
            primitive_id: hit.fragment.primitive_id,
            barycentrics: DVec3::new(1.0 - uv.x - uv.y, uv.x, uv.y),
            cell_index: hit.cell_index,
        })
    }

    /// Tests for any intersection, with the same segment rules as [`Self::nearest_hit`].
    #[must_use]
    pub fn is_occluded(&self, origin: Vec3, direction: Vec3, range: RangeInclusive<f64>) -> bool {
        self.trace::<true>(origin, direction, range).is_some()
    }

    fn trace<const ANY: bool>(
        &self,
        origin: Vec3,
        direction: Vec3,
        range: RangeInclusive<f64>,
    ) -> Option<Intersection<'_>> {
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
        let world = SlabRay {
            origin: origin.as_dvec3(),
            inverse: direction.recip(),
        };
        let mut query = Query {
            span: Span { start, end },
            hit: None,
        };
        for (cell_index, (cell, placed)) in self.asset.0.cells.iter().zip(&self.cells).enumerate() {
            let Some(root) = cell.nodes.first() else {
                continue;
            };
            if world.entry(placed.bounds, query.span).is_none() {
                continue;
            }
            // Transform into this cell's unit rest tetrahedron without normalizing
            // the direction, so the same t applies to both rays.
            let ray = LocalRay::new(
                placed.frame.point(world.origin),
                placed.frame.vector(direction),
            );
            if ray.slab.entry(Bounds::from(root), query.span).is_some()
                && cell.visit::<ANY>(0, &ray, &mut query, cell_index)
            {
                break;
            }
        }
        query.hit
    }
}

#[derive(Clone, Copy, Debug)]
struct Bounds {
    min: Vec3,
    max: Vec3,
}

impl Bounds {
    fn of<const N: usize>(points: [Vec3; N]) -> Self {
        points[1..].iter().fold(
            Self {
                min: points[0],
                max: points[0],
            },
            |bounds, &point| Self {
                min: bounds.min.min(point),
                max: bounds.max.max(point),
            },
        )
    }

    fn overlaps(self, other: Self) -> bool {
        self.min.cmple(other.max).all() && other.min.cmple(self.max).all()
    }
}

impl From<&BvhNode> for Bounds {
    fn from(node: &BvhNode) -> Self {
        Self {
            min: node.min,
            max: node.max,
        }
    }
}

// Maps points into the reference tetrahedron with vertices 0, X, Y, Z.
#[derive(Debug)]
struct Frame {
    base: DVec3,
    inverse_rows: [DVec3; 3],
}

impl Frame {
    fn new(points: [Vec3; 4]) -> Option<(Self, bool)> {
        let [base, a, b, c] = points.map(|point| point.as_dvec3());
        let (a, b, c) = (a - base, b - base, c - base);
        let cofactors = [b.cross(c), c.cross(a), a.cross(b)];
        let det = a.dot(cofactors[0]);
        let extent = a.abs().max(b.abs()).max(c.abs()).max_element();
        let cofactor = cofactors[0]
            .abs()
            .max(cofactors[1].abs())
            .max(cofactors[2].abs())
            .max_element();
        // A relative inverse condition estimate rejects singular and numerically
        // unstable maps without imposing a minimum cell size in world units.
        if det.abs() <= (32.0 * f64::EPSILON) * extent * cofactor {
            return None;
        }
        let inv_det = det.recip();
        Some((
            Self {
                base,
                inverse_rows: cofactors.map(|row| row * inv_det),
            },
            det < 0.0,
        ))
    }

    fn point(&self, point: DVec3) -> DVec3 {
        self.vector(point - self.base)
    }

    fn vector(&self, vector: DVec3) -> DVec3 {
        let [x, y, z] = &self.inverse_rows;
        DVec3::new(x.dot(vector), y.dot(vector), z.dot(vector))
    }
}

#[derive(Debug)]
struct PlacedCell {
    frame: Frame,
    bounds: Bounds,
}

#[derive(Clone, Copy, Default)]
struct ClipVertex {
    cell: DVec4,
    source: DVec2,
}

#[derive(Debug)]
struct Fragment {
    points: [DVec3; 3],
    source: [DVec2; 3],
    primitive_id: u32,
}

impl Fragment {
    fn bounds(&self) -> (Vec3, Vec3) {
        let [a, b, c] = self.points;
        let min = a.min(b).min(c).as_vec3();
        let max = a.max(b).max(c).as_vec3();
        // Mini-BLASes use f32 bounds around f64 triangles.
        (
            Vec3::new(min.x.next_down(), min.y.next_down(), min.z.next_down()),
            Vec3::new(max.x.next_up(), max.y.next_up(), max.z.next_up()),
        )
    }
}

fn has_area([a, b, c]: [DVec3; 3]) -> bool {
    (b - a).cross(c - a) != DVec3::ZERO
}

fn clip_triangle(
    fragments: &mut Vec<Fragment>,
    points: &[DVec3; 3],
    primitive_id: u32,
    frame: &Frame,
    cell: usize,
) -> Result<(), CageError> {
    let vertices = (*points).map(|point| {
        let local = frame.point(point);
        DVec4::new(1.0 - local.x - local.y - local.z, local.x, local.y, local.z)
    });
    let source = [DVec2::ZERO, DVec2::X, DVec2::Y];
    if vertices.iter().all(|w| w.cmpge(DVec4::ZERO).all()) {
        push_fragment(
            fragments,
            vertices.map(fragment_point),
            source,
            primitive_id,
        );
        return Ok(());
    }

    // Exact convex clipping adds at most one vertex per plane. Leave room for
    // the worst-case alternating signs (3 → 4 → 6 → 9 → 13) under rounding.
    const CAPACITY: usize = 13;
    let mut polygon = [ClipVertex::default(); CAPACITY];
    let mut scratch = [ClipVertex::default(); CAPACITY];
    let (mut polygon, mut scratch) = (&mut polygon[..], &mut scratch[..]);
    let mut count = 3;
    for (slot, (cell, source)) in polygon.iter_mut().zip(vertices.into_iter().zip(source)) {
        *slot = ClipVertex { cell, source };
    }
    for axis in 0..4 {
        let mut next = 0;
        let mut prev = polygon[count - 1];
        for &cur in &polygon[..count] {
            let (a, b) = (prev.cell[axis], cur.cell[axis]);
            // Zero-weight endpoints already belong to the closed polygon.
            if (a < 0.0 && b > 0.0) || (a > 0.0 && b < 0.0) {
                // Symmetric interpolation gives shared edges the same result
                // regardless of the direction in which they're clipped.
                let (a, b) = (a.abs(), b.abs());
                let sum = a + b;
                let mut cell = (prev.cell * b + cur.cell * a) / sum;
                cell[axis] = 0.0;
                scratch[next] = ClipVertex {
                    cell,
                    source: (prev.source * b + cur.source * a) / sum,
                };
                next += 1;
            }
            if b >= 0.0 {
                scratch[next] = cur;
                next += 1;
            }
            prev = cur;
        }
        if next < 3 {
            return Ok(());
        }
        std::mem::swap(&mut polygon, &mut scratch);
        count = next;
    }
    // Barycentric weights must sum to one. If cancellation has lost more
    // than the source positions' precision in reference space, fail the build
    // rather than constructing bounds around a malformed fragment.
    if polygon[..count].iter().any(|vertex| {
        let [a, b, c, d] = vertex.cell.to_array();
        (a + b + c + d - 1.0).abs() > f64::from(f32::EPSILON)
    }) {
        return Err(CageError(CageErrorKind::UnstableClip {
            cell,
            primitive_id,
        }));
    }
    for edge in polygon[1..count].windows(2) {
        let corners = [polygon[0], edge[0], edge[1]];
        push_fragment(
            fragments,
            corners.map(|corner| fragment_point(corner.cell)),
            corners.map(|corner| corner.source),
            primitive_id,
        );
    }
    Ok(())
}

fn fragment_point(weights: DVec4) -> DVec3 {
    DVec3::new(weights.y, weights.z, weights.w)
}

fn push_fragment(
    fragments: &mut Vec<Fragment>,
    points: [DVec3; 3],
    source: [DVec2; 3],
    primitive_id: u32,
) {
    if has_area(points) {
        fragments.push(Fragment {
            points,
            source,
            primitive_id,
        });
    }
}

#[derive(Clone, Copy)]
struct Span {
    start: f64,
    end: f64,
}

struct SlabRay {
    origin: DVec3,
    inverse: DVec3,
}

impl SlabRay {
    fn entry(&self, bounds: Bounds, mut span: Span) -> Option<f64> {
        for axis in 0..3 {
            let origin = self.origin[axis];
            let inverse = self.inverse[axis];
            let (min, max) = (f64::from(bounds.min[axis]), f64::from(bounds.max[axis]));
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
            let entry = (near - origin) * inverse;
            let exit = (far - origin) * inverse;
            // Bounds traversal must tolerate rounding; triangle tests retain the
            // original closed interval and the exact best hit found so far.
            span.start = span.start.max(entry.next_down().next_down().next_down());
            span.end = span.end.min(exit.next_up().next_up().next_up());
            if span.start > span.end {
                return None;
            }
        }
        Some(span.start)
    }
}

struct LocalRay {
    slab: SlabRay,
    axes: [usize; 3],
    shear: [f64; 2],
    dominant: f64,
}

impl LocalRay {
    fn new(origin: DVec3, direction: DVec3) -> Self {
        let abs = direction.abs();
        let mut z = if abs.y > abs.x { 1 } else { 0 };
        if abs.z > abs[z] {
            z = 2;
        }
        let (x, y) = ((z + 1) % 3, (z + 2) % 3);
        let inverse = direction.recip();
        Self {
            slab: SlabRay { origin, inverse },
            axes: [x, y, z],
            shear: [direction[x] / direction[z], direction[y] / direction[z]],
            dominant: direction[z],
        }
    }

    fn intersect(&self, points: &[DVec3; 3], span: Span) -> Option<(f64, DVec3)> {
        let [x, y, z] = self.axes;
        let [sx, sy] = self.shear;
        // Project along the ray. Adjacent fragments with matching endpoints
        // evaluate their shared edge with the same products in reverse order.
        let [a, b, c] = points.each_ref().map(|point| {
            let p = *point - self.slab.origin;
            DVec3::new(p[x] - sx * p[z], p[y] - sy * p[z], p[z])
        });
        let u = b.x * c.y - b.y * c.x;
        let v = c.x * a.y - c.y * a.x;
        let w = a.x * b.y - a.y * b.x;
        if (u < 0.0 || v < 0.0 || w < 0.0) && (u > 0.0 || v > 0.0 || w > 0.0) {
            return None;
        }
        let det = u + v + w;
        if det == 0.0 {
            return None;
        }
        let depth = ((u * a.z + v * b.z + w * c.z) / det)
            .clamp(a.z.min(b.z).min(c.z), a.z.max(b.z).max(c.z));
        let t = depth / self.dominant;
        (span.start <= t && t <= span.end).then_some((t, DVec3::new(u, v, w)))
    }
}

struct Intersection<'asset> {
    fragment: &'asset Fragment,
    t: f64,
    weights: DVec3,
    cell_index: usize,
}

struct Query<'asset> {
    span: Span,
    hit: Option<Intersection<'asset>>,
}

impl Cell {
    fn visit<'asset, const ANY: bool>(
        &'asset self,
        index: usize,
        ray: &LocalRay,
        query: &mut Query<'asset>,
        cell_index: usize,
    ) -> bool {
        let node = &self.nodes[index];
        let first = node.first as usize;
        if node.count > 0 {
            for fragment in &self.fragments[first..first + node.count as usize] {
                let Some((t, weights)) = ray.intersect(&fragment.points, query.span) else {
                    continue;
                };
                // The span already excludes hits farther than the current best.
                if query.hit.as_ref().is_none_or(|hit| {
                    t < hit.t || fragment.primitive_id < hit.fragment.primitive_id
                }) {
                    query.span.end = t;
                    query.hit = Some(Intersection {
                        fragment,
                        t,
                        weights,
                        cell_index,
                    });
                }
                if ANY {
                    return true;
                }
            }
            return false;
        }
        let mut children = [first, first + 1].map(|child| {
            (
                child,
                ray.slab.entry(Bounds::from(&self.nodes[child]), query.span),
            )
        });
        if children[0].1.unwrap_or(f64::INFINITY) > children[1].1.unwrap_or(f64::INFINITY) {
            children.swap(0, 1);
        }
        children.into_iter().any(|(child, entry)| {
            entry.is_some_and(|entry| {
                entry <= query.span.end && self.visit::<ANY>(child, ray, query, cell_index)
            })
        })
    }
}
