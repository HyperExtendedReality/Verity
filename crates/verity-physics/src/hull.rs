//! Convex hulls for colliders, and the polytope and GJK queries contacts use.
//!
//! [`ConvexHull::new`] runs Quickhull (Barber et al. 1996), adding the point
//! farthest outside the hull so far each time. Stopped at a vertex budget,
//! that greedy order gives the hull that misses the least of the input, which
//! is how a render mesh becomes a cheap collider. Coplanar triangles are then
//! merged into polygons: contact clipping wants a box's face as one quad.

use crate::Bounds;
use glam::{Mat3, Vec3};
use std::collections::HashMap;

/// Most vertices a hull keeps. Contact clipping works on fixed buffers sized
/// from it, and past it a hull costs more than it adds (PhysX caps GPU hulls
/// at the same number).
pub const MAX_HULL_VERTICES: usize = 64;
/// Polygon buffer for clipping: an incident face, plus one vertex per side plane.
pub(crate) const HULL_POLY: usize = 2 * MAX_HULL_VERTICES;
/// Adjacent triangles this close in direction are one face.
const COPLANAR_COS: f32 = 0.9999;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Plane {
    pub normal: Vec3,
    pub offset: f32,
}

impl Plane {
    pub fn distance(&self, p: Vec3) -> f32 {
        self.normal.dot(p) - self.offset
    }
}

/// A face's vertices, counter-clockwise seen from outside, as a run of `loops`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Face {
    pub first: u16,
    pub count: u16,
}

/// An edge and the normals either side of it: its arc on the Gauss map.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Edge {
    pub a: u8,
    pub b: u8,
    pub normals: [Vec3; 2],
}

/// A convex polytope's features, borrowed: a hull's, a box's or a triangle's.
#[derive(Clone, Copy)]
pub(crate) struct Poly<'a> {
    pub vertices: &'a [Vec3],
    pub planes: &'a [Plane],
    pub faces: &'a [Face],
    pub loops: &'a [u8],
    pub edges: &'a [Edge],
}

impl Poly<'_> {
    pub fn support(&self, direction: Vec3) -> usize {
        support(self.vertices, direction)
    }

    pub fn face(&self, face: usize) -> impl Iterator<Item = Vec3> + '_ {
        let Face { first, count } = self.faces[face];
        let run = &self.loops[first as usize..(first + count) as usize];
        run.iter().map(|&v| self.vertices[v as usize])
    }
}

pub(crate) fn support(points: &[Vec3], direction: Vec3) -> usize {
    let mut best = (0, f32::NEG_INFINITY);
    for (i, p) in points.iter().enumerate() {
        let d = p.dot(direction);
        if d > best.1 {
            best = (i, d);
        }
    }
    best.0
}

/// A convex collider, in metres, centred on its centroid.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvexHull {
    vertices: Vec<Vec3>,
    planes: Vec<Plane>,
    faces: Vec<Face>,
    loops: Vec<u8>,
    edges: Vec<Edge>,
    center: Vec3,
    volume: f32,
    /// Inertia tensor about the centroid at unit density.
    inertia: Mat3,
    radius: f32,
    inner: f32,
    bounds: Bounds,
}

impl ConvexHull {
    /// The hull of `points` with at most [`MAX_HULL_VERTICES`] vertices; `None`
    /// for fewer than four points, or points that are flat or not finite.
    /// Vertices are moved so the centroid is the origin: [`center`](Self::center)
    /// says where it was, so a mesh drawn for the body moves by its negative.
    pub fn new(points: &[Vec3]) -> Option<Self> {
        Self::with_max_vertices(points, MAX_HULL_VERTICES)
    }

    /// The hull keeping at most `max_vertices` (4 to [`MAX_HULL_VERTICES`]),
    /// each the input point farthest outside those kept before it. Fewer
    /// vertices are cheaper contacts; the hull lies inside the input's, by at
    /// most [`error`](Self::error) of the given points.
    pub fn with_max_vertices(points: &[Vec3], max_vertices: usize) -> Option<Self> {
        if points.iter().any(|p| !p.is_finite()) {
            return None;
        }
        let max_vertices = max_vertices.clamp(4, MAX_HULL_VERTICES);
        let triangles = quickhull(points, max_vertices)?;
        let mut hull = Self::from_triangles(points, &triangles)?;
        hull.recenter();
        Some(hull)
    }

    /// The vertices, relative to the centroid.
    pub fn vertices(&self) -> &[Vec3] {
        &self.vertices
    }

    /// The centroid in the input points' frame.
    pub fn center(&self) -> Vec3 {
        self.center
    }

    pub fn volume(&self) -> f32 {
        self.volume
    }

    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    /// Whether `point`, relative to the centroid, is inside or on the hull.
    pub fn contains(&self, point: Vec3) -> bool {
        self.signed_distance(point) <= 0.0
    }

    /// How far the farthest of `points` (in the input's frame) lies outside
    /// the hull: zero when it holds them all.
    pub fn error(&self, points: &[Vec3]) -> f32 {
        (points.iter())
            .map(|&p| self.signed_distance(p - self.center))
            .fold(0.0, f32::max)
    }

    pub(crate) fn poly(&self) -> Poly<'_> {
        Poly {
            vertices: &self.vertices,
            planes: &self.planes,
            faces: &self.faces,
            loops: &self.loops,
            edges: &self.edges,
        }
    }

    pub(crate) fn mass_properties(&self, density: f32) -> (f32, Mat3) {
        (self.volume * density, self.inertia * density)
    }

    /// The farthest a vertex lies from the centroid.
    pub(crate) fn radius(&self) -> f32 {
        self.radius
    }

    /// The nearest a face lies to the centroid: the inscribed sphere.
    pub(crate) fn inner(&self) -> f32 {
        self.inner
    }

    pub(crate) fn bounds(&self) -> Bounds {
        self.bounds
    }

    /// The largest plane distance: negative inside, a lower bound outside.
    fn signed_distance(&self, p: Vec3) -> f32 {
        (self.planes.iter())
            .map(|plane| plane.distance(p))
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Where a line enters and leaves the hull grown by `pad`, and the normal
    /// where it enters; `None` if it misses.
    fn clip_line(&self, origin: Vec3, direction: Vec3, pad: f32) -> Option<(f32, f32, Vec3)> {
        let (mut enter, mut leave, mut normal) = (f32::NEG_INFINITY, f32::INFINITY, Vec3::ZERO);
        for plane in &self.planes {
            let (along, distance) = (plane.normal.dot(direction), plane.distance(origin) - pad);
            if along.abs() < 1.0e-12 {
                if distance > 0.0 {
                    return None;
                }
                continue;
            }
            let t = -distance / along;
            if along < 0.0 {
                if t > enter {
                    (enter, normal) = (t, plane.normal);
                }
            } else {
                leave = leave.min(t);
            }
        }
        (enter <= leave).then_some((enter, leave, normal))
    }

    /// Where a ray from outside enters, and the outward normal there.
    pub(crate) fn ray(&self, origin: Vec3, direction: Vec3, max: f32) -> Option<(f32, Vec3)> {
        let (enter, _, normal) = self.clip_line(origin, direction, 0.0)?;
        (enter > 0.0 && enter < max).then_some((enter, normal))
    }

    /// Where a point moving by `direction` comes within `pad`, as a fraction
    /// of the move (zero if already within), and the normal there.
    pub(crate) fn sweep(&self, origin: Vec3, direction: Vec3, pad: f32) -> Option<(f32, Vec3)> {
        let (enter, leave, normal) = self.clip_line(origin, direction, pad)?;
        (leave >= 0.0 && enter.is_finite()).then_some((enter.max(0.0), normal))
    }

    /// The surface point nearest `p`, the outward normal there and the signed
    /// distance to it.
    pub(crate) fn nearest(&self, p: Vec3) -> (Vec3, Vec3, f32) {
        let (on_point, on_hull, distance) = gjk(&[p], &self.vertices);
        if distance > 1.0e-6 {
            return (on_hull, (on_point - on_hull) / distance, distance);
        }
        let plane = (self.planes.iter())
            .max_by(|a, b| a.distance(p).total_cmp(&b.distance(p)))
            .expect("a hull has faces");
        let depth = plane.distance(p);
        (p - plane.normal * depth, plane.normal, depth)
    }

    /// Merge the hull's triangles into faces and keep only their vertices.
    fn from_triangles(points: &[Vec3], triangles: &[[u32; 3]]) -> Option<Self> {
        let normal = |t: &[u32; 3]| {
            let [a, b, c] = t.map(|i| points[i as usize]);
            (b - a).cross(c - a)
        };
        let mut owner: HashMap<(u32, u32), usize> = HashMap::new();
        for (i, t) in triangles.iter().enumerate() {
            for k in 0..3 {
                owner.insert((t[k], t[(k + 1) % 3]), i);
            }
        }
        // Union coplanar neighbours.
        let mut group: Vec<usize> = (0..triangles.len()).collect();
        fn root(group: &mut [usize], mut i: usize) -> usize {
            while group[i] != i {
                group[i] = group[group[i]];
                i = group[i];
            }
            i
        }
        let units: Vec<Vec3> = triangles.iter().map(|t| normal(t).normalize_or_zero()).collect();
        for (i, t) in triangles.iter().enumerate() {
            for k in 0..3 {
                let &j = owner.get(&(t[(k + 1) % 3], t[k]))?;
                if units[i].dot(units[j]) > COPLANAR_COS {
                    let (ri, rj) = (root(&mut group, i), root(&mut group, j));
                    group[ri] = rj;
                }
            }
        }
        // Each group's boundary, walked from its directed edges.
        let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..triangles.len() {
            let r = root(&mut group, i);
            members.entry(r).or_default().push(i);
        }
        let mut groups: Vec<Vec<usize>> = members.into_values().collect();
        groups.sort_by_key(|g| g[0]);
        let mut remap: HashMap<u32, u8> = HashMap::new();
        let mut vertices = Vec::new();
        let (mut planes, mut faces, mut loops) = (Vec::new(), Vec::new(), Vec::new());
        for members in &groups {
            let mut next: HashMap<u32, u32> = HashMap::new();
            let mut area = Vec3::ZERO;
            for &i in members {
                let t = triangles[i];
                area += normal(&t);
                for k in 0..3 {
                    let (a, b) = (t[k], t[(k + 1) % 3]);
                    if !members.contains(&owner[&(b, a)]) {
                        next.insert(a, b);
                    }
                }
            }
            let start = *next.keys().min()?;
            let first = loops.len();
            let mut at = start;
            loop {
                let index = *remap.entry(at).or_insert_with(|| {
                    vertices.push(points[at as usize]);
                    (vertices.len() - 1) as u8
                });
                loops.push(index);
                at = *next.get(&at)?;
                if at == start || loops.len() - first > next.len() {
                    break;
                }
            }
            let n = area.normalize();
            let offset = (loops[first..].iter())
                .map(|&v| n.dot(vertices[v as usize]))
                .fold(f32::NEG_INFINITY, f32::max);
            planes.push(Plane { normal: n, offset });
            faces.push(Face {
                first: first as u16,
                count: (loops.len() - first) as u16,
            });
        }
        let edges = edges(&faces, &loops, &planes)?;
        Some(Self {
            vertices,
            planes,
            faces,
            loops,
            edges,
            center: Vec3::ZERO,
            volume: 0.0,
            inertia: Mat3::ZERO,
            radius: 0.0,
            inner: 0.0,
            bounds: Bounds::EMPTY,
        })
    }

    /// Move the centroid to the origin and find the mass properties.
    fn recenter(&mut self) {
        let fan = |hull: &Self, mut each: Box<dyn FnMut(Vec3, Vec3, Vec3) + '_>| {
            for face in 0..hull.faces.len() {
                let corners: Vec<Vec3> = hull.poly().face(face).collect();
                for k in 1..corners.len() - 1 {
                    each(corners[0], corners[k], corners[k + 1]);
                }
            }
        };
        let inside = self.vertices.iter().sum::<Vec3>() / self.vertices.len() as f32;
        let (mut volume, mut moment) = (0.0, Vec3::ZERO);
        fan(
            self,
            Box::new(|a, b, c| {
                let v = (a - inside).dot((b - inside).cross(c - inside)) / 6.0;
                volume += v;
                moment += v * (inside + a + b + c) / 4.0;
            }),
        );
        let center = moment / volume;
        for v in &mut self.vertices {
            *v -= center;
        }
        for plane in &mut self.planes {
            plane.offset -= plane.normal.dot(center);
        }
        // Covariance of tetrahedra from the centroid (Blow and Binstock 2004).
        let canonical = Mat3::from_cols_array(&[2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0]) / 120.0;
        let mut covariance = Mat3::ZERO;
        fan(
            self,
            Box::new(|a, b, c| {
                let m = Mat3::from_cols(a, b, c);
                covariance += (m * canonical * m.transpose()) * m.determinant();
            }),
        );
        let trace = covariance.x_axis.x + covariance.y_axis.y + covariance.z_axis.z;
        self.inertia = Mat3::from_diagonal(Vec3::splat(trace)) - covariance;
        self.center = center;
        self.volume = volume;
        self.radius = self.vertices.iter().map(|v| v.length()).fold(0.0, f32::max);
        self.inner = self.planes.iter().map(|p| p.offset).fold(f32::INFINITY, f32::min);
        self.bounds = (self.vertices.iter()).fold(Bounds::EMPTY, |b, &v| Bounds {
            min: b.min.min(v),
            max: b.max.max(v),
        });
    }
}

/// Each edge once, with the normals of the faces it runs between.
fn edges(faces: &[Face], loops: &[u8], planes: &[Plane]) -> Option<Vec<Edge>> {
    let mut by_direction: HashMap<(u8, u8), usize> = HashMap::new();
    for (f, face) in faces.iter().enumerate() {
        let run = &loops[face.first as usize..(face.first + face.count) as usize];
        for k in 0..run.len() {
            by_direction.insert((run[k], run[(k + 1) % run.len()]), f);
        }
    }
    let mut edges: Vec<Edge> = (by_direction.iter())
        .filter(|((a, b), _)| a < b)
        .map(|(&(a, b), &f)| {
            let back = *by_direction.get(&(b, a))?;
            Some(Edge {
                a,
                b,
                normals: [planes[f].normal, planes[back].normal],
            })
        })
        .collect::<Option<_>>()?;
    edges.sort_by_key(|e| (e.a, e.b));
    Some(edges)
}

struct Triangle {
    v: [u32; 3],
    plane: Plane,
    outside: Vec<u32>,
    /// The farthest outside point and its distance.
    far: (u32, f32),
    alive: bool,
}

impl Triangle {
    fn new(points: &[Vec3], v: [u32; 3]) -> Self {
        let [a, b, c] = v.map(|i| points[i as usize]);
        let normal = (b - a).cross(c - a).normalize_or_zero();
        Self {
            v,
            plane: Plane {
                normal,
                offset: normal.dot(a),
            },
            outside: Vec::new(),
            far: (0, 0.0),
            alive: true,
        }
    }

    fn take(&mut self, point: u32, distance: f32) {
        self.outside.push(point);
        if distance > self.far.1 {
            self.far = (point, distance);
        }
    }
}

/// Give each point to the triangle it lies farthest outside, if any.
fn assign(points: &[Vec3], candidates: &[u32], triangles: &mut [Triangle], new: &[usize], eps: f32) {
    for &p in candidates {
        let best = (new.iter())
            .map(|&t| (t, triangles[t].plane.distance(points[p as usize])))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((t, d)) = best
            && d > eps
        {
            triangles[t].take(p, d);
        }
    }
}

/// Outward-wound triangles of the hull of `points`, with at most
/// `max_vertices` vertices added.
fn quickhull(points: &[Vec3], max_vertices: usize) -> Option<Vec<[u32; 3]>> {
    if points.len() < 4 {
        return None;
    }
    let (lo, hi) = (points.iter()).fold((Vec3::INFINITY, Vec3::NEG_INFINITY), |(lo, hi), &p| {
        (lo.min(p), hi.max(p))
    });
    let eps = 1.0e-5 * (hi - lo).max_element().max(hi.abs().max(lo.abs()).max_element());
    let at = |i: u32| points[i as usize];
    // Extreme points along the axes, then the widest pair among them.
    let mut extremes = [0u32; 6];
    for axis in 0..3 {
        let by = |a: &&Vec3, b: &&Vec3| a[axis].total_cmp(&b[axis]);
        extremes[2 * axis] = points.iter().enumerate().min_by(|a, b| by(&a.1, &b.1))?.0 as u32;
        extremes[2 * axis + 1] = points.iter().enumerate().max_by(|a, b| by(&a.1, &b.1))?.0 as u32;
    }
    let (mut i0, mut i1, mut widest) = (0, 0, -1.0);
    for &a in &extremes {
        for &b in &extremes {
            let d = at(a).distance_squared(at(b));
            if d > widest {
                (i0, i1, widest) = (a, b, d);
            }
        }
    }
    let all = 0..points.len() as u32;
    let line = (at(i1) - at(i0)).normalize_or_zero();
    let off_line = |i: u32| {
        let d = at(i) - at(i0);
        (d - line * d.dot(line)).length()
    };
    let i2 = all.clone().max_by(|&a, &b| off_line(a).total_cmp(&off_line(b)))?;
    let base = (at(i1) - at(i0)).cross(at(i2) - at(i0)).normalize_or_zero();
    let off_plane = |i: u32| base.dot(at(i) - at(i0)).abs();
    let i3 = all.clone().max_by(|&a, &b| off_plane(a).total_cmp(&off_plane(b)))?;
    if off_line(i2) <= eps || off_plane(i3) <= eps {
        return None;
    }
    let inside = (at(i0) + at(i1) + at(i2) + at(i3)) / 4.0;
    let mut triangles: Vec<Triangle> = [[i0, i1, i2], [i0, i3, i1], [i1, i3, i2], [i2, i3, i0]]
        .into_iter()
        .map(|v| {
            let t = Triangle::new(points, v);
            if t.plane.distance(inside) > 0.0 {
                Triangle::new(points, [v[0], v[2], v[1]])
            } else {
                t
            }
        })
        .collect();
    let rest: Vec<u32> = all.filter(|i| ![i0, i1, i2, i3].contains(i)).collect();
    assign(points, &rest, &mut triangles, &[0, 1, 2, 3], eps);
    let mut count = 4;
    while count < max_vertices {
        let Some(t) = (0..triangles.len())
            .filter(|&t| triangles[t].alive && !triangles[t].outside.is_empty())
            .max_by(|&a, &b| triangles[a].far.1.total_cmp(&triangles[b].far.1))
        else {
            break;
        };
        let eye = triangles[t].far.0;
        let visible: Vec<usize> = (0..triangles.len())
            .filter(|&t| triangles[t].alive && triangles[t].plane.distance(at(eye)) > eps)
            .collect();
        let mut directed = std::collections::HashSet::new();
        for &t in &visible {
            let v = triangles[t].v;
            for k in 0..3 {
                directed.insert((v[k], v[(k + 1) % 3]));
            }
        }
        let mut orphans = Vec::new();
        let mut horizon = Vec::new();
        for &t in &visible {
            let v = triangles[t].v;
            for k in 0..3 {
                let (a, b) = (v[k], v[(k + 1) % 3]);
                if !directed.contains(&(b, a)) {
                    horizon.push((a, b));
                }
            }
            triangles[t].alive = false;
            orphans.extend(std::mem::take(&mut triangles[t].outside));
        }
        let new: Vec<usize> = (horizon.iter())
            .map(|&(a, b)| {
                triangles.push(Triangle::new(points, [a, b, eye]));
                triangles.len() - 1
            })
            .collect();
        orphans.retain(|&p| p != eye);
        assign(points, &orphans, &mut triangles, &new, eps);
        count += 1;
    }
    Some(triangles.iter().filter(|t| t.alive).map(|t| t.v).collect())
}

/// Closest points of the hulls of two point sets (GJK, with Ericson's
/// closest-point-on-simplex), and the distance between them: zero when they
/// overlap.
pub(crate) fn gjk(a: &[Vec3], b: &[Vec3]) -> (Vec3, Vec3, f32) {
    // Simplex vertices: the Minkowski point and the points it came from.
    let mut simplex: [(Vec3, Vec3, Vec3); 4] = [(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO); 4];
    let mut weights = [1.0f32, 0.0, 0.0, 0.0];
    simplex[0] = (a[0] - b[0], a[0], b[0]);
    let mut len = 1;
    let mut v = simplex[0].0;
    let points = |simplex: &[(Vec3, Vec3, Vec3)], weights: &[f32]| {
        (simplex.iter().zip(weights)).fold((Vec3::ZERO, Vec3::ZERO), |(pa, pb), (s, w)| {
            (pa + s.1 * *w, pb + s.2 * *w)
        })
    };
    for _ in 0..32 {
        let vv = v.length_squared();
        if vv < 1.0e-12 {
            let (pa, pb) = points(&simplex[..len], &weights[..len]);
            return (pa, pb, 0.0);
        }
        let (sa, sb) = (a[support(a, -v)], b[support(b, v)]);
        let w = sa - sb;
        // No closer point along -v, or a repeat: converged.
        if vv - v.dot(w) <= 1.0e-6 * vv || simplex[..len].iter().any(|s| s.0 == w) {
            break;
        }
        simplex[len] = (w, sa, sb);
        len += 1;
        let (reduced, count, closest) = nearest_on_simplex(&simplex[..len]);
        if count == 4 {
            let (pa, pb) = points(&simplex[..len], &reduced);
            return (pa, pb, 0.0);
        }
        // Keep only the vertices the closest point uses.
        let mut kept = 0;
        for i in 0..len {
            if reduced[i] > 0.0 {
                simplex[kept] = simplex[i];
                weights[kept] = reduced[i];
                kept += 1;
            }
        }
        len = kept;
        v = closest;
    }
    let (pa, pb) = points(&simplex[..len], &weights[..len]);
    (pa, pb, (pa - pb).length())
}

/// Barycentric weights of the point of a simplex nearest the origin, how many
/// vertices it has (4 means the origin is inside), and the point.
fn nearest_on_simplex(s: &[(Vec3, Vec3, Vec3)]) -> ([f32; 4], usize, Vec3) {
    let w = |i: usize| s[i].0;
    match s.len() {
        1 => ([1.0, 0.0, 0.0, 0.0], 1, w(0)),
        2 => {
            let (a, b) = (w(0), w(1));
            let ab = b - a;
            let t = (-a.dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
            let t = if t.is_finite() { t } else { 0.0 };
            ([1.0 - t, t, 0.0, 0.0], 2, a + ab * t)
        }
        3 => {
            let (u, p) = triangle_nearest(Vec3::ZERO, w(0), w(1), w(2));
            ([u[0], u[1], u[2], 0.0], 3, p)
        }
        _ => {
            let [a, b, c, d] = [w(0), w(1), w(2), w(3)];
            // Faces with the origin on their far side from the fourth vertex.
            let faces = [([0, 1, 2], d), ([0, 1, 3], c), ([0, 2, 3], b), ([1, 2, 3], a)];
            let mut best: Option<([f32; 4], f32, Vec3)> = None;
            for (f, opposite) in faces {
                let [x, y, z] = f.map(w);
                let n = (y - x).cross(z - x);
                let (o, q) = (n.dot(-x), n.dot(opposite - x));
                if o * q >= 0.0 && q != 0.0 {
                    continue;
                }
                let (u, p) = triangle_nearest(Vec3::ZERO, x, y, z);
                let d = p.length_squared();
                if best.is_none_or(|b| d < b.1) {
                    let mut weights = [0.0; 4];
                    for (k, &i) in f.iter().enumerate() {
                        weights[i] = u[k];
                    }
                    best = Some((weights, d, p));
                }
            }
            match best {
                Some((weights, _, p)) => (weights, 3, p),
                None => ([0.25; 4], 4, Vec3::ZERO),
            }
        }
    }
}

/// Barycentric weights of the point of triangle `abc` nearest `p`, and the
/// point (Ericson, Real-Time Collision Detection 5.1.5).
pub(crate) fn triangle_nearest(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> ([f32; 3], Vec3) {
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return ([1.0, 0.0, 0.0], a);
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return ([0.0, 1.0, 0.0], b);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return ([1.0 - v, v, 0.0], a + ab * v);
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return ([0.0, 0.0, 1.0], c);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return ([1.0 - w, 0.0, w], a + ac * w);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return ([0.0, 1.0 - w, w], b + (c - b) * w);
    }
    let denom = va + vb + vc;
    if denom.abs() < 1.0e-30 {
        // Degenerate: the nearest corner.
        let corners = [a, b, c];
        let k = (0..3)
            .min_by(|&i, &j| p.distance_squared(corners[i]).total_cmp(&p.distance_squared(corners[j])))
            .unwrap();
        let mut u = [0.0; 3];
        u[k] = 1.0;
        return (u, corners[k]);
    }
    let (v, w) = (vb / denom, vc / denom);
    ([1.0 - v - w, v, w], a + ab * v + ac * w)
}

/// A box as a polytope: the unit cube's topology with these half extents.
pub(crate) struct BoxPoly {
    vertices: [Vec3; 8],
    planes: [Plane; 6],
}

static UNIT_BOX: std::sync::LazyLock<ConvexHull> = std::sync::LazyLock::new(|| {
    let corners: Vec<Vec3> = (0..8)
        .map(|i| Vec3::new(sign(i, 0), sign(i, 1), sign(i, 2)))
        .collect();
    ConvexHull::new(&corners).expect("a cube has a hull")
});

fn sign(bits: usize, axis: usize) -> f32 {
    if bits >> axis & 1 == 0 { -1.0 } else { 1.0 }
}

impl BoxPoly {
    pub fn new(half: Vec3) -> Self {
        let unit = &*UNIT_BOX;
        Self {
            vertices: std::array::from_fn(|i| unit.vertices[i] * half),
            planes: std::array::from_fn(|i| {
                let normal = unit.planes[i].normal;
                Plane {
                    normal,
                    offset: normal.abs().dot(half),
                }
            }),
        }
    }

    pub fn poly(&self) -> Poly<'_> {
        let unit = &*UNIT_BOX;
        Poly {
            vertices: &self.vertices,
            planes: &self.planes,
            faces: &unit.faces,
            loops: &unit.loops,
            edges: &unit.edges,
        }
    }
}

/// A triangle as a polytope: two faces back to back, and each edge's Gauss
/// arc split at its outward in-plane normal, so edge pruning sees a half circle.
pub(crate) struct TrianglePoly {
    vertices: [Vec3; 3],
    planes: [Plane; 2],
    edges: [Edge; 6],
}

const TRIANGLE_FACES: [Face; 2] = [Face { first: 0, count: 3 }, Face { first: 3, count: 3 }];
const TRIANGLE_LOOPS: [u8; 6] = [0, 1, 2, 2, 1, 0];

impl TrianglePoly {
    /// `None` for a triangle without area.
    pub fn new([a, b, c]: [Vec3; 3]) -> Option<Self> {
        let n = (b - a).cross(c - a).try_normalize()?;
        let vertices = [a, b, c];
        let edges = std::array::from_fn(|i| {
            let k = i / 2;
            let (p, q) = (vertices[k], vertices[(k + 1) % 3]);
            let out = (q - p).cross(n).normalize_or_zero();
            let normals = if i % 2 == 0 { [n, out] } else { [out, -n] };
            Edge {
                a: k as u8,
                b: ((k + 1) % 3) as u8,
                normals,
            }
        });
        Some(Self {
            vertices,
            planes: [
                Plane {
                    normal: n,
                    offset: n.dot(a),
                },
                Plane {
                    normal: -n,
                    offset: -n.dot(a),
                },
            ],
            edges,
        })
    }

    pub fn normal(&self) -> Vec3 {
        self.planes[0].normal
    }

    pub fn poly(&self) -> Poly<'_> {
        Poly {
            vertices: &self.vertices,
            planes: &self.planes,
            faces: &TRIANGLE_FACES,
            loops: &TRIANGLE_LOOPS,
            edges: &self.edges,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_hull_has_six_quads_and_exact_mass() {
        let hull = &*UNIT_BOX;
        assert_eq!(hull.vertices().len(), 8);
        assert_eq!(hull.face_count(), 6);
        assert!(hull.faces.iter().all(|f| f.count == 4));
        assert_eq!(hull.edges.len(), 12);
        assert!((hull.volume() - 8.0).abs() < 1.0e-4);
        // A 2 m cube of unit density: 8 kg, moment m(h² + h²)/3 = 16/3.
        let (mass, inertia) = hull.mass_properties(1.0);
        assert!((mass - 8.0).abs() < 1.0e-4);
        let expected = Mat3::from_diagonal(Vec3::splat(16.0 / 3.0));
        assert!(inertia.abs_diff_eq(expected, 1.0e-3), "{inertia:?}");
    }

    #[test]
    fn hull_ignores_interior_points_and_recenters() {
        let mut points: Vec<Vec3> = (0..8)
            .map(|i| Vec3::new(sign(i, 0), sign(i, 1), sign(i, 2)) + Vec3::new(5.0, 0.0, 1.0))
            .collect();
        points.extend((0..50).map(|i| Vec3::new(5.0, 0.0, 1.0) + Vec3::splat(i as f32 / 60.0 - 0.4)));
        let hull = ConvexHull::new(&points).unwrap();
        assert_eq!(hull.vertices().len(), 8);
        assert!(hull.center().abs_diff_eq(Vec3::new(5.0, 0.0, 1.0), 1.0e-4));
        assert!(hull.error(&points) < 1.0e-4);
    }

    #[test]
    fn sphere_points_keep_the_vertex_budget_and_stay_close() {
        let points: Vec<Vec3> = (0..2000)
            .map(|i| {
                // A Fibonacci sphere.
                let y = 1.0 - 2.0 * (i as f32 + 0.5) / 2000.0;
                let r = (1.0 - y * y).sqrt();
                let a = i as f32 * 2.399_963;
                Vec3::new(r * a.cos(), y, r * a.sin())
            })
            .collect();
        let hull = ConvexHull::with_max_vertices(&points, 32).unwrap();
        assert!(hull.vertices().len() <= 32);
        // Euler: a closed hull's faces, edges and vertices.
        assert_eq!(
            hull.vertices().len() + hull.face_count(),
            hull.edges.len() + 2
        );
        assert!(hull.error(&points) < 0.25, "{}", hull.error(&points));
        let full = ConvexHull::new(&points).unwrap();
        assert!(full.error(&points) < hull.error(&points));
        let sphere = 4.0 / 3.0 * std::f32::consts::PI;
        assert!(full.volume() < sphere && full.volume() > 0.9 * sphere);
    }

    #[test]
    fn flat_points_have_no_hull() {
        let points = [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::new(1.0, 1.0, 0.0)];
        assert!(ConvexHull::new(&points).is_none());
    }

    #[test]
    fn gjk_finds_distance_between_a_point_and_a_box() {
        let cube = &*UNIT_BOX;
        let (p, q, d) = gjk(&[Vec3::new(3.0, 0.5, 0.0)], cube.vertices());
        assert!((d - 2.0).abs() < 1.0e-4, "{d}");
        assert!(p.abs_diff_eq(Vec3::new(3.0, 0.5, 0.0), 1.0e-5));
        assert!(q.abs_diff_eq(Vec3::new(1.0, 0.5, 0.0), 1.0e-4), "{q}");
        let (_, _, inside) = gjk(&[Vec3::new(0.2, 0.1, 0.0)], cube.vertices());
        assert_eq!(inside, 0.0);
        // Segment past an edge.
        let (_, _, d) = gjk(&[Vec3::new(2.0, 2.0, -5.0), Vec3::new(2.0, 2.0, 5.0)], cube.vertices());
        assert!((d - 2.0f32.sqrt()).abs() < 1.0e-4, "{d}");
    }

    #[test]
    fn ray_enters_through_the_facing_face() {
        let cube = &*UNIT_BOX;
        let (t, n) = cube.ray(Vec3::new(-5.0, 0.2, 0.3), Vec3::X, 100.0).unwrap();
        assert!((t - 4.0).abs() < 1.0e-5);
        assert_eq!(n, -Vec3::X);
        assert!(cube.ray(Vec3::new(-5.0, 3.0, 0.0), Vec3::X, 100.0).is_none());
    }
}
