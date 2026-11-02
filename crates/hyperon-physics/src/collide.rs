//! Contact generation for cuboids and spheres. Cuboid pairs use the separating
//! axis test with reference-face clipping, ported from the AVBD 3D demo.

use crate::Collider;
use crate::solver::{Body, Contact};
use glam::{Mat3, Vec3};

/// A face manifold keeps four points spanning the largest area, enough for a
/// stable support polygon at half the solver work of the full clip.
pub(crate) const MAX_CONTACTS: usize = 4;
const MAX_POLY: usize = 16;
const AXIS_EPSILON: f32 = 1.0e-6;
const PLANE_EPSILON: f32 = 1.0e-5;
const MERGE_DISTANCE_SQ: f32 = 1.0e-6;

const FACE_A: u32 = 0;
const FACE_B: u32 = 1;
const EDGE: u32 = 2;

/// Contact basis (normal from B to A, then tangents) and contacts with anchors
/// in each body's frame; `None` when the shapes don't touch.
pub(crate) fn collide(a: &Body, b: &Body) -> Option<([Vec3; 3], [Contact; MAX_CONTACTS], usize)> {
    let mut points = Points::default();
    let normal = match (a.shape, b.shape) {
        (Collider::Sphere { radius: ra }, Collider::Sphere { radius: rb }) => {
            let d = a.position - b.position;
            let distance = d.length();
            if distance > ra + rb {
                return None;
            }
            let n = if distance > AXIS_EPSILON {
                d / distance
            } else {
                Vec3::Z
            };
            points.push(a.position - n * ra, b.position + n * rb, 0);
            n
        }
        (Collider::Sphere { radius }, Collider::Cuboid { half_extents }) => {
            let (n, xa, xb) = sphere_box(a.position, radius, &obb(b, half_extents))?;
            points.push(xa, xb, 0);
            n
        }
        (Collider::Cuboid { half_extents }, Collider::Sphere { radius }) => {
            let (n, xb, xa) = sphere_box(b.position, radius, &obb(a, half_extents))?;
            points.push(xa, xb, 0);
            -n
        }
        (Collider::Cuboid { half_extents: ha }, Collider::Cuboid { half_extents: hb }) => {
            -box_box(&obb(a, ha), &obb(b, hb), &mut points)?
        }
    };
    let (t1, t2) = normal.any_orthonormal_pair();
    let (ia, ib) = (a.rotation.conjugate(), b.rotation.conjugate());
    let contacts = std::array::from_fn(|i| {
        let (xa, xb, feature) = points.p[i];
        Contact::new(feature, ia * (xa - a.position), ib * (xb - b.position))
    });
    Some(([normal, t1, t2], contacts, points.len))
}

#[derive(Default)]
struct Points {
    p: [(Vec3, Vec3, u32); MAX_CONTACTS],
    len: usize,
}

impl Points {
    /// Add a contact unless it duplicates one already found.
    fn push(&mut self, xa: Vec3, xb: Vec3, feature: u32) {
        let mid = (xa + xb) * 0.5;
        if self.len == MAX_CONTACTS
            || self.p[..self.len]
                .iter()
                .any(|&(a, b, _)| ((a + b) * 0.5).distance_squared(mid) < MERGE_DISTANCE_SQ)
        {
            return;
        }
        self.p[self.len] = (xa, xb, feature);
        self.len += 1;
    }
}

struct Obb {
    center: Vec3,
    half: Vec3,
    axis: [Vec3; 3],
}

fn obb(body: &Body, half: Vec3) -> Obb {
    let m = Mat3::from_quat(body.rotation);
    Obb {
        center: body.position,
        half,
        axis: [m.x_axis, m.y_axis, m.z_axis],
    }
}

fn signum(x: f32) -> f32 {
    if x >= 0.0 { 1.0 } else { -1.0 }
}

impl Obb {
    fn support(&self, dir: Vec3) -> Vec3 {
        (0..3).fold(self.center, |p, i| {
            p + self.axis[i] * (self.half[i] * signum(dir.dot(self.axis[i])))
        })
    }

    /// Face tangents and their half extents for the face normal to `axis`.
    fn face_axes(&self, axis: usize) -> (Vec3, Vec3, f32, f32) {
        let (u, v) = match axis {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        (self.axis[u], self.axis[v], self.half[u], self.half[v])
    }

    /// Projected radius onto unit `n`.
    fn radius(&self, n: Vec3) -> f32 {
        (0..3)
            .map(|i| self.half[i] * n.dot(self.axis[i]).abs())
            .sum()
    }
}

/// Normal from box to sphere, the sphere's deepest point, and the box surface point.
fn sphere_box(center: Vec3, radius: f32, b: &Obb) -> Option<(Vec3, Vec3, Vec3)> {
    let d = center - b.center;
    let local = Vec3::new(d.dot(b.axis[0]), d.dot(b.axis[1]), d.dot(b.axis[2]));
    let clamped = local.clamp(-b.half, b.half);
    let (n_local, surface) = if clamped != local {
        let offset = local - clamped;
        let distance = offset.length();
        if distance > radius {
            return None;
        }
        (offset / distance, clamped)
    } else {
        // Center inside: push out through the nearest face.
        let gap = b.half - local.abs();
        let k = if gap.x <= gap.y && gap.x <= gap.z {
            0
        } else if gap.y <= gap.z {
            1
        } else {
            2
        };
        let mut n = Vec3::ZERO;
        n[k] = signum(local[k]);
        let mut surface = local;
        surface[k] = n[k] * b.half[k];
        (n, surface)
    };
    let world = |v: Vec3| b.axis[0] * v.x + b.axis[1] * v.y + b.axis[2] * v.z;
    let n = world(n_local);
    Some((n, center - n * radius, b.center + world(surface)))
}

struct Axis {
    kind: u32,
    a: usize,
    b: usize,
    separation: f32,
    /// Unit normal pointing from A toward B.
    normal: Vec3,
}

/// Returns false when `axis` separates the boxes.
#[allow(clippy::too_many_arguments)]
fn test_axis(
    a: &Obb,
    b: &Obb,
    delta: Vec3,
    axis: Vec3,
    kind: u32,
    ia: usize,
    ib: usize,
    best: &mut Option<Axis>,
) -> bool {
    let length_sq = axis.length_squared();
    if length_sq < AXIS_EPSILON {
        return true;
    }
    let mut n = axis / length_sq.sqrt();
    if n.dot(delta) < 0.0 {
        n = -n;
    }
    let separation = delta.dot(n).abs() - (a.radius(n) + b.radius(n));
    if separation > 0.0 {
        return false;
    }
    if best
        .as_ref()
        .is_none_or(|best| separation > best.separation)
    {
        *best = Some(Axis {
            kind,
            a: ia,
            b: ib,
            separation,
            normal: n,
        });
    }
    true
}

/// Fill `points` and return the normal from A to B.
fn box_box(a: &Obb, b: &Obb, points: &mut Points) -> Option<Vec3> {
    let delta = b.center - a.center;
    let (mut face, mut edge) = (None, None);
    for i in 0..3 {
        if !test_axis(a, b, delta, a.axis[i], FACE_A, i, 0, &mut face)
            || !test_axis(a, b, delta, b.axis[i], FACE_B, 0, i, &mut face)
        {
            return None;
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            if !test_axis(
                a,
                b,
                delta,
                a.axis[i].cross(b.axis[j]),
                EDGE,
                i,
                j,
                &mut edge,
            ) {
                return None;
            }
        }
    }
    let face = face?;
    // Prefer faces unless an edge axis is clearly better: face manifolds are stable.
    let best = match edge {
        Some(edge) if 0.95 * edge.separation > face.separation + 0.01 => edge,
        _ => face,
    };
    match best.kind {
        EDGE => edge_contact(a, b, &best, points),
        FACE_A => face_contacts(a, b, true, best.a, best.normal, points),
        _ => face_contacts(a, b, false, best.b, best.normal, points),
    }
    if points.len == 0 {
        points.push(
            a.support(best.normal),
            b.support(-best.normal),
            best.kind << 24,
        );
    }
    Some(best.normal)
}

/// Clip the incident face against the reference face's side planes.
fn face_contacts(
    a: &Obb,
    b: &Obb,
    reference_is_a: bool,
    axis: usize,
    normal_ab: Vec3,
    points: &mut Points,
) {
    let (reference, incident) = if reference_is_a { (a, b) } else { (b, a) };
    let outward = if reference_is_a {
        normal_ab
    } else {
        -normal_ab
    };
    let normal = reference.axis[axis] * signum(outward.dot(reference.axis[axis]));
    let center = reference.center + normal * reference.half[axis];
    let (u, v, eu, ev) = reference.face_axes(axis);

    let incident_axis = (0..3)
        .max_by(|&i, &j| {
            incident.axis[i]
                .dot(normal)
                .abs()
                .total_cmp(&incident.axis[j].dot(normal).abs())
        })
        .unwrap();
    let facing = incident.axis[incident_axis].dot(normal) > 0.0;
    let face_normal = incident.axis[incident_axis] * if facing { -1.0 } else { 1.0 };
    let face_center = incident.center + face_normal * incident.half[incident_axis];
    let (iu, iv, ieu, iev) = incident.face_axes(incident_axis);
    let mut polygon = [Vec3::ZERO; MAX_POLY];
    polygon[..4].copy_from_slice(&[
        face_center + iu * ieu + iv * iev,
        face_center - iu * ieu + iv * iev,
        face_center - iu * ieu - iv * iev,
        face_center + iu * ieu - iv * iev,
    ]);
    let mut count = 4;
    for (plane, extent) in [(u, eu), (-u, eu), (v, ev), (-v, ev)] {
        count = clip(&mut polygon, count, plane, plane.dot(center) + extent);
        if count == 0 {
            return;
        }
    }

    let kind = if reference_is_a { FACE_A } else { FACE_B };
    let prefix = (kind << 24) | ((axis as u32) << 16) | ((incident_axis as u32) << 8);
    // Penetrating points: polygon index and depth below the reference face.
    let mut found = [(0usize, 0.0f32); MAX_POLY];
    let mut n = 0;
    for (i, &p) in polygon[..count].iter().enumerate() {
        let distance = (p - center).dot(normal);
        if distance <= PLANE_EPSILON {
            found[n] = (i, distance);
            n += 1;
        }
    }
    let (picked, len) = reduce(&polygon, &found[..n], normal);
    for &i in &picked[..len] {
        let (index, distance) = found[i];
        let p = polygon[index];
        let on_reference = p - normal * distance;
        let (xa, xb) = if reference_is_a {
            (on_reference, p)
        } else {
            (p, on_reference)
        };
        points.push(xa, xb, prefix | (index as u32 & 0xFF));
    }
}

/// Up to four of `found` (indices into `polygon`): the deepest, the farthest
/// from it, then the two spanning the most area on either side of that line.
fn reduce(
    polygon: &[Vec3; MAX_POLY],
    found: &[(usize, f32)],
    normal: Vec3,
) -> ([usize; MAX_CONTACTS], usize) {
    if found.len() <= MAX_CONTACTS {
        return (std::array::from_fn(|i| i), found.len());
    }
    let at = |i: usize| polygon[found[i].0];
    let first = (0..found.len())
        .min_by(|&i, &j| found[i].1.total_cmp(&found[j].1))
        .unwrap();
    let second = (0..found.len())
        .max_by(|&i, &j| at(i).distance_squared(at(first)).total_cmp(&at(j).distance_squared(at(first))))
        .unwrap();
    let area = |i: usize| (at(second) - at(first)).cross(at(i) - at(first)).dot(normal);
    let third = (0..found.len()).max_by(|&i, &j| area(i).total_cmp(&area(j))).unwrap();
    let fourth = (0..found.len()).min_by(|&i, &j| area(i).total_cmp(&area(j))).unwrap();
    let mut picked = [first, second, 0, 0];
    let mut len = 2;
    for i in [third, fourth] {
        if !picked[..len].contains(&i) {
            picked[len] = i;
            len += 1;
        }
    }
    (picked, len)
}

/// Sutherland-Hodgman: keep the part of the polygon with `n·p <= offset`.
fn clip(polygon: &mut [Vec3; MAX_POLY], count: usize, n: Vec3, offset: f32) -> usize {
    let input = *polygon;
    let mut out = 0;
    let mut a = input[count - 1];
    let mut da = n.dot(a) - offset;
    for &b in &input[..count] {
        let db = n.dot(b) - offset;
        let (a_in, b_in) = (da <= PLANE_EPSILON, db <= PLANE_EPSILON);
        if a_in != b_in && out < MAX_POLY {
            let denom = da - db;
            let t = if denom.abs() > AXIS_EPSILON {
                (da / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            polygon[out] = a + (b - a) * t;
            out += 1;
        }
        if b_in && out < MAX_POLY {
            polygon[out] = b;
            out += 1;
        }
        (a, da) = (b, db);
    }
    out
}

fn edge_contact(a: &Obb, b: &Obb, axis: &Axis, points: &mut Points) {
    let (a0, a1) = support_edge(a, axis.a, axis.normal);
    let (b0, b1) = support_edge(b, axis.b, -axis.normal);
    let (xa, xb) = closest_on_segments(a0, a1, b0, b1);
    points.push(
        xa,
        xb,
        (EDGE << 24) | ((axis.a as u32) << 8) | axis.b as u32,
    );
}

fn support_edge(o: &Obb, axis: usize, dir: Vec3) -> (Vec3, Vec3) {
    let (i, j) = ((axis + 1) % 3, (axis + 2) % 3);
    let center = o.center
        + o.axis[i] * (o.half[i] * signum(dir.dot(o.axis[i])))
        + o.axis[j] * (o.half[j] * signum(dir.dot(o.axis[j])));
    let along = o.axis[axis] * o.half[axis];
    (center - along, center + along)
}

fn closest_on_segments(p0: Vec3, p1: Vec3, q0: Vec3, q1: Vec3) -> (Vec3, Vec3) {
    let (d1, d2, r) = (p1 - p0, q1 - q0, p0 - q0);
    let (a, e, f) = (d1.dot(d1), d2.dot(d2), d2.dot(r));
    if a <= AXIS_EPSILON && e <= AXIS_EPSILON {
        return (p0, q0);
    }
    let (mut s, mut t) = (0.0, 0.0);
    if a <= AXIS_EPSILON {
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e <= AXIS_EPSILON {
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b;
            if denom.abs() > AXIS_EPSILON {
                s = ((b * f - c * e) / denom).clamp(0.0, 1.0);
            }
            t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
        }
    }
    (p0 + d1 * s, q0 + d2 * t)
}
