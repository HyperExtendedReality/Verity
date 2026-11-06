//! Contact generation for cuboids, spheres and ground. Cuboid pairs use the
//! separating axis test with reference-face clipping, ported from the AVBD 3D
//! demo. Ground is a field, so a body asks it about points on its own surface.

use crate::solver::{Body, CONTACT_DISTANCE, Contact};
use crate::{Collider, Ground};
use glam::{Quat, Vec3};

/// A face manifold keeps four points spanning the largest area, enough for a
/// stable support polygon at half the solver work of the full clip.
pub(crate) const MAX_CONTACTS: usize = 4;
const MAX_POLY: usize = 16;
const AXIS_EPSILON: f32 = 1.0e-6;
const PLANE_EPSILON: f32 = 1.0e-5;
const MERGE_DISTANCE_SQ: f32 = 1.0e-6;
/// Most poses tested along a body's path in one step.
const MAX_SAMPLES: usize = 16;
/// Bisections of a hit found between two of those poses.
const REFINEMENTS: usize = 3;

const FACE_A: u32 = 0;
const FACE_B: u32 = 1;
const EDGE: u32 = 2;

/// The widest gap between points tested on a cuboid against ground, in metres:
/// a ridge narrower than this can reach into a face.
const GROUND_SPACING: f32 = 0.75;
/// Divisions of a cuboid's edge, at most. Longer edges test further apart.
const GROUND_DIVISIONS: usize = 4;
/// Metres across a cuboid worth giving up per metre of depth when choosing its
/// contacts: a bump a few centimetres proud of the rest gets one, noise does not.
const GROUND_DEPTH_WEIGHT: f32 = 25.0;
/// Most points on a cuboid's surface.
const GROUND_POINTS: usize = 98;
/// How far off the ground a cuboid's points still count as contacts. Contacts
/// with a gap do nothing until it closes, but a body on a bump then has the
/// corner it tips towards ready to stop it, instead of rocking between two sets.
const GROUND_SKIN: f32 = 0.05;

/// Whether A, moved by `shift`, comes within `margin` of B.
fn touches(a: &Body, b: &Body, shift: Vec3, margin: f32, ground: Option<&dyn Ground>) -> bool {
    let center = a.position + shift;
    match (a.shape, b.shape) {
        (Collider::Ground, Collider::Ground) => false,
        (Collider::Ground, _) => {
            ground.is_some_and(|g| body_touches(g, (center, a.rotation), b, Vec3::ZERO, margin))
        }
        (_, Collider::Ground) => {
            ground.is_some_and(|g| body_touches(g, (b.position, b.rotation), a, shift, margin))
        }
        (Collider::Sphere { radius: ra }, Collider::Sphere { radius: rb }) => {
            let reach = ra + rb + margin;
            center.distance_squared(b.position) <= reach * reach
        }
        (Collider::Sphere { radius }, Collider::Cuboid { half_extents }) => {
            sphere_box(center, radius, &obb(b, half_extents), margin).is_some()
        }
        (Collider::Cuboid { half_extents }, Collider::Sphere { radius }) => {
            let box_a = moved_obb(a, half_extents, shift);
            sphere_box(b.position, radius, &box_a, margin).is_some()
        }
        (Collider::Cuboid { half_extents: ha }, Collider::Cuboid { half_extents: hb }) => {
            box_axes(&moved_obb(a, ha, shift), &obb(b, hb), margin).is_some()
        }
    }
}

/// A's first pose this step that touches B, as a shift from now; `None` if it
/// passes clear. Contacts come from that pose, not this one: a contact plane
/// from here would stop a body flying past a corner, and catch a box sliding
/// over a seam on the far tile's side.
fn predicted_hit(a: &Body, b: &Body, margin: f32, ground: Option<&dyn Ground>) -> Option<Vec3> {
    let travel = a.travel - b.travel;
    if travel.length_squared() < CONTACT_DISTANCE * CONTACT_DISTANCE * 0.01 {
        return Some(Vec3::ZERO);
    }
    let thickness = |shape| match shape {
        Collider::Cuboid { half_extents } => half_extents.min_element(),
        Collider::Sphere { radius } => radius,
        Collider::Ground => 0.0,
    };
    let spacing = 0.5 * (thickness(a.shape) + thickness(b.shape));
    let samples = ((travel.length() / spacing).ceil() as usize).clamp(1, MAX_SAMPLES);
    if samples == 1 {
        // The contact generator makes this one test itself.
        return Some(travel);
    }
    let touching = |t: f32| touches(a, b, travel * t, margin, ground);
    let hit = (1..=samples).find(|&i| touching(i as f32 / samples as f32))?;
    // A deep first sample could lie past a thin wall's middle, which would
    // push the body through, so find first touch more closely.
    let (mut low, mut high) = (
        (hit - 1) as f32 / samples as f32,
        hit as f32 / samples as f32,
    );
    for _ in 0..REFINEMENTS {
        let middle = 0.5 * (low + high);
        if touching(middle) {
            high = middle;
        } else {
            low = middle;
        }
    }
    Some(travel * high)
}

/// Contact basis (normal from B to A, then tangents) and contacts with anchors
/// in each body's frame; `None` when the shapes do not meet this step. Shapes
/// about to meet get speculative contacts, whose gap stays in the constraint,
/// so a fast body stops at the surface, not inside it.
pub(crate) fn collide(
    a: &Body,
    b: &Body,
    ground: Option<&dyn Ground>,
) -> Option<([Vec3; 3], [Contact; MAX_CONTACTS], usize)> {
    let mut points = Points::default();
    let margin = CONTACT_DISTANCE + a.reach + b.reach;
    let shift = predicted_hit(a, b, margin, ground)?;
    let center = a.position + shift;
    let normal = match (a.shape, b.shape) {
        (Collider::Ground, Collider::Ground) => return None,
        (Collider::Ground, _) => {
            let frame = (center, a.rotation);
            -ground_contacts(ground?, frame, b, Vec3::ZERO, margin, true, &mut points)?
        }
        (_, Collider::Ground) => {
            let frame = (b.position, b.rotation);
            ground_contacts(ground?, frame, a, shift, margin, false, &mut points)?
        }
        (Collider::Sphere { radius: ra }, Collider::Sphere { radius: rb }) => {
            let d = center - b.position;
            let distance_squared = d.length_squared();
            let reach = ra + rb + margin;
            if distance_squared > reach * reach
                || (distance_squared == f32::INFINITY && reach.is_finite())
            {
                return None;
            }
            let distance = distance_squared.sqrt();
            let n = if distance > AXIS_EPSILON {
                d / distance
            } else {
                Vec3::Z
            };
            points.push(center - n * ra, b.position + n * rb, 0);
            n
        }
        (Collider::Sphere { radius }, Collider::Cuboid { half_extents }) => {
            let (n, xa, xb) = sphere_box(center, radius, &obb(b, half_extents), margin)?;
            points.push(xa, xb, 0);
            n
        }
        (Collider::Cuboid { half_extents }, Collider::Sphere { radius }) => {
            let box_a = moved_obb(a, half_extents, shift);
            let (n, xb, xa) = sphere_box(b.position, radius, &box_a, margin)?;
            points.push(xa, xb, 0);
            -n
        }
        (Collider::Cuboid { half_extents: ha }, Collider::Cuboid { half_extents: hb }) => {
            -box_box(&moved_obb(a, ha, shift), &obb(b, hb), margin, &mut points)?
        }
    };
    let (t1, t2) = normal.any_orthonormal_pair();
    let (ia, ib) = (a.rotation.conjugate(), b.rotation.conjugate());
    let mut contacts = [Contact::default(); MAX_CONTACTS];
    // Back to the real poses, leaving only the gap along the normal: sideways
    // would read as friction error. The lighter body's anchors move less, so
    // a rolling ball's stay under it and the floor's take the difference.
    let slide = shift - normal * normal.dot(shift);
    let total = a.mass + b.mass;
    let share = if total > 0.0 { b.mass / total } else { 0.5 };
    let (slide_a, slide_b) = (slide * share, slide * (1.0 - share));
    for (contact, &(xa, xb, feature)) in contacts.iter_mut().zip(&points.p[..points.len]) {
        let (ra, rb) = (xa - center + slide_a, xb - b.position - slide_b);
        *contact = Contact::new(feature, ia * ra, ib * rb);
    }
    Some(([normal, t1, t2], contacts, points.len))
}

/// Where a body's own points lie for a ground test, and which feature each is.
struct BodyPoints {
    points: [(Vec3, u32); GROUND_POINTS],
    len: usize,
}

/// Points on a cuboid's surface: a lattice over it, in the ground's frame.
fn cuboid_points(box_: &Obb, to_ground: impl Fn(Vec3) -> Vec3) -> BodyPoints {
    let divisions = box_
        .half
        .to_array()
        .map(|half| ((2.0 * half / GROUND_SPACING).ceil() as usize).clamp(2, GROUND_DIVISIONS));
    let mut out = BodyPoints {
        points: [(Vec3::ZERO, 0); GROUND_POINTS],
        len: 0,
    };
    let at = |i: usize, axis: usize| (2 * i) as f32 / divisions[axis] as f32 - 1.0;
    for x in 0..=divisions[0] {
        for y in 0..=divisions[1] {
            for z in 0..=divisions[2] {
                let outside = x == 0
                    || y == 0
                    || z == 0
                    || x == divisions[0]
                    || y == divisions[1]
                    || z == divisions[2];
                if !outside {
                    continue;
                }
                let local = Vec3::new(at(x, 0), at(y, 1), at(z, 2)) * box_.half;
                let world = box_.center
                    + box_.axis[0] * local.x
                    + box_.axis[1] * local.y
                    + box_.axis[2] * local.z;
                // Stable across steps, so contacts warm start by feature.
                let feature = ((x * 8 + y) * 8 + z) as u32;
                out.points[out.len] = (to_ground(world), feature);
                out.len += 1;
            }
        }
    }
    out
}

/// Whether a body, moved by `shift`, comes within `margin` of the ground.
fn body_touches(
    ground: &dyn Ground,
    (position, rotation): (Vec3, Quat),
    body: &Body,
    shift: Vec3,
    margin: f32,
) -> bool {
    let inverse = rotation.conjugate();
    let to_ground = |p: Vec3| inverse * (p - position);
    match body.shape {
        Collider::Sphere { radius } => {
            ground.distance(to_ground(body.position + shift)) <= radius + margin
        }
        Collider::Cuboid { half_extents } => {
            let box_ = moved_obb(body, half_extents, shift);
            let points = cuboid_points(&box_, to_ground);
            (points.points[..points.len].iter()).any(|(p, _)| ground.distance(*p) <= margin)
        }
        Collider::Ground => false,
    }
}

/// Contacts of a body, moved by `shift`, on the ground placed at `frame`; the
/// normal, from ground to body, if they meet. Contacts share one normal, as
/// every manifold does: where the ground bends under a body, the mean of the
/// normals at its contacts, weighted by how deep each is.
fn ground_contacts(
    ground: &dyn Ground,
    (position, rotation): (Vec3, Quat),
    body: &Body,
    shift: Vec3,
    margin: f32,
    ground_is_a: bool,
    out: &mut Points,
) -> Option<Vec3> {
    let inverse = rotation.conjugate();
    let to_ground = |p: Vec3| inverse * (p - position);
    let to_world = |p: Vec3| position + rotation * p;
    let mut push = |on_body: Vec3, on_ground: Vec3, feature: u32| {
        let (body_point, ground_point) = (to_world(on_body), to_world(on_ground));
        if ground_is_a {
            out.push(ground_point, body_point, feature);
        } else {
            out.push(body_point, ground_point, feature);
        }
    };
    let center = to_ground(body.position + shift);
    match body.shape {
        Collider::Sphere { radius } => {
            let distance = ground.distance(center);
            // The estimate reads up to twice the truth.
            if distance > 2.0 * (radius + margin) {
                return None;
            }
            // Project onto the surface, twice, as distances are only estimates.
            let surface = center - ground.normal(center) * distance;
            let surface = surface - ground.normal(surface) * ground.distance(surface);
            let normal = ground.normal(surface);
            let gap = (center - surface).dot(normal) - radius;
            if gap > margin {
                return None;
            }
            push(center - normal * radius, surface, 0);
            Some(rotation * normal)
        }
        Collider::Cuboid { half_extents } => {
            if ground.distance(center) > 2.0 * (half_extents.length() + margin) {
                return None;
            }
            let box_ = moved_obb(body, half_extents, shift);
            let points = cuboid_points(&box_, to_ground);
            let mut near = [(Vec3::ZERO, 0.0f32, 0u32); GROUND_POINTS];
            let mut count = 0;
            let margin = margin + GROUND_SKIN;
            for &(point, feature) in &points.points[..points.len] {
                let distance = ground.distance(point);
                if distance <= margin {
                    near[count] = (point, distance, feature);
                    count += 1;
                }
            }
            // The two axes lying most along the ground, unit length for the score.
            let up = ground.normal(center);
            let mut axes = box_.axis.map(|axis| inverse * axis);
            axes.sort_by(|a, b| a.dot(up).abs().total_cmp(&b.dot(up).abs()));
            let chosen = spread(&near[..count], center, axes[0], axes[1]);
            let mut normal = Vec3::ZERO;
            for &(point, distance, _) in chosen.iter().flatten() {
                normal += ground.normal(point) * (margin - distance + 1.0e-3);
            }
            let normal = normal.try_normalize()?;
            for &(point, distance, feature) in chosen.iter().flatten() {
                push(point, point - normal * distance, feature);
            }
            Some(rotation * normal)
        }
        Collider::Ground => None,
    }
}

/// Up to four points that hold a body up: for each corner of the plane of `u`
/// and `v`, the point furthest that way, unless another is much deeper. Choosing
/// this way changes smoothly: picking by depth alone lets a millimetre swap the
/// whole set each time contacts refresh, and the body rocks for ever.
fn spread(
    points: &[(Vec3, f32, u32)],
    center: Vec3,
    u: Vec3,
    v: Vec3,
) -> [Option<(Vec3, f32, u32)>; MAX_CONTACTS] {
    let mut chosen = [None; MAX_CONTACTS];
    for (slot, (su, sv)) in [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)]
        .into_iter()
        .enumerate()
    {
        let score = |&(point, distance, _): &(Vec3, f32, u32)| {
            let offset = point - center;
            su * offset.dot(u) + sv * offset.dot(v) - GROUND_DEPTH_WEIGHT * distance
        };
        let best = (points.iter()).max_by(|a, b| score(a).total_cmp(&score(b)));
        let fresh = |p: &&(Vec3, f32, u32)| {
            !chosen
                .iter()
                .flatten()
                .any(|c: &(Vec3, f32, u32)| c.2 == p.2)
        };
        chosen[slot] = best.filter(fresh).copied();
    }
    chosen
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
    moved_obb(body, half, Vec3::ZERO)
}

fn moved_obb(body: &Body, half: Vec3, shift: Vec3) -> Obb {
    let m = body.initial_basis;
    Obb {
        center: body.position + shift,
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
}

/// Normal from box to sphere, the sphere's deepest point, and the box surface point.
fn sphere_box(center: Vec3, radius: f32, b: &Obb, margin: f32) -> Option<(Vec3, Vec3, Vec3)> {
    let d = center - b.center;
    let local = Vec3::new(d.dot(b.axis[0]), d.dot(b.axis[1]), d.dot(b.axis[2]));
    let clamped = local.clamp(-b.half, b.half);
    let (n_local, surface) = if clamped != local {
        let offset = local - clamped;
        let distance_squared = offset.length_squared();
        let reach = radius + margin;
        if distance_squared > reach * reach
            || (distance_squared == f32::INFINITY && radius.is_finite())
        {
            return None;
        }
        let distance = distance_squared.sqrt();
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

/// Test an unnormalized SAT axis. Normalize only a new minimum-penetration
/// candidate, not every axis (especially not axes that separate the boxes).
#[allow(clippy::too_many_arguments)]
fn test_axis(
    delta: Vec3,
    axis: Vec3,
    separation: f32,
    kind: u32,
    ia: usize,
    ib: usize,
    margin: f32,
    best: &mut Option<Axis>,
) -> bool {
    let length_sq = axis.length_squared();
    if length_sq < AXIS_EPSILON {
        return true;
    }
    if separation > 0.0 && separation * separation > margin * margin * length_sq {
        return false;
    }
    // The largest separation, compared without normalizing: squares keep sign.
    let squared = |s: f32| s * s.abs();
    if best
        .as_ref()
        .is_none_or(|best| squared(separation) > squared(best.separation) * length_sq)
    {
        let inverse_length = length_sq.sqrt().recip();
        let n = axis * inverse_length;
        *best = Some(Axis {
            kind,
            a: ia,
            b: ib,
            separation: separation * inverse_length,
            normal: if n.dot(delta) < 0.0 { -n } else { n },
        });
    }
    true
}

/// The best face and edge axes, or `None` if an axis separates by over `margin`.
fn box_axes(a: &Obb, b: &Obb, margin: f32) -> Option<(Option<Axis>, Option<Axis>)> {
    let delta = b.center - a.center;
    // Project the relative orientation once. The 15 box SAT tests can then
    // share these nine dot products instead of projecting six axes per test.
    let rotation: [[f32; 3]; 3] =
        std::array::from_fn(|i| std::array::from_fn(|j| a.axis[i].dot(b.axis[j])));
    let absolute = rotation.map(|row| row.map(f32::abs));
    let ta = a.axis.map(|axis| delta.dot(axis));
    let tb = b.axis.map(|axis| delta.dot(axis));
    let (mut face, mut edge) = (None, None);
    for i in 0..3 {
        let ra = (0..3).map(|j| b.half[j] * absolute[i][j]).sum::<f32>();
        let rb = (0..3).map(|j| a.half[j] * absolute[j][i]).sum::<f32>();
        if !test_axis(
            delta,
            a.axis[i],
            ta[i].abs() - (a.half[i] + ra),
            FACE_A,
            i,
            0,
            margin,
            &mut face,
        ) || !test_axis(
            delta,
            b.axis[i],
            tb[i].abs() - (b.half[i] + rb),
            FACE_B,
            0,
            i,
            margin,
            &mut face,
        ) {
            return None;
        }
    }
    for i in 0..3 {
        let (i1, i2) = ((i + 1) % 3, (i + 2) % 3);
        for j in 0..3 {
            let (j1, j2) = ((j + 1) % 3, (j + 2) % 3);
            let ra = a.half[i1] * absolute[i2][j] + a.half[i2] * absolute[i1][j];
            let rb = b.half[j1] * absolute[i][j2] + b.half[j2] * absolute[i][j1];
            let distance = (ta[i2] * rotation[i1][j] - ta[i1] * rotation[i2][j]).abs();
            if !test_axis(
                delta,
                a.axis[i].cross(b.axis[j]),
                distance - (ra + rb),
                EDGE,
                i,
                j,
                margin,
                &mut edge,
            ) {
                return None;
            }
        }
    }
    Some((face, edge))
}

/// Fill `points` and return the normal from A to B.
fn box_box(a: &Obb, b: &Obb, margin: f32, points: &mut Points) -> Option<Vec3> {
    let (face, edge) = box_axes(a, b, margin)?;
    let face = face?;
    // Prefer faces unless an edge axis is clearly better: face manifolds are stable.
    let best = match edge {
        Some(edge) if edge.separation > 0.95 * face.separation + 0.01 => edge,
        _ => face,
    };
    match best.kind {
        EDGE => edge_contact(a, b, &best, points),
        FACE_A => face_contacts(a, b, true, best.a, best.normal, margin, points),
        _ => face_contacts(a, b, false, best.b, best.normal, margin, points),
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
    margin: f32,
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
    // Alternate buffers so each clipping plane reads only the live vertices,
    // rather than copying the entire fixed-capacity polygon on every pass.
    let mut scratch = [Vec3::ZERO; MAX_POLY];
    let (mut input, mut output) = (&mut polygon, &mut scratch);
    let mut count = 4;
    for (plane, extent) in [(u, eu), (-u, eu), (v, ev), (-v, ev)] {
        count = clip(input, output, count, plane, plane.dot(center) + extent);
        if count == 0 {
            return;
        }
        std::mem::swap(&mut input, &mut output);
    }

    let kind = if reference_is_a { FACE_A } else { FACE_B };
    let prefix = (kind << 24) | ((axis as u32) << 16) | ((incident_axis as u32) << 8);
    // Points within reach: polygon index and depth below the reference face.
    let mut found = [(0usize, 0.0f32); MAX_POLY];
    let mut n = 0;
    for (i, &p) in input[..count].iter().enumerate() {
        let distance = (p - center).dot(normal);
        if distance <= PLANE_EPSILON + margin {
            found[n] = (i, distance);
            n += 1;
        }
    }
    let (picked, len) = reduce(input, &found[..n], normal);
    for &i in &picked[..len] {
        let (index, distance) = found[i];
        let p = input[index];
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
    // Compute each distance and signed area once. `max_by` picks the last
    // equal element and `min_by` the first, including for total-ordered NaNs.
    let origin = at(first);
    let mut second = 0;
    let mut max_distance = at(0).distance_squared(origin);
    for i in 1..found.len() {
        let distance = at(i).distance_squared(origin);
        if distance.total_cmp(&max_distance).is_ge() {
            second = i;
            max_distance = distance;
        }
    }
    let line = at(second) - origin;
    let area = |i: usize| line.cross(at(i) - origin).dot(normal);
    let (mut third, mut fourth) = (0, 0);
    let mut max_area = area(0);
    let mut min_area = max_area;
    for i in 1..found.len() {
        let value = area(i);
        if value.total_cmp(&max_area).is_ge() {
            third = i;
            max_area = value;
        }
        if value.total_cmp(&min_area).is_lt() {
            fourth = i;
            min_area = value;
        }
    }
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
fn clip(
    input: &[Vec3; MAX_POLY],
    output: &mut [Vec3; MAX_POLY],
    count: usize,
    n: Vec3,
    offset: f32,
) -> usize {
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
            output[out] = a + (b - a) * t;
            out += 1;
        }
        if b_in && out < MAX_POLY {
            output[out] = b;
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

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat3, Quat};

    #[test]
    fn clipping_retains_polygon_winding_and_intersections() {
        let mut polygon = [Vec3::ZERO; MAX_POLY];
        polygon[..4].copy_from_slice(&[
            Vec3::new(-2.0, -1.0, 0.0),
            Vec3::new(2.0, -1.0, 0.0),
            Vec3::new(2.0, 1.0, 0.0),
            Vec3::new(-2.0, 1.0, 0.0),
        ]);
        let mut scratch = [Vec3::ZERO; MAX_POLY];
        let count = clip(&polygon, &mut scratch, 4, Vec3::X, 1.0);
        let count = clip(&scratch, &mut polygon, count, -Vec3::X, 1.0);
        assert_eq!(
            &polygon[..count],
            &[
                Vec3::new(-1.0, -1.0, 0.0),
                Vec3::new(1.0, -1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(-1.0, 1.0, 0.0),
            ]
        );
    }

    #[test]
    fn reduction_spans_both_sides_of_the_support_polygon() {
        let mut polygon = [Vec3::ZERO; MAX_POLY];
        polygon[..6].copy_from_slice(&[
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.3, 0.0),
            Vec3::new(-0.5, 1.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(-0.5, -0.8, 0.0),
            Vec3::new(0.5, -1.2, 0.0),
        ]);
        let found = [
            (0, -0.2),
            (1, -0.1),
            (2, -0.1),
            (3, -0.1),
            (4, -0.1),
            (5, -0.1),
        ];
        let (picked, len) = reduce(&polygon, &found, Vec3::Z);
        assert_eq!(&picked[..len], &[0, 3, 5, 1]);
    }

    fn corners(b: &Obb) -> [Vec3; 8] {
        std::array::from_fn(|bits| {
            (0..3).fold(b.center, |point, i| {
                let sign = if bits & (1 << i) == 0 { -1.0 } else { 1.0 };
                point + b.axis[i] * (sign * b.half[i])
            })
        })
    }

    // Independent oracle: project corners rather than reusing the cached
    // relative-orientation formulas, including the nine edge-cross axes.
    fn corner_separation(a: &Obb, b: &Obb) -> f32 {
        let (va, vb) = (corners(a), corners(b));
        let edges = a.axis.into_iter().flat_map(|x| b.axis.map(|y| x.cross(y)));
        a.axis
            .into_iter()
            .chain(b.axis)
            .chain(edges)
            .filter(|axis| axis.length_squared() >= AXIS_EPSILON)
            .map(|axis| {
                let axis = axis.normalize();
                let range = |points: &[Vec3; 8]| {
                    points
                        .iter()
                        .map(|p| p.dot(axis))
                        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), p| {
                            (lo.min(p), hi.max(p))
                        })
                };
                let ((amin, amax), (bmin, bmax)) = (range(&va), range(&vb));
                (bmin - amax).max(amin - bmax)
            })
            .fold(f32::NEG_INFINITY, f32::max)
    }

    #[test]
    fn box_sat_matches_corner_projection_for_rotated_and_nearly_parallel_boxes() {
        let mut seed = 0x741b_a963u32;
        let mut random = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        for case in 0..2000 {
            let mut make_box = || {
                let center = Vec3::new(random(), random(), random()) * 6.0 - 3.0;
                let half = Vec3::new(random(), random(), random()) * 1.9 + 0.1;
                let axis = Vec3::new(random() + 0.1, random(), random()).normalize();
                let rotation = Quat::from_axis_angle(axis, random() * std::f32::consts::TAU);
                let matrix = Mat3::from_quat(rotation);
                Obb {
                    center,
                    half,
                    axis: [matrix.x_axis, matrix.y_axis, matrix.z_axis],
                }
            };
            let (a, mut b) = (make_box(), make_box());
            if case % 4 == 0 {
                let rotation = Quat::from_rotation_y(1.0e-5);
                b.axis = a.axis.map(|axis| rotation * axis);
            }
            let separation = corner_separation(&a, &b);
            let mut points = Points::default();
            let normal = box_box(&a, &b, 0.0, &mut points);
            // Both formulas use f32; do not pin classifications at roundoff.
            if separation.abs() > 1.0e-4 {
                assert_eq!(
                    normal.is_some(),
                    separation < 0.0,
                    "case {case}: separation {separation}"
                );
            }
            if let Some(normal) = normal {
                assert!(
                    (normal.length_squared() - 1.0).abs() < 1.0e-4,
                    "case {case}"
                );
                for &(xa, xb, _) in &points.p[..points.len] {
                    assert!(
                        (xb - xa).dot(normal) <= 1.0e-3,
                        "case {case}: separated contact anchors"
                    );
                }
            }
        }
    }
}
