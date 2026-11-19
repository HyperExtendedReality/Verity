//! Contact generation for cuboids, spheres, capsules, hulls, meshes and ground.
//! Cuboid pairs use the separating axis test with reference-face clipping,
//! ported from the AVBD 3D demo. Hulls use the same test generalized with
//! Gregorius' Gauss-map edge pruning (GDC 2013), and spheres and capsules meet
//! them through GJK. Ground is a field, so a body asks it about points on its
//! own surface. A mesh is tested triangle by triangle, against active edges only.

use crate::hull::{BoxPoly, HULL_POLY, MAX_HULL_VERTICES, Plane, Poly, TrianglePoly, gjk};
use crate::solver::{Body, CONTACT_DISTANCE, Contact};
use crate::{Bounds, Collider, Ground, TriMesh};
use glam::{Mat3, Quat, Vec3};

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
    match (&a.shape, &b.shape) {
        (Collider::Ground | Collider::Mesh(_), Collider::Ground | Collider::Mesh(_)) => false,
        (Collider::Mesh(mesh), _) => {
            let mut found = Candidates::default();
            mesh_candidates((mesh, Pose::of(a, shift)), b, Vec3::ZERO, margin, &mut found);
            found.len > 0
        }
        (_, Collider::Mesh(mesh)) => {
            let mut found = Candidates::default();
            mesh_candidates((mesh, Pose::of(b, Vec3::ZERO)), a, shift, margin, &mut found);
            found.len > 0
        }
        (Collider::Ground, _) => {
            ground.is_some_and(|g| body_touches(g, (center, a.rotation), b, Vec3::ZERO, margin))
        }
        (_, Collider::Ground) => {
            ground.is_some_and(|g| body_touches(g, (b.position, b.rotation), a, shift, margin))
        }
        (&Collider::Sphere { radius: ra }, &Collider::Sphere { radius: rb }) => {
            let reach = ra + rb + margin;
            center.distance_squared(b.position) <= reach * reach
        }
        (&Collider::Sphere { radius }, &Collider::Cuboid { half_extents }) => {
            sphere_box(center, radius, &obb(b, half_extents), margin).is_some()
        }
        (&Collider::Cuboid { half_extents }, &Collider::Sphere { radius }) => {
            let box_a = moved_obb(a, half_extents, shift);
            sphere_box(b.position, radius, &box_a, margin).is_some()
        }
        (&Collider::Cuboid { half_extents: ha }, &Collider::Cuboid { half_extents: hb }) => {
            box_axes(&moved_obb(a, ha, shift), &obb(b, hb), margin).is_some()
        }
        // The rest are cheap enough to test by making their contacts.
        _ => convex_contacts(a, b, shift, margin, &mut Points::default()).is_some(),
    }
}

/// A sphere's or capsule's core, moved by `shift`: the segment its surface is
/// `radius` from. A sphere's is a point.
fn core(body: &Body, shift: Vec3) -> Option<(Vec3, Vec3, f32)> {
    let center = body.position + shift;
    match body.shape {
        Collider::Sphere { radius } => Some((center, center, radius)),
        Collider::Capsule {
            radius,
            half_height,
        } => {
            let axis = body.initial_basis.z_axis * half_height;
            Some((center - axis, center + axis, radius))
        }
        Collider::Cuboid { .. } | Collider::Ground | Collider::Hull(_) | Collider::Mesh(_) => None,
    }
}

/// Where a shape is for the narrowphase: its step-start rotation, as
/// [`moved_obb`] uses, at its position moved by a shift.
#[derive(Clone, Copy)]
struct Pose {
    r: Mat3,
    t: Vec3,
}

impl Pose {
    fn of(body: &Body, shift: Vec3) -> Self {
        Self {
            r: body.initial_basis,
            t: body.position + shift,
        }
    }

    fn point(&self, p: Vec3) -> Vec3 {
        self.r * p + self.t
    }

    fn local(&self, p: Vec3) -> Vec3 {
        self.r.transpose() * (p - self.t)
    }
}

/// Calls `f` with a box's or hull's polytope and pose; `None` for other shapes.
fn with_poly<R>(body: &Body, shift: Vec3, f: impl FnOnce(Poly, Pose) -> R) -> Option<R> {
    let pose = Pose::of(body, shift);
    match body.shape {
        Collider::Cuboid { half_extents } => Some(f(BoxPoly::new(half_extents).poly(), pose)),
        Collider::Hull(ref hull) => Some(f(hull.poly(), pose)),
        _ => None,
    }
}

/// Contacts of convex pairs without ground or a mesh, A moved by `shift`;
/// the normal from B to A, if they meet.
fn convex_contacts(
    a: &Body,
    b: &Body,
    shift: Vec3,
    margin: f32,
    points: &mut Points,
) -> Option<Vec3> {
    match (core(a, shift), core(b, Vec3::ZERO), &a.shape, &b.shape) {
        (Some(ca), Some(cb), ..) => round_round(ca, cb, margin, points),
        (Some(ca), None, _, &Collider::Cuboid { half_extents }) => {
            round_box(ca, &obb(b, half_extents), margin, true, points)
        }
        (None, Some(cb), &Collider::Cuboid { half_extents }, _) => {
            let box_a = moved_obb(a, half_extents, shift);
            round_box(cb, &box_a, margin, false, points).map(|n| -n)
        }
        (Some(ca), None, ..) => with_poly(b, Vec3::ZERO, |poly, pose| {
            round_poly(ca, poly, pose, margin, true, points)
        })?,
        (None, Some(cb), ..) => {
            with_poly(a, shift, |poly, pose| round_poly(cb, poly, pose, margin, false, points))?
                .map(|n| -n)
        }
        (None, None, ..) => with_poly(a, shift, |pa, ta| {
            with_poly(b, Vec3::ZERO, |pb, tb| poly_poly(pa, ta, pb, tb, margin, points))
        })??
        .map(|n| -n),
    }
}

/// Two round cores: their closest points and, where they run side by side,
/// the ends of the stretch they share, so a capsule lying on another rests.
fn round_round(
    (a0, a1, ra): (Vec3, Vec3, f32),
    (b0, b1, rb): (Vec3, Vec3, f32),
    margin: f32,
    points: &mut Points,
) -> Option<Vec3> {
    let reach = ra + rb + margin;
    let (pa, pb) = closest_on_segments(a0, a1, b0, b1);
    let d = pa - pb;
    let distance_squared = d.length_squared();
    if distance_squared > reach * reach {
        return None;
    }
    let distance = distance_squared.sqrt();
    let n = if distance > AXIS_EPSILON {
        d / distance
    } else {
        // Crossing cores: any direction across both.
        (a1 - a0).cross(b1 - b0).try_normalize().unwrap_or(Vec3::Z)
    };
    points.push(pa - n * ra, pb + n * rb, 0);
    let ends = [
        closest_on_segments(a0, a0, b0, b1),
        closest_on_segments(a1, a1, b0, b1),
        closest_on_segments(a0, a1, b0, b0),
        closest_on_segments(a0, a1, b1, b1),
    ];
    for (feature, (pa, pb)) in (1..).zip(ends) {
        if pa.distance_squared(pb) <= reach * reach {
            points.push(pa - n * ra, pb + n * rb, feature);
        }
    }
    Some(n)
}

/// A round core against a box: the core's point nearest the box and its ends
/// where they lie against the same face, so a capsule lying on a box rests on
/// both. The normal is from the box to the core.
fn round_box(
    (p0, p1, radius): (Vec3, Vec3, f32),
    b: &Obb,
    margin: f32,
    round_is_a: bool,
    points: &mut Points,
) -> Option<Vec3> {
    // Alternating projections converge on the closest points of two convex sets.
    let mut p = 0.5 * (p0 + p1);
    for _ in 0..8 {
        let q = b.clamp(p);
        p = closest_on_segments(q, q, p0, p1).1;
    }
    let (n, _, on_box) = sphere_box(p, radius, b, margin)?;
    let mut push = |center: Vec3, on_box: Vec3, feature| {
        let on_round = center - n * radius;
        if round_is_a {
            points.push(on_round, on_box, feature);
        } else {
            points.push(on_box, on_round, feature);
        }
    };
    push(p, on_box, 0);
    for (feature, end) in [(1, p0), (2, p1)] {
        if let Some((end_n, _, on_box)) = sphere_box(end, radius, b, margin)
            && end_n.dot(n) > 0.95
        {
            push(end, on_box, feature);
        }
    }
    Some(n)
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
    let thickness = |shape: &Collider| match *shape {
        Collider::Cuboid { half_extents } => half_extents.min_element(),
        Collider::Sphere { radius } | Collider::Capsule { radius, .. } => radius,
        Collider::Hull(ref hull) => hull.inner(),
        Collider::Ground | Collider::Mesh(_) => 0.0,
    };
    let spacing = 0.5 * (thickness(&a.shape) + thickness(&b.shape));
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
    let normal = match (&a.shape, &b.shape) {
        (Collider::Ground, Collider::Ground) => return None,
        // Meshes make several manifolds: see `collide_mesh`.
        (Collider::Mesh(_), _) | (_, Collider::Mesh(_)) => return None,
        (Collider::Ground, _) => {
            let frame = (center, a.rotation);
            -ground_contacts(ground?, frame, b, Vec3::ZERO, margin, true, &mut points)?
        }
        (_, Collider::Ground) => {
            let frame = (b.position, b.rotation);
            ground_contacts(ground?, frame, a, shift, margin, false, &mut points)?
        }
        (&Collider::Sphere { radius: ra }, &Collider::Sphere { radius: rb }) => {
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
        (&Collider::Sphere { radius }, &Collider::Cuboid { half_extents }) => {
            let (n, xa, xb) = sphere_box(center, radius, &obb(b, half_extents), margin)?;
            points.push(xa, xb, 0);
            n
        }
        (&Collider::Cuboid { half_extents }, &Collider::Sphere { radius }) => {
            let box_a = moved_obb(a, half_extents, shift);
            let (n, xb, xa) = sphere_box(b.position, radius, &box_a, margin)?;
            points.push(xa, xb, 0);
            -n
        }
        (&Collider::Cuboid { half_extents: ha }, &Collider::Cuboid { half_extents: hb }) => {
            -box_box(&moved_obb(a, ha, shift), &obb(b, hb), margin, &mut points)?
        }
        _ => convex_contacts(a, b, shift, margin, &mut points)?,
    };
    Some(finish(a, b, shift, normal, &points))
}

/// Contact basis and contacts with anchors in each body's frame, from points
/// found with A moved by `shift`.
fn finish(
    a: &Body,
    b: &Body,
    shift: Vec3,
    normal: Vec3,
    points: &Points,
) -> ([Vec3; 3], [Contact; MAX_CONTACTS], usize) {
    let center = a.position + shift;
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
    ([normal, t1, t2], contacts, points.len)
}

/// Where a body's own points lie for a ground test, and which feature each is.
struct BodyPoints<const N: usize = GROUND_POINTS> {
    points: [(Vec3, u32); N],
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
        Collider::Capsule { radius, .. } => (capsule_samples(body, shift).iter())
            .any(|&p| ground.distance(to_ground(p)) <= radius + margin),
        Collider::Cuboid { half_extents } => {
            let box_ = moved_obb(body, half_extents, shift);
            let points = cuboid_points(&box_, to_ground);
            (points.points[..points.len].iter()).any(|(p, _)| ground.distance(*p) <= margin)
        }
        Collider::Hull(ref hull) => {
            let points = hull_points(hull.poly(), Pose::of(body, shift), to_ground);
            (points.points[..points.len].iter()).any(|(p, _)| ground.distance(*p) <= margin)
        }
        Collider::Ground | Collider::Mesh(_) => false,
    }
}

/// Most points on a hull's surface for a ground test: vertices and face centers.
const HULL_GROUND_POINTS: usize = 3 * MAX_HULL_VERTICES;

/// A hull's vertices and face centers in the ground's frame. Face centers
/// find a ridge under a broad face, as the cuboid's lattice does.
fn hull_points(
    poly: Poly,
    pose: Pose,
    to_ground: impl Fn(Vec3) -> Vec3,
) -> BodyPoints<HULL_GROUND_POINTS> {
    let mut out = BodyPoints {
        points: [(Vec3::ZERO, 0); HULL_GROUND_POINTS],
        len: 0,
    };
    let faces = (0..poly.faces.len()).map(|f| {
        let count = poly.faces[f].count as f32;
        poly.face(f).sum::<Vec3>() / count
    });
    for (feature, p) in (0..).zip(poly.vertices.iter().copied().chain(faces)) {
        if out.len == HULL_GROUND_POINTS {
            break;
        }
        out.points[out.len] = (to_ground(pose.point(p)), feature);
        out.len += 1;
    }
    out
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
            let (normal, surface, _) = ground_sphere(ground, center, radius, margin)?;
            push(center - normal * radius, surface, 0);
            Some(rotation * normal)
        }
        Collider::Capsule { radius, .. } => {
            let mut hits = [None; MAX_CONTACTS];
            let mut normal = Vec3::ZERO;
            for (hit, p) in hits.iter_mut().zip(capsule_samples(body, shift)) {
                let p = to_ground(p);
                *hit = ground_sphere(ground, p, radius, margin).map(|(n, surface, gap)| {
                    normal += n * (margin - gap + 1.0e-3);
                    (p, surface)
                });
            }
            // One normal for all, weighted by depth as for cuboids.
            let normal = normal.try_normalize()?;
            for (feature, hit) in (0..).zip(hits) {
                if let Some((p, surface)) = hit {
                    push(p - normal * radius, surface, feature);
                }
            }
            Some(rotation * normal)
        }
        Collider::Cuboid { half_extents } => {
            if ground.distance(center) > 2.0 * (half_extents.length() + margin) {
                return None;
            }
            let box_ = moved_obb(body, half_extents, shift);
            let points = cuboid_points(&box_, to_ground);
            // The two axes lying most along the ground, unit length for the score.
            let up = ground.normal(center);
            let mut axes = box_.axis.map(|axis| inverse * axis);
            axes.sort_by(|a, b| a.dot(up).abs().total_cmp(&b.dot(up).abs()));
            let mut near = [(Vec3::ZERO, 0.0f32, 0u32); GROUND_POINTS];
            let found = &points.points[..points.len];
            let (normal, chosen) =
                rest_on(ground, found, &mut near, center, (axes[0], axes[1]), margin)?;
            for &(point, distance, feature) in chosen.iter().flatten() {
                push(point, point - normal * distance, feature);
            }
            Some(rotation * normal)
        }
        Collider::Hull(ref hull) => {
            if ground.distance(center) > 2.0 * (hull.radius() + margin) {
                return None;
            }
            let points = hull_points(hull.poly(), Pose::of(body, shift), to_ground);
            let (u, v) = ground.normal(center).any_orthonormal_pair();
            let mut near = [(Vec3::ZERO, 0.0f32, 0u32); HULL_GROUND_POINTS];
            let found = &points.points[..points.len];
            let (normal, chosen) = rest_on(ground, found, &mut near, center, (u, v), margin)?;
            for &(point, distance, feature) in chosen.iter().flatten() {
                push(point, point - normal * distance, feature);
            }
            Some(rotation * normal)
        }
        Collider::Ground | Collider::Mesh(_) => None,
    }
}

/// A body's points near the ground, in its frame: up to four spread over the
/// plane of `u` and `v`, and the depth-weighted normal under them.
#[allow(clippy::type_complexity)]
fn rest_on(
    ground: &dyn Ground,
    points: &[(Vec3, u32)],
    near: &mut [(Vec3, f32, u32)],
    center: Vec3,
    (u, v): (Vec3, Vec3),
    margin: f32,
) -> Option<(Vec3, [Option<(Vec3, f32, u32)>; MAX_CONTACTS])> {
    let mut count = 0;
    let margin = margin + GROUND_SKIN;
    for &(point, feature) in points {
        let distance = ground.distance(point);
        if distance <= margin {
            near[count] = (point, distance, feature);
            count += 1;
        }
    }
    let chosen = spread(&near[..count], center, u, v);
    let mut normal = Vec3::ZERO;
    for &(point, distance, _) in chosen.iter().flatten() {
        normal += ground.normal(point) * (margin - distance + 1.0e-3);
    }
    Some((normal.try_normalize()?, chosen))
}

/// A sphere's normal, surface point and gap on the ground, in the ground's
/// frame, if within `margin`.
fn ground_sphere(
    ground: &dyn Ground,
    center: Vec3,
    radius: f32,
    margin: f32,
) -> Option<(Vec3, Vec3, f32)> {
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
    (gap <= margin).then_some((normal, surface, gap))
}

/// Sphere centers along a capsule's core for a ground test, at most
/// `GROUND_SPACING` apart, ends included.
fn capsule_samples(body: &Body, shift: Vec3) -> Vec<Vec3> {
    let Some((p0, p1, _)) = core(body, shift) else {
        return Vec::new();
    };
    let gaps = ((p0.distance(p1) / GROUND_SPACING).ceil() as usize).clamp(1, MAX_CONTACTS - 1);
    (0..=gaps)
        .map(|i| p0.lerp(p1, i as f32 / gaps as f32))
        .collect()
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
    /// The point of the box nearest `p`.
    fn clamp(&self, p: Vec3) -> Vec3 {
        let d = p - self.center;
        (0..3).fold(self.center, |q, i| {
            q + self.axis[i] * d.dot(self.axis[i]).clamp(-self.half[i], self.half[i])
        })
    }

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
    polygon: &[Vec3],
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
fn clip(input: &[Vec3], output: &mut [Vec3], count: usize, n: Vec3, offset: f32) -> usize {
    let mut out = 0;
    let mut a = input[count - 1];
    let mut da = n.dot(a) - offset;
    for &b in &input[..count] {
        let db = n.dot(b) - offset;
        let (a_in, b_in) = (da <= PLANE_EPSILON, db <= PLANE_EPSILON);
        if a_in != b_in && out < output.len() {
            let denom = da - db;
            let t = if denom.abs() > AXIS_EPSILON {
                (da / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            output[out] = a + (b - a) * t;
            out += 1;
        }
        if b_in && out < output.len() {
            output[out] = b;
            out += 1;
        }
        (a, da) = (b, db);
    }
    out
}

/// Contacts of two polytopes (boxes, hulls, triangles) at their poses: the
/// separating axis test over face normals and the edge pairs whose Gauss arcs
/// cross, which are the only edge pairs that can touch (Gregorius, "The
/// Separating Axis Test between Convex Polyhedra", GDC 2013). Fills `points`
/// and returns the normal from A to B, or `None` if they are over `margin` apart.
fn poly_poly(
    a: Poly,
    pa: Pose,
    b: Poly,
    pb: Pose,
    margin: f32,
    points: &mut Points,
) -> Option<Vec3> {
    // B in A's frame.
    let ra = pa.r.transpose();
    let r = ra * pb.r;
    let t = ra * (pb.t - pa.t);
    let rt = r.transpose();
    let b_in_a = |p: Vec3| r * p + t;
    let a_in_b = |p: Vec3| rt * (p - t);
    let mut face_a = (f32::NEG_INFINITY, 0);
    for (i, plane) in a.planes.iter().enumerate() {
        let deepest = b_in_a(b.vertices[b.support(rt * -plane.normal)]);
        let s = plane.distance(deepest);
        if s > margin {
            return None;
        }
        if s > face_a.0 {
            face_a = (s, i);
        }
    }
    let mut face_b = (f32::NEG_INFINITY, 0);
    for (i, plane) in b.planes.iter().enumerate() {
        let deepest = a_in_b(a.vertices[a.support(r * -plane.normal)]);
        let s = plane.distance(deepest);
        if s > margin {
            return None;
        }
        if s > face_b.0 {
            face_b = (s, i);
        }
    }
    let mut edge = (f32::NEG_INFINITY, 0, 0, Vec3::ZERO);
    for (j, eb) in b.edges.iter().enumerate() {
        let q0 = b_in_a(b.vertices[eb.a as usize]);
        let along_b = b_in_a(b.vertices[eb.b as usize]) - q0;
        // B's arc, negated: the Minkowski difference's.
        let (c, d) = (-(r * eb.normals[0]), -(r * eb.normals[1]));
        let dxc = d.cross(c);
        for (i, ea) in a.edges.iter().enumerate() {
            let [na, nb] = ea.normals;
            let bxa = nb.cross(na);
            let (cba, dba) = (c.dot(bxa), d.dot(bxa));
            let (adc, bdc) = (na.dot(dxc), nb.dot(dxc));
            if !(cba * dba < 0.0 && adc * bdc < 0.0 && cba * bdc > 0.0) {
                continue;
            }
            let p0 = a.vertices[ea.a as usize];
            let along_a = a.vertices[ea.b as usize] - p0;
            let axis = along_a.cross(along_b);
            let length_sq = axis.length_squared();
            // Parallel edges: their faces' axes already cover them.
            if length_sq < 1.0e-6 * along_a.length_squared() * along_b.length_squared() {
                continue;
            }
            let mut axis = axis / length_sq.sqrt();
            if axis.dot(na + nb) < 0.0 {
                axis = -axis;
            }
            let s = axis.dot(q0 - p0);
            if s > margin {
                return None;
            }
            if s > edge.0 {
                edge = (s, i, j, axis);
            }
        }
    }
    // Prefer A's faces, then faces over edges, unless clearly worse: stable manifolds.
    let (face, reference_is_a) = if face_b.0 > face_a.0 + 1.0e-3 {
        (face_b, false)
    } else {
        (face_a, true)
    };
    if edge.0 > 0.95 * face.0 + 0.01 {
        let (_, i, j, axis) = edge;
        let (ea, eb) = (a.edges[i], b.edges[j]);
        let (p0, p1) = (a.vertices[ea.a as usize], a.vertices[ea.b as usize]);
        let (q0, q1) = (
            b_in_a(b.vertices[eb.a as usize]),
            b_in_a(b.vertices[eb.b as usize]),
        );
        let (xa, xb) = closest_on_segments(p0, p1, q0, q1);
        let feature = (EDGE << 24) | ((i as u32 & 0xFF) << 8) | (j as u32 & 0xFF);
        points.push(pa.point(xa), pa.point(xb), feature);
        return Some(pa.r * axis);
    }
    let normal = if reference_is_a {
        poly_face(a, pa, face.1, b, pb, margin, true, points);
        pa.r * a.planes[face.1].normal
    } else {
        poly_face(b, pb, face.1, a, pa, margin, false, points);
        -(pb.r * b.planes[face.1].normal)
    };
    if points.len == 0 {
        let sa = a.support(pa.r.transpose() * normal);
        let sb = b.support(pb.r.transpose() * -normal);
        let kind = if reference_is_a { FACE_A } else { FACE_B };
        points.push(pa.point(a.vertices[sa]), pb.point(b.vertices[sb]), kind << 24);
    }
    Some(normal)
}

/// Clip the incident face against the reference face's side planes, in the
/// world, keeping points within `margin` below the reference face.
#[allow(clippy::too_many_arguments)]
fn poly_face(
    reference: Poly,
    rp: Pose,
    face: usize,
    incident: Poly,
    ip: Pose,
    margin: f32,
    reference_is_a: bool,
    points: &mut Points,
) {
    let plane = reference.planes[face];
    let normal = rp.r * plane.normal;
    let offset = plane.offset + normal.dot(rp.t);
    let local = ip.r.transpose() * normal;
    let incident_face = (0..incident.planes.len())
        .min_by(|&i, &j| {
            let dot = |k: usize| incident.planes[k].normal.dot(local);
            dot(i).total_cmp(&dot(j))
        })
        .expect("a polytope has faces");
    let mut polygon = [Vec3::ZERO; HULL_POLY];
    let mut count = 0;
    for p in incident.face(incident_face).take(HULL_POLY) {
        polygon[count] = ip.point(p);
        count += 1;
    }
    let mut scratch = [Vec3::ZERO; HULL_POLY];
    let (mut input, mut output) = (&mut polygon, &mut scratch);
    let mut corners = [Vec3::ZERO; MAX_HULL_VERTICES];
    let mut sides = 0;
    for p in reference.face(face).take(MAX_HULL_VERTICES) {
        corners[sides] = rp.point(p);
        sides += 1;
    }
    for k in 0..sides {
        let (p, q) = (corners[k], corners[(k + 1) % sides]);
        // Outward in the face's plane: its loop runs counter-clockwise.
        let side = (q - p).cross(normal).normalize_or_zero();
        count = clip(&input[..], &mut output[..], count, side, side.dot(p));
        if count == 0 {
            return;
        }
        std::mem::swap(&mut input, &mut output);
    }
    let kind = if reference_is_a { FACE_A } else { FACE_B };
    let prefix =
        (kind << 24) | ((face as u32 & 0xFF) << 16) | ((incident_face as u32 & 0xFF) << 8);
    let mut found = [(0usize, 0.0f32); HULL_POLY];
    let mut n = 0;
    for (i, &p) in input[..count].iter().enumerate() {
        let distance = normal.dot(p) - offset;
        if distance <= PLANE_EPSILON + margin {
            found[n] = (i, distance);
            n += 1;
        }
    }
    let (picked, len) = reduce(&input[..count], &found[..n], normal);
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

/// A round core (sphere or capsule) against a polytope at `pose`, by GJK:
/// the nearest points, and the core's ends where they lie against the same
/// side, so a capsule lying on a hull rests on both. If the core reaches
/// inside, it leaves through the face it is least deep behind. The normal
/// is from the polytope to the core.
fn round_poly(
    (p0, p1, radius): (Vec3, Vec3, f32),
    poly: Poly,
    pose: Pose,
    margin: f32,
    round_is_a: bool,
    points: &mut Points,
) -> Option<Vec3> {
    let ends = [pose.local(p0), pose.local(p1)];
    let core = if p0 == p1 { &ends[..1] } else { &ends[..] };
    let reach = radius + margin;
    let (on_core, on_poly, distance) = gjk(core, poly.vertices);
    if distance > reach {
        return None;
    }
    let (n, on_core, on_poly) = if distance > 1.0e-6 {
        ((on_core - on_poly) / distance, on_core, on_poly)
    } else {
        let depth = |plane: &&Plane| {
            (core.iter()).map(|&c| plane.distance(c)).fold(f32::INFINITY, f32::min)
        };
        let plane = (poly.planes.iter())
            .max_by(|a, b| depth(a).total_cmp(&depth(b)))
            .expect("a polytope has faces");
        let deep = (core.iter().copied())
            .min_by(|&a, &b| plane.distance(a).total_cmp(&plane.distance(b)))
            .expect("a core has a point");
        (plane.normal, deep, deep - plane.normal * plane.distance(deep))
    };
    let mut push = |center: Vec3, on_poly: Vec3, feature: u32| {
        let (round, other) = (pose.point(center - n * radius), pose.point(on_poly));
        if round_is_a {
            points.push(round, other, feature);
        } else {
            points.push(other, round, feature);
        }
    };
    push(on_core, on_poly, 0);
    if core.len() == 2 {
        for (feature, end) in [(1, ends[0]), (2, ends[1])] {
            let (_, on_end, d) = gjk(&[end], poly.vertices);
            if d > 1.0e-6 && d <= reach && ((end - on_end) / d).dot(n) > 0.95 {
                push(end, on_end, feature);
            }
        }
    }
    Some(pose.r * n)
}

/// Most manifolds a body has with one mesh: one per direction it is pushed.
pub(crate) const MAX_PARTS: usize = 4;
/// Triangle contacts this close in normal share a manifold (about 25 degrees).
const PART_COS: f32 = 0.9;
/// Most triangle contacts one test gathers from a mesh.
const MESH_CANDIDATES: usize = 64;
/// How near an edge a contact on a triangle must be to count as on it.
const EDGE_TOLERANCE: f32 = 1.0e-3;

/// One manifold of a body against a mesh.
#[derive(Default, Clone, Copy)]
pub(crate) struct Part {
    pub basis: [Vec3; 3],
    pub contacts: [Contact; MAX_CONTACTS],
    pub len: usize,
    /// Stable while the body touches the same side of the mesh, from 1.
    pub id: u8,
}

/// Triangle contacts: on the body, on the mesh, the normal from mesh to
/// body, and a feature.
struct Candidates {
    found: [(Vec3, Vec3, Vec3, u32); MESH_CANDIDATES],
    len: usize,
}

impl Default for Candidates {
    fn default() -> Self {
        Self {
            found: [(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, 0); MESH_CANDIDATES],
            len: 0,
        }
    }
}

/// Contacts of `body`, moved by `shift`, with each triangle of `mesh` it nears.
fn mesh_candidates(
    (mesh, at): (&TriMesh, Pose),
    body: &Body,
    shift: Vec3,
    margin: f32,
    out: &mut Candidates,
) {
    let reach = body.bound() + margin;
    let center = at.local(body.position + shift);
    let window = Bounds {
        min: center - reach,
        max: center + reach,
    };
    let round = core(body, shift);
    mesh.overlapping(window, |slot| {
        let corners = mesh.triangles()[slot];
        let Some(triangle) = TrianglePoly::new(corners) else {
            return;
        };
        let mut points = Points::default();
        let normal = match round {
            Some(core) => round_poly(core, triangle.poly(), at, margin, true, &mut points),
            None => with_poly(body, shift, |poly, pose| {
                poly_poly(poly, pose, triangle.poly(), at, margin, &mut points).map(|n| -n)
            })
            .flatten(),
        };
        let Some(mut normal) = normal else {
            return;
        };
        let face = at.r * triangle.normal();
        let face = if face.dot(normal) < 0.0 { -face } else { face };
        // Off an inactive edge, or a vertex between inactive edges, push off
        // the face instead: the neighbour's face holds the body there.
        let active = mesh.active(slot);
        let on_active = |p: Vec3| {
            let p = at.local(p);
            (0..3).any(|k| {
                let (_, q) = closest_on_segments(p, p, corners[k], corners[(k + 1) % 3]);
                active >> k & 1 != 0 && q.distance(p) <= EDGE_TOLERANCE
            })
        };
        let pairs = &mut points.p[..points.len];
        if normal.dot(face) < 0.999 && !pairs.iter().any(|&(_, on_mesh, _)| on_active(on_mesh)) {
            normal = face;
            let plane = face.dot(at.point(corners[0]));
            for (on_body, on_mesh, _) in pairs.iter_mut() {
                *on_mesh = *on_body - face * (face.dot(*on_body) - plane);
            }
        }
        for &(on_body, on_mesh, feature) in pairs.iter() {
            if out.len < MESH_CANDIDATES {
                let feature = (slot as u32) << 8 | (feature & 0xFF);
                out.found[out.len] = (on_body, on_mesh, normal, feature);
                out.len += 1;
            }
        }
    });
}

/// A body's manifolds with a mesh, normals B to A as [`collide`] gives them,
/// and how many there are. Triangle contacts are grouped by normal: a group
/// shares its depth-weighted mean, and keeps the four spanning the most area,
/// as a box's face does.
pub(crate) fn collide_mesh(a: &Body, b: &Body) -> ([Part; MAX_PARTS], usize) {
    let mut parts = [Part::default(); MAX_PARTS];
    let margin = CONTACT_DISTANCE + a.reach + b.reach;
    let Some(shift) = predicted_hit(a, b, margin, None) else {
        return (parts, 0);
    };
    let mut found = Candidates::default();
    let (mesh_pose, mesh_is_a) = match (&a.shape, &b.shape) {
        (Collider::Mesh(_), Collider::Mesh(_) | Collider::Ground)
        | (Collider::Ground, Collider::Mesh(_)) => return (parts, 0),
        (Collider::Mesh(mesh), _) => {
            let at = Pose::of(a, shift);
            mesh_candidates((mesh, at), b, Vec3::ZERO, margin, &mut found);
            (at, true)
        }
        (_, Collider::Mesh(mesh)) => {
            let at = Pose::of(b, Vec3::ZERO);
            mesh_candidates((mesh, at), a, shift, margin, &mut found);
            (at, false)
        }
        _ => return (parts, 0),
    };
    let found = &found.found[..found.len];
    let weight = |i: usize| {
        let (on_body, on_mesh, n, _) = found[i];
        margin - (on_body - on_mesh).dot(n) + 1.0e-3
    };
    // Deepest first, so each group's direction starts from its deepest contact.
    let mut order = [0usize; MESH_CANDIDATES];
    for (i, slot) in order.iter_mut().enumerate() {
        *slot = i;
    }
    let order = &mut order[..found.len()];
    order.sort_by(|&i, &j| weight(j).total_cmp(&weight(i)));
    let mut sums = [Vec3::ZERO; MAX_PARTS];
    let mut group = [0usize; MESH_CANDIDATES];
    let mut count = 0;
    for &i in order.iter() {
        let n = found[i].2;
        let along = |g: usize| sums[g].normalize_or_zero().dot(n);
        let nearest = (0..count).max_by(|&g, &h| along(g).total_cmp(&along(h)));
        let g = match nearest {
            Some(g) if along(g) > PART_COS || count == MAX_PARTS => g,
            _ => {
                count += 1;
                count - 1
            }
        };
        sums[g] += n * weight(i);
        group[i] = g;
    }
    let mut kept = 0;
    let mut used = [0u8; MAX_PARTS];
    for (g, sum) in sums[..count].iter().enumerate() {
        let Some(n) = sum.try_normalize() else {
            continue;
        };
        let mut middles = [Vec3::ZERO; MESH_CANDIDATES];
        let mut depths = [(0usize, 0.0f32); MESH_CANDIDATES];
        let mut members = [0usize; MESH_CANDIDATES];
        let mut len = 0;
        for i in (0..found.len()).filter(|&i| group[i] == g) {
            let (on_body, on_mesh, ..) = found[i];
            middles[len] = 0.5 * (on_body + on_mesh);
            depths[len] = (len, (on_body - on_mesh).dot(n));
            members[len] = i;
            len += 1;
        }
        let (picked, picked_len) = reduce(&middles[..len], &depths[..len], n);
        let mut points = Points::default();
        for &k in &picked[..picked_len] {
            let (on_body, on_mesh, _, feature) = found[members[k]];
            if mesh_is_a {
                points.push(on_mesh, on_body, feature);
            } else {
                points.push(on_body, on_mesh, feature);
            }
        }
        // Named by the side of the mesh it pushes from, so it keeps its warm start.
        let local = mesh_pose.r.transpose() * n;
        let size = local.abs();
        let axis = if size.x >= size.y && size.x >= size.z {
            0
        } else if size.y >= size.z {
            1
        } else {
            2
        };
        let mut id = 1 + 2 * axis as u8 + u8::from(local[axis] < 0.0);
        while used[..kept].contains(&id) {
            id += 6;
        }
        used[kept] = id;
        let normal = if mesh_is_a { -n } else { n };
        let (basis, contacts, len) = finish(a, b, shift, normal, &points);
        parts[kept] = Part {
            basis,
            contacts,
            len,
            id,
        };
        kept += 1;
    }
    (parts, kept)
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
