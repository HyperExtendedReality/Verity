//! Shared 45-degree frames for wide-node child bounds. Only box tests use this
//! frame; the ray and triangles remain in mesh space for triangle intersection.
//!
//! For frame 0 (xy), 1 (xz), or 2 (yz), let `(u, v)` be the selected pair in
//! that order. In those two slots the shader computes `precise (u + v) * s`
//! and `precise (v - u) * s`, with `s = asfloat(0x3f3504f3u)`. The third slot
//! is unchanged. Apply this to both ray origin and direction, without
//! translating or normalizing: a common linear map preserves ray parameter t.
//!
//! The baked padding encloses the exact linear image of each triangle despite
//! rounded corner projection. It cannot cover cancellation in arbitrary rays.
//! For each rotated slab the shader additionally expands both decoded endpoints
//! by `error(origin) + exit * error(direction)`, where
//! `error(p) = 8 * EPSILON * max(abs(p[u]), abs(p[v])) + 8 * MIN_POSITIVE` and
//! `exit` is a conservative upper bound on t from the mesh-space root AABB and
//! the current ray limit. The expansion must round outward (or have spare error
//! budget for that rounding). If the ray transform, expansion, or slab reciprocal
//! is unsafe to evaluate, the oriented test must keep the child at entry zero.

use std::ops::Range;

use glam::Vec3;

const LANES: usize = 4;
const PLANES: [[usize; 2]; 3] = [[0, 1], [0, 2], [1, 2]];
const SCALE: f32 = std::f32::consts::FRAC_1_SQRT_2;
const RELATIVE_PAD: f32 = 8.0 * f32::EPSILON;
const ABSOLUTE_PAD: f32 = 8.0 * f32::MIN_POSITIVE;
// Leave room for projected sums, outward padding, and the full grid extent.
const MAX_CORNER: f32 = f32::MAX / 8.0;
// Require more than 25% less proxy traversal work to pay for projecting the ray.
const AREA_BUDGET: f64 = 0.75;

type Bounds = (Vec3, Vec3);
const EMPTY: Bounds = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));

/// One shared frame and its conservative child bounds.
#[derive(Debug)]
pub(super) struct OrientedChildren {
    /// Serialized frame index: 0 = xy, 1 = xz, 2 = yz.
    pub(super) frame: u32,
    /// Quantizable bounds in input lane order. Only populated lanes are valid.
    pub(super) bounds: [(Vec3, Vec3); LANES],
}

/// Selects a profitable shared frame from the actual triangles of each child.
///
/// `child_spans` indexes triangles in `corners` and must match `child_bounds`
/// in length and lane order. Returns `None` if a frame would not improve the
/// area proxy enough or if coordinates cannot be projected safely.
#[cfg(test)]
pub(super) fn choose(
    corners: &[[Vec3; 3]],
    child_spans: &[Range<usize>],
    child_bounds: &[(Vec3, Vec3)],
) -> Option<OrientedChildren> {
    choose_indexed(corners, &[], child_spans, child_bounds)
}

/// Selects bounds over direct spans or indirect mesh-relative references.
pub(super) fn choose_indexed(
    corners: &[[Vec3; 3]],
    references: &[u32],
    child_spans: &[Range<usize>],
    child_bounds: &[Bounds],
) -> Option<OrientedChildren> {
    let len = child_spans.len();
    if len == 0 || len > LANES || len != child_bounds.len() {
        return None;
    }

    let mut baseline = 0.0;
    for &(min, max) in child_bounds {
        if !min.is_finite() || !max.is_finite() || min.cmpgt(max).any() {
            return None;
        }
        baseline += half_area((min, max));
    }
    if baseline == 0.0 {
        return None;
    }

    let mut candidates = [[EMPTY; LANES]; PLANES.len()];
    let mut costs = [0.0; PLANES.len()];
    for (lane, span) in child_spans.iter().enumerate() {
        if span.is_empty() {
            return None;
        }

        let mut bounds = [EMPTY; PLANES.len()];
        let mut magnitude = Vec3::ZERO;
        for index in span.clone() {
            // Full source triangles also enclose the clipped spatial fragments.
            let source = if references.is_empty() {
                index
            } else {
                *references.get(index)? as usize
            };
            for &corner in corners.get(source)? {
                let abs = corner.abs();
                if !corner.is_finite() || abs.max_element() > MAX_CORNER {
                    return None;
                }
                magnitude = magnitude.max(abs);
                for ((min, max), point) in bounds.iter_mut().zip(project_unscaled(corner)) {
                    *min = min.min(point);
                    *max = max.max(point);
                }
            }
        }

        for (frame, [u, v]) in PLANES.into_iter().enumerate() {
            let (mut min, mut max) = bounds[frame];
            // A linear function takes its triangle extrema at the vertices.
            // The positive scale preserves their order, so multiply only the
            // extrema. Both projected operations and subnormal flushes fit
            // inside the pad, and next_down/up round the endpoints outward.
            let pad = rounding_pad(magnitude[u].max(magnitude[v]));
            for axis in [u, v] {
                min[axis] = (min[axis] * SCALE - pad).next_down();
                max[axis] = (max[axis] * SCALE + pad).next_up();
            }
            candidates[frame][lane] = (min, max);
            costs[frame] += half_area((min, max));
        }
    }

    // Summed child surface area proxies the number of child boxes a ray enters.
    // Compare total costs, not independent best frames for individual children.
    let mut budget = baseline * AREA_BUDGET;
    let mut best = None;
    for (frame, cost) in costs.into_iter().enumerate() {
        if cost < budget {
            best = Some(frame);
            budget = cost;
        }
    }
    let frame = best?;
    Some(OrientedChildren {
        frame: frame as u32,
        bounds: candidates[frame],
    })
}

fn project_unscaled(Vec3 { x, y, z }: Vec3) -> [Vec3; PLANES.len()] {
    [
        Vec3::new(x + y, y - x, z),
        Vec3::new(x + z, y, z - x),
        Vec3::new(x, y + z, z - y),
    ]
}

fn rounding_pad(magnitude: f32) -> f32 {
    magnitude * RELATIVE_PAD + ABSOLUTE_PAD
}

fn half_area((min, max): Bounds) -> f64 {
    let d = max.as_dvec3() - min.as_dvec3();
    d.x * d.y + d.y * d.z + d.z * d.x
}

#[cfg(test)]
pub(super) fn packed_entry(
    frame: u32,
    (min, max): Bounds,
    (root_min, root_max): Bounds,
    origin: Vec3,
    direction: Vec3,
    limit: f32,
) -> Option<f32> {
    let mut exit = limit;
    for axis in 0..3 {
        let d = direction[axis];
        if !(1.0e-30..=1.0e30).contains(&d.abs()) {
            continue;
        }
        let endpoint = if d > 0.0 {
            root_max[axis]
        } else {
            root_min[axis]
        };
        let delta = endpoint - origin[axis];
        let time = ((delta * d.signum() + ABSOLUTE_PAD).max(0.0)) / d.abs();
        if delta.is_finite() && time.is_finite() {
            exit = exit.min(time + rounding_pad(time.abs()));
        }
    }
    let frame = frame as usize;
    let [u, v] = PLANES[frame];
    let error = |p: Vec3| rounding_pad(p[u].abs().max(p[v].abs()));
    let mut pad = Vec3::splat(ABSOLUTE_PAD + exit * ABSOLUTE_PAD);
    pad[u] = error(origin) + exit * error(direction);
    pad[v] = pad[u];
    let project = |p| {
        let mut q = project_unscaled(p)[frame];
        q[u] *= SCALE;
        q[v] *= SCALE;
        q
    };
    let (o, d) = (project(origin), project(direction));
    pad += min.abs().max(max.abs()) * RELATIVE_PAD;
    let (lo, hi) = (min - pad, max + pad);
    if [o, d, pad, lo, hi].into_iter().any(|p| !p.is_finite()) {
        return Some(0.0);
    }
    let mut near = 0.0_f32;
    let mut far = exit;
    for axis in 0..3 {
        if d[axis] == 0.0 {
            if !(lo[axis]..=hi[axis]).contains(&o[axis]) {
                return None;
            }
        } else {
            if !(1.0e-30..=1.0e30).contains(&d[axis].abs()) {
                return Some(0.0);
            }
            let a = ((lo[axis] - o[axis]) - ABSOLUTE_PAD) / d[axis];
            let b = ((hi[axis] - o[axis]) + ABSOLUTE_PAD) / d[axis];
            if !a.is_finite() || !b.is_finite() {
                return Some(0.0);
            }
            let (a, b) = (a.min(b), a.max(b));
            near = near.max(a - rounding_pad(a.abs()));
            far = far.min(b + rounding_pad(b.abs()));
        }
    }
    (near <= far).then_some(near)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aabb(triangle: [Vec3; 3]) -> Bounds {
        triangle
            .into_iter()
            .fold(EMPTY, |(lo, hi), p| (lo.min(p), hi.max(p)))
    }

    fn in_plane(p: Vec3, frame: usize) -> Vec3 {
        match frame {
            0 => p,
            1 => Vec3::new(p.x, p.z, p.y),
            2 => Vec3::new(p.z, p.x, p.y),
            _ => unreachable!("only three test frames"),
        }
    }

    fn diagonal_children(frame: usize, translation: Vec3) -> [[Vec3; 3]; 2] {
        let triangle = [
            Vec3::ZERO,
            Vec3::new(128.0, 128.0, 0.0),
            Vec3::new(64.5, 63.5, 2.0),
        ];
        [Vec3::ZERO, Vec3::new(200.0, 198.0, 4.0)]
            .map(|child| triangle.map(|p| in_plane(p + child, frame) + translation))
    }

    #[test]
    fn long_diagonal_children_share_the_tight_frame_in_each_plane() {
        for frame in 0..PLANES.len() {
            let corners = diagonal_children(frame, Vec3::ZERO);
            let axis_aligned = corners.map(aabb);
            let oriented = choose(&corners, &[0..1, 1..2], &axis_aligned)
                .expect("diagonal child triangles benefit from rotation");
            assert_eq!(oriented.frame as usize, frame);
            for (bounds, axis_aligned) in oriented.bounds.into_iter().zip(axis_aligned) {
                assert!(half_area(bounds) < 0.05 * half_area(axis_aligned));
            }
        }
    }

    #[test]
    fn axis_aligned_children_keep_the_cheaper_unrotated_test() {
        let a = [Vec3::ZERO, Vec3::X * 16.0, Vec3::Y * 16.0];
        let corners = [a, a.map(|p| p + Vec3::splat(32.0))];
        assert!(choose(&corners, &[0..1, 1..2], &corners.map(aabb)).is_none());
    }

    #[test]
    fn extreme_corners_keep_the_unrotated_test() {
        let a = [
            Vec3::ZERO,
            Vec3::splat(f32::MAX / 2.0),
            Vec3::new(f32::MAX / 2.0, 1.0, 2.0),
        ];
        let corners = [a, a.map(|p| -p)];
        assert!(choose(&corners, &[0..1, 1..2], &corners.map(aabb)).is_none());
    }

    #[test]
    fn vertex_edge_and_interior_hits_survive_projected_slabs_with_translation() {
        let translations = [
            Vec3::ZERO,
            Vec3::new(16384.0, 32768.0, -4096.0),
            Vec3::new(-262144.0, 262144.0, 4096.0),
        ];
        let directions = [
            Vec3::new(0.0, 0.0, -2.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(32768.0, -32768.0625, 2.0),
        ];
        for frame in 0..PLANES.len() {
            for translation in translations {
                let corners = diagonal_children(frame, translation);
                let oriented = choose(&corners, &[0..1, 1..2], &corners.map(aabb))
                    .expect("translated geometry still benefits from rotation");
                let [u, v] = PLANES[oriented.frame as usize];
                for ([a, b, c], bounds) in corners.into_iter().zip(oriented.bounds) {
                    let root = aabb([a, b, c]);
                    let edge = a + (b - a) * 0.5;
                    let inside = edge + (c - a) * 0.25;
                    for hit in [a, b, c, edge, inside] {
                        let mut projected = hit.as_dvec3();
                        let (first, second) = (projected[u], projected[v]);
                        let scale = f64::from(SCALE);
                        projected[u] = (first + second) * scale;
                        projected[v] = (second - first) * scale;
                        assert!(bounds.0.as_dvec3().cmple(projected).all());
                        assert!(projected.cmple(bounds.1.as_dvec3()).all());

                        for direction in directions.map(|p| in_plane(p, frame)) {
                            let t = 4.0;
                            let origin = hit - direction * t;
                            assert_eq!(
                                origin.as_dvec3() + direction.as_dvec3() * f64::from(t),
                                hit.as_dvec3()
                            );
                            let entry =
                                packed_entry(oriented.frame, bounds, root, origin, direction, t);
                            assert!(
                                entry.is_some_and(|entry| entry <= t),
                                "projected slabs rejected {origin:?} + {t} * {direction:?}",
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn unsafe_ray_arithmetic_keeps_children_at_entry_zero() {
        let bounds = (Vec3::ZERO, Vec3::ONE);
        let root = (Vec3::splat(-f32::MAX), Vec3::splat(f32::MAX));
        for (origin, direction, limit) in [
            (Vec3::splat(f32::MAX), Vec3::NEG_Z, 4.0),
            (Vec3::ZERO, Vec3::splat(1.0e-32), 4.0),
            (Vec3::ZERO, Vec3::splat(1.0e-32), f32::INFINITY),
        ] {
            assert_eq!(
                packed_entry(0, bounds, root, origin, direction, limit),
                Some(0.0)
            );
        }
    }
}
