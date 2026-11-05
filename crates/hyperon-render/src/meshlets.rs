//! CPU meshlet preparation and conservative homogeneous-frustum testing.

use glam::{Mat4, Vec4};
use hyperon_strata::Bounds;

/// Only draw ranges and bounds are needed: the GPU keeps the original indices.
pub(crate) struct Meshlet {
    pub(crate) first_triangle: u32,
    pub(crate) triangle_count: u32,
    pub(crate) bounds: Bounds,
}

/// Failure leaves the original draw available as an uncullled fallback.
pub(crate) fn build(positions: &[[f32; 3]], indices: &[u32]) -> Option<Vec<Meshlet>> {
    hyperon_strata::build_meshlets(positions, indices)
        .ok()
        .map(|meshlets| {
            meshlets
                .into_iter()
                .map(|meshlet| Meshlet {
                    first_triangle: meshlet.first_triangle,
                    triangle_count: meshlet.triangle_count,
                    bounds: meshlet.bounds,
                })
                .collect()
        })
}

/// Emit contiguous runs in primitive-local vertex coordinates. The caller uses
/// each run's start as Vulkan firstVertex, retaining the original triangle ID.
pub(crate) fn visible_runs(
    meshlets: &[Meshlet],
    clip_from_world: Mat4,
    model: Mat4,
    mut emit: impl FnMut(u32, u32),
) {
    let mut run: Option<(u32, u32)> = None;
    for meshlet in meshlets {
        if visible(&meshlet.bounds, clip_from_world, model) {
            let first = meshlet.first_triangle * 3;
            let count = meshlet.triangle_count * 3;
            match &mut run {
                Some((start, length)) if *start + *length == first => *length += count,
                _ => {
                    if let Some((start, length)) = run.take() {
                        emit(start, length);
                    }
                    run = Some((first, count));
                }
            }
        } else if let Some((start, length)) = run.take() {
            emit(start, length);
        }
    }
    if let Some((start, length)) = run {
        emit(start, length);
    }
}

/// Reject only if every corner is strictly outside one Vulkan clip plane.
/// Non-finite transforms or arithmetic cannot establish invisibility.
pub(crate) fn visible(bounds: &Bounds, clip_from_world: Mat4, model: Mat4) -> bool {
    let mut outside = [true; 6];
    for x in [bounds.min[0], bounds.max[0]] {
        for y in [bounds.min[1], bounds.max[1]] {
            for z in [bounds.min[2], bounds.max[2]] {
                let object = Vec4::new(x, y, z, 1.0);
                let p = clip_from_world * (model * object);
                // Bound intermediate magnitudes, not just the final coordinates:
                // cancellation in either shader multiplication can amplify error.
                let abs_mul = |m: Mat4, v: Vec4| {
                    m.x_axis.abs() * v.x
                        + m.y_axis.abs() * v.y
                        + m.z_axis.abs() * v.z
                        + m.w_axis.abs() * v.w
                };
                let magnitude = abs_mul(clip_from_world, abs_mul(model, object.abs()));
                if !p.is_finite() || !magnitude.is_finite() {
                    return true;
                }
                let epsilon = 1.0e-5 * magnitude.max_element().max(1.0);
                let distances = [p.x + p.w, p.w - p.x, p.y + p.w, p.w - p.y, p.z, p.w - p.z];
                for (all_outside, distance) in outside.iter_mut().zip(distances) {
                    *all_outside &= distance < -epsilon;
                }
            }
        }
    }
    !outside.into_iter().any(|all_outside| all_outside)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn clip_bounds_are_conservative_for_transforms_and_invalid_values() {
        let bounds = Bounds {
            min: [-0.5; 3],
            max: [0.5; 3],
        };
        let visible_model = |model| visible(&bounds, Mat4::IDENTITY, model);
        assert!(visible_model(Mat4::from_translation(Vec3::new(
            0.0, 0.0, 0.5
        ))));
        assert!(!visible_model(Mat4::from_translation(Vec3::new(
            3.0, 0.0, 0.5
        ))));
        assert!(visible_model(Mat4::from_scale_rotation_translation(
            Vec3::new(-3.0, 0.1, 1.0),
            glam::Quat::IDENTITY,
            Vec3::new(2.0, 0.0, 0.5),
        )));
        assert!(visible_model(Mat4::from_scale(Vec3::splat(f32::NAN))));
        assert!(visible_model(Mat4::from_translation(Vec3::new(
            1.5, 0.0, 0.5
        ))));
    }

    #[test]
    fn reverse_z_and_behind_camera_are_handled() {
        let bounds = Bounds {
            min: [-0.1; 3],
            max: [0.1; 3],
        };
        let projection = Mat4::perspective_infinite_reverse_rh(1.0, 1.0, 0.1);
        assert!(visible(
            &bounds,
            projection,
            Mat4::from_translation(Vec3::new(0.0, 0.0, -1.0))
        ));
        assert!(!visible(
            &bounds,
            projection,
            Mat4::from_translation(Vec3::new(0.0, 0.0, 1.0))
        ));
    }

    #[test]
    fn runs_preserve_primitive_triangle_offsets_and_merge_adjacent_meshlets() {
        let positions = [[0.0, 0.0, 0.5], [4.0, 0.0, 0.5]];
        let mut indices = Vec::new();
        for vertex in [0, 0, 1, 0] {
            for _ in 0..124 {
                indices.extend_from_slice(&[vertex; 3]);
            }
        }
        let meshlets = build(&positions, &indices).unwrap();
        let mut runs = Vec::new();
        visible_runs(&meshlets, Mat4::IDENTITY, Mat4::IDENTITY, |first, count| {
            runs.push((first, count));
        });
        assert_eq!(runs, [(0, 2 * 124 * 3), (3 * 124 * 3, 124 * 3)]);
    }
}
