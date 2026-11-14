//! CPU meshlet preparation and conservative homogeneous-frustum testing.

use glam::Mat4;
use verity_matrix::Bounds;

/// Only draw ranges and bounds are needed: the GPU keeps the original indices.
pub(crate) struct Meshlet {
    pub(crate) first_triangle: u32,
    pub(crate) triangle_count: u32,
    pub(crate) bounds: Bounds,
}

/// Failure leaves the original draw available as an uncullled fallback.
pub(crate) fn build(positions: &[[f32; 3]], indices: &[u32]) -> Option<Vec<Meshlet>> {
    verity_matrix::build_meshlets(positions, indices)
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
    let frustum = Frustum::new(clip_from_world, model);
    for meshlet in meshlets {
        if frustum.visible(&meshlet.bounds) {
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
struct Frustum {
    /// `clip * model`, and `|clip| * |model|`, which bounds how far a box's
    /// extent can move each clip coordinate. Precomputed: they cost half of
    /// each meshlet's test.
    clip_model: Mat4,
    abs_clip_model: Mat4,
}

impl Frustum {
    fn new(clip: Mat4, model: Mat4) -> Self {
        let abs = |m: Mat4| {
            Mat4::from_cols(
                m.x_axis.abs(),
                m.y_axis.abs(),
                m.z_axis.abs(),
                m.w_axis.abs(),
            )
        };
        Self {
            clip_model: clip * model,
            abs_clip_model: abs(clip) * abs(model),
        }
    }

    fn visible(&self, bounds: &Bounds) -> bool {
        if bounds
            .min
            .iter()
            .chain(&bounds.max)
            .any(|&v| v != 0.0 && v.abs() < 2.0 * f32::MIN_POSITIVE)
        {
            return true;
        }
        let min = glam::Vec3::from_array(bounds.min) * 0.5;
        let max = glam::Vec3::from_array(bounds.max) * 0.5;
        let center = (min + max).extend(1.0);
        let extent = (max - min).extend(0.0);
        let p = self.clip_model * center;
        let radius = self.abs_clip_model * extent;
        let magnitude = self.abs_clip_model * (center.abs() + extent);
        if !p.is_finite() || !radius.is_finite() || !magnitude.is_finite() {
            return true;
        }
        let epsilon = 1.0e-5 * magnitude.max_element().max(1.0);
        let distances = [
            p.x + p.w + radius.x + radius.w,
            p.w - p.x + radius.w + radius.x,
            p.y + p.w + radius.y + radius.w,
            p.w - p.y + radius.w + radius.y,
            p.z + radius.z,
            p.w - p.z + radius.w + radius.z,
        ];
        !distances.into_iter().any(|distance| distance < -epsilon)
    }
}

/// Whether any of `bounds`, placed by `model`, may land inside `clip_from_world`.
pub(crate) fn visible(bounds: &Bounds, clip_from_world: Mat4, model: Mat4) -> bool {
    Frustum::new(clip_from_world, model).visible(bounds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec3, Vec4};

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
    fn subnormal_bounds_can_be_scaled_into_view() {
        let bounds = Bounds {
            min: [0.0; 3],
            max: [f32::from_bits(1), 0.0, 0.0],
        };
        let model = Mat4::from_scale_rotation_translation(
            Vec3::new(1e38, 1.0, 1.0),
            glam::Quat::IDENTITY,
            Vec3::new(-1.4e-7, 0.0, 0.5),
        );
        assert!(visible(
            &bounds,
            Mat4::from_scale(Vec3::new(1e8, 1.0, 1.0)),
            model
        ));
    }

    #[test]
    fn intersecting_bounds_are_not_culled() {
        let projection = Mat4::perspective_infinite_reverse_rh(1.0, 1.5, 0.1);
        let bounds = Bounds {
            min: [-0.5; 3],
            max: [0.5; 3],
        };
        for x in -12..=12 {
            for z in -12..=2 {
                let model = Mat4::from_scale_rotation_translation(
                    Vec3::new(-1.3, 0.7, 2.0),
                    glam::Quat::from_rotation_y(0.7),
                    Vec3::new(x as f32 * 0.25, 0.1, z as f32 * 0.25),
                );
                let corners: Vec<_> = [-0.5, 0.5]
                    .into_iter()
                    .flat_map(|x| {
                        [-0.5, 0.5].into_iter().flat_map(move |y| {
                            [-0.5, 0.5]
                                .into_iter()
                                .map(move |z| projection * model * Vec4::new(x, y, z, 1.0))
                        })
                    })
                    .collect();
                let outside = (0..6).any(|plane| {
                    corners.iter().all(|p| match plane {
                        0 => p.x + p.w < -1e-4,
                        1 => p.w - p.x < -1e-4,
                        2 => p.y + p.w < -1e-4,
                        3 => p.w - p.y < -1e-4,
                        4 => p.z < -1e-4,
                        _ => p.w - p.z < -1e-4,
                    })
                });
                if !outside {
                    assert!(visible(&bounds, projection, model));
                }
            }
        }
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
