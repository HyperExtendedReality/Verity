//! Viewport picking for the editor's cube entities.

use egui::{Pos2, Rect};
use glam::{Mat4, Vec3};
use verity_core::Scene;
use verity_render::Camera;

/// The nearest visible entity under a point in the viewport.
pub fn entity(scene: &Scene, camera: &Camera, viewport: Rect, point: Pos2) -> Option<usize> {
    if !viewport.contains(point)
        || viewport.width() <= 0.0
        || viewport.height() <= 0.0
        || !camera.eye.is_finite()
        || !camera.target.is_finite()
        || !camera.vertical_fov.is_finite()
        || !(0.01..3.13).contains(&camera.vertical_fov)
    {
        return None;
    }
    let forward = camera.target - camera.eye;
    if forward.length_squared() < 1e-8 || forward.cross(verity_render::UP).length_squared() < 1e-8 {
        return None;
    }
    let mut projection = Mat4::perspective_infinite_reverse_rh(
        camera.vertical_fov,
        viewport.width() / viewport.height(),
        0.1,
    );
    projection.y_axis.y *= -1.0;
    let inverse = (projection * Mat4::look_at_rh(camera.eye, camera.target, verity_render::UP)).inverse();
    let ndc = Vec3::new(
        (point.x - viewport.left()) * 2.0 / viewport.width() - 1.0,
        (point.y - viewport.top()) * 2.0 / viewport.height() - 1.0,
        1.0,
    );
    let origin = inverse.project_point3(ndc);
    let direction = (origin - camera.eye).normalize_or_zero();
    if !origin.is_finite() || !direction.is_finite() || direction == Vec3::ZERO {
        return None;
    }

    scene.entities.iter().enumerate().filter_map(|(index, entity)| {
        let matrix = verity_render::entity_matrix(&entity.transform)?;
        if !matrix.is_finite() {
            return None;
        }
        let inverse = matrix.inverse();
        let origin = inverse.transform_point3(origin);
        let direction = inverse.transform_vector3(direction);
        cube_hit(origin, direction).map(|distance| (index, distance))
    }).min_by(|a, b| a.1.total_cmp(&b.1)).map(|(index, _)| index)
}

fn cube_hit(origin: Vec3, direction: Vec3) -> Option<f32> {
    if !origin.is_finite() || !direction.is_finite() {
        return None;
    }
    let (mut near, mut far) = (0.0f32, f32::INFINITY);
    for axis in 0..3 {
        if direction[axis] == 0.0 {
            if !(-0.5..=0.5).contains(&origin[axis]) {
                return None;
            }
            continue;
        }
        let a = (-0.5 - origin[axis]) / direction[axis];
        let b = (0.5 - origin[axis]) / direction[axis];
        near = near.max(a.min(b));
        far = far.min(a.max(b));
        if near > far {
            return None;
        }
    }
    Some(near)
}

#[cfg(test)]
mod tests {
    use super::*;
    use verity_core::{Entity, Transform};

    #[test]
    fn picks_nearest_cube_at_the_clicked_viewport_position() {
        let camera = Camera { eye: Vec3::new(0.0, -5.0, 0.0), target: Vec3::ZERO, vertical_fov: 60f32.to_radians() };
        let viewport = Rect::from_min_size(Pos2::new(30.0, 40.0), egui::vec2(800.0, 600.0));
        let mut scene = Scene::default();
        scene.entities = [2.0, 0.0].map(|y| Entity { transform: Transform { translation: [0.0, y, 0.0], ..Default::default() }, ..Default::default() }).to_vec();
        assert_eq!(entity(&scene, &camera, viewport, viewport.center()), Some(1));
        assert_eq!(entity(&scene, &camera, viewport, viewport.left_top()), None);
        scene.entities[1].transform.scale = [0.0; 3];
        assert_eq!(entity(&scene, &camera, viewport, viewport.center()), Some(0));
    }

    #[test]
    fn transformed_cubes_and_parallel_rays_are_handled() {
        let rotation = glam::Quat::from_rotation_z(0.7);
        let camera = Camera { eye: Vec3::new(0.0, -5.0, 0.0), target: Vec3::ZERO, vertical_fov: 60f32.to_radians() };
        let viewport = Rect::from_min_size(Pos2::ZERO, egui::vec2(800.0, 600.0));
        let mut scene = Scene::default();
        scene.entities = vec![Entity { transform: Transform { rotation: rotation.to_array(), scale: [-2.0, 0.5, 1.0], ..Default::default() }, ..Default::default() }];
        assert_eq!(entity(&scene, &camera, viewport, viewport.center()), Some(0));
        assert!(cube_hit(Vec3::new(1.0, 0.0, 0.0), Vec3::Y).is_none());
        assert_eq!(cube_hit(Vec3::ZERO, Vec3::Y), Some(0.0));
    }
}
