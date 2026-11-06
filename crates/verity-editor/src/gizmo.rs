use egui::{Color32, Pos2, Rect, Sense, Stroke, Vec2};
use glam::{Mat4, Quat, Vec3};
use verity_core::Transform;
use verity_render::Camera;

const AXES: [Vec3; 3] = [Vec3::X, Vec3::Y, Vec3::Z];
const COLORS: [Color32; 3] = [
    Color32::from_rgb(240, 75, 75),
    Color32::from_rgb(92, 220, 100),
    Color32::from_rgb(80, 150, 255),
];
const HIT_RADIUS: f32 = 8.0;
const RING_SEGMENTS: usize = 64;

/// The active viewport tool.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    /// Pick only; no gizmo.
    Select,
    #[default]
    Translate,
    Rotate,
    Scale,
}

struct Drag {
    mode: Mode,
    axis: usize,
    start: Pos2,
    initial: Transform,
    last: Transform,
    // Screen-space derivative, in points per world unit or radian.
    derivative: Vec2,
    length: f32,
}

/// A viewport-space editor for a selected entity's transform.
#[derive(Default)]
pub struct Gizmo {
    mode: Mode,
    drag: Option<Drag>,
}

impl Gizmo {
    /// Cancel an in-progress interaction when the selected entity changes.
    pub fn cancel_drag(&mut self) {
        self.drag = None;
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    pub fn set_mode(&mut self, mode: Mode) {
        if mode != self.mode {
            self.mode = mode;
            self.drag = None;
        }
    }

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        viewport: Rect,
        transform: &mut Transform,
        camera: &Camera,
    ) -> bool {
        // External edits invalidate the drag snapshot. The caller must also
        // cancel the drag on selection changes, even if transforms match.
        if self
            .drag
            .as_ref()
            .is_some_and(|drag| drag.last != *transform)
        {
            self.drag = None;
        }
        if self.mode == Mode::Select {
            return false;
        }
        let Some(projection) = Projection::new(viewport, camera, transform) else {
            self.drag = None;
            return false;
        };
        let painter = ui
            .painter()
            .with_clip_rect(viewport.intersect(ui.clip_rect()));
        let pointer = ui.input(|input| input.pointer.interact_pos());
        let pointer_inside =
            pointer.is_some_and(|p| viewport.intersect(ui.clip_rect()).contains(p));
        let mut hit: Option<(f32, usize, Vec2)> = None;
        let origin = Vec3::from_array(transform.translation);
        let rotation = normalized_rotation(transform.rotation);

        for axis in 0..3 {
            let color = COLORS[axis];
            if self.mode == Mode::Rotate {
                let (u, v) = ring_basis(axis);
                let mut previous: Option<(Pos2, f32)> = None;
                for step in 0..=RING_SEGMENTS {
                    let angle = std::f32::consts::TAU * step as f32 / RING_SEGMENTS as f32;
                    let point = origin + projection.length * (u * angle.cos() + v * angle.sin());
                    let projected = projection.project(point);
                    if let (Some((last_pos, last_angle)), Some(pos)) = (previous, projected) {
                        painter.line_segment([last_pos, pos], Stroke::new(2.5, color));
                        if pointer_inside {
                            let pointer = pointer.unwrap();
                            let (distance, fraction) = segment_distance(pointer, last_pos, pos);
                            let theta = last_angle + (angle - last_angle) * fraction;
                            let tangent = projection.length * (-u * theta.sin() + v * theta.cos());
                            let world_point =
                                origin + projection.length * (u * theta.cos() + v * theta.sin());
                            let epsilon = 0.01;
                            if let (Some(a), Some(b)) = (
                                projection.project(world_point - tangent * epsilon),
                                projection.project(world_point + tangent * epsilon),
                            ) {
                                consider_hit(&mut hit, distance, axis, (b - a) / (2.0 * epsilon));
                            }
                        }
                    }
                    previous = projected.map(|p| (p, angle));
                }
            } else {
                let direction = if self.mode == Mode::Scale {
                    rotation * AXES[axis]
                } else {
                    AXES[axis]
                };
                let Some(end) = projection.project(origin + direction * projection.length) else {
                    continue;
                };
                let start = projection.center;
                let screen_axis = end - start;
                if screen_axis.length_sq() < 16.0 {
                    continue;
                }
                painter.line_segment([start, end], Stroke::new(3.0, color));
                if self.mode == Mode::Scale {
                    painter.rect_filled(Rect::from_center_size(end, Vec2::splat(9.0)), 1.0, color);
                } else {
                    painter.circle_filled(end, 5.0, color);
                }
                if pointer_inside {
                    let (distance, _) =
                        segment_distance(pointer.unwrap(), start + screen_axis * 0.15, end);
                    consider_hit(&mut hit, distance, axis, screen_axis / projection.length);
                }
            }
        }

        let down = ui.input(|input| input.pointer.primary_down());
        if !down {
            self.drag = None;
        }
        // Only register a widget when the pointer is actually on a handle (or
        // while that widget owns a drag), rather than covering the viewport.
        if let Some(pos) = pointer
            && (hit.is_some() || self.drag.is_some())
        {
            let widget_pos = Pos2::new(
                pos.x.clamp(viewport.left(), viewport.right()),
                pos.y.clamp(viewport.top(), viewport.bottom()),
            );
            let response = ui.interact(
                Rect::from_center_size(widget_pos, Vec2::splat(HIT_RADIUS * 2.0))
                    .intersect(viewport),
                ui.make_persistent_id("transform_gizmo"),
                Sense::drag(),
            );
            if response.is_pointer_button_down_on()
                && self.drag.is_none()
                && let Some((_, axis, derivative)) = hit
            {
                self.drag = Some(Drag {
                    mode: self.mode,
                    axis,
                    start: pos,
                    initial: transform.clone(),
                    last: transform.clone(),
                    derivative,
                    length: projection.length,
                });
            }
            if response.dragged()
                && let Some(drag) = &mut self.drag
            {
                let amount = (pos - drag.start).dot(drag.derivative) / drag.derivative.length_sq();
                if amount.is_finite() {
                    let mut result = drag.initial.clone();
                    match drag.mode {
                        // Select never starts a drag.
                        Mode::Select => {}
                        Mode::Translate => {
                            result.translation = (Vec3::from_array(drag.initial.translation)
                                + AXES[drag.axis] * amount)
                                .to_array();
                        }
                        Mode::Rotate => {
                            result.rotation = (Quat::from_axis_angle(AXES[drag.axis], amount)
                                * normalized_rotation(drag.initial.rotation))
                            .normalize()
                            .to_array();
                        }
                        Mode::Scale => {
                            let initial = drag.initial.scale[drag.axis];
                            let sign = if initial < 0.0 { -1.0 } else { 1.0 };
                            // One handle length corresponds to the original scale magnitude.
                            let magnitude = (initial.abs().max(0.001)
                                * (1.0 + amount / drag.length))
                                .max(0.001);
                            result.scale[drag.axis] = sign * magnitude;
                        }
                    }
                    if result
                        .translation
                        .iter()
                        .chain(&result.rotation)
                        .chain(&result.scale)
                        .all(|v| v.is_finite())
                        && result != *transform
                    {
                        *transform = result;
                        drag.last = transform.clone();
                        return true;
                    }
                }
            }
        }
        false
    }
}

fn normalized_rotation(rotation: [f32; 4]) -> Quat {
    let q = Quat::from_array(rotation);
    if q.is_finite() && q.length_squared() > 1e-12 {
        q.normalize()
    } else {
        Quat::IDENTITY
    }
}

fn ring_basis(axis: usize) -> (Vec3, Vec3) {
    match axis {
        0 => (Vec3::Y, Vec3::Z),
        1 => (Vec3::Z, Vec3::X),
        _ => (Vec3::X, Vec3::Y),
    }
}

fn segment_distance(p: Pos2, a: Pos2, b: Pos2) -> (f32, f32) {
    let line = b - a;
    let length_sq = line.length_sq();
    let t = if length_sq > 1e-6 {
        ((p - a).dot(line) / length_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (p.distance(a + line * t), t)
}

fn consider_hit(
    hit: &mut Option<(f32, usize, Vec2)>,
    distance: f32,
    axis: usize,
    derivative: Vec2,
) {
    if distance <= HIT_RADIUS
        && derivative.length_sq() >= 1e-8
        && derivative.is_finite()
        && hit.as_ref().is_none_or(|(best, _, _)| distance < *best)
    {
        *hit = Some((distance, axis, derivative));
    }
}

struct Projection {
    matrix: Mat4,
    viewport: Rect,
    center: Pos2,
    length: f32,
}

impl Projection {
    fn new(viewport: Rect, camera: &Camera, transform: &Transform) -> Option<Self> {
        if viewport.width() < 16.0
            || viewport.height() < 16.0
            || !camera.eye.is_finite()
            || !camera.target.is_finite()
            || !camera.vertical_fov.is_finite()
            || !(0.01..3.13).contains(&camera.vertical_fov)
        {
            return None;
        }
        let forward = camera.target - camera.eye;
        if forward.length_squared() < 1e-8
            || forward.cross(verity_render::UP).length_squared() < 1e-8
        {
            return None;
        }
        let origin = Vec3::from_array(transform.translation);
        if !origin.is_finite() {
            return None;
        }
        let depth = (origin - camera.eye).dot(forward.normalize());
        if depth <= 0.1 {
            return None;
        }
        let mut perspective = Mat4::perspective_infinite_reverse_rh(
            camera.vertical_fov,
            viewport.width() / viewport.height(),
            0.1,
        );
        perspective.y_axis.y *= -1.0;
        let matrix = perspective * Mat4::look_at_rh(camera.eye, camera.target, verity_render::UP);
        let length = depth * (camera.vertical_fov * 0.5).tan() * 2.0 * 72.0 / viewport.height();
        let mut result = Self {
            matrix,
            viewport,
            center: viewport.center(),
            length,
        };
        result.center = result.project(origin)?;
        Some(result)
    }

    fn project(&self, point: Vec3) -> Option<Pos2> {
        let clip = self.matrix * point.extend(1.0);
        if !clip.is_finite() || clip.w <= 0.1 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        Some(Pos2::new(
            self.viewport.left() + (ndc.x + 1.0) * 0.5 * self.viewport.width(),
            self.viewport.top() + (ndc.y + 1.0) * 0.5 * self.viewport.height(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dragging_handles_edits_the_selected_transform() {
        let viewport = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
        let camera = Camera {
            eye: Vec3::new(0.0, -5.0, 0.0),
            target: Vec3::ZERO,
            vertical_fov: std::f32::consts::FRAC_PI_2,
        };
        for mode in [Mode::Translate, Mode::Rotate, Mode::Scale] {
            let context = egui::Context::default();
            let mut gizmo = Gizmo { mode, drag: None };
            let mut transform = Transform::default();
            let projection = Projection::new(viewport, &camera, &transform).unwrap();
            let (start, delta) = if mode == Mode::Rotate {
                let angle = std::f32::consts::FRAC_PI_4;
                (
                    projection
                        .project(Vec3::new(angle.cos(), angle.sin(), 0.0) * projection.length)
                        .unwrap(),
                    Vec2::new(-24.0, -24.0),
                )
            } else {
                (
                    projection
                        .project(Vec3::X * projection.length * 0.8)
                        .unwrap(),
                    Vec2::new(30.0, 0.0),
                )
            };
            let mut frame = |events| {
                let input = egui::RawInput {
                    screen_rect: Some(viewport),
                    events,
                    ..Default::default()
                };
                let mut output = context.run_ui(input, |ui| {
                    gizmo.show(ui, viewport, &mut transform, &camera);
                });
                output.textures_delta.clear();
            };
            frame(vec![]);
            frame(vec![egui::Event::PointerMoved(start)]);
            frame(vec![egui::Event::PointerButton {
                pos: start,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            }]);
            frame(vec![egui::Event::PointerMoved(start + delta)]);
            frame(vec![egui::Event::PointerButton {
                pos: start + delta,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            }]);
            match mode {
                Mode::Translate => assert!(transform.translation[0] > 0.1, "{:?}", transform),
                Mode::Rotate => assert!(transform.rotation[2].abs() > 0.1, "{:?}", transform),
                Mode::Scale => assert!(transform.scale[0] > 1.1, "{:?}", transform),
                Mode::Select => unreachable!(),
            }
        }
    }

    #[test]
    fn projection_matches_vulkan_viewport_orientation() {
        let viewport = Rect::from_min_size(Pos2::new(40.0, 20.0), Vec2::new(800.0, 400.0));
        let camera = Camera {
            eye: Vec3::new(0.0, -5.0, 0.0),
            target: Vec3::ZERO,
            vertical_fov: std::f32::consts::FRAC_PI_2,
        };
        let projection = Projection::new(viewport, &camera, &Transform::default()).unwrap();
        let center = projection.project(Vec3::ZERO).unwrap();
        assert!(center.distance(viewport.center()) < 0.001);
        // Z is up on screen; behind the camera does not project.
        assert!(projection.project(Vec3::Z).unwrap().y < center.y);
        assert!(projection.project(Vec3::X).unwrap().x > center.x);
        assert!(projection.project(Vec3::new(0.0, -6.0, 0.0)).is_none());
    }

    #[test]
    fn hit_testing_clamps_to_segment() {
        let (distance, fraction) = segment_distance(
            Pos2::new(15.0, 4.0),
            Pos2::new(0.0, 0.0),
            Pos2::new(10.0, 0.0),
        );
        assert_eq!(fraction, 1.0);
        assert!((distance - 41.0_f32.sqrt()).abs() < 1e-5);
    }
}
