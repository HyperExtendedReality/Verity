use std::path::Path;

use egui::{
    Color32, CornerRadius, DragValue, Frame, Margin, Pos2, Rect, RichText, Sense, Stroke, Vec2,
};
use glam::Vec3;
use verity_core::{
    ComponentRegistry, Entity, FieldMut, LightingMethod, Material, PointLight, Scene, Transform,
};
use verity_render::UP;
use winit::window::ResizeDirection;

use crate::gizmo::Mode;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selection {
    Entity(usize),
    Light(usize),
    Camera(usize),
    Sun,
    Lighting,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    New,
    Open,
    Save,
    SaveAs,
    Play,
    Pause,
    Step,
    Stop,
    Undo,
    Redo,
}

/// Whether the scene is being edited or simulated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlayState {
    #[default]
    Editing,
    Playing,
    Paused,
}

/// What the custom title bar asks of the window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowCommand {
    Minimize,
    ToggleMaximize,
    Close,
    Drag,
    Resize(ResizeDirection),
}

pub struct UiResponse {
    pub action: Option<Action>,
    pub asset_action: Option<crate::asset_browser::AssetAction>,
    pub asset_remove: Option<usize>,
    pub window: Option<WindowCommand>,
    pub changed: bool,
    /// The unobscured scene viewport, in egui points.
    pub viewport: egui::Rect,
}

/// Fly-camera orientation and speed; `Tools::camera` follows it.
pub struct Fly {
    /// Radians about Z from +X.
    pub yaw: f32,
    /// Radians above the horizon.
    pub pitch: f32,
    /// Metres per second.
    pub speed: f32,
}

impl Fly {
    fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cp * cy, cp * sy, sp)
    }
}

pub struct Tools {
    pub browser: crate::asset_browser::AssetBrowser,
    pub profiler: crate::profiler::Profiler,
    pub gizmo: crate::gizmo::Gizmo,
    pub camera: verity_render::Camera,
    pub fly: Fly,
    /// The right mouse button is held in the viewport: WASD moves the camera.
    pub flying: bool,
    pub maximized: bool,
    pub light_limit: usize,
    /// Enable Matrix geometry selection for the current editor session.
    pub matrix_enabled: bool,
    /// Display the renderer's Matrix geometry diagnostics.
    pub matrix_debug: bool,
    /// Draw as fast as the GPU allows instead of at the display's refresh rate.
    pub uncapped: bool,
    /// GI readiness, shown while the scene uses Split Radiance Cascades.
    pub gi_status: Option<String>,
    /// While playing: each jointed entity's load as text, and its share of
    /// breaking, by entity index.
    pub joint_status: std::collections::HashMap<usize, (String, f32)>,
    /// Component types the inspector can add and edit.
    pub registry: ComponentRegistry,
    /// While not editing, the scene shows the simulation and is read-only.
    pub play: PlayState,
    /// Whether the editor has steps to undo and redo.
    pub can_undo: bool,
    pub can_redo: bool,
}

impl Default for Tools {
    fn default() -> Self {
        let camera = verity_render::Camera::default();
        let direction = (camera.target - camera.eye).normalize();
        Self {
            browser: Default::default(),
            profiler: Default::default(),
            gizmo: Default::default(),
            fly: Fly {
                yaw: direction.y.atan2(direction.x),
                pitch: direction.z.asin(),
                speed: 5.0,
            },
            camera,
            flying: false,
            maximized: false,
            light_limit: verity_render::MAX_LIGHTS,
            matrix_enabled: false,
            matrix_debug: false,
            uncapped: false,
            gi_status: None,
            joint_status: Default::default(),
            registry: verity_runtime::registry(),
            play: PlayState::Editing,
            can_undo: false,
            can_redo: false,
        }
    }
}

// Neutral dark grays; hacker green for focus, synthwave neons for kinds of thing.
pub(crate) const BACKGROUND: Color32 = Color32::from_rgb(22, 22, 22);
pub(crate) const PANEL: Color32 = Color32::from_rgb(31, 32, 34);
pub(crate) const BAR: Color32 = BACKGROUND;
pub(crate) const SURFACE: Color32 = Color32::from_rgb(42, 43, 46);
pub(crate) const HOVER: Color32 = Color32::from_rgb(52, 54, 58);
pub(crate) const BORDER: Color32 = Color32::from_rgb(46, 48, 52);
pub(crate) const TEXT: Color32 = Color32::from_rgb(228, 231, 233);
pub(crate) const MUTED: Color32 = Color32::from_rgb(138, 144, 150);
pub(crate) const ACCENT: Color32 = Color32::from_rgb(57, 255, 136);
pub(crate) const ACCENT_DARK: Color32 = Color32::from_rgb(22, 56, 38);
pub(crate) const PINK: Color32 = Color32::from_rgb(255, 61, 172);
pub(crate) const CYAN: Color32 = Color32::from_rgb(38, 214, 255);
pub(crate) const AMBER: Color32 = Color32::from_rgb(255, 196, 64);
const RADIUS: u8 = 10;
const CLOSE_HOVER: Color32 = Color32::from_rgb(214, 38, 90);
const TITLE_HEIGHT: f32 = 36.0;
/// How close to a window edge the pointer resizes instead of interacting.
const RESIZE_MARGIN: f32 = 6.0;

fn panel_frame(color: Color32) -> Frame {
    Frame::new()
        .fill(color)
        .stroke(Stroke::new(1.0, BORDER))
        .corner_radius(RADIUS)
        .inner_margin(Margin::symmetric(12, 10))
        .outer_margin(Margin::same(4))
}

/// The pink-to-cyan-to-green synthwave rule under the title bar.
fn neon_rule(painter: &egui::Painter, rect: Rect) {
    let mut mesh = egui::Mesh::default();
    let stops = [PINK, CYAN, ACCENT];
    for (i, color) in stops.iter().enumerate() {
        let x = rect.left() + rect.width() * i as f32 / (stops.len() - 1) as f32;
        mesh.colored_vertex(Pos2::new(x, rect.top()), *color);
        mesh.colored_vertex(Pos2::new(x, rect.bottom()), *color);
        if i > 0 {
            let v = i as u32 * 2;
            mesh.add_triangle(v - 2, v - 1, v);
            mesh.add_triangle(v - 1, v, v + 1);
        }
    }
    painter.add(mesh);
}

/// Background-coloured fillets over the scene's square corners, so the
/// viewport reads as a rounded card like the panels.
fn round_corners(painter: &egui::Painter, rect: Rect, radius: f32) {
    let corners = [
        (rect.left_top(), egui::vec2(1.0, 1.0)),
        (rect.right_top(), egui::vec2(-1.0, 1.0)),
        (rect.right_bottom(), egui::vec2(-1.0, -1.0)),
        (rect.left_bottom(), egui::vec2(1.0, -1.0)),
    ];
    for (corner, inward) in corners {
        let center = corner + inward * radius;
        let mut mesh = egui::Mesh::default();
        mesh.colored_vertex(corner, BACKGROUND);
        for step in 0..=8 {
            let a = step as f32 / 8.0 * std::f32::consts::FRAC_PI_2;
            let arc = center - egui::vec2(inward.x * a.cos(), inward.y * a.sin()) * radius;
            mesh.colored_vertex(arc, BACKGROUND);
            if step > 0 {
                mesh.add_triangle(0, step, step + 1);
            }
        }
        painter.add(mesh);
    }
    painter.rect_stroke(
        rect,
        radius,
        Stroke::new(1.0, BORDER),
        egui::StrokeKind::Inside,
    );
}

fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(10.0);
    ui.label(RichText::new(title).size(10.0).strong().color(MUTED));
    ui.add_space(4.0);
}

fn panel_heading(ui: &mut egui::Ui, title: &str, detail: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(title).size(12.5).strong().color(TEXT));
        ui.label(RichText::new(detail).size(10.5).color(MUTED));
    });
    ui.add_space(6.0);
    let rect = ui.available_rect_before_wrap();
    ui.painter()
        .hline(rect.x_range(), rect.top(), Stroke::new(1.0, BORDER));
    ui.add_space(2.0);
}

fn brand_mark(ui: &mut egui::Ui) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(20.0, 20.0), Sense::hover());
    ui.painter().rect_filled(rect, 5, ACCENT);
    let c = rect.center();
    let stroke = Stroke::new(1.8, BACKGROUND);
    for x in [-4.0, 4.0] {
        ui.painter()
            .line_segment([c + egui::vec2(x, -5.0), c + egui::vec2(x, 5.0)], stroke);
    }
    ui.painter().line_segment(
        [c + egui::vec2(-4.0, 0.0), c + egui::vec2(4.0, 0.0)],
        stroke,
    );
}

fn hierarchy_row(ui: &mut egui::Ui, name: &str, selected: bool, light: bool) -> bool {
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), 26.0), Sense::click());
    let painter = ui.painter();
    if selected {
        painter.rect_filled(rect, 6, ACCENT_DARK);
        painter.rect_filled(
            Rect::from_min_size(rect.min, egui::vec2(2.0, rect.height())),
            1,
            ACCENT,
        );
    } else if response.hovered() {
        painter.rect_filled(rect, 6, HOVER);
    }
    let icon = rect.left_center() + egui::vec2(14.0, 0.0);
    let color = if light { PINK } else { CYAN };
    if light {
        painter.circle_stroke(icon, 4.0, Stroke::new(1.2, color));
        painter.circle_filled(icon, 1.3, color);
    } else {
        let cube = Rect::from_center_size(icon, egui::vec2(8.0, 8.0));
        painter.rect_stroke(cube, 1, Stroke::new(1.2, color), egui::StrokeKind::Inside);
    }
    let galley = painter.layout_no_wrap(
        name.to_owned(),
        egui::FontId::proportional(12.5),
        if selected {
            TEXT
        } else {
            Color32::from_gray(214)
        },
    );
    let text_rect = Rect::from_min_max(
        rect.left_top() + egui::vec2(28.0, 0.0),
        rect.right_bottom() - egui::vec2(6.0, 0.0),
    );
    painter.with_clip_rect(text_rect).galley(
        Pos2::new(text_rect.left(), rect.center().y - galley.size().y / 2.0),
        galley,
        TEXT,
    );
    response.clicked()
}

fn vector<const N: usize>(ui: &mut egui::Ui, label: &str, values: &mut [f32; N]) -> bool {
    let mut changed = false;
    ui.label(RichText::new(label).size(11.5).color(MUTED));
    ui.horizontal(|ui| {
        let width = ((ui.available_width()
            - (N.saturating_sub(1) as f32 * ui.spacing().item_spacing.x))
            / N as f32)
            .max(35.0);
        for (index, value) in values.iter_mut().enumerate() {
            // Axis tint on the label, like the gizmo's colours.
            let (prefix, tint) = [
                ("X", Color32::from_rgb(224, 92, 84)),
                ("Y", Color32::from_rgb(120, 196, 110)),
                ("Z", Color32::from_rgb(96, 150, 232)),
                ("W", MUTED),
            ][index];
            let field = ui.add_sized(
                [width, 22.0],
                DragValue::new(value)
                    .speed(0.02)
                    .prefix(format!("{prefix}  ")),
            );
            ui.painter().vline(
                field.rect.left() + 1.0,
                field.rect.y_range().shrink(5.0),
                Stroke::new(2.0, tint),
            );
            changed |= field.changed();
        }
    });
    changed
}

fn scalar(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    speed: f32,
) -> bool {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).size(11.5).color(MUTED));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_sized(
                [110.0, 22.0],
                DragValue::new(value).speed(speed).range(range),
            )
            .changed()
        })
        .inner
    })
    .inner
}

fn color(ui: &mut egui::Ui, label: &str, value: &mut [f32; 3]) -> bool {
    let mut edited = *value;
    let changed = ui
        .horizontal(|ui| {
            ui.label(RichText::new(label).size(11.5).color(MUTED));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.color_edit_button_rgb(&mut edited).changed()
            })
            .inner
        })
        .inner;
    if changed && edited != *value {
        *value = edited;
        true
    } else {
        false
    }
}

/// An entity's components, edited through their reflected fields. Components
/// the registry does not know are kept as saved.
fn components(
    ui: &mut egui::Ui,
    registry: &ComponentRegistry,
    components: &mut std::collections::BTreeMap<String, serde_json::Value>,
    names: &[String],
    joint_status: Option<&(String, f32)>,
) -> bool {
    let mut changed = false;
    let mut remove = None;
    for (name, value) in components.iter_mut() {
        ui.push_id(name.as_str(), |ui| {
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(name.to_uppercase())
                        .size(10.0)
                        .strong()
                        .color(MUTED),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .small_button("✕")
                        .on_hover_text("Remove component")
                        .clicked()
                    {
                        remove = Some(name.clone());
                    }
                });
            });
            ui.add_space(4.0);
            match registry.get(name) {
                None => {
                    ui.label(RichText::new("Unknown component, kept as saved").color(MUTED));
                }
                Some(info) => match (info.inspect)(value, &mut |f| fields(ui, f, names)) {
                    Ok(edited) => changed |= edited,
                    Err(error) => {
                        ui.label(RichText::new(error).color(Color32::from_rgb(224, 92, 84)));
                    }
                },
            }
            if name == "Soft Body" {
                ui.label(
                    RichText::new("Replaces this entity's cube during Play.")
                        .size(11.0)
                        .color(MUTED),
                );
            }
            if name == "Joint" {
                ui.label(
                    RichText::new("A preset material replaces the stiffness and load fields.")
                        .size(11.0)
                        .color(MUTED),
                );
                if let Some((text, share)) = joint_status {
                    let color = match *share {
                        s if s >= 1.0 => Color32::from_rgb(224, 92, 84),
                        s if s >= 0.5 => Color32::from_rgb(232, 180, 80),
                        _ => MUTED,
                    };
                    ui.label(RichText::new(text).size(11.5).color(color));
                }
            }
            if name == "Rigid Body" {
                ui.label(
                    RichText::new(
                        "Collider size is separate from visual scale. Spheres are shown as cubes.",
                    )
                    .size(11.0)
                    .color(MUTED),
                );
            }
        });
    }
    if let Some(name) = remove {
        components.remove(&name);
        changed = true;
    }
    let missing: Vec<_> = registry
        .iter()
        .filter(|info| info.name != "Rigid Body" && !components.contains_key(info.name))
        .collect();
    if !missing.is_empty() {
        ui.add_space(10.0);
        ui.menu_button("＋ Add component", |ui| {
            for info in missing {
                if ui.button(info.name).clicked() {
                    components.insert(info.name.to_owned(), (info.default)());
                    changed = true;
                    ui.close();
                }
            }
        });
    }
    changed
}

/// A widget per reflected field; true if any changed.
fn fields(
    ui: &mut egui::Ui,
    fields: &mut [(&'static str, FieldMut<'_>)],
    names: &[String],
) -> bool {
    let mut changed = false;
    for (label, field) in fields.iter_mut() {
        changed |= match field {
            FieldMut::Float(value, range) => scalar(ui, label, value, range.clone(), 0.01),
            FieldMut::Vec3(values, range) => {
                let edited = vector(ui, label, values);
                if edited {
                    for value in values.iter_mut() {
                        *value = value.clamp(*range.start(), *range.end());
                    }
                }
                edited
            }
            FieldMut::Color(value) => color(ui, label, value),
            FieldMut::Bool(value) => ui.checkbox(value, *label).changed(),
            FieldMut::Entity(value) => {
                let mut edited = false;
                ui.horizontal(|ui| {
                    ui.label(RichText::new(*label).size(11.5).color(MUTED));
                    let shown = if value.is_empty() {
                        "World"
                    } else {
                        value.as_str()
                    };
                    egui::ComboBox::from_id_salt(*label)
                        .width(ui.available_width())
                        .selected_text(shown.to_owned())
                        .show_ui(ui, |ui| {
                            edited |= ui.selectable_label(value.is_empty(), "World").clicked()
                                && !value.is_empty();
                            if edited {
                                value.clear();
                            }
                            for name in names {
                                if ui.selectable_label(**value == *name, name).clicked()
                                    && **value != *name
                                {
                                    **value = name.clone();
                                    edited = true;
                                }
                            }
                        });
                });
                edited
            }
            FieldMut::Choice(choice) => {
                let options = choice.options();
                let mut index = choice.index();
                ui.horizontal(|ui| {
                    ui.label(RichText::new(*label).size(11.5).color(MUTED));
                    egui::ComboBox::from_id_salt(*label)
                        .width(ui.available_width())
                        .selected_text(options[index])
                        .show_ui(ui, |ui| {
                            for (i, option) in options.iter().enumerate() {
                                ui.selectable_value(&mut index, i, *option);
                            }
                        });
                });
                let edited = index != choice.index();
                if edited {
                    choice.select(index);
                }
                edited
            }
        };
    }
    changed
}

/// One of the title bar's window buttons, drawn like the platform's own.
fn window_button(ui: &mut egui::Ui, glyph: WindowCommand, maximized: bool) -> bool {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(46.0, TITLE_HEIGHT), Sense::click());
    let close = glyph == WindowCommand::Close;
    if response.hovered() {
        ui.painter()
            .rect_filled(rect, 0, if close { CLOSE_HOVER } else { HOVER });
    }
    let c = rect.center();
    let stroke = Stroke::new(
        1.0,
        if response.hovered() && close {
            Color32::WHITE
        } else {
            TEXT
        },
    );
    let painter = ui.painter();
    match glyph {
        WindowCommand::Minimize => {
            painter.hline(c.x - 5.0..=c.x + 5.0, c.y, stroke);
        }
        WindowCommand::ToggleMaximize if maximized => {
            let back = Rect::from_center_size(c + egui::vec2(2.0, -2.0), egui::vec2(8.0, 8.0));
            painter.rect_stroke(back, 1, stroke, egui::StrokeKind::Inside);
            let front = Rect::from_center_size(c + egui::vec2(-1.0, 1.0), egui::vec2(8.0, 8.0));
            painter.rect_filled(front, 0, BAR);
            painter.rect_stroke(front, 1, stroke, egui::StrokeKind::Inside);
        }
        WindowCommand::ToggleMaximize => {
            painter.rect_stroke(
                Rect::from_center_size(c, egui::vec2(10.0, 10.0)),
                1,
                stroke,
                egui::StrokeKind::Inside,
            );
        }
        _ => {
            painter.line_segment(
                [c + egui::vec2(-5.0, -5.0), c + egui::vec2(5.0, 5.0)],
                stroke,
            );
            painter.line_segment(
                [c + egui::vec2(5.0, -5.0), c + egui::vec2(-5.0, 5.0)],
                stroke,
            );
        }
    }
    response
        .on_hover_text(match glyph {
            WindowCommand::Minimize => "Minimize",
            WindowCommand::ToggleMaximize if maximized => "Restore",
            WindowCommand::ToggleMaximize => "Maximize",
            _ => "Close",
        })
        .clicked()
}

/// A toolbar tool button with a painted icon.
fn tool_button(ui: &mut egui::Ui, mode: Mode, active: bool, hint: &str) -> bool {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(30.0, 26.0), Sense::click());
    let painter = ui.painter();
    if active {
        painter.rect_filled(rect, 8, ACCENT_DARK);
        painter.rect_stroke(rect, 8, Stroke::new(1.0, ACCENT), egui::StrokeKind::Inside);
    } else if response.hovered() {
        painter.rect_filled(rect, 8, HOVER);
    }
    let c = rect.center();
    let stroke = Stroke::new(1.4, if active { ACCENT } else { TEXT });
    match mode {
        Mode::Select => {
            let tip = c + egui::vec2(-4.0, -6.0);
            let points = vec![
                tip,
                tip + egui::vec2(0.0, 11.0),
                tip + egui::vec2(3.0, 8.0),
                tip + egui::vec2(7.0, 8.0),
            ];
            painter.add(egui::Shape::closed_line(points, stroke));
        }
        Mode::Translate => {
            for d in [
                egui::vec2(1.0, 0.0),
                egui::vec2(-1.0, 0.0),
                egui::vec2(0.0, 1.0),
                egui::vec2(0.0, -1.0),
            ] {
                let end = c + d * 7.0;
                painter.line_segment([c, end], stroke);
                let side = egui::vec2(-d.y, d.x) * 2.5;
                painter.line_segment([end, end - d * 2.5 + side], stroke);
                painter.line_segment([end, end - d * 2.5 - side], stroke);
            }
        }
        Mode::Rotate => {
            let points: Vec<Pos2> = (0..=20)
                .map(|i| {
                    let a = 0.6 + i as f32 / 20.0 * 5.0;
                    c + egui::vec2(a.cos(), a.sin()) * 6.5
                })
                .collect();
            let end = *points.last().unwrap();
            painter.add(egui::Shape::line(points, stroke));
            painter.line_segment([end, end + egui::vec2(3.0, 0.5)], stroke);
            painter.line_segment([end, end + egui::vec2(0.5, 3.0)], stroke);
        }
        Mode::Scale => {
            painter.rect_stroke(
                Rect::from_center_size(c + egui::vec2(-1.5, 1.5), egui::vec2(9.0, 9.0)),
                1,
                stroke,
                egui::StrokeKind::Inside,
            );
            painter.line_segment(
                [c + egui::vec2(0.0, 0.0), c + egui::vec2(6.0, -6.0)],
                stroke,
            );
            painter.line_segment(
                [c + egui::vec2(6.0, -6.0), c + egui::vec2(2.5, -6.0)],
                stroke,
            );
            painter.line_segment(
                [c + egui::vec2(6.0, -6.0), c + egui::vec2(6.0, -2.5)],
                stroke,
            );
        }
    }
    response.on_hover_text(hint).clicked()
}

/// Which edge the pointer is on, for resizing the undecorated window.
fn resize_edge(screen: Rect, pointer: Pos2) -> Option<ResizeDirection> {
    let (left, right) = (
        pointer.x - screen.left() < RESIZE_MARGIN,
        screen.right() - pointer.x < RESIZE_MARGIN,
    );
    let (top, bottom) = (
        pointer.y - screen.top() < RESIZE_MARGIN,
        screen.bottom() - pointer.y < RESIZE_MARGIN,
    );
    Some(match (left, right, top, bottom) {
        (true, _, true, _) => ResizeDirection::NorthWest,
        (_, true, true, _) => ResizeDirection::NorthEast,
        (true, _, _, true) => ResizeDirection::SouthWest,
        (_, true, _, true) => ResizeDirection::SouthEast,
        (true, ..) => ResizeDirection::West,
        (_, true, ..) => ResizeDirection::East,
        (_, _, true, _) => ResizeDirection::North,
        (.., true) => ResizeDirection::South,
        _ => return None,
    })
}

/// Right mouse: look and fly with WASD/QE (Shift is faster, the wheel sets
/// speed). Otherwise the wheel dollies and the middle button pans.
fn navigate(ui: &egui::Ui, viewport: Rect, tools: &mut Tools) {
    let hovered = ui.rect_contains_pointer(viewport);
    let (look, pan, delta, scroll, dt, shift) = ui.input(|i| {
        (
            i.pointer.secondary_down(),
            i.pointer.middle_down(),
            i.pointer.delta(),
            i.smooth_scroll_delta.y,
            i.stable_dt.min(0.1),
            i.modifiers.shift,
        )
    });
    // Flying starts only over the viewport but continues if the pointer leaves it.
    tools.flying = look && (tools.flying || hovered);
    let fly = &mut tools.fly;
    let camera = &mut tools.camera;
    let forward = fly.forward();
    let right = forward.cross(UP).normalize_or_zero();
    if tools.flying {
        fly.yaw -= delta.x * 0.004;
        fly.pitch = (fly.pitch - delta.y * 0.004).clamp(-1.54, 1.54);
        if scroll != 0.0 {
            fly.speed = (fly.speed * 1.15f32.powf(scroll.signum())).clamp(0.25, 200.0);
        }
        let axis = |positive: egui::Key, negative: egui::Key| {
            ui.input(|i| i.key_down(positive) as i32 - i.key_down(negative) as i32) as f32
        };
        let forward = fly.forward();
        let motion = forward * axis(egui::Key::W, egui::Key::S)
            + right * axis(egui::Key::D, egui::Key::A)
            + UP * axis(egui::Key::E, egui::Key::Q);
        camera.eye += motion.normalize_or_zero() * fly.speed * if shift { 3.0 } else { 1.0 } * dt;
        ui.ctx().set_cursor_icon(egui::CursorIcon::None);
        ui.ctx().request_repaint();
    } else if hovered {
        if scroll != 0.0 {
            camera.eye += forward * scroll * 0.01 * fly.speed;
        }
        if pan && delta != Vec2::ZERO {
            let up = right.cross(forward);
            camera.eye += (-right * delta.x + up * delta.y) * 0.002 * fly.speed;
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
        }
    }
    camera.target = camera.eye + fly.forward();
}

/// An XYZ orientation triad in the viewport corner.
fn orientation(ui: &egui::Ui, viewport: Rect, fly: &Fly) {
    let forward = fly.forward();
    let right = forward.cross(UP).normalize_or_zero();
    let up = right.cross(forward);
    let center = Pos2::new(viewport.left() + 38.0, viewport.bottom() - 38.0);
    let painter = ui.painter_at(viewport);
    painter.circle_filled(center, 30.0, Color32::from_black_alpha(90));
    let mut axes = [
        (Vec3::X, "X", Color32::from_rgb(224, 92, 84)),
        (Vec3::Y, "Y", Color32::from_rgb(120, 196, 110)),
        (Vec3::Z, "Z", Color32::from_rgb(96, 150, 232)),
    ];
    // Far axes first, so near ones draw on top.
    axes.sort_by(|a, b| a.0.dot(forward).total_cmp(&b.0.dot(forward)).reverse());
    for (axis, label, tint) in axes {
        let end = center + egui::vec2(axis.dot(right), -axis.dot(up)) * 20.0;
        painter.line_segment([center, end], Stroke::new(2.0, tint));
        painter.circle_filled(end, 6.5, tint);
        painter.text(
            end,
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::proportional(9.0),
            BACKGROUND,
        );
    }
}

/// Draw the editor around a transparent scene viewport.
pub fn show(
    ui: &mut egui::Ui,
    scene: &mut Scene,
    selected: &mut Selection,
    path: Option<&Path>,
    dirty: bool,
    tools: &mut Tools,
) -> UiResponse {
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = PANEL;
    visuals.window_fill = SURFACE;
    visuals.window_stroke = Stroke::new(1.0, BORDER);
    visuals.extreme_bg_color = BACKGROUND;
    visuals.faint_bg_color = SURFACE;
    visuals.override_text_color = Some(TEXT);
    visuals.window_corner_radius = CornerRadius::same(12);
    visuals.menu_corner_radius = CornerRadius::same(12);
    for (widget, fill) in [
        (&mut visuals.widgets.noninteractive, PANEL),
        (&mut visuals.widgets.inactive, SURFACE),
        (&mut visuals.widgets.hovered, HOVER),
        (&mut visuals.widgets.active, ACCENT_DARK),
        (&mut visuals.widgets.open, SURFACE),
    ] {
        widget.bg_fill = fill;
        widget.weak_bg_fill = fill;
        widget.corner_radius = CornerRadius::same(7);
        widget.bg_stroke = Stroke::new(1.0, BORDER);
    }
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
    visuals.widgets.inactive.bg_stroke = Stroke::NONE;
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, Color32::from_gray(80));
    visuals.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
    visuals.selection.bg_fill = ACCENT_DARK;
    visuals.selection.stroke = Stroke::new(1.0, ACCENT);
    ui.style_mut().visuals = visuals;
    ui.spacing_mut().item_spacing = egui::vec2(6.0, 5.0);
    ui.spacing_mut().button_padding = egui::vec2(8.0, 4.0);
    ui.spacing_mut().interact_size.y = 24.0;
    ui.style_mut()
        .text_styles
        .insert(egui::TextStyle::Body, egui::FontId::proportional(12.5));
    ui.style_mut()
        .text_styles
        .insert(egui::TextStyle::Button, egui::FontId::proportional(12.5));
    let previous_selection = *selected;

    if matches!(*selected, Selection::Entity(i) if i >= scene.entities.len())
        || matches!(*selected, Selection::Light(i) if i >= scene.lights.len())
        || matches!(*selected, Selection::Camera(i) if i >= scene.cameras.len())
    {
        *selected = Selection::None;
    }
    let mut action = None;
    let mut asset_action = None;
    let mut asset_remove = None;
    let mut window = None;
    let mut changed = false;
    let filename = path
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .unwrap_or("Untitled");

    // Window edges resize the undecorated window, unless it is maximized.
    let screen = ui.max_rect();
    if !tools.maximized
        && !tools.flying
        && let Some(direction) = ui
            .ctx()
            .pointer_hover_pos()
            .and_then(|p| resize_edge(screen, p))
    {
        use ResizeDirection as R;
        ui.ctx().set_cursor_icon(match direction {
            R::East | R::West => egui::CursorIcon::ResizeHorizontal,
            R::North | R::South => egui::CursorIcon::ResizeVertical,
            R::NorthEast | R::SouthWest => egui::CursorIcon::ResizeNeSw,
            R::NorthWest | R::SouthEast => egui::CursorIcon::ResizeNwSe,
        });
        if ui.input(|i| i.pointer.primary_pressed()) {
            window = Some(WindowCommand::Resize(direction));
        }
    }

    // Q W E R pick tools, like other engine editors.
    if !tools.flying && !ui.ctx().egui_wants_keyboard_input() {
        for (key, mode) in [
            (egui::Key::Q, Mode::Select),
            (egui::Key::W, Mode::Translate),
            (egui::Key::E, Mode::Rotate),
            (egui::Key::R, Mode::Scale),
        ] {
            if ui.input(|i| i.key_pressed(key) && !i.modifiers.any()) {
                tools.gizmo.set_mode(mode);
            }
        }
    }

    egui::Panel::top("editor_title")
        .exact_size(TITLE_HEIGHT)
        .frame(Frame::new().fill(BAR))
        .show(ui, |ui| {
            let bar = ui.max_rect();
            neon_rule(
                ui.painter(),
                Rect::from_min_max(
                    Pos2::new(bar.left(), bar.bottom() - 1.5),
                    bar.right_bottom(),
                ),
            );
            ui.horizontal_centered(|ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                // Menus sit flat on the bar until hovered.
                ui.visuals_mut().widgets.inactive.weak_bg_fill = Color32::TRANSPARENT;
                ui.add_space(10.0);
                brand_mark(ui);
                ui.add_space(8.0);
                for (label, menu) in [("File", 0), ("Edit", 1), ("Create", 2)] {
                    ui.menu_button(RichText::new(label).size(12.5), |ui| {
                        ui.set_min_width(190.0);
                        match menu {
                            0 => {
                                for (text, shortcut, item) in [
                                    ("New scene", "Ctrl+N", Action::New),
                                    ("Open…", "Ctrl+O", Action::Open),
                                    ("Save", "Ctrl+S", Action::Save),
                                    ("Save as…", "Ctrl+Shift+S", Action::SaveAs),
                                ] {
                                    if ui
                                        .add(egui::Button::new(text).shortcut_text(shortcut))
                                        .clicked()
                                    {
                                        action = Some(item);
                                        ui.close();
                                    }
                                }
                                ui.separator();
                                if ui.button("Exit").clicked() {
                                    window = Some(WindowCommand::Close);
                                    ui.close();
                                }
                            }
                            1 => {
                                let editing = tools.play == PlayState::Editing;
                                for (enabled, label, shortcut, choice) in [
                                    (tools.can_undo, "Undo", "Ctrl+Z", Action::Undo),
                                    (tools.can_redo, "Redo", "Ctrl+Y", Action::Redo),
                                ] {
                                    if ui
                                        .add_enabled(
                                            editing && enabled,
                                            egui::Button::new(label).shortcut_text(shortcut),
                                        )
                                        .clicked()
                                    {
                                        action = Some(choice);
                                        ui.close();
                                    }
                                }
                                ui.separator();
                                ui.add_enabled(
                                    false,
                                    egui::Button::new("Copy entity").shortcut_text("Ctrl+C"),
                                );
                                ui.add_enabled(
                                    false,
                                    egui::Button::new("Paste entity").shortcut_text("Ctrl+V"),
                                );
                                ui.label(
                                    RichText::new("Use the shortcuts in the viewport")
                                        .size(10.5)
                                        .color(MUTED),
                                );
                            }
                            _ => {
                                if tools.play != PlayState::Editing {
                                    ui.disable();
                                }
                                if ui
                                    .add(egui::Button::new("Cube").shortcut_text("Insert"))
                                    .clicked()
                                {
                                    add_cube(scene, selected);
                                    changed = true;
                                    ui.close();
                                }
                                if ui
                                    .add_enabled(
                                        scene.lights.len() < tools.light_limit,
                                        egui::Button::new("Point light").shortcut_text("L"),
                                    )
                                    .clicked()
                                {
                                    add_light(scene, selected);
                                    changed = true;
                                    ui.close();
                                }
                                if ui.button("Camera").clicked() {
                                    add_camera(scene, selected, &tools.camera);
                                    changed = true;
                                    ui.close();
                                }
                            }
                        }
                    });
                }
                ui.menu_button(RichText::new("View").size(12.5), |ui| {
                    ui.checkbox(&mut tools.matrix_enabled, "Enable Matrix")
                        .on_hover_text("Use Matrix virtualized geometry");
                    ui.add_enabled_ui(tools.matrix_enabled, |ui| {
                        ui.checkbox(&mut tools.matrix_debug, "Matrix LOD debug view")
                            .on_hover_text("Color geometry by the LOD selected for rendering");
                    });
                    ui.separator();
                    ui.checkbox(&mut tools.uncapped, "Uncap FPS")
                        .on_hover_text("Draw continuously, without waiting for vsync; may tear");
                });
                // The rest of the bar drags the window; double-click maximizes.
                let buttons = 46.0 * 3.0;
                let free = Rect::from_min_max(
                    Pos2::new(ui.cursor().left() + 8.0, ui.max_rect().top()),
                    Pos2::new(ui.max_rect().right() - buttons, ui.max_rect().bottom()),
                );
                let drag = ui.interact(free, ui.id().with("title_drag"), Sense::click_and_drag());
                if drag.double_clicked() {
                    window = Some(WindowCommand::ToggleMaximize);
                } else if drag.drag_started() {
                    window = Some(WindowCommand::Drag);
                }
                let title = format!("{filename}{}", if dirty { "  •" } else { "" });
                ui.painter().text(
                    free.center(),
                    egui::Align2::CENTER_CENTER,
                    title,
                    egui::FontId::proportional(12.5),
                    if dirty { TEXT } else { MUTED },
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;
                    for glyph in [
                        WindowCommand::Close,
                        WindowCommand::ToggleMaximize,
                        WindowCommand::Minimize,
                    ] {
                        if window_button(ui, glyph, tools.maximized) {
                            window = Some(glyph);
                        }
                    }
                });
            });
        });

    egui::Panel::top("editor_toolbar")
        .exact_size(48.0)
        .frame(
            Frame::new()
                .fill(PANEL)
                .stroke(Stroke::new(1.0, BORDER))
                .corner_radius(RADIUS)
                .inner_margin(Margin::symmetric(10, 5))
                .outer_margin(Margin::symmetric(4, 4)),
        )
        .show(ui, |ui| {
            ui.horizontal_centered(|ui| {
                ui.spacing_mut().item_spacing.x = 3.0;
                let playing = tools.play == PlayState::Playing;
                if ui
                    .button(RichText::new(if playing { "⏸ Pause" } else { "▶ Play" }).color(ACCENT))
                    .on_hover_text("Simulate the scene · Ctrl+P")
                    .clicked()
                {
                    action = Some(if playing { Action::Pause } else { Action::Play });
                }
                if ui
                    .add_enabled(tools.play == PlayState::Paused, egui::Button::new("⏭ Step"))
                    .on_hover_text("Advance one physics tick")
                    .clicked()
                {
                    action = Some(Action::Step);
                }
                if ui
                    .add_enabled(
                        tools.play != PlayState::Editing,
                        egui::Button::new("⏹ Stop"),
                    )
                    .on_hover_text("Stop and restore the scene")
                    .clicked()
                {
                    action = Some(Action::Stop);
                }
                ui.add_space(8.0);
                ui.separator();
                ui.add_space(8.0);
                if tools.play != PlayState::Editing {
                    ui.disable();
                }
                let mode = tools.gizmo.mode();
                for (tool, hint) in [
                    (Mode::Select, "Select · Q"),
                    (Mode::Translate, "Move · W"),
                    (Mode::Rotate, "Rotate · E"),
                    (Mode::Scale, "Scale · R"),
                ] {
                    if tool_button(ui, tool, mode == tool, hint) {
                        tools.gizmo.set_mode(tool);
                    }
                }
                ui.add_space(8.0);
                ui.separator();
                ui.add_space(8.0);
                if ui
                    .button(RichText::new("＋ Cube").color(CYAN))
                    .on_hover_text("Add a cube · Insert")
                    .clicked()
                {
                    add_cube(scene, selected);
                    changed = true;
                }
                if ui
                    .add_enabled(
                        scene.lights.len() < tools.light_limit,
                        egui::Button::new(RichText::new("＋ Light").color(PINK)),
                    )
                    .on_hover_text("Add a point light · L")
                    .clicked()
                {
                    add_light(scene, selected);
                    changed = true;
                }
                if ui
                    .button(RichText::new("＋ Camera"))
                    .on_hover_text("Save the current view as a scene camera")
                    .clicked()
                {
                    add_camera(scene, selected, &tools.camera);
                    changed = true;
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        RichText::new("RMB + WASD fly  ·  Q/E down/up  ·  wheel speed  ·  MMB pan")
                            .size(10.5)
                            .color(MUTED),
                    );
                    ui.add_space(10.0);
                    ui.add(
                        DragValue::new(&mut tools.fly.speed)
                            .range(0.25..=200.0)
                            .speed(0.1)
                            .suffix(" m/s"),
                    )
                    .on_hover_text("Fly speed");
                    ui.label(RichText::new("Speed").size(11.0).color(AMBER));
                });
            });
        });

    egui::Panel::bottom("editor_status")
        .exact_size(24.0)
        .frame(
            Frame::new()
                .fill(BAR)
                .inner_margin(Margin::symmetric(12, 0)),
        )
        .show(ui, |ui| {
            ui.horizontal_centered(|ui| {
                let (dot, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), Sense::hover());
                ui.painter()
                    .circle_filled(dot.center(), 3.0, if dirty { PINK } else { ACCENT });
                ui.label(
                    RichText::new(if tools.play != PlayState::Editing {
                        "Playing · changes are discarded on Stop"
                    } else if dirty {
                        "Unsaved changes"
                    } else if path.is_none() {
                        "Untitled scene"
                    } else {
                        "Saved"
                    })
                    .size(11.0)
                    .color(MUTED),
                );
                ui.label(RichText::new("·").color(MUTED));
                ui.label(
                    RichText::new(format!(
                        "{} objects  ·  {} lights  ·  {} models",
                        scene.entities.len(),
                        scene.lights.len(),
                        scene.assets.len()
                    ))
                    .size(11.0)
                    .color(MUTED),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let eye = tools.camera.eye;
                    ui.label(
                        RichText::new(format!(
                            "Camera  {:.1}, {:.1}, {:.1}   ·   Z up",
                            eye.x, eye.y, eye.z
                        ))
                        .size(11.0)
                        .color(MUTED),
                    );
                });
            });
        });

    egui::Panel::left("editor_hierarchy")
        .resizable(true)
        .default_size(250.0)
        .min_size(180.0)
        .max_size(380.0)
        .frame(panel_frame(PANEL))
        .show(ui, |ui| {
            panel_heading(
                ui,
                "Hierarchy",
                &format!(
                    "{}",
                    scene.entities.len() + scene.lights.len() + scene.cameras.len()
                ),
            );
            egui::ScrollArea::vertical().show(ui, |ui| {
                section(ui, "ENTITIES");
                if scene.entities.is_empty() {
                    ui.label(RichText::new("No entities").color(MUTED).italics());
                }
                for (index, entity) in scene.entities.iter().enumerate() {
                    let name = if entity.name.is_empty() {
                        "Unnamed entity"
                    } else {
                        &entity.name
                    };
                    if hierarchy_row(ui, name, *selected == Selection::Entity(index), false) {
                        *selected = Selection::Entity(index);
                    }
                }
                section(ui, "IMPORTED MODELS");
                if scene.assets.is_empty() {
                    ui.label(RichText::new("No imported models").color(MUTED).italics());
                }
                for (index, asset) in scene.assets.iter().enumerate() {
                    ui.push_id(index, |ui| {
                        ui.horizontal(|ui| {
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .small_button("✕")
                                        .on_hover_text("Remove model from scene")
                                        .clicked()
                                    {
                                        asset_remove = Some(index);
                                    }
                                    let name = Path::new(asset)
                                        .file_name()
                                        .and_then(|name| name.to_str())
                                        .unwrap_or(asset);
                                    ui.add(
                                        egui::Label::new(
                                            RichText::new(name).color(Color32::from_gray(214)),
                                        )
                                        .truncate(),
                                    )
                                    .on_hover_text(asset);
                                },
                            );
                        });
                    });
                }
                section(ui, "CAMERAS");
                if scene.cameras.is_empty() {
                    ui.label(RichText::new("No cameras").color(MUTED).italics());
                }
                for (index, camera) in scene.cameras.iter().enumerate() {
                    let name = if camera.name.is_empty() {
                        "Unnamed camera"
                    } else {
                        &camera.name
                    };
                    if hierarchy_row(ui, name, *selected == Selection::Camera(index), false) {
                        *selected = Selection::Camera(index);
                    }
                }
                section(ui, "SCENE & LIGHTING");
                if hierarchy_row(
                    ui,
                    "Environment & Physics",
                    *selected == Selection::Lighting,
                    true,
                ) {
                    *selected = Selection::Lighting;
                }
                if hierarchy_row(ui, "Sun", *selected == Selection::Sun, true) {
                    *selected = Selection::Sun;
                }
                for index in 0..scene.lights.len() {
                    if hierarchy_row(
                        ui,
                        &format!("Point light {}", index + 1),
                        *selected == Selection::Light(index),
                        true,
                    ) {
                        *selected = Selection::Light(index);
                    }
                }
            });
        });

    egui::Panel::right("editor_inspector")
        .resizable(true)
        .default_size(300.0)
        .min_size(230.0)
        .max_size(440.0)
        .frame(panel_frame(PANEL))
        .show(ui, |ui| {
            panel_heading(ui, "Inspector", "");
            if tools.play != PlayState::Editing {
                ui.disable();
            }
            let mut delete = false;
            egui::ScrollArea::vertical().show(ui, |ui| match *selected {
                Selection::Entity(index) => {
                    let names: Vec<_> = (scene.entities.iter().enumerate())
                        .filter(|&(i, e)| i != index && e.components.contains_key("Rigid Body"))
                        .map(|(_, e)| e.name.clone())
                        .collect();
                    let entity = &mut scene.entities[index];
                    section(ui, "ENTITY");
                    changed |= ui
                        .add(
                            egui::TextEdit::singleline(&mut entity.name)
                                .desired_width(f32::INFINITY),
                        )
                        .changed();
                    if !entity.components.contains_key("Rigid Body")
                        && let Some(info) = tools.registry.get("Rigid Body")
                        && ui.button("＋ Add rigid body").clicked()
                    {
                        entity.components.insert(info.name.to_owned(), (info.default)());
                        changed = true;
                    }
                    section(ui, "TRANSFORM");
                    changed |= vector(ui, "Position", &mut entity.transform.translation);
                    changed |= vector(ui, "Rotation · quaternion", &mut entity.transform.rotation);
                    changed |= vector(ui, "Scale", &mut entity.transform.scale);
                    changed |= components(
                        ui,
                        &tools.registry,
                        &mut entity.components,
                        &names,
                        tools.joint_status.get(&index),
                    );
                    section(ui, "MATERIAL");
                    changed |= color(ui, "Base color", &mut entity.material.base_color);
                    changed |= scalar(
                        ui,
                        "Metallic",
                        &mut entity.material.metallic,
                        0.0..=1.0,
                        0.01,
                    );
                    changed |= scalar(
                        ui,
                        "Roughness",
                        &mut entity.material.roughness,
                        0.0..=1.0,
                        0.01,
                    );
                    ui.add_space(14.0);
                    delete = ui.button("Delete entity").clicked();
                }
                Selection::Light(index) => {
                    let light = &mut scene.lights[index];
                    section(ui, &format!("POINT LIGHT {}", index + 1));
                    changed |= vector(ui, "Position", &mut light.position);
                    section(ui, "LIGHT");
                    changed |= color(ui, "Color", &mut light.color);
                    changed |= scalar(ui, "Intensity", &mut light.intensity, 0.0..=f32::MAX, 0.1);
                    changed |= scalar(ui, "Radius", &mut light.radius, 0.0..=f32::MAX, 0.1);
                    ui.add_space(14.0);
                    delete = ui.button("Delete light").clicked();
                }
                Selection::Camera(index) => {
                    let camera = &mut scene.cameras[index];
                    section(ui, "CAMERA");
                    changed |= ui
                        .add(
                            egui::TextEdit::singleline(&mut camera.name)
                                .desired_width(f32::INFINITY),
                        )
                        .changed();
                    section(ui, "VIEW");
                    changed |= vector(ui, "Position", &mut camera.position);
                    changed |= vector(ui, "Target", &mut camera.target);
                    changed |= scalar(
                        ui,
                        "Vertical FOV °",
                        &mut camera.vertical_fov,
                        1.0..=170.0,
                        0.5,
                    );
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui
                            .button("View through")
                            .on_hover_text("Move the viewport to this camera")
                            .clicked()
                        {
                            look_through(tools, camera);
                        }
                        if ui
                            .button("Set from view")
                            .on_hover_text("Move this camera to the viewport")
                            .clicked()
                        {
                            *camera = camera_from_view(camera.name.clone(), &tools.camera);
                            changed = true;
                        }
                    });
                    ui.add_space(14.0);
                    delete = ui.button("Delete camera").clicked();
                }
                Selection::Sun => {
                    section(ui, "SUN · DIRECTIONAL LIGHT");
                    let mut enabled = scene.sun.is_some();
                    if ui.checkbox(&mut enabled, "Enabled").changed() {
                        scene.sun = enabled.then(verity_core::Sun::default);
                        changed = true;
                    }
                    if let Some(sun) = &mut scene.sun {
                        changed |= vector(ui, "Direction light travels", &mut sun.direction);
                        ui.label(
                            RichText::new("Zero direction points straight down (−Z)")
                                .size(11.0)
                                .color(MUTED),
                        );
                        changed |= color(ui, "Color", &mut sun.color);
                        changed |= scalar(ui, "Intensity", &mut sun.intensity, 0.0..=f32::MAX, 0.1);
                    }
                }
                Selection::Lighting => {
                    section(ui, "SCENE PHYSICS");
                    changed |= vector(ui, "Gravity", &mut scene.physics.gravity);
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Iterations").size(11.5).color(MUTED));
                        let iterations = DragValue::new(&mut scene.physics.iterations);
                        changed |= ui.add(iterations.range(1..=u32::MAX)).changed();
                    });
                    ui.label(
                        RichText::new("Play uses these settings until Stop.")
                            .size(11.0)
                            .color(MUTED),
                    );
                    ui.label(
                        RichText::new("Verity supports cuboids and spheres; capsules and bounce are unavailable.")
                            .size(11.0)
                            .color(MUTED),
                    );
                    section(ui, "SCENE LIGHTING");
                    let previous = scene.lighting.method;
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Method").size(11.5).color(MUTED));
                        egui::ComboBox::from_id_salt("lighting_method")
                            .width(ui.available_width())
                            .selected_text(lighting_name(scene.lighting.method))
                            .show_ui(ui, |ui| {
                                for method in [
                                    LightingMethod::DirectOnly,
                                    LightingMethod::DirectAndHemisphere,
                                ] {
                                    ui.selectable_value(
                                        &mut scene.lighting.method,
                                        method,
                                        lighting_name(method),
                                    );
                                }
                                ui.separator();
                                for info in verity_lux::METHODS {
                                    ui.selectable_value(
                                        &mut scene.lighting.method,
                                        info.lighting,
                                        info.name,
                                    )
                                    .on_hover_text(info.about);
                                }
                            });
                    });
                    changed |= previous != scene.lighting.method;
                    if let Some(status) = &tools.gi_status {
                        ui.label(RichText::new(status).size(11.0).color(MUTED));
                    }
                    section(ui, "HEMISPHERE AMBIENT");
                    changed |= color(ui, "Sky", &mut scene.lighting.sky);
                    changed |= color(ui, "Ground", &mut scene.lighting.ground);
                }
                Selection::None => {
                    ui.add_space(24.0);
                    ui.vertical_centered(|ui| {
                        ui.label(RichText::new("Nothing selected").strong());
                        ui.label(
                            RichText::new(
                                "Select something in the hierarchy\nto edit its properties.",
                            )
                            .size(11.5)
                            .color(MUTED),
                        );
                    });
                }
            });
            if delete {
                tools.gizmo.cancel_drag();
                *selected = match *selected {
                    Selection::Entity(index) => {
                        scene.entities.remove(index);
                        if scene.entities.is_empty() {
                            Selection::None
                        } else {
                            Selection::Entity(index.min(scene.entities.len() - 1))
                        }
                    }
                    Selection::Light(index) => {
                        scene.lights.remove(index);
                        if scene.lights.is_empty() {
                            Selection::None
                        } else {
                            Selection::Light(index.min(scene.lights.len() - 1))
                        }
                    }
                    Selection::Camera(index) => {
                        scene.cameras.remove(index);
                        if scene.cameras.is_empty() {
                            Selection::None
                        } else {
                            Selection::Camera(index.min(scene.cameras.len() - 1))
                        }
                    }
                    other => other,
                };
                changed = true;
            }
        });

    egui::Panel::bottom("editor_assets")
        .resizable(true)
        .default_size(260.0)
        .min_size(150.0)
        .frame(panel_frame(PANEL))
        .show(ui, |ui| {
            panel_heading(ui, "Content Browser", "Double-click to open or import");
            asset_action = tools.browser.show(ui);
        });

    if *selected != previous_selection {
        tools.gizmo.cancel_drag();
    }

    let mut viewport = egui::Rect::NOTHING;
    egui::CentralPanel::default()
        .frame(Frame::new().fill(Color32::TRANSPARENT).inner_margin(4))
        .show(ui, |ui| {
            viewport = ui.available_rect_before_wrap();
            round_corners(&ui.painter_at(viewport), viewport, RADIUS as f32);
            navigate(ui, viewport, tools);
            orientation(ui, viewport, &tools.fly);
            tools.profiler.badge(ui, viewport);
            if let Selection::Entity(index) = *selected
                && tools.play == PlayState::Editing
            {
                changed |= tools.gizmo.show(
                    ui,
                    viewport,
                    &mut scene.entities[index].transform,
                    &tools.camera,
                );
            }
        });
    let scale = ui.ctx().pixels_per_point();
    let details = [
        (
            "Viewport",
            format!(
                "{:.0} × {:.0} px",
                viewport.width() * scale,
                viewport.height() * scale
            ),
        ),
        ("Entities", scene.entities.len().to_string()),
        (
            "Lights",
            format!("{} / {}", scene.lights.len(), tools.light_limit),
        ),
        ("Cameras", scene.cameras.len().to_string()),
        (
            "Lighting",
            tools.gi_status.clone().unwrap_or_else(|| "No GI".into()),
        ),
    ];
    tools.profiler.window(ui.ctx(), &details);
    UiResponse {
        action,
        asset_action,
        asset_remove,
        window,
        changed,
        viewport,
    }
}

fn add_cube(scene: &mut Scene, selected: &mut Selection) {
    scene.entities.push(Entity {
        name: format!("Cube {}", scene.entities.len() + 1),
        transform: Transform {
            translation: [0.0, 0.0, 0.5],
            ..Transform::default()
        },
        material: Material::default(),
        ..Entity::default()
    });
    *selected = Selection::Entity(scene.entities.len() - 1);
}

/// A lighting method's name: the GI methods name themselves.
fn lighting_name(method: LightingMethod) -> &'static str {
    match method {
        LightingMethod::DirectOnly => "Direct only",
        LightingMethod::DirectAndHemisphere => "Direct + hemisphere",
        method => verity_lux::method_info(method).map_or("Unknown", |info| info.name),
    }
}

fn camera_from_view(name: String, view: &verity_render::Camera) -> verity_core::Camera {
    verity_core::Camera {
        name,
        position: view.eye.to_array(),
        target: view.target.to_array(),
        vertical_fov: view.vertical_fov.to_degrees(),
    }
}

fn add_camera(scene: &mut Scene, selected: &mut Selection, view: &verity_render::Camera) {
    let name = format!("Camera {}", scene.cameras.len() + 1);
    scene.cameras.push(camera_from_view(name, view));
    *selected = Selection::Camera(scene.cameras.len() - 1);
}

/// Point the viewport, and the fly controls that drive it, along a scene camera.
pub fn look_through(tools: &mut Tools, camera: &verity_core::Camera) {
    let eye = Vec3::from_array(camera.position);
    let Some(direction) = (Vec3::from_array(camera.target) - eye).try_normalize() else {
        return;
    };
    tools.fly.yaw = direction.y.atan2(direction.x);
    tools.fly.pitch = direction.z.clamp(-1.0, 1.0).asin();
    tools.camera = verity_render::Camera {
        eye,
        target: eye + tools.fly.forward(),
        vertical_fov: camera.vertical_fov.clamp(1.0, 170.0).to_radians(),
    };
}

fn add_light(scene: &mut Scene, selected: &mut Selection) {
    scene.lights.push(PointLight {
        position: [0.0, 0.0, 3.0],
        color: [1.0, 1.0, 1.0],
        intensity: 30.0,
        radius: 15.0,
    });
    *selected = Selection::Light(scene.lights.len() - 1);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_position(shape: &egui::Shape, label: &str) -> Option<Pos2> {
        match shape {
            egui::Shape::Text(text) if text.galley.job.text == label => {
                Some(text.pos + text.galley.size() * 0.5)
            }
            egui::Shape::Vec(shapes) => shapes.iter().find_map(|s| text_position(s, label)),
            _ => None,
        }
    }

    #[test]
    fn viewport_remains_usable_at_minimum_window_size() {
        let context = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(800.0, 520.0),
            )),
            ..Default::default()
        };
        let mut scene = Scene::default();
        let mut selected = Selection::Entity(0);
        let mut tools = Tools::default();
        let mut viewport = egui::Rect::NOTHING;
        let mut output = context.run_ui(input, |ui| {
            viewport = show(ui, &mut scene, &mut selected, None, false, &mut tools).viewport;
        });
        output.textures_delta.clear();
        assert!(
            viewport.width() >= 100.0 && viewport.height() >= 80.0,
            "{viewport:?}"
        );
        assert!(
            viewport.right() <= 800.0 && viewport.bottom() <= 520.0,
            "{viewport:?}"
        );
    }

    #[test]
    fn inspector_shows_components_without_rewriting_them() {
        let mut scene = Scene::default();
        let body = serde_json::json!({ "body_type": "Static", "shape": "Sphere", "radius": 2.0 });
        scene.entities[0].components = [
            ("Rigid Body".to_owned(), body),
            ("Mana".to_owned(), serde_json::json!({ "points": 3 })),
        ]
        .into();
        let before = scene.clone();
        let (mut selected, mut tools) = (Selection::Entity(0), Tools::default());
        let context = egui::Context::default();
        for _ in 0..2 {
            let mut output = context.run_ui(egui::RawInput::default(), |ui| {
                assert!(!show(ui, &mut scene, &mut selected, None, false, &mut tools).changed);
            });
            output.textures_delta.clear();
        }
        assert_eq!(scene, before);
    }

    #[test]
    fn inspector_offers_other_bodies_for_a_joint() {
        let mut scene = Scene::default();
        let body = serde_json::json!({});
        scene.entities[0]
            .components
            .insert("Rigid Body".into(), body.clone());
        let mut other = scene.entities[0].clone();
        other.name = "Other".into();
        scene.entities.push(other);
        let joint = serde_json::to_value(verity_physics_joint()).unwrap();
        scene.entities[0].components.insert("Joint".into(), joint);
        let (mut selected, mut tools) = (Selection::Entity(0), Tools::default());
        tools
            .joint_status
            .insert(0, ("Yielded · 12 N · 3 N·m".into(), 0.6));
        let context = egui::Context::default();
        let output = context.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, egui::vec2(1600.0, 2400.0))),
                ..Default::default()
            },
            |ui| {
                show(ui, &mut scene, &mut selected, None, false, &mut tools);
            },
        );
        for label in ["Connected to", "World", "Yielded · 12 N · 3 N·m"] {
            let shown = output
                .shapes
                .iter()
                .any(|s| text_position(&s.shape, label).is_some());
            assert!(shown, "{label} is visible");
        }
    }

    fn verity_physics_joint() -> serde_json::Value {
        (verity_runtime::registry().get("Joint").unwrap().default)()
    }

    #[test]
    fn inspector_adds_and_removes_a_rigid_body() {
        let context = egui::Context::default();
        let mut scene = Scene::default();
        let (mut selected, mut tools) = (Selection::Entity(0), Tools::default());
        let input = |events| egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, egui::vec2(1600.0, 1200.0))),
            events,
            ..Default::default()
        };
        let mut output = context.run_ui(input(vec![]), |ui| {
            show(ui, &mut scene, &mut selected, None, false, &mut tools);
        });
        let pos = output
            .shapes
            .iter()
            .find_map(|s| text_position(&s.shape, "＋ Add rigid body"))
            .expect("add button is visible");
        output.textures_delta.clear();
        for pressed in [true, false] {
            let events = vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                },
            ];
            let mut output = context.run_ui(input(events), |ui| {
                show(ui, &mut scene, &mut selected, None, false, &mut tools);
            });
            output.textures_delta.clear();
        }
        assert!(scene.entities[0].components.contains_key("Rigid Body"));

        let mut output = context.run_ui(input(vec![]), |ui| {
            show(ui, &mut scene, &mut selected, None, false, &mut tools);
        });
        let header = output
            .shapes
            .iter()
            .find_map(|s| text_position(&s.shape, "RIGID BODY"))
            .expect("rigid body is visible");
        let pos = output
            .shapes
            .iter()
            .filter_map(|s| text_position(&s.shape, "✕"))
            .find(|p| (p.y - header.y).abs() < 12.0)
            .expect("remove button is visible");
        output.textures_delta.clear();
        for pressed in [true, false] {
            let events = vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                },
            ];
            let mut output = context.run_ui(input(events), |ui| {
                show(ui, &mut scene, &mut selected, None, false, &mut tools);
            });
            output.textures_delta.clear();
        }
        assert!(!scene.entities[0].components.contains_key("Rigid Body"));
    }

    #[test]
    fn scene_physics_inspector_does_not_change_settings_when_idle() {
        let context = egui::Context::default();
        let mut scene = Scene::default();
        let before = scene.clone();
        let (mut selected, mut tools) = (Selection::Lighting, Tools::default());
        let mut output = context.run_ui(egui::RawInput::default(), |ui| {
            assert!(!show(ui, &mut scene, &mut selected, None, false, &mut tools).changed);
        });
        output.textures_delta.clear();
        assert_eq!(scene, before);
    }

    #[test]
    fn edges_resize_and_the_inside_does_not() {
        let screen = Rect::from_min_size(Pos2::ZERO, egui::vec2(800.0, 600.0));
        assert_eq!(
            resize_edge(screen, Pos2::new(2.0, 300.0)),
            Some(ResizeDirection::West)
        );
        assert_eq!(
            resize_edge(screen, Pos2::new(798.0, 598.0)),
            Some(ResizeDirection::SouthEast)
        );
        assert_eq!(
            resize_edge(screen, Pos2::new(400.0, 1.0)),
            Some(ResizeDirection::North)
        );
        assert_eq!(resize_edge(screen, Pos2::new(400.0, 300.0)), None);
    }

    #[test]
    fn look_through_matches_scene_camera() {
        let mut tools = Tools::default();
        let camera = verity_core::Camera {
            name: "Nave".into(),
            position: [8.0, -4.0, 3.0],
            target: [-4.0, -4.0, 5.0],
            vertical_fov: 60.0,
        };
        look_through(&mut tools, &camera);
        let saved = camera_from_view(camera.name.clone(), &tools.camera);
        let direction = |c: &verity_core::Camera| {
            (Vec3::from_array(c.target) - Vec3::from_array(c.position)).normalize()
        };
        assert_eq!(saved.position, camera.position);
        assert!(direction(&saved).abs_diff_eq(direction(&camera), 1e-5));
        assert!((saved.vertical_fov - 60.0).abs() < 1e-3);
    }

    #[test]
    fn default_fly_camera_looks_where_the_camera_does() {
        let tools = Tools::default();
        let direction = (tools.camera.target - tools.camera.eye).normalize();
        assert!(tools.fly.forward().distance(direction) < 1e-4);
    }
}
