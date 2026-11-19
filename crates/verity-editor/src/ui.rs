use std::path::Path;

use egui::{
    Color32, CornerRadius, DragValue, Frame, Margin, Pos2, Rect, RichText, Sense, Stroke, Vec2,
};
use glam::Vec3;
use verity_core::{
    ComponentRegistry, Entity, FieldMut, LightingMethod, Material, PointLight, Scene, Transform,
    Widget, WidgetKind,
};
use verity_render::UP;
use winit::window::ResizeDirection;

use crate::dock::Panel;
use crate::gizmo::Mode;

/// Fast enough to cross a planet.
const MAX_FLY_SPEED: f32 = 20_000.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selection {
    Entity(usize),
    Light(usize),
    Camera(usize),
    /// A widget of the game UI.
    Widget(usize),
    Sun,
    Lighting,
    Terrain,
    City,
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
    Copy,
    Paste,
}

/// Whether the scene is being edited or simulated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlayState {
    #[default]
    Editing,
    Playing,
    Paused,
}

/// How far a scene or model load has come.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LoadProgress {
    /// Steps finished, of `total`.
    pub done: usize,
    pub total: usize,
    /// What the current step is doing.
    pub label: String,
}

impl LoadProgress {
    fn fraction(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            (self.done as f32 / self.total as f32).clamp(0.0, 1.0)
        }
    }
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
    pub anti_aliasing: verity_render::AntiAliasing,
    /// Temporal anti-aliasing sharpening, 0 to 1.
    pub sharpening: f32,
    pub ambient_occlusion: bool,
    /// World radius ambient occlusion searches for occluders.
    pub ao_radius: f32,
    /// Draw the scene's game UI over the viewport.
    pub show_game_ui: bool,
    /// Show the viewport's world-axis indicator.
    show_axes: bool,
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
    /// The hierarchy's search text.
    pub filter: String,
    /// While models load: their progress. The editor takes no edits meanwhile.
    pub loading: Option<LoadProgress>,
    /// What the terrain is doing, while there is one.
    pub terrain_status: Option<String>,
    /// Where the panels sit; the editor loads and saves it.
    pub dock: egui_dock::DockState<Panel>,
    pub city: crate::city::CityTool,
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
            anti_aliasing: verity_render::AntiAliasing::default(),
            sharpening: 0.5,
            ambient_occlusion: true,
            ao_radius: 1.0,
            show_game_ui: true,
            show_axes: true,
            gi_status: None,
            joint_status: Default::default(),
            registry: verity_runtime::registry(),
            play: PlayState::Editing,
            can_undo: false,
            can_redo: false,
            filter: String::new(),
            loading: None,
            terrain_status: None,
            dock: crate::dock::default_layout(),
            city: Default::default(),
        }
    }
}

// Graphite chrome with a blue interaction accent and semantic colors.
pub(crate) const BACKGROUND: Color32 = Color32::from_rgb(21, 23, 28);
pub(crate) const PANEL: Color32 = Color32::from_rgb(30, 33, 39);
pub(crate) const BAR: Color32 = Color32::from_rgb(25, 28, 34);
pub(crate) const SURFACE: Color32 = Color32::from_rgb(42, 46, 55);
pub(crate) const HOVER: Color32 = Color32::from_rgb(53, 62, 77);
pub(crate) const BORDER: Color32 = Color32::from_rgb(53, 58, 68);
pub(crate) const TEXT: Color32 = Color32::from_rgb(230, 234, 241);
pub(crate) const MUTED: Color32 = Color32::from_rgb(163, 173, 190);
pub(crate) const ACCENT: Color32 = Color32::from_rgb(132, 180, 255);
pub(crate) const ACCENT_DARK: Color32 = Color32::from_rgb(39, 62, 99);
pub(crate) const PINK: Color32 = Color32::from_rgb(247, 145, 166);
pub(crate) const CYAN: Color32 = Color32::from_rgb(127, 205, 217);
pub(crate) const AMBER: Color32 = Color32::from_rgb(241, 192, 109);
const RADIUS: u8 = 5;
const CLOSE_HOVER: Color32 = Color32::from_rgb(175, 46, 67);
const TITLE_HEIGHT: f32 = 36.0;
/// How close to a window edge the pointer resizes instead of interacting.
const RESIZE_MARGIN: f32 = 6.0;

/// A soft neon halo around a frame.
fn glow(color: Color32, strength: f32) -> egui::Shadow {
    egui::Shadow {
        offset: [0, 0],
        blur: 14,
        spread: 0,
        color: color.gamma_multiply(strength),
    }
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
    ui.label(RichText::new(title).size(11.0).strong().color(MUTED));
    ui.add_space(4.0);
}

fn panel_heading(ui: &mut egui::Ui, title: &str, detail: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(title).size(13.0).strong().color(TEXT));
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
    ui.painter().add(egui::Shape::line(
        vec![
            c + egui::vec2(-5.0, -4.0),
            c + egui::vec2(0.0, 5.0),
            c + egui::vec2(5.0, -4.0),
        ],
        stroke,
    ));
}

#[derive(Clone, Copy, PartialEq)]
enum Icon {
    Entity,
    Light,
    Camera,
}

fn hierarchy_row(ui: &mut egui::Ui, name: &str, selected: bool, kind: Icon) -> egui::Response {
    nested_row(ui, name, selected, kind, 0)
}

/// A hierarchy row indented `depth` levels under its parent.
fn nested_row(
    ui: &mut egui::Ui,
    name: &str,
    selected: bool,
    kind: Icon,
    depth: usize,
) -> egui::Response {
    let indent = 14.0 * depth.min(12) as f32;
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), 26.0), Sense::click());
    // Rows scrolled out of the hierarchy keep their space but skip text layout and painting.
    if !ui.is_rect_visible(rect) {
        return response;
    }
    let painter = ui.painter();
    if selected {
        painter.rect_filled(rect, 6, ACCENT_DARK);
        painter.rect_stroke(
            rect,
            6,
            Stroke::new(1.0, ACCENT.gamma_multiply(0.35)),
            egui::StrokeKind::Inside,
        );
        painter.rect_filled(
            Rect::from_min_size(rect.min, egui::vec2(2.0, rect.height())),
            1,
            ACCENT,
        );
    } else if response.hovered() {
        painter.rect_filled(rect, 6, HOVER);
    }
    let icon = rect.left_center() + egui::vec2(14.0 + indent, 0.0);
    match kind {
        Icon::Light => {
            painter.circle_stroke(icon, 4.0, Stroke::new(1.2, PINK));
            painter.circle_filled(icon, 1.3, PINK);
        }
        Icon::Entity => {
            let cube = Rect::from_center_size(icon, egui::vec2(8.0, 8.0));
            painter.rect_stroke(cube, 1, Stroke::new(1.2, CYAN), egui::StrokeKind::Inside);
        }
        Icon::Camera => {
            let body = Rect::from_center_size(icon - egui::vec2(1.5, 0.0), egui::vec2(7.0, 6.0));
            painter.rect_stroke(body, 1, Stroke::new(1.2, AMBER), egui::StrokeKind::Inside);
            let lens = vec![
                icon + egui::vec2(2.0, 0.0),
                icon + egui::vec2(5.0, -3.0),
                icon + egui::vec2(5.0, 3.0),
            ];
            painter.add(egui::Shape::closed_line(lens, Stroke::new(1.2, AMBER)));
        }
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
        rect.left_top() + egui::vec2(28.0 + indent, 0.0),
        rect.right_bottom() - egui::vec2(6.0, 0.0),
    );
    painter.with_clip_rect(text_rect).galley(
        Pos2::new(text_rect.left(), rect.center().y - galley.size().y / 2.0),
        galley,
        TEXT,
    );
    response
}

fn vector<const N: usize>(ui: &mut egui::Ui, label: &str, values: &mut [f32; N]) -> bool {
    vector_speed(ui, label, values, 0.02)
}

fn vector_speed<const N: usize>(
    ui: &mut egui::Ui,
    label: &str,
    values: &mut [f32; N],
    speed: f32,
) -> bool {
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
                    .speed(speed)
                    .max_decimals(3)
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
            FieldMut::File(value, extensions) => {
                let mut edited = false;
                ui.horizontal(|ui| {
                    ui.label(RichText::new(*label).size(11.5).color(MUTED));
                    if ui.button("…").on_hover_text("Choose a file").clicked()
                        && let Some(path) = rfd::FileDialog::new()
                            .add_filter(extensions.join(", "), extensions)
                            .pick_file()
                    {
                        **value = path.to_string_lossy().into_owned();
                        edited = true;
                    }
                    edited |= ui
                        .add(egui::TextEdit::singleline(*value).desired_width(f32::INFINITY))
                        .changed();
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
const TRANSFORM_TOOLS: [(egui::Key, Mode, &str, &str); 4] = [
    (
        egui::Key::Q,
        Mode::Select,
        "Select",
        "Select without a transform gizmo",
    ),
    (
        egui::Key::W,
        Mode::Translate,
        "Move",
        "Translate along world axes",
    ),
    (
        egui::Key::E,
        Mode::Rotate,
        "Rotate",
        "Rotate around world axes",
    ),
    (egui::Key::R, Mode::Scale, "Scale", "Scale along local axes"),
];

fn add_menu(ui: &mut egui::Ui, scene: &mut Scene, selected: &mut Selection, tools: &Tools) -> bool {
    ui.set_min_width(180.0);
    if ui
        .add(egui::Button::new("Cube").shortcut_text("Insert"))
        .clicked()
    {
        add_cube(scene, selected);
    } else if ui
        .add_enabled(
            scene.lights.len() < tools.light_limit,
            egui::Button::new("Point Light").shortcut_text("L"),
        )
        .clicked()
    {
        add_light(scene, selected);
    } else if ui.button("Camera from View").clicked() {
        add_camera(scene, selected, &tools.camera);
    } else {
        ui.separator();
        return ui
            .menu_button("UI Widget", |ui| {
                for (label, kind) in WIDGET_KINDS {
                    if ui.button(label).clicked() {
                        add_widget(scene, selected, kind);
                        ui.close();
                        return true;
                    }
                }
                false
            })
            .inner
            .unwrap_or(false);
    }
    ui.close();
    true
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
            fly.speed = (fly.speed * 1.15f32.powf(scroll.signum())).clamp(0.25, MAX_FLY_SPEED);
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
    visuals.window_corner_radius = CornerRadius::same(6);
    visuals.menu_corner_radius = CornerRadius::same(6);
    for (widget, fill) in [
        (&mut visuals.widgets.noninteractive, PANEL),
        (&mut visuals.widgets.inactive, SURFACE),
        (&mut visuals.widgets.hovered, HOVER),
        (&mut visuals.widgets.active, ACCENT_DARK),
        (&mut visuals.widgets.open, SURFACE),
    ] {
        widget.bg_fill = fill;
        widget.weak_bg_fill = fill;
        widget.corner_radius = CornerRadius::same(4);
        widget.bg_stroke = Stroke::new(1.0, BORDER);
    }
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
    visuals.widgets.inactive.bg_stroke = Stroke::NONE;
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, CYAN.gamma_multiply(0.6));
    visuals.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
    visuals.widgets.open.bg_stroke = Stroke::new(1.0, CYAN.gamma_multiply(0.6));
    visuals.selection.bg_fill = ACCENT_DARK;
    visuals.selection.stroke = Stroke::new(1.0, ACCENT);
    visuals.text_cursor.stroke = Stroke::new(2.0, ACCENT);
    visuals.hyperlink_color = CYAN;
    visuals.warn_fg_color = AMBER;
    visuals.error_fg_color = PINK;
    visuals.window_shadow = egui::Shadow {
        offset: [0, 4],
        blur: 16,
        spread: 0,
        color: Color32::from_black_alpha(100),
    };
    visuals.popup_shadow = visuals.window_shadow;
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
        || matches!(*selected, Selection::Widget(i) if i >= scene.ui.len())
    {
        *selected = Selection::None;
    }
    let mut action = None;
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

    // Q W E R pick tools and F frames the selection, like other engine editors.
    if !tools.flying && !ui.ctx().egui_wants_keyboard_input() {
        if ui.input(|i| i.key_pressed(egui::Key::F) && !i.modifiers.any()) {
            focus(scene, *selected, tools);
        }
        for (key, mode, _, _) in TRANSFORM_TOOLS {
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
            // A quiet divider becomes the progress track while loading.
            let rule = match &tools.loading {
                Some(progress) => {
                    let track = Rect::from_min_max(
                        Pos2::new(bar.left(), bar.bottom() - 3.0),
                        bar.right_bottom(),
                    );
                    ui.painter().rect_filled(track, 0, BORDER);
                    let mut fill = track;
                    fill.set_width(track.width() * progress.fraction().max(0.02));
                    fill
                }
                None => Rect::from_min_max(
                    Pos2::new(bar.left(), bar.bottom() - 1.5),
                    bar.right_bottom(),
                ),
            };
            ui.painter().rect_filled(
                rule,
                0,
                if tools.loading.is_some() {
                    ACCENT
                } else {
                    BORDER
                },
            );
            ui.horizontal_centered(|ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                // Menus sit flat on the bar until hovered.
                ui.visuals_mut().widgets.inactive.weak_bg_fill = Color32::TRANSPARENT;
                ui.add_space(10.0);
                brand_mark(ui);
                ui.add_space(8.0);
                ui.add_enabled_ui(tools.loading.is_none(), |ui| {
                    for label in ["File", "Edit"] {
                        ui.menu_button(RichText::new(label).size(12.5), |ui| {
                            ui.set_min_width(190.0);
                            match label {
                                "File" => {
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
                                "Edit" => {
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
                                    let entity = matches!(*selected, Selection::Entity(_));
                                    for (enabled, label, shortcut, choice) in [
                                        (entity, "Copy entity", "Ctrl+C", Action::Copy),
                                        (true, "Paste entity", "Ctrl+V", Action::Paste),
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
                                    if ui
                                        .add_enabled(
                                            editing && entity,
                                            egui::Button::new("Duplicate").shortcut_text("Ctrl+D"),
                                        )
                                        .clicked()
                                    {
                                        changed |= duplicate(scene, selected);
                                        ui.close();
                                    }
                                    if ui
                                        .add_enabled(
                                            *selected != Selection::None,
                                            egui::Button::new("Frame selected").shortcut_text("F"),
                                        )
                                        .clicked()
                                    {
                                        focus(scene, *selected, tools);
                                        ui.close();
                                    }
                                }
                                _ => unreachable!(),
                            }
                        });
                    }
                    ui.menu_button("View", |ui| {
                        ui.set_min_width(200.0);
                        ui.menu_button("Overlays", |ui| {
                            ui.checkbox(&mut tools.show_axes, "World axes");
                            ui.checkbox(&mut tools.show_game_ui, "Game UI preview");
                        });
                        ui.menu_button("Anti-aliasing", |ui| {
                            for mode in verity_render::AntiAliasing::ALL {
                                ui.radio_value(&mut tools.anti_aliasing, mode, mode.label());
                            }
                            ui.add_enabled(
                                tools.anti_aliasing.sharpened(),
                                egui::Slider::new(&mut tools.sharpening, 0.0..=1.0)
                                    .text("Sharpening"),
                            );
                        });
                        ui.menu_button("Ambient Occlusion", |ui| {
                            ui.checkbox(&mut tools.ambient_occlusion, "Enable");
                            ui.add_enabled(
                                tools.ambient_occlusion,
                                egui::Slider::new(&mut tools.ao_radius, 0.1..=5.0)
                                    .logarithmic(true)
                                    .text("Radius"),
                            )
                            .on_hover_text("How far, in world units, to look for occluders");
                        });
                        ui.menu_button("Matrix Geometry", |ui| {
                            ui.checkbox(&mut tools.matrix_enabled, "Enable Matrix");
                            ui.add_enabled_ui(tools.matrix_enabled, |ui| {
                                ui.checkbox(&mut tools.matrix_debug, "Matrix LOD debug view")
                                    .on_hover_text(
                                        "Color geometry by the LOD selected for rendering",
                                    );
                            });
                        });
                        ui.separator();
                        ui.checkbox(&mut tools.uncapped, "Uncap FPS")
                            .on_hover_text("Draw without waiting for vsync; may tear");
                    });
                    ui.menu_button("Window", |ui| {
                        ui.set_min_width(170.0);
                        for panel in Panel::ALL {
                            if ui.button(panel.title()).clicked() {
                                crate::dock::open_panel(&mut tools.dock, panel);
                                ui.close();
                            }
                        }
                        ui.separator();
                        if ui.button("Reset Layout").clicked() {
                            tools.dock = crate::dock::default_layout();
                            ui.close();
                        }
                    });
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
                if free.is_positive() {
                    ui.painter().with_clip_rect(free).text(
                        free.center(),
                        egui::Align2::CENTER_CENTER,
                        title,
                        egui::FontId::proportional(12.5),
                        if dirty { TEXT } else { MUTED },
                    );
                }
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

    // Everything below the title bar waits for a load to finish.
    if tools.loading.is_some() {
        ui.disable();
    }
    egui::Panel::top("editor_toolbar")
        .exact_size(42.0)
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
                ui.spacing_mut().item_spacing.x = 4.0;
                let editing = tools.play == PlayState::Editing;
                ui.add_enabled_ui(editing, |ui| {
                    ui.menu_button(RichText::new("+ Add").strong().color(ACCENT), |ui| {
                        changed |= add_menu(ui, scene, selected, tools);
                    });
                    ui.separator();
                    let mode = tools.gizmo.mode();
                    for (key, tool, label, description) in TRANSFORM_TOOLS {
                        if ui
                            .add(egui::Button::selectable(mode == tool, label))
                            .on_hover_text(format!("{description} · {}", key.name()))
                            .clicked()
                        {
                            tools.gizmo.set_mode(tool);
                        }
                    }
                });
                ui.separator();
                if ui
                    .add_enabled(*selected != Selection::None, egui::Button::new("Frame"))
                    .on_hover_text("Frame selected · F")
                    .clicked()
                {
                    focus(scene, *selected, tools);
                }
                ui.menu_button("Camera", |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Fly speed");
                        ui.add(
                            DragValue::new(&mut tools.fly.speed)
                                .range(0.25..=MAX_FLY_SPEED)
                                .speed(0.1)
                                .suffix(" m/s"),
                        );
                    });
                    ui.separator();
                    for hint in [
                        "Right mouse + WASD to fly",
                        "Q / E to descend / ascend",
                        "Scroll to adjust speed",
                        "Middle mouse to pan",
                    ] {
                        ui.label(RichText::new(hint).small().color(MUTED));
                    }
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add_enabled(!editing, egui::Button::new("Stop"))
                        .on_hover_text("Stop and restore the scene")
                        .clicked()
                    {
                        action = Some(Action::Stop);
                    }
                    if ui
                        .add_enabled(tools.play == PlayState::Paused, egui::Button::new("Step"))
                        .on_hover_text("Advance one physics tick")
                        .clicked()
                    {
                        action = Some(Action::Step);
                    }
                    let (label, command) = match tools.play {
                        PlayState::Editing => ("▶ Play", Action::Play),
                        PlayState::Playing => ("Pause", Action::Pause),
                        PlayState::Paused => ("▶ Resume", Action::Play),
                    };
                    if ui
                        .add(egui::Button::new(RichText::new(label).strong()).fill(ACCENT_DARK))
                        .on_hover_text("Play, pause or resume simulation · Ctrl+P")
                        .clicked()
                    {
                        action = Some(command);
                    }
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
                ui.style_mut().override_font_id = Some(egui::FontId::monospace(10.5));
                let playing = tools.play != PlayState::Editing;
                let state = if playing {
                    AMBER
                } else if dirty {
                    AMBER
                } else {
                    ACCENT
                };
                let (dot, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), Sense::hover());
                ui.painter()
                    .circle_filled(dot.center(), 5.0, state.gamma_multiply(0.25));
                ui.painter().circle_filled(dot.center(), 3.0, state);
                ui.label(
                    RichText::new(if playing {
                        if tools.play == PlayState::Paused {
                            "Paused · Stop to restore scene"
                        } else {
                            "Simulating · Stop to restore scene"
                        }
                    } else if dirty {
                        "Unsaved changes"
                    } else if path.is_none() {
                        "Untitled scene"
                    } else {
                        "Saved"
                    })
                    .color(if playing { AMBER } else { MUTED }),
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

    let mut dock = std::mem::replace(&mut tools.dock, egui_dock::DockState::new(Vec::new()));
    let profiler_tab = dock.find_tab(&Panel::Profiler).is_some();
    tools.profiler.open = profiler_tab;
    let mut panels = Panels {
        scene,
        selected,
        tools,
        changed,
        delete: false,
        asset_action: None,
        asset_remove: None,
        viewport: egui::Rect::NOTHING,
    };
    egui::CentralPanel::default()
        .frame(Frame::new().fill(Color32::TRANSPARENT).inner_margin(4))
        .show(ui, |ui| {
            egui_dock::DockArea::new(&mut dock)
                .style(dock_style(ui.style()))
                .show_leaf_collapse_buttons(false)
                .show_leaf_close_all_buttons(false)
                .show_inside(ui, &mut panels);
        });
    let Panels {
        scene,
        selected,
        tools,
        mut changed,
        delete,
        asset_action,
        asset_remove,
        viewport,
    } = panels;
    // The badge toggles the Profiler tab.
    if tools.profiler.open != profiler_tab {
        match dock.find_tab(&Panel::Profiler) {
            Some(path) => {
                dock.remove_tab(path);
            }
            None => crate::dock::open_panel(&mut dock, Panel::Profiler),
        }
    }
    tools.profiler.open = dock.find_tab(&Panel::Profiler).is_some();
    tools.dock = dock;

    if delete {
        tools.gizmo.cancel_drag();
        *selected = match *selected {
            Selection::Entity(index) => {
                scene.remove_entity(index);
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
            Selection::Widget(index) => {
                scene.ui.remove(index);
                if scene.ui.is_empty() {
                    Selection::None
                } else {
                    Selection::Widget(index.min(scene.ui.len() - 1))
                }
            }
            other => other,
        };
        changed = true;
    }

    if *selected != previous_selection {
        tools.gizmo.cancel_drag();
    }

    let scale = ui.ctx().pixels_per_point();
    if tools.profiler.open {
        let profiler = &mut tools.profiler;
        profiler.counter(
            "Viewport",
            format!(
                "{:.0} × {:.0} px",
                viewport.width() * scale,
                viewport.height() * scale
            ),
        );
        profiler.counter("Entities", scene.entities.len().to_string());
        profiler.counter(
            "Lights",
            format!("{} / {}", scene.lights.len(), tools.light_limit),
        );
        profiler.counter("Cameras", scene.cameras.len().to_string());
        profiler.counter(
            "Lighting",
            tools.gi_status.clone().unwrap_or_else(|| "No GI".into()),
        );
    }
    if let Some(progress) = &tools.loading {
        loading_popup(ui.ctx(), progress);
    }
    UiResponse {
        action,
        asset_action,
        asset_remove,
        window,
        changed,
        viewport,
    }
}

impl Panels<'_> {
    fn hierarchy(&mut self, ui: &mut egui::Ui) {
        let (scene, selected, tools) = (&mut *self.scene, &mut *self.selected, &mut *self.tools);
        let (mut changed, mut delete, mut asset_remove) = (false, false, None);
        panel_heading(
            ui,
            "Hierarchy",
            &format!(
                "{}",
                scene.entities.len() + scene.lights.len() + scene.cameras.len()
            ),
        );
        ui.add(
            egui::TextEdit::singleline(&mut tools.filter)
                .hint_text("🔍 Search…")
                .desired_width(f32::INFINITY),
        );
        let filter = tools.filter.to_lowercase();
        let shown = |name: &str| filter.is_empty() || name.to_lowercase().contains(&filter);
        let editing = tools.play == PlayState::Editing;
        let mut pending = None;
        let mut menu = |row: egui::Response, item: Selection| {
            if row.clicked() || row.secondary_clicked() {
                pending = Some((item, None));
            }
            if let Some(chosen) = row_menu(&row, item, editing) {
                pending = Some((item, Some(chosen)));
            }
        };
        egui::ScrollArea::vertical().show(ui, |ui| {
            section(ui, "ENTITIES");
            if scene.entities.is_empty() {
                ui.label(RichText::new("No entities").color(MUTED).italics());
            }
            // Unfiltered rows are uniform, so rows out of view become one spacer
            // instead of thousands of allocations and hit tests.
            let total = scene.entities.len();
            let step = 26.0 + ui.spacing().item_spacing.y;
            let (first, last) = if filter.is_empty() {
                let (top, clip) = (ui.cursor().top(), ui.clip_rect());
                let row = |y: f32| (((y - top) / step).max(0.0) as usize).min(total);
                (row(clip.top()), (row(clip.bottom()) + 1).min(total))
            } else {
                (0, total)
            };
            let last = last.max(first);
            ui.add_space(first as f32 * step);
            // Keep row ids stable while scrolling, so an open context menu stays open.
            ui.skip_ahead_auto_ids(first);
            let tree = scene.tree();
            for &(index, depth) in &tree[first..last] {
                let entity = &scene.entities[index];
                let name = if entity.name.is_empty() {
                    "Unnamed entity"
                } else {
                    &entity.name
                };
                if !shown(name) {
                    continue;
                }
                let item = Selection::Entity(index);
                // Filtered rows lose their parents, so indenting them would mislead.
                let depth = if filter.is_empty() { depth } else { 0 };
                menu(
                    nested_row(ui, name, *selected == item, Icon::Entity, depth),
                    item,
                );
            }
            ui.add_space((total - last) as f32 * step);
            ui.skip_ahead_auto_ids(total - last);
            section(ui, "IMPORTED MODELS");
            if scene.assets.is_empty() {
                ui.label(RichText::new("No imported models").color(MUTED).italics());
            }
            for (index, asset) in scene.assets.iter().enumerate() {
                ui.push_id(index, |ui| {
                    let name = Path::new(asset)
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or(asset);
                    let row = hierarchy_row(ui, name, false, Icon::Entity).on_hover_text(asset);
                    row.context_menu(|ui| {
                        let button = ui.add_enabled(editing, egui::Button::new("    Remove model"));
                        let c = button.rect.left_center() + egui::vec2(11.0, 0.0);
                        let stroke = Stroke::new(1.2, if editing { TEXT } else { MUTED });
                        ui.painter().rect_stroke(
                            Rect::from_min_max(
                                c + egui::vec2(-3.5, -2.5),
                                c + egui::vec2(3.5, 4.5),
                            ),
                            1,
                            stroke,
                            egui::StrokeKind::Inside,
                        );
                        ui.painter().line_segment(
                            [c + egui::vec2(-4.5, -4.0), c + egui::vec2(4.5, -4.0)],
                            stroke,
                        );
                        ui.painter().line_segment(
                            [c + egui::vec2(-1.5, -5.5), c + egui::vec2(1.5, -5.5)],
                            stroke,
                        );
                        if button.clicked() {
                            asset_remove = Some(index);
                            ui.close();
                        }
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
                if shown(name) {
                    let item = Selection::Camera(index);
                    menu(
                        hierarchy_row(ui, name, *selected == item, Icon::Camera),
                        item,
                    );
                }
            }
            section(ui, "GAME UI");
            if scene.ui.is_empty() {
                ui.label(RichText::new("No widgets · use Add").color(MUTED).italics());
            }
            for (index, widget) in scene.ui.iter().enumerate() {
                if shown(&widget.name) {
                    let item = Selection::Widget(index);
                    menu(
                        hierarchy_row(ui, &widget.name, *selected == item, Icon::Entity),
                        item,
                    );
                }
            }
            section(ui, "SCENE & LIGHTING");
            for (name, item) in [
                ("Environment & Physics", Selection::Lighting),
                ("Sun", Selection::Sun),
                ("Terrain", Selection::Terrain),
                ("City", Selection::City),
            ] {
                if shown(name) && hierarchy_row(ui, name, *selected == item, Icon::Light).clicked()
                {
                    *selected = item;
                }
            }
            for index in 0..scene.lights.len() {
                let name = format!("Point light {}", index + 1);
                if shown(&name) {
                    let item = Selection::Light(index);
                    menu(
                        hierarchy_row(ui, &name, *selected == item, Icon::Light),
                        item,
                    );
                }
            }
        });
        if let Some((item, chosen)) = pending {
            *selected = item;
            match chosen {
                Some(RowAction::Frame) => focus(scene, item, tools),
                Some(RowAction::Duplicate) => changed |= duplicate(scene, selected),
                Some(RowAction::Delete) => delete = true,
                None => {}
            }
        }
        self.changed |= changed;
        self.delete |= delete;
        self.asset_remove = self.asset_remove.or(asset_remove);
    }

    fn inspector(&mut self, ui: &mut egui::Ui) {
        let (scene, selected, tools) = (&mut *self.scene, &mut *self.selected, &mut *self.tools);
        let (mut changed, mut delete) = (false, false);
        panel_heading(ui, "Inspector", "");
        if tools.play != PlayState::Editing {
            ui.disable();
        }

        egui::ScrollArea::vertical().show(ui, |ui| match *selected {
            Selection::Entity(index) => {
                let names: Vec<_> = (scene.entities.iter().enumerate())
                    .filter(|&(i, e)| i != index && e.components.contains_key("Rigid Body"))
                    .map(|(_, e)| e.name.clone())
                    .collect();
                let below = scene.descendants(index);
                let parents: Vec<_> = (scene.entities.iter().enumerate())
                    .filter(|&(i, _)| i != index && !below.contains(&i))
                    .map(|(i, e)| (i, e.name.clone()))
                    .collect();
                let entity = &mut scene.entities[index];
                let before = entity.transform.clone();
                let mut parent = entity.parent;
                section(ui, "ENTITY");
                changed |= ui
                    .add(egui::TextEdit::singleline(&mut entity.name).desired_width(f32::INFINITY))
                    .changed();
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Parent").size(11.5).color(MUTED));
                    let name = |p: Option<usize>| {
                        let found = parents.iter().find(|(i, _)| Some(*i) == p);
                        found.map_or("None", |(_, name)| name.as_str()).to_owned()
                    };
                    egui::ComboBox::from_id_salt("entity_parent")
                        .width(ui.available_width())
                        .selected_text(name(parent))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut parent, None, "None");
                            for (i, name) in &parents {
                                ui.selectable_value(&mut parent, Some(*i), name);
                            }
                        });
                });
                if !entity.components.contains_key("Rigid Body")
                    && let Some(info) = tools.registry.get("Rigid Body")
                    && ui.button("＋ Add rigid body").clicked()
                {
                    entity
                        .components
                        .insert(info.name.to_owned(), (info.default)());
                    changed = true;
                }
                section(ui, "TRANSFORM");
                changed |= vector(ui, "Position", &mut entity.transform.translation);
                changed |= euler(ui, &mut entity.transform.rotation);
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
                delete |= ui
                    .button(RichText::new("Delete entity").color(PINK))
                    .clicked();
                if scene.entities[index].transform != before {
                    scene.carry_children(index, &before);
                }
                if parent != scene.entities[index].parent {
                    // The picker only offers parents that cannot form a cycle.
                    changed |= scene.set_parent(index, parent).is_ok();
                }
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
                delete |= ui
                    .button(RichText::new("Delete light").color(PINK))
                    .clicked();
            }
            Selection::Camera(index) => {
                let camera = &mut scene.cameras[index];
                section(ui, "CAMERA");
                changed |= ui
                    .add(egui::TextEdit::singleline(&mut camera.name).desired_width(f32::INFINITY))
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
                delete |= ui
                    .button(RichText::new("Delete camera").color(PINK))
                    .clicked();
            }
            Selection::Widget(index) => {
                changed |= widget_inspector(ui, &mut scene.ui[index]);
                ui.add_space(14.0);
                delete |= ui
                    .button(RichText::new("Delete widget").color(PINK))
                    .clicked();
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
                    RichText::new(
                        "Verity supports cuboids and spheres; capsules and bounce are unavailable.",
                    )
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
                            for (heading, methods) in [
                                (None, verity_lux::METHODS),
                                (Some("Experimental GI"), verity_lux::EXPERIMENTAL_METHODS),
                            ] {
                                ui.separator();
                                if let Some(heading) = heading {
                                    ui.label(RichText::new(heading).small().color(MUTED));
                                }
                                for info in methods {
                                    ui.selectable_value(
                                        &mut scene.lighting.method,
                                        info.lighting,
                                        info.name,
                                    )
                                    .on_hover_text(info.about);
                                }
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
                section(ui, "SKY · VOLUMETRIC CLOUDS");
                let mut enabled = scene.lighting.clouds.is_some();
                if ui.checkbox(&mut enabled, "Enabled").changed() {
                    scene.lighting.clouds = enabled.then(verity_core::Clouds::default);
                    changed = true;
                }
                if let Some(clouds) = &mut scene.lighting.clouds {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new("Weather").size(11.5).color(MUTED));
                        // Coverage, genus, density, base, thickness, cirrus.
                        for (name, [coverage, kind, density, bottom, thickness, cirrus]) in [
                            ("Fair", [0.18, 0.95, 0.008, 1200.0, 1200.0, 0.25]),
                            ("Partly cloudy", [0.35, 0.85, 0.012, 1200.0, 2000.0, 0.4]),
                            ("Cloudy", [0.7, 0.5, 0.015, 1000.0, 1500.0, 0.2]),
                            ("Overcast", [0.95, 0.1, 0.02, 800.0, 1200.0, 0.0]),
                            ("Storm", [0.75, 1.0, 0.05, 700.0, 7000.0, 0.6]),
                        ] {
                            if ui.small_button(name).clicked() {
                                *clouds = verity_core::Clouds {
                                    coverage,
                                    kind,
                                    density,
                                    bottom,
                                    thickness,
                                    cirrus,
                                    ..*clouds
                                };
                                changed = true;
                            }
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Formation").size(11.5).color(MUTED));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("Randomize").clicked() {
                                let nanos = std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .map_or(0, |d| d.subsec_nanos());
                                clouds.seed = nanos ^ clouds.seed.rotate_left(13);
                                changed = true;
                            }
                            changed |= ui.add(DragValue::new(&mut clouds.seed)).changed();
                        });
                    });
                    changed |= scalar(ui, "Coverage", &mut clouds.coverage, 0.0..=1.0, 0.01);
                    changed |= scalar(ui, "Type", &mut clouds.kind, 0.0..=1.0, 0.01);
                    ui.label(
                        RichText::new("0 stratus · 0.5 stratocumulus · 1 cumulus")
                            .size(11.0)
                            .color(MUTED),
                    );
                    changed |= scalar(ui, "Density", &mut clouds.density, 0.0..=1.0, 0.001);
                    changed |= scalar(ui, "Base altitude", &mut clouds.bottom, 0.0..=20000.0, 10.0);
                    changed |= scalar(ui, "Thickness", &mut clouds.thickness, 1.0..=10000.0, 10.0);
                    changed |= scalar(ui, "Wind X", &mut clouds.wind[0], -100.0..=100.0, 0.1);
                    changed |= scalar(ui, "Wind Y", &mut clouds.wind[1], -100.0..=100.0, 0.1);
                    changed |= scalar(ui, "Cirrus", &mut clouds.cirrus, 0.0..=1.0, 0.01);
                    ui.label(RichText::new("Voxel cloudscape").size(11.5).color(MUTED));
                    egui::ComboBox::from_id_salt("cloudscape")
                        .width(ui.available_width())
                        .selected_text(format!("{:?}", clouds.cloudscape))
                        .show_ui(ui, |ui| {
                            for kind in verity_core::Cloudscape::ALL {
                                changed |= ui
                                    .selectable_value(&mut clouds.cloudscape, kind, format!("{kind:?}"))
                                    .changed();
                            }
                        });
                }
            }
            Selection::Terrain => changed |= terrain_inspector(ui, scene, tools),
            Selection::City => changed |= city_inspector(ui, scene, tools),
            Selection::None => {
                ui.add_space(24.0);
                ui.vertical_centered(|ui| {
                    ui.label(RichText::new("Nothing selected").strong());
                    ui.label(
                        RichText::new("Select something in the hierarchy\nto edit its properties.")
                            .size(11.5)
                            .color(MUTED),
                    );
                });
            }
        });
        self.changed |= changed;
        self.delete |= delete;
    }

    fn assets(&mut self, ui: &mut egui::Ui) {
        let tools = &mut *self.tools;
        panel_heading(ui, "Content Browser", "Double-click to open or import");
        let asset_action = tools.browser.show(ui);
        self.asset_action = self.asset_action.take().or(asset_action);
    }

    fn viewport(&mut self, ui: &mut egui::Ui) {
        let (scene, selected, tools) = (&mut *self.scene, &mut *self.selected, &mut *self.tools);
        let mut changed = false;
        let viewport = ui.available_rect_before_wrap();
        round_corners(&ui.painter_at(viewport), viewport, RADIUS as f32);
        // Match the status bar's simulation color around the live scene.
        if tools.play != PlayState::Editing {
            let painter = ui.painter_at(viewport);
            painter.rect_stroke(
                viewport,
                RADIUS,
                Stroke::new(2.0, AMBER),
                egui::StrokeKind::Inside,
            );
            let label = if tools.play == PlayState::Paused {
                "❚❚ PAUSED"
            } else {
                "▶ PLAYING"
            };
            painter.text(
                viewport.center_top() + egui::vec2(0.0, 16.0),
                egui::Align2::CENTER_CENTER,
                label,
                egui::FontId::monospace(11.0),
                AMBER,
            );
        }
        navigate(ui, viewport, tools);
        if tools.show_axes {
            orientation(ui, viewport, &tools.fly);
        }
        tools.profiler.badge(ui, viewport);
        if tools.show_game_ui && tools.loading.is_none() {
            // Buttons press only while playing; editing selects and drags widgets instead.
            let playing = tools.play == PlayState::Playing;
            for index in verity_ui::show(ui, viewport, &scene.ui, playing) {
                crate::console::info(format!("UI: {} clicked", scene.ui[index].name));
            }
            if tools.play == PlayState::Editing {
                changed |= edit_widgets(ui, viewport, scene, selected);
            }
        }
        if *selected == Selection::City
            && tools.play == PlayState::Editing
            && tools.loading.is_none()
        {
            changed |= crate::city::viewport(ui, viewport, scene, &tools.camera, &mut tools.city);
        }
        if let Selection::Entity(index) = *selected
            && tools.play == PlayState::Editing
            && tools.loading.is_none()
        {
            let before = scene.entities[index].transform.clone();
            if tools.gizmo.show(
                ui,
                viewport,
                &mut scene.entities[index].transform,
                &tools.camera,
            ) {
                scene.carry_children(index, &before);
                changed = true;
            }
        }
        self.viewport = viewport;
        self.changed |= changed;
    }
}

const WIDGET_KINDS: [(&str, WidgetKind); 4] = [
    ("Panel", WidgetKind::Panel),
    ("Text", WidgetKind::Text),
    ("Button", WidgetKind::Button),
    ("Bar", WidgetKind::Bar),
];

fn add_widget(scene: &mut Scene, selected: &mut Selection, kind: WidgetKind) {
    let label = WIDGET_KINDS
        .iter()
        .find(|(_, k)| *k == kind)
        .map_or("Widget", |(l, _)| l);
    scene.ui.push(Widget {
        name: format!("{label} {}", scene.ui.len() + 1),
        kind,
        text: match kind {
            WidgetKind::Text | WidgetKind::Button => label.into(),
            WidgetKind::Panel | WidgetKind::Bar => String::new(),
        },
        background: match kind {
            WidgetKind::Text => [0; 4],
            WidgetKind::Button => [10, 50, 36, 235],
            WidgetKind::Panel | WidgetKind::Bar => Widget::default().background,
        },
        color: match kind {
            WidgetKind::Bar => [57, 255, 136, 255],
            _ => Widget::default().color,
        },
        size: match kind {
            WidgetKind::Bar => [400.0, 24.0],
            _ => Widget::default().size,
        },
        ..Widget::default()
    });
    *selected = Selection::Widget(scene.ui.len() - 1);
}

/// Click a game UI widget in the viewport to select it, drag to move it.
fn edit_widgets(
    ui: &mut egui::Ui,
    viewport: Rect,
    scene: &mut Scene,
    selected: &mut Selection,
) -> bool {
    if viewport.height() <= 0.0 {
        return false;
    }
    let mut changed = false;
    // Each widget takes its own hit area, so the rest of the viewport keeps camera input.
    for index in 0..scene.ui.len() {
        let widget = &mut scene.ui[index];
        if !widget.visible {
            continue;
        }
        let rect = verity_ui::rect(widget, viewport);
        let response = ui.interact(
            rect,
            ui.id().with(("edit_widget", index)),
            Sense::click_and_drag(),
        );
        if response.clicked() || response.drag_started() {
            *selected = Selection::Widget(index);
        }
        let delta = response.drag_delta() / verity_ui::scale(viewport);
        if response.dragged() && *selected == Selection::Widget(index) && delta != Vec2::ZERO {
            widget.offset[0] += delta.x;
            widget.offset[1] += delta.y;
            changed = true;
        }
        if response.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Move);
        }
    }
    if let Selection::Widget(index) = *selected {
        verity_ui::outline(
            &ui.painter_at(viewport),
            &scene.ui[index],
            viewport,
            Stroke::new(1.5, ACCENT),
        );
    }
    changed
}

fn widget_inspector(ui: &mut egui::Ui, widget: &mut Widget) -> bool {
    let mut changed = false;
    section(ui, "UI WIDGET");
    changed |= ui
        .add(egui::TextEdit::singleline(&mut widget.name).desired_width(f32::INFINITY))
        .changed();
    ui.horizontal(|ui| {
        ui.label(RichText::new("Kind").size(11.5).color(MUTED));
        for (label, kind) in WIDGET_KINDS {
            changed |= ui.selectable_value(&mut widget.kind, kind, label).changed();
        }
    });
    changed |= ui.checkbox(&mut widget.visible, "Visible").changed();
    section(ui, "LAYOUT");
    ui.horizontal(|ui| {
        ui.label(RichText::new("Anchor").size(11.5).color(MUTED))
            .on_hover_text("Sets the anchor and pivot to a screen edge or corner");
        egui::Grid::new("anchor_presets")
            .spacing([2.0, 2.0])
            .show(ui, |ui| {
                for y in [0.0, 0.5, 1.0] {
                    for x in [0.0, 0.5, 1.0] {
                        let on = widget.anchor == [x, y] && widget.pivot == [x, y];
                        if ui.selectable_label(on, "■").clicked() && !on {
                            widget.anchor = [x, y];
                            widget.pivot = [x, y];
                            changed = true;
                        }
                    }
                    ui.end_row();
                }
            });
    });
    changed |= vector_speed(ui, "Anchor (0–1 of screen)", &mut widget.anchor, 0.005);
    changed |= vector_speed(ui, "Pivot (0–1 of widget)", &mut widget.pivot, 0.005);
    changed |= vector_speed(ui, "Offset (px @1080p)", &mut widget.offset, 1.0);
    changed |= vector_speed(ui, "Size (px @1080p)", &mut widget.size, 1.0);
    section(ui, "LOOK");
    if widget.kind != WidgetKind::Bar {
        ui.label(RichText::new("Text").size(11.5).color(MUTED));
        changed |= ui
            .add(
                egui::TextEdit::multiline(&mut widget.text)
                    .desired_rows(2)
                    .desired_width(f32::INFINITY),
            )
            .changed();
        changed |= scalar(ui, "Font size", &mut widget.font_size, 1.0..=400.0, 0.5);
    } else {
        changed |= scalar(ui, "Value", &mut widget.value, 0.0..=1.0, 0.005);
    }
    for (label, value) in [
        (
            if widget.kind == WidgetKind::Bar {
                "Fill"
            } else {
                "Text color"
            },
            &mut widget.color,
        ),
        ("Background", &mut widget.background),
    ] {
        ui.horizontal(|ui| {
            ui.label(RichText::new(label).size(11.5).color(MUTED));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let mut edited =
                    Color32::from_rgba_unmultiplied(value[0], value[1], value[2], value[3]);
                if ui.color_edit_button_srgba(&mut edited).changed() {
                    let unmultiplied = edited.to_srgba_unmultiplied();
                    changed |= *value != unmultiplied;
                    *value = unmultiplied;
                }
            });
        });
    }
    changed |= scalar(
        ui,
        "Corner radius",
        &mut widget.corner_radius,
        0.0..=200.0,
        0.5,
    );
    changed
}

/// What the docked panels edit, and what they report back to [`show`].
struct Panels<'a> {
    scene: &'a mut Scene,
    selected: &'a mut Selection,
    tools: &'a mut Tools,
    changed: bool,
    /// Delete the selection, as asked from the hierarchy or inspector.
    delete: bool,
    asset_action: Option<crate::asset_browser::AssetAction>,
    asset_remove: Option<usize>,
    /// The unobscured scene viewport, in egui points.
    viewport: Rect,
}

impl egui_dock::TabViewer for Panels<'_> {
    type Tab = Panel;

    fn id(&mut self, tab: &mut Panel) -> egui::Id {
        egui::Id::new(*tab)
    }

    fn title(&mut self, tab: &mut Panel) -> egui::WidgetText {
        tab.title().into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Panel) {
        match tab {
            Panel::Viewport => self.viewport(ui),
            Panel::Hierarchy => self.hierarchy(ui),
            Panel::Inspector => self.inspector(ui),
            Panel::Assets => self.assets(ui),
            Panel::Profiler => self.tools.profiler.panel(ui),
            Panel::Console => console_panel(ui),
        }
    }

    fn is_closeable(&self, tab: &Panel) -> bool {
        *tab != Panel::Viewport
    }

    fn allowed_in_windows(&self, tab: &mut Panel) -> bool {
        // The scene draws beneath egui at the main viewport only.
        *tab != Panel::Viewport
    }

    fn clear_background(&self, tab: &Panel) -> bool {
        *tab != Panel::Viewport
    }

    fn scroll_bars(&self, _tab: &Panel) -> [bool; 2] {
        // Panels scroll themselves.
        [false, false]
    }
}

/// The dock in the editor's colors: panels as cards, tabs on the bar color.
fn dock_style(style: &egui::Style) -> egui_dock::Style {
    let mut dock = egui_dock::Style::from_egui(style);
    dock.main_surface_border_stroke = Stroke::NONE;
    dock.tab_bar.bg_fill = BAR;
    dock.tab_bar.hline_color = BORDER;
    dock.tab.tab_body.bg_fill = PANEL;
    dock.tab.tab_body.stroke = Stroke::new(1.0, BORDER);
    dock.tab.tab_body.inner_margin = Margin::symmetric(10, 8);
    dock.tab.active.bg_fill = PANEL;
    dock.tab.active.text_color = ACCENT;
    dock.tab.focused.bg_fill = PANEL;
    dock.tab.focused.text_color = ACCENT;
    dock.tab.inactive.text_color = MUTED;
    dock.separator.color_idle = BACKGROUND;
    dock.separator.color_hovered = HOVER;
    dock.separator.color_dragged = ACCENT;
    dock
}

/// Editor messages, newest last, with a Clear button.
fn console_panel(ui: &mut egui::Ui) {
    crate::console::with(|log| {
        ui.horizontal(|ui| {
            if ui.button("Clear").clicked() {
                log.clear();
            }
            let errors = log
                .iter()
                .filter(|(level, _)| *level == crate::console::Level::Error)
                .count();
            ui.label(
                RichText::new(format!("{} messages · {errors} errors", log.len())).color(MUTED),
            );
        });
        ui.separator();
        if log.is_empty() {
            ui.label(RichText::new("No messages").color(MUTED).italics());
            return;
        }
        let font = egui::FontId::monospace(11.0);
        let height = ui.ctx().fonts_mut(|fonts| fonts.row_height(&font));
        // One line per message, so only the rows in view are laid out; hover shows the rest.
        egui::ScrollArea::vertical()
            .auto_shrink(false)
            .stick_to_bottom(true)
            .show_rows(ui, height, log.len(), |ui, rows| {
                for (level, message) in &log[rows] {
                    let color = match level {
                        crate::console::Level::Info => TEXT,
                        crate::console::Level::Error => PINK,
                    };
                    ui.add(
                        egui::Label::new(RichText::new(message).font(font.clone()).color(color))
                            .truncate(),
                    );
                }
            });
    });
}

/// A dimmed editor with a centred progress card, while a load holds the renderer.
fn loading_popup(ctx: &egui::Context, progress: &LoadProgress) {
    let screen = ctx.content_rect();
    ctx.layer_painter(egui::LayerId::new(
        egui::Order::Middle,
        egui::Id::new("load_scrim"),
    ))
    .rect_filled(screen, 0, Color32::from_black_alpha(150));
    egui::Window::new("loading")
        .title_bar(false)
        .order(egui::Order::Foreground)
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .fixed_size([360.0, 0.0])
        .frame(
            Frame::new()
                .corner_radius(12)
                .fill(PANEL)
                .stroke(Stroke::new(1.0, CYAN.gamma_multiply(0.5)))
                .shadow(glow(CYAN, 0.35))
                .inner_margin(Margin::same(18)),
        )
        .show(ctx, |ui| {
            ui.label(
                RichText::new("// LOADING SCENE")
                    .monospace()
                    .size(12.0)
                    .color(CYAN),
            );
            ui.add_space(10.0);
            ui.label(RichText::new(&progress.label).color(TEXT));
            ui.add_space(8.0);
            ui.add(
                egui::ProgressBar::new(progress.fraction())
                    .fill(ACCENT.gamma_multiply(0.8))
                    .desired_height(6.0)
                    .corner_radius(3),
            );
            ui.add_space(6.0);
            ui.label(
                RichText::new(format!(
                    "step {} of {}  ·  {:.0}%",
                    (progress.done + 1).min(progress.total),
                    progress.total,
                    progress.fraction() * 100.0
                ))
                .monospace()
                .size(10.5)
                .color(MUTED),
            );
        });
}

#[derive(Clone, Copy)]
enum RowAction {
    Frame,
    Duplicate,
    Delete,
}

/// A hierarchy row's right-click menu.
fn row_menu(row: &egui::Response, item: Selection, editing: bool) -> Option<RowAction> {
    let mut chosen = None;
    row.context_menu(|ui| {
        for (action, label, shortcut, enabled) in [
            (RowAction::Frame, "Frame", "F", true),
            (
                RowAction::Duplicate,
                "Duplicate",
                "Ctrl+D",
                editing && matches!(item, Selection::Entity(_) | Selection::Widget(_)),
            ),
            (RowAction::Delete, "Delete", "Del", editing),
        ] {
            let button = egui::Button::new(label).shortcut_text(shortcut);
            if ui.add_enabled(enabled, button).clicked() {
                chosen = Some(action);
                ui.close();
            }
        }
    });
    chosen
}

/// Rotation as XYZ Euler degrees, like other engine editors; stored as a quaternion.
fn euler(ui: &mut egui::Ui, rotation: &mut [f32; 4]) -> bool {
    let quat = glam::Vec4::from_array(*rotation)
        .try_normalize()
        .map_or(glam::Quat::IDENTITY, glam::Quat::from_vec4);
    let (x, y, z) = quat.to_euler(glam::EulerRot::XYZ);
    // + 0.0 turns -0 into 0 so the fields do not read "-0.0".
    let mut degrees = [x, y, z].map(|r| r.to_degrees() + 0.0);
    let changed = vector_speed(ui, "Rotation °", &mut degrees, 0.5);
    if changed {
        let [x, y, z] = degrees.map(f32::to_radians);
        *rotation = glam::Quat::from_euler(glam::EulerRot::XYZ, x, y, z).to_array();
    }
    changed
}

/// Move the view to look at the selection along the current view direction.
fn focus(scene: &Scene, selected: Selection, tools: &mut Tools) {
    let target = match selected {
        Selection::Entity(i) => scene.entities.get(i).map(|e| {
            let size = Vec3::from(e.transform.scale).abs().max_element();
            (Vec3::from(e.transform.translation), size)
        }),
        Selection::Light(i) => scene.lights.get(i).map(|l| (Vec3::from(l.position), 1.0)),
        Selection::Camera(i) => scene.cameras.get(i).map(|c| (Vec3::from(c.position), 1.0)),
        _ => None,
    };
    if let Some((center, size)) = target {
        tools.camera.eye = center - tools.fly.forward() * (size * 2.5 + 1.5);
        tools.camera.target = center;
    }
}

/// Copy the selected entity beside itself and select the copy.
pub(crate) fn duplicate(scene: &mut Scene, selected: &mut Selection) -> bool {
    if let Selection::Widget(index) = *selected
        && let Some(mut copy) = scene.ui.get(index).cloned()
    {
        copy.name.push_str(" copy");
        copy.offset[0] += 20.0;
        copy.offset[1] += 20.0;
        scene.ui.push(copy);
        *selected = Selection::Widget(scene.ui.len() - 1);
        return true;
    }
    let Selection::Entity(index) = *selected else {
        return false;
    };
    let Some(mut copy) = scene.entities.get(index).cloned() else {
        return false;
    };
    copy.name.push_str(" copy");
    copy.transform.translation[0] += 1.0;
    scene.entities.push(copy);
    *selected = Selection::Entity(scene.entities.len() - 1);
    true
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
fn count(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut usize,
    range: std::ops::RangeInclusive<usize>,
) -> bool {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).size(11.5).color(MUTED));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_sized([110.0, 22.0], DragValue::new(value).range(range))
                .changed()
        })
        .inner
    })
    .inner
}

/// A seed from the clock, for rolling new worlds.
fn fresh_seed() -> u32 {
    (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH))
        .map_or(1, |d| d.subsec_nanos() ^ d.as_secs() as u32)
}

/// The scene's terrain recipe. Every change regenerates in the background, like World Creator.
fn city_inspector(ui: &mut egui::Ui, scene: &mut Scene, tools: &mut Tools) -> bool {
    use verity_pcg::{City, Kind, MAX_LANES};
    let mut city = crate::city::recipe(scene);
    let before = city.clone();
    let tool = &mut tools.city;
    section(ui, "CITY");
    let mut enabled = city.is_some();
    if ui.checkbox(&mut enabled, "Enabled").changed() {
        city = enabled.then(City::default);
        tool.drawing = None;
        tool.edge = None;
    }
    let Some(c) = &mut city else {
        ui.label(
            RichText::new("Draw roads, highways and rivers as splines; blocks, buildings, traffic and pedestrians follow from them.")
                .size(11.0)
                .color(MUTED),
        );
        scene.city = None;
        return before.is_some();
    };
    ui.horizontal_wrapped(|ui| {
        for kind in Kind::ALL {
            let on = tool.drawing == Some(kind);
            if ui
                .selectable_label(on, format!("✏ {}", kind.name()))
                .clicked()
            {
                tool.drawing = (!on).then_some(kind);
                tool.pending.clear();
            }
        }
    });
    ui.label(
        RichText::new(if tool.drawing.is_some() {
            "Click the ground to place points; Enter or double-click finishes, Esc stops. Ends snap to nodes and split roads."
        } else {
            "Click a spline to select it; drag its points to edit. Delete removes it."
        })
        .size(11.0)
        .color(MUTED),
    );
    ui.horizontal(|ui| {
        ui.label(RichText::new("Seed").size(11.5).color(MUTED));
        ui.add(DragValue::new(&mut c.seed));
        if ui
            .button("🎲")
            .on_hover_text("Roll new buildings")
            .clicked()
        {
            c.seed = fresh_seed();
        }
    });
    if let Some(e) = tool.edge.filter(|&e| e < c.edges.len()) {
        section(ui, &format!("EDGE {}", e + 1));
        let edge = &mut c.edges[e];
        let mut kind = edge.kind;
        ui.horizontal(|ui| {
            ui.label(RichText::new("Kind").size(11.5).color(MUTED));
            egui::ComboBox::from_id_salt("city_edge_kind")
                .selected_text(kind.name())
                .show_ui(ui, |ui| {
                    for k in Kind::ALL {
                        ui.selectable_value(&mut kind, k, k.name());
                    }
                });
        });
        if kind != edge.kind {
            (edge.lanes, edge.width) = kind.defaults();
            edge.kind = kind;
        }
        if kind.drives() {
            for (name, lanes) in [("Lanes forward", 0), ("Lanes back", 1)] {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(name).size(11.5).color(MUTED));
                    ui.add(DragValue::new(&mut edge.lanes[lanes]).range(0..=MAX_LANES));
                });
            }
        }
        scalar(ui, "Width m", &mut edge.width, 1.0..=200.0, 0.1);
        if ui
            .button(RichText::new("Delete edge").color(PINK))
            .clicked()
        {
            c.remove_edge(e);
            tool.edge = None;
        }
    }
    let rules = &mut c.rules;
    section(ui, "BLOCKS & BUILDINGS");
    scalar(ui, "Sidewalk m", &mut rules.sidewalk, 0.5..=20.0, 0.1);
    scalar(ui, "Lot width m", &mut rules.lot_width, 4.0..=200.0, 0.5);
    scalar(ui, "Lot depth m", &mut rules.lot_depth, 4.0..=200.0, 0.5);
    scalar(ui, "Setback m", &mut rules.setback, 0.0..=20.0, 0.1);
    for (name, k) in [("Fewest floors", 0), ("Most floors", 1)] {
        ui.horizontal(|ui| {
            ui.label(RichText::new(name).size(11.5).color(MUTED));
            ui.add(DragValue::new(&mut rules.floors[k]).range(1..=200));
        });
    }
    scalar(
        ui,
        "Floor height m",
        &mut rules.floor_height,
        2.0..=10.0,
        0.05,
    );
    scalar(
        ui,
        "Downtown radius m",
        &mut rules.downtown,
        1.0..=10_000.0,
        5.0,
    );
    section(ui, "LIFE · WHILE PLAYING");
    scalar(
        ui,
        "Vehicles per lane km",
        &mut rules.vehicles_per_km,
        0.0..=200.0,
        1.0,
    );
    scalar(
        ui,
        "Pedestrians per km",
        &mut rules.walkers_per_km,
        0.0..=1000.0,
        1.0,
    );
    scalar(ui, "Green light s", &mut rules.green, 2.0..=120.0, 0.5);
    section(ui, "VIEW");
    ui.checkbox(&mut tool.overlay, "Lanes and sidewalks");
    ui.checkbox(&mut tool.interiors, "Interiors near the camera");
    if let Some(status) = &tool.status {
        ui.add_space(6.0);
        ui.label(RichText::new(status).size(11.0).color(MUTED));
    }
    if city == before {
        return false;
    }
    scene.city = city.map(|c| serde_json::to_value(c).expect("city recipes serialize"));
    true
}

fn terrain_inspector(ui: &mut egui::Ui, scene: &mut Scene, tools: &mut Tools) -> bool {
    use verity_terra::{Filter, Generator, Landform, Settings, planet::PlanetSettings};
    let mut generator = crate::terrain::generator(scene);
    let before = generator.clone();
    section(ui, "TERRAIN GENERATOR");
    let kind = |g: &Option<Generator>| match g {
        None => 0,
        Some(Generator::Landscape(_)) => 1,
        Some(Generator::Planet(_)) => 2,
    };
    let mut chosen = kind(&generator);
    ui.horizontal(|ui| {
        ui.label(RichText::new("Kind").size(11.5).color(MUTED));
        egui::ComboBox::from_id_salt("terrain_kind")
            .width(ui.available_width())
            .selected_text(["None", "Landscape", "Planet"][chosen])
            .show_ui(ui, |ui| {
                for (value, name) in [(0, "None"), (1, "Landscape"), (2, "Planet")] {
                    ui.selectable_value(&mut chosen, value, name);
                }
            });
    });
    if chosen != kind(&generator) {
        generator = match chosen {
            1 => Some(Generator::Landscape(Settings::default())),
            2 => Some(Generator::Planet(PlanetSettings::default())),
            _ => None,
        };
    }
    if generator.is_some() {
        ui.horizontal(|ui| {
            if ui
                .button("🎲 Auto generate")
                .on_hover_text("Roll a new seed and a whole new recipe")
                .clicked()
            {
                generator = match generator.take() {
                    Some(Generator::Landscape(_)) => {
                        Some(Generator::Landscape(Settings::random(fresh_seed())))
                    }
                    Some(Generator::Planet(p)) => Some(Generator::Planet(p.random(fresh_seed()))),
                    None => None,
                };
            }
            if ui
                .button("Frame")
                .on_hover_text("Move the view to see it all")
                .clicked()
            {
                let fov = tools.camera.vertical_fov.to_degrees();
                let (position, target, speed) = match &generator {
                    Some(Generator::Planet(p)) => {
                        let r = p.radius;
                        ([0.0, -2.4 * r, -0.2 * r], [0.0, 0.0, -r], r * 0.25)
                    }
                    _ => ([-150.0, -175.0, 150.0], [0.0, 0.0, 20.0], 30.0),
                };
                let view = verity_core::Camera {
                    name: String::new(),
                    position,
                    target,
                    vertical_fov: fov,
                };
                look_through(tools, &view);
                tools.fly.speed = speed.clamp(0.25, MAX_FLY_SPEED);
            }
        });
    }
    match &mut generator {
        None => {
            ui.label(
                RichText::new("A landscape is an eroded world, 4 km across by default, that you can dig near the camera; a planet is a whole eroded sphere whose detail follows the camera.")
                    .size(11.0)
                    .color(MUTED),
            );
        }
        Some(Generator::Landscape(s)) => {
            section(ui, "SHAPE");
            let mut landform = s.landform;
            ui.horizontal(|ui| {
                ui.label(RichText::new("Landform").size(11.5).color(MUTED));
                egui::ComboBox::from_id_salt("terrain_landform")
                    .width(ui.available_width())
                    .selected_text(format!("{landform:?}"))
                    .show_ui(ui, |ui| {
                        for form in Landform::ALL {
                            ui.selectable_value(&mut landform, form, format!("{form:?}"));
                        }
                    });
            });
            if landform != s.landform {
                // The preset's shape, keeping the seed and filter stack.
                *s = Settings {
                    seed: s.seed,
                    filters: std::mem::take(&mut s.filters),
                    ..landform.into()
                };
            }
            ui.horizontal(|ui| {
                ui.label(RichText::new("Seed").size(11.5).color(MUTED));
                ui.add(DragValue::new(&mut s.seed));
            });
            let mut sea = s.sea_level.is_some();
            ui.checkbox(&mut sea, "Sea");
            if sea != s.sea_level.is_some() {
                s.sea_level = sea.then_some(0.0);
            }
            if let Some(level) = &mut s.sea_level {
                scalar(ui, "Sea level m", level, -200.0..=500.0, 0.5);
            }
            scalar(ui, "Mountains m", &mut s.mountains, 0.0..=3_000.0, 1.0);
            scalar(ui, "Hills m", &mut s.hills, 0.0..=300.0, 0.5);
            count(ui, "Samples per side", &mut s.size, 65..=4_097);
            scalar(ui, "Sample spacing m", &mut s.cell, 0.5..=32.0, 0.05);
            section(ui, "EROSION");
            count(ui, "River age (steps)", &mut s.fluvial_steps, 0..=400);
            count(ui, "Droplets", &mut s.droplets, 0..=100_000);
            count(ui, "Thermal steps", &mut s.thermal_steps, 0..=200);
            scalar(ui, "Erosion rate", &mut s.erosion_rate, 0.0..=1.0, 0.005);
            scalar(ui, "Deposition", &mut s.deposition_rate, 0.0..=1.0, 0.005);
            scalar(ui, "Evaporation", &mut s.evaporation, 0.001..=0.5, 0.001);
            scalar(ui, "Talus rise/run", &mut s.talus, 0.05..=2.0, 0.01);
            section(ui, "FILTERS · APPLIED IN ORDER");
            let mut remove = None;
            let mut raise = None;
            for (index, filter) in s.filters.iter_mut().enumerate() {
                ui.push_id(index, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(filter.name()).strong());
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("✕").on_hover_text("Remove").clicked() {
                                remove = Some(index);
                            }
                            if index > 0
                                && ui
                                    .small_button("⏶")
                                    .on_hover_text("Apply earlier")
                                    .clicked()
                            {
                                raise = Some(index);
                            }
                        });
                    });
                    match filter {
                        Filter::Terraces { step, sharpness } => {
                            scalar(ui, "Step m", step, 0.5..=200.0, 0.1);
                            scalar(ui, "Sharpness", sharpness, 0.0..=1.0, 0.01);
                        }
                        Filter::Smooth { strength, passes } => {
                            scalar(ui, "Strength", strength, 0.0..=1.0, 0.01);
                            let mut n = *passes as usize;
                            count(ui, "Passes", &mut n, 1..=32);
                            *passes = n as u32;
                        }
                        Filter::Plateau { height, strength } => {
                            scalar(ui, "Height m", height, -200.0..=3_000.0, 1.0);
                            scalar(ui, "Strength", strength, 0.0..=1.0, 0.01);
                        }
                        Filter::Sediment { strength } => {
                            scalar(ui, "Strength", strength, 0.0..=1.0, 0.01);
                        }
                        Filter::Level { scale, offset } => {
                            scalar(ui, "Scale", scale, 0.0..=3.0, 0.01);
                            scalar(ui, "Offset m", offset, -500.0..=500.0, 0.5);
                        }
                    }
                });
            }
            if let Some(index) = remove {
                s.filters.remove(index);
            }
            if let Some(index) = raise {
                s.filters.swap(index - 1, index);
            }
            ui.menu_button("＋ Add filter", |ui| {
                for preset in Filter::PRESETS {
                    if ui.button(preset.name()).clicked() {
                        s.filters.push(preset);
                        ui.close();
                    }
                }
            });
        }
        Some(Generator::Planet(p)) => {
            section(ui, "PLANET");
            ui.horizontal(|ui| {
                ui.label(RichText::new("Seed").size(11.5).color(MUTED));
                ui.add(DragValue::new(&mut p.seed));
            });
            scalar(ui, "Radius m", &mut p.radius, 50.0..=50_000.0, 5.0);
            scalar(ui, "Mountains m", &mut p.mountains, 0.0..=5_000.0, 1.0);
            scalar(ui, "Hills m", &mut p.hills, 0.0..=1_000.0, 0.5);
            section(ui, "OCEANS & CLIMATE");
            scalar(ui, "Ocean cover", &mut p.ocean, 0.0..=1.0, 0.005);
            scalar(ui, "Ocean depth m", &mut p.ocean_depth, 0.0..=5_000.0, 1.0);
            scalar(ui, "Continents", &mut p.continents, 0.2..=8.0, 0.01);
            scalar(ui, "Ice caps", &mut p.ice, 0.0..=1.0, 0.005);
            section(ui, "LEVEL OF DETAIL");
            scalar(ui, "Detail", &mut p.detail, 0.5..=8.0, 0.02);
            count(ui, "Erosion grid", &mut p.resolution, 16..=1_024);
            ui.label(
                RichText::new("Patches split while closer than this many of their widths. The origin sits on the equator; the poles lie along Y.")
                    .size(11.0)
                    .color(MUTED),
            );
        }
    }
    if let Some(status) = &tools.terrain_status {
        ui.add_space(6.0);
        ui.label(RichText::new(status).size(11.0).color(MUTED));
    }
    if generator == before {
        return false;
    }
    scene.terrain = generator.map(|g| serde_json::to_value(g).expect("terrain recipes serialize"));
    true
}

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
    fn duplicate_selects_an_offset_copy_of_entities_only() {
        let mut scene = Scene::default();
        add_cube(&mut scene, &mut Selection::None);
        let last = scene.entities.len() - 1;
        let mut selected = Selection::Entity(last);
        assert!(duplicate(&mut scene, &mut selected));
        assert_eq!(selected, Selection::Entity(last + 1));
        let (a, b) = (&scene.entities[last], &scene.entities[last + 1]);
        assert_eq!(b.name, format!("{} copy", a.name));
        assert_eq!(b.transform.translation[0], a.transform.translation[0] + 1.0);
        assert!(!duplicate(&mut scene, &mut Selection::Sun));
        assert_eq!(scene.entities.len(), last + 2);
    }

    #[test]
    fn hierarchy_lays_out_only_visible_entities() {
        let mut scene = Scene::default();
        let template = scene.entities[0].clone();
        scene.entities = (0..5000)
            .map(|i| Entity {
                name: format!("Row {i}"),
                ..template.clone()
            })
            .collect();
        let (mut selected, mut tools) = (Selection::None, Tools::default());
        let context = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, egui::vec2(1280.0, 800.0))),
            ..Default::default()
        };
        let mut output = context.run_ui(input, |ui| {
            show(ui, &mut scene, &mut selected, None, false, &mut tools);
        });
        output.textures_delta.clear();
        let drawn = |label: &str| {
            output
                .shapes
                .iter()
                .any(|s| text_position(&s.shape, label).is_some())
        };
        assert!(drawn("Row 0"));
        assert!(!drawn("Row 4999"));
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
    fn workspace_menus_route_creation_and_reopen_closed_panels() {
        let context = egui::Context::default();
        let mut scene = Scene::default();
        let before = scene.entities.len();
        let (mut selected, mut tools) = (Selection::None, Tools::default());
        let inspector = tools.dock.find_tab(&Panel::Inspector).unwrap();
        tools.dock.remove_tab(inspector);
        let mut render = |events| {
            let input = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, egui::vec2(800.0, 520.0))),
                events,
                ..Default::default()
            };
            let mut output = context.run_ui(input, |ui| {
                show(ui, &mut scene, &mut selected, None, false, &mut tools);
            });
            output.textures_delta.clear();
            output
        };
        let mut output = render(vec![]);
        for label in [
            "+ Add",
            "Cube",
            "Window",
            "Inspector",
            "View",
            "Overlays",
            "World axes",
        ] {
            let pos = output
                .shapes
                .iter()
                .find_map(|s| text_position(&s.shape, label))
                .unwrap_or_else(|| panic!("{label} is visible"));
            render(vec![egui::Event::PointerMoved(pos)]);
            for pressed in [true, false] {
                render(vec![egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                }]);
            }
            // A fresh popup first runs an invisible sizing pass.
            output = render(vec![]);
        }
        assert_eq!(scene.entities.len(), before + 1);
        assert_eq!(selected, Selection::Entity(before));
        assert!(tools.dock.find_tab(&Panel::Inspector).is_some());
        assert!(!tools.show_axes);
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
        let mut output = context.run_ui(
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
        output.textures_delta.clear();
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
