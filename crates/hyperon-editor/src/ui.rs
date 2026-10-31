use std::path::Path;

use egui::{Color32, DragValue, Frame, Margin, RichText, Stroke};
use hyperon_core::{Entity, LightingMethod, Material, PointLight, Scene, Transform};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selection {
    Entity(usize),
    Light(usize),
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
}

pub struct UiResponse {
    pub action: Option<Action>,
    pub asset_action: Option<crate::asset_browser::AssetAction>,
    pub asset_remove: Option<usize>,
    pub changed: bool,
    /// The unobscured scene viewport, in egui points.
    pub viewport: egui::Rect,
}

pub struct Tools {
    pub browser: crate::asset_browser::AssetBrowser,
    pub gizmo: crate::gizmo::Gizmo,
    pub camera: hyperon_render::Camera,
    pub light_limit: usize,
}

impl Default for Tools {
    fn default() -> Self {
        Self {
            browser: Default::default(),
            gizmo: Default::default(),
            camera: Default::default(),
            light_limit: hyperon_render::MAX_LIGHTS,
        }
    }
}

pub(crate) const BACKGROUND: Color32 = Color32::from_rgb(14, 16, 23);
pub(crate) const PANEL: Color32 = Color32::from_rgb(21, 24, 33);
pub(crate) const BAR: Color32 = Color32::from_rgb(25, 28, 39);
pub(crate) const SURFACE: Color32 = Color32::from_rgb(34, 38, 51);
pub(crate) const BORDER: Color32 = Color32::from_rgb(52, 57, 73);
pub(crate) const TEXT: Color32 = Color32::from_rgb(228, 231, 241);
pub(crate) const MUTED: Color32 = Color32::from_rgb(151, 160, 180);
pub(crate) const ACCENT: Color32 = Color32::from_rgb(168, 151, 255);
pub(crate) const ACCENT_DARK: Color32 = Color32::from_rgb(57, 49, 88);
pub(crate) const AQUA: Color32 = Color32::from_rgb(108, 209, 205);

fn panel_frame(color: Color32) -> Frame {
    Frame::new()
        .fill(color)
        .stroke(Stroke::new(1.0, BORDER))
        .inner_margin(Margin::symmetric(14, 12))
}

fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(12.0);
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(egui::vec2(3.0, 12.0), egui::Sense::hover());
        ui.painter().rect_filled(rect, 2, ACCENT);
        ui.label(RichText::new(title).size(10.5).strong().color(MUTED));
    });
    ui.add_space(6.0);
}

fn brand_mark(ui: &mut egui::Ui) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(32.0, 32.0), egui::Sense::hover());
    ui.painter().rect_filled(rect, 9, ACCENT_DARK);
    let center = rect.center();
    let stroke = Stroke::new(2.2, ACCENT);
    ui.painter().line_segment(
        [
            center + egui::vec2(-7.0, -7.0),
            center + egui::vec2(-7.0, 7.0),
        ],
        stroke,
    );
    ui.painter().line_segment(
        [
            center + egui::vec2(7.0, -7.0),
            center + egui::vec2(7.0, 7.0),
        ],
        stroke,
    );
    ui.painter().line_segment(
        [
            center + egui::vec2(-7.0, 4.0),
            center + egui::vec2(7.0, -4.0),
        ],
        stroke,
    );
}

fn panel_heading(ui: &mut egui::Ui, title: &str, subtitle: &str) {
    ui.vertical(|ui| {
        ui.label(RichText::new(title).size(14.0).strong().color(TEXT));
        ui.label(RichText::new(subtitle).size(10.5).color(MUTED));
    });
}
fn hierarchy_row(ui: &mut egui::Ui, name: &str, selected: bool, light: bool) -> bool {
    let response = ui.add_sized(
        [ui.available_width(), 32.0],
        egui::Button::selectable(selected, RichText::new(format!("      {name}")).color(TEXT))
            .right_text("")
            .truncate()
            .corner_radius(6),
    );
    if selected {
        let indicator = egui::Rect::from_min_size(
            response.rect.left_top() + egui::vec2(0.0, 7.0),
            egui::vec2(3.0, response.rect.height() - 14.0),
        );
        ui.painter().rect_filled(indicator, 2, ACCENT);
    }
    let center = response.rect.left_center() + egui::vec2(13.0, 0.0);
    let color = if light { AQUA } else { ACCENT };
    if light {
        ui.painter()
            .circle_stroke(center, 4.0, Stroke::new(1.3, color));
        ui.painter().circle_filled(center, 1.2, color);
    } else {
        let rect = egui::Rect::from_center_size(center, egui::vec2(8.0, 8.0));
        ui.painter()
            .rect_stroke(rect, 1, Stroke::new(1.2, color), egui::StrokeKind::Inside);
    }
    response.clicked()
}

fn vector<const N: usize>(ui: &mut egui::Ui, label: &str, values: &mut [f32; N]) -> bool {
    let mut changed = false;
    ui.label(RichText::new(label).color(MUTED));
    ui.horizontal(|ui| {
        let width = ((ui.available_width()
            - (N.saturating_sub(1) as f32 * ui.spacing().item_spacing.x))
            / N as f32)
            .max(35.0);
        for (index, value) in values.iter_mut().enumerate() {
            let prefix = ["X ", "Y ", "Z ", "W "][index];
            changed |= ui
                .add_sized(
                    [width, 23.0],
                    DragValue::new(value).speed(0.02).prefix(prefix),
                )
                .changed();
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
        ui.label(RichText::new(label).color(MUTED));
        ui.add(DragValue::new(value).speed(speed).range(range))
            .changed()
    })
    .inner
}

fn color(ui: &mut egui::Ui, label: &str, value: &mut [f32; 3]) -> bool {
    let mut edited = *value;
    let changed = ui
        .horizontal(|ui| {
            ui.label(RichText::new(label).color(MUTED));
            ui.color_edit_button_rgb(&mut edited).changed()
        })
        .inner;
    if changed && edited != *value {
        *value = edited;
        true
    } else {
        false
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
    let Tools {
        browser,
        gizmo,
        camera,
        light_limit,
    } = tools;
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = PANEL;
    visuals.window_fill = BAR;
    visuals.extreme_bg_color = BACKGROUND;
    visuals.faint_bg_color = SURFACE;
    visuals.override_text_color = Some(TEXT);
    visuals.widgets.noninteractive.bg_fill = PANEL;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
    visuals.widgets.inactive.bg_fill = SURFACE;
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, BORDER);
    visuals.widgets.hovered.bg_fill = Color32::from_rgb(49, 52, 70);
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, ACCENT);
    visuals.widgets.active.bg_fill = ACCENT_DARK;
    visuals.selection.bg_fill = ACCENT_DARK;
    visuals.selection.stroke = Stroke::new(1.0, ACCENT);
    ui.style_mut().visuals = visuals;
    ui.spacing_mut().item_spacing = egui::vec2(8.0, 6.0);
    ui.spacing_mut().button_padding = egui::vec2(9.0, 5.0);
    ui.spacing_mut().interact_size.y = 28.0;
    ui.style_mut()
        .text_styles
        .insert(egui::TextStyle::Body, egui::FontId::proportional(13.0));
    ui.style_mut()
        .text_styles
        .insert(egui::TextStyle::Button, egui::FontId::proportional(12.0));
    let previous_selection = *selected;

    if matches!(*selected, Selection::Entity(i) if i >= scene.entities.len())
        || matches!(*selected, Selection::Light(i) if i >= scene.lights.len())
    {
        *selected = Selection::None;
    }
    let mut action = None;
    let mut asset_action = None;
    let mut asset_remove = None;
    let mut changed = false;
    let filename = path
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .unwrap_or("Untitled");

    egui::Panel::top("editor_toolbar")
        .frame(
            Frame::new()
                .fill(BAR)
                .stroke(Stroke::new(1.0, BORDER))
                .inner_margin(Margin::symmetric(16, 8)),
        )
        .show(ui, |ui| {
            ui.horizontal_centered(|ui| {
                brand_mark(ui);
                ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing.y = 0.0;
                    ui.label(RichText::new("HYPERON").size(14.0).strong().color(TEXT));
                    ui.label(
                        RichText::new("SCENE EDITOR")
                            .size(9.0)
                            .strong()
                            .color(ACCENT),
                    );
                });
                ui.add_space(18.0);
                ui.separator();
                ui.add_space(8.0);
                if ui
                    .button("New")
                    .on_hover_text("New scene · Ctrl+N")
                    .clicked()
                {
                    action = Some(Action::New);
                }
                if ui
                    .button("Open")
                    .on_hover_text("Open scene · Ctrl+O")
                    .clicked()
                {
                    action = Some(Action::Open);
                }
                if ui
                    .button("Save as")
                    .on_hover_text("Save scene as · Ctrl+Shift+S")
                    .clicked()
                {
                    action = Some(Action::SaveAs);
                }
                if ui
                    .add(
                        egui::Button::new(
                            RichText::new("Save scene")
                                .strong()
                                .color(Color32::from_rgb(20, 17, 32)),
                        )
                        .fill(ACCENT)
                        .corner_radius(6),
                    )
                    .on_hover_text("Save scene · Ctrl+S")
                    .clicked()
                {
                    action = Some(Action::Save);
                }
                if ui.available_width() > 160.0 {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new(if dirty {
                                "Modified"
                            } else if path.is_none() {
                                "New"
                            } else {
                                "Saved"
                            })
                            .size(11.0)
                            .color(if dirty { ACCENT } else { AQUA }),
                        );
                        ui.label(RichText::new(filename).size(12.0).color(TEXT));
                        ui.label(RichText::new("SCENE /").size(10.0).color(MUTED));
                    });
                }
            });
        });

    egui::Panel::bottom("editor_status")
        .frame(
            Frame::new()
                .fill(BACKGROUND)
                .inner_margin(Margin::symmetric(16, 6)),
        )
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let status = if dirty {
                    "UNSAVED CHANGES"
                } else if path.is_none() {
                    "UNTITLED SCENE"
                } else {
                    "SAVED"
                };
                let (dot, _) = ui.allocate_exact_size(egui::vec2(7.0, 7.0), egui::Sense::hover());
                ui.painter()
                    .circle_filled(dot.center(), 3.0, if dirty { ACCENT } else { AQUA });
                ui.label(RichText::new(status).size(10.0).strong().color(MUTED));
                ui.separator();
                ui.label(
                    RichText::new(format!(
                        "{} {}   ·   {} {}",
                        scene.entities.len(),
                        if scene.entities.len() == 1 {
                            "object"
                        } else {
                            "objects"
                        },
                        scene.lights.len(),
                        if scene.lights.len() == 1 {
                            "light"
                        } else {
                            "lights"
                        }
                    ))
                    .size(11.0)
                    .color(MUTED),
                );
                if ui.available_width() > 240.0 {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new("CTRL + S  SAVE    /    TAB  SELECT")
                                .size(10.0)
                                .color(MUTED),
                        );
                    });
                }
            });
        });

    egui::Panel::left("editor_hierarchy")
        .resizable(true)
        .default_size(240.0)
        .min_size(170.0)
        .max_size(350.0)
        .frame(panel_frame(PANEL))
        .show(ui, |ui| {
            panel_heading(ui, "Scene", "OUTLINER");
            ui.add_space(12.0);
            ui.horizontal_wrapped(|ui| {
                if ui.button("+ Object").clicked() {
                    scene.entities.push(Entity {
                        name: format!("Entity {}", scene.entities.len() + 1),
                        transform: Transform::default(),
                        material: Material::default(),
                    });
                    *selected = Selection::Entity(scene.entities.len() - 1);
                    changed = true;
                }
                if ui
                    .add_enabled(
                        scene.lights.len() < *light_limit,
                        egui::Button::new("+ Light"),
                    )
                    .on_hover_text(format!(
                        "Scene can add up to {} point lights after imported lights",
                        *light_limit
                    ))
                    .clicked()
                {
                    scene.lights.push(PointLight {
                        position: [0.0, 3.0, 0.0],
                        color: [1.0, 1.0, 1.0],
                        intensity: 30.0,
                        radius: 15.0,
                    });
                    *selected = Selection::Light(scene.lights.len() - 1);
                    changed = true;
                }
            });
            ui.add_space(5.0);
            ui.separator();
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
                ui.label(
                    RichText::new("Whole scene geometry · not editable entities")
                        .size(11.0)
                        .color(MUTED),
                );
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
                                        .small_button("Remove")
                                        .on_hover_text("Remove model from scene")
                                        .clicked()
                                    {
                                        asset_remove = Some(index);
                                    }
                                    let name = Path::new(asset)
                                        .file_name()
                                        .and_then(|name| name.to_str())
                                        .unwrap_or(asset);
                                    ui.add(egui::Label::new(name).truncate())
                                        .on_hover_text(asset);
                                },
                            );
                        });
                    });
                }
                section(ui, "LIGHTING");
                if hierarchy_row(ui, "Environment", *selected == Selection::Lighting, true) {
                    *selected = Selection::Lighting;
                }
                if hierarchy_row(ui, "Sun · Directional", *selected == Selection::Sun, true) {
                    *selected = Selection::Sun;
                }
                if scene.lights.is_empty() {
                    ui.label(RichText::new("No point lights").color(MUTED).italics());
                }
                for index in 0..scene.lights.len() {
                    if hierarchy_row(
                        ui,
                        &format!("Point Light {}", index + 1),
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
        .min_size(220.0)
        .max_size(420.0)
        .frame(panel_frame(PANEL))
        .show(ui, |ui| {
            panel_heading(ui, "Inspector", "PROPERTIES");
            ui.add_space(10.0);
            ui.separator();
            let mut delete = false;
            egui::ScrollArea::vertical().show(ui, |ui| match *selected {
                Selection::Entity(index) => {
                    let entity = &mut scene.entities[index];
                    section(ui, "ENTITY");
                    ui.label(RichText::new("Name").color(MUTED));
                    changed |= ui.text_edit_singleline(&mut entity.name).changed();
                    section(ui, "TRANSFORM");
                    changed |= vector(ui, "Position", &mut entity.transform.translation);
                    changed |= vector(ui, "Rotation · quaternion", &mut entity.transform.rotation);
                    changed |= vector(ui, "Scale", &mut entity.transform.scale);
                    ui.add_space(6.0);
                    ui.separator();
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
                    ui.add_space(6.0);
                    ui.separator();
                    section(ui, "LIGHT");
                    changed |= color(ui, "Color", &mut light.color);
                    changed |= scalar(ui, "Intensity", &mut light.intensity, 0.0..=f32::MAX, 0.1);
                    changed |= scalar(ui, "Radius", &mut light.radius, 0.0..=f32::MAX, 0.1);
                    ui.add_space(14.0);
                    delete = ui.button("Delete light").clicked();
                }
                Selection::Sun => {
                    section(ui, "SUN · DIRECTIONAL LIGHT");
                    let mut enabled = scene.sun.is_some();
                    if ui.checkbox(&mut enabled, "Enabled").changed() {
                        scene.sun = enabled.then(hyperon_core::Sun::default);
                        changed = true;
                    }
                    if let Some(sun) = &mut scene.sun {
                        changed |= vector(ui, "Direction light travels", &mut sun.direction);
                        ui.label(RichText::new("Zero direction uses straight down").color(MUTED));
                        changed |= color(ui, "Color", &mut sun.color);
                        changed |= scalar(ui, "Intensity", &mut sun.intensity, 0.0..=f32::MAX, 0.1);
                    }
                }
                Selection::Lighting => {
                    section(ui, "SCENE LIGHTING");
                    ui.label(RichText::new("Method").color(MUTED));
                    let previous = scene.lighting.method;
                    egui::ComboBox::from_id_salt("lighting_method")
                        .selected_text(match scene.lighting.method {
                            LightingMethod::DirectOnly => "Direct only",
                            LightingMethod::DirectAndHemisphere => "Direct + Hemisphere ambient",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut scene.lighting.method,
                                LightingMethod::DirectOnly,
                                "Direct only",
                            );
                            ui.selectable_value(
                                &mut scene.lighting.method,
                                LightingMethod::DirectAndHemisphere,
                                "Direct + Hemisphere ambient",
                            );
                        });
                    changed |= previous != scene.lighting.method;
                    ui.add_space(6.0);
                    ui.separator();
                    section(ui, "HEMISPHERE AMBIENT");
                    ui.label(RichText::new("Linear sky and ground radiance").color(MUTED));
                    changed |= color(ui, "Sky", &mut scene.lighting.sky);
                    changed |= color(ui, "Ground", &mut scene.lighting.ground);
                }
                Selection::None => {
                    ui.add_space(20.0);
                    ui.label(RichText::new("Nothing selected").strong());
                    ui.label(
                        RichText::new(
                            "Select an entity or light in the hierarchy to edit its properties.",
                        )
                        .color(MUTED),
                    );
                }
            });
            if delete {
                gizmo.cancel_drag();
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
                    Selection::Sun => Selection::Sun,
                    Selection::Lighting => Selection::Lighting,
                    Selection::None => Selection::None,
                };
                changed = true;
            }
        });

    egui::Panel::bottom("editor_assets")
        .resizable(true)
        .default_size(220.0)
        .min_size(130.0)
        .frame(panel_frame(PANEL))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Assets").size(14.0).strong().color(TEXT));
                ui.label(
                    RichText::new("PROJECT LIBRARY")
                        .size(10.0)
                        .strong()
                        .color(MUTED),
                );
                if ui.available_width() > 220.0 {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new("Double-click to open or import")
                                .size(10.5)
                                .color(MUTED),
                        );
                    });
                }
            });
            ui.add_space(4.0);
            asset_action = browser.show(ui);
        });

    if *selected != previous_selection {
        gizmo.cancel_drag();
    }

    let mut viewport = egui::Rect::NOTHING;
    egui::CentralPanel::default()
        .frame(Frame::new().fill(Color32::TRANSPARENT))
        .show(ui, |ui| {
            Frame::new()
                .fill(BAR)
                .stroke(Stroke::new(1.0, BORDER))
                .inner_margin(Margin::symmetric(12, 6))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("VIEWPORT").size(10.0).strong().color(ACCENT));
                        ui.separator();
                        gizmo.toolbar(ui);
                        if ui.available_width() > 150.0 {
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    ui.label(
                                        RichText::new("PERSPECTIVE  ·  VULKAN")
                                            .size(10.0)
                                            .color(MUTED),
                                    );
                                },
                            );
                        }
                    });
                });
            viewport = ui.available_rect_before_wrap();
            if let Selection::Entity(index) = *selected {
                changed |= gizmo.show(ui, viewport, &mut scene.entities[index].transform, camera);
            }
        });
    UiResponse {
        action,
        asset_action,
        asset_remove,
        changed,
        viewport,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
