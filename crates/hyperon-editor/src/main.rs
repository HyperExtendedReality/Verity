mod asset_browser;
mod gizmo;
mod ui;

use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use hyperon_core::{Entity, Material, PointLight, Scene, Transform};
use hyperon_render::{Renderer, UiFrame};
use ui::{Action, Selection};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, ModifiersState, NamedKey},
    window::{Window, WindowId},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::<Instant>::with_user_event().build()?;
    event_loop.set_control_flow(ControlFlow::Wait);
    let context = egui::Context::default();
    let proxy = event_loop.create_proxy();
    context.set_request_repaint_callback(move |info| {
        if let Some(when) = Instant::now().checked_add(info.delay) {
            let _ = proxy.send_event(when);
        }
    });
    event_loop.run_app(&mut Editor::new(context))?;
    Ok(())
}

struct Editor {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    context: egui::Context,
    egui_state: Option<egui_winit::State>,
    next_repaint: Option<Instant>,
    scene: Scene,
    path: Option<PathBuf>,
    selected: Selection,
    tools: ui::Tools,
    modifiers: ModifiersState,
    dirty: bool,
}

impl Editor {
    fn new(context: egui::Context) -> Self {
        Self {
            window: None,
            renderer: None,
            context,
            egui_state: None,
            next_repaint: None,
            scene: Scene::default(),
            path: None,
            selected: Selection::Entity(0),
            tools: ui::Tools::default(),
            modifiers: ModifiersState::empty(),
            dirty: false,
        }
    }

    fn error(message: impl std::fmt::Display) {
        eprintln!("{message}");
        rfd::MessageDialog::new()
            .set_title("Hyperon error")
            .set_description(message.to_string())
            .show();
    }

    fn can_discard(&self) -> bool {
        !self.dirty
            || rfd::MessageDialog::new()
                .set_title("Unsaved changes")
                .set_description("Discard unsaved scene changes?")
                .set_buttons(rfd::MessageButtons::YesNo)
                .show()
                == rfd::MessageDialogResult::Yes
    }

    fn title(&self) {
        if let Some(window) = &self.window {
            let file = self
                .path
                .as_deref()
                .and_then(Path::file_name)
                .and_then(|s| s.to_str())
                .unwrap_or("Untitled");
            window.set_title(&format!(
                "Hyperon — {file}{}",
                if self.dirty { "*" } else { "" }
            ));
        }
    }

    fn redraw(&self) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn changed(&mut self) {
        self.dirty = true;
        self.title();
        self.redraw();
    }
    /// Resolve stored project-relative asset paths without following them outside assets/.
    fn asset_path(path: &str) -> Result<PathBuf, String> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let assets = root
            .join("assets")
            .canonicalize()
            .map_err(|e| e.to_string())?;
        let resolved = root
            .join(path)
            .canonicalize()
            .map_err(|e| format!("{path}: {e}"))?;
        if !resolved.starts_with(&assets)
            || !resolved
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("glb"))
        {
            return Err(format!(
                "{path}: expected a GLB inside the project assets directory"
            ));
        }
        Ok(resolved)
    }

    fn load_assets(
        renderer: &mut Renderer,
        assets: &[String],
        scene_lights: usize,
    ) -> Result<usize, String> {
        let paths = assets
            .iter()
            .map(|path| Self::asset_path(path))
            .collect::<Result<Vec<_>, _>>()?;
        renderer.clear_gltf().map_err(|e| e.to_string())?;
        let mut count = scene_lights;
        for path in paths {
            let stats = renderer
                .load_gltf(&path, 512)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            count += stats.lights;
            if count > hyperon_render::MAX_LIGHTS {
                return Err(format!(
                    "Imported scenes and scene lights exceed the {}-light limit",
                    hyperon_render::MAX_LIGHTS
                ));
            }
        }
        // The editor's Sun toggle owns directional lighting, including when a
        // model carries a glTF directional light. Imported point lights remain.
        renderer.set_sun(None);
        Ok(count - scene_lights)
    }

    /// On failure restore the previous scene's imported geometry.
    fn replace_assets(&mut self, assets: &[String], scene_lights: usize) -> Result<(), String> {
        let renderer = self.renderer.as_mut().ok_or("Renderer is not available")?;
        match Self::load_assets(renderer, assets, scene_lights) {
            Ok(imported) => {
                self.tools.light_limit = hyperon_render::MAX_LIGHTS - imported;
                Ok(())
            }
            Err(error) => {
                match Self::load_assets(renderer, &self.scene.assets, self.scene.lights.len()) {
                    Ok(imported) => self.tools.light_limit = hyperon_render::MAX_LIGHTS - imported,
                    Err(restore) => {
                        return Err(format!(
                            "{error}\nCould not restore the previous models: {restore}"
                        ));
                    }
                }
                Err(error)
            }
        }
    }

    fn import_model(&mut self, path: PathBuf) {
        let Some(path) = path.to_str().map(str::to_owned) else {
            Self::error("The model path is not valid Unicode");
            return;
        };
        if self.scene.assets.contains(&path) {
            return;
        }
        let mut assets = self.scene.assets.clone();
        assets.push(path);
        match self.replace_assets(&assets, self.scene.lights.len()) {
            Ok(()) => {
                self.scene.assets = assets;
                self.changed();
            }
            Err(error) => Self::error(format!("Cannot import model: {error}")),
        }
    }

    fn remove_model(&mut self, index: usize) {
        if index >= self.scene.assets.len() {
            return;
        }
        let mut assets = self.scene.assets.clone();
        assets.remove(index);
        match self.replace_assets(&assets, self.scene.lights.len()) {
            Ok(()) => {
                self.scene.assets = assets;
                self.changed();
            }
            Err(error) => Self::error(format!("Cannot remove model: {error}")),
        }
    }

    fn new_scene(&mut self) {
        if self.can_discard() {
            if !self.scene.assets.is_empty()
                && let Err(error) = self.replace_assets(&[], 0)
            {
                Self::error(format!("Cannot create scene: {error}"));
                return;
            }
            self.tools.gizmo.cancel_drag();
            self.scene = Scene::default();
            self.path = None;
            self.selected = Selection::Entity(0);
            self.dirty = false;
            self.title();
            self.redraw();
        }
    }

    fn open(&mut self) {
        if !self.can_discard() {
            return;
        }
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Hyperon scene", &["hyperon"])
            .pick_file()
        else {
            return;
        };
        self.open_path(path);
    }

    fn open_path(&mut self, path: PathBuf) {
        let result = std::fs::read(&path)
            .map_err(|e| e.to_string())
            .and_then(|data| serde_json::from_slice::<Scene>(&data).map_err(|e| e.to_string()));
        match result {
            Ok(scene) if scene.lights.len() > hyperon_render::MAX_LIGHTS => {
                Self::error(format!(
                    "Cannot open {}: the renderer supports at most {} point lights",
                    path.display(),
                    hyperon_render::MAX_LIGHTS
                ));
            }
            Ok(scene) => {
                if let Err(error) = self.replace_assets(&scene.assets, scene.lights.len()) {
                    Self::error(format!("Cannot open {}: {error}", path.display()));
                    return;
                }
                self.tools.gizmo.cancel_drag();
                self.selected = if scene.entities.is_empty() {
                    Selection::None
                } else {
                    Selection::Entity(0)
                };
                self.scene = scene;
                self.path = Some(path);
                self.dirty = false;
                self.title();
                self.redraw();
            }
            Err(error) => Self::error(format!("Cannot open {}: {error}", path.display())),
        }
    }

    fn save(&mut self, save_as: bool) {
        let path = if save_as || self.path.is_none() {
            rfd::FileDialog::new()
                .add_filter("Hyperon scene", &["hyperon"])
                .set_file_name("scene.hyperon")
                .save_file()
        } else {
            self.path.clone()
        };
        let Some(path) = path else { return };
        let result = serde_json::to_vec_pretty(&self.scene)
            .map_err(|e| e.to_string())
            .and_then(|data| std::fs::write(&path, data).map_err(|e| e.to_string()));
        match result {
            Ok(()) => {
                self.path = Some(path);
                self.dirty = false;
                self.title();
                self.redraw();
            }
            Err(error) => Self::error(format!("Cannot save {}: {error}", path.display())),
        }
    }

    fn key(&mut self, key: &Key) {
        let ctrl = self.modifiers.control_key() || self.modifiers.super_key();
        if ctrl {
            if let Key::Character(character) = key {
                match character.to_lowercase().as_str() {
                    "n" => self.new_scene(),
                    "o" => self.open(),
                    "s" => self.save(self.modifiers.shift_key()),
                    "c" => {
                        if let Selection::Entity(index) = self.selected
                            && let Some(entity) = self.scene.entities.get(index)
                        {
                            let result = serde_json::to_string(entity)
                                .map_err(|e| e.to_string())
                                .and_then(|text| {
                                    arboard::Clipboard::new()
                                        .and_then(|mut c| c.set_text(text))
                                        .map_err(|e| e.to_string())
                                });
                            if let Err(error) = result {
                                Self::error(format!("Cannot copy entity: {error}"));
                            }
                        }
                    }
                    "v" => {
                        let result = arboard::Clipboard::new()
                            .and_then(|mut c| c.get_text())
                            .map_err(|e| e.to_string())
                            .and_then(|text| {
                                serde_json::from_str::<Entity>(&text).map_err(|e| e.to_string())
                            });
                        match result {
                            Ok(mut entity) => {
                                entity.name.push_str(" copy");
                                entity.transform.translation[0] += 1.0;
                                self.scene.entities.push(entity);
                                self.selected = Selection::Entity(self.scene.entities.len() - 1);
                                self.changed();
                            }
                            Err(error) => Self::error(format!("Cannot paste entity: {error}")),
                        }
                    }
                    _ => {}
                }
            }
            return;
        }
        match key {
            Key::Named(NamedKey::Tab) => {
                let count = self.scene.entities.len() + self.scene.lights.len() + 2;
                let current = match self.selected {
                    Selection::Entity(i) => i,
                    Selection::Light(i) => self.scene.entities.len() + i,
                    Selection::Sun => count - 2,
                    Selection::Lighting | Selection::None => count - 1,
                };
                let next = (current + 1) % count;
                self.selected = if next < self.scene.entities.len() {
                    Selection::Entity(next)
                } else if next < count - 2 {
                    Selection::Light(next - self.scene.entities.len())
                } else if next == count - 2 {
                    Selection::Sun
                } else {
                    Selection::Lighting
                };
                self.tools.gizmo.cancel_drag();
                self.redraw();
            }
            Key::Named(NamedKey::Insert) => {
                self.scene.entities.push(Entity {
                    name: "Cube".into(),
                    transform: Transform {
                        translation: [0.0, 0.5, 0.0],
                        ..Transform::default()
                    },
                    material: Material {
                        base_color: [0.7, 0.5, 0.3],
                        metallic: 0.0,
                        roughness: 0.5,
                    },
                });
                self.selected = Selection::Entity(self.scene.entities.len() - 1);
                self.changed();
            }
            Key::Named(NamedKey::Delete) => {
                let removed = match self.selected {
                    Selection::Entity(i) if i < self.scene.entities.len() => {
                        self.scene.entities.remove(i);
                        true
                    }
                    Selection::Light(i) if i < self.scene.lights.len() => {
                        self.scene.lights.remove(i);
                        true
                    }
                    _ => false,
                };
                if removed {
                    self.selected = Selection::None;
                    self.changed();
                }
            }
            Key::Character(character) if character.eq_ignore_ascii_case("l") => {
                if self.scene.lights.len() >= self.tools.light_limit {
                    Self::error(format!(
                        "The renderer supports at most {} point lights",
                        hyperon_render::MAX_LIGHTS
                    ));
                } else {
                    self.scene.lights.push(PointLight {
                        position: [0.0, 3.0, 0.0],
                        color: [1.0, 1.0, 1.0],
                        intensity: 30.0,
                        radius: 8.0,
                    });
                    self.selected = Selection::Light(self.scene.lights.len() - 1);
                    self.changed();
                }
            }
            Key::Named(name) => {
                let delta = match name {
                    NamedKey::ArrowLeft => Some((0, -0.25)),
                    NamedKey::ArrowRight => Some((0, 0.25)),
                    NamedKey::ArrowUp => Some((2, -0.25)),
                    NamedKey::ArrowDown => Some((2, 0.25)),
                    NamedKey::PageUp => Some((1, 0.25)),
                    NamedKey::PageDown => Some((1, -0.25)),
                    _ => None,
                };
                if let Some((axis, amount)) = delta {
                    let position = match self.selected {
                        Selection::Entity(i) => self
                            .scene
                            .entities
                            .get_mut(i)
                            .map(|e| &mut e.transform.translation),
                        Selection::Light(i) => {
                            self.scene.lights.get_mut(i).map(|l| &mut l.position)
                        }
                        Selection::Sun | Selection::Lighting | Selection::None => None,
                    };
                    if let Some(position) = position {
                        position[axis] += amount;
                        self.changed();
                    }
                }
            }
            _ => {}
        }
    }

    fn frame(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let Some(window) = self.window.clone() else {
            return Ok(());
        };
        let Some(state) = &mut self.egui_state else {
            return Ok(());
        };
        let input = state.take_egui_input(&window);
        let context = self.context.clone();
        let mut response = None;
        let output = context.run_ui(input, |root| {
            response = Some(ui::show(
                root,
                &mut self.scene,
                &mut self.selected,
                self.path.as_deref(),
                self.dirty,
                &mut self.tools,
            ));
        });
        let Some(response) = response else {
            return Ok(());
        };
        state.handle_platform_output(&window, output.platform_output);
        let scale = output.pixels_per_point;
        let size = window.inner_size();
        let rect = response.viewport;
        let x = (rect.min.x * scale).floor().max(0.0).min(size.width as f32) as u32;
        let y = (rect.min.y * scale)
            .floor()
            .max(0.0)
            .min(size.height as f32) as u32;
        let right = (rect.max.x * scale)
            .ceil()
            .max(x as f32)
            .min(size.width as f32) as u32;
        let bottom = (rect.max.y * scale)
            .ceil()
            .max(y as f32)
            .min(size.height as f32) as u32;
        let frame = UiFrame {
            primitives: context.tessellate(output.shapes, scale),
            textures_delta: output.textures_delta,
            pixels_per_point: scale,
            viewport: [x, y, right - x, bottom - y],
        };
        if let Some(renderer) = &mut self.renderer {
            renderer.draw_ui(&self.scene, frame)?;
        }
        if response.changed {
            self.changed();
        }
        match response.action {
            Some(Action::New) => self.new_scene(),
            Some(Action::Open) => self.open(),
            Some(Action::Save) => self.save(false),
            Some(Action::SaveAs) => self.save(true),
            None => {}
        }
        if let Some(index) = response.asset_remove {
            self.remove_model(index);
        }
        match response.asset_action {
            Some(asset_browser::AssetAction::OpenScene(path)) if self.can_discard() => {
                self.open_path(path)
            }
            Some(asset_browser::AssetAction::ImportModel(path)) => self.import_model(path),
            _ => {}
        }
        Ok(())
    }
}

impl ApplicationHandler<Instant> for Editor {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let result = event_loop.create_window(
            Window::default_attributes()
                .with_title("Hyperon")
                .with_theme(Some(winit::window::Theme::Dark))
                .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0))
                .with_min_inner_size(winit::dpi::LogicalSize::new(800.0, 520.0))
                .with_maximized(true),
        );
        let window = match result {
            Ok(window) => Arc::new(window),
            Err(error) => {
                Self::error(error);
                event_loop.exit();
                return;
            }
        };
        match Renderer::new(window.clone()) {
            Ok(renderer) => {
                self.egui_state = Some(egui_winit::State::new(
                    self.context.clone(),
                    egui::ViewportId::ROOT,
                    &*window,
                    Some(window.scale_factor() as f32),
                    window.theme(),
                    None,
                ));
                self.renderer = Some(renderer);
                self.window = Some(window);
                self.title();
                self.redraw();
            }
            Err(error) => {
                Self::error(format!("Unable to initialize Vulkan: {error}"));
                event_loop.exit();
            }
        }
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, when: Instant) {
        if when <= Instant::now() {
            self.redraw();
        } else if self.next_repaint.is_none_or(|current| when < current) {
            self.next_repaint = Some(when);
            event_loop.set_control_flow(ControlFlow::WaitUntil(when));
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.next_repaint.is_some_and(|when| when <= Instant::now()) {
            self.next_repaint = None;
            event_loop.set_control_flow(ControlFlow::Wait);
            self.redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let Some(window) = self.window.clone() else {
            return;
        };
        if window.id() != id {
            return;
        }
        let consumed = if let Some(state) = &mut self.egui_state {
            let response = state.on_window_event(&window, &event);
            if response.repaint {
                window.request_redraw();
            }
            response.consumed
        } else {
            false
        };
        match event {
            WindowEvent::CloseRequested => {
                if self.can_discard() {
                    event_loop.exit();
                }
            }
            WindowEvent::ModifiersChanged(modifiers) => self.modifiers = modifiers.state(),
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                let movement = matches!(
                    &event.logical_key,
                    Key::Named(
                        NamedKey::ArrowLeft
                            | NamedKey::ArrowRight
                            | NamedKey::ArrowUp
                            | NamedKey::ArrowDown
                            | NamedKey::PageUp
                            | NamedKey::PageDown
                    )
                );
                let shortcut = (self.modifiers.control_key() || self.modifiers.super_key())
                    && matches!(&event.logical_key, Key::Character(c) if
                        ["n", "o", "s"].iter().any(|s| c.eq_ignore_ascii_case(s))
                            || (!self.context.text_edit_focused() && ["c", "v"].iter().any(|s| c.eq_ignore_ascii_case(s))));
                if (!event.repeat || movement)
                    && (shortcut || (!consumed && !self.context.egui_wants_keyboard_input()))
                {
                    let previous_selection = self.selected;
                    self.key(&event.logical_key);
                    if self.selected != previous_selection {
                        self.tools.gizmo.cancel_drag();
                    }
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size.width, size.height);
                }
                if size.width > 0 && size.height > 0 {
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                if let Err(error) = self.frame() {
                    Self::error(format!("Render failed: {error}"));
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }
}
