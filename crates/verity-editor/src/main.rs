mod asset_browser;
mod gizmo;
mod profiler;
mod ui;

use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use ui::{Action, PlayState, Selection, WindowCommand};
use verity_core::EntityHandle;
use verity_core::{Entity, LightingMethod, Material, PointLight, Scene, Transform};
use verity_lux::{
    gi::{GiStatus, Lux},
    settings::{Sampling, Settings},
};
use verity_render::{Renderer, UiFrame};
use verity_runtime::Runtime;
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
    /// The custom title bar's close button, handled where the event loop is.
    close_requested: bool,
    play: Option<Play>,
    history: History,
    loading: Option<Loading>,
}

/// Models loading on a worker thread that holds the renderer until it is done.
struct Loading {
    /// The renderer, the imported light count when the GPU holds a known
    /// scene, and the outcome.
    thread: std::thread::JoinHandle<(Renderer, Option<usize>, Result<(), String>)>,
    done: Box<dyn FnOnce(&mut Editor, Result<(), String>)>,
}

/// Undo steps beyond this are forgotten.
const HISTORY_LIMIT: usize = 200;

/// Whole-scene snapshots: scenes hold paths and parameters, not meshes, so
/// they are small. `committed` is the state the last step ended in.
#[derive(Default)]
struct History {
    undo: Vec<Scene>,
    redo: Vec<Scene>,
    committed: Scene,
}

/// A simulation of the scene. The editor's scene shows its transforms; Stop
/// restores `snapshot`.
struct Play {
    runtime: Runtime,
    /// World handles in the scene's entity order.
    handles: Vec<EntityHandle>,
    snapshot: Scene,
    dirty: bool,
    paused: bool,
    last: Instant,
}

/// The simulation's soft bodies as meshes, in their entities' materials.
fn soft_meshes(play: Option<&Play>) -> Vec<verity_render::DeformingMesh> {
    let Some(play) = play else {
        return Vec::new();
    };
    (play.runtime.soft_meshes())
        .filter_map(|(entity, positions, triangles)| {
            let material = play.runtime.world.get(entity)?.material.clone();
            let indices = triangles.iter().flatten().copied().collect();
            Some(verity_render::DeformingMesh::from_triangles(
                positions, indices, material,
            ))
        })
        .collect()
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
            close_requested: false,
            play: None,
            history: History::default(),
            loading: None,
        }
    }

    fn error(message: impl std::fmt::Display) {
        eprintln!("{message}");
        rfd::MessageDialog::new()
            .set_title("Verity error")
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
                "Verity — {file}{}{}",
                if self.dirty { "*" } else { "" },
                if self.loading.is_some() {
                    " — loading models…"
                } else {
                    ""
                }
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
        method: LightingMethod,
    ) -> Result<usize, String> {
        let paths = assets
            .iter()
            .map(|path| Self::asset_path(path))
            .collect::<Result<Vec<_>, _>>()?;
        // GI only learns about models loaded after it is installed, and the
        // renderer cannot clear models under an extension: reinstall it each time.
        renderer.remove_extension().map_err(|e| e.to_string())?;
        renderer.clear_gltf().map_err(|e| e.to_string())?;
        if method == LightingMethod::SplitRadianceCascades {
            let settings = Settings {
                sampling: Sampling {
                    // Lux's benchmarked budget and a ~300 MB world cache, as in sponza_gi.
                    max_cache_entries: 262_144,
                    max_updates_per_frame: 512,
                    max_traversal_queries: 16_384,
                    ..Sampling::default()
                },
                ..Settings::default()
            };
            renderer
                .set_extension(Box::new(Lux::new(settings)))
                .map_err(|e| e.to_string())?;
        }
        let mut count = scene_lights;
        for path in paths {
            let stats = renderer
                .load_gltf(&path, 512)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            count += stats.lights;
            if count > verity_render::MAX_LIGHTS {
                return Err(format!(
                    "Imported scenes and scene lights exceed the {}-light limit",
                    verity_render::MAX_LIGHTS
                ));
            }
        }
        // The editor's Sun toggle owns directional lighting, including when a
        // model carries a glTF directional light. Imported point lights remain.
        renderer.set_sun(None);
        Ok(count - scene_lights)
    }

    /// Load models on a worker thread, which holds the renderer meanwhile, so
    /// the window keeps responding: Sponza takes seconds, and Windows closes
    /// windows that stop responding. `done` runs on the event loop afterwards.
    /// On failure restore the previous scene's imported geometry.
    fn replace_assets(
        &mut self,
        assets: Vec<String>,
        scene_lights: usize,
        method: LightingMethod,
        done: impl FnOnce(&mut Self, Result<(), String>) + 'static,
    ) {
        let Some(mut renderer) = self.renderer.take() else {
            return done(self, Err("Renderer is not available".to_owned()));
        };
        let previous = self.scene.assets.clone();
        let (previous_lights, previous_method) =
            (self.scene.lights.len(), self.scene.lighting.method);
        let context = self.context.clone();
        let thread = std::thread::spawn(move || {
            let (imported, result) =
                match Self::load_assets(&mut renderer, &assets, scene_lights, method) {
                    Ok(imported) => (Some(imported), Ok(())),
                    Err(error) => match Self::load_assets(
                        &mut renderer,
                        &previous,
                        previous_lights,
                        previous_method,
                    ) {
                        Ok(imported) => (Some(imported), Err(error)),
                        Err(restore) => (
                            None,
                            Err(format!(
                                "{error}\nCould not restore the previous models: {restore}"
                            )),
                        ),
                    },
                };
            // Wakes the event loop to finish in `frame`.
            context.request_repaint();
            (renderer, imported, result)
        });
        self.loading = Some(Loading {
            thread,
            done: Box::new(done),
        });
        self.title();
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
        let (lights, method) = (self.scene.lights.len(), self.scene.lighting.method);
        self.replace_assets(
            assets.clone(),
            lights,
            method,
            |editor, result| match result {
                Ok(()) => {
                    editor.scene.assets = assets;
                    editor.changed();
                }
                Err(error) => Self::error(format!("Cannot import model: {error}")),
            },
        );
    }

    fn remove_model(&mut self, index: usize) {
        if index >= self.scene.assets.len() {
            return;
        }
        let mut assets = self.scene.assets.clone();
        assets.remove(index);
        let (lights, method) = (self.scene.lights.len(), self.scene.lighting.method);
        self.replace_assets(
            assets.clone(),
            lights,
            method,
            |editor, result| match result {
                Ok(()) => {
                    editor.scene.assets = assets;
                    editor.changed();
                }
                Err(error) => Self::error(format!("Cannot remove model: {error}")),
            },
        );
    }

    fn new_scene(&mut self) {
        if self.can_discard() {
            // Also runs without models, to drop a GI extension the old scene installed.
            self.replace_assets(
                Vec::new(),
                0,
                LightingMethod::default(),
                |editor, result| {
                    if let Err(error) = result {
                        Self::error(format!("Cannot create scene: {error}"));
                        return;
                    }
                    editor.tools.gizmo.cancel_drag();
                    editor.scene = Scene::default();
                    editor.reset_history();
                    editor.path = None;
                    editor.selected = Selection::Entity(0);
                    editor.dirty = false;
                    editor.title();
                    editor.redraw();
                },
            );
        }
    }

    fn open(&mut self) {
        if !self.can_discard() {
            return;
        }
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Verity scene", &["verity"])
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
            Ok(scene) if scene.lights.len() > verity_render::MAX_LIGHTS => {
                Self::error(format!(
                    "Cannot open {}: the renderer supports at most {} point lights",
                    path.display(),
                    verity_render::MAX_LIGHTS
                ));
            }
            Ok(scene) => {
                let (assets, lights, method) = (
                    scene.assets.clone(),
                    scene.lights.len(),
                    scene.lighting.method,
                );
                self.replace_assets(assets, lights, method, |editor, result| {
                    if let Err(error) = result {
                        Self::error(format!("Cannot open {}: {error}", path.display()));
                        return;
                    }
                    editor.tools.gizmo.cancel_drag();
                    editor.selected = if scene.entities.is_empty() {
                        Selection::None
                    } else {
                        Selection::Entity(0)
                    };
                    if let Some(camera) = scene.cameras.first() {
                        ui::look_through(&mut editor.tools, camera);
                    }
                    editor.scene = scene;
                    editor.reset_history();
                    editor.path = Some(path);
                    editor.dirty = false;
                    editor.title();
                    editor.redraw();
                });
            }
            Err(error) => Self::error(format!("Cannot open {}: {error}", path.display())),
        }
    }

    fn save(&mut self, save_as: bool) {
        let path = if save_as || self.path.is_none() {
            rfd::FileDialog::new()
                .add_filter("Verity scene", &["verity"])
                .set_file_name("scene.verity")
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

    /// Record the scene's edits since the last step as one undo step.
    fn commit(&mut self) {
        if self.play.is_some() || self.scene == self.history.committed {
            return;
        }
        let previous = std::mem::replace(&mut self.history.committed, self.scene.clone());
        self.history.undo.push(previous);
        if self.history.undo.len() > HISTORY_LIMIT {
            self.history.undo.remove(0);
        }
        self.history.redo.clear();
    }

    /// Start a fresh history at the current scene, after opening or creating one.
    fn reset_history(&mut self) {
        self.history = History {
            committed: self.scene.clone(),
            ..History::default()
        };
    }

    /// Undo, or redo when `forward`.
    fn travel(&mut self, forward: bool) {
        if self.play.is_some() {
            return;
        }
        self.commit();
        let stack = if forward {
            &mut self.history.redo
        } else {
            &mut self.history.undo
        };
        let Some(target) = stack.pop() else { return };
        // Models and GI live on the GPU: reload them when the step changes them.
        if target.assets != self.scene.assets
            || target.lighting.method != self.scene.lighting.method
        {
            let (assets, lights, method) = (
                target.assets.clone(),
                target.lights.len(),
                target.lighting.method,
            );
            self.replace_assets(assets, lights, method, move |editor, result| match result {
                Ok(()) => editor.arrive(target, forward),
                Err(error) => {
                    let stack = if forward {
                        &mut editor.history.redo
                    } else {
                        &mut editor.history.undo
                    };
                    stack.push(target);
                    Self::error(format!(
                        "Cannot {}: {error}",
                        if forward { "redo" } else { "undo" }
                    ));
                }
            });
        } else {
            self.arrive(target, forward);
        }
    }

    /// Make an undo or redo step's scene current.
    fn arrive(&mut self, target: Scene, forward: bool) {
        let previous = std::mem::replace(&mut self.history.committed, target.clone());
        if forward {
            self.history.undo.push(previous);
        } else {
            self.history.redo.push(previous);
        }
        self.scene = target;
        self.tools.gizmo.cancel_drag();
        self.changed();
    }

    fn play(&mut self) {
        if let Some(play) = &mut self.play {
            play.paused = false;
            play.last = Instant::now();
            return;
        }
        match Runtime::from_scene(&self.scene) {
            Ok(runtime) => {
                self.tools.gizmo.cancel_drag();
                self.play = Some(Play {
                    handles: runtime.world.handles().to_vec(),
                    runtime,
                    snapshot: self.scene.clone(),
                    dirty: self.dirty,
                    paused: false,
                    last: Instant::now(),
                });
                self.redraw();
            }
            Err(error) => Self::error(format!("Cannot play the scene: {error}")),
        }
    }

    fn stop(&mut self) {
        if let Some(play) = self.play.take() {
            self.tools.joint_status.clear();
            self.scene = play.snapshot;
            self.dirty = play.dirty;
            self.title();
            self.redraw();
        }
    }

    /// Advance the simulation and show its transforms in the scene.
    fn tick(&mut self, step: bool) {
        let Some(play) = &mut self.play else { return };
        let now = Instant::now();
        if step {
            play.runtime.advance(play.runtime.fixed_timestep());
        } else if !play.paused {
            // shortcut: drops time past 100 ms (hitches, breakpoints) instead of
            // catching up; use a real clock policy once games need wall time.
            play.runtime
                .advance((now - play.last).min(Duration::from_millis(100)));
        }
        play.last = now;
        for (entity, &handle) in self.scene.entities.iter_mut().zip(&play.handles) {
            if let Some(simulated) = play.runtime.world.get(handle) {
                entity.transform = simulated.transform.clone();
            }
        }
        // A soft body draws as its mesh, in place of the entity's cube.
        for (entity, &handle) in self.scene.entities.iter_mut().zip(&play.handles) {
            if play.runtime.soft_meshes().any(|(soft, ..)| soft == handle) {
                entity.transform.scale = [0.0; 3];
            }
        }
        self.tools.joint_status.clear();
        for (index, &handle) in play.handles.iter().enumerate() {
            if let Some(status) = play.runtime.joint_status(handle) {
                let text = format!(
                    "{:?} · {:.0} N · {:.0} N·m",
                    status.stage, status.force, status.torque
                );
                self.tools.joint_status.insert(index, (text, status.share));
                // Load shows in the viewport too: redder nearer breaking.
                if let (Some(entity), Some(authored)) = (
                    self.scene.entities.get_mut(index),
                    play.snapshot.entities.get(index),
                ) {
                    let share = status.share.clamp(0.0, 1.0);
                    let base = authored.material.base_color;
                    entity.material.base_color = [
                        base[0] + (0.95 - base[0]) * share,
                        base[1] * (1.0 - share),
                        base[2] * (1.0 - share),
                    ];
                }
            }
        }
    }

    fn key(&mut self, key: &Key) {
        let ctrl = self.modifiers.control_key() || self.modifiers.super_key();
        let character = |c: &str| matches!(key, Key::Character(k) if k.eq_ignore_ascii_case(c));
        if ctrl && character("p") {
            if self.play.as_ref().is_some_and(|p| !p.paused) {
                self.play.as_mut().unwrap().paused = true;
            } else {
                self.play();
            }
            return;
        }
        if self.play.is_some() {
            // File shortcuts act on the authoring scene; other edits wait for Stop.
            if ctrl && ["n", "o", "s"].iter().any(|c| character(c)) {
                self.stop();
            } else if !matches!(key, Key::Named(NamedKey::Tab)) {
                return;
            }
        }
        if ctrl {
            if let Key::Character(character) = key {
                match character.to_lowercase().as_str() {
                    "n" => self.new_scene(),
                    "o" => self.open(),
                    "s" => self.save(self.modifiers.shift_key()),
                    "z" if self.modifiers.shift_key() => self.travel(true),
                    "z" => self.travel(false),
                    "y" => self.travel(true),
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
                let entities = self.scene.entities.len();
                let lights = entities + self.scene.lights.len();
                let count = lights + self.scene.cameras.len() + 2;
                let current = match self.selected {
                    Selection::Entity(i) => i,
                    Selection::Light(i) => entities + i,
                    Selection::Camera(i) => lights + i,
                    Selection::Sun => count - 2,
                    Selection::Lighting | Selection::None => count - 1,
                };
                let next = (current + 1) % count;
                self.selected = if next < entities {
                    Selection::Entity(next)
                } else if next < lights {
                    Selection::Light(next - entities)
                } else if next < count - 2 {
                    Selection::Camera(next - lights)
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
                        translation: [0.0, 0.0, 0.5],
                        ..Transform::default()
                    },
                    material: Material {
                        base_color: [0.7, 0.5, 0.3],
                        metallic: 0.0,
                        roughness: 0.5,
                    },
                    ..Entity::default()
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
                    Selection::Camera(i) if i < self.scene.cameras.len() => {
                        self.scene.cameras.remove(i);
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
                        verity_render::MAX_LIGHTS
                    ));
                } else {
                    self.scene.lights.push(PointLight {
                        position: [0.0, 0.0, 3.0],
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
                    // Z is up: arrows move in the ground plane, pages vertically.
                    NamedKey::ArrowUp => Some((1, 0.25)),
                    NamedKey::ArrowDown => Some((1, -0.25)),
                    NamedKey::PageUp => Some((2, 0.25)),
                    NamedKey::PageDown => Some((2, -0.25)),
                    _ => None,
                };
                if let Some((axis, amount)) = delta
                    && let Selection::Camera(i) = self.selected
                {
                    // Move the target too, so the camera keeps its heading.
                    if let Some(camera) = self.scene.cameras.get_mut(i) {
                        camera.position[axis] += amount;
                        camera.target[axis] += amount;
                        self.changed();
                    }
                } else if let Some((axis, amount)) = delta {
                    let position = match self.selected {
                        Selection::Entity(i) => self
                            .scene
                            .entities
                            .get_mut(i)
                            .map(|e| &mut e.transform.translation),
                        Selection::Light(i) => {
                            self.scene.lights.get_mut(i).map(|l| &mut l.position)
                        }
                        Selection::Camera(_)
                        | Selection::Sun
                        | Selection::Lighting
                        | Selection::None => None,
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
        if let Some(loading) = self.loading.take_if(|loading| loading.thread.is_finished()) {
            let (mut renderer, imported, result) = loading
                .thread
                .join()
                .map_err(|_| "Loading models panicked")?;
            if let Some(imported) = imported {
                self.tools.light_limit = verity_render::MAX_LIGHTS - imported;
            }
            // Resizes during the load went unseen.
            let size = window.inner_size();
            renderer.resize(size.width, size.height);
            self.renderer = Some(renderer);
            (loading.done)(self, result);
            self.title();
        }
        // The UI stays frozen until the models arrive, so no edit races the load.
        if self.loading.is_some() {
            return Ok(());
        }
        self.tick(false);
        self.tools.can_undo = !self.history.undo.is_empty() || self.scene != self.history.committed;
        self.tools.can_redo = !self.history.redo.is_empty();
        self.tools.play = match &self.play {
            None => PlayState::Editing,
            Some(play) if play.paused => PlayState::Paused,
            Some(_) => PlayState::Playing,
        };
        let Some(state) = &mut self.egui_state else {
            return Ok(());
        };
        self.tools.maximized = window.is_maximized();
        let input = state.take_egui_input(&window);
        let context = self.context.clone();
        let mut response = None;
        let method = self.scene.lighting.method;
        let ui_start = Instant::now();
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
        let ui_time = ui_start.elapsed();
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
            let camera = &self.tools.camera;
            renderer.set_camera(camera.eye, camera.target, camera.vertical_fov);
            renderer.set_strata_enabled(self.tools.strata_enabled);
            renderer.set_strata_debug(self.tools.strata_enabled && self.tools.strata_debug);
            renderer.set_deforming_meshes(soft_meshes(self.play.as_ref()));
            let render_start = Instant::now();
            renderer.draw_ui(&self.scene, frame)?;
            self.tools.profiler.record(ui_time, render_start.elapsed());
            self.tools.gi_status = renderer
                .extension_mut::<Lux>()
                .map(|lux| match lux.status() {
                    GiStatus::Compiling { since } => format!(
                        "GI compiling pipelines… {}s (hemisphere until ready)",
                        since.elapsed().as_secs()
                    ),
                    GiStatus::Ready { compile_time } => {
                        format!("GI active (compiled in {:.1}s)", compile_time.as_secs_f32())
                    }
                    GiStatus::Failed(error) => format!("GI failed: {error}"),
                });
            // GI polls its compile and converges over frames, and play mode
            // simulates, so keep drawing; an open profiler measures continuous frames.
            if self.tools.gi_status.is_some()
                || self.tools.play == PlayState::Playing
                || self.tools.profiler.open
            {
                window.request_redraw();
            }
        }
        match response.window {
            Some(WindowCommand::Minimize) => window.set_minimized(true),
            Some(WindowCommand::ToggleMaximize) => window.set_maximized(!window.is_maximized()),
            Some(WindowCommand::Close) => self.close_requested = true,
            // Both fail only when the pointer is no longer pressed.
            Some(WindowCommand::Drag) => {
                let _ = window.drag_window();
            }
            Some(WindowCommand::Resize(direction)) => {
                let _ = window.drag_resize_window(direction);
            }
            None => {}
        }
        if response.changed {
            self.changed();
        }
        if self.scene.lighting.method != method {
            // replace_assets restores from the scene on failure, so it must hold the old method.
            let new = std::mem::replace(&mut self.scene.lighting.method, method);
            let (assets, lights) = (self.scene.assets.clone(), self.scene.lights.len());
            self.replace_assets(assets, lights, new, move |editor, result| match result {
                Ok(()) => editor.scene.lighting.method = new,
                Err(error) => Self::error(format!("Cannot change the lighting method: {error}")),
            });
        }
        // File and model actions apply to the authoring scene.
        if matches!(
            response.action,
            Some(Action::New | Action::Open | Action::Save | Action::SaveAs)
        ) || response.asset_remove.is_some()
            || response.asset_action.is_some()
        {
            self.stop();
        }
        match response.action {
            Some(Action::Play) => self.play(),
            Some(Action::Pause) => {
                if let Some(play) = &mut self.play {
                    play.paused = true;
                }
            }
            Some(Action::Step) => {
                self.tick(true);
                self.redraw();
            }
            Some(Action::Stop) => self.stop(),
            Some(Action::Undo) => self.travel(false),
            Some(Action::Redo) => self.travel(true),
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
        // A drag or a text edit becomes one step once it ends.
        if !self.context.input(|i| i.pointer.any_down()) && !self.context.text_edit_focused() {
            self.commit();
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
                .with_title("Verity")
                .with_theme(Some(winit::window::Theme::Dark))
                // The editor draws its own title bar and window controls.
                .with_decorations(false)
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
                if let Some(path) = std::env::args_os().nth(1) {
                    self.open_path(path.into());
                }
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
                        ["n", "o", "s", "p"].iter().any(|s| c.eq_ignore_ascii_case(s))
                            || (!self.context.text_edit_focused() && ["c", "v", "z", "y"].iter().any(|s| c.eq_ignore_ascii_case(s))));
                if !self.tools.flying
                    && self.loading.is_none()
                    && (!event.repeat || movement)
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
                if std::mem::take(&mut self.close_requested) && self.can_discard() {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn playground_runs_verity_physics_and_stop_restores_the_scene() {
        let scene: Scene = serde_json::from_str(include_str!(
            "../../../assets/scenes/physics-playground.verity"
        ))
        .unwrap();
        assert_eq!(scene.physics.engine, verity_core::PhysicsEngine::Verity);
        assert_eq!(scene.entities.len(), 427);
        let falling = scene
            .entities
            .iter()
            .position(|entity| entity.name == "Falling cube 001")
            .unwrap();
        let start_z = scene.entities[falling].transform.translation[2];
        let mut editor = Editor::new(egui::Context::default());
        editor.scene = scene.clone();

        // Validate before invoking play, which reports errors in a dialog.
        Runtime::from_scene(&scene).unwrap();
        editor.play();
        assert!(
            editor
                .play
                .as_ref()
                .unwrap()
                .runtime
                .verity_physics
                .is_some()
        );
        editor.play.as_mut().unwrap().paused = true;
        for _ in 0..10 {
            editor.tick(true);
        }
        assert!(editor.scene.entities[falling].transform.translation[2] < start_z);
        editor.stop();
        assert_eq!(editor.scene, scene);
    }

    #[test]
    fn joints_scene_snaps_its_chain_stops_its_elbow_and_shows_loads() {
        let scene: Scene =
            serde_json::from_str(include_str!("../../../assets/scenes/joints.verity")).unwrap();
        let index = |name: &str| scene.entities.iter().position(|e| e.name == name).unwrap();
        let mut editor = Editor::new(egui::Context::default());
        editor.scene = scene.clone();
        Runtime::from_scene(&scene).unwrap();
        editor.play();
        editor.play.as_mut().unwrap().paused = true;
        let (mut yielded, mut broke) = (0, 0);
        for _ in 0..600 {
            editor.tick(true);
            for event in &editor.play.as_ref().unwrap().runtime.joint_events {
                let stage = format!("{:?}", event.stage);
                yielded += usize::from(stage == "Yielded");
                broke += usize::from(stage == "Broken");
            }
        }
        assert!(broke >= 1, "the chain snapped ({yielded} yielded)");
        let z = |editor: &Editor, name| editor.scene.entities[index(name)].transform.translation[2];
        assert!(z(&editor, "Chain load") < 3.0, "the load fell");
        assert!(z(&editor, "Rubber block") > 7.0, "the rubber held");
        // Stopped at 60 degrees: its center hangs 0.5 sin 60° below the elbow.
        assert!(
            (z(&editor, "Forearm") - 9.57).abs() < 0.05,
            "the elbow held"
        );
        let (text, _) = &editor.tools.joint_status[&index("Rubber block")];
        assert!(text.contains(" N"), "{text}");
        editor.stop();
        assert!(editor.tools.joint_status.is_empty());
        assert_eq!(editor.scene, scene);
    }

    #[test]
    fn soft_bodies_scene_runs_and_each_soft_body_shows_as_a_mesh() {
        let scene: Scene =
            serde_json::from_str(include_str!("../../../assets/scenes/soft-bodies.verity"))
                .unwrap();
        let index = |name: &str| scene.entities.iter().position(|e| e.name == name).unwrap();
        let mut editor = Editor::new(egui::Context::default());
        editor.scene = scene.clone();
        Runtime::from_scene(&scene).unwrap();
        editor.play();
        editor.play.as_mut().unwrap().paused = true;
        let start = soft_meshes(editor.play.as_ref());
        for _ in 0..240 {
            editor.tick(true);
        }
        let meshes = soft_meshes(editor.play.as_ref());
        assert_eq!(meshes.len(), 5);
        for mesh in &meshes {
            assert!(mesh.vertices.iter().all(|v| v.position.iter().all(|p| p.is_finite())));
        }
        // The heavy ball tore the trampoline, which duplicates vertices.
        let torn = meshes.iter().zip(&start).any(|(m, s)| m.vertices.len() > s.vertices.len());
        assert!(torn, "something tore");
        let hidden = editor.scene.entities[index("Curtain")].transform.scale;
        assert_eq!(hidden, [0.0; 3], "the entity's cube makes way for its mesh");
        editor.stop();
        assert_eq!(editor.scene, scene);
        assert!(soft_meshes(editor.play.as_ref()).is_empty());
    }

    #[test]
    fn play_simulates_locks_edits_and_stop_restores() {
        let mut editor = Editor::new(egui::Context::default());
        editor.scene.entities[0].transform.translation = [0.0, 0.0, 5.0];
        editor.scene.entities[0].components.insert(
            "Rigid Body".into(),
            serde_json::to_value(verity_box3d::RigidBodyComponent::default()).unwrap(),
        );
        editor.dirty = true;
        let authored = editor.scene.clone();

        editor.play();
        editor.play.as_mut().unwrap().paused = true;
        for _ in 0..10 {
            editor.tick(true);
        }
        let z = editor.scene.entities[0].transform.translation[2];
        assert!(z < 5.0, "{z}");
        assert_eq!(editor.scene.entities[1], authored.entities[1]);

        editor.key(&Key::Named(NamedKey::Delete));
        assert_eq!(editor.scene.entities.len(), authored.entities.len());

        editor.stop();
        assert!(editor.play.is_none());
        assert_eq!(editor.scene, authored);
        assert!(editor.dirty);
    }
    #[test]
    fn undo_and_redo_walk_committed_steps() {
        let mut editor = Editor::new(egui::Context::default());
        let original = editor.scene.clone();
        editor.scene.entities[0].transform.translation[2] = 2.0;
        editor.commit();
        let moved = editor.scene.clone();
        editor.key(&Key::Character("l".into()));
        assert_eq!(editor.scene.lights.len(), 2);
        // Uncommitted edits are captured before undoing.
        editor.travel(false);
        assert_eq!(editor.scene, moved);
        editor.travel(false);
        assert_eq!(editor.scene, original);
        editor.travel(false);
        assert_eq!(editor.scene, original, "nothing left to undo");
        editor.travel(true);
        assert_eq!(editor.scene, moved);

        // A new edit forgets the redo branch.
        editor.scene.entities[1].name = "Renamed".into();
        editor.commit();
        editor.travel(true);
        assert_eq!(editor.scene.entities[1].name, "Renamed");
        assert_eq!(editor.scene.lights.len(), 1);

        // Play mode owns the scene: undo waits for Stop.
        editor.play();
        editor.travel(false);
        assert_eq!(editor.scene.entities[1].name, "Renamed");
        editor.stop();
        editor.travel(false);
        assert_eq!(editor.scene, moved);
    }
}
