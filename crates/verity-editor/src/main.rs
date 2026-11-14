mod asset_browser;
mod city;
mod console;
mod dock;
mod gizmo;
mod profiler;
mod terrain;
mod ui;

use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use ui::{Action, PlayState, Selection, WindowCommand};
use verity_core::EntityHandle;
use verity_core::{Copies, Entity, LightingMethod, Material, PointLight, Scene, Transform, World};
use verity_lux::Lux;
use verity_physics::PhysicsWorld;
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
    let mut editor = Editor::new(context);
    editor.tools.dock = dock::load_layout();
    event_loop.run_app(&mut editor)?;
    dock::save_layout(&editor.tools.dock);
    Ok(())
}

struct Editor {
    /// The lighting method the renderer has installed, which the scene's lags
    /// behind while models reload.
    installed_method: LightingMethod,
    /// The scene as PXTGI sees it outside play mode, when nothing simulates it.
    trace_world: World,
    /// Stands in for a physics world outside play mode.
    no_physics: PhysicsWorld,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    context: egui::Context,
    egui_state: Option<egui_winit::State>,
    next_repaint: Option<Instant>,
    /// The last frame wants another straight away: GI, play, profiler or uncapped.
    continuous: bool,
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
    terrain: terrain::TerrainView,
    city: city::CityView,
}

/// Models loading on a worker thread that holds the renderer until it is done.
struct Loading {
    /// The renderer, the imported light count when the GPU holds a known
    /// scene, and the outcome.
    thread: std::thread::JoinHandle<(Renderer, Option<usize>, Result<(), String>)>,
    done: Box<dyn FnOnce(&mut Editor, Result<(), String>)>,
    shared: Arc<Mutex<Shared>>,
}

/// What the event loop and a loading thread exchange: the thread reports its
/// progress, and the event loop leaves its latest UI frame and window size for
/// the thread to draw between steps, since the thread holds the renderer.
#[derive(Default)]
struct Shared {
    progress: ui::LoadProgress,
    frame: Option<(UiFrame, [u32; 2])>,
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

/// Keep `mirror` holding `scene`'s entities, in its order, keeping their
/// handles while the count stands so PXTGI sees moves and not new entities.
fn mirror_scene(scene: &Scene, mirror: &mut World) {
    if mirror.len() != scene.entities.len() {
        *mirror = World::new();
        for entity in &scene.entities {
            mirror.spawn(Entity {
                components: Default::default(),
                ..entity.clone()
            });
        }
    }
    for (found, entity) in mirror.iter_mut().zip(&scene.entities) {
        *found.transform = entity.transform.clone();
        *found.material = entity.material.clone();
    }
}

/// A static body shaped like `ground`, which nothing draws.
fn add_ground(runtime: &mut Runtime, ground: Arc<dyn verity_physics::Ground>) {
    let entity = runtime.world.spawn(Entity {
        name: "Terrain".into(),
        transform: Transform {
            scale: [0.0; 3],
            ..Transform::default()
        },
        ..Entity::default()
    });
    let body = verity_physics::PhysicsBody::new(
        verity_physics::RigidBody::Static,
        verity_physics::Collider::Ground,
    );
    (runtime.physics)
        .insert(&runtime.world, entity, body)
        .expect("the entity was just spawned");
    runtime
        .physics
        .set_ground(ground, verity_physics::Bounds::EMPTY);
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
            continuous: false,
            installed_method: LightingMethod::default(),
            trace_world: World::new(),
            no_physics: PhysicsWorld::default(),
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
            terrain: Default::default(),
            city: Default::default(),
        }
    }

    fn error(message: impl std::fmt::Display) {
        eprintln!("{message}");
        console::error(message.to_string());
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
        copies: &[Copies],
        scene_lights: usize,
        method: LightingMethod,
        step: &mut dyn FnMut(&mut Renderer, ui::LoadProgress),
    ) -> Result<usize, String> {
        let paths = assets
            .iter()
            .map(|path| Self::asset_path(path))
            .collect::<Result<Vec<_>, _>>()?;
        let total = 2 + 2 * paths.len();
        let mut report = |renderer: &mut Renderer, done: usize, label: String| {
            step(renderer, ui::LoadProgress { done, total, label })
        };
        report(renderer, 0, "Preparing the renderer".into());
        // GI only learns about models loaded after it is installed, and the
        // renderer cannot clear models under an extension: reinstall it each time.
        renderer.remove_extension().map_err(|e| e.to_string())?;
        renderer.clear_gltf().map_err(|e| e.to_string())?;
        if let Some(lux) = Lux::for_lighting(method) {
            renderer
                .set_extension(Box::new(lux))
                .map_err(|e| e.to_string())?;
        }
        let mut count = scene_lights;
        let mut draws = std::collections::HashMap::new();
        let options = verity_render::asset::LoadOptions {
            format: renderer.texture_format(),
            max_texture_size: 512,
        };
        std::thread::scope(|scope| {
            // Read the next model while the renderer takes the current one.
            let (sender, parsed) = std::sync::mpsc::sync_channel(1);
            let paths = &paths;
            scope.spawn(move || {
                for path in paths {
                    let data = verity_render::asset::load_gltf(path, options)
                        .map_err(|e| format!("{}: {e}", path.display()));
                    // The loader stopped early, after an error.
                    if sender.send(data).is_err() {
                        break;
                    }
                }
            });
            for (index, (path, asset)) in paths.iter().zip(assets).enumerate() {
                let name = path
                    .file_name()
                    .map_or(asset.into(), |n| n.to_string_lossy());
                report(renderer, 1 + 2 * index, format!("Reading {name}"));
                let data = parsed
                    .recv()
                    .map_err(|_| format!("{}: reading stopped", path.display()))??;
                report(
                    renderer,
                    2 + 2 * index,
                    format!("Uploading {name} and building GI"),
                );
                let first = renderer.model_draw_count();
                let stats = renderer
                    .add_scene(data)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                draws.insert(asset.as_str(), first..renderer.model_draw_count());
                count += stats.lights;
                if count > verity_render::MAX_LIGHTS {
                    return Err(format!(
                        "Imported scenes and scene lights exceed the {}-light limit",
                        verity_render::MAX_LIGHTS
                    ));
                }
            }
            Ok(())
        })?;
        for copies in copies {
            let range = (draws.get(copies.asset.as_str())).ok_or_else(|| {
                format!(
                    "{}: copies of a model the scene does not load",
                    copies.asset
                )
            })?;
            let transforms: Vec<_> = (copies.transforms.iter())
                .filter_map(verity_render::entity_matrix)
                .collect();
            (renderer.place_copies(range.clone(), &transforms))
                .map_err(|e| format!("{}: {e}", copies.asset))?;
        }
        // The editor's Sun toggle owns directional lighting, including when a
        // model carries a glTF directional light. Imported point lights remain.
        renderer.set_sun(None);
        // The one progress frame that draws the scene, so GI builds here and
        // not in the editor's first frame.
        report(renderer, total - 1, "Building GI".into());
        Ok(count - scene_lights)
    }

    /// Load models on a worker thread, which holds the renderer meanwhile, so
    /// the window keeps responding: Sponza takes seconds, and Windows closes
    /// windows that stop responding. `done` runs on the event loop afterwards.
    /// On failure restore the previous scene's imported geometry.
    fn replace_assets(
        &mut self,
        assets: Vec<String>,
        copies: Vec<Copies>,
        scene_lights: usize,
        method: LightingMethod,
        done: impl FnOnce(&mut Self, Result<(), String>) + 'static,
    ) {
        let Some(mut renderer) = self.renderer.take() else {
            return done(self, Err("Renderer is not available".to_owned()));
        };
        let (previous, previous_copies) = (self.scene.assets.clone(), self.scene.copies.clone());
        let (previous_lights, previous_method) = (self.scene.lights.len(), self.installed_method);
        self.installed_method = method;
        let done = move |editor: &mut Self, result: Result<(), String>| {
            if result.is_err() {
                editor.installed_method = previous_method;
            }
            done(editor, result);
        };
        let context = self.context.clone();
        let shared = Arc::new(Mutex::new(Shared::default()));
        let (scene, worker_shared) = (self.scene.clone(), shared.clone());
        let mut size = self
            .window
            .as_ref()
            .map(|w| [w.inner_size().width, w.inner_size().height]);
        // Between steps, publish progress and draw the editor's newest frame.
        let mut step = move |renderer: &mut Renderer, progress: ui::LoadProgress| {
            worker_shared.lock().unwrap().progress = progress;
            context.request_repaint();
            // shortcut: waits a fixed moment for the event loop to draw the new
            // progress; a frame counter with a condvar would wait exactly.
            std::thread::sleep(Duration::from_millis(50));
            let pending = worker_shared.lock().unwrap().frame.take();
            if let Some((frame, current)) = pending {
                if size != Some(current) {
                    renderer.resize(current[0], current[1]);
                    size = Some(current);
                }
                // A failed progress frame only costs that frame.
                let _ = renderer.draw_ui(&scene, frame);
            }
        };
        let context = self.context.clone();
        let thread = std::thread::spawn(move || {
            let (imported, result) = match Self::load_assets(
                &mut renderer,
                &assets,
                &copies,
                scene_lights,
                method,
                &mut step,
            ) {
                Ok(imported) => (Some(imported), Ok(())),
                Err(error) => match Self::load_assets(
                    &mut renderer,
                    &previous,
                    &previous_copies,
                    previous_lights,
                    previous_method,
                    &mut step,
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
            shared,
        });
        self.title();
    }

    /// Follow the scene's lighting method. GI only learns about models loaded
    /// after it is installed, so changing it reloads them, on a worker thread.
    fn apply_lighting_method(&mut self) {
        let (method, installed) = (self.scene.lighting.method, self.installed_method);
        if method == installed || self.play.is_some() || self.renderer.is_none() {
            return;
        }
        let uses_gi = |method| verity_lux::method_info(method).is_some();
        if !uses_gi(method) && !uses_gi(installed) {
            self.installed_method = method;
            return;
        }
        let (assets, lights) = (self.scene.assets.clone(), self.scene.lights.len());
        let copies = self.scene.copies.clone();
        self.replace_assets(assets, copies, lights, method, move |editor, result| {
            if let Err(error) = result {
                editor.scene.lighting.method = installed;
                Self::error(format!("Cannot change the lighting method: {error}"));
            }
            editor.redraw();
        });
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
            self.scene.copies.clone(),
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
        let removed = assets.remove(index);
        let mut copies = self.scene.copies.clone();
        copies.retain(|copies| copies.asset != removed);
        let (lights, method) = (self.scene.lights.len(), self.scene.lighting.method);
        self.replace_assets(
            assets.clone(),
            copies.clone(),
            lights,
            method,
            |editor, result| match result {
                Ok(()) => {
                    editor.scene.assets = assets;
                    editor.scene.copies = copies;
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
                let copies = scene.copies.clone();
                self.replace_assets(assets, copies, lights, method, |editor, result| {
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
                    console::info(format!("Opened {}", path.display()));
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
                console::info(format!("Saved {}", path.display()));
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
            || target.copies != self.scene.copies
            || target.lighting.method != self.scene.lighting.method
        {
            let (assets, lights, method) = (
                target.assets.clone(),
                target.lights.len(),
                target.lighting.method,
            );
            let copies = target.copies.clone();
            self.replace_assets(
                assets,
                copies,
                lights,
                method,
                move |editor, result| match result {
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
                },
            );
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
            Ok(mut runtime) => {
                self.tools.gizmo.cancel_drag();
                // Taken before the ground joins the world: they pair with the scene's entities.
                let handles = runtime.world.handles().to_vec();
                if let Some(ground) = self.terrain.ground() {
                    add_ground(&mut runtime, ground);
                }
                // Crowds and the city stand on the landscape, not the flat ground they started on.
                if let Some(land) = self.terrain.landscape() {
                    let crowd_ground = land.clone();
                    let height: Arc<verity_runtime::agents::Height> =
                        Arc::new(move |p: glam::Vec2| Some(crowd_ground.height(p.x, p.y)));
                    runtime.agents.navigate(&runtime.world, Some(height));
                    if let Some(run) = runtime.city.take() {
                        let height = |x: f32, y: f32| Some(land.height(x, y));
                        runtime.city = Some(verity_runtime::CityRun::new(run.city, &height));
                    }
                }
                self.play = Some(Play {
                    handles,
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

    /// Physics phases of this frame's ticks, under the Simulation row.
    fn profile_physics(&mut self) {
        let Some(play) = &self.play else { return };
        let profiler = &mut self.tools.profiler;
        let p = play.runtime.last_profile;
        if play.runtime.last_ticks > 0 {
            for (name, time) in [
                ("Setup", p.setup),
                ("Broad phase", p.broad_phase),
                ("Narrow phase", p.narrow_phase),
                ("Solve", p.solve),
                ("Write-back", p.write_back),
                ("Soft bodies", p.soft),
                ("Agents & city", play.runtime.last_agents),
            ] {
                profiler.cpu(name, 2, time);
            }
        }
        if profiler.open {
            let city = play.runtime.city.as_ref();
            profiler.counter(
                "Agents",
                format!(
                    "{} crowd · {} vehicles · {} pedestrians",
                    play.runtime.agents.crowd.len(),
                    city.map_or(0, |c| c.sim.traffic.len()),
                    city.map_or(0, |c| c.sim.crowd.len()),
                ),
            );
        }
        if profiler.open {
            profiler.counter(
                "Physics",
                format!(
                    "{} ticks · {} bodies · {} contacts · {} joints · {} soft particles",
                    play.runtime.last_ticks, p.bodies, p.contacts, p.joints, p.soft_particles
                ),
            );
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

    fn copy(&self) {
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

    fn paste(&mut self) {
        let result = arboard::Clipboard::new()
            .and_then(|mut c| c.get_text())
            .map_err(|e| e.to_string())
            .and_then(|text| serde_json::from_str::<Entity>(&text).map_err(|e| e.to_string()));
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

    fn key(&mut self, key: &Key) {
        // Shortcuts would edit the scene the load is replacing.
        if self.loading.is_some() {
            return;
        }
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
                    "c" => self.copy(),
                    "v" => self.paste(),
                    "d" if ui::duplicate(&mut self.scene, &mut self.selected) => self.changed(),
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
                    Selection::Lighting
                    | Selection::Terrain
                    | Selection::City
                    | Selection::Widget(_)
                    | Selection::None => count - 1,
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
                        self.scene.remove_entity(i);
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
                    Selection::Widget(i) if i < self.scene.ui.len() => {
                        self.scene.ui.remove(i);
                        true
                    }
                    Selection::City => {
                        let edge = self.tools.city.edge.take();
                        if let (Some(e), Some(mut city)) = (edge, city::recipe(&self.scene)) {
                            city.remove_edge(e);
                            self.scene.city =
                                Some(serde_json::to_value(city).expect("city recipes serialize"));
                            self.changed();
                        }
                        false
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
                        | Selection::Widget(_)
                        | Selection::Sun
                        | Selection::Lighting
                        | Selection::Terrain
                        | Selection::City
                        | Selection::None => None,
                    };
                    if let Some(position) = position {
                        position[axis] += amount;
                        if let Selection::Entity(i) = self.selected {
                            for child in self.scene.descendants(i) {
                                self.scene.entities[child].transform.translation[axis] += amount;
                            }
                        }
                        self.changed();
                    }
                }
            }
            _ => {}
        }
    }

    fn window_command(&mut self, window: &Window, command: Option<WindowCommand>) {
        match command {
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
    }

    fn draw(&mut self, event_loop: &ActiveEventLoop) {
        if let Err(error) = self.frame() {
            Self::error(format!("Render failed: {error}"));
            event_loop.exit();
        }
        if std::mem::take(&mut self.close_requested) && self.can_discard() {
            event_loop.exit();
        }
    }

    fn frame(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.continuous = false;
        let Some(window) = self.window.clone() else {
            return Ok(());
        };
        // Font texture updates in a frame the loading thread never drew.
        let mut undrawn = None;
        if let Some(loading) = self.loading.take_if(|loading| loading.thread.is_finished()) {
            undrawn = (loading.shared.lock().unwrap().frame.take()).map(|(f, _)| f.textures_delta);
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
        // While loading, the UI shows progress but takes no edits, so none races the load.
        self.tools.loading =
            (self.loading.as_ref()).map(|loading| loading.shared.lock().unwrap().progress.clone());
        self.tools.profiler.begin_frame();
        if self.loading.is_none() {
            if let Some(play) = &mut self.play {
                play.runtime.focus = vec![self.tools.camera.eye];
            }
            let start = Instant::now();
            self.tick(false);
            self.tools.profiler.cpu("Simulation", 1, start.elapsed());
            self.profile_physics();
        }
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
        self.tools.profiler.cpu("UI", 1, ui_start.elapsed());
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
        let mut textures_delta = undrawn.unwrap_or_default();
        textures_delta.append(output.textures_delta);
        let frame = UiFrame {
            primitives: context.tessellate(output.shapes, scale),
            textures_delta,
            pixels_per_point: scale,
            viewport: [x, y, right - x, bottom - y],
        };
        if let Some(loading) = &self.loading {
            let mut shared = loading.shared.lock().unwrap();
            let mut frame = frame;
            // The loading popup covers the viewport, so progress frames skip
            // the 3D scene, which would make GI upload each model as it lands,
            // until the last step, which draws it to build GI.
            if (self.tools.loading.as_ref()).is_none_or(|p| p.done + 1 < p.total) {
                frame.viewport[2] = 0;
                frame.viewport[3] = 0;
            }
            // Keep the texture updates of a frame this one replaces.
            if let Some((older, _)) = shared.frame.take() {
                let mut textures = older.textures_delta;
                textures.append(frame.textures_delta);
                frame.textures_delta = textures;
            }
            shared.frame = Some((frame, [size.width, size.height]));
            drop(shared);
            self.window_command(&window, response.window);
            // Keep a fresh frame ready for the loading thread's next step.
            context.request_repaint_after(Duration::from_millis(33));
            return Ok(());
        }
        if let Some(renderer) = &mut self.renderer {
            let camera = &self.tools.camera;
            renderer.set_camera(camera.eye, camera.target, camera.vertical_fov);
            renderer.set_matrix_enabled(self.tools.matrix_enabled);
            renderer.set_matrix_debug(self.tools.matrix_enabled && self.tools.matrix_debug);
            renderer.set_uncapped(self.tools.uncapped);
            renderer.set_anti_aliasing(self.tools.anti_aliasing);
            renderer.set_sharpening(self.tools.sharpening);
            let mut deforming = soft_meshes(self.play.as_ref());
            deforming.extend(city::agents(
                self.play.as_ref().map(|p| &p.runtime),
                camera.eye,
            ));
            renderer.set_deforming_meshes(deforming);
            renderer.set_profiling(self.tools.profiler.open);
            let profiler = &mut self.tools.profiler;
            if let Some(lux) = renderer.extension_mut::<Lux>()
                && lux.follows_physics()
            {
                profiler.time("GI sync", 1, || match &self.play {
                    Some(play) => lux.sync(&play.runtime.physics, &play.runtime.world),
                    None => {
                        mirror_scene(&self.scene, &mut self.trace_world);
                        lux.sync(&self.no_physics, &self.trace_world);
                    }
                });
            }
            let terrain_busy = profiler.time("Terrain", 1, || {
                (self.terrain).update(terrain::generator(&self.scene), renderer, camera.eye)
            });
            self.tools.terrain_status = self.terrain.status();
            let ground = self.terrain.landscape();
            let interiors = self.tools.city.interiors;
            // Rebuilding waits for a drag to end, so dragging a point stays smooth.
            let holding = self.context.input(|i| i.pointer.any_down());
            profiler.time("City", 1, || {
                let wanted = city::recipe(&self.scene);
                self.city
                    .update(wanted, &ground, renderer, camera.eye, interiors, holding)
            });
            self.tools.city.ground = ground;
            self.tools.city.status = self.city.status();
            self.tools.city.lines = self.city.lines();
            // The sea draws with the scene but is not part of it: push it for
            // the draw and pop it after, rather than cloning the whole scene.
            let sea = self.terrain.sea_level().map(terrain::sea);
            let with_sea = sea.is_some();
            self.scene.entities.extend(sea);
            let render_start = Instant::now();
            let drawn = renderer.draw_ui(&self.scene, frame);
            if with_sea {
                self.scene.entities.pop();
            }
            drawn?;
            profiler.cpu("Render", 1, render_start.elapsed());
            if profiler.open {
                profile_renderer(profiler, renderer);
            }
            self.tools.gi_status = renderer.extension_mut::<Lux>().map(|lux| lux.status_line());
            // GI polls its compile and converges over frames, and play mode
            // simulates, so keep drawing; an open profiler measures continuous
            // frames, and uncapped means drawing whether or not anything changed.
            // A minimized window presents nothing to pace a polling loop.
            self.continuous = window.is_minimized() != Some(true)
                && (self.tools.gi_status.is_some()
                    || terrain_busy
                    || self.tools.play == PlayState::Playing
                    || self.tools.profiler.open
                    || self.tools.uncapped);
        }
        self.window_command(&window, response.window);
        if response.changed {
            self.changed();
        }
        if self.scene.lighting.method != method {
            // replace_assets restores from the scene on failure, so it must hold the old method.
            let new = std::mem::replace(&mut self.scene.lighting.method, method);
            let (assets, lights) = (self.scene.assets.clone(), self.scene.lights.len());
            let copies = self.scene.copies.clone();
            self.replace_assets(
                assets,
                copies,
                lights,
                new,
                move |editor, result| match result {
                    Ok(()) => editor.scene.lighting.method = new,
                    Err(error) => {
                        Self::error(format!("Cannot change the lighting method: {error}"))
                    }
                },
            );
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
            Some(Action::Copy) => self.copy(),
            Some(Action::Paste) => self.paste(),
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
        self.apply_lighting_method();
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
        // Windows delivers request_redraw as a low-priority WM_PAINT that only
        // arrives promptly while input wakes the loop, so continuous frames
        // draw from here instead; presenting paces them to the display.
        if self.continuous {
            event_loop.set_control_flow(ControlFlow::Poll);
            self.draw(event_loop);
            return;
        }
        if self.next_repaint.is_some_and(|when| when <= Instant::now()) {
            self.next_repaint = None;
            self.redraw();
        }
        event_loop.set_control_flow(match self.next_repaint {
            Some(when) => ControlFlow::WaitUntil(when),
            None => ControlFlow::Wait,
        });
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
            WindowEvent::RedrawRequested => self.draw(event_loop),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn play_lands_bodies_on_the_terrain() {
        use verity_physics::Ground;
        let planet = verity_terra::planet::Planet::new(Default::default());
        let ground = 12.0 - planet.distance(glam::Vec3::new(0.0, 0.0, 12.0));
        let info = verity_runtime::registry();
        let mut cube = Entity {
            name: "Cube".into(),
            transform: Transform {
                translation: [0.0, 0.0, ground + 5.0],
                ..Transform::default()
            },
            ..Entity::default()
        };
        let body = info.get("Rigid Body").unwrap();
        cube.components
            .insert(body.name.to_owned(), (body.default)());
        let scene = Scene {
            entities: vec![cube],
            ..Scene::default()
        };
        let resting = |ground: Option<Arc<dyn Ground>>| {
            let mut runtime = Runtime::from_scene(&scene).unwrap();
            let cube = runtime.world.handles()[0];
            if let Some(ground) = ground {
                add_ground(&mut runtime, ground);
            }
            runtime.advance(Duration::from_secs(3));
            runtime.world.get(cube).unwrap().transform.translation[2]
        };
        let landed = resting(Some(Arc::new(planet)));
        assert!((landed - ground).abs() < 2.0, "{landed} on {ground}");
        assert!(
            resting(None) < ground - 20.0,
            "without terrain it falls through"
        );
    }

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
            assert!(
                mesh.vertices
                    .iter()
                    .all(|v| v.position.iter().all(|p| p.is_finite()))
            );
        }
        // The heavy ball tore the trampoline, which duplicates vertices.
        let torn = meshes
            .iter()
            .zip(&start)
            .any(|(m, s)| m.vertices.len() > s.vertices.len());
        assert!(torn, "something tore");
        let hidden = editor.scene.entities[index("Curtain")].transform.scale;
        assert_eq!(hidden, [0.0; 3], "the entity's cube makes way for its mesh");
        editor.stop();
        assert_eq!(editor.scene, scene);
        assert!(soft_meshes(editor.play.as_ref()).is_empty());
    }

    #[test]
    fn play_uses_verity_physics_for_a_scene_that_names_box3d_and_leaves_it_saved_so() {
        let mut editor = Editor::new(egui::Context::default());
        editor.scene.physics.engine = verity_core::PhysicsEngine::Box3d;
        editor.scene.entities[0]
            .components
            .insert("Rigid Body".into(), serde_json::json!({}));
        editor.play();
        let runtime = &editor.play.as_ref().unwrap().runtime;
        assert_eq!(runtime.physics.body_count(), 1);
        editor.stop();
        assert_eq!(
            editor.scene.physics.engine,
            verity_core::PhysicsEngine::Box3d
        );
    }

    #[test]
    fn play_simulates_locks_edits_and_stop_restores() {
        let mut editor = Editor::new(egui::Context::default());
        editor.scene.entities[0].transform.translation = [0.0, 0.0, 5.0];
        editor.scene.entities[0]
            .components
            .insert("Rigid Body".into(), serde_json::json!({}));
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

/// The renderer's CPU stages under Render, its GPU stages with GI passes
/// under GI, and its counters.
fn profile_renderer(profiler: &mut profiler::Profiler, renderer: &mut Renderer) {
    if let Some(profile) = renderer.profile() {
        let names = verity_render::RENDER_STAGE_NAMES;
        // Indented names are sub-stages of "Record scene", listed last.
        let split = names
            .iter()
            .position(|n| n.starts_with("  "))
            .unwrap_or(names.len());
        for (i, name) in names[..split].iter().enumerate() {
            profiler.cpu(name, 2, profile.stages[i]);
            if *name == "Record scene" {
                for (sub, &time) in names[split..].iter().zip(&profile.stages[split..]) {
                    profiler.cpu(sub.trim_start(), 3, time);
                }
            }
        }
        if let Some(gpu) = profile.gpu_ms {
            profiler.gpu("GPU frame", 0, gpu.iter().sum());
            for (name, ms) in verity_render::GPU_STAGE_NAMES.iter().zip(gpu) {
                profiler.gpu(name, 1, ms);
                if *name == "GI"
                    && let Some(lux) = renderer.extension_mut::<Lux>()
                {
                    for &(pass, ms) in lux.pass_times() {
                        profiler.gpu(pass, 2, ms);
                    }
                }
            }
        }
    }
    let m = renderer.matrix_stats();
    profiler.counter(
        "Matrix",
        format!(
            "{} clusters · {} bricks · {} imposters · shadows {} clusters, {} bricks",
            m.clusters, m.bricks, m.imposters, m.shadow_clusters, m.shadow_bricks
        ),
    );
    let t = renderer.terrain_texture_stats();
    profiler.counter(
        "Terrain pages",
        format!(
            "{} / {} resident · {} wanted · {} loading · {} uploaded",
            t.resident, t.capacity, t.wanted, t.loading, t.uploaded
        ),
    );
    profiler.counter("Model draws", renderer.model_draw_count().to_string());
}
