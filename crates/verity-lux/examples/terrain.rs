//! An eroded voxel landscape to fly over and dig into.
//!
//! `cargo run --release -p verity-lux --example terrain -- --landform island --seed 7 --gi pxtgi`
//! Landforms: valley, island, archipelago, continent, mountains, plains.
//! GI: none (the default), lux for Split Radiance Cascades, or pxtgi.
//!
//! Click to capture the mouse, then WASD to fly, Q/E to descend and rise, Shift
//! to go faster, hold the left button to dig and the right to fill, F to throw
//! a box, wheel for brush size, Esc to release. F5 saves edits and F9 loads
//! them, from `--edits` (default `terrain.edits`), which also loads at start.
//! Chunks stream in and out around the eye. The ground is textured with the
//! virtual-textured layers in `assets/packed/terrain` when they exist (see
//! `scripts/convert_terrain_layers.py`); `--no-layers` keeps it procedural.
//! Set `VERITY_SMOKE` to drop boxes,
//! dig under them through the real renderer, report and exit, pausing on each
//! view.
use glam::{IVec3, Vec3};
use std::{
    collections::HashSet,
    error::Error,
    fs::{self, File},
    io::{BufReader, BufWriter},
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};
use verity_core::{Entity, EntityHandle, Material, Transform, World};
use verity_lux::{
    GiStatus, Lux, Method,
    cascades::settings::Settings as LuxSettings,
    pxtgi::{Pxtgi, PxtgiSettings},
};
use verity_physics::{
    Bounds, Collider, PhysicsBody, PhysicsSettings, PhysicsWorld, RigidBody, Velocity,
};
use verity_render::{
    GpuMaterial, Renderer, RetainedMesh, Sun, TERRAIN_ROLES, TerrainLayer, TerrainRole, Vertex,
};
use verity_terra::{
    CELLS, Landform, Landscape, MeshData, Mesher, Settings, Streamer, Terrain, VOXEL, chunk_origin,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

/// Finished meshes accepted per frame, which bounds each frame's upload.
const UPLOAD_BUDGET: usize = 64;
const REACH: f32 = 90.0;
/// How far from the eye chunks stay meshed.
const STREAM_RADIUS: f32 = 200.0;
/// Physics steps at this rate, however fast frames come.
const PHYSICS_STEP: f32 = 1.0 / 60.0;
/// Boxes in the world at most: a new one replaces the oldest.
const MAX_BOXES: usize = 256;
const BOX_SIZE: f32 = 1.0;
const BOX_MATERIAL: Material = Material {
    base_color: [0.72, 0.33, 0.12],
    metallic: 0.0,
    roughness: 0.6,
};
/// Radiance of the sky, for ambient light and for GI rays that escape.
const SKY: Vec3 = Vec3::new(0.42, 0.52, 0.68);
const SMOKE_PAUSE: Duration = Duration::from_secs(4);
/// Brush strokes in the smoke test's trench, 80 ms apart.
const TRENCH_STROKES: u32 = 40;

/// The converted layer for each role, and the metres it covers before repeating.
fn terrain_layers() -> Option<[TerrainLayer; TERRAIN_ROLES]> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/packed/terrain");
    let layers = TerrainRole::ALL.map(|role| {
        let (name, size) = match role {
            TerrainRole::Grass => ("Grass002", 3.0),
            TerrainRole::Meadow => ("Grass003", 3.0),
            TerrainRole::Soil => ("Ground051", 3.0),
            TerrainRole::Loam => ("Ground110", 3.0),
            TerrainRole::Subsoil => ("Ground112", 2.5),
            TerrainRole::Rock => ("Rock031", 6.0),
            TerrainRole::Scree => ("Ground003", 3.0),
            TerrainRole::Gravel => ("Ground063", 2.5),
            TerrainRole::Sand => ("Ground079S", 3.0),
        };
        TerrainLayer::converted(&dir, name, size)
    });
    layers
        .iter()
        .all(|l| l.albedo.exists() && l.surface.exists())
        .then_some(layers)
}

fn key(coord: IVec3) -> u64 {
    let [x, y, z] = coord.to_array().map(|c| c as u16 as u64);
    x | y << 16 | z << 32
}

fn retained(mesh: MeshData) -> Option<RetainedMesh> {
    let vertices = (mesh
        .positions
        .iter()
        .zip(&mesh.normals)
        .zip(&mesh.terrain_uv))
    .map(|((p, n), uv)| Vertex {
        position: [p[0], p[1], p[2], uv[0]],
        normal: [n[0], n[1], n[2], uv[1]],
        tangent: [0.0; 4],
        color: [1.0; 4],
    })
    .collect();
    (!mesh.indices.is_empty()).then_some(RetainedMesh {
        vertices,
        indices: mesh.indices,
        bounds: mesh.bounds,
        plain: false,
    })
}

#[derive(Clone, Copy, PartialEq)]
enum Gi {
    None,
    Lux,
    Pxtgi,
}

#[derive(Clone, Copy, PartialEq)]
enum Smoke {
    Off,
    Landscape,
    Closeup,
    /// Digging along a line, like holding the button down, with strokes left.
    Trench(Vec3, u32),
    Dug(Vec3),
    Done,
}

struct Viewer {
    terrain: Terrain,
    mesher: Mesher,
    streamer: Streamer,
    /// Where F5 saves edits and F9 loads them.
    edits: PathBuf,
    world: World,
    physics: PhysicsWorld,
    physics_settings: PhysicsSettings,
    /// Time not yet stepped.
    physics_time: f32,
    boxes: Vec<EntityHandle>,
    gi: Gi,
    /// Terrain textures, or `None` for procedural ground.
    layers: Option<[TerrainLayer; TERRAIN_ROLES]>,
    sea_level: Option<f32>,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    eye: Vec3,
    yaw: f32,
    pitch: f32,
    keys: HashSet<KeyCode>,
    digging: bool,
    filling: bool,
    grabbed: bool,
    radius: f32,
    /// Earliest time the brush may stroke again, and where it last did.
    next_stroke: Instant,
    last_center: Option<Vec3>,
    strokes: u64,
    last_frame: Instant,
    title_at: Instant,
    frames: u32,
    frame_seconds: f32,
    /// The longest frame since the smoke test last reported.
    worst_frame: f32,
    smoke: Smoke,
    settled_at: Option<Instant>,
}

impl Viewer {
    fn forward(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        Vec3::new(cos_pitch * cos_yaw, cos_pitch * sin_yaw, sin_pitch)
    }

    fn look_at(&mut self, eye: Vec3, target: Vec3) {
        let direction = (target - eye).normalize();
        self.eye = eye;
        self.yaw = direction.y.atan2(direction.x);
        self.pitch = direction.z.asin();
    }

    fn spawn_box(&mut self, at: Vec3, velocity: Vec3) {
        if self.boxes.len() >= MAX_BOXES {
            let oldest = self.boxes.remove(0);
            self.physics.remove(oldest);
            self.world.despawn(oldest);
        }
        let entity = self.world.spawn(Entity {
            name: "Box".into(),
            transform: Transform {
                translation: at.to_array(),
                scale: [BOX_SIZE; 3],
                ..Transform::default()
            },
            material: BOX_MATERIAL,
            components: Default::default(),
        });
        let collider = Collider::Cuboid {
            half_extents: Vec3::splat(BOX_SIZE * 0.5),
        };
        let mut body = PhysicsBody::new(RigidBody::Dynamic, collider);
        body.velocity = Velocity {
            linear: velocity,
            ..Velocity::default()
        };
        self.physics
            .insert(&self.world, entity, body)
            .expect("just spawned");
        self.boxes.push(entity);
    }

    /// Dig or fill, then give physics the new ground: bodies near the brush wake.
    fn brush(&mut self, center: Vec3, radius: f32, add: bool) -> usize {
        let dirty = self.terrain.brush(center, radius, add);
        let changed = dirty.len();
        self.mesher.queue(dirty);
        if changed > 0 {
            let reach = Vec3::splat(radius + 1.0);
            let region = Bounds {
                min: center - reach,
                max: center + reach,
            };
            self.physics
                .set_ground(Arc::new(self.terrain.clone()), region);
        }
        changed
    }

    /// Written aside and renamed over the old save, so a failed save loses nothing.
    fn save_edits(&self) {
        let temporary = self.edits.with_extension("tmp");
        let result = File::create(&temporary).and_then(|file| {
            let mut out = BufWriter::new(file);
            self.terrain.save_edits(&mut out)?;
            out.into_inner()?.sync_all()?;
            fs::rename(&temporary, &self.edits)
        });
        match result {
            Ok(()) => println!(
                "saved {} edited chunks to {}",
                self.terrain.edited().count(),
                self.edits.display()
            ),
            Err(error) => eprintln!("save {}: {error}", self.edits.display()),
        }
    }

    fn load_edits(&mut self) {
        let file = File::open(&self.edits).map(BufReader::new);
        let changed = match file.and_then(|file| self.terrain.load_edits(file)) {
            Ok(changed) => changed,
            Err(error) => return eprintln!("load {}: {error}", self.edits.display()),
        };
        println!("loaded {} edited chunks", self.terrain.edited().count());
        let Some(&first) = changed.first() else {
            return;
        };
        let size = Vec3::splat((CELLS + 2) as f32 * VOXEL);
        let mut region = Bounds {
            min: chunk_origin(first),
            max: chunk_origin(first) + size,
        };
        for &coord in &changed {
            region.min = region.min.min(chunk_origin(coord));
            region.max = region.max.max(chunk_origin(coord) + size);
        }
        // Chunks out of reach have no mesh, and get one when they come into it.
        let streamer = &self.streamer;
        (self.mesher).queue(changed.into_iter().filter(|c| streamer.is_resident(*c)));
        self.physics
            .set_ground(Arc::new(self.terrain.clone()), region);
    }

    fn step_physics(&mut self, seconds: f32) {
        // Falling behind stays bounded: time past a few steps is dropped.
        self.physics_time = (self.physics_time + seconds).min(4.0 * PHYSICS_STEP);
        while self.physics_time >= PHYSICS_STEP {
            self.physics_time -= PHYSICS_STEP;
            (self.physics).step(&mut self.world, &self.physics_settings, PHYSICS_STEP);
        }
    }

    fn fly(&mut self, seconds: f32) {
        let forward = self.forward();
        let right = forward.cross(Vec3::Z).normalize_or_zero();
        let mut movement = Vec3::ZERO;
        for (keys, direction) in [
            ([KeyCode::KeyW, KeyCode::ArrowUp], forward),
            ([KeyCode::KeyS, KeyCode::ArrowDown], -forward),
            ([KeyCode::KeyD, KeyCode::ArrowRight], right),
            ([KeyCode::KeyA, KeyCode::ArrowLeft], -right),
            ([KeyCode::KeyE, KeyCode::KeyE], Vec3::Z),
            ([KeyCode::KeyQ, KeyCode::KeyQ], -Vec3::Z),
        ] {
            if keys.iter().any(|key| self.keys.contains(key)) {
                movement += direction;
            }
        }
        let fast = self.keys.contains(&KeyCode::ShiftLeft);
        let speed = if fast { 48.0 } else { 14.0 };
        self.eye += movement.normalize_or_zero() * speed * seconds.min(0.05);
        self.eye = self.eye.clamp(
            Vec3::new(-185.0, -185.0, -43.0),
            Vec3::new(185.0, 185.0, 180.0),
        );
    }

    fn sculpt(&mut self) {
        let now = Instant::now();
        if !self.grabbed || !(self.digging || self.filling) {
            self.last_center = None;
            return;
        }
        if now < self.next_stroke {
            return;
        }
        let Some(hit) = self.terrain.raycast(self.eye, self.forward(), REACH) else {
            self.last_center = None;
            return;
        };
        let normal = self.terrain.normal(hit);
        let offset = if self.filling { 0.4 } else { -0.35 };
        let center = hit + normal * self.radius * offset;
        if (self.last_center).is_some_and(|last| last.distance(center) < 0.25) {
            return;
        }
        let changed = self.brush(center, self.radius, self.filling);
        self.strokes += u64::from(changed > 0);
        self.next_stroke = now + Duration::from_millis(80);
        self.last_center = Some(center);
    }

    /// Port of the viewer's smoke test: dig where the camera looks, and check it.
    fn smoke_step(&mut self, event_loop: &ActiveEventLoop) {
        if let Smoke::Trench(center, strokes) = self.smoke {
            if Instant::now() >= self.next_stroke {
                self.next_stroke = Instant::now() + Duration::from_millis(80);
                assert!(
                    self.brush(center, 7.0, false) > 0,
                    "digging must change samples"
                );
                let along = center + Vec3::new(0.7, -0.5, -0.1);
                self.smoke = match strokes {
                    1 => Smoke::Dug(center),
                    _ => Smoke::Trench(along, strokes - 1),
                };
            }
            return;
        }
        if self.smoke == Smoke::Off || self.mesher.outstanding() > 0 || !self.gi_ready() {
            self.settled_at = None;
            return;
        }
        let settled = *self.settled_at.get_or_insert_with(|| {
            println!("settled");
            Instant::now()
        });
        if settled.elapsed() < SMOKE_PAUSE {
            return;
        }
        self.settled_at = None;
        let drawn = self
            .renderer
            .as_ref()
            .map_or((0, 0), |r| r.retained_drawn());
        let pages = (self.renderer.as_ref())
            .map(|r| r.terrain_texture_stats())
            .unwrap_or_default();
        println!(
            "{:.2} ms a frame, worst {:.1} ms, drawing {} meshes, {} triangles, {} of {} pages wanted resident, {} uploaded",
            self.frame_seconds * 1000.0,
            self.worst_frame * 1000.0,
            drawn.0,
            drawn.1,
            pages.resident,
            pages.wanted,
            pages.uploaded,
        );
        self.worst_frame = 0.0;
        match self.smoke {
            Smoke::Landscape => {
                let height = self.terrain.landscape.height(24.0, 52.0);
                let target = Vec3::new(24.0, 52.0, height);
                self.look_at(target + Vec3::new(10.0, 18.0, 12.0), target);
                self.smoke = Smoke::Closeup;
            }
            Smoke::Closeup => {
                let hit = self.terrain.raycast(self.eye, self.forward(), REACH);
                let hit = hit.expect("the camera looks at the ground");
                // Boxes fall in, and the trench opens under those that landed.
                for i in 0..12 {
                    let offset = Vec3::new((i % 4) as f32 * 2.0, (i / 4) as f32 * 2.0, 6.0);
                    self.spawn_box(hit + offset - Vec3::new(3.0, 2.0, 0.0), Vec3::ZERO);
                }
                self.smoke = Smoke::Trench(hit - Vec3::Z * 2.0, TRENCH_STROKES);
            }
            Smoke::Dug(center) => {
                assert!(
                    self.terrain.sample(center) > 0.0,
                    "digging must remove ground"
                );
                // What the tracer sees must be the dug ground, not the old surface.
                let (eye, forward) = (self.eye, self.forward());
                let lux = (self.renderer.as_mut()).and_then(|r| r.extension_mut::<Lux>());
                let pxtgi = lux.and_then(|lux| lux.method_mut::<Pxtgi>());
                if let Some(pxtgi) = pxtgi {
                    let field = self.terrain.raycast(eye, forward, REACH).expect("ground");
                    let hit = (pxtgi.ray_scene().ray(eye, forward, REACH)).expect("traced ground");
                    let gap = (eye + forward * hit.t).distance(field);
                    assert!(gap < 1.0, "the tracer is {gap} m from the terrain");
                    println!("traced ground within {gap:.3} m of the field");
                }
                for entity in &self.boxes {
                    let found = self.world.get(*entity).expect("alive");
                    let at = Vec3::from_array(found.transform.translation);
                    assert!(at.is_finite(), "a box exploded");
                    let depth = -self.terrain.sample(at);
                    assert!(depth < 0.5, "a box is {depth} m underground at {at}");
                }
                println!(
                    "{} boxes on the ground, {} contacts",
                    self.boxes.len(),
                    self.physics.contact_count()
                );
                println!(
                    "smoke test passed, {} stale meshes rejected",
                    self.mesher.rejected
                );
                self.smoke = Smoke::Done;
                event_loop.exit();
            }
            Smoke::Off | Smoke::Trench(..) | Smoke::Done => {}
        }
    }

    /// Whether the GI method is running, so a smoke test really exercises it.
    fn gi_ready(&mut self) -> bool {
        let Some(renderer) = &mut self.renderer else {
            return false;
        };
        match renderer.extension_mut::<Lux>().map(|lux| lux.status()) {
            Some(GiStatus::Compiling { .. }) => false,
            Some(GiStatus::Failed(error)) => panic!("lux: {error}"),
            _ => true,
        }
    }

    fn set_grab(&mut self, grabbed: bool) {
        let Some(window) = &self.window else {
            return;
        };
        let mode = if grabbed {
            CursorGrabMode::Locked
        } else {
            CursorGrabMode::None
        };
        // Not every platform can lock the cursor; confining it still works for looking.
        let result = (window.set_cursor_grab(mode))
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
        if let Err(error) = result {
            eprintln!("cursor: {error}");
        }
        window.set_cursor_visible(!grabbed);
        self.grabbed = grabbed;
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        let now = Instant::now();
        let seconds = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;
        self.frame_seconds += (seconds - self.frame_seconds) * 0.05;
        self.worst_frame = self.worst_frame.max(seconds);
        if self.grabbed {
            self.fly(seconds);
        }
        self.sculpt();
        self.step_physics(seconds);
        let unloaded = (self.streamer).update(&mut self.terrain, &mut self.mesher, self.eye);
        let ready = (self.mesher).update(&mut self.terrain, self.eye, UPLOAD_BUDGET);
        let (eye, forward) = (self.eye, self.forward());
        if let Some(renderer) = &mut self.renderer {
            for coord in unloaded {
                renderer.set_retained_mesh(key(coord), None);
            }
            for (coord, mesh) in ready {
                renderer.set_retained_mesh(key(coord), retained(mesh));
            }
            renderer.set_camera(eye, eye + forward, 60f32.to_radians());
            if let Some(lux) = renderer.extension_mut::<Lux>() {
                lux.sync(&self.physics, &self.world);
            }
            if let Err(error) = renderer.draw_world(&self.world) {
                eprintln!("render: {error}");
                event_loop.exit();
                return;
            }
        }
        self.frames += 1;
        self.smoke_step(event_loop);
        if now.duration_since(self.title_at) > Duration::from_millis(250) {
            self.title_at = now;
            if let (Some(window), Some(renderer)) = (&self.window, &self.renderer) {
                let (meshes, triangles) = renderer.retained_drawn();
                let pages = renderer.terrain_texture_stats();
                window.set_title(&format!(
                    "Terra: {:.1} ms, {meshes} chunks, {}k triangles, {} pending, {} edits, brush {:.1} m, pages {}/{} ({} loading)",
                    self.frame_seconds * 1000.0,
                    triangles / 1000,
                    self.mesher.outstanding(),
                    self.strokes,
                    self.radius,
                    pages.resident,
                    pages.capacity,
                    pages.loading,
                ));
            }
        }
    }
}

impl ApplicationHandler for Viewer {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title("Terra")
            .with_inner_size(PhysicalSize::new(1440, 900));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                eprintln!("window: {error}");
                event_loop.exit();
                return;
            }
        };
        match Renderer::new(window.clone()) {
            Ok(mut renderer) => {
                renderer.set_clear_color([0.46, 0.62, 0.84, 1.0]);
                renderer.set_ambient(SKY, Vec3::new(0.13, 0.11, 0.09));
                renderer.set_sun(Some(Sun {
                    direction: Vec3::new(100.0, -70.0, -125.0).normalize(),
                    color: Vec3::new(2.5, 2.3, 1.95),
                }));
                // Sand needs a waterline: a dry world's sits below all of its ground.
                let sea = self.sea_level.unwrap_or(-1.0e4);
                renderer.set_retained_material(GpuMaterial::terrain(0.19, sea));
                if let Some(layers) = &self.layers
                    && let Err(error) = renderer.set_terrain_layers(layers)
                {
                    eprintln!("terrain layers: {error}; the ground stays procedural");
                }
                renderer.set_shadow_distance(260.0);
                renderer.set_uncapped(self.smoke != Smoke::Off);
                let method = match self.gi {
                    Gi::None => None,
                    Gi::Lux => Some(Method::SplitRadianceCascades(LuxSettings::landscape(
                        SKY.to_array(),
                    ))),
                    Gi::Pxtgi => Some(Method::Pxtgi(PxtgiSettings::default())),
                };
                let installed =
                    method.map_or(Ok(()), |m| renderer.set_extension(Box::new(Lux::new(m))));
                if let Err(error) = installed {
                    eprintln!("gi: {error}");
                    event_loop.exit();
                    return;
                }
                self.renderer = Some(renderer);
                self.window = Some(window);
                self.last_frame = Instant::now();
            }
            Err(error) => {
                eprintln!("renderer: {error}");
                event_loop.exit();
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.smoke != Smoke::Off {
            if let Some(window) = &self.window {
                window.request_redraw();
            }
            event_loop.set_control_flow(ControlFlow::Poll);
            return;
        }
        let next = self.last_frame + Duration::from_millis(8);
        if Instant::now() >= next
            && let Some(window) = &self.window
        {
            window.request_redraw();
        }
        event_loop.set_control_flow(ControlFlow::WaitUntil(next));
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        if let (true, DeviceEvent::MouseMotion { delta }) = (self.grabbed, event) {
            self.yaw -= delta.0 as f32 * 0.002;
            self.pitch = (self.pitch - delta.1 as f32 * 0.002).clamp(-1.5, 1.5);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if self.window.as_ref().map(|window| window.id()) != Some(id) {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size.width, size.height);
                }
            }
            WindowEvent::Focused(false) => self.set_grab(false),
            WindowEvent::KeyboardInput { event, .. } => {
                let PhysicalKey::Code(code) = event.physical_key else {
                    return;
                };
                if event.state == ElementState::Pressed {
                    self.keys.insert(code);
                    if code == KeyCode::KeyF && self.grabbed && !event.repeat {
                        let forward = self.forward();
                        self.spawn_box(self.eye + forward * 2.0, forward * 25.0);
                    }
                    if code == KeyCode::Escape {
                        self.set_grab(false);
                    }
                    if !event.repeat && code == KeyCode::F5 {
                        self.save_edits();
                    }
                    if !event.repeat && code == KeyCode::F9 {
                        self.load_edits();
                    }
                } else {
                    self.keys.remove(&code);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                if pressed && !self.grabbed {
                    // The click that captures the mouse does not dig.
                    self.set_grab(true);
                    self.next_stroke = Instant::now() + Duration::from_millis(200);
                    return;
                }
                match button {
                    MouseButton::Left => self.digging = pressed,
                    MouseButton::Right => self.filling = pressed,
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                self.radius = (self.radius + lines * 0.35).clamp(1.5, 12.0);
            }
            WindowEvent::RedrawRequested => self.redraw(event_loop),
            _ => {}
        }
    }
}

fn landform(name: &str) -> Option<Landform> {
    Some(match name {
        "valley" => Landform::Valley,
        "island" => Landform::Island,
        "archipelago" => Landform::Archipelago,
        "continent" => Landform::Continent,
        "mountains" => Landform::Mountains,
        "plains" => Landform::Plains,
        _ => return None,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let value = |flag: &str| {
        let at = args.iter().position(|arg| arg == flag)?;
        args.get(at + 1).map(String::as_str)
    };
    let mut settings = Settings::default();
    if let Some(name) = value("--landform") {
        settings = landform(name).ok_or("unknown --landform")?.into();
    }
    if let Some(seed) = value("--seed") {
        settings.seed = seed.parse()?;
    }
    let gi = match value("--gi") {
        None | Some("none") => Gi::None,
        Some("lux") => Gi::Lux,
        Some("pxtgi") => Gi::Pxtgi,
        Some(_) => return Err("unknown --gi".into()),
    };

    let start = Instant::now();
    let landscape = Landscape::generate(settings.clone());
    println!(
        "generated in {:.0} ms",
        start.elapsed().as_secs_f64() * 1000.0
    );
    let sea = settings.sea_level.unwrap_or(f32::NEG_INFINITY);
    // The valley opens on its river floor; every other landform on an overview.
    let (eye, target) = if settings.landform == Landform::Valley {
        (
            Vec3::new(54.0, 100.0, landscape.height(54.0, 100.0) + 18.0),
            Vec3::new(10.0, 35.0, landscape.height(10.0, 35.0)),
        )
    } else {
        let ground = landscape.height(-150.0, -175.0).max(sea);
        (
            Vec3::new(-150.0, -175.0, (ground + 90.0).min(175.0)),
            Vec3::new(0.0, 0.0, landscape.height(0.0, 0.0).max(sea) * 0.5),
        )
    };
    let terrain = Terrain::new(landscape);
    let mut physics = PhysicsWorld::default();
    let streamer = Streamer::new(&terrain, STREAM_RADIUS);
    let mut world = World::new();
    if let Some(level) = settings.sea_level {
        // Extends past the 4 km world so the sea reaches the horizon.
        world.spawn(Entity {
            name: "Sea".into(),
            transform: Transform {
                translation: [0.0, 0.0, level - 0.5],
                scale: [8000.0, 8000.0, 1.0],
                ..Transform::default()
            },
            material: Material {
                base_color: [0.05, 0.19, 0.24],
                metallic: 0.0,
                roughness: 0.06,
            },
            components: Default::default(),
        });
    }
    // The ground is a body like any other, drawn by its chunks rather than a cube.
    let ground = world.spawn(Entity {
        name: "Ground".into(),
        transform: Transform {
            scale: [0.0; 3],
            ..Transform::default()
        },
        material: Material::default(),
        components: Default::default(),
    });
    let body = PhysicsBody::new(RigidBody::Static, Collider::Ground);
    physics.insert(&world, ground, body)?;
    physics.set_ground(Arc::new(terrain.clone()), Bounds::EMPTY);
    let now = Instant::now();
    let mut viewer = Viewer {
        terrain,
        mesher: Mesher::default(),
        streamer,
        edits: PathBuf::from(value("--edits").unwrap_or("terrain.edits")),
        world,
        physics,
        physics_settings: PhysicsSettings::default(),
        physics_time: 0.0,
        boxes: Vec::new(),
        gi,
        layers: if args.iter().any(|a| a == "--no-layers") {
            None
        } else {
            let layers = terrain_layers();
            if layers.is_none() {
                println!("no terrain layers in assets/packed/terrain: the ground is procedural");
            }
            layers
        },
        sea_level: settings.sea_level,
        window: None,
        renderer: None,
        eye,
        yaw: 0.0,
        pitch: 0.0,
        keys: HashSet::new(),
        digging: false,
        filling: false,
        grabbed: false,
        radius: 3.5,
        next_stroke: now,
        last_center: None,
        strokes: 0,
        last_frame: now,
        title_at: now,
        frames: 0,
        frame_seconds: 0.016,
        worst_frame: 0.0,
        smoke: if std::env::var_os("VERITY_SMOKE").is_some() {
            Smoke::Landscape
        } else {
            Smoke::Off
        },
        settled_at: None,
    };
    viewer.look_at(eye, target);
    if viewer.smoke == Smoke::Off && viewer.edits.exists() {
        viewer.load_edits();
    }
    let event_loop = EventLoop::new()?;
    event_loop.run_app(&mut viewer)?;
    Ok(())
}
