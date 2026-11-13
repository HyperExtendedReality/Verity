//! Sponza with Split Radiance Cascades GI.
//!
//! GI pipelines compile in the background; the first run takes a while
//! (later runs reuse the pipeline cache). Until then the scene renders without GI.
//! `cargo run --release -p verity-lux --example sponza_gi -- --gi`
//!
//! `--bench=DIR` measures instead of viewing: from a fixed camera it warms up,
//! prints each pass's mean GPU time, saves consecutive frames to `DIR` as PFM,
//! then does the same along a moving camera. `--size=WxH`, `--warm=N`,
//! `--eye=X,Y,Z`, `--target=X,Y,Z`, `--updates=N` and `--uncapped` adjust it, `--no-gi-light`
//! saves direct light alone, and `--pxtgi` (diffuse only without `--reflections`),
//! `--voxel` and `--reference` run those instead. `--reference` needs a high `--warm`.
//! H switches between hardware and software tracing while viewing.
//! `--turn=N` looks the other way for N frames before saving the still view.
//! `--lights=N` hangs N point lights in two rows along the nave. `--software`
//! traces without hardware ray queries, and `--world=M` sets PXTGI's ray reach.
//! `--scene=A.glb,B.glb` loads those from `assets/packed` instead of Sponza.
//! `--no-surface-cache` turns the surface cache off; `--surface-level=N` and
//! `--surface-texel=M` set the first cascade read from it and its texel size;
//! `--surface-updates=N` and `--surface-rays=N` its texels shaded a frame and
//! bounce rays each; `--surface-reach=M` caps software bounce rays.
use glam::Vec3;
use std::{collections::HashMap, path::Path, sync::Arc, time::Instant};
use verity_core::{PointLight, World};
use verity_lux::{
    Lux, Method,
    cascades::settings::Settings,
    pxtgi::{Capture, PxtgiDebug, PxtgiReflections, PxtgiSettings},
    voxel::ScreenVoxelSettings,
};
use verity_render::{Renderer, Sun};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    keyboard::Key,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

/// Frames each measurement averages or saves.
const TIMED: u32 = 120;
const SAVED: u32 = 8;

/// Where a benchmark has got to.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Stage {
    /// Waiting for pipelines, then this many frames.
    Warm(u32),
    Time(u32),
    /// Looking the other way for this many frames before saving (`--turn`).
    Away(u32),
    Save(u32),
}

/// A `--bench` run: a stationary camera, then a moving one.
struct Bench {
    dir: String,
    warm: u32,
    /// Frames to look away before saving the still view, testing how GI
    /// stands up to the camera turning around and back.
    turn: u32,
    stage: Stage,
    moving: bool,
    sums: HashMap<&'static str, (u32, f32, f32)>,
    order: Vec<&'static str>,
    /// The slowest frame timed: all of its passes.
    worst: f32,
    /// Advances only while moving, so saved frames repeat between runs.
    pose: f32,
}

fn point(name: &str) -> Option<Vec3> {
    let text = arg(name)?;
    let v: Vec<f32> = text.split(',').map(|c| c.parse().expect("X,Y,Z")).collect();
    Some(Vec3::new(v[0], v[1], v[2]))
}

fn arg(name: &str) -> Option<String> {
    std::env::args().find_map(|a| a.strip_prefix(&format!("--{name}=")).map(str::to_owned))
}

fn write_pfm(path: &str, capture: &Capture) -> std::io::Result<()> {
    let mut out = format!("PF\n{} {}\n-1.0\n", capture.width, capture.height).into_bytes();
    for row in capture.pixels.chunks(capture.width as usize).rev() {
        for pixel in row {
            for channel in &pixel[..3] {
                out.extend_from_slice(&channel.to_le_bytes());
            }
        }
    }
    std::fs::write(path, out)
}

impl Bench {
    fn report(&mut self) {
        let label = if self.moving { "moving" } else { "still" };
        let mut total = 0.0;
        // Per frame timed, as some passes run only on some frames.
        let frames = self.sums.values().map(|s| s.0).max().unwrap_or(1);
        for name in &self.order {
            let (n, sum, peak) = self.sums[name];
            let mean = sum / frames as f32;
            total += mean;
            let ran = if n < frames { format!("  in {n} of {frames} frames") } else { String::new() };
            println!("{label} {name:<20} {mean:8.3} ms  peak {peak:8.3}{ran}");
        }
        println!("{label} {:<20} {total:8.3} ms", "TOTAL");
        println!("{label} {:<20} {:8.3} ms", "WORST FRAME", self.worst);
        self.worst = 0.0;
        self.sums.clear();
        self.order.clear();
    }

    /// One drawn frame's bookkeeping, given the method's pass times and any
    /// finished capture. Returns whether the run is over and whether to capture.
    fn advance(&mut self, times: &[(&'static str, f32)], capture: Option<Capture>) -> (bool, bool) {
        let ready = !times.is_empty();
        if let Some(capture) = capture {
            let label = if self.moving { "moving" } else { "still" };
            let Stage::Save(left) = self.stage else {
                return (false, false);
            };
            let path = format!("{}/{label}{}.pfm", self.dir, SAVED - left);
            write_pfm(&path, &capture).expect("write capture");
            self.stage = if left > 1 {
                Stage::Save(left - 1)
            } else if self.moving {
                return (true, false);
            } else {
                self.moving = true;
                Stage::Warm(self.warm / 4)
            };
        }
        match self.stage {
            Stage::Warm(_) if !ready => {}
            Stage::Warm(0) => self.stage = Stage::Time(TIMED),
            Stage::Warm(left) => self.stage = Stage::Warm(left - 1),
            Stage::Time(0) => {
                self.report();
                self.stage = match self.turn {
                    0 => Stage::Save(SAVED),
                    turn => Stage::Away(turn),
                };
            }
            Stage::Away(0) => self.stage = Stage::Save(SAVED),
            Stage::Away(left) => self.stage = Stage::Away(left - 1),
            Stage::Time(left) => {
                for &(name, ms) in times {
                    let entry = self.sums.entry(name).or_insert_with(|| {
                        self.order.push(name);
                        (0, 0.0, 0.0)
                    });
                    *entry = (entry.0 + 1, entry.1 + ms, entry.2.max(ms));
                }
                self.worst = self.worst.max(times.iter().map(|(_, ms)| ms).sum());
                self.stage = Stage::Time(left - 1);
            }
            Stage::Save(_) => {}
        }
        if self.moving {
            self.pose += 1.0 / 60.0;
        }
        (false, matches!(self.stage, Stage::Save(_)))
    }
}

struct Viewer {
    world: World,
    bench: Option<Bench>,
    /// How far the camera swings around its centre.
    orbit: f32,
    /// Tracing with hardware ray queries; H switches.
    hardware_rays: bool,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    start: Instant,
    frames: u64,
    timer: Instant,
}

impl ApplicationHandler for Viewer {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let size = arg("size").unwrap_or("1280x720".into());
        let (width, height) = size.split_once('x').expect("--size=WxH");
        let attributes = Window::default_attributes()
            .with_title("Verity Sponza GI")
            .with_inner_size(winit::dpi::PhysicalSize::new(
                width.parse::<u32>().expect("width"),
                height.parse::<u32>().expect("height"),
            ));
        let window = Arc::new(event_loop.create_window(attributes).expect("window"));
        let result = Renderer::new(window.clone()).and_then(|mut renderer| {
            let flag = |name: &str| std::env::args().any(|a| a == name);
            if flag("--gi") || self.bench.is_some() {
                let mut settings = if flag("--uncapped") {
                    Settings::default()
                } else {
                    Settings::budgeted()
                };
                if let Some(updates) = arg("updates") {
                    settings.sampling.max_updates_per_frame = updates.parse().expect("--updates=N");
                    settings.sampling.max_traversal_queries =
                        settings.sampling.max_updates_per_frame * 32;
                }
                if let Some(divisor) = arg("grid") {
                    settings.sampling.grid_divisor = divisor.parse().expect("--grid=N");
                }
                if let Some(scale) = arg("resolve") {
                    settings.sampling.resolve_scale = scale.parse().expect("--resolve=N");
                }
                if let Some(entries) = arg("cache") {
                    settings.sampling.max_cache_entries = entries.parse().expect("--cache=N");
                }
                if flag("--no-gi-light") {
                    settings.intensity = 0.0;
                }
                settings.hardware_rays = !flag("--software");
                settings.surface_cache = !flag("--no-surface-cache");
                if let Some(level) = arg("surface-level") {
                    settings.surface_from_level = level.parse().expect("--surface-level=N");
                }
                if let Some(updates) = arg("surface-updates") {
                    settings.surface_updates = updates.parse().expect("--surface-updates=N");
                }
                if let Some(reach) = arg("surface-reach") {
                    settings.surface_software_reach = reach.parse().expect("--surface-reach=M");
                }
                if let Some(rays) = arg("surface-rays") {
                    settings.surface_bounce_rays = rays.parse().expect("--surface-rays=N");
                }
                if let Some(texel) = arg("surface-texel") {
                    settings.surface_texel = texel.parse().expect("--surface-texel=M");
                }
                if let Some(steps) = arg("filter") {
                    settings.spatial_filter_steps = steps.parse().expect("--filter=N");
                }
                if let Some(spacing) = arg("spacing") {
                    settings.probe_spacing = spacing.parse().expect("--spacing=M");
                }
                if let Some(length) = arg("interval") {
                    settings.interval_length = length.parse().expect("--interval=M");
                }
                let lux = Lux::new(if flag("--reference") {
                    Method::Pxtgi(PxtgiSettings {
                        debug: PxtgiDebug::Reference,
                        reflections: PxtgiReflections::Off,
                        world_ray_distance: 200.0,
                        hardware_rays: !flag("--software"),
                        ..PxtgiSettings::default()
                    })
                } else if flag("--pxtgi") {
                    Method::Pxtgi(PxtgiSettings {
                        rays_per_texel: arg("rays").map_or(2, |r| r.parse().expect("--rays=N")),
                        world_ray_distance: arg("world").map_or(30.0, |w| w.parse().expect("--world=M")),
                        debug: if flag("--world-only") {
                            PxtgiDebug::WorldOnly
                        } else {
                            PxtgiDebug::Off
                        },
                        hardware_rays: !flag("--software"),
                        multi_bounce: !flag("--single-bounce"),
                        max_distance: arg("screen").map_or(0.0, |m| m.parse().expect("--screen=M")),
                        reflections: if flag("--reflections") {
                            PxtgiReflections::Traced
                        } else {
                            PxtgiReflections::Off
                        },
                        ..PxtgiSettings::default()
                    })
                } else if flag("--voxel") {
                    Method::ScreenVoxel(ScreenVoxelSettings {
                        max_distance: if flag("--no-screen") { 0.0 } else { 8.0 },
                        rays_per_texel: arg("rays").map_or(4, |r| r.parse().expect("--rays=N")),
                        ..ScreenVoxelSettings::default()
                    })
                } else {
                    Method::SplitRadianceCascades(settings)
                });
                if self.bench.is_some() {
                    renderer.set_uncapped(true);
                }
                renderer.set_extension(Box::new(lux))?;
            }
            let packed = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/packed");
            let scenes = arg("scene").unwrap_or("NewSponza_Main.glb,NewSponza_Curtains.glb".into());
            for scene in scenes.split(',') {
                renderer.load_gltf(packed.join(scene), 512)?;
            }
            renderer.set_sun(Some(Sun {
                direction: Vec3::new(-0.3, -0.2, -1.0).normalize(),
                color: Vec3::splat(3.0),
            }));
            Ok(renderer)
        });
        match result {
            Ok(renderer) => {
                self.renderer = Some(renderer);
                self.window = Some(window);
                self.start = Instant::now();
            }
            Err(error) => {
                eprintln!("{error}");
                event_loop.exit();
            }
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if self.window.as_ref().map(|w| w.id()) != Some(id) {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. }
                if event.state.is_pressed()
                    && !event.repeat
                    && event.logical_key == Key::Character("h".into()) =>
            {
                if let Some(lux) = self.renderer.as_mut().and_then(|r| r.extension_mut::<Lux>()) {
                    self.hardware_rays = !self.hardware_rays;
                    lux.set_hardware_rays(self.hardware_rays);
                    let mode = if self.hardware_rays { "hardware" } else { "software" };
                    println!("tracing in {mode}");
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                let Some(renderer) = &mut self.renderer else {
                    return;
                };
                let t = match &self.bench {
                    Some(bench) => bench.pose * 0.5,
                    None => self.start.elapsed().as_secs_f32() * 0.05,
                };
                let center = Vec3::new(2.0, -4.0, 3.0);
                let orbit = Vec3::new(6.0 * t.cos(), -1.2 * t.sin(), 0.0);
                let eye = point("eye").unwrap_or(center) + orbit * self.orbit;
                let mut target = point("target").unwrap_or(center + Vec3::Z * 2.0) - orbit * self.orbit;
                if self.bench.as_ref().is_some_and(|b| matches!(b.stage, Stage::Away(_))) {
                    target = eye * 2.0 - target;
                }
                renderer.set_camera(eye, target, 60f32.to_radians());
                if let Err(error) = renderer.draw_world(&self.world) {
                    eprintln!("render: {error}");
                    event_loop.exit();
                }
                self.frames += 1;
                if let Some(bench) = &mut self.bench {
                    let lux = renderer.extension_mut::<Lux>().expect("GI is installed");
                    let finished = lux.take_capture();
                    let times = lux.pass_times().to_vec();
                    let (done, capture) = bench.advance(&times, finished);
                    if capture {
                        lux.request_capture();
                    }
                    if done {
                        event_loop.exit();
                    }
                    return;
                }
                if self.frames.is_multiple_of(120) {
                    let status = renderer.extension_mut::<Lux>().map(|lux| lux.status());
                    println!(
                        "{:.2} ms/frame, GI {status:?}",
                        self.timer.elapsed().as_secs_f64() * 1000.0 / 120.0
                    );
                    self.timer = Instant::now();
                }
            }
            _ => {}
        }
    }
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let bench = arg("bench").map(|dir| {
        std::fs::create_dir_all(&dir).expect("bench directory");
        let warm = arg("warm").map_or(600, |w| w.parse().expect("--warm=N"));
        Bench {
            dir,
            warm,
            turn: arg("turn").map_or(0, |n| n.parse().expect("--turn=N")),
            stage: Stage::Warm(warm),
            moving: false,
            sums: HashMap::new(),
            order: Vec::new(),
            worst: 0.0,
            pose: 0.0,
        }
    });
    let mut world = World::new();
    let count: u32 = arg("lights").map_or(0, |n| n.parse().expect("--lights=N"));
    for i in 0..count {
        let along = (i / 2) as f32 / (count.div_ceil(2).max(2) - 1) as f32;
        world.lights.push(PointLight {
            position: [-10.0 + 22.0 * along, if i % 2 == 0 { -6.0 } else { -2.0 }, 3.0],
            color: [1.0, 0.8, 0.6],
            intensity: 400.0 / count as f32,
            radius: 12.0,
        });
    }
    EventLoop::new()?.run_app(&mut Viewer {
        world,
        bench,
        orbit: if arg("eye").is_some() { 0.1 } else { 1.0 },
        hardware_rays: !std::env::args().any(|a| a == "--software"),
        window: None,
        renderer: None,
        start: Instant::now(),
        frames: 0,
        timer: Instant::now(),
    })?;
    Ok(())
}
