//! Intel's New Sponza from `assets/packed` (run `scripts/convert_assets.py` first),
//! with a slow orbiting camera, a sun and a few warm point lights.
//!
//! `cargo run --release -p verity-render --example sponza [max texture size]`
//!
//! `--matrix` draws through Matrix; `--matrix-debug` also colors by tier.
//! `--uncapped` ignores vsync, for timing. `--smaa` anti-aliases with SMAA T2x.
use glam::Vec3;
use std::{
    error::Error,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};
use verity_core::{PointLight, World};
use verity_render::{Renderer, Sun};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

const SCENES: [&str; 3] = [
    "NewSponza_Main.glb",
    "NewSponza_Curtains.glb",
    "NewSponza_IvyGrowth.glb",
];

struct Viewer {
    world: World,
    max_texture_size: u32,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    start: Instant,
    /// Frames drawn and when counting began, for a periodic frame-time line.
    frames: (u32, Instant),
    view: Option<(Vec3, Vec3)>,
}

impl Viewer {
    fn load(&self, renderer: &mut Renderer) -> Result<(), Box<dyn Error>> {
        let packed = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/packed");
        println!(
            "textures as {:?}, capped at {}",
            renderer.texture_format(),
            self.max_texture_size
        );
        for scene in SCENES {
            let started = Instant::now();
            let stats = renderer.load_gltf(packed.join(scene), self.max_texture_size)?;
            println!("{scene}: {stats:?} in {:.2?}", started.elapsed());
        }
        let args: Vec<String> = std::env::args().collect();
        let debug = args.iter().any(|a| a == "--matrix-debug");
        renderer.set_matrix_enabled(debug || args.iter().any(|a| a == "--matrix"));
        renderer.set_matrix_debug(debug);
        renderer.set_uncapped(args.iter().any(|a| a == "--uncapped"));
        if args.iter().any(|a| a == "--smaa") {
            renderer.set_anti_aliasing(verity_render::AntiAliasing::Smaa);
        }
        // Until GI: a blue sky above, warm stone bouncing up from below.
        renderer.set_ambient(Vec3::new(0.25, 0.3, 0.4), Vec3::new(0.12, 0.1, 0.08));
        renderer.set_sun(Some(Sun {
            direction: Vec3::new(-0.3, -0.2, -1.0).normalize(),
            color: Vec3::new(1.0, 0.95, 0.88) * 3.0,
        }));
        Ok(())
    }
}

impl ApplicationHandler for Viewer {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title("Verity Sponza")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                eprintln!("window: {error}");
                event_loop.exit();
                return;
            }
        };
        let result = Renderer::new(window.clone()).and_then(|mut renderer| {
            self.load(&mut renderer)?;
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

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
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
            WindowEvent::RedrawRequested => {
                let Some(renderer) = &mut self.renderer else {
                    return;
                };
                // Orbit the open middle of the courtyard, clear of the columns.
                let t = self.start.elapsed().as_secs_f32() * 0.08;
                let center = Vec3::new(2.0, -4.0, 3.0);
                let eye = center + Vec3::new(6.0 * t.cos(), -1.2 * t.sin(), 0.0);
                let target = center + Vec3::new(-6.0 * t.cos(), 1.2 * t.sin(), 2.0);
                // `--view ex ey ez tx ty tz` holds the camera still, for inspection.
                let (eye, target) = self.view.unwrap_or((eye, target));
                renderer.set_camera(eye, target, 60f32.to_radians());
                if let Err(error) = renderer.draw_world(&self.world) {
                    eprintln!("render: {error}");
                    event_loop.exit();
                }
                self.frames.0 += 1;
                let elapsed = self.frames.1.elapsed();
                if elapsed >= Duration::from_secs(5) {
                    // Without `--uncapped` this is capped at the display's refresh rate.
                    println!(
                        "{:.2} ms/frame {:?}",
                        elapsed.as_secs_f64() * 1000.0 / self.frames.0 as f64,
                        renderer.matrix_stats()
                    );
                    self.frames = (0, Instant::now());
                }
            }
            _ => {}
        }
        event_loop.set_control_flow(ControlFlow::WaitUntil(
            Instant::now() + Duration::from_millis(1),
        ));
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let max_texture_size = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(4096);
    let mut world = World::new();
    // The export's own lights all have zero intensity; light the arcades by hand.
    for (i, x) in [-9.0f32, -3.0, 3.0, 9.0].into_iter().enumerate() {
        world.lights.push(PointLight {
            position: [x, -4.0 - if i % 2 == 0 { -2.5 } else { 2.5 }, 1.5],
            color: [1.0, 0.7, 0.4],
            intensity: 8.0,
            radius: 8.0,
        });
    }
    EventLoop::new()?.run_app(&mut Viewer {
        world,
        max_texture_size,
        window: None,
        renderer: None,
        start: Instant::now(),
        frames: (0, Instant::now()),
        view: fixed_view(),
    })?;
    Ok(())
}

/// Eye and target from `--view ex ey ez tx ty tz`.
fn fixed_view() -> Option<(Vec3, Vec3)> {
    let args: Vec<String> = std::env::args().collect();
    let at = args.iter().position(|a| a == "--view")?;
    let v: Vec<f32> = args
        .get(at + 1..at + 7)?
        .iter()
        .filter_map(|a| a.parse().ok())
        .collect();
    (v.len() == 6).then(|| (Vec3::new(v[0], v[1], v[2]), Vec3::new(v[3], v[4], v[5])))
}
