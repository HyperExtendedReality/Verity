//! Sponza with Split Radiance Cascades GI.
//!
//! GI pipelines compile in the background; the first run takes a while
//! (later runs reuse the pipeline cache). Until then the scene renders without GI.
//! `cargo run --release -p hyperon-lux --example sponza_gi -- --gi`
use glam::Vec3;
use hyperon_core::World;
use hyperon_lux::{
    gi::Lux,
    settings::{Sampling, Settings},
};
use hyperon_render::{Renderer, Sun};
use std::{path::Path, sync::Arc, time::Instant};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

struct Viewer {
    world: World,
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
        let attributes = Window::default_attributes()
            .with_title("Hyperon Sponza GI")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
        let window = Arc::new(event_loop.create_window(attributes).expect("window"));
        let result = Renderer::new(window.clone()).and_then(|mut renderer| {
            if std::env::args().any(|a| a == "--gi") {
                let settings = Settings {
                    sampling: Sampling {
                        // About 300 MB of world cache instead of the uncapped default.
                        max_cache_entries: 262_144,
                        // Lux's benchmarked budget: 512 updates, 16,384 queries a frame.
                        max_updates_per_frame: 512,
                        max_traversal_queries: 16_384,
                        ..Sampling::default()
                    },
                    ..Settings::default()
                };
                renderer.set_extension(Box::new(Lux::new(settings)))?;
            }
            let packed = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/packed");
            for scene in ["NewSponza_Main.glb", "NewSponza_Curtains.glb"] {
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
            WindowEvent::Resized(size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                let Some(renderer) = &mut self.renderer else {
                    return;
                };
                let t = self.start.elapsed().as_secs_f32() * 0.05;
                let center = Vec3::new(2.0, -4.0, 3.0);
                let eye = center + Vec3::new(6.0 * t.cos(), -1.2 * t.sin(), 0.0);
                let target = center + Vec3::new(-6.0 * t.cos(), 1.2 * t.sin(), 2.0);
                renderer.set_camera(eye, target, 60f32.to_radians());
                if let Err(error) = renderer.draw_world(&self.world) {
                    eprintln!("render: {error}");
                    event_loop.exit();
                }
                self.frames += 1;
                if self.frames.is_multiple_of(120) {
                    let status = renderer
                        .extension_mut::<Lux>()
                        .map(|lux| lux.status().clone());
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
    EventLoop::new()?.run_app(&mut Viewer {
        world: World::new(),
        window: None,
        renderer: None,
        start: Instant::now(),
        frames: 0,
        timer: Instant::now(),
    })?;
    Ok(())
}
