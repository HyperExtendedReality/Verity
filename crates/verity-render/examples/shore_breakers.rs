//! GPU-animated breakers driven by the same shore profile as CPU water queries.
//! Run with `cargo run --release -p verity-render --example shore_breakers`.
use std::{error::Error, io, sync::Arc, time::Instant};

use glam::{Vec2, Vec3};
use verity_core::{Entity, Material, Transform, World};
use verity_ocean::{Seabed, ShoreSettings, ShoreWaves, Water};
use verity_render::{breaker, Renderer, Sun};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

struct Demo {
    shore: ShoreWaves,
    world: World,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    start: Instant,
    frames: u32,
    frame_limit: Option<u32>,
    error: Option<String>,
}

impl Demo {
    fn fail(&mut self, event_loop: &ActiveEventLoop, error: impl std::fmt::Display) {
        self.error = Some(error.to_string());
        event_loop.exit();
    }
}

impl ApplicationHandler for Demo {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = match event_loop.create_window(
            Window::default_attributes()
                .with_title("Verity shore breakers")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720)),
        ) {
            Ok(window) => Arc::new(window),
            Err(error) => return self.fail(event_loop, error),
        };
        let mut renderer = match Renderer::new(window.clone()) {
            Ok(renderer) => renderer,
            Err(error) => return self.fail(event_loop, error),
        };
        renderer.set_camera(
            Vec3::new(39.0, -24.0, 17.0),
            Vec3::new(23.0, 19.0, 0.0),
            50f32.to_radians(),
        );
        renderer.set_sun(Some(Sun {
            direction: Vec3::new(-0.4, 0.2, -1.0).normalize(),
            color: Vec3::splat(3.0),
        }));
        let grid = self.shore.breaker_grid();
        let profile = self.shore.profile();
        let upload = breaker::BreakerGrid {
            width: grid.width,
            height: grid.height,
            cell: grid.cell,
            origin: grid.origin,
            sea_level: grid.sea_level,
            omega: grid.omega,
            time: grid.time,
            lean: grid.lean,
            phase: grid.phase,
            direction: grid.direction,
            wave_height: grid.wave_height,
            wavenumber: grid.wavenumber,
            depth: grid.depth,
            breaking: grid.breaking,
        };
        let profile = breaker::BreakerProfile {
            horizontal_second_harmonic: profile.horizontal_second_harmonic,
            vertical_second_harmonic: profile.vertical_second_harmonic,
        };
        if let Err(error) = renderer.set_breakers(upload, profile) {
            return self.fail(event_loop, error);
        }
        self.start = Instant::now();
        self.renderer = Some(renderer);
        self.window = Some(window);
        println!("breaker demo ready");
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
                let time = self.start.elapsed().as_secs_f32();
                self.shore.set_time(time);
                renderer.set_breaker_time(time);
                if let Err(error) = renderer.draw_world(&self.world) {
                    return self.fail(event_loop, error);
                }
                self.frames += 1;
                if self.frame_limit.is_some_and(|limit| self.frames >= limit) {
                    let water = self.shore.sample(Vec2::new(30.0, 20.0));
                    println!(
                        "rendered {} frames; CPU water height {:.4}, velocity {:?}",
                        self.frames, water.height, water.velocity
                    );
                    event_loop.exit();
                }
            }
            _ => {}
        }
        event_loop.set_control_flow(ControlFlow::Poll);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let frame_limit = std::env::args()
        .nth(1)
        .map(|arg| arg.parse::<u32>())
        .transpose()?;
    let mut seabed = Seabed::flat(96, 80, 0.5, 0.0);
    let mut world = World::new();
    world.spawn(Entity {
        name: "Beach".into(),
        transform: Transform {
            translation: [44.0, 20.0, -0.2],
            scale: [8.0, 40.0, 0.4],
            ..Default::default()
        },
        material: Material {
            base_color: [0.65, 0.51, 0.32],
            roughness: 1.0,
            ..Default::default()
        },
        ..Default::default()
    });
    for y in 0..seabed.height {
        for x in 0..seabed.width {
            let point = seabed.cell_center(x, y);
            seabed.elevation[x + y * seabed.width] = -4.0 + 0.1 * point.x;
        }
    }
    let shore = ShoreWaves::new(
        &seabed,
        0.0,
        ShoreSettings {
            period: 5.0,
            height: 1.0,
            ..Default::default()
        },
    );
    let mut demo = Demo {
        shore,
        world,
        window: None,
        renderer: None,
        start: Instant::now(),
        frames: 0,
        frame_limit,
        error: None,
    };
    EventLoop::new()?.run_app(&mut demo)?;
    if let Some(error) = demo.error {
        return Err(io::Error::other(error).into());
    }
    Ok(())
}
