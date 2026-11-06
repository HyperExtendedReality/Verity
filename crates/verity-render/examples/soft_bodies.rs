//! The soft-bodies demo scene, simulated and drawn as deforming meshes.
//! Set `VERITY_FRAMES` to exit after that many frames.
use glam::Vec3;
use std::{
    error::Error,
    sync::Arc,
    time::{Duration, Instant},
};
use verity_core::Scene;
use verity_render::{DeformingMesh, Renderer};
use verity_runtime::Runtime;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

struct Demo {
    runtime: Runtime,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    last_frame: Instant,
    frames: u32,
    limit: u32,
}

impl Demo {
    fn meshes(&self) -> Vec<DeformingMesh> {
        (self.runtime.soft_meshes())
            .filter_map(|(entity, positions, triangles)| {
                let material = self.runtime.world.get(entity)?.material.clone();
                let indices = triangles.iter().flatten().copied().collect();
                Some(DeformingMesh::from_triangles(positions, indices, material))
            })
            .collect()
    }
}

impl ApplicationHandler for Demo {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title("Verity soft bodies")
            .with_inner_size(PhysicalSize::new(1280, 720));
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
                let (eye, target) = (Vec3::new(0.0, -15.0, 7.0), Vec3::new(0.0, 0.0, 2.5));
                renderer.set_camera(eye, target, 0.9);
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
        let next = self.last_frame + Duration::from_millis(16);
        if Instant::now() >= next {
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
        event_loop.set_control_flow(ControlFlow::WaitUntil(next));
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
                let now = Instant::now();
                self.runtime.advance(now.duration_since(self.last_frame));
                self.last_frame = now;
                let meshes = self.meshes();
                if let Some(renderer) = &mut self.renderer {
                    renderer.set_deforming_meshes(meshes);
                    if let Err(error) = renderer.draw_world(&self.runtime.world) {
                        eprintln!("render: {error}");
                        event_loop.exit();
                    }
                }
                self.frames += 1;
                if self.frames >= self.limit {
                    println!("drew {} frames", self.frames);
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let scene: Scene =
        serde_json::from_str(include_str!("../../../assets/scenes/soft-bodies.verity"))?;
    let mut runtime = Runtime::from_scene(&scene)?;
    // A soft body draws as its mesh, in place of the entity's cube.
    let soft: Vec<_> = runtime.soft_meshes().map(|(entity, ..)| entity).collect();
    for entity in soft {
        if let Some(object) = runtime.world.get_mut(entity) {
            *object.transform = verity_core::Transform {
                scale: [0.0; 3],
                ..object.transform.clone()
            };
        }
    }
    let limit = std::env::var("VERITY_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(u32::MAX);
    let mut demo = Demo {
        runtime,
        window: None,
        renderer: None,
        last_frame: Instant::now(),
        frames: 0,
        limit,
    };
    let event_loop = EventLoop::new()?;
    event_loop.run_app(&mut demo)?;
    Ok(())
}
