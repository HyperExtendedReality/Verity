//! A Box3D simulation rendered directly from Hyperon's runtime world.
use glam::Vec3;
use hyperon_box3d::{Collider, PhysicsBody, RigidBody};
use hyperon_core::{Entity, Material, PointLight, Transform};
use hyperon_render::Renderer;
use hyperon_runtime::Runtime;
use std::{
    error::Error,
    sync::Arc,
    time::{Duration, Instant},
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

struct Demo {
    runtime: Runtime,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    last_frame: Instant,
}

impl ApplicationHandler for Demo {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = match event_loop
            .create_window(Window::default_attributes().with_title("Hyperon Box3D"))
        {
            Ok(window) => Arc::new(window),
            Err(error) => {
                eprintln!("window: {error}");
                event_loop.exit();
                return;
            }
        };
        match Renderer::new(window.clone()) {
            Ok(renderer) => {
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
                if let Some(renderer) = &mut self.renderer {
                    if let Err(error) = renderer.draw_world(&self.runtime.world) {
                        eprintln!("render: {error}");
                        event_loop.exit();
                    }
                }
            }
            _ => {}
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut runtime = Runtime::new();
    let floor = runtime.world.spawn(Entity {
        name: "Floor".into(),
        transform: Transform {
            translation: [0.0, -0.5, 0.0],
            scale: [8.0, 1.0, 8.0],
            ..Transform::default()
        },
        material: Material {
            base_color: [0.25, 0.32, 0.38],
            ..Material::default()
        },
        ..Entity::default()
    });
    let cube = runtime.world.spawn(Entity {
        name: "Falling cube".into(),
        transform: Transform {
            translation: [0.0, 3.0, 0.0],
            ..Transform::default()
        },
        material: Material {
            base_color: [0.8, 0.35, 0.12],
            ..Material::default()
        },
        ..Entity::default()
    });
    runtime.physics.insert(
        &runtime.world,
        floor,
        PhysicsBody::new(
            RigidBody::Static,
            Collider::Cuboid {
                half_extents: Vec3::new(4.0, 0.5, 4.0),
            },
        ),
    )?;
    runtime.physics.insert(
        &runtime.world,
        cube,
        PhysicsBody::new(
            RigidBody::Dynamic,
            Collider::Cuboid {
                half_extents: Vec3::splat(0.5),
            },
        ),
    )?;
    runtime.world.lights.push(PointLight {
        position: [2.0, 4.0, 3.0],
        color: [1.0, 0.95, 0.85],
        intensity: 30.0,
        radius: 15.0,
    });
    EventLoop::new()?.run_app(&mut Demo {
        runtime,
        window: None,
        renderer: None,
        last_frame: Instant::now(),
    })?;
    Ok(())
}
