//! Traffic of detailed cars on lattices, a few crashing into a wall, deformed on
//! the GPU. It checks the GPU's vertices against the CPU's and fails if they
//! disagree. Set `VERITY_FRAMES` for its length and `VERITY_CARS` for its size.
use glam::{Mat4, Quat, Vec3};
use std::{error::Error, sync::Arc, time::Instant};
use verity_core::{Entity, EntityHandle, Material, Transform, World};
use verity_physics::{
    Collider, PhysicsBody, PhysicsSettings, PhysicsWorld, RigidBody, SoftBody, SoftId, SoftMaterial,
};
use verity_render::{Dent, EmbeddedInstanceId, EmbeddedModel, GpuMaterial, Renderer, Vertex};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

const DT: f32 = 1.0 / 60.0;
const SHELL: Vec3 = Vec3::new(2.2, 0.9, 0.7);
const CORE: Vec3 = Vec3::new(1.0, 0.7, 0.5);
const SPEED: f32 = 15.6;
const LANE: f32 = 6.0;
const CRASHING: usize = 4;
/// Squares along each side of each face of a car's mesh.
const DETAIL: usize = 48;
/// Share of a crush the mesh shows as sharp detail, and how wide it is. The
/// lattice has the rest: more would count the same dent twice.
const DENT_DEPTH: f32 = 0.12;
const DENT_RADIUS: f32 = 0.6;

/// A box of `DETAIL` squares a side, the shape of a shell.
fn car_mesh() -> (Vec<Vertex>, Vec<u32>) {
    let (mut vertices, mut indices) = (Vec::new(), Vec::new());
    for (normal, u_axis) in [
        (Vec3::X, Vec3::Y),
        (Vec3::NEG_X, Vec3::NEG_Y),
        (Vec3::Y, Vec3::NEG_X),
        (Vec3::NEG_Y, Vec3::X),
        (Vec3::Z, Vec3::X),
        (Vec3::NEG_Z, Vec3::NEG_X),
    ] {
        let v_axis = normal.cross(u_axis);
        let first = vertices.len() as u32;
        for j in 0..=DETAIL {
            for i in 0..=DETAIL {
                let (u, v) = (i as f32 / DETAIL as f32, j as f32 / DETAIL as f32);
                let p = (normal + u_axis * (2.0 * u - 1.0) + v_axis * (2.0 * v - 1.0)) * SHELL;
                vertices.push(Vertex {
                    position: [p.x, p.y, p.z, u],
                    normal: [normal.x, normal.y, normal.z, v],
                    tangent: [u_axis.x, u_axis.y, u_axis.z, 1.0],
                    color: [1.0; 4],
                });
            }
        }
        let row = DETAIL as u32 + 1;
        for j in 0..DETAIL as u32 {
            for i in 0..DETAIL as u32 {
                let a = first + j * row + i;
                indices.extend_from_slice(&[a, a + 1, a + row, a + 1, a + row + 1, a + row]);
            }
        }
    }
    (vertices, indices)
}

struct Car {
    entity: EntityHandle,
    soft: SoftId,
    instance: Option<EmbeddedInstanceId>,
    revision: u64,
    dents: Vec<Dent>,
}

struct Demo {
    world: World,
    physics: PhysicsWorld,
    cars: Vec<Car>,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    frames: u32,
    limit: u32,
    /// Dispatches when the last quarter began, and frame time spent.
    settled: u64,
    busy: f64,
    failed: Option<String>,
}

impl Demo {
    fn update(&mut self) -> Result<(), Box<dyn Error>> {
        let settings = PhysicsSettings {
            gravity: Vec3::ZERO,
            ..Default::default()
        };
        self.physics.step(&mut self.world, &settings, DT);
        let renderer = self.renderer.as_mut().ok_or("no renderer")?;
        for (index, car) in self.cars.iter_mut().enumerate() {
            let transform = &self.world.get(car.entity).expect("cars stay").transform;
            let position = Vec3::from_array(transform.translation);
            let rotation = Quat::from_array(transform.rotation).normalize();
            let instance = car.instance.ok_or("no instance")?;
            renderer.set_embedded_pose(
                instance,
                Mat4::from_rotation_translation(rotation, position),
            );
            let body = self.physics.soft_body_mut(car.soft).ok_or("no body")?;
            // Car 1 keeps to its lattice, to compare with the CPU.
            for impact in body.take_impacts().iter().filter(|_| index != 1) {
                car.dents.push(Dent {
                    point: impact.point,
                    normal: impact.normal,
                    depth: DENT_DEPTH * body.crush_depth(impact),
                    radius: DENT_RADIUS * impact.radius,
                });
            }
            if body.revision() != car.revision {
                car.revision = body.revision();
                let nodes: Vec<Vec3> = (body.positions().iter())
                    .map(|&p| rotation.conjugate() * (p - position))
                    .collect();
                renderer.set_embedded_shape(instance, &nodes, &car.dents)?;
            }
        }
        Ok(())
    }

    /// Compare the GPU's work with the CPU's, once the crashes are over.
    fn check(&mut self) -> Result<(), Box<dyn Error>> {
        let (rest, _) = car_mesh();
        let renderer = self.renderer.as_mut().ok_or("no renderer")?;
        let start = |p: Vec3, lane: usize| p + Vec3::new(0.0, lane as f32 * LANE, 3.0);
        let points: Vec<Vec3> = (rest.iter())
            .map(|v| start(Vec3::from_slice(&v.position[..3]), 0))
            .collect();

        let plain = &self.cars[1];
        let dented = &self.cars[0];
        let body = self.physics.soft_body(plain.soft).ok_or("no body")?;
        if body.permanent_strain() == 0.0 {
            return Err("car 1 never crashed".into());
        }
        let skin = {
            let built = self.physics.soft_body(dented.soft).ok_or("no body")?;
            built.embed(&points)
        };
        let transform = &self.world.get(plain.entity).ok_or("no car")?.transform;
        let position = Vec3::from_array(transform.translation);
        let rotation = Quat::from_array(transform.rotation).normalize();
        let expected: Vec<Vec3> = (body.deformed(&skin).iter())
            .map(|&p| rotation.conjugate() * (p - position))
            .collect();
        let instance = plain.instance.ok_or("no instance")?;
        let gpu = renderer.read_embedded_vertices(instance)?;
        let mut worst = 0.0f32;
        let mut moved = 0.0f32;
        for ((v, want), r) in gpu.iter().zip(&expected).zip(&rest) {
            let got = Vec3::from_slice(&v.position[..3]);
            worst = worst.max(got.distance(*want));
            moved = moved.max(got.distance(Vec3::from_slice(&r.position[..3])));
            let length = Vec3::from_slice(&v.normal[..3]).length();
            if !(0.99..1.01).contains(&length) {
                return Err(format!("bad normal of length {length}").into());
            }
            if v.position[3] != r.position[3] || v.normal[3] != r.normal[3] {
                return Err("texture coordinates changed".into());
            }
        }
        println!("GPU against CPU: {worst:.6} m, over a crush of {moved:.3} m");
        if worst > 1.0e-3 {
            return Err(format!("GPU vertices are {worst} m from the CPU's").into());
        }
        if moved < 0.1 {
            return Err(format!("the crash moved vertices only {moved} m").into());
        }

        let instance = dented.instance.ok_or("no instance")?;
        let sharp = renderer.read_embedded_vertices(instance)?;
        let detail = (sharp.iter().zip(&gpu)).fold(0.0f32, |d, (a, b)| {
            d.max(Vec3::from_slice(&a.position[..3]).distance(Vec3::from_slice(&b.position[..3])))
        });
        println!(
            "dent detail reaches {detail:.3} m from {} dents",
            dented.dents.len()
        );
        if dented.dents.is_empty() || detail < 0.005 {
            return Err("car 0 shows no dent detail".into());
        }

        let spent = renderer.embedded_dispatches() - self.settled;
        let frames = (self.limit / 4) as f32;
        let rate = spent as f32 / frames;
        println!(
            "{} cars: {rate:.1} dispatches a frame once settled, {:.2} ms a frame",
            self.cars.len(),
            self.busy * 1000.0 / self.frames as f64,
        );
        if rate > CRASHING as f32 {
            return Err("cars that are only driving were deformed".into());
        }
        Ok(())
    }
}

impl ApplicationHandler for Demo {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title("Verity embedded cars")
            .with_inner_size(PhysicalSize::new(1280, 720));
        let setup = (|| -> Result<(Arc<Window>, Renderer), Box<dyn Error>> {
            let window = Arc::new(event_loop.create_window(attributes)?);
            let mut renderer = Renderer::new(window.clone())?;
            renderer.set_camera(
                Vec3::new(32.0, -22.0, 14.0),
                Vec3::new(42.0, 12.0, 3.0),
                0.9,
            );
            let (vertices, indices) = car_mesh();
            let built = self.physics.soft_body(self.cars[0].soft).ok_or("no body")?;
            let points: Vec<Vec3> = (vertices.iter())
                .map(|v| Vec3::from_slice(&v.position[..3]) + Vec3::new(0.0, 0.0, 3.0))
                .collect();
            let model = renderer.add_embedded_model(EmbeddedModel {
                skin: built.embed(&points),
                rest_to_cell: built.cell_basis().ok_or("no grid")?,
                vertices,
                indices,
                material: GpuMaterial::plain(&Material {
                    base_color: [0.6, 0.05, 0.05],
                    metallic: 0.8,
                    roughness: 0.35,
                }),
            })?;
            let instances = renderer.add_embedded_instances(model, self.cars.len())?;
            for (car, instance) in self.cars.iter_mut().zip(instances) {
                car.instance = Some(instance);
                // The first frame must have a shape to draw.
                car.revision = u64::MAX;
            }
            Ok((window, renderer))
        })();
        match setup {
            Ok((window, renderer)) => {
                self.window = Some(window);
                self.renderer = Some(renderer);
            }
            Err(error) => {
                self.failed = Some(error.to_string());
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
                let began = Instant::now();
                let frame = self.update().and_then(|()| {
                    let renderer = self.renderer.as_mut().ok_or("no renderer")?;
                    renderer.draw_world(&self.world)
                });
                self.busy += began.elapsed().as_secs_f64();
                self.frames += 1;
                if self.frames == self.limit - self.limit / 4 {
                    self.settled = self
                        .renderer
                        .as_ref()
                        .map_or(0, Renderer::embedded_dispatches);
                }
                let result = if self.frames >= self.limit {
                    frame.and_then(|()| self.check())
                } else {
                    frame
                };
                if let Err(error) = result {
                    self.failed = Some(error.to_string());
                }
                if self.failed.is_some() || self.frames >= self.limit {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }
}

fn spawn(world: &mut World, position: Vec3, scale: [f32; 3], base_color: [f32; 3]) -> EntityHandle {
    world.spawn(Entity {
        name: "body".into(),
        transform: Transform {
            translation: position.to_array(),
            scale,
            ..Default::default()
        },
        material: Material {
            base_color,
            ..Material::default()
        },
        ..Entity::default()
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let setting = |name: &str, default: usize| {
        (std::env::var(name).ok())
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let (count, limit) = (setting("VERITY_CARS", 64), setting("VERITY_FRAMES", 300));
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    spawn(
        &mut world,
        Vec3::new(50.0, 100.0, 1.0),
        [400.0, 400.0, 1.0],
        [0.3, 0.3, 0.32],
    );
    // A wall across only the first lanes: the rest drive past.
    let reach = LANE * CRASHING as f32 * 0.5;
    let half = Vec3::new(0.5, reach, 3.0);
    let at = Vec3::new(40.0, reach - LANE * 0.5, 3.0);
    let wall = spawn(&mut world, at, (half * 2.0).to_array(), [0.5, 0.45, 0.4]);
    let collider = Collider::Cuboid { half_extents: half };
    physics.insert(&world, wall, PhysicsBody::new(RigidBody::Static, collider))?;
    let cars = (0..count)
        .map(|lane| {
            let start = Vec3::new(0.0, lane as f32 * LANE, 3.0);
            let collider = Collider::Cuboid { half_extents: CORE };
            let mut core = PhysicsBody::new(RigidBody::Dynamic, collider);
            core.material.density = 1200.0 / (8.0 * CORE.x * CORE.y * CORE.z);
            core.velocity.linear = Vec3::X * SPEED;
            let entity = spawn(&mut world, start, [0.0; 3], [0.0; 3]);
            physics.insert(&world, entity, core)?;
            let material = SoftMaterial::crumple_zone();
            let mut shell = SoftBody::lattice(start, Quat::IDENTITY, SHELL, [11, 4, 3], material);
            shell.set_velocity(Vec3::X * SPEED);
            let soft = physics.add_soft_body(shell);
            let inside = move |p: Vec3| (p - start).abs().cmple(CORE + 0.05).all();
            physics.pin_soft_body(&world, soft, inside, entity, f32::INFINITY)?;
            Ok(Car {
                entity,
                soft,
                instance: None,
                revision: 0,
                dents: Vec::new(),
            })
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let mut demo = Demo {
        world,
        physics,
        cars,
        window: None,
        renderer: None,
        frames: 0,
        limit: limit as u32,
        settled: 0,
        busy: 0.0,
        failed: None,
    };
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut demo)?;
    drop(demo.renderer.take());
    match demo.failed {
        Some(error) => Err(error.into()),
        None => Ok(()),
    }
}
