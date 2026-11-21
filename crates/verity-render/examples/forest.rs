//! A procedural forest for Matrix: clusters up close, voxels further out and
//! imposters in the distance, with frame times and selection counts.
//!
//! `cargo run --release -p verity-render --example forest -- [options]`
//!
//! - `--trees N`, `--leaves N`: how many, and leaves per tree
//! - `--exact`: draw every triangle instead, to compare
//! - `--debug`: color by tier: green, blue and yellow for cluster levels, orange
//!   for voxels, purple for imposters
//! - `--shadows M`: how far shadows reach, in meters
//! - `--model file.glb`, `--spacing M`: plant a model instead of the tree
//! - `--view ex ey ez tx ty tz`: hold the camera still
use glam::{Mat4, Quat, Vec3};
use std::{
    error::Error,
    sync::Arc,
    time::{Duration, Instant},
};
use verity_core::{Cloudscape, Clouds, World};
use verity_render::{
    asset::{Draw, MaterialData, Primitive, SceneData},
    Renderer, Sun, Vertex,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

/// A small deterministic generator, so runs compare.
struct Random(u32);

impl Random {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0 as f32 / u32::MAX as f32
    }

    fn signed(&mut self) -> f32 {
        self.next() * 2.0 - 1.0
    }
}

fn vertex(position: Vec3, normal: Vec3, color: [f32; 4]) -> Vertex {
    Vertex {
        position: position.extend(0.0).to_array(),
        normal: normal.extend(0.0).to_array(),
        tangent: [0.0; 4],
        color,
    }
}

fn add_primitive(scene: &mut SceneData, vertices: Vec<Vertex>, indices: Vec<u32>, material: u32) {
    scene.primitives.push(Primitive {
        first_index: scene.indices.len() as u32,
        index_count: indices.len() as u32,
        base_vertex: scene.vertices.len() as u32,
        material,
    });
    scene.draws.push(Draw {
        transform: Mat4::IDENTITY,
        primitive: scene.primitives.len() as u32 - 1,
    });
    scene.vertices.extend(vertices);
    scene.indices.extend(indices);
}

fn material(double_sided: bool, roughness: f32) -> MaterialData {
    MaterialData {
        double_sided,
        roughness,
        metallic: 0.0,
        ..MaterialData::default()
    }
}

/// A tapered trunk and a leaf cloud in a few lobes, 12 m tall.
fn tree(scene: &mut SceneData, leaves: u32) {
    let mut random = Random(0x1234_5678);
    let bark = [0.16, 0.11, 0.07, 1.0];
    let (mut vertices, mut indices) = (Vec::new(), Vec::new());
    const SIDES: u32 = 12;
    for ring in 0..=8 {
        let height = ring as f32 * 1.1;
        let radius = 0.35 * (1.0 - ring as f32 / 10.0);
        for side in 0..=SIDES {
            let angle = side as f32 / SIDES as f32 * std::f32::consts::TAU;
            let normal = Vec3::new(angle.cos(), angle.sin(), 0.0);
            vertices.push(vertex(normal * radius + Vec3::Z * height, normal, bark));
        }
    }
    let row = SIDES + 1;
    for ring in 0..8 {
        for side in 0..SIDES {
            let a = ring * row + side;
            indices.extend([a, a + 1, a + row + 1, a, a + row + 1, a + row]);
        }
    }
    add_primitive(scene, vertices, indices, 0);

    let lobes: Vec<(Vec3, f32)> = (0..9)
        .map(|_| {
            let center = Vec3::new(
                random.signed() * 2.2,
                random.signed() * 2.2,
                7.0 + random.signed() * 2.8,
            );
            (center, 1.6 + random.next() * 1.0)
        })
        .collect();
    let (mut vertices, mut indices) = (Vec::new(), Vec::new());
    for leaf in 0..leaves {
        let (center, radius) = lobes[leaf as usize % lobes.len()];
        let offset = loop {
            let p = Vec3::new(random.signed(), random.signed(), random.signed());
            if p.length_squared() <= 1.0 {
                break p;
            }
        };
        let position = center + offset * radius;
        let normal = Vec3::new(random.signed(), random.signed(), random.next() + 0.2).normalize();
        let u = normal.cross(Vec3::new(0.31, 0.53, 0.79)).normalize();
        let v = normal.cross(u);
        let shade = 0.65 + random.next() * 0.5;
        let color = [0.1 * shade, 0.3 * shade, 0.06 * shade, 1.0];
        let base = vertices.len() as u32;
        for (a, b) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            vertices.push(vertex(position + (u * a + v * b) * 0.09, normal, color));
        }
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    add_primitive(scene, vertices, indices, 1);
}

fn scene(leaves: u32) -> SceneData {
    let mut scene = SceneData {
        vertices: Vec::new(),
        indices: Vec::new(),
        primitives: Vec::new(),
        draws: Vec::new(),
        materials: vec![
            material(false, 0.9),
            // Leaves let a third of their diffuse light through, yellow-green.
            MaterialData {
                transmission: [0.6, 0.8, 0.25, 0.35],
                ..material(true, 0.6)
            },
            material(false, 1.0),
        ],
        textures: Vec::new(),
        lights: Vec::new(),
        sun: None,
    };
    tree(&mut scene, leaves);
    let ground = [0.12, 0.13, 0.07, 1.0];
    let size = 4000.0;
    let vertices = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        .map(|(x, y)| vertex(Vec3::new(x * size, y * size, 0.0), Vec3::Z, ground))
        .to_vec();
    add_primitive(&mut scene, vertices, vec![0, 1, 2, 0, 2, 3], 2);
    scene
}

/// Trees on a jittered grid around the origin, leaving a clearing at the center.
fn placements(trees: u32, spacing: f32) -> Vec<Mat4> {
    let mut random = Random(0x9e37_79b9);
    let side = (trees as f32).sqrt().ceil() as i32;
    let mut transforms = Vec::with_capacity(trees as usize);
    for y in 0..side {
        for x in 0..side {
            if transforms.len() as u32 >= trees {
                return transforms;
            }
            let position = Vec3::new(
                (x - side / 2) as f32 * spacing + random.signed() * 4.0,
                (y - side / 2) as f32 * spacing + random.signed() * 4.0,
                0.0,
            );
            let scale = 0.8 + random.next() * 0.5;
            let spin = Quat::from_rotation_z(random.next() * std::f32::consts::TAU);
            transforms.push(Mat4::from_scale_rotation_translation(
                Vec3::splat(scale),
                spin,
                position,
            ));
        }
    }
    transforms
}

struct Options {
    trees: u32,
    leaves: u32,
    exact: bool,
    debug: bool,
    shadows: u32,
    spacing: u32,
    model: Option<String>,
    view: Option<(Vec3, Vec3)>,
}

struct Viewer {
    options: Options,
    world: World,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    start: Instant,
    frames: (u32, Instant),
}

impl Viewer {
    fn load(&self, renderer: &mut Renderer) -> Result<(), Box<dyn Error>> {
        let started = Instant::now();
        let stats = renderer.add_scene(scene(self.options.leaves))?;
        println!("{stats:?} cooked in {:.2?}", started.elapsed());
        let transforms = placements(self.options.trees, self.options.spacing as f32);
        // The first two draws are one tree; the original stays at the origin.
        if let Some(model) = &self.options.model {
            let first = renderer.model_draw_count();
            println!("{:?}", renderer.load_gltf(model, 1024)?);
            renderer.place_copies(first..renderer.model_draw_count(), &transforms)?;
        } else {
            renderer.place_copies(0..2, &transforms)?;
        }
        renderer.set_matrix_enabled(!self.options.exact);
        renderer.set_matrix_debug(self.options.debug);
        renderer.set_uncapped(true);
        renderer.set_profiling(true);
        renderer.set_shadow_distance(self.options.shadows as f32);
        renderer.set_clear_color([0.45, 0.6, 0.8, 1.0]);
        renderer.set_ambient(Vec3::new(0.35, 0.42, 0.55), Vec3::new(0.1, 0.1, 0.06));
        renderer.set_sun(Some(Sun {
            direction: Vec3::new(-0.45, -0.3, -0.85).normalize(),
            color: Vec3::new(1.0, 0.95, 0.85) * 3.0,
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
            .with_title("Verity Forest")
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
                self.frames = (0, Instant::now());
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
                let t = self.start.elapsed().as_secs_f32() * 0.05;
                let eye = Vec3::new(30.0 * t.cos(), 30.0 * t.sin(), 4.0);
                let target = Vec3::new(120.0 * (t + 0.6).cos(), 120.0 * (t + 0.6).sin(), 6.0);
                let (eye, target) = self.options.view.unwrap_or((eye, target));
                renderer.set_camera(eye, target, 60f32.to_radians());
                if let Err(error) = renderer.draw_world(&self.world) {
                    eprintln!("render: {error}");
                    event_loop.exit();
                }
                self.frames.0 += 1;
                let elapsed = self.frames.1.elapsed();
                if elapsed >= Duration::from_secs(2) {
                    let gpu = renderer.profile().and_then(|p| p.gpu_ms).map_or(String::new(), |ms| {
                        // Shadows and Resolve, in `GPU_STAGE_NAMES` order.
                        format!(" gpu: shadows {:.2} ms, resolve {:.2} ms", ms[2], ms[5])
                    });
                    println!(
                        "{:.2} ms/frame{gpu} {:?}",
                        elapsed.as_secs_f64() * 1000.0 / self.frames.0 as f64,
                        renderer.matrix_stats()
                    );
                    self.frames = (0, Instant::now());
                }
            }
            _ => {}
        }
        event_loop.set_control_flow(ControlFlow::Poll);
    }
}

fn number(args: &[String], name: &str, default: u32) -> u32 {
    args.iter()
        .position(|a| a == name)
        .and_then(|at| args.get(at + 1)?.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let view = args.iter().position(|a| a == "--view").and_then(|at| {
        let v: Vec<f32> = args
            .get(at + 1..at + 7)?
            .iter()
            .filter_map(|a| a.parse().ok())
            .collect();
        (v.len() == 6).then(|| (Vec3::new(v[0], v[1], v[2]), Vec3::new(v[3], v[4], v[5])))
    });
    // `--clouds cumulus` (or parkouring, stormbird, none): a cloudy sky with that voxel cloudscape.
    let mut world = World::new();
    if let Some(name) = args.iter().position(|a| a == "--clouds").and_then(|at| args.get(at + 1)) {
        let cloudscape = (Cloudscape::ALL.into_iter())
            .find(|kind| format!("{kind:?}").eq_ignore_ascii_case(name))
            .ok_or_else(|| format!("unknown cloudscape {name}"))?;
        world.lighting.clouds = Some(Clouds {
            cloudscape,
            ..Clouds::default()
        });
    }
    EventLoop::new()?.run_app(&mut Viewer {
        options: Options {
            trees: number(&args, "--trees", 10_000),
            leaves: number(&args, "--leaves", 20_000),
            exact: args.iter().any(|a| a == "--exact"),
            debug: args.iter().any(|a| a == "--debug"),
            shadows: number(&args, "--shadows", 1000),
            spacing: number(&args, "--spacing", 11),
            model: args
                .iter()
                .position(|a| a == "--model")
                .and_then(|at| args.get(at + 1).cloned()),
            view,
        },
        world,
        window: None,
        renderer: None,
        start: Instant::now(),
        frames: (0, Instant::now()),
    })?;
    Ok(())
}
