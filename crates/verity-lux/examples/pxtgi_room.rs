//! PXTGI in a physics room: colored walls, falling boxes, a ball and a hanging
//! cloth, all traced through the same `RayScene` the simulation fills.
//!
//! `cargo run --release -p verity-lux --example pxtgi_room`
//!
//! `--sponza` adds the Sponza scene as static trace meshes. For measurements:
//! `--frames=N` quits after N frames, `--dump=file.pfm` saves the last one,
//! `--debug=hits|bounce|reflections|world|position|roundtrip` picks a view,
//! `--no-gi` runs the plain renderer, `--software` traces without ray queries, `--lights=N` spreads the light over N, `--extinction` puts the light out at
//! frame 240 and prints how fast its bounce light goes, and `--reference`
//! path traces the bounce light on the CPU to compare with `--debug=bounce`.
use glam::{Quat, Vec3};
use std::{f32::consts::PI, sync::Arc, time::Instant};
use verity_core::{Entity, EntityHandle, Material, PointLight, Transform, World};
use verity_lux::pxtgi::{Capture, Pxtgi, PxtgiDebug, PxtgiSettings};
use verity_physics::{
    Collider, PhysicsBody, PhysicsSettings, PhysicsWorld, RayScene, RaySource, RigidBody, SoftBody,
    SoftMaterial,
};
use verity_render::{DeformingMesh, Renderer, Sun};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

const STEP: f32 = 1.0 / 60.0;
/// When `--extinction` puts the light out.
const LIGHT_OUT: u64 = 240;

fn flag(name: &str) -> Option<String> {
    let prefix = format!("--{name}");
    std::env::args().find_map(|a| {
        let rest = a.strip_prefix(&prefix)?;
        (rest.is_empty() || rest.starts_with('=')).then(|| rest.trim_start_matches('=').to_owned())
    })
}

struct Room {
    world: World,
    physics: PhysicsWorld,
    settings: PhysicsSettings,
    ball: EntityHandle,
}

fn body(room: &mut Room, at: Vec3, half: Vec3, color: [f32; 3], kind: RigidBody) -> EntityHandle {
    let entity = room.world.spawn(Entity {
        name: "box".into(),
        transform: Transform {
            translation: at.to_array(),
            // Unit cubes are drawn; the collider carries the real size.
            scale: (half * 2.0).to_array(),
            ..Transform::default()
        },
        material: Material {
            base_color: color,
            roughness: 0.6,
            ..Material::default()
        },
        ..Entity::default()
    });
    let collider = Collider::Cuboid { half_extents: half };
    (room.physics)
        .insert(&room.world, entity, PhysicsBody::new(kind, collider))
        .expect("the entity is alive");
    entity
}

fn room() -> Room {
    let mut world = World::new();
    let ball = world.spawn(Entity {
        name: "ball".into(),
        transform: Transform {
            translation: [2.2, -0.5, 3.0],
            // Drawn by `meshes`, not as a cube.
            scale: [0.0; 3],
            ..Transform::default()
        },
        ..Entity::default()
    });
    let mut room = Room {
        world,
        physics: PhysicsWorld::default(),
        settings: PhysicsSettings::default(),
        ball,
    };
    let wall = |room: &mut Room, at, half, color| {
        body(room, at, half, color, RigidBody::Static);
    };
    let (white, red, green) = ([0.75; 3], [0.75, 0.08, 0.06], [0.1, 0.65, 0.12]);
    wall(
        &mut room,
        Vec3::new(0.0, 0.0, -0.25),
        Vec3::new(5.0, 5.0, 0.25),
        white,
    );
    wall(
        &mut room,
        Vec3::new(0.0, 0.0, 5.25),
        Vec3::new(5.0, 5.0, 0.25),
        white,
    );
    wall(
        &mut room,
        Vec3::new(0.0, 5.25, 2.5),
        Vec3::new(5.0, 0.25, 2.75),
        white,
    );
    wall(
        &mut room,
        Vec3::new(-5.25, 0.0, 2.5),
        Vec3::new(0.25, 5.5, 2.75),
        red,
    );
    wall(
        &mut room,
        Vec3::new(5.25, 0.0, 2.5),
        Vec3::new(0.25, 5.5, 2.75),
        green,
    );
    wall(
        &mut room,
        Vec3::new(0.0, -5.25, 2.5),
        Vec3::new(5.0, 0.25, 2.75),
        white,
    );
    for (i, color) in [[0.9, 0.55, 0.1], [0.15, 0.3, 0.8], [0.8, 0.8, 0.75]]
        .into_iter()
        .enumerate()
    {
        let at = Vec3::new(-1.6 + 1.5 * i as f32, 1.0, 1.5 + 2.0 * i as f32);
        let block = body(&mut room, at, Vec3::splat(0.6), color, RigidBody::Dynamic);
        let spin = Quat::from_rotation_z(0.4 * i as f32);
        room.world.get_mut(block).unwrap().transform.rotation = spin.to_array();
    }
    let sphere = Collider::Sphere { radius: 0.7 };
    (room.physics)
        .insert(
            &room.world,
            ball,
            PhysicsBody::new(RigidBody::Dynamic, sphere),
        )
        .expect("the entity is alive");
    let cloth = SoftBody::cloth(
        Vec3::new(0.0, 3.2, 4.2),
        Quat::from_rotation_x(PI / 2.0),
        [4.0, 1.4],
        [32, 12],
        SoftMaterial::cloth(),
    );
    let id = room.physics.add_soft_body(cloth);
    (room.physics.soft_body_mut(id).unwrap()).pin(|p| p.z > 4.85, 1.0);
    // `--lights=N` shares the light's power among N in a grid under the
    // ceiling, past the 8 PXTGI shades exactly.
    let count = flag("lights").and_then(|n| n.parse().ok()).unwrap_or(1u32).max(1);
    let side = (count as f32).sqrt().ceil() as u32;
    for i in 0..count {
        let at = |k: u32| (k as f32 + 0.5) / side as f32 * 8.0 - 4.0;
        let position = if count == 1 { [0.0, -1.0, 4.4] } else { [at(i % side), at(i / side), 4.4] };
        room.world.lights.push(PointLight {
            position,
            color: [1.0, 0.93, 0.82],
            intensity: 30.0 / count as f32,
            radius: 16.0,
        });
    }
    room
}

/// Soft bodies and the ball, which the renderer draws as meshes.
fn meshes(room: &Room) -> Vec<DeformingMesh> {
    let cloth = Material {
        base_color: [0.7, 0.2, 0.55],
        ..Material::default()
    };
    let mut out: Vec<_> = (room.physics.soft_bodies())
        .map(|(_, soft)| {
            let indices = soft.triangles().iter().flatten().copied().collect();
            DeformingMesh::from_triangles(soft.positions(), indices, cloth.clone())
        })
        .collect();
    let (rows, columns) = (16u32, 24u32);
    let pose = room.world.get(room.ball).unwrap().transform;
    let center = Vec3::from_array(pose.translation);
    let positions: Vec<_> = (0..=rows)
        .flat_map(|r| (0..=columns).map(move |c| (r, c)))
        .map(|(r, c)| {
            let (theta, phi) = (
                PI * r as f32 / rows as f32,
                2.0 * PI * c as f32 / columns as f32,
            );
            center
                + 0.7
                    * Vec3::new(
                        theta.sin() * phi.cos(),
                        theta.sin() * phi.sin(),
                        theta.cos(),
                    )
        })
        .collect();
    let at = |r: u32, c: u32| r * (columns + 1) + c;
    let indices = (0..rows)
        .flat_map(|r| (0..columns).map(move |c| (r, c)))
        .flat_map(|(r, c)| {
            [
                at(r, c),
                at(r + 1, c),
                at(r, c + 1),
                at(r, c + 1),
                at(r + 1, c),
                at(r + 1, c + 1),
            ]
        })
        .collect();
    let ball = Material {
        base_color: [0.95, 0.9, 0.8],
        roughness: 0.12,
        ..Material::default()
    };
    out.push(DeformingMesh::from_triangles(&positions, indices, ball));
    out
}

fn mean_luminance(capture: &Capture) -> f32 {
    let sum: f32 = (capture.pixels.iter())
        .map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2])
        .sum();
    sum / capture.pixels.len() as f32
}

/// Portable float map: a header, then RGB rows from the bottom.
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

/// Points on the walls, floor and ceiling, each with the way it faces.
const PROBES: [(&str, Vec3, Vec3); 7] = [
    ("floor by red", Vec3::new(-4.2, -0.5, 0.0), Vec3::Z),
    ("floor by green", Vec3::new(4.2, -0.5, 0.0), Vec3::Z),
    ("floor middle", Vec3::new(0.0, -0.8, 0.0), Vec3::Z),
    ("ceiling", Vec3::new(0.0, -0.5, 5.0), Vec3::NEG_Z),
    ("back left", Vec3::new(-3.0, 5.0, 3.5), Vec3::NEG_Y),
    ("back right", Vec3::new(3.0, 5.0, 3.5), Vec3::NEG_Y),
    ("red wall", Vec3::new(-5.0, 0.0, 3.0), Vec3::X),
];

/// What the renderer's BRDF leaves for diffuse after specular takes its share
/// on these rough dielectrics: its `1 - specular_albedo`, near normal incidence.
const DIFFUSE_SHARE: f32 = 0.95;

/// A deterministic generator, so references repeat.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
    fn cosine(&mut self, n: Vec3) -> Vec3 {
        let (t, b) = n.any_orthonormal_pair();
        let (u, phi) = (self.unit(), 2.0 * PI * self.unit());
        let r = u.sqrt();
        (t * (r * phi.cos()) + b * (r * phi.sin()) + n * (1.0 - u).sqrt()).normalize()
    }
}

/// The light arriving at a surface point along cosine-weighted rays, as PXTGI
/// holds it, by plain path tracing against the same scene.
fn reference(
    scene: &RayScene,
    room: &Room,
    at: Vec3,
    normal: Vec3,
    paths: u32,
    rng: &mut Rng,
) -> Vec3 {
    let albedo = |item: usize| match scene.items()[item].source {
        RaySource::Body(e) => {
            let m = room.world.get(e).unwrap().material;
            Vec3::from_array(m.base_color) * (1.0 - m.metallic) * DIFFUSE_SHARE
        }
        _ => Vec3::splat(0.6),
    };
    let direct = |x: Vec3, n: Vec3| {
        let mut sum = Vec3::ZERO;
        for light in &room.world.lights {
            let to = Vec3::from_array(light.position) - x;
            let (d2, d) = (to.length_squared().max(1e-2), to.length());
            let l = to / d;
            let w = (1.0 - (d2 / (light.radius * light.radius)).powi(2)).clamp(0.0, 1.0);
            let origin = x + n * 0.02;
            if n.dot(l) > 0.0 && w > 0.0 && scene.ray(origin, l, d - 0.05).is_none() {
                sum +=
                    Vec3::from_array(light.color) * (light.intensity / PI) * n.dot(l) * w * w / d2;
            }
        }
        sum
    };
    let mut total = Vec3::ZERO;
    for _ in 0..paths {
        let (mut x, mut n, mut weight) = (at, normal, Vec3::ONE);
        for _ in 0..24 {
            let d = rng.cosine(n);
            let Some(hit) = scene.ray(x + n * 0.03, d, 40.0) else {
                break;
            };
            x = x + n * 0.03 + d * hit.t;
            n = hit.normal;
            weight *= albedo(hit.item);
            total += weight * direct(x, n);
        }
    }
    total / paths as f32
}

/// Where a world point lands in a capture, and the mean there, if it is seen.
fn seen_at(capture: &Capture, clip: glam::Mat4, at: Vec3) -> Option<Vec3> {
    let c = clip.project_point3(at);
    let (x, y) = (
        ((c.x * 0.5 + 0.5) * capture.width as f32) as i32,
        ((c.y * 0.5 + 0.5) * capture.height as f32) as i32,
    );
    let (mut sum, mut count) = (Vec3::ZERO, 0.0);
    for v in y - 5..=y + 5 {
        for u in x - 5..=x + 5 {
            if u >= 0 && v >= 0 && u < capture.width as i32 && v < capture.height as i32 {
                let [r, g, b, _] = capture.pixels[(v as u32 * capture.width + u as u32) as usize];
                sum += Vec3::new(r, g, b);
                count += 1.0;
            }
        }
    }
    (count > 0.0).then(|| sum / count)
}

const EYE: Vec3 = Vec3::new(0.0, -4.8, 2.4);
const LOOK: Vec3 = Vec3::new(0.0, 2.0, 1.9);
const FOV_DEGREES: f32 = 72.0;

fn compare(capture: &Capture, scene: &RayScene, room: &Room) {
    let aspect = capture.width as f32 / capture.height as f32;
    // Y flipped, as a capture's rows run from the top.
    let clip = glam::Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0))
        * glam::Mat4::perspective_infinite_reverse_rh(FOV_DEGREES.to_radians(), aspect, 0.1)
        * glam::Mat4::look_at_rh(EYE, LOOK, Vec3::Z);
    let mut rng = Rng(1);
    let (mut error, mut truth) = (0.0, 0.0);
    println!("{:16} {:>20} {:>20}", "", "path traced", "PXTGI");
    for (name, at, normal) in PROBES {
        let Some(got) = seen_at(capture, clip, at + normal * 0.01) else {
            println!("{name:16} off screen");
            continue;
        };
        let want = reference(scene, room, at, normal, 40_000, &mut rng);
        println!(
            "{name:16} {:>20} {:>20}",
            format!("{want:.3}"),
            format!("{got:.3}")
        );
        error += (got - want).abs().element_sum();
        truth += want.element_sum();
    }
    println!("mean error {:.1}% of the light", 100.0 * error / truth);
}

struct Viewer {
    room: Room,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    frames: u64,
    limit: Option<u64>,
    timer: Instant,
    /// Mean brightness just before `--extinction` put the light out.
    lit: Option<f32>,
}

impl Viewer {
    fn frame(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let extinction = flag("extinction").is_some();
        let room = &mut self.room;
        room.physics.step(&mut room.world, &room.settings, STEP);
        if extinction && self.frames == LIGHT_OUT {
            // `--extinction=half` dims it instead of putting it out.
            match flag("extinction").as_deref() {
                Some("half") => room.world.lights[0].intensity *= 0.5,
                _ => room.world.lights.clear(),
            }
        }
        let renderer = self.renderer.as_mut().unwrap();
        renderer.set_deforming_meshes(meshes(room));
        renderer.set_camera(EYE, LOOK, FOV_DEGREES.to_radians());
        let pxtgi = renderer.extension_mut::<Pxtgi>();
        if let Some(pxtgi) = pxtgi {
            pxtgi.sync(&room.physics, &room.world);
            // Every other frame is as often as a capture can come back.
            let near_out = self.frames + 2 >= LIGHT_OUT && self.frames < LIGHT_OUT + 40;
            let last = self.limit.is_some_and(|limit| self.frames + 4 == limit);
            if (extinction && near_out) || last {
                pxtgi.request_capture();
            }
            if let Some(capture) = pxtgi.take_capture() {
                let mean = mean_luminance(&capture);
                if extinction {
                    if self.frames < LIGHT_OUT + 2 {
                        self.lit = Some(mean);
                    } else if let Some(lit) = self.lit {
                        let age = self.frames - LIGHT_OUT - 2;
                        println!(
                            "{age:3} frames dark: {:5.1}% of the lit room",
                            100.0 * mean / lit
                        );
                    }
                }
                if flag("reference").is_some() {
                    compare(&capture, pxtgi.ray_scene(), room);
                }
                if let Some(path) = flag("dump") {
                    write_pfm(&path, &capture)?;
                    println!("mean luminance {mean:.4}, saved {path}");
                }
            }
        }
        renderer.draw_world(&room.world)?;
        self.frames += 1;
        if self.frames.is_multiple_of(120) {
            let ms = self.timer.elapsed().as_secs_f64() * 1000.0 / 120.0;
            let gi = renderer.extension_mut::<Pxtgi>().and_then(|p| p.gpu_time());
            println!(
                "frame {}: {ms:.2} ms/frame, PXTGI {gi:.2?} GPU",
                self.frames
            );
            self.timer = Instant::now();
        }
        Ok(())
    }
}

impl ApplicationHandler for Viewer {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title("Verity PXTGI")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
        let window = Arc::new(event_loop.create_window(attributes).expect("window"));
        let result = Renderer::new(window.clone()).and_then(|mut renderer| {
            renderer.set_ambient(Vec3::splat(0.0), Vec3::splat(0.0));
            if flag("no-gi").is_none() {
                let debug = match flag("debug").as_deref() {
                    Some("roundtrip") => PxtgiDebug::RoundTrip,
                    Some("world") => PxtgiDebug::WorldOnly,
                    Some("hits") => PxtgiDebug::WorldHits,
                    Some("position") => PxtgiDebug::WorldPosition,
                    Some("bounce") => PxtgiDebug::BounceLight,
                    Some("reflections") => PxtgiDebug::Reflections,
                    _ => PxtgiDebug::Off,
                };
                let settings = PxtgiSettings {
                    debug,
                    hardware_rays: flag("software").is_none(),
                    ..PxtgiSettings::default()
                };
                renderer.set_extension(Box::new(Pxtgi::new(settings)))?;
            }
            if flag("sponza").is_some() {
                let packed =
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/packed");
                let start = Instant::now();
                renderer.load_gltf(packed.join("NewSponza_Main.glb"), 512)?;
                println!("Sponza loaded in {:.1} s", start.elapsed().as_secs_f32());
                renderer.set_sun(Some(Sun {
                    direction: Vec3::new(-0.3, -0.2, -1.0).normalize(),
                    color: Vec3::splat(3.0),
                }));
            }
            Ok(renderer)
        });
        match result {
            Ok(renderer) => {
                self.renderer = Some(renderer);
                self.window = Some(window);
                self.timer = Instant::now();
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
            WindowEvent::RedrawRequested if self.renderer.is_some() => {
                if let Err(error) = self.frame() {
                    eprintln!("render: {error}");
                    event_loop.exit();
                }
                if self.limit.is_some_and(|limit| self.frames >= limit) {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let limit = flag("frames").and_then(|n| n.parse().ok());
    EventLoop::new()?.run_app(&mut Viewer {
        room: room(),
        window: None,
        renderer: None,
        frames: 0,
        limit,
        timer: Instant::now(),
        lit: None,
    })?;
    Ok(())
}
