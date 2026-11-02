//! hyperon-physics against Box3D on the same Z-up scenes:
//! `cargo run --release -p hyperon-physics --example compare [scale]`.
//! Set `RAYON_NUM_THREADS=1` to compare single-threaded (the Box3D bridge uses one worker).

use glam::Vec3;
use hyperon_core::{Entity, EntityHandle, Transform, World};
use std::time::Instant;

const DT: f32 = 1.0 / 60.0;
const ACTIVE: usize = 60;
const GRAVITY: Vec3 = Vec3::new(0.0, 0.0, -9.81);

/// What both engines are asked to simulate.
enum Kind {
    Floor,
    Cube,
    Ball,
}

struct Scene {
    name: &'static str,
    bodies: Vec<(Kind, Vec3)>,
    steps: usize,
}

/// A 2D pyramid of unit cubes on the XZ plane: the classic stacking test.
fn pyramid(rows: usize) -> Scene {
    let mut bodies = vec![(Kind::Floor, Vec3::new(0.0, 0.0, -0.5))];
    for row in 0..rows {
        for i in 0..rows - row {
            let x = i as f32 - (rows - row) as f32 * 0.5;
            bodies.push((Kind::Cube, Vec3::new(x * 1.01, 0.0, 0.5 + row as f32)));
        }
    }
    Scene { name: "pyramid", bodies, steps: 300 }
}

/// Deterministic pseudo-random value in [-1, 1).
fn jitter(seed: usize) -> f32 {
    let x = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(17);
    (x.wrapping_mul(0xBF58_476D_1CE4_E5B9) >> 40) as f32 / (1u64 << 23) as f32 - 1.0
}

/// A block of cubes and balls dropped onto the floor: broadphase and piles.
/// Jitter breaks the grid's symmetry so it collapses into a real pile.
fn rain(side: usize) -> Scene {
    let mut bodies = vec![(Kind::Floor, Vec3::new(0.0, 0.0, -0.5))];
    for z in 0..side {
        for y in 0..side {
            for x in 0..side {
                let n = bodies.len() * 3;
                let p = Vec3::new(x as f32, y as f32, z as f32) * 1.2
                    + Vec3::new(-(side as f32) * 0.6, -(side as f32) * 0.6, 1.0)
                    + Vec3::new(jitter(n), jitter(n + 1), jitter(n + 2)) * 0.08;
                let shape = if (x + y + z) % 2 == 0 { Kind::Cube } else { Kind::Ball };
                bodies.push((shape, p));
            }
        }
    }
    Scene { name: "rain", bodies, steps: 300 }
}

fn spawn(world: &mut World, p: Vec3) -> EntityHandle {
    world.spawn(Entity {
        transform: Transform {
            translation: p.to_array(),
            ..Default::default()
        },
        ..Entity::default()
    })
}

/// Milliseconds per step and the highest body's height afterwards.
struct Result {
    /// The first second, while everything moves.
    active: f64,
    ms: f64,
    top: f32,
    lowest: f32,
}

fn measure(world: &World, steps: usize, start: Instant, active: f64) -> Result {
    let heights = || world.iter().skip(1).map(|e| e.transform.translation[2]);
    Result {
        active,
        ms: start.elapsed().as_secs_f64() * 1000.0 / steps as f64,
        top: heights().fold(f32::MIN, f32::max),
        lowest: heights().fold(f32::MAX, f32::min),
    }
}

fn hyperon(scene: &Scene) -> Result {
    use hyperon_physics::*;
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    for (shape, p) in &scene.bodies {
        let entity = spawn(&mut world, *p);
        let body = match shape {
            Kind::Floor => PhysicsBody::new(
                RigidBody::Static,
                Collider::Cuboid { half_extents: Vec3::new(100.0, 100.0, 0.5) },
            ),
            Kind::Cube => PhysicsBody::new(
                RigidBody::Dynamic,
                Collider::Cuboid { half_extents: Vec3::splat(0.5) },
            ),
            Kind::Ball => PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius: 0.5 }),
        };
        physics.insert(&world, entity, body).unwrap();
    }
    let settings = PhysicsSettings::default();
    let start = Instant::now();
    let mut active = 0.0;
    for step in 0..scene.steps {
        physics.step(&mut world, &settings, DT);
        if step == ACTIVE - 1 {
            active = start.elapsed().as_secs_f64() * 1000.0 / ACTIVE as f64;
        }
    }
    measure(&world, scene.steps, start, active)
}

fn box3d(scene: &Scene) -> Result {
    use hyperon_box3d::*;
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    for (shape, p) in &scene.bodies {
        let entity = spawn(&mut world, *p);
        let body = match shape {
            Kind::Floor => PhysicsBody::new(
                RigidBody::Static,
                Collider::Cuboid { half_extents: Vec3::new(100.0, 100.0, 0.5) },
            ),
            Kind::Cube => PhysicsBody::new(
                RigidBody::Dynamic,
                Collider::Cuboid { half_extents: Vec3::splat(0.5) },
            ),
            Kind::Ball => PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius: 0.5 }),
        };
        physics.insert(&world, entity, body).unwrap();
    }
    let settings = Box3dSettings { gravity: GRAVITY, ..Default::default() };
    let start = Instant::now();
    let mut active = 0.0;
    for step in 0..scene.steps {
        physics.step(&mut world, &settings, DT);
        if step == ACTIVE - 1 {
            active = start.elapsed().as_secs_f64() * 1000.0 / ACTIVE as f64;
        }
    }
    measure(&world, scene.steps, start, active)
}

fn main() {
    let scale: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    for scene in [pyramid(20 * scale), pyramid(40 * scale), rain(8 * scale), rain(14 * scale)] {
        // Best of three: other load on the machine only ever adds time.
        let best = |run: fn(&Scene) -> Result| {
            (0..3).map(|_| run(&scene)).reduce(|a, b| Result {
                active: a.active.min(b.active),
                ms: a.ms.min(b.ms),
                ..b
            })
        };
        let (h, b) = (best(hyperon).unwrap(), best(box3d).unwrap());
        println!(
            "{:8} {:5} bodies | active {:7.3} vs {:7.3} ms {:5.2}x | all {:7.3} vs {:7.3} ms {:5.2}x | top {:6.2} vs {:6.2} low {:5.2} vs {:5.2}",
            scene.name,
            scene.bodies.len() - 1,
            h.active, b.active, b.active / h.active,
            h.ms, b.ms, b.ms / h.ms,
            h.top, b.top, h.lowest, b.lowest,
        );
    }
}
