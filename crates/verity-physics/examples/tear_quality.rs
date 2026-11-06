//! How tears behave: do they follow a sheet's threads, and does a tear cost the
//! same however fine the mesh?

use glam::{Quat, Vec3};
use verity_core::{Entity, EntityHandle, Transform, World};
use verity_physics::*;

const DT: f32 = 1.0 / 60.0;

fn spawn(
    world: &mut World,
    physics: &mut PhysicsWorld,
    at: Vec3,
    body: PhysicsBody,
) -> EntityHandle {
    let entity = world.spawn(Entity {
        transform: Transform {
            translation: at.to_array(),
            ..Default::default()
        },
        ..Entity::default()
    });
    physics.insert(world, entity, body).unwrap();
    entity
}

fn ball(radius: f32, density: f32) -> PhysicsBody {
    let mut body = PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius });
    body.material.density = density;
    body
}

/// Pull a sheet apart slowly by a bar along its bottom edge, until it parts:
/// its tear, and how many steps that took.
fn pulled(material: SoftMaterial, warp_degrees: f32, cells: u32) -> (Vec<(Vec3, Vec3)>, u32) {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let center = Vec3::new(0.0, 0.0, 4.0);
    let mut cloth = SoftBody::cloth(center, Quat::IDENTITY, [2.0, 2.0], [cells, cells], material);
    let (sin, cos) = warp_degrees.to_radians().sin_cos();
    cloth.set_warp(Vec3::new(cos, sin, 0.0));
    cloth.pin(|p| p.y > 0.99, f32::INFINITY);
    cloth.nick(center);
    let shape = Collider::Cuboid {
        half_extents: Vec3::new(1.2, 0.02, 0.02),
    };
    let mut bar = PhysicsBody::new(RigidBody::Kinematic, shape);
    bar.velocity.linear = Vec3::NEG_Y * 0.05;
    let bar = spawn(&mut world, &mut physics, Vec3::new(0.0, -1.0, 4.0), bar);
    let id = physics.add_soft_body(cloth);
    physics
        .pin_soft_body(&world, id, |p| p.y < -0.99, bar, f32::INFINITY)
        .unwrap();
    let settings = PhysicsSettings {
        gravity: Vec3::ZERO,
        ..Default::default()
    };
    for step in 0..3000 {
        physics.step(&mut world, &settings, DT);
        if physics.soft_body(id).unwrap().pieces() > 1 {
            return (physics.soft_body(id).unwrap().tear_path(), step);
        }
    }
    (physics.soft_body(id).unwrap().tear_path(), 3000)
}

/// A tear's overall direction in degrees from X, how straight it runs (1 is a
/// line), and its length.
fn direction(path: &[(Vec3, Vec3)]) -> (f32, f32, f32) {
    let (mut cos, mut sin, mut total) = (0.0, 0.0, 0.0);
    for &(a, b) in path {
        let edge = b - a;
        let length = edge.length();
        let angle = 2.0 * edge.y.atan2(edge.x);
        cos += length * angle.cos();
        sin += length * angle.sin();
        total += length;
    }
    let total = f32::max(total, 1.0e-6);
    let degrees = 0.5 * f32::atan2(sin, cos).to_degrees();
    (degrees, (cos * cos + sin * sin).sqrt() / total, total)
}

/// How fast a ball leaves a sheet it fell through, and how long the tear is.
fn punched(toughness: f32, cells: u32) -> (f32, usize) {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let material = SoftMaterial {
        toughness,
        strength: Strength::brittle(1500.0),
        ..SoftMaterial::cloth()
    };
    let center = Vec3::new(0.0, 0.0, 4.0);
    let mut cloth = SoftBody::cloth(center, Quat::IDENTITY, [3.0, 3.0], [cells, cells], material);
    cloth.pin(|p| p.x.abs() > 1.49 || p.y.abs() > 1.49, f32::INFINITY);
    let id = physics.add_soft_body(cloth);
    let ball = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 6.0),
        ball(0.25, 400.0),
    );
    let mut lowest = 0.0f32;
    for _ in 0..120 {
        physics.step(&mut world, &PhysicsSettings::default(), DT);
        let z = world.get(ball).unwrap().transform.translation[2];
        if z < 2.5 && lowest == 0.0 {
            lowest = physics.get(ball).unwrap().velocity.linear.z;
        }
    }
    (lowest.abs(), physics.soft_body(id).unwrap().tears())
}

/// Draw a tear, for looking at.
fn map(path: &[(Vec3, Vec3)]) {
    let (width, height) = (60usize, 22usize);
    let mut rows = vec![vec![b'.'; width]; height];
    for &(a, b) in path {
        for k in 0..=12 {
            let p = a + (b - a) * (k as f32 / 12.0);
            let x = (p.x + 1.0) * 0.5 * (width - 1) as f32;
            let y = (1.0 - p.y) * 0.5 * (height - 1) as f32;
            if let Some(cell) = rows.get_mut(y as usize).and_then(|r| r.get_mut(x as usize)) {
                *cell = b'#';
            }
        }
    }
    for row in rows {
        println!("{}", String::from_utf8_lossy(&row));
    }
}

fn main() {
    // Keep the cloth preset's strength and toughness in proportion: the zone
    // where a tear's bonds hold is stiffness x toughness / strength squared, so
    // weakening only one spreads it across the sheet and no stress gathers.
    let weak = |grain: f32, weave: f32| SoftMaterial {
        grain,
        weave,
        ..SoftMaterial::cloth()
    };
    println!("Pulled along Y, warp at 30 degrees: each tear's direction from X");
    for (name, grain, weave) in [
        ("felt", 1.0, 0.0),
        ("plain weave", 1.0, 1.0),
        ("strong grain", 0.3, 0.0),
    ] {
        for cells in [24, 40] {
            let (path, steps) = pulled(weak(grain, weave), 30.0, cells);
            let (degrees, straight, length) = direction(&path);
            if cells == 40 {
                map(&path);
            }
            println!(
                "{name:13} {cells:3} cells  {degrees:6.1} deg  straightness {straight:.2}  length {length:5.2} m  {} cuts  parted at step {steps}",
                path.len(),
            );
        }
    }
    println!("A ball through a pinned sheet: its speed below, at each resolution");
    for toughness in [0.0, 30.0, 120.0] {
        for cells in [32, 64, 96] {
            let (speed, tears) = punched(toughness, cells);
            println!("toughness {toughness:5}  {cells:3} cells  {speed:6.3} m/s  {tears:4} cuts");
        }
    }
}
