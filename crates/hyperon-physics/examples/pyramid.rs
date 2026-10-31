//! Times a box pyramid: `cargo run --release -p hyperon-physics --example pyramid [rows]`.

use glam::Vec3;
use hyperon_core::{Entity, Material, Transform, World};
use hyperon_physics::*;
use std::time::Instant;

fn main() {
    let rows: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let mut add = |world: &mut World, position: Vec3, body: PhysicsBody| {
        let entity = world.spawn(Entity {
            name: "box".into(),
            transform: Transform {
                translation: position.to_array(),
                ..Default::default()
            },
            material: Material::default(),
        });
        physics.insert(world, entity, body).unwrap();
    };
    let floor = Collider::Cuboid {
        half_extents: Vec3::new(100.0, 0.5, 100.0),
    };
    add(
        &mut world,
        Vec3::new(0.0, -0.5, 0.0),
        PhysicsBody::new(RigidBody::Static, floor),
    );
    let cube = Collider::Cuboid {
        half_extents: Vec3::splat(0.5),
    };
    for row in 0..rows {
        for i in 0..rows - row {
            let x = i as f32 - (rows - row) as f32 * 0.5;
            add(
                &mut world,
                Vec3::new(x * 1.01, 0.5 + row as f32, 0.0),
                PhysicsBody::new(RigidBody::Dynamic, cube),
            );
        }
    }

    let settings = PhysicsSettings::default();
    let steps = 300;
    let start = Instant::now();
    for _ in 0..steps {
        physics.step(&mut world, &settings, 1.0 / 60.0);
    }
    let elapsed = start.elapsed().as_secs_f64() * 1000.0 / steps as f64;
    let top = world
        .iter()
        .map(|e| e.transform.translation[1])
        .fold(0.0, f32::max);
    println!(
        "{} bodies, {} contacts: {elapsed:.2} ms/step, top at {top:.2} (built {:.2})",
        physics.body_count(),
        physics.contact_count(),
        rows as f32 - 0.5
    );
}
