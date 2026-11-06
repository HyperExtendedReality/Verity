//! Time per step of each kind of soft body, on one thread.

use glam::{Quat, Vec3};
use std::time::Instant;
use verity_core::{Entity, Transform, World};
use verity_physics::*;

fn time(name: &str, body: SoftBody) {
    let (particles, mut world, mut physics) = (
        body.positions().len(),
        World::new(),
        PhysicsWorld::default(),
    );
    physics.add_soft_body(body);
    let settings = PhysicsSettings::default();
    for _ in 0..30 {
        physics.step(&mut world, &settings, 1.0 / 60.0);
    }
    let (steps, start) = (120, Instant::now());
    for _ in 0..steps {
        physics.step(&mut world, &settings, 1.0 / 60.0);
    }
    let ms = start.elapsed().as_secs_f64() * 1000.0 / steps as f64;
    println!("{name:24} {particles:6} particles {ms:7.3} ms/step");
}

fn cloth(cells: u32) -> SoftBody {
    let mut body = SoftBody::cloth(
        Vec3::ZERO,
        Quat::IDENTITY,
        [2.0, 2.0],
        [cells, cells],
        SoftMaterial::cloth(),
    );
    body.pin(|p| p.y > 0.99, f32::INFINITY);
    body
}

fn solid(half: Vec3, cells: [u32; 3]) -> SoftBody {
    let metal = SoftMaterial::sheet_metal();
    let mut body = SoftBody::lattice(Vec3::ZERO, Quat::IDENTITY, half, cells, metal);
    body.pin(|p| p.x < -half.x + 0.01, f32::INFINITY);
    body
}

/// Many cars driving in a line: shells on rigid cores, as in a traffic scene.
fn traffic(cars: usize) {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let half = Vec3::new(1.0, 0.7, 0.5);
    let shell = Vec3::new(2.2, 0.9, 0.7);
    let weightless = PhysicsSettings {
        gravity: Vec3::ZERO,
        ..Default::default()
    };
    for i in 0..cars {
        let start = Vec3::new(0.0, i as f32 * 5.0, 3.0);
        let mut core =
            PhysicsBody::new(RigidBody::Dynamic, Collider::Cuboid { half_extents: half });
        core.velocity.linear = Vec3::X * 15.0;
        let entity = world.spawn(Entity {
            transform: Transform {
                translation: start.to_array(),
                ..Default::default()
            },
            ..Entity::default()
        });
        physics.insert(&world, entity, core).unwrap();
        let material = SoftMaterial::crumple_zone();
        let mut body = SoftBody::lattice(start, Quat::IDENTITY, shell, [11, 4, 3], material);
        body.set_velocity(Vec3::X * 15.0);
        let id = physics.add_soft_body(body);
        let inside = move |p: Vec3| (p - start).abs().cmple(half + 0.05).all();
        physics
            .pin_soft_body(&world, id, inside, entity, f32::INFINITY)
            .unwrap();
    }
    let measure = |physics: &mut PhysicsWorld, world: &mut World, steps: usize| {
        let start = Instant::now();
        for _ in 0..steps {
            physics.step(world, &weightless, 1.0 / 60.0);
        }
        start.elapsed().as_secs_f64() * 1000.0 / steps as f64
    };
    let solving = measure(&mut physics, &mut world, 10);
    measure(&mut physics, &mut world, 60);
    let riding = measure(&mut physics, &mut world, 120);
    println!("{cars} cars: {solving:.3} ms/step solving, {riding:.3} ms/step riding");
}

fn main() {
    traffic(100);
    time("cloth 20 x 20", cloth(20));
    time("cloth 40 x 40", cloth(40));
    time("cloth 64 x 64", cloth(64));
    time(
        "plate 12 x 12 x 1",
        solid(Vec3::new(1.2, 1.2, 0.06), [12, 12, 1]),
    );
    let flesh = |cells: u32| {
        let half = Vec3::splat(0.5);
        let block = SoftBody::lattice(
            Vec3::ZERO,
            Quat::IDENTITY,
            half,
            [cells; 3],
            SoftMaterial::flesh(),
        );
        let mut block = block.finite_elements();
        block.pin(|p| p.z > 0.49, f32::INFINITY);
        block
    };
    time("flesh 6 x 6 x 6", flesh(6));
    time("flesh 12 x 12 x 12", flesh(12));
    time(
        "car shell 8 x 3 x 2",
        solid(Vec3::new(1.4, 0.7, 0.45), [8, 3, 2]),
    );
    time(
        "car shell 16 x 6 x 4",
        solid(Vec3::new(1.4, 0.7, 0.45), [16, 6, 4]),
    );
}
