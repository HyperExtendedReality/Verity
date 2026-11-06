//! Time per step of each kind of soft body, on one thread.

use glam::{Quat, Vec3};
use std::time::Instant;
use verity_core::World;
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

fn main() {
    time("cloth 20 x 20", cloth(20));
    time("cloth 40 x 40", cloth(40));
    time("cloth 64 x 64", cloth(64));
    time(
        "plate 12 x 12 x 1",
        solid(Vec3::new(1.2, 1.2, 0.06), [12, 12, 1]),
    );
    time(
        "car shell 8 x 3 x 2",
        solid(Vec3::new(1.4, 0.7, 0.45), [8, 3, 2]),
    );
    time(
        "car shell 16 x 6 x 4",
        solid(Vec3::new(1.4, 0.7, 0.45), [16, 6, 4]),
    );
}
