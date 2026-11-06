//! How closely a cloth follows its material, whatever its mesh. A sheet hanging
//! under its own weight stretches by `density g length² / (2 stiffness)`, at
//! any resolution.

use glam::{Quat, Vec3};
use verity_core::World;
use verity_physics::*;

/// The stretch of a 1 m square sheet of `density` hanging from its top edge.
fn stretch(cells: u32, density: f32) -> f32 {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let mut material = SoftMaterial::cloth();
    material.strength = Strength::default();
    material.density = density;
    material.poisson = 0.0;
    material.damping = 4.0;
    let upright = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
    let center = Vec3::new(0.0, 0.0, 5.0);
    let mut cloth = SoftBody::cloth(center, upright, [1.0, 1.0], [cells, cells], material);
    cloth.pin(|p| p.z > 5.49, f32::INFINITY);
    let id = physics.add_soft_body(cloth);
    let settings = PhysicsSettings::default();
    for _ in 0..900 {
        physics.step(&mut world, &settings, 1.0 / 60.0);
    }
    let body = physics.soft_body(id).unwrap();
    let (top, bottom) =
        (body.positions().iter()).fold((f32::MIN, f32::MAX), |(t, b), p| (t.max(p.z), b.min(p.z)));
    top - bottom - 1.0
}

fn main() {
    let material = SoftMaterial::cloth();
    let density = 40.0;
    let expected = density * 9.81 / (2.0 * material.stiffness);
    println!("expecting {expected:.4} m");
    for cells in [8, 16, 32, 64] {
        let got = stretch(cells, density);
        println!(
            "{cells:3} cells: stretch {got:.4} m, {:+.1}%",
            (got - expected) / expected * 100.0
        );
    }
}
