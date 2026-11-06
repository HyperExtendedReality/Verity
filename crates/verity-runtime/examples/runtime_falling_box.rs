use std::{error::Error, time::Duration};

use glam::Vec3;
use verity_core::{Entity, Material, Transform};
use verity_physics::{Collider, PhysicsBody, RigidBody};
use verity_runtime::Runtime;

fn main() -> Result<(), Box<dyn Error>> {
    let mut runtime = Runtime::new();
    let floor = runtime.world.spawn(Entity {
        name: "Floor".into(),
        transform: Transform {
            translation: [0.0, 0.0, -0.5],
            ..Transform::default()
        },
        material: Material::default(),
        ..Entity::default()
    });
    let cube = runtime.world.spawn(Entity {
        name: "Cube".into(),
        transform: Transform {
            translation: [0.0, 0.0, 3.0],
            ..Transform::default()
        },
        material: Material::default(),
        ..Entity::default()
    });
    runtime.physics.insert(
        &runtime.world,
        floor,
        PhysicsBody::new(
            RigidBody::Static,
            Collider::Cuboid {
                half_extents: Vec3::new(5.0, 5.0, 0.5),
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
    let initial_z = runtime
        .world
        .get(cube)
        .ok_or("cube missing")?
        .transform
        .translation[2];
    let ticks = runtime.advance(Duration::from_secs(2));
    let final_z = runtime
        .world
        .get(cube)
        .ok_or("cube missing")?
        .transform
        .translation[2];
    println!("{ticks} fixed ticks: cube z {initial_z:.3} -> {final_z:.3}");
    if !(0.45..0.65).contains(&final_z) {
        return Err(format!("cube did not settle on the floor: z={final_z}").into());
    }
    Ok(())
}
