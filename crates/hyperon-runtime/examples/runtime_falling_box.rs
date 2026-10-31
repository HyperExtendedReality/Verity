use std::{error::Error, time::Duration};

use glam::Vec3;
use hyperon_box3d::{Collider, PhysicsBody, RigidBody};
use hyperon_core::{Entity, Material, Transform};
use hyperon_runtime::Runtime;

fn main() -> Result<(), Box<dyn Error>> {
    let mut runtime = Runtime::new();
    let floor = runtime.world.spawn(Entity {
        name: "Floor".into(),
        transform: Transform {
            translation: [0.0, -0.5, 0.0],
            ..Transform::default()
        },
        material: Material::default(),
    });
    let cube = runtime.world.spawn(Entity {
        name: "Cube".into(),
        transform: Transform {
            translation: [0.0, 3.0, 0.0],
            ..Transform::default()
        },
        material: Material::default(),
    });
    runtime.physics.insert(
        &runtime.world,
        floor,
        PhysicsBody::new(
            RigidBody::Static,
            Collider::Cuboid {
                half_extents: Vec3::new(5.0, 0.5, 5.0),
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
    let initial_y = runtime
        .world
        .get(cube)
        .ok_or("cube missing")?
        .transform
        .translation[1];
    let ticks = runtime.advance(Duration::from_secs(2));
    let final_y = runtime
        .world
        .get(cube)
        .ok_or("cube missing")?
        .transform
        .translation[1];
    println!("{ticks} fixed ticks: cube y {initial_y:.3} -> {final_y:.3}");
    if !(0.45..0.65).contains(&final_y) {
        return Err(format!("cube did not settle on the floor: y={final_y}").into());
    }
    Ok(())
}
