use glam::Vec3;
use verity_box3d::{Box3dSettings, Collider, PhysicsBody, PhysicsWorld, RigidBody};
use verity_core::{Entity, Material, Transform, World};

fn main() {
    let mut world = World::new();
    let mut physics = PhysicsWorld::default();
    let ground = world.spawn(Entity {
        name: "ground".into(),
        transform: Transform {
            translation: [0.0, -0.5, 0.0],
            ..Default::default()
        },
        material: Material::default(),
        ..Entity::default()
    });
    physics
        .insert(
            &world,
            ground,
            PhysicsBody::new(
                RigidBody::Static,
                Collider::Cuboid {
                    half_extents: Vec3::new(10.0, 0.5, 10.0),
                },
            ),
        )
        .unwrap();
    let cube = world.spawn(Entity {
        name: "cube".into(),
        transform: Transform {
            translation: [0.0, 4.0, 0.0],
            ..Default::default()
        },
        material: Material::default(),
        ..Entity::default()
    });
    physics
        .insert(
            &world,
            cube,
            PhysicsBody::new(
                RigidBody::Dynamic,
                Collider::Cuboid {
                    half_extents: Vec3::splat(0.5),
                },
            ),
        )
        .unwrap();
    let settings = Box3dSettings::default();
    for tick in 0..180 {
        physics.step(&mut world, &settings, 1.0 / 60.0);
        if tick % 30 == 0 {
            println!(
                "tick {tick}: {:?}",
                world.get(cube).unwrap().transform.translation
            );
        }
    }
    let height = world.get(cube).unwrap().transform.translation[1];
    assert!((height - 0.5).abs() < 0.08);
}
