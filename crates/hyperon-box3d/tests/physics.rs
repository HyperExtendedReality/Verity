use glam::{Quat, Vec3};
use hyperon_box3d::*;
use hyperon_core::{Entity, EntityHandle, Material, Transform, World};

fn spawn(
    world: &mut World,
    physics: &mut PhysicsWorld,
    position: Vec3,
    body: PhysicsBody,
) -> EntityHandle {
    let entity = world.spawn(Entity {
        name: "body".into(),
        transform: Transform {
            translation: position.to_array(),
            ..Default::default()
        },
        material: Material::default(),
        ..Entity::default()
    });
    physics.insert(world, entity, body).unwrap();
    entity
}

fn step(world: &mut World, physics: &mut PhysicsWorld, gravity: Vec3, count: usize) {
    let settings = Box3dSettings {
        gravity,
        ..Default::default()
    };
    for _ in 0..count {
        physics.step(world, &settings, 1.0 / 60.0);
    }
}

#[test]
fn colliders_fall_and_rest_on_ground() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let ground = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, -0.5, 0.0),
        PhysicsBody::new(
            RigidBody::Static,
            Collider::Cuboid {
                half_extents: Vec3::new(10.0, 0.5, 10.0),
            },
        ),
    );
    let shapes = [
        (
            Collider::Cuboid {
                half_extents: Vec3::splat(0.5),
            },
            0.5,
        ),
        (Collider::Sphere { radius: 0.5 }, 0.5),
        (
            Collider::Capsule {
                radius: 0.5,
                half_height: 0.5,
            },
            1.0,
        ),
    ];
    let bodies: Vec<_> = shapes
        .iter()
        .enumerate()
        .map(|(i, (shape, _))| {
            spawn(
                &mut world,
                &mut physics,
                Vec3::new(i as f32 * 3.0 - 3.0, 4.0, 0.0),
                PhysicsBody::new(RigidBody::Dynamic, shape.clone()),
            )
        })
        .collect();
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    step(&mut world, &mut physics, gravity, 30);
    for entity in &bodies {
        assert!(world.get(*entity).unwrap().transform.translation[1] < 3.0);
    }
    step(&mut world, &mut physics, gravity, 270);
    for (entity, (_, height)) in bodies.iter().zip(shapes) {
        let y = world.get(*entity).unwrap().transform.translation[1];
        assert!(
            (y - height).abs() < 0.08,
            "unexpected resting position: {y}"
        );
        assert!(physics.get(*entity).unwrap().velocity.linear.length() < 0.1);
    }
    assert_eq!(world.get(ground).unwrap().transform.translation[1], -0.5);
    assert_eq!(physics.body_count(), 4);
}

#[test]
fn edits_impulses_and_removal_reconcile_native_bodies() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let mut body = PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius: 0.5 });
    body.material.density = 1.0;
    body.impulse.linear = Vec3::X;
    let entity = spawn(&mut world, &mut physics, Vec3::ZERO, body);
    physics.sync(&world);
    assert_eq!(physics.get(entity).unwrap().impulse.linear, Vec3::X);
    step(&mut world, &mut physics, Vec3::ZERO, 1);
    assert_eq!(physics.get(entity).unwrap().impulse.linear, Vec3::ZERO);
    let speed = physics.get(entity).unwrap().velocity.linear.x;
    assert!(speed > 1.0);
    step(&mut world, &mut physics, Vec3::ZERO, 1);
    assert!((physics.get(entity).unwrap().velocity.linear.x - speed).abs() < 0.001);
    let rotation = Quat::from_rotation_z(0.3);
    *world.get_mut(entity).unwrap().transform = Transform {
        translation: [10.0, 5.0, 0.0],
        rotation: rotation.to_array(),
        scale: [2.0; 3],
    };
    let body = physics.get_mut(entity).unwrap();
    body.velocity.linear = Vec3::new(2.0, 0.0, 0.0);
    body.collider = Collider::Cuboid {
        half_extents: Vec3::splat(0.75),
    };
    step(&mut world, &mut physics, Vec3::ZERO, 1);
    let transform = world.get(entity).unwrap().transform;
    assert!((transform.translation[0] - (10.0 + 2.0 / 60.0)).abs() < 0.001);
    assert!((Quat::from_array(transform.rotation).dot(rotation).abs() - 1.0).abs() < 0.001);
    assert_eq!(transform.scale, [2.0; 3]);
    assert!(physics.remove(entity).is_some());
    assert!(!physics.contains(entity));
    physics
        .insert(
            &world,
            entity,
            PhysicsBody::new(RigidBody::Kinematic, Collider::Sphere { radius: 1.0 }),
        )
        .unwrap();
    step(&mut world, &mut physics, Vec3::ZERO, 1);
    assert!(physics.contains(entity));
    world.despawn(entity);
    assert!(
        physics
            .insert(
                &world,
                entity,
                PhysicsBody::new(RigidBody::Static, Collider::Sphere { radius: 1.0 })
            )
            .is_err()
    );
    let replacement = world.spawn(Entity {
        name: "replacement".into(),
        transform: Transform::default(),
        material: Material::default(),
        ..Entity::default()
    });
    assert_ne!(replacement, entity);
    physics.sync(&world);
    assert_eq!(physics.body_count(), 0);
    assert!(!physics.contains(replacement));
}

#[test]
fn kinematic_motion_force_and_body_type_changes() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let mut body = PhysicsBody::new(RigidBody::Kinematic, Collider::Sphere { radius: 0.5 });
    body.velocity.linear = Vec3::X;
    let kinematic = spawn(&mut world, &mut physics, Vec3::ZERO, body);
    let mut body = PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius: 0.5 });
    body.material.density = 1.0;
    body.force = ExternalForce {
        force: Vec3::X,
        torque: Vec3::Z,
    };
    let dynamic = spawn(&mut world, &mut physics, Vec3::new(0.0, 10.0, 0.0), body);
    step(&mut world, &mut physics, Vec3::ZERO, 60);
    assert!((world.get(kinematic).unwrap().transform.translation[0] - 1.0).abs() < 0.001);
    let velocity = physics.get(dynamic).unwrap().velocity;
    assert!(velocity.linear.x > 1.0 && velocity.angular.z > 1.0);
    physics.get_mut(dynamic).unwrap().body_type = RigidBody::Static;
    step(&mut world, &mut physics, Vec3::ZERO, 1);
    let position = world.get(dynamic).unwrap().transform.translation;
    step(&mut world, &mut physics, Vec3::ZERO, 10);
    assert_eq!(world.get(dynamic).unwrap().transform.translation, position);
    physics.get_mut(dynamic).unwrap().body_type = RigidBody::Dynamic;
    step(&mut world, &mut physics, Vec3::ZERO, 10);
    assert!(world.get(dynamic).unwrap().transform.translation[0] > position[0]);
}

#[test]
#[should_panic(expected = "sphere radius must be finite and positive")]
fn invalid_shape_is_rejected_before_ffi() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    spawn(
        &mut world,
        &mut physics,
        Vec3::ZERO,
        PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius: f32::NAN }),
    );
    step(&mut world, &mut physics, Vec3::ZERO, 1);
}

#[test]
fn convex_hull_rests_on_triangle_mesh_ground() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let floor = TriMesh::new(
        &[
            Vec3::new(-10.0, 0.0, -10.0),
            Vec3::new(-10.0, 0.0, 10.0),
            Vec3::new(10.0, 0.0, 10.0),
            Vec3::new(10.0, 0.0, -10.0),
        ],
        &[[0, 1, 2], [0, 2, 3]],
    )
    .unwrap();
    assert_eq!(floor.triangles().len(), 6);
    assert!(TriMesh::new(&[Vec3::ZERO; 3], &[[0, 1, 3]]).is_none());
    let corners: Vec<_> = (0..8)
        .map(|i| {
            Vec3::new(
                (i & 1) as f32 - 0.5,
                ((i >> 1) & 1) as f32 - 0.5,
                (i >> 2) as f32 - 0.5,
            )
        })
        .collect();
    let cube = ConvexHull::new(&corners).unwrap();
    assert_eq!(cube.planes().len(), 6);
    assert!(ConvexHull::new(&corners[..3]).is_none());
    spawn(
        &mut world,
        &mut physics,
        Vec3::ZERO,
        PhysicsBody::new(RigidBody::Static, Collider::TriMesh(floor)),
    );
    let body = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 3.0, 0.0),
        PhysicsBody::new(RigidBody::Dynamic, Collider::ConvexHull(cube)),
    );
    step(&mut world, &mut physics, Vec3::new(0.0, -9.81, 0.0), 240);
    let height = world.get(body).unwrap().transform.translation[1];
    assert!((height - 0.5).abs() < 0.08, "hull rests at {height}");
    physics.get_mut(body).unwrap().collider =
        Collider::ConvexHull(ConvexHull::new(&corners).unwrap());
    step(&mut world, &mut physics, Vec3::ZERO, 1);
    world.despawn(body);
    step(&mut world, &mut physics, Vec3::ZERO, 1);
    assert_eq!(physics.body_count(), 1);
}

#[test]
fn material_change_updates_mass_and_sync_does_not_consume_impulse() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let mut body = PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius: 0.5 });
    body.material.density = 1.0;
    body.impulse.linear = Vec3::X;
    let entity = spawn(&mut world, &mut physics, Vec3::ZERO, body);
    physics.sync(&world);
    physics.sync(&world);
    step(&mut world, &mut physics, Vec3::ZERO, 1);
    let first = physics.get(entity).unwrap().velocity.linear.x;
    let body = physics.get_mut(entity).unwrap();
    body.material.density = 2.0;
    body.impulse.linear = Vec3::X;
    step(&mut world, &mut physics, Vec3::ZERO, 1);
    let second = physics.get(entity).unwrap().velocity.linear.x - first;
    assert!((second / first - 0.5).abs() < 0.02);
}
