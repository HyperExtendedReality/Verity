use glam::{Quat, Vec3};
use hyperon_core::{Entity, EntityHandle, Material, Transform, World};
use hyperon_physics::*;

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
    });
    physics.insert(world, entity, body).unwrap();
    entity
}

fn ground(world: &mut World, physics: &mut PhysicsWorld) -> EntityHandle {
    let collider = Collider::Cuboid {
        half_extents: Vec3::new(20.0, 0.5, 20.0),
    };
    spawn(
        world,
        physics,
        Vec3::new(0.0, -0.5, 0.0),
        PhysicsBody::new(RigidBody::Static, collider),
    )
}

fn cube(half: f32) -> PhysicsBody {
    PhysicsBody::new(
        RigidBody::Dynamic,
        Collider::Cuboid {
            half_extents: Vec3::splat(half),
        },
    )
}

fn position(world: &World, entity: EntityHandle) -> Vec3 {
    Vec3::from_array(world.get(entity).unwrap().transform.translation)
}

fn run(world: &mut World, physics: &mut PhysicsWorld, steps: usize) {
    let settings = PhysicsSettings::default();
    for _ in 0..steps {
        physics.step(world, &settings, 1.0 / 60.0);
    }
}

#[test]
fn shapes_fall_and_rest_on_ground() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    ground(&mut world, &mut physics);
    let cube = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 3.0, 0.0),
        cube(0.5),
    );
    let ball = spawn(
        &mut world,
        &mut physics,
        Vec3::new(3.0, 2.0, 0.0),
        PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius: 0.5 }),
    );
    run(&mut world, &mut physics, 180);
    for entity in [cube, ball] {
        let p = position(&world, entity);
        assert!((p.y - 0.5).abs() < 0.03, "rest height {p}");
        assert!(physics.get(entity).unwrap().velocity.linear.length() < 0.05);
    }
    assert!(physics.contact_count() >= 5);
}

#[test]
fn box_stack_stays_upright() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    ground(&mut world, &mut physics);
    let stack: Vec<_> = (0..10)
        .map(|i| {
            spawn(
                &mut world,
                &mut physics,
                Vec3::new(0.0, 0.5 + i as f32, 0.0),
                cube(0.5),
            )
        })
        .collect();
    run(&mut world, &mut physics, 300);
    for (i, &entity) in stack.iter().enumerate() {
        let p = position(&world, entity);
        assert!(
            p.x.abs() < 0.05 && p.z.abs() < 0.05,
            "box {i} drifted to {p}"
        );
        assert!((p.y - (0.5 + i as f32)).abs() < 0.2, "box {i} at {p}");
    }
}

#[test]
fn heavy_box_on_light_box_holds() {
    // Mass ratio 1000:1, the case AVBD's ramped penalties handle without substeps.
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    ground(&mut world, &mut physics);
    let light = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.25, 0.0),
        cube(0.25),
    );
    let mut heavy = cube(1.0);
    heavy.material.density = 1000.0 * 1000.0 / 64.0;
    let heavy = spawn(&mut world, &mut physics, Vec3::new(0.0, 1.5, 0.0), heavy);
    run(&mut world, &mut physics, 240);
    assert!(position(&world, light).y > 0.2, "light box crushed");
    assert!((position(&world, heavy).y - 1.5).abs() < 0.1);
}

#[test]
fn friction_holds_box_on_slope() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let tilt = Quat::from_rotation_z(0.3); // tan(0.3) ≈ 0.31 < friction 0.6
    let slope = world.spawn(Entity {
        name: "slope".into(),
        transform: Transform {
            rotation: tilt.to_array(),
            ..Default::default()
        },
        material: Material::default(),
    });
    let plank = Collider::Cuboid {
        half_extents: Vec3::new(10.0, 0.5, 10.0),
    };
    physics
        .insert(&world, slope, PhysicsBody::new(RigidBody::Static, plank))
        .unwrap();
    let start = tilt * Vec3::new(0.0, 1.0, 0.0);
    let entity = world.spawn(Entity {
        name: "box".into(),
        transform: Transform {
            translation: start.to_array(),
            rotation: tilt.to_array(),
            ..Default::default()
        },
        material: Material::default(),
    });
    physics.insert(&world, entity, cube(0.5)).unwrap();
    run(&mut world, &mut physics, 240);
    assert!(
        position(&world, entity).distance(start) < 0.1,
        "slid to {}",
        position(&world, entity)
    );
}

#[test]
fn despawned_and_removed_bodies_leave() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let floor = ground(&mut world, &mut physics);
    let a = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 1.0, 0.0),
        cube(0.5),
    );
    let b = spawn(
        &mut world,
        &mut physics,
        Vec3::new(2.0, 1.0, 0.0),
        cube(0.5),
    );
    run(&mut world, &mut physics, 10);
    world.despawn(a);
    assert!(physics.remove(floor).is_some());
    run(&mut world, &mut physics, 10);
    assert_eq!(physics.body_count(), 1);
    assert!(physics.contains(b) && !physics.contains(a));
    assert!(position(&world, b).y < 1.0, "still falls without the floor");
}

#[test]
fn impulse_and_kinematic_motion() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let settings = PhysicsSettings {
        gravity: Vec3::ZERO,
        ..Default::default()
    };
    let mut body = cube(0.5);
    body.impulse.linear = Vec3::new(1000.0, 0.0, 0.0); // 1000 kg cube → 1 m/s
    let pushed = spawn(&mut world, &mut physics, Vec3::ZERO, body);
    let mut mover = PhysicsBody::new(RigidBody::Kinematic, Collider::Sphere { radius: 0.5 });
    mover.velocity.linear = Vec3::Y;
    let mover = spawn(&mut world, &mut physics, Vec3::new(0.0, 0.0, 5.0), mover);
    for _ in 0..60 {
        physics.step(&mut world, &settings, 1.0 / 60.0);
    }
    assert!((position(&world, pushed).x - 1.0).abs() < 0.02);
    assert_eq!(
        physics.get(pushed).unwrap().impulse,
        ExternalImpulse::default()
    );
    assert!((position(&world, mover).y - 1.0).abs() < 1e-3);
}
