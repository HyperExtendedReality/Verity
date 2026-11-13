use glam::{Quat, Vec3};
use verity_core::{Entity, EntityHandle, Material, Transform, World};
use verity_physics::*;

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

fn ground(world: &mut World, physics: &mut PhysicsWorld) -> EntityHandle {
    let collider = Collider::Cuboid {
        half_extents: Vec3::new(20.0, 20.0, 0.5),
    };
    spawn(
        world,
        physics,
        Vec3::new(0.0, 0.0, -0.5),
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

/// A body on a floor wide enough for any of these runs.
fn on_long_floor(body: PhysicsBody, spin: Vec3, speed: f32) -> (World, PhysicsWorld, EntityHandle) {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, -0.5),
        PhysicsBody::new(
            RigidBody::Static,
            Collider::Cuboid {
                half_extents: Vec3::new(200.0, 200.0, 0.5),
            },
        ),
    );
    let mut body = body;
    body.velocity = Velocity {
        linear: Vec3::new(speed, 0.0, 0.0),
        angular: spin,
    };
    let entity = spawn(&mut world, &mut physics, Vec3::new(0.0, 0.0, 0.5), body);
    (world, physics, entity)
}

#[test]
fn sliding_friction_stops_a_box_at_the_coulomb_distance() {
    for speed in [5.0f32, 10.0] {
        let (mut world, mut physics, entity) = on_long_floor(cube(0.5), Vec3::ZERO, speed);
        run(&mut world, &mut physics, 300);
        let expected = speed * speed / (2.0 * 0.6 * 9.81);
        let x = position(&world, entity).x;
        assert!((x - expected).abs() < expected * 0.05, "{x} for {expected}");
    }
}

#[test]
fn sliding_ball_ends_rolling_at_five_sevenths_of_its_speed() {
    let ball = PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius: 0.5 });
    let (mut world, mut physics, entity) = on_long_floor(ball, Vec3::ZERO, 7.0);
    run(&mut world, &mut physics, 180);
    let velocity = physics.get(entity).unwrap().velocity;
    assert!((velocity.linear.x - 5.0).abs() < 0.05, "{:?}", velocity);
    assert!(
        (velocity.angular.y * 0.5 - 5.0).abs() < 0.05,
        "{:?}",
        velocity
    );
}

#[test]
fn free_spin_is_not_damped() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let mut ball = PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius: 0.5 });
    ball.velocity.angular = Vec3::new(0.0, 0.0, 30.0);
    let entity = spawn(&mut world, &mut physics, Vec3::new(0.0, 0.0, 50.0), ball);
    let settings = PhysicsSettings {
        gravity: Vec3::ZERO,
        ..Default::default()
    };
    for _ in 0..300 {
        physics.step(&mut world, &settings, 1.0 / 60.0);
    }
    let spin = physics.get(entity).unwrap().velocity.angular.z;
    assert!((spin - 30.0).abs() < 1.0e-3, "{spin}");
}

#[test]
fn tumbling_slab_conserves_momentum_and_energy_and_flips() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let half = Vec3::new(0.5, 0.3, 0.15);
    let mut slab = PhysicsBody::new(RigidBody::Dynamic, Collider::Cuboid { half_extents: half });
    // Nearly about the intermediate axis, which is unstable.
    let spin = Vec3::new(0.02, 4.0, 0.02);
    slab.velocity.angular = spin;
    let entity = spawn(&mut world, &mut physics, Vec3::new(0.0, 0.0, 50.0), slab);
    let settings = PhysicsSettings {
        gravity: Vec3::ZERO,
        ..Default::default()
    };
    for _ in 0..300 {
        physics.step(&mut world, &settings, 1.0 / 60.0);
    }
    let square = half * half;
    let moments = Vec3::new(
        square.y + square.z,
        square.x + square.z,
        square.x + square.y,
    );
    let rotation = Quat::from_array(world.get(entity).unwrap().transform.rotation);
    let local = rotation.inverse() * physics.get(entity).unwrap().velocity.angular;
    let momentum = rotation * (moments * local);
    assert!((momentum - moments * spin).length() < 1.0e-3 * (moments * spin).length());
    let energy = (moments * local).dot(local) / (moments * spin).dot(spin);
    assert!((energy - 1.0).abs() < 1.0e-3, "energy ratio {energy}");
    assert!(local.y < -3.5, "spin must have flipped: {local}");
}

#[test]
fn impact_stops_at_the_surface_and_rests_there() {
    let ball = PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius: 0.5 });
    for body in [cube(0.5), ball] {
        let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
        ground(&mut world, &mut physics);
        let entity = spawn(&mut world, &mut physics, Vec3::new(0.0, 0.0, 20.5), body);
        for _ in 0..240 {
            run(&mut world, &mut physics, 1);
            assert!(position(&world, entity).z > 0.498, "sank into the floor");
        }
        // Neither inside the floor nor hovering above it.
        assert!((position(&world, entity).z - 0.5).abs() < 1.0e-3);
        assert_eq!(physics.get(entity).unwrap().velocity.linear, Vec3::ZERO);
    }
}

#[test]
fn a_gap_is_closed_in_the_step_that_reaches_it() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    ground(&mut world, &mut physics);
    let mut body = cube(0.5);
    body.velocity.linear = Vec3::new(0.0, 0.0, -30.0);
    let entity = spawn(&mut world, &mut physics, Vec3::new(0.0, 0.0, 0.7), body);
    run(&mut world, &mut physics, 1);
    let z = position(&world, entity).z;
    assert!((z - 0.5).abs() < 0.01, "stopped short at {z}");
}

/// A world without gravity and with `fixed` as its only static body.
fn weightless(fixed: Vec3) -> (World, PhysicsWorld) {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let collider = Collider::Cuboid {
        half_extents: fixed,
    };
    let body = PhysicsBody::new(RigidBody::Static, collider);
    spawn(&mut world, &mut physics, Vec3::ZERO, body);
    (world, physics)
}

fn run_weightless(world: &mut World, physics: &mut PhysicsWorld, steps: usize) {
    let settings = PhysicsSettings {
        gravity: Vec3::ZERO,
        ..Default::default()
    };
    for _ in 0..steps {
        physics.step(world, &settings, 1.0 / 60.0);
    }
}

#[test]
fn a_box_slides_over_floor_tile_seams() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    for i in -5..30 {
        let collider = Collider::Cuboid {
            half_extents: Vec3::new(0.5, 10.0, 0.5),
        };
        let tile = PhysicsBody::new(RigidBody::Static, collider);
        spawn(
            &mut world,
            &mut physics,
            Vec3::new(i as f32, 0.0, -0.5),
            tile,
        );
    }
    let mut body = cube(0.5);
    body.velocity.linear = Vec3::new(10.0, 0.0, 0.0);
    let entity = spawn(&mut world, &mut physics, Vec3::new(0.0, 0.0, 0.5), body);
    run(&mut world, &mut physics, 240);
    let expected = 10.0 * 10.0 / (2.0 * 0.6 * 9.81);
    let x = position(&world, entity).x;
    assert!((x - expected).abs() < expected * 0.05, "{x} for {expected}");
}

#[test]
fn static_friction_does_not_creep_down_a_slope() {
    // tan(30 degrees) is below the default friction of 0.6.
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let collider = Collider::Cuboid {
        half_extents: Vec3::new(50.0, 50.0, 0.5),
    };
    let floor = PhysicsBody::new(RigidBody::Static, collider);
    spawn(&mut world, &mut physics, Vec3::new(0.0, 0.0, -0.5), floor);
    let entity = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 0.5),
        cube(0.5),
    );
    let angle = 30.0f32.to_radians();
    let settings = PhysicsSettings {
        gravity: Vec3::new(angle.sin(), 0.0, -angle.cos()) * 9.81,
        ..Default::default()
    };
    for _ in 0..300 {
        physics.step(&mut world, &settings, 1.0 / 60.0);
    }
    let x = position(&world, entity).x;
    assert!(x.abs() < 5.0e-4, "crept {x} m");
}

#[test]
fn fast_bodies_do_not_tunnel_through_a_thin_wall() {
    let ball = PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius: 0.5 });
    for shape in [cube(0.5), ball] {
        for speed in [20.0, 60.0, 150.0] {
            // Offsets vary where in a step the body meets the wall.
            for phase in 0..7 {
                let mut body = shape.clone();
                body.velocity.linear = Vec3::new(speed, 0.0, 0.0);
                let (mut world, mut physics) = weightless(Vec3::new(0.05, 20.0, 20.0));
                let start = Vec3::new(-8.0 - phase as f32 * 0.13, 0.0, 0.0);
                let entity = spawn(&mut world, &mut physics, start, body);
                run_weightless(&mut world, &mut physics, 120);
                let x = position(&world, entity).x;
                assert!(x < 0.0, "through at {speed} m/s, phase {phase}: {x}");
            }
        }
    }
}

#[test]
fn passing_a_corner_is_not_a_collision() {
    let (along, across) = (
        Vec3::new(1.0, 1.0, 0.0).normalize(),
        Vec3::new(1.0, -1.0, 0.0).normalize(),
    );
    for speed in [30.0, 80.0] {
        for phase in 0..7 {
            let (mut world, mut physics) = weightless(Vec3::splat(0.5));
            let mut ball = PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius: 0.5 });
            ball.velocity.linear = along * speed;
            // The cube's far corner, the ball's radius, and 5 cm to spare.
            let line = across * (0.7072 + 0.5 + 0.05);
            let start = line - along * (6.0 + phase as f32 * 0.037 * speed / 10.0);
            let entity = spawn(&mut world, &mut physics, start, ball);
            run_weightless(&mut world, &mut physics, (14.0 / speed * 60.0) as usize);
            let velocity = physics.get(entity).unwrap().velocity.linear;
            assert!((velocity - along * speed).length() < 1.0e-3 * speed);
        }
    }
}

#[test]
fn collisions_between_free_bodies_keep_their_momentum() {
    for speed in [5.0f32, 20.0] {
        let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
        let moving = |x: f32| {
            let mut body = cube(0.5);
            body.velocity.linear = Vec3::new(x, 0.0, 0.0);
            body
        };
        let a = spawn(
            &mut world,
            &mut physics,
            Vec3::new(-3.0, 0.0, 0.0),
            moving(speed),
        );
        let b = spawn(
            &mut world,
            &mut physics,
            Vec3::new(3.0, 0.0, 0.0),
            moving(-speed),
        );
        run_weightless(&mut world, &mut physics, 120);
        let x = |e| physics.get(e).unwrap().velocity.linear.x;
        assert!((x(a) + x(b)).abs() < 0.05 * speed, "{} {}", x(a), x(b));

        let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
        let a = spawn(
            &mut world,
            &mut physics,
            Vec3::new(-3.0, 0.0, 0.0),
            moving(speed),
        );
        let b = spawn(
            &mut world,
            &mut physics,
            Vec3::new(3.0, 0.0, 0.0),
            moving(0.0),
        );
        run_weightless(&mut world, &mut physics, 120);
        let x = |e| physics.get(e).unwrap().velocity.linear.x;
        assert!(
            (x(a) + x(b) - speed).abs() < 0.02 * speed,
            "{} {}",
            x(a),
            x(b)
        );
    }
}

#[test]
fn shapes_fall_and_rest_on_ground() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    ground(&mut world, &mut physics);
    let cube = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 3.0),
        cube(0.5),
    );
    let ball = spawn(
        &mut world,
        &mut physics,
        Vec3::new(3.0, 0.0, 2.0),
        PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius: 0.5 }),
    );
    run(&mut world, &mut physics, 180);
    for entity in [cube, ball] {
        let p = position(&world, entity);
        assert!((p.z - 0.5).abs() < 0.03, "rest height {p}");
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
                Vec3::new(0.0, 0.0, 0.5 + i as f32),
                cube(0.5),
            )
        })
        .collect();
    run(&mut world, &mut physics, 300);
    for (i, &entity) in stack.iter().enumerate() {
        let p = position(&world, entity);
        assert!(
            p.x.abs() < 0.05 && p.y.abs() < 0.05,
            "box {i} drifted to {p}"
        );
        assert!((p.z - (0.5 + i as f32)).abs() < 0.2, "box {i} at {p}");
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
        Vec3::new(0.0, 0.0, 0.25),
        cube(0.25),
    );
    let mut heavy = cube(1.0);
    heavy.material.density = 1000.0 * 1000.0 / 64.0;
    let heavy = spawn(&mut world, &mut physics, Vec3::new(0.0, 0.0, 1.5), heavy);
    run(&mut world, &mut physics, 240);
    assert!(position(&world, light).z > 0.2, "light box crushed");
    assert!((position(&world, heavy).z - 1.5).abs() < 0.1);
}

#[test]
fn friction_holds_box_on_slope() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let tilt = Quat::from_rotation_x(0.3); // tan(0.3) ≈ 0.31 < friction 0.6
    let slope = world.spawn(Entity {
        name: "slope".into(),
        transform: Transform {
            rotation: tilt.to_array(),
            ..Default::default()
        },
        material: Material::default(),
        ..Entity::default()
    });
    let plank = Collider::Cuboid {
        half_extents: Vec3::new(10.0, 10.0, 0.5),
    };
    physics
        .insert(&world, slope, PhysicsBody::new(RigidBody::Static, plank))
        .unwrap();
    let start = tilt * Vec3::new(0.0, 0.0, 1.0);
    let entity = world.spawn(Entity {
        name: "box".into(),
        transform: Transform {
            translation: start.to_array(),
            rotation: tilt.to_array(),
            ..Default::default()
        },
        material: Material::default(),
        ..Entity::default()
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
fn frictionless_ground_preserves_tangential_velocity() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let floor = ground(&mut world, &mut physics);
    let floor_material = &mut physics.get_mut(floor).unwrap().material;
    floor_material.friction = 0.0;
    floor_material.static_friction = 0.0;
    let mut body = cube(0.5);
    body.material.friction = 0.0;
    body.material.static_friction = 0.0;
    body.velocity.linear = Vec3::new(2.0, 0.0, 0.0);
    let entity = spawn(&mut world, &mut physics, Vec3::new(0.0, 0.0, 0.5), body);
    run(&mut world, &mut physics, 60);
    let velocity = physics.get(entity).unwrap().velocity.linear;
    assert!(velocity.x > 1.8, "frictionless speed fell to {velocity}");
    assert!(position(&world, entity).x > 1.8);
}

#[test]
fn contact_does_not_restrain_separating_body() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    ground(&mut world, &mut physics);
    let entity = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 0.5),
        cube(0.5),
    );
    run(&mut world, &mut physics, 120);
    physics.get_mut(entity).unwrap().velocity.linear = Vec3::new(0.0, 0.0, 5.0);
    run(&mut world, &mut physics, 1);
    let speed = physics.get(entity).unwrap().velocity.linear.z;
    assert!(speed > 4.5, "separation was impeded: {speed}");
}

#[test]
fn releasing_loaded_friction_does_not_keep_a_box_stuck() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let floor = ground(&mut world, &mut physics);
    let entity = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 0.5),
        cube(0.5),
    );
    run(&mut world, &mut physics, 120);
    physics.get_mut(entity).unwrap().force.force = Vec3::new(3_000.0, 0.0, 0.0);
    run(&mut world, &mut physics, 120);
    let floor_material = &mut physics.get_mut(floor).unwrap().material;
    floor_material.friction = 0.0;
    floor_material.static_friction = 0.0;
    let body = physics.get_mut(entity).unwrap();
    body.material.friction = 0.0;
    body.material.static_friction = 0.0;
    body.force.force = Vec3::ZERO;
    body.velocity.linear = Vec3::new(2.0, 0.0, 0.0);
    run(&mut world, &mut physics, 30);
    let velocity = physics.get(entity).unwrap().velocity.linear;
    assert!(velocity.x > 1.8, "released box slowed to {velocity}");
}

#[test]
fn despawned_and_removed_bodies_leave() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let floor = ground(&mut world, &mut physics);
    let a = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 1.0),
        cube(0.5),
    );
    let b = spawn(
        &mut world,
        &mut physics,
        Vec3::new(2.0, 0.0, 1.0),
        cube(0.5),
    );
    run(&mut world, &mut physics, 10);
    world.despawn(a);
    assert!(physics.remove(floor).is_some());
    run(&mut world, &mut physics, 10);
    assert_eq!(physics.body_count(), 1);
    assert!(physics.contains(b) && !physics.contains(a));
    assert!(position(&world, b).z < 1.0, "still falls without the floor");
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

#[test]
fn changing_density_shape_and_rotation_updates_impulse_response() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let entity = spawn(&mut world, &mut physics, Vec3::ZERO, cube(0.5));
    let settings = PhysicsSettings {
        gravity: Vec3::ZERO,
        ..Default::default()
    };
    physics.step(&mut world, &settings, 1.0 / 60.0);

    let body = physics.get_mut(entity).unwrap();
    body.collider = Collider::Sphere { radius: 0.5 };
    body.material.density = 500.0;
    body.impulse.linear = Vec3::X * 1000.0;
    physics.step(&mut world, &settings, 1.0 / 60.0);
    let mass = (4.0 / 3.0) * std::f32::consts::PI * 0.5_f32.powi(3) * 500.0;
    assert!((physics.get(entity).unwrap().velocity.linear.x - 1000.0 / mass).abs() < 1e-4);

    let body = physics.get_mut(entity).unwrap();
    body.collider = Collider::Cuboid {
        half_extents: Vec3::new(1.0, 0.5, 0.5),
    };
    body.material.density = 1000.0;
    body.velocity = Velocity::default();
    body.impulse.angular = Vec3::X * (2500.0 / 3.0);
    world.get_mut(entity).unwrap().transform.rotation =
        Quat::from_rotation_z(std::f32::consts::FRAC_PI_2).to_array();
    physics.step(&mut world, &settings, 1.0 / 60.0);
    let omega = physics.get(entity).unwrap().velocity.angular;
    assert!((omega - Vec3::X).length() < 1e-3, "{omega:?}");
}

#[test]
fn default_gravity_points_down_z() {
    assert_eq!(
        PhysicsSettings::default().gravity,
        Vec3::new(0.0, 0.0, -9.81)
    );
}

#[test]
fn rolling_and_spinning_spheres_stay_on_the_ground() {
    // Regression: re-linearizing contacts at the current rotation sank
    // rolling spheres through the floor and sped them up.
    for (linear, angular) in [
        (Vec3::new(5.0, 0.0, 0.0), Vec3::new(0.0, 10.0, 0.0)),
        (Vec3::new(5.0, 0.0, 0.0), Vec3::ZERO),
        (Vec3::ZERO, Vec3::new(0.0, 0.0, 20.0)),
    ] {
        let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
        ground(&mut world, &mut physics);
        let mut body = PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius: 0.5 });
        body.velocity = Velocity { linear, angular };
        let ball = spawn(&mut world, &mut physics, Vec3::new(0.0, 0.0, 0.5), body);
        let mut lowest = f32::MAX;
        for _ in 0..120 {
            run(&mut world, &mut physics, 1);
            lowest = lowest.min(position(&world, ball).z);
        }
        assert!(lowest > 0.45, "sank to {lowest} from {linear} {angular}");
        let speed = physics.get(ball).unwrap().velocity.linear.length();
        assert!(speed <= linear.length() + 0.01, "gained speed: {speed}");
    }
}

#[test]
fn resting_stack_sleeps_and_stays_put() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    ground(&mut world, &mut physics);
    let stack: Vec<_> = (0..5)
        .map(|i| {
            spawn(
                &mut world,
                &mut physics,
                Vec3::new(0.0, 0.0, 0.5 + i as f32),
                cube(0.5),
            )
        })
        .collect();
    run(&mut world, &mut physics, 120);
    assert!(
        stack.iter().all(|&e| physics.is_sleeping(e)),
        "stack still awake"
    );
    let before: Vec<_> = stack.iter().map(|&e| position(&world, e)).collect();
    run(&mut world, &mut physics, 60);
    let after: Vec<_> = stack.iter().map(|&e| position(&world, e)).collect();
    assert_eq!(before, after);
    assert!(physics.contact_count() >= 20, "sleeping contacts are kept");
}

#[test]
fn sleeping_world_consumes_impulses_and_responds_to_force() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    ground(&mut world, &mut physics);
    let entity = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 0.5),
        cube(0.5),
    );
    run(&mut world, &mut physics, 120);
    assert!(physics.is_sleeping(entity));

    physics.get_mut(entity).unwrap().impulse.linear.x = -0.0;
    run(&mut world, &mut physics, 1);
    assert_eq!(physics.get(entity).unwrap().impulse.linear.x.to_bits(), 0);
    assert!(physics.is_sleeping(entity));

    physics.get_mut(entity).unwrap().force.force = Vec3::new(200_000.0, 0.0, 0.0);
    run(&mut world, &mut physics, 1);
    assert!(!physics.is_sleeping(entity), "force must wake the body");
}

#[test]
#[should_panic(expected = "friction coefficients must be finite and nonnegative")]
fn sleeping_world_rejects_invalid_material() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    ground(&mut world, &mut physics);
    let entity = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 0.5),
        cube(0.5),
    );
    run(&mut world, &mut physics, 120);
    assert!(physics.is_sleeping(entity));
    physics.get_mut(entity).unwrap().material.static_friction = f32::NAN;
    run(&mut world, &mut physics, 1);
}

#[test]
fn changing_a_quiet_static_body_to_kinematic_updates_its_pose() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let entity = spawn(
        &mut world,
        &mut physics,
        Vec3::ZERO,
        PhysicsBody::new(RigidBody::Static, Collider::Sphere { radius: 0.5 }),
    );
    world.get_mut(entity).unwrap().transform.rotation = [0.0, 0.0, 0.0, 0.9999];
    run(&mut world, &mut physics, 1);
    physics.get_mut(entity).unwrap().body_type = RigidBody::Kinematic;
    run(&mut world, &mut physics, 1);
    assert_eq!(
        world.get(entity).unwrap().transform.rotation,
        Quat::IDENTITY.to_array()
    );
}

#[test]
fn penetration_blocks_sleep_even_with_strong_regularization() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    ground(&mut world, &mut physics);
    let body = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 0.4),
        cube(0.5),
    );
    let settings = PhysicsSettings {
        gravity: Vec3::ZERO,
        alpha: 0.9999,
        ..Default::default()
    };
    for _ in 0..90 {
        physics.step(&mut world, &settings, 1.0 / 60.0);
    }
    assert!(
        position(&world, body).z < 0.48,
        "fixture no longer penetrates"
    );
    assert!(
        !physics.is_sleeping(body),
        "regularized error hid the remaining penetration"
    );
}

#[test]
fn falling_ball_wakes_a_sleeping_stack() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    ground(&mut world, &mut physics);
    let stack: Vec<_> = (0..3)
        .map(|i| {
            spawn(
                &mut world,
                &mut physics,
                Vec3::new(0.0, 0.0, 0.5 + i as f32),
                cube(0.5),
            )
        })
        .collect();
    run(&mut world, &mut physics, 120);
    assert!(physics.is_sleeping(stack[0]));
    let mut heavy = PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius: 0.5 });
    heavy.material.density = 20_000.0;
    let ball = spawn(&mut world, &mut physics, Vec3::new(0.3, 0.0, 6.0), heavy);
    for _ in 0..90 {
        run(&mut world, &mut physics, 1);
        assert!(
            position(&world, ball).z > 2.6 || !physics.is_sleeping(stack[2]),
            "hit a sleeper"
        );
    }
    // The off-centre hit must have moved the stack, which was asleep.
    let top = position(&world, stack[2]);
    assert!(
        top.distance(Vec3::new(0.0, 0.0, 2.5)) > 0.01,
        "top box never woke: {top}"
    );
}

#[test]
fn collision_wake_preserves_other_islands_warm_start() {
    fn advance(prewake: bool) -> (Vec3, Vec3) {
        let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
        ground(&mut world, &mut physics);
        let sleeper = spawn(
            &mut world,
            &mut physics,
            Vec3::new(0.0, 0.0, 0.5),
            cube(0.5),
        );
        let settings = PhysicsSettings {
            gamma: 0.8,
            ..Default::default()
        };
        for _ in 0..120 {
            physics.step(&mut world, &settings, 1.0 / 60.0);
        }
        assert!(physics.is_sleeping(sleeper));

        let mut sliding = cube(0.5);
        sliding.material.friction = 0.0;
        sliding.material.static_friction = 0.0;
        sliding.velocity.linear = Vec3::new(2.0, 0.0, 0.0);
        let observer = spawn(&mut world, &mut physics, Vec3::new(5.0, 0.0, 0.5), sliding);
        for _ in 0..10 {
            physics.step(&mut world, &settings, 1.0 / 60.0);
        }
        assert!(physics.is_sleeping(sleeper));
        if prewake {
            let x = &mut world.get_mut(sleeper).unwrap().transform.translation[0];
            *x = if *x == 0.0 {
                1.0e-20
            } else {
                f32::from_bits(x.to_bits() + 1)
            };
        }
        let mut impact = PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius: 0.5 });
        impact.velocity.linear.z = -1.0;
        spawn(&mut world, &mut physics, Vec3::new(0.0, 0.0, 1.4), impact);
        physics.step(&mut world, &settings, 1.0 / 60.0);
        assert!(
            !physics.is_sleeping(sleeper),
            "impact failed to wake the island"
        );
        (
            position(&world, observer),
            physics.get(observer).unwrap().velocity.linear,
        )
    }

    let (collision_position, collision_velocity) = advance(false);
    let (requested_position, requested_velocity) = advance(true);
    assert!(collision_position.distance(requested_position) < 1.0e-5);
    assert!(collision_velocity.distance(requested_velocity) < 1.0e-5);
}

#[test]
fn caller_changes_and_lost_support_wake_sleepers() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let floor = ground(&mut world, &mut physics);
    let a = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 0.5),
        cube(0.5),
    );
    let b = spawn(
        &mut world,
        &mut physics,
        Vec3::new(3.0, 0.0, 0.5),
        cube(0.5),
    );
    run(&mut world, &mut physics, 120);
    assert!(physics.is_sleeping(a) && physics.is_sleeping(b));

    physics.get_mut(a).unwrap().velocity.linear = Vec3::new(2.0, 0.0, 0.0);
    run(&mut world, &mut physics, 10);
    assert!(position(&world, a).x > 0.1, "velocity woke it");

    physics.remove(floor);
    run(&mut world, &mut physics, 30);
    assert!(position(&world, b).z < 0.0, "fell once the floor went");
}

#[test]
fn despawn_does_not_hide_moved_support_from_sleepers() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let removed = spawn(
        &mut world,
        &mut physics,
        Vec3::new(5.0, 0.0, 0.5),
        cube(0.5),
    );
    let floor = ground(&mut world, &mut physics);
    let sleeper = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 0.5),
        cube(0.5),
    );
    run(&mut world, &mut physics, 120);
    assert!(physics.is_sleeping(sleeper));

    world.despawn(removed);
    world.get_mut(floor).unwrap().transform.translation[2] -= 5.0;
    run(&mut world, &mut physics, 30);
    assert!(!physics.contains(removed));
    assert!(
        position(&world, sleeper).z < 0.0,
        "sleeper did not lose its support"
    );
}

/// A pile big enough that a step solves on several threads.
fn pile() -> (World, PhysicsWorld, Vec<EntityHandle>) {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    ground(&mut world, &mut physics);
    let mut bodies = Vec::new();
    for i in 0..400 {
        let (x, y, z) = ((i % 10) as f32, (i / 10 % 10) as f32, (i / 100) as f32);
        let p =
            Vec3::new(x * 1.1 - 5.0, y * 1.1 - 5.0, 0.6 + z * 1.2) + Vec3::splat(i as f32 * 1e-3);
        let body = if i % 2 == 0 {
            cube(0.5)
        } else {
            PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius: 0.5 })
        };
        bodies.push(spawn(&mut world, &mut physics, p, body));
    }
    // Joints and springs along each row, so links are solved in parallel too.
    for i in (0..400).filter(|i| i % 10 != 9) {
        let (a, b) = (bodies[i], bodies[i + 1]);
        let (right, left) = (Vec3::new(0.55, 0.0, 0.0), Vec3::new(-0.55, 0.0, 0.0));
        if i % 2 == 0 {
            physics
                .add_joint(&world, Joint::new(Some(a), b, right, left))
                .unwrap();
        } else {
            let spring = Spring {
                body_a: Some(a),
                body_b: b,
                anchor_a: right,
                anchor_b: left,
                stiffness: 1.0e5,
                rest_length: -1.0,
            };
            physics.add_spring(&world, spring).unwrap();
        }
    }
    (world, physics, bodies)
}

#[test]
fn parallel_steps_match_the_serial_path_exactly() {
    let simulate = || {
        let (mut world, mut physics, bodies) = pile();
        run(&mut world, &mut physics, 90);
        bodies
            .iter()
            .map(|&e| world.get(e).unwrap().transform.clone())
            .collect::<Vec<_>>()
    };
    let serial = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(simulate);
    assert_eq!(simulate(), serial);
}

#[test]
fn worlds_step_concurrently() {
    let worlds: Vec<_> = (0..2)
        .map(|_| {
            std::thread::spawn(|| {
                let (mut world, mut physics, _) = pile();
                run(&mut world, &mut physics, 30);
            })
        })
        .collect();
    for world in worlds {
        world.join().unwrap();
    }
}

fn bar() -> PhysicsBody {
    PhysicsBody::new(
        RigidBody::Dynamic,
        Collider::Cuboid {
            half_extents: Vec3::new(0.5, 0.1, 0.1),
        },
    )
}

fn point(world: &World, entity: EntityHandle, local: Vec3) -> Vec3 {
    let t = &world.get(entity).unwrap().transform;
    Vec3::from_array(t.translation) + Quat::from_array(t.rotation) * local
}

#[test]
fn hard_joint_chain_holds_without_gaining_energy() {
    // The AVBD demo chain: hard ball joints hold while it swings, and the
    // swing never gains energy.
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let top = Vec3::new(0.0, 0.0, 12.0);
    let links: Vec<_> = (0..10)
        .map(|i| {
            spawn(
                &mut world,
                &mut physics,
                top + Vec3::new(0.5 + i as f32, 0.0, 0.0),
                bar(),
            )
        })
        .collect();
    let (left, right) = (Vec3::new(-0.5, 0.0, 0.0), Vec3::new(0.5, 0.0, 0.0));
    physics
        .add_joint(&world, Joint::new(None, links[0], top, left))
        .unwrap();
    for pair in links.windows(2) {
        physics
            .add_joint(&world, Joint::new(Some(pair[0]), pair[1], right, left))
            .unwrap();
    }
    let error = |world: &World| {
        let mut worst = point(world, links[0], left).distance(top);
        for pair in links.windows(2) {
            worst = worst.max(point(world, pair[0], right).distance(point(world, pair[1], left)));
        }
        worst
    };
    // Per kilogram: every bar weighs the same.
    let energy = |world: &World, physics: &PhysicsWorld| -> f32 {
        (links.iter())
            .map(|&e| {
                9.81 * position(world, e).z
                    + 0.5 * physics.get(e).unwrap().velocity.linear.length_squared()
            })
            .sum()
    };
    let start = energy(&world, &physics);
    let (mut swing, mut highest) = (0.0f32, f32::MIN);
    for _ in 0..1200 {
        run(&mut world, &mut physics, 1);
        swing = swing.max(error(&world));
        highest = highest.max(energy(&world, &physics));
    }
    assert!(swing < 0.05, "joint error while swinging {swing}");
    assert!(error(&world) < 0.005, "joint error {}", error(&world));
    assert!(
        highest <= start * 1.001,
        "gained energy: {highest} from {start}"
    );
    assert!(position(&world, links[9]).z < 11.0, "chain swung down");
}

/// A level chain of `n` bars, the last `load` times as dense and thrown at
/// `kick` m/s. Returns the highest energy reached, as a multiple of the start's,
/// and the worst joint error at the end.
fn chain_energy(n: usize, load: f32, iterations: u32, dt: f32, kick: f32) -> (f32, f32) {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let top = Vec3::new(0.0, 0.0, 100.0);
    let half = Vec3::new(0.5, 0.1, 0.1);
    let volume = 8.0 * half.x * half.y * half.z;
    let masses: Vec<f32> = (0..n)
        .map(|i| 1000.0 * volume * if i + 1 == n { load } else { 1.0 })
        .collect();
    let bars: Vec<_> = (0..n)
        .map(|i| {
            let mut body = bar();
            body.material.density = masses[i] / volume;
            if i + 1 == n {
                body.velocity.linear = Vec3::new(0.0, kick, 0.0);
            }
            spawn(
                &mut world,
                &mut physics,
                top + Vec3::new(0.5 + i as f32, 0.0, 0.0),
                body,
            )
        })
        .collect();
    let (left, right) = (Vec3::new(-0.5, 0.0, 0.0), Vec3::new(0.5, 0.0, 0.0));
    physics
        .add_joint(&world, Joint::new(None, bars[0], top, left))
        .unwrap();
    for pair in bars.windows(2) {
        physics
            .add_joint(&world, Joint::new(Some(pair[0]), pair[1], right, left))
            .unwrap();
    }
    let rest: f32 = (0..n)
        .map(|i| masses[i] * 9.81 * (top.z - 0.5 - i as f32))
        .sum();
    let energy = |world: &World, physics: &PhysicsWorld| -> f32 {
        (0..n)
            .map(|i| {
                let v = physics.get(bars[i]).unwrap().velocity;
                let q = Quat::from_array(world.get(bars[i]).unwrap().transform.rotation);
                let w = q.inverse() * v.angular;
                let h = half * half;
                let spin = (w * w).dot(Vec3::new(h.y + h.z, h.x + h.z, h.x + h.y)) / 3.0;
                let z = world.get(bars[i]).unwrap().transform.translation[2];
                masses[i] * (9.81 * z + 0.5 * v.linear.length_squared() + 0.5 * spin)
            })
            .sum::<f32>()
            - rest
    };
    let settings = PhysicsSettings {
        iterations,
        ..Default::default()
    };
    let (start, mut highest) = (energy(&world, &physics), f32::MIN);
    for _ in 0..(40.0 / dt) as usize {
        physics.step(&mut world, &settings, dt);
        highest = highest.max(energy(&world, &physics));
        if !highest.is_finite() {
            break;
        }
    }
    let at = |e: EntityHandle, l: Vec3| {
        let t = &world.get(e).unwrap().transform;
        Vec3::from_array(t.translation) + Quat::from_array(t.rotation) * l
    };
    let error = bars
        .windows(2)
        .fold(at(bars[0], left).distance(top), |w, p| {
            w.max(at(p[0], right).distance(at(p[1], left)))
        });
    (highest / start, error)
}

#[test]
fn hard_joints_gain_no_energy_in_harsh_chains() {
    for (n, load, iterations, dt, limit) in [
        (10, 1.0, 6, 1.0 / 60.0, 1.01),
        (30, 1.0, 6, 1.0 / 60.0, 1.01),
        (30, 100.0, 6, 1.0 / 60.0, 1.01),
        (10, 1.0, 3, 1.0 / 60.0, 1.01),
        (10, 1.0, 6, 1.0 / 30.0, 1.05),
    ] {
        let (highest, _) = chain_energy(n, load, iterations, dt, 0.0);
        assert!(highest <= limit, "{n} bars, load {load}: gained {highest}x");
    }
}

#[test]
fn fast_thin_bodies_on_joints_do_not_explode() {
    // Explicit gyroscopic steps once fed these energy without bound.
    let (highest, error) = chain_energy(3, 1.0, 6, 1.0 / 60.0, 60.0);
    assert!(highest < 3.0 && error < 0.01, "{highest}x, error {error}");
    let (highest, error) = chain_energy(30, 1.0, 6, 1.0 / 30.0, 0.0);
    assert!(highest < 2.0 && error < 0.1, "{highest}x, error {error}");
}

#[test]
fn a_pendulum_keeps_its_swing_and_period() {
    // Backward Euler would leave exp(-ω²ht/2) = 29% of the swing after 30 s.
    let (length, angle) = (2.0f32, 0.2f32);
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let top = Vec3::new(0.0, 0.0, 10.0);
    let offset = Vec3::new(angle.sin(), 0.0, -angle.cos()) * length;
    let collider = Collider::Sphere { radius: 0.1 };
    let ball = PhysicsBody::new(RigidBody::Dynamic, collider);
    let entity = spawn(&mut world, &mut physics, top + offset, ball);
    physics
        .add_joint(&world, Joint::new(None, entity, top, -offset))
        .unwrap();
    let (mut crossings, mut last, mut peak) = (Vec::new(), offset.x, 0.0f32);
    for step in 0..1800 {
        run(&mut world, &mut physics, 1);
        let x = position(&world, entity).x;
        if last < 0.0 && x >= 0.0 {
            crossings.push(step as f32 / 60.0);
        }
        last = x;
        if step >= 1680 {
            peak = peak.max(x.abs());
        }
    }
    assert!(
        peak > 0.95 * offset.x,
        "swing fell to {peak} of {}",
        offset.x
    );
    let period = (crossings[crossings.len() - 1] - crossings[0]) / (crossings.len() - 1) as f32;
    let expected = 2.8455;
    assert!((period - expected).abs() < 0.02, "period {period}");
}

#[test]
fn jointed_bodies_sleep_and_wake_as_one_island() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let floor = ground(&mut world, &mut physics);
    let a = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 0.5),
        cube(0.5),
    );
    let b = spawn(
        &mut world,
        &mut physics,
        Vec3::new(1.5, 0.0, 0.5),
        cube(0.5),
    );
    physics
        .add_joint(
            &world,
            Joint::new(
                Some(a),
                b,
                Vec3::new(0.75, 0.0, 0.0),
                Vec3::new(-0.75, 0.0, 0.0),
            ),
        )
        .unwrap();
    run(&mut world, &mut physics, 180);
    assert!(physics.is_sleeping(a) && physics.is_sleeping(b));
    physics.remove(floor);
    run(&mut world, &mut physics, 30);
    assert!(
        position(&world, a).z < 0.0 && position(&world, b).z < 0.0,
        "both fell"
    );
}

#[test]
fn welded_bodies_keep_their_relative_rotation() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let a = spawn(&mut world, &mut physics, Vec3::new(0.5, 0.0, 5.0), bar());
    let b = spawn(&mut world, &mut physics, Vec3::new(1.5, 0.0, 5.0), bar());
    physics
        .add_joint(
            &world,
            Joint::new(None, a, Vec3::new(0.0, 0.0, 5.0), Vec3::new(-0.5, 0.0, 0.0)),
        )
        .unwrap();
    let mut weld = Joint::new(
        Some(a),
        b,
        Vec3::new(0.5, 0.0, 0.0),
        Vec3::new(-0.5, 0.0, 0.0),
    );
    weld.angular_stiffness = f32::INFINITY;
    physics.add_joint(&world, weld).unwrap();
    for _ in 0..120 {
        run(&mut world, &mut physics, 1);
    }
    let rotation = |e| Quat::from_array(world.get(e).unwrap().transform.rotation);
    let relative = rotation(a).conjugate() * rotation(b);
    assert!(
        relative.angle_between(Quat::IDENTITY) < 0.01,
        "bent by {relative}"
    );
    assert!(position(&world, b).z < 4.5, "swung as one bar");
}

#[test]
fn soft_joint_stretches_by_weight_over_stiffness() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let body = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 5.0),
        cube(0.5),
    );
    let mut joint = Joint::new(
        None,
        body,
        Vec3::new(0.0, 0.0, 5.5),
        Vec3::new(0.0, 0.0, 0.5),
    );
    joint.linear_stiffness = 1.0e6; // 1000 kg: 9.81 mm under gravity
    physics.add_joint(&world, joint).unwrap();
    run(&mut world, &mut physics, 600);
    let stretch = 5.0 - position(&world, body).z;
    assert!((stretch - 0.00981).abs() < 0.001, "stretched {stretch}");
}

#[test]
fn stiffness_ratio_springs_converge_in_five_iterations() {
    // AVBD Fig. 2: blocks on springs 10,000x apart in stiffness. Plain VBD
    // overstretches the soft spring; ramped penalties (Eq. 16) do not.
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let anchor = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 10.0),
        PhysicsBody::new(
            RigidBody::Static,
            Collider::Cuboid {
                half_extents: Vec3::splat(0.5),
            },
        ),
    );
    let middle = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 7.0),
        cube(0.5),
    );
    let bottom = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 4.0),
        cube(0.5),
    );
    let (down, up) = (Vec3::new(0.0, 0.0, -0.5), Vec3::new(0.0, 0.0, 0.5));
    let spring = |a, b, stiffness| Spring {
        body_a: Some(a),
        body_b: b,
        anchor_a: down,
        anchor_b: up,
        stiffness,
        rest_length: 2.0,
    };
    // Stiff on top (two blocks of weight), soft below (one block).
    physics
        .add_spring(&world, spring(anchor, middle, 1.0e9))
        .unwrap();
    physics
        .add_spring(&world, spring(middle, bottom, 1.0e5))
        .unwrap();
    let settings = PhysicsSettings {
        iterations: 5,
        ..Default::default()
    };
    for _ in 0..900 {
        physics.step(&mut world, &settings, 1.0 / 60.0);
    }
    let weight = 1000.0 * 9.81;
    let soft = position(&world, middle).z - 0.5 - (position(&world, bottom).z + 0.5) - 2.0;
    let stiff = 9.5 - (position(&world, middle).z + 0.5) - 2.0;
    assert!(
        (soft - weight / 1.0e5).abs() < 0.01,
        "soft spring stretched {soft}"
    );
    assert!(stiff.abs() < 0.01, "stiff spring stretched {stiff}");
}

#[test]
fn overloaded_joint_breaks() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let a = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 5.0),
        PhysicsBody::new(
            RigidBody::Static,
            Collider::Cuboid {
                half_extents: Vec3::splat(0.5),
            },
        ),
    );
    let b = spawn(
        &mut world,
        &mut physics,
        Vec3::new(1.0, 0.0, 5.0),
        cube(0.5),
    );
    let mut weld = Joint::new(
        Some(a),
        b,
        Vec3::new(0.5, 0.0, 0.0),
        Vec3::new(-0.5, 0.0, 0.0),
    );
    weld.angular_stiffness = f32::INFINITY;
    weld.angular_strength = Strength::brittle(1000.0);
    let id = physics.add_joint(&world, weld).unwrap();
    run(&mut world, &mut physics, 60);
    assert_eq!(physics.is_broken(id), Some(true));
    assert!(position(&world, b).z < 4.0, "fell once broken");
    assert!(physics.remove_link(id));
    assert_eq!(physics.is_broken(id), None);
}

#[test]
fn jointed_bodies_do_not_collide() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let settings = PhysicsSettings {
        gravity: Vec3::ZERO,
        ..Default::default()
    };
    // Overlapping halves of one rigid assembly.
    let a = spawn(&mut world, &mut physics, Vec3::ZERO, cube(0.5));
    let b = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.6, 0.0, 0.0),
        cube(0.5),
    );
    let mut weld = Joint::new(
        Some(a),
        b,
        Vec3::new(0.3, 0.0, 0.0),
        Vec3::new(-0.3, 0.0, 0.0),
    );
    weld.angular_stiffness = f32::INFINITY;
    physics.add_joint(&world, weld).unwrap();
    for _ in 0..60 {
        physics.step(&mut world, &settings, 1.0 / 60.0);
    }
    assert!(position(&world, a).length() < 1e-3, "pushed apart");
    assert!(
        position(&world, b).distance(Vec3::new(0.6, 0.0, 0.0)) < 1e-3,
        "pushed apart"
    );
    assert_eq!(physics.contact_count(), 0);
}

#[test]
fn static_friction_holds_until_the_box_slides() {
    // Sec. 3.3: tan(0.38) is about 0.4, between kinetic 0.2 and static 0.6.
    let slide = |push: f32| {
        let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
        let tilt = Quat::from_rotation_x(0.38);
        let slope = world.spawn(Entity {
            transform: Transform {
                rotation: tilt.to_array(),
                ..Default::default()
            },
            ..Entity::default()
        });
        let plank = Collider::Cuboid {
            half_extents: Vec3::new(20.0, 20.0, 0.5),
        };
        let mut body = PhysicsBody::new(RigidBody::Static, plank);
        body.material.friction = 0.2;
        physics.insert(&world, slope, body).unwrap();
        let start = tilt * Vec3::new(0.0, 0.0, 1.0);
        let boxed = world.spawn(Entity {
            transform: Transform {
                translation: start.to_array(),
                rotation: tilt.to_array(),
                ..Default::default()
            },
            ..Entity::default()
        });
        let mut body = cube(0.5);
        body.material.friction = 0.2;
        body.velocity.linear = tilt * Vec3::new(0.0, -push, 0.0);
        physics.insert(&world, boxed, body).unwrap();
        run(&mut world, &mut physics, 120);
        position(&world, boxed).distance(start)
    };
    assert!(slide(0.0) < 0.05, "static friction let it slide");
    assert!(slide(1.0) > 2.0, "kinetic friction held it");
}

#[test]
fn dropped_pile_stays_on_the_ground() {
    // Regression: rescaling the normal row of a penetrating contact (Eq. 14)
    // let impacts bounce bodies between iterations and sink through the floor.
    // The compare example's 512-body rain scene, where it first showed.
    let jitter = |seed: usize| {
        let x = (seed as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .rotate_left(17);
        (x.wrapping_mul(0xBF58_476D_1CE4_E5B9) >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    };
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    ground(&mut world, &mut physics);
    let mut bodies = Vec::new();
    for z in 0..8 {
        for y in 0..8 {
            for x in 0..8 {
                let n = (bodies.len() + 1) * 3;
                let p = Vec3::new(x as f32, y as f32, z as f32) * 1.2
                    + Vec3::new(-4.8, -4.8, 1.0)
                    + Vec3::new(jitter(n), jitter(n + 1), jitter(n + 2)) * 0.08;
                let body = if (x + y + z) % 2 == 0 {
                    cube(0.5)
                } else {
                    PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius: 0.5 })
                };
                bodies.push(spawn(&mut world, &mut physics, p, body));
            }
        }
    }
    let mut lowest = f32::MAX;
    for _ in 0..120 {
        run(&mut world, &mut physics, 1);
        lowest = bodies
            .iter()
            .map(|&e| position(&world, e).z)
            .fold(lowest, f32::min);
    }
    assert!(lowest > 0.35, "a body sank to {lowest}");
}

/// A 1000 kg cube hanging from a world point by `joint`; its weight is 9810 N.
fn hanging(setup: impl FnOnce(&mut Joint)) -> (World, PhysicsWorld, EntityHandle, LinkId) {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let top = Vec3::new(0.0, 0.0, 20.0);
    let body = spawn(&mut world, &mut physics, top - Vec3::Z * 2.0, cube(0.5));
    let mut joint = Joint::new(None, body, top, Vec3::Z * 2.0);
    setup(&mut joint);
    let id = physics.add_joint(&world, joint).unwrap();
    (world, physics, body, id)
}

#[test]
fn a_joint_reports_the_load_it_carries() {
    let (mut world, mut physics, _, id) = hanging(|_| {});
    run(&mut world, &mut physics, 120);
    let status = physics.joint_status(id).unwrap();
    assert!((status.force - 9810.0).abs() < 100.0, "{status:?}");
    assert_eq!(status.stage, JointStage::Intact);
}

#[test]
fn a_ball_joint_holds_below_its_break_force_and_snaps_above_it() {
    let hold = |joint: &mut Joint| joint.linear_strength = Strength::brittle(10_500.0);
    let (mut world, mut physics, _, id) = hanging(hold);
    run(&mut world, &mut physics, 300);
    assert_eq!(physics.is_broken(id), Some(false));

    let snap = |joint: &mut Joint| joint.linear_strength = Strength::brittle(9_000.0);
    let (mut world, mut physics, body, id) = hanging(snap);
    let mut events = Vec::new();
    for _ in 0..60 {
        run(&mut world, &mut physics, 1);
        events.extend_from_slice(physics.joint_events());
    }
    assert_eq!(physics.is_broken(id), Some(true));
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(events[0].stage, JointStage::Broken);
    assert_eq!((events[0].link, events[0].body_b), (id, body));
    assert!(position(&world, body).z < 16.0, "falls once broken");
}

#[test]
fn a_ductile_joint_stretches_for_good_and_then_holds_or_snaps() {
    // Hardening from 9 to 20 kN over 0.5 m carries 9.81 kN after 3.7 cm, and a
    // load applied at once overshoots to twice that, as in any spring.
    let holds = |joint: &mut Joint| {
        joint.linear_strength = Strength {
            yield_load: 9_000.0,
            break_load: 20_000.0,
            ductility: 0.5,
        }
    };
    let (mut world, mut physics, body, id) = hanging(holds);
    run(&mut world, &mut physics, 300);
    let status = physics.joint_status(id).unwrap();
    assert_eq!(status.stage, JointStage::Yielded);
    let stretch = 18.0 - position(&world, body).z;
    assert!((stretch - 0.074).abs() < 0.015, "stretch {stretch}");

    // Never hardens enough for the weight, so it runs out of ductility.
    let snaps = |joint: &mut Joint| {
        joint.linear_strength = Strength {
            yield_load: 9_000.0,
            break_load: 9_500.0,
            ductility: 0.2,
        }
    };
    let (mut world, mut physics, body, id) = hanging(snaps);
    let (mut stages, mut stretch) = (Vec::new(), 0.0);
    for _ in 0..240 {
        run(&mut world, &mut physics, 1);
        stages.extend(physics.joint_events().iter().map(|e| e.stage));
        if physics.is_broken(id) == Some(false) {
            stretch = 18.0 - position(&world, body).z;
        }
    }
    assert_eq!(stages, [JointStage::Yielded, JointStage::Broken]);
    assert!(stretch > 0.1 && stretch < 0.4, "broke at {stretch}");
}

#[test]
fn a_chain_snaps_at_the_link_carrying_the_most() {
    // Every link is as strong; the top one carries all the links below it.
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let top = Vec3::new(0.0, 0.0, 30.0);
    let (up, down) = (Vec3::X * -0.5, Vec3::X * 0.5);
    let bars: Vec<_> = (0..5)
        .map(|i| {
            let p = top + Vec3::new(0.5 + i as f32, 0.0, 0.0);
            spawn(&mut world, &mut physics, p, bar())
        })
        .collect();
    let strong = Strength::brittle(5_000.0);
    let mut first = Joint::new(None, bars[0], top, up);
    first.linear_strength = strong;
    let mut ids = vec![physics.add_joint(&world, first).unwrap()];
    for pair in bars.windows(2) {
        let mut joint = Joint::new(Some(pair[0]), pair[1], down, up);
        joint.linear_strength = strong;
        ids.push(physics.add_joint(&world, joint).unwrap());
    }
    // 160 kg bars: 7.8 kN hanging still, more while swinging.
    let mut broke = None;
    for _ in 0..600 {
        run(&mut world, &mut physics, 1);
        if let Some(event) = physics.joint_events().first() {
            broke = Some(event.link);
            break;
        }
    }
    assert_eq!(broke, Some(ids[0]));
}

/// A 2 m bar pinned at one end, level, free to fall around Y: B's rest frame is
/// the world's. Returns the bar's turn around Y after two seconds.
fn levered(setup: impl FnOnce(&mut Joint)) -> (World, PhysicsWorld, EntityHandle, LinkId) {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let top = Vec3::new(0.0, 0.0, 20.0);
    let shape = Collider::Cuboid {
        half_extents: Vec3::new(1.0, 0.1, 0.1),
    };
    let body = PhysicsBody::new(RigidBody::Dynamic, shape);
    let body = spawn(&mut world, &mut physics, top + Vec3::X, body);
    let mut joint = Joint::new(None, body, top, -Vec3::X);
    setup(&mut joint);
    let id = physics.add_joint(&world, joint).unwrap();
    (world, physics, body, id)
}

fn turn_y(world: &World, body: EntityHandle) -> f32 {
    let rotation = Quat::from_array(world.get(body).unwrap().transform.rotation);
    (rotation * Vec3::X).z.atan2((rotation * Vec3::X).x).abs()
}

fn hinge(joint: &mut Joint) {
    joint.limit_min = Vec3::new(0.0, -0.5, 0.0);
    joint.limit_max = Vec3::new(0.0, 0.5, 0.0);
}

#[test]
fn a_rotation_limit_stops_a_hinge_at_its_range() {
    let (mut world, mut physics, body, id) = levered(hinge);
    let mut worst = 0.0f32;
    for _ in 0..300 {
        run(&mut world, &mut physics, 1);
        worst = worst.max(turn_y(&world, body));
    }
    let settled = turn_y(&world, body);
    assert!((settled - 0.5).abs() < 0.01, "settled at {settled}");
    assert!(worst < 0.56, "swung to {worst}");
    let rotation = Quat::from_array(world.get(body).unwrap().transform.rotation);
    assert!((rotation * Vec3::Y).y > 0.999, "locked axes held");
    // An 80 kg bar's weight at 1 m, with the bar at 0.5 rad.
    let torque = physics.joint_status(id).unwrap().torque;
    let expected = 80.0 * 9.81 * 0.5f32.cos();
    assert!(
        (torque - expected).abs() < 0.05 * expected,
        "torque {torque}"
    );
}

#[test]
fn an_overloaded_limit_bends_for_good_then_rips() {
    // 689 N·m at the stop, but the bar arrives moving. Energy balance: the work
    // 500s + 250s² of bending s past the stop equals the weight's 785 sin(0.5 + s).
    let bends = |joint: &mut Joint| {
        hinge(joint);
        joint.angular_strength = Strength {
            yield_load: 500.0,
            break_load: 1_000.0,
            ductility: 1.0,
        };
    };
    let (mut world, mut physics, body, id) = levered(bends);
    run(&mut world, &mut physics, 300);
    assert_eq!(physics.joint_status(id).unwrap().stage, JointStage::Yielded);
    let bent = turn_y(&world, body);
    assert!((bent - 1.5).abs() < 0.1, "bent to {bent}");

    let rips = |joint: &mut Joint| {
        hinge(joint);
        joint.angular_strength = Strength::brittle(400.0);
    };
    let (mut world, mut physics, body, id) = levered(rips);
    run(&mut world, &mut physics, 120);
    assert_eq!(physics.is_broken(id), Some(true));
    assert!(position(&world, body).z < 15.0, "fell away");
}

#[test]
fn rubber_stretches_stiffens_and_snaps() {
    // Linear: 9810 N at 10 kN/m stretches 0.98 m. Doubling every 0.5 m settles
    // where k(1 + (s/0.5)²)s = 9810: about 0.5 m.
    let linear = |joint: &mut Joint| joint.linear_stiffness = 10_000.0;
    let (mut world, mut physics, body, _) = hanging(linear);
    run(&mut world, &mut physics, 3000);
    let plain = 18.0 - position(&world, body).z;
    assert!((plain - 0.98).abs() < 0.03, "linear stretch {plain}");

    let stiffening = |joint: &mut Joint| {
        joint.linear_stiffness = 10_000.0;
        joint.stiffening = 0.5;
    };
    let (mut world, mut physics, body, _) = hanging(stiffening);
    run(&mut world, &mut physics, 3000);
    let stiff = 18.0 - position(&world, body).z;
    assert!((stiff - 0.5).abs() < 0.06, "stiffened stretch {stiff}");

    let snaps = |joint: &mut Joint| {
        joint.linear_stiffness = 10_000.0;
        joint.linear_strength = Strength::brittle(5_000.0);
    };
    let (mut world, mut physics, _, id) = hanging(snaps);
    run(&mut world, &mut physics, 120);
    assert_eq!(physics.is_broken(id), Some(true));
}

#[test]
fn a_limit_stop_does_not_throw_a_body_back() {
    let (mut world, mut physics, body, _) = levered(hinge);
    let (mut reached, mut rebound) = (false, f32::MAX);
    for _ in 0..300 {
        run(&mut world, &mut physics, 1);
        let turn = turn_y(&world, body);
        reached |= turn >= 0.5;
        if reached {
            rebound = rebound.min(turn);
        }
    }
    assert!(rebound > 0.47, "rebounded to {rebound}");
}

fn arm_muscle(activation: f32) -> Muscle {
    Muscle {
        axis: Vec3::Y,
        section: 20.0,
        specific_tension: 30.0,
        moment_arm: 2.0,
        activation,
        target: 0.0,
        optimal_angle: 0.0,
        angle_width: 3.0,
        max_speed: 15.0,
        stiffness: 20_000.0,
    }
}

#[test]
fn hill_muscle_torque_follows_size_activation_length_and_speed() {
    let muscle = arm_muscle(1.0);
    let peak = 20.0 * 30.0 * 2.0;
    assert!((muscle.torque_limit(0.0, 0.0) - peak).abs() < 0.01);
    assert!((arm_muscle(0.5).torque_limit(0.0, 0.0) - peak / 2.0).abs() < 0.01);
    let stretched = muscle.torque_limit(3.0, 0.0);
    assert!((stretched - peak / std::f32::consts::E).abs() < 1.0);
    assert_eq!(muscle.torque_limit(0.0, 15.0), 0.0);
    assert!(muscle.torque_limit(0.0, 5.0) < peak / 2.0);
    let forced = muscle.torque_limit(0.0, -1000.0);
    assert!(forced > 1.45 * peak && forced <= 1.5 * peak, "{forced}");
}

#[test]
fn the_elbow_preset_gives_its_stated_torque() {
    let elbow = Muscle::elbow_flexors();
    let peak = elbow.torque_limit(elbow.optimal_angle, 0.0);
    assert!((peak - 13.3).abs() < 0.1, "{peak}");
}

#[test]
fn a_tensed_arm_holds_what_breaks_a_limp_one() {
    // The bar's 785 N·m at level, held by a muscle or falling on a 400 N·m stop.
    let setup = |activation: f32| {
        move |joint: &mut Joint| {
            hinge(joint);
            joint.angular_strength = Strength::brittle(400.0);
            joint.limit_stiffness = 20_000.0;
            joint.muscle = Some(arm_muscle(activation));
        }
    };
    let (mut world, mut physics, body, id) = levered(setup(1.0));
    run(&mut world, &mut physics, 300);
    assert_eq!(physics.is_broken(id), Some(false));
    assert!(turn_y(&world, body) < 0.1, "held near level");

    let (mut world, mut physics, _, id) = levered(setup(0.0));
    run(&mut world, &mut physics, 300);
    assert_eq!(physics.is_broken(id), Some(true));

    // Relaxing a tensed arm lets the same load through.
    let (mut world, mut physics, _, id) = levered(setup(1.0));
    run(&mut world, &mut physics, 60);
    assert!(physics.drive_muscle(id, 0.0, 0.0));
    run(&mut world, &mut physics, 240);
    assert_eq!(physics.is_broken(id), Some(true));
}

#[test]
fn a_muscle_moves_a_joint_to_its_target() {
    let driven = |joint: &mut Joint| joint.muscle = Some(arm_muscle(1.0));
    let (mut world, mut physics, body, id) = levered(driven);
    assert!(physics.drive_muscle(id, 1.0, -0.5));
    run(&mut world, &mut physics, 300);
    let turn = turn_y(&world, body);
    assert!((turn - 0.5).abs() < 0.08, "raised to {turn}");
}

#[test]
fn a_compliant_stop_spreads_an_impact_a_rigid_one_takes_at_once() {
    let peak = |stiffness: f32| {
        let setup = |joint: &mut Joint| {
            hinge(joint);
            joint.limit_stiffness = stiffness;
        };
        let (mut world, mut physics, body, id) = levered(setup);
        let (mut highest, mut furthest) = (0.0f32, 0.0f32);
        for _ in 0..300 {
            run(&mut world, &mut physics, 1);
            highest = highest.max(physics.joint_status(id).unwrap().torque);
            furthest = furthest.max(turn_y(&world, body));
        }
        (highest, furthest)
    };
    let (rigid, _) = peak(f32::INFINITY);
    let (soft, furthest) = peak(5_000.0);
    assert!(soft < 0.7 * rigid, "peak {soft} against {rigid}");
    assert!(furthest > 0.55 && furthest < 1.0, "gave to {furthest}");
}

/// Highest center height after the first bounce of a ball dropped from 2 m.
fn bounce_peak(restitution: f32) -> f32 {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    ground(&mut world, &mut physics);
    let mut ball = PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius: 0.5 });
    ball.material.restitution = restitution;
    let entity = spawn(&mut world, &mut physics, Vec3::new(0.0, 0.0, 2.5), ball);
    let (mut bounced, mut peak) = (false, 0.0f32);
    for _ in 0..240 {
        run(&mut world, &mut physics, 1);
        let v = physics.get(entity).unwrap().velocity.linear.z;
        bounced |= v > 0.5;
        if bounced {
            peak = peak.max(position(&world, entity).z - 0.5);
        }
    }
    peak
}

#[test]
fn restitution_bounces_a_ball_to_its_squared_share_of_the_drop() {
    for e in [0.5f32, 0.8] {
        let peak = bounce_peak(e);
        let expected = e * e * 2.0;
        assert!((peak - expected).abs() < 0.15 * expected, "{e}: {peak} for {expected}");
    }
    assert!(bounce_peak(0.0) < 0.02, "no restitution, no bounce");
}

/// A capsule dropped onto the floor, with its core along `axis`.
fn dropped_capsule(axis: Vec3) -> (World, PhysicsWorld, EntityHandle) {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    ground(&mut world, &mut physics);
    let capsule = Collider::Capsule {
        radius: 0.3,
        half_height: 0.6,
    };
    let entity = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 1.5),
        PhysicsBody::new(RigidBody::Dynamic, capsule),
    );
    world.get_mut(entity).unwrap().transform.rotation =
        Quat::from_rotation_arc(Vec3::Z, axis).to_array();
    run(&mut world, &mut physics, 240);
    (world, physics, entity)
}

#[test]
fn a_capsule_rests_lying_on_its_side_and_standing_on_a_cap() {
    let (world, physics, entity) = dropped_capsule(Vec3::Y);
    let z = position(&world, entity).z;
    assert!((z - 0.3).abs() < 0.01, "lying at {z}");
    let rotation = Quat::from_array(world.get(entity).unwrap().transform.rotation);
    assert!((rotation * Vec3::Z).z.abs() < 0.01, "stays level");
    assert!(physics.get(entity).unwrap().velocity.linear.length() < 0.01);

    let (world, _, entity) = dropped_capsule(Vec3::Z);
    let z = position(&world, entity).z;
    assert!((z - 0.9).abs() < 0.01, "standing at {z}");
}

#[test]
fn capsules_rest_crossed_on_capsules_and_hold_spheres() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let capsule = Collider::Capsule {
        radius: 0.3,
        half_height: 0.6,
    };
    let log = spawn(
        &mut world,
        &mut physics,
        Vec3::ZERO,
        PhysicsBody::new(RigidBody::Static, capsule),
    );
    world.get_mut(log).unwrap().transform.rotation =
        Quat::from_rotation_arc(Vec3::Z, Vec3::X).to_array();
    let top = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 1.0),
        PhysicsBody::new(RigidBody::Dynamic, capsule),
    );
    world.get_mut(top).unwrap().transform.rotation =
        Quat::from_rotation_arc(Vec3::Z, Vec3::Y).to_array();
    run(&mut world, &mut physics, 180);
    let z = position(&world, top).z;
    assert!((z - 0.6).abs() < 0.01, "crossed at {z}");

    let ball = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 2.0),
        PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius: 0.2 }),
    );
    run(&mut world, &mut physics, 30);
    let z = position(&world, ball).z;
    assert!(z > 0.6 + 0.3 + 0.2 - 0.02, "ball held at {z}");
}

/// Exact flat ground at z = 0.
struct Flat;

impl Ground for Flat {
    fn distance(&self, p: Vec3) -> f32 {
        p.z
    }
    fn normal(&self, _: Vec3) -> Vec3 {
        Vec3::Z
    }
}

#[test]
fn a_capsule_lies_level_on_ground() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let field = PhysicsBody::new(RigidBody::Static, Collider::Ground);
    spawn(&mut world, &mut physics, Vec3::ZERO, field);
    physics.set_ground(std::sync::Arc::new(Flat), Bounds::EMPTY);
    let capsule = Collider::Capsule {
        radius: 0.3,
        half_height: 1.0,
    };
    let entity = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 1.0),
        PhysicsBody::new(RigidBody::Dynamic, capsule),
    );
    world.get_mut(entity).unwrap().transform.rotation =
        Quat::from_rotation_arc(Vec3::Z, Vec3::X).to_array();
    run(&mut world, &mut physics, 240);
    let z = position(&world, entity).z;
    assert!((z - 0.3).abs() < 0.01, "lying at {z}");
    let rotation = Quat::from_array(world.get(entity).unwrap().transform.rotation);
    assert!((rotation * Vec3::Z).z.abs() < 0.01, "stays level");
    assert!(physics.is_sleeping(entity), "a capsule at rest must sleep");
}

#[test]
fn step_profile_times_each_phase_within_the_step() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    ground(&mut world, &mut physics);
    // Enough touching boxes that every phase takes measurable time.
    for i in 0..64 {
        let at = Vec3::new((i % 8) as f32 * 1.5, (i / 8) as f32 * 1.5, 0.49);
        spawn(&mut world, &mut physics, at, cube(0.5));
    }
    run(&mut world, &mut physics, 2);
    let start = std::time::Instant::now();
    run(&mut world, &mut physics, 1);
    let measured = start.elapsed();
    let profile = physics.last_profile();
    assert!(profile.contacts > 0, "boxes rest on the ground");
    assert_eq!(profile.bodies, 65);
    for (name, time) in [
        ("broad", profile.broad_phase),
        ("narrow", profile.narrow_phase),
        ("solve", profile.solve),
    ] {
        assert!(!time.is_zero(), "{name} phase was not timed");
    }
    assert!(profile.total() <= measured, "{:?} > {measured:?}", profile.total());
}
