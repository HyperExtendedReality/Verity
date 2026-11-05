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
    weld.fracture = 1000.0;
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
