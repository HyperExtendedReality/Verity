use glam::{Quat, Vec3};
use verity_core::{Entity, EntityHandle, Material, Transform, World};
use verity_physics::*;

const DT: f32 = 1.0 / 60.0;

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

fn run(world: &mut World, physics: &mut PhysicsWorld, steps: usize) {
    let settings = PhysicsSettings::default();
    for _ in 0..steps {
        physics.step(world, &settings, DT);
    }
}

fn ground(world: &mut World, physics: &mut PhysicsWorld) {
    let shape = Collider::Cuboid {
        half_extents: Vec3::new(50.0, 50.0, 0.5),
    };
    let body = PhysicsBody::new(RigidBody::Static, shape);
    spawn(world, physics, Vec3::new(0.0, 0.0, -0.5), body);
}

fn ball(radius: f32, density: f32) -> PhysicsBody {
    let mut body = PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius });
    body.material.density = density;
    body
}

fn z(world: &World, entity: EntityHandle) -> f32 {
    world.get(entity).unwrap().transform.translation[2]
}

/// A 2 m cloth, level, 4 m up.
fn sheet(material: SoftMaterial) -> SoftBody {
    let center = Vec3::new(0.0, 0.0, 4.0);
    SoftBody::cloth(center, Quat::IDENTITY, [2.0, 2.0], [20, 20], material)
}

#[test]
fn a_pinned_cloth_hangs_without_stretching_or_gaining_energy() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let mut cloth = sheet(SoftMaterial::cloth());
    assert_eq!(cloth.pin(|p| p.y > 0.99, f32::INFINITY), 21);
    let id = physics.add_soft_body(cloth);
    let mut lowest = f32::MAX;
    for _ in 0..1500 {
        run(&mut world, &mut physics, 1);
        let body = physics.soft_body(id).unwrap();
        lowest = body.positions().iter().fold(lowest, |l, p| l.min(p.z));
    }
    let body = physics.soft_body(id).unwrap();
    assert!(body.positions().iter().all(|p| p.is_finite()));
    let end = body.positions().iter().fold(f32::MAX, |l, p| l.min(p.z));
    // Hanging from a line 2 m long, it reaches 2 m below that line.
    assert!((end - 2.0).abs() < 0.15, "hangs to {end}");
    assert!(lowest > 1.5, "swung down to {lowest}");
    let speed = body
        .velocities()
        .iter()
        .fold(0.0f32, |s, v| s.max(v.length()));
    assert!(speed < 0.5, "still moving at {speed}");
}

#[test]
fn cloth_drapes_over_a_sphere_and_stays_on_the_outside() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let rock = PhysicsBody::new(RigidBody::Static, Collider::Sphere { radius: 0.6 });
    let center = Vec3::new(0.0, 0.0, 3.0);
    spawn(&mut world, &mut physics, center, rock);
    let id = physics.add_soft_body(sheet(SoftMaterial::cloth()));
    run(&mut world, &mut physics, 300);
    let body = physics.soft_body(id).unwrap();
    let closest = (body.positions().iter())
        .map(|p| p.distance(center))
        .fold(f32::MAX, f32::min);
    assert!(closest > 0.59, "reached {closest} of the 0.6 m sphere");
    let top = body.positions().iter().fold(f32::MIN, |t, p| t.max(p.z));
    assert!(top > 3.55 && top < 3.8, "top rests at {top}");
    let hanging = body.positions().iter().fold(f32::MAX, |l, p| l.min(p.z));
    assert!(hanging < 3.0, "the edges fall past it: {hanging}");
}

#[test]
fn friction_holds_cloth_on_a_slope_it_would_slide_off_without() {
    let slide = |friction: f32| {
        let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
        let tilt = Quat::from_rotation_y(0.3);
        let shape = Collider::Cuboid {
            half_extents: Vec3::new(10.0, 10.0, 0.5),
        };
        let slab = PhysicsBody::new(RigidBody::Static, shape);
        let slab = spawn(&mut world, &mut physics, Vec3::ZERO, slab);
        world.get_mut(slab).unwrap().transform.rotation = tilt.to_array();
        let mut material = SoftMaterial::cloth();
        material.friction = friction;
        let center = tilt * Vec3::new(0.0, 0.0, 0.6);
        let cloth = SoftBody::cloth(center, tilt, [1.0, 1.0], [8, 8], material);
        let id = physics.add_soft_body(cloth);
        run(&mut world, &mut physics, 120);
        let body = physics.soft_body(id).unwrap();
        let sum: Vec3 = body.positions().iter().copied().sum();
        (sum / body.positions().len() as f32).distance(center)
    };
    let (held, slid) = (slide(1.0), slide(0.0));
    assert!(held < 0.2, "held cloth moved {held}");
    assert!(slid > 2.0, "slippery cloth moved {slid}");
}

#[test]
fn a_ball_is_caught_by_cloth_and_pushes_it_down() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let mut cloth = sheet(SoftMaterial::cloth());
    cloth.pin(|p| p.x.abs() > 0.99 || p.y.abs() > 0.99, f32::INFINITY);
    let id = physics.add_soft_body(cloth);
    let body = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 4.6),
        ball(0.25, 50.0),
    );
    run(&mut world, &mut physics, 300);
    let height = z(&world, body);
    assert!(height > 3.0 && height < 4.0, "ball rests at {height}");
    let sheet = physics.soft_body(id).unwrap();
    let lowest = sheet.positions().iter().fold(f32::MAX, |l, p| l.min(p.z));
    assert!(lowest < 3.9, "the cloth sagged to {lowest}");
    assert_eq!(sheet.tears(), 0);
}

#[test]
fn a_heavy_ball_tears_through_cloth_into_pieces() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let mut cloth = sheet(SoftMaterial::cloth());
    cloth.pin(|p| p.x.abs() > 0.99 || p.y.abs() > 0.99, f32::INFINITY);
    let particles = cloth.positions().len();
    let id = physics.add_soft_body(cloth);
    let body = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 6.0),
        ball(0.25, 8000.0),
    );
    run(&mut world, &mut physics, 300);
    let sheet = physics.soft_body(id).unwrap();
    assert!(sheet.tears() > 10, "{} edges tore", sheet.tears());
    assert!(
        sheet.positions().len() > particles,
        "tearing duplicates particles"
    );
    assert!(
        z(&world, body) < 0.0,
        "the ball fell through: {}",
        z(&world, body)
    );
    assert!(sheet.positions().iter().all(|p| p.is_finite()));
}

#[test]
fn tearing_never_flings_fragments() {
    // Fine cloth and a heavy ball: many particles torn free at once, each
    // lighter than the last, none balanced by its neighbours.
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let center = Vec3::new(0.0, 0.0, 3.0);
    let mut cloth = SoftBody::cloth(
        center,
        Quat::IDENTITY,
        [3.0, 3.0],
        [28, 28],
        SoftMaterial::cloth(),
    );
    cloth.pin(|p| p.x.abs() > 1.49 || p.y.abs() > 1.49, f32::INFINITY);
    let id = physics.add_soft_body(cloth);
    spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 6.0),
        ball(0.3, 6000.0),
    );
    for step in 0..300 {
        run(&mut world, &mut physics, 1);
        let body = physics.soft_body(id).unwrap();
        let fastest = body
            .velocities()
            .iter()
            .fold(0.0f32, |s, v| s.max(v.length()));
        assert!(
            fastest < 80.0,
            "a fragment reached {fastest} m/s at step {step}"
        );
    }
    assert!(physics.soft_body(id).unwrap().tears() > 10);
}

#[test]
fn pulling_a_sheet_apart_makes_two_pieces_and_keeps_its_mass() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let mut material = SoftMaterial::cloth();
    material.strength = Strength::brittle(30.0);
    let mut cloth = sheet(material);
    cloth.pin(|p| p.y > 0.99, f32::INFINITY);
    let mass = cloth.mass();
    // A weight on the lower edge, held by weak pins that pass the load through.
    let weight = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, -1.0, 4.0),
        ball(0.2, 2000.0),
    );
    let id = physics.add_soft_body(cloth);
    let pinned = physics
        .pin_soft_body(&world, id, |p| p.y < -0.99, weight, f32::INFINITY)
        .unwrap();
    assert_eq!(pinned, 21);
    run(&mut world, &mut physics, 300);
    let sheet = physics.soft_body(id).unwrap();
    assert!(sheet.pieces() >= 2, "{} pieces", sheet.pieces());
    assert!(
        (sheet.mass() - mass).abs() < 0.01 * mass,
        "mass {}",
        sheet.mass()
    );
    assert!(
        z(&world, weight) < 1.0,
        "the weight fell: {}",
        z(&world, weight)
    );
}

#[test]
fn a_cloth_torn_from_a_weak_pin_falls() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let mut cloth = sheet(SoftMaterial::cloth());
    cloth.pin(|p| p.y > 0.99, 0.01);
    let id = physics.add_soft_body(cloth);
    run(&mut world, &mut physics, 120);
    let sheet = physics.soft_body(id).unwrap();
    assert_eq!(sheet.pin_count(), 0);
    let top = sheet.positions().iter().fold(f32::MIN, |t, p| t.max(p.z));
    assert!(top < 0.0, "fell to {top}");
}

/// A 2 × 2 × 0.1 m steel plate on two supports, with a ball dropped on it.
fn struck_plate(drop_height: f32, density: f32) -> (SoftBody, f32, f32) {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let shape = Collider::Cuboid {
        half_extents: Vec3::new(0.2, 2.0, 0.5),
    };
    for x in [-0.9, 0.9] {
        let support = PhysicsBody::new(RigidBody::Static, shape);
        spawn(&mut world, &mut physics, Vec3::new(x, 0.0, 0.5), support);
    }
    let plate = SoftBody::lattice(
        Vec3::new(0.0, 0.0, 1.06),
        Quat::IDENTITY,
        Vec3::new(1.0, 1.0, 0.05),
        [10, 10, 1],
        SoftMaterial::sheet_metal(),
    );
    let id = physics.add_soft_body(plate);
    run(&mut world, &mut physics, 60);
    let settled = center_height(physics.soft_body(id).unwrap());
    let start = Vec3::new(0.0, 0.0, 1.5 + drop_height);
    let struck = spawn(&mut world, &mut physics, start, ball(0.3, density));
    run(&mut world, &mut physics, 240);
    let height = z(&world, struck);
    (physics.soft_body(id).unwrap().clone(), settled, height)
}

fn center_height(plate: &SoftBody) -> f32 {
    let center = (plate.positions().iter())
        .filter(|p| p.x.abs() < 0.15 && p.y.abs() < 0.15)
        .map(|p| p.z);
    let (sum, count) = center.fold((0.0, 0), |(s, n), z| (s + z, n + 1));
    sum / count as f32
}

#[test]
fn a_metal_plate_holds_its_shape_under_its_own_weight() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    ground(&mut world, &mut physics);
    let center = Vec3::new(0.0, 0.0, 0.6);
    let half = Vec3::new(0.5, 0.5, 0.5);
    let block = SoftBody::lattice(
        center,
        Quat::IDENTITY,
        half,
        [4, 4, 4],
        SoftMaterial::sheet_metal(),
    );
    let id = physics.add_soft_body(block);
    run(&mut world, &mut physics, 120);
    let settled = physics.soft_body(id).unwrap().permanent_strain();
    run(&mut world, &mut physics, 240);
    let body = physics.soft_body(id).unwrap();
    let top = body.positions().iter().fold(f32::MIN, |t, p| t.max(p.z));
    assert!((top - 1.01).abs() < 0.02, "the block's top is at {top}");
    assert_eq!(body.permanent_strain(), settled, "it does not creep");
}

#[test]
fn metal_springs_back_from_a_light_blow_and_stays_dented_by_a_hard_one() {
    let (light, before, resting) = struck_plate(0.2, 100.0);
    assert!(
        (resting - 1.42).abs() < 0.05,
        "the ball rests on the plate: {resting}"
    );
    let kept = before - center_height(&light);
    assert!(kept.abs() < 0.005, "a light blow left {kept} m");

    let (hard, _, _) = struck_plate(6.0, 8000.0);
    assert!(
        hard.permanent_strain() > 0.01,
        "strain {}",
        hard.permanent_strain()
    );
    let dent = before - center_height(&hard);
    assert!(dent > 0.01, "dented {dent} m");
    assert!(hard.positions().iter().all(|p| p.is_finite()));
}

#[test]
fn a_dent_stays_when_what_made_it_is_gone() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    ground(&mut world, &mut physics);
    let block = SoftBody::lattice(
        Vec3::new(0.0, 0.0, 0.26),
        Quat::IDENTITY,
        Vec3::new(1.0, 1.0, 0.25),
        [8, 8, 2],
        SoftMaterial::crumple_zone(),
    );
    let id = physics.add_soft_body(block);
    run(&mut world, &mut physics, 30);
    let flat = center_height(physics.soft_body(id).unwrap());
    let press = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 4.0),
        ball(0.3, 20_000.0),
    );
    run(&mut world, &mut physics, 90);
    physics.remove(press);
    run(&mut world, &mut physics, 120);
    let body = physics.soft_body(id).unwrap();
    let dent = flat - center_height(body);
    assert!(dent > 0.005, "kept {dent} m of dent");
}

#[test]
fn a_car_shell_crumples_around_its_rigid_core_and_slows_it() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let wall_shape = Collider::Cuboid {
        half_extents: Vec3::new(0.5, 5.0, 3.0),
    };
    let wall = PhysicsBody::new(RigidBody::Static, wall_shape);
    spawn(&mut world, &mut physics, Vec3::new(6.0, 0.0, 3.0), wall);
    ground(&mut world, &mut physics);
    let shape = Collider::Cuboid {
        half_extents: Vec3::new(0.8, 0.5, 0.3),
    };
    let mut core = PhysicsBody::new(RigidBody::Dynamic, shape);
    core.velocity.linear = Vec3::X * 12.0;
    core.material.density = 300.0;
    core.material.friction = 0.0;
    core.material.static_friction = 0.0;
    let start = Vec3::new(0.0, 0.0, 0.6);
    let car = spawn(&mut world, &mut physics, start, core);
    let material = SoftMaterial::crumple_zone();
    let mut shell = SoftBody::lattice(
        start,
        Quat::IDENTITY,
        Vec3::new(1.6, 0.7, 0.45),
        [8, 3, 2],
        material,
    );
    shell.set_velocity(Vec3::X * 12.0);
    let id = physics.add_soft_body(shell);
    let inside = |p: Vec3| p.x.abs() < 0.85 && p.y.abs() < 0.55 && (p.z - 0.6).abs() < 0.35;
    let pinned = physics
        .pin_soft_body(&world, id, inside, car, f32::INFINITY)
        .unwrap();
    assert!(pinned > 0);
    let nose = |physics: &PhysicsWorld| {
        let body = physics.soft_body(id).unwrap();
        body.positions().iter().fold(f32::MIN, |n, p| n.max(p.x))
    };
    // Nothing happens to it before it arrives.
    run(&mut world, &mut physics, 10);
    let (body, core) = (physics.soft_body(id).unwrap(), physics.get(car).unwrap());
    assert_eq!(body.permanent_strain(), 0.0, "undamaged on the way");
    assert!(
        core.velocity.linear.x > 11.0,
        "still at {}",
        core.velocity.linear.x
    );
    run(&mut world, &mut physics, 230);
    let (body, core) = (physics.soft_body(id).unwrap(), physics.get(car).unwrap());
    assert!(body.positions().iter().all(|p| p.is_finite()));
    assert!(
        body.permanent_strain() > 0.02,
        "strain {}",
        body.permanent_strain()
    );
    assert!(
        core.velocity.linear.x < 3.0,
        "core still at {}",
        core.velocity.linear.x
    );
    let x = world.get(car).unwrap().transform.translation[0];
    // The wall's face is at 5.5, the core's nose 0.8 ahead of its center.
    assert!(x + 0.8 < 5.51, "the core stayed out of the wall: {x}");
    assert!(nose(&physics) < 5.6, "the shell stopped at the wall");
    let crumpled = 1.6 - (nose(&physics) - x);
    assert!(crumpled > 0.1, "front crumpled {crumpled} m");
}
