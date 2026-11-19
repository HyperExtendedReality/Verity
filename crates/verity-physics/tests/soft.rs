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
    assert!(held < 0.3, "held cloth moved {held}");
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
    assert!(height > 3.8 && height < 4.3, "ball rests at {height}");
    let sheet = physics.soft_body(id).unwrap();
    let lowest = sheet.positions().iter().fold(f32::MAX, |l, p| l.min(p.z));
    assert!(lowest < 3.99, "the cloth sagged to {lowest}");
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
    material.strength = Strength::brittle(150.0);
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
        let support = PhysicsBody::new(RigidBody::Static, shape.clone());
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

#[test]
fn a_solid_can_be_any_shape_and_keeps_it() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    ground(&mut world, &mut physics);
    let radius = 0.5;
    let ball = SoftBody::voxels(
        Vec3::new(0.0, 0.0, 0.55),
        Quat::IDENTITY,
        Vec3::splat(radius),
        [8, 8, 8],
        SoftMaterial::sheet_metal(),
        |p| p.length() <= radius,
    );
    let full = 9 * 9 * 9;
    assert!(ball.positions().len() < full, "corners are left out");
    let volume = 4.0 / 3.0 * std::f32::consts::PI * radius.powi(3);
    let mass = 7800.0 * volume;
    assert!(
        (ball.mass() - mass).abs() < 0.1 * mass,
        "mass {}",
        ball.mass()
    );
    let id = physics.add_soft_body(ball);
    run(&mut world, &mut physics, 180);
    let body = physics.soft_body(id).unwrap();
    let top = body.positions().iter().fold(f32::MIN, |t, p| t.max(p.z));
    let bottom = body.positions().iter().fold(f32::MAX, |b, p| b.min(p.z));
    assert!((bottom - 0.01).abs() < 0.03, "it landed at {bottom}");
    assert!(
        (top - bottom - 2.0 * radius).abs() < 0.05,
        "height {}",
        top - bottom
    );
}

#[test]
fn points_tied_to_a_solid_follow_it() {
    let center = Vec3::new(0.0, 0.0, 5.0);
    let half = Vec3::new(1.0, 0.5, 0.5);
    let mut block = SoftBody::lattice(
        center,
        Quat::IDENTITY,
        half,
        [4, 2, 2],
        SoftMaterial::rubber(),
    );
    let points = [center + Vec3::new(0.3, 0.1, -0.2), center];
    let tied = block.embed(&points);
    let at_rest = block.deformed(&tied);
    assert!(at_rest[0].distance(points[0]) < 1.0e-4, "{:?}", at_rest[0]);
    block.pin(|p| p.x < -0.99, f32::INFINITY);
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let id = physics.add_soft_body(block);
    run(&mut world, &mut physics, 60);
    let body = physics.soft_body(id).unwrap();
    let moved = body.deformed(&tied);
    assert!(
        moved[0].z < points[0].z - 0.02,
        "it sagged with the body: {:?}",
        moved[0]
    );
    let nearest = (body.positions().iter())
        .map(|p| p.distance(moved[0]))
        .fold(f32::MAX, f32::min);
    assert!(nearest < 0.5, "it stayed inside: {nearest}");
}

/// A 3 m concrete beam held at both ends with a heavy ball dropped on it.
fn struck_beam(material: SoftMaterial, density: f32) -> SoftBody {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let half = Vec3::new(1.5, 0.2, 0.15);
    let center = Vec3::new(0.0, 0.0, 2.0);
    let mut beam = SoftBody::lattice(center, Quat::IDENTITY, half, [12, 2, 2], material);
    beam.pin(|p| p.x.abs() > 1.3, f32::INFINITY);
    let id = physics.add_soft_body(beam);
    spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 4.0),
        ball(0.3, density),
    );
    run(&mut world, &mut physics, 180);
    physics.soft_body(id).unwrap().clone()
}

#[test]
fn a_brittle_solid_cracks_into_pieces_and_a_light_blow_does_not() {
    let light = struck_beam(SoftMaterial::concrete(), 100.0);
    assert_eq!(light.pieces(), 1, "{} cracks", light.tears());
    let heavy = struck_beam(SoftMaterial::concrete(), 8000.0);
    assert!(heavy.pieces() >= 2, "{} pieces", heavy.pieces());
    assert!(heavy.positions().iter().all(|p| p.is_finite()));
    let lowest = heavy.positions().iter().fold(f32::MAX, |l, p| l.min(p.z));
    assert!(lowest < 1.5, "the middle fell: {lowest}");
}

#[test]
fn metal_tears_only_after_its_ductility_runs_out() {
    let mut weak = SoftMaterial::crumple_zone();
    weak.strength.ductility = 0.2;
    let snaps = struck_beam(weak, 8000.0);
    assert!(snaps.tears() > 0);
    let bends = struck_beam(SoftMaterial::crumple_zone(), 8000.0);
    assert_eq!(bends.tears(), 0, "ductile enough to bend, not break");
    assert!(
        bends.permanent_strain() > 0.05,
        "strain {}",
        bends.permanent_strain()
    );
}

/// The closest two particles of a sheet that are not neighbours at rest.
fn closest_folds(sheet: &SoftBody, built: &[Vec3], spacing: f32) -> f32 {
    let p = sheet.positions();
    let mut closest = f32::MAX;
    for i in 0..built.len() {
        for j in i + 1..built.len() {
            if built[i].distance(built[j]) > 2.0 * spacing {
                closest = closest.min(p[i].distance(p[j]));
            }
        }
    }
    closest
}

#[test]
fn a_sheet_falling_on_itself_stays_in_layers() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    ground(&mut world, &mut physics);
    // Dropped upright, it folds down onto itself on the floor.
    let upright = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
    let center = Vec3::new(0.0, 0.0, 1.1);
    let cloth = SoftBody::cloth(center, upright, [1.0, 2.0], [10, 20], SoftMaterial::cloth());
    let built = cloth.positions().to_vec();
    let id = physics.add_soft_body(cloth);
    run(&mut world, &mut physics, 240);
    let sheet = physics.soft_body(id).unwrap();
    let closest = closest_folds(sheet, &built, 0.1);
    assert!(closest > 0.05, "layers came within {closest}");
}

#[test]
fn one_soft_body_lands_on_another() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let material = SoftMaterial::cloth();
    let center = Vec3::new(0.0, 0.0, 3.0);
    let mut lower = SoftBody::cloth(center, Quat::IDENTITY, [2.0, 2.0], [16, 16], material);
    lower.pin(|p| p.x.abs() > 0.99 || p.y.abs() > 0.99, f32::INFINITY);
    let lower = physics.add_soft_body(lower);
    let upper = SoftBody::cloth(
        center + Vec3::Z,
        Quat::IDENTITY,
        [1.0, 1.0],
        [8, 8],
        material,
    );
    let upper = physics.add_soft_body(upper);
    run(&mut world, &mut physics, 180);
    let lowest =
        (physics.soft_body(upper).unwrap().positions().iter()).fold(f32::MAX, |l, p| l.min(p.z));
    let middle =
        (physics.soft_body(lower).unwrap().positions().iter()).fold(f32::MAX, |l, p| l.min(p.z));
    assert!(lowest > middle, "it stayed on top: {lowest} over {middle}");
    assert!(lowest > 2.5, "and did not fall through: {lowest}");
}

#[test]
fn fast_cloth_does_not_pass_through_a_thin_wall() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let shape = Collider::Cuboid {
        half_extents: Vec3::new(0.02, 3.0, 3.0),
    };
    let wall = PhysicsBody::new(RigidBody::Static, shape);
    spawn(&mut world, &mut physics, Vec3::new(2.0, 0.0, 5.0), wall);
    let upright = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
    let center = Vec3::new(0.0, 0.0, 5.0);
    let mut cloth = SoftBody::cloth(center, upright, [1.0, 1.0], [8, 8], SoftMaterial::cloth());
    cloth.set_velocity(Vec3::X * 60.0);
    let id = physics.add_soft_body(cloth);
    run(&mut world, &mut physics, 30);
    let body = physics.soft_body(id).unwrap();
    let furthest = body.positions().iter().fold(f32::MIN, |f, p| f.max(p.x));
    assert!(furthest < 2.0, "reached {furthest} past the wall at 2");
}

#[test]
fn a_small_fast_ball_hits_cloth_between_its_particles() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let mut cloth = sheet(SoftMaterial::cloth());
    cloth.pin(|p| p.x.abs() > 0.99 || p.y.abs() > 0.99, f32::INFINITY);
    let id = physics.add_soft_body(cloth);
    // Between grid points, smaller than the 0.1 m cells, at 30 m/s.
    let mut small = ball(0.03, 2000.0);
    small.velocity.linear = Vec3::Z * -30.0;
    let body = spawn(&mut world, &mut physics, Vec3::new(0.05, 0.05, 5.0), small);
    run(&mut world, &mut physics, 12);
    let sheet = physics.soft_body(id).unwrap();
    let dipped = sheet.positions().iter().fold(f32::MAX, |l, p| l.min(p.z));
    assert!(dipped < 3.97, "the cloth felt it: {dipped}");
    // Free fall from 30 m/s would be 32 by now.
    let speed = physics.get(body).unwrap().velocity.linear.length();
    assert!(speed < 31.5, "it lost speed: {speed}");
}

#[test]
fn a_car_at_56_kmh_crushes_about_as_far_as_a_real_one() {
    // Published frontal tests at this speed crush 0.49 to 0.65 m.
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let shape = Collider::Cuboid {
        half_extents: Vec3::new(0.5, 5.0, 3.0),
    };
    let wall = PhysicsBody::new(RigidBody::Static, shape);
    spawn(&mut world, &mut physics, Vec3::new(6.0, 0.0, 3.0), wall);
    let start = Vec3::new(0.0, 0.0, 3.0);
    let half = Vec3::new(1.0, 0.7, 0.5);
    let mut core = PhysicsBody::new(RigidBody::Dynamic, Collider::Cuboid { half_extents: half });
    core.material.density = 1200.0 / (8.0 * half.x * half.y * half.z);
    core.velocity.linear = Vec3::X * 15.6;
    let car = spawn(&mut world, &mut physics, start, core);
    let material = SoftMaterial::crumple_zone();
    let body = Vec3::new(2.2, 0.9, 0.7);
    let mut shell = SoftBody::lattice(start, Quat::IDENTITY, body, [11, 4, 3], material);
    shell.set_velocity(Vec3::X * 15.6);
    let id = physics.add_soft_body(shell);
    let inside = |p: Vec3| (p - start).abs().cmple(half + 0.05).all();
    physics
        .pin_soft_body(&world, id, inside, car, f32::INFINITY)
        .unwrap();
    let settings = PhysicsSettings {
        gravity: Vec3::ZERO,
        ..Default::default()
    };
    let mut crush = 0.0f32;
    for _ in 0..120 {
        physics.step(&mut world, &settings, DT);
        let x = world.get(car).unwrap().transform.translation[0];
        let sheet = physics.soft_body(id).unwrap();
        let nose = sheet.positions().iter().fold(f32::MIN, |n, p| n.max(p.x));
        crush = crush.max(body.x - (nose - x));
    }
    assert!((0.45..0.7).contains(&crush), "crushed {crush} m");
}

/// A lattice shell pinned to a rigid core, moving along X, with no gravity.
fn car(
    world: &mut World,
    physics: &mut PhysicsWorld,
    start: Vec3,
    speed: f32,
) -> (EntityHandle, SoftId) {
    turning_car(world, physics, start, speed, 0.0)
}

/// A car with a steady torque about Z, as steering gives it.
fn turning_car(
    world: &mut World,
    physics: &mut PhysicsWorld,
    start: Vec3,
    speed: f32,
    torque: f32,
) -> (EntityHandle, SoftId) {
    let half = Vec3::new(1.0, 0.7, 0.5);
    let mut core = PhysicsBody::new(RigidBody::Dynamic, Collider::Cuboid { half_extents: half });
    core.material.density = 1200.0 / (8.0 * half.x * half.y * half.z);
    core.velocity.linear = Vec3::X * speed;
    core.force.torque = Vec3::Z * torque;
    let entity = spawn(world, physics, start, core);
    let body = Vec3::new(2.2, 0.9, 0.7);
    let material = SoftMaterial::crumple_zone();
    let mut shell = SoftBody::lattice(start, Quat::IDENTITY, body, [11, 4, 3], material);
    shell.set_velocity(Vec3::X * speed);
    let id = physics.add_soft_body(shell);
    let inside = move |p: Vec3| (p - start).abs().cmple(half + 0.05).all();
    physics
        .pin_soft_body(world, id, inside, entity, f32::INFINITY)
        .unwrap();
    (entity, id)
}

fn weightless() -> PhysicsSettings {
    PhysicsSettings {
        gravity: Vec3::ZERO,
        ..Default::default()
    }
}

#[test]
fn a_driving_car_stops_solving_and_keeps_its_shape_on_its_core() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let (entity, id) = car(&mut world, &mut physics, Vec3::new(0.0, 0.0, 3.0), 10.0);
    let settings = weightless();
    for _ in 0..60 {
        physics.step(&mut world, &settings, DT);
    }
    assert!(physics.soft_body(id).unwrap().riding());
    let core = |world: &World| Vec3::from_array(world.get(entity).unwrap().transform.translation);
    let shape = |physics: &PhysicsWorld, at: Vec3| -> Vec<Vec3> {
        let sheet = physics.soft_body(id).unwrap();
        sheet.positions().iter().map(|&p| p - at).collect()
    };
    let before = shape(&physics, core(&world));
    for _ in 0..60 {
        physics.step(&mut world, &settings, DT);
    }
    let after = shape(&physics, core(&world));
    let drift = (before.iter().zip(&after)).fold(0.0f32, |d, (a, b)| d.max(a.distance(*b)));
    assert!(drift < 1.0e-3, "shape moved {drift} m");
    assert!(physics.soft_body(id).unwrap().velocities()[0].x > 9.9);
}

#[test]
fn a_riding_car_still_crushes_against_a_wall() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let shape = Collider::Cuboid {
        half_extents: Vec3::new(0.5, 5.0, 5.0),
    };
    let wall = PhysicsBody::new(RigidBody::Static, shape);
    spawn(&mut world, &mut physics, Vec3::new(22.0, 0.0, 3.0), wall);
    let start = Vec3::new(0.0, 0.0, 3.0);
    let (entity, id) = car(&mut world, &mut physics, start, 15.6);
    let settings = weightless();
    let mut rode = false;
    let mut crush = 0.0f32;
    for _ in 0..180 {
        physics.step(&mut world, &settings, DT);
        let body = physics.soft_body(id).unwrap();
        rode |= body.riding();
        let x = world.get(entity).unwrap().transform.translation[0];
        let nose = body.positions().iter().fold(f32::MIN, |n, p| n.max(p.x));
        crush = crush.max(2.2 - (nose - x));
    }
    assert!(rode, "the car never rode");
    assert!((0.45..0.7).contains(&crush), "crushed {crush} m");
}

#[test]
fn riding_cars_wake_for_each_other() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let (_, a) = car(&mut world, &mut physics, Vec3::new(0.0, 0.0, 3.0), 8.0);
    let (_, b) = car(&mut world, &mut physics, Vec3::new(24.0, 0.0, 3.0), -8.0);
    let settings = weightless();
    let mut rode = false;
    for _ in 0..120 {
        physics.step(&mut world, &settings, DT);
        rode |= physics.soft_body(a).unwrap().riding();
    }
    assert!(rode, "the cars never rode");
    let dented = |id| physics.soft_body(id).unwrap().permanent_strain();
    assert!(dented(a) > 0.0 && dented(b) > 0.0);
}

#[test]
fn the_weight_of_a_riding_shell_stays_on_its_core() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let start = Vec3::new(0.0, 0.0, 50.0);
    let half = Vec3::new(1.0, 0.7, 0.5);
    let material = SoftMaterial::crumple_zone();
    let body = Vec3::new(2.2, 0.9, 0.7);
    let shell = SoftBody::lattice(start, Quat::IDENTITY, body, [11, 4, 3], material);
    let gravity = Vec3::new(0.0, 0.0, -2.0);
    // Exactly what holds core and shell up: if the shell's weight went missing
    // once it rode, the car would speed upward.
    let mut core = PhysicsBody::new(RigidBody::Dynamic, Collider::Cuboid { half_extents: half });
    core.material.density = 1200.0 / (8.0 * half.x * half.y * half.z);
    core.force.force = -gravity * (1200.0 + shell.mass());
    let entity = spawn(&mut world, &mut physics, start, core);
    let id = physics.add_soft_body(shell);
    let inside = move |p: Vec3| (p - start).abs().cmple(half + 0.05).all();
    physics
        .pin_soft_body(&world, id, inside, entity, f32::INFINITY)
        .unwrap();
    let settings = PhysicsSettings {
        gravity,
        ..Default::default()
    };
    for _ in 0..300 {
        physics.step(&mut world, &settings, DT);
        if physics.soft_body(id).unwrap().riding() {
            break;
        }
    }
    assert!(physics.soft_body(id).unwrap().riding());
    let speed = |physics: &PhysicsWorld| physics.get(entity).unwrap().velocity.linear.z;
    let before = speed(&physics);
    for _ in 0..120 {
        physics.step(&mut world, &settings, DT);
    }
    let change = speed(&physics) - before;
    assert!(change.abs() < 1.0e-3, "speed changed by {change} m/s");
}

#[test]
fn soft_bodies_give_the_same_result_on_any_number_of_threads() {
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
            ground(&mut world, &mut physics);
            let ids: Vec<SoftId> = (0..6)
                .map(|i| {
                    let center = Vec3::new(i as f32 * 3.0, 0.0, 2.0 + 0.5 * i as f32);
                    let material = SoftMaterial::rubber();
                    let half = Vec3::splat(0.5);
                    let block =
                        SoftBody::lattice(center, Quat::IDENTITY, half, [3, 3, 3], material);
                    physics.add_soft_body(block)
                })
                .collect();
            run(&mut world, &mut physics, 90);
            (ids.iter())
                .flat_map(|&id| physics.soft_body(id).unwrap().positions().to_vec())
                .collect::<Vec<Vec3>>()
        })
    };
    assert_eq!(run(1), run(4));
}

/// The shell's particles in its core's frame.
fn in_core_frame(
    world: &World,
    physics: &PhysicsWorld,
    car: EntityHandle,
    id: SoftId,
) -> Vec<Vec3> {
    let transform = &world.get(car).unwrap().transform;
    let position = Vec3::from_array(transform.translation);
    let rotation = Quat::from_array(transform.rotation).normalize();
    let shell = physics.soft_body(id).unwrap();
    (shell.positions().iter())
        .map(|&p| rotation.conjugate() * (p - position))
        .collect()
}

/// The shell's particle nearest the middle of a car's nose, as built.
fn nose_of(points: &[Vec3]) -> usize {
    let tip = Vec3::new(2.2, 0.0, 0.0);
    (0..points.len())
        .min_by(|&a, &b| points[a].distance(tip).total_cmp(&points[b].distance(tip)))
        .unwrap()
}

/// A static wall facing a car that starts at the origin, 3 m up.
fn wall(world: &mut World, physics: &mut PhysicsWorld, x: f32) {
    let shape = Collider::Cuboid {
        half_extents: Vec3::new(0.5, 5.0, 5.0),
    };
    let wall = PhysicsBody::new(RigidBody::Static, shape);
    spawn(world, physics, Vec3::new(x, 0.0, 3.0), wall);
}

#[test]
fn a_car_that_turned_while_riding_wakes_in_shape() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let start = Vec3::new(0.0, 0.0, 3.0);
    let (car, id) = turning_car(&mut world, &mut physics, start, 0.0, 150.0);
    let settings = weightless();
    // Regions last saw the car half a turn ago, the worst case for finding
    // their rotation from where they left off.
    let turned = |world: &World| {
        let rotation = Quat::from_array(world.get(car).unwrap().transform.rotation);
        rotation.to_axis_angle().1
    };
    for _ in 0..1200 {
        physics.step(&mut world, &settings, DT);
        if turned(&world) > 3.0 {
            break;
        }
    }
    assert!(turned(&world) > 3.0, "turned only {} rad", turned(&world));
    assert!(physics.soft_body(id).unwrap().riding());
    let shape = in_core_frame(&world, &physics, car, id);
    // Something appears beside a corner, close enough to wake it, not to touch.
    let corner = physics.soft_body(id).unwrap().positions();
    let far = corner.iter().fold(start, |f, &p| {
        if (p - start).length() > (f - start).length() {
            p
        } else {
            f
        }
    });
    let outward = (far - start).normalize();
    let pebble = PhysicsBody::new(RigidBody::Static, Collider::Sphere { radius: 0.01 });
    spawn(&mut world, &mut physics, far + outward * 0.1, pebble);
    physics.step(&mut world, &settings, DT);
    assert!(!physics.soft_body(id).unwrap().riding(), "never woke");
    for _ in 0..30 {
        physics.step(&mut world, &settings, DT);
    }
    let now = in_core_frame(&world, &physics, car, id);
    let drift = (shape.iter().zip(&now)).fold(0.0f32, |d, (a, b)| d.max(a.distance(*b)));
    assert!(drift < 0.05, "shape moved {drift} m");
}

#[test]
fn a_crash_is_one_impact_and_a_nudge_is_none() {
    let crash = |speed: f32| {
        let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
        wall(&mut world, &mut physics, 4.0);
        let (_, id) = car(&mut world, &mut physics, Vec3::new(0.0, 0.0, 3.0), speed);
        let settings = weightless();
        for _ in 0..300 {
            physics.step(&mut world, &settings, DT);
        }
        physics.soft_body_mut(id).unwrap().take_impacts()
    };
    let impacts = crash(15.6);
    assert_eq!(impacts.len(), 1, "{impacts:?}");
    let [hit] = impacts[..] else { unreachable!() };
    let kinetic = 0.5 * 1200.0 * 15.6 * 15.6;
    assert!(
        hit.energy > kinetic && hit.energy < 2.0 * kinetic,
        "{hit:?}"
    );
    assert!(hit.normal.x < -0.9, "{hit:?}");
    assert!(hit.point.x > 1.5, "{hit:?}");
    assert!(crash(0.2).is_empty());
}

#[test]
fn a_crash_nobody_saw_dents_the_car_when_it_comes_into_view() {
    let crash = |culled: bool| {
        let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
        wall(&mut world, &mut physics, 6.0);
        let (car, id) = car(&mut world, &mut physics, Vec3::new(0.0, 0.0, 3.0), 15.6);
        let far = Interest {
            center: Vec3::new(1000.0, 0.0, 0.0),
            radius: 10.0,
            frustum: None,
        };
        if culled {
            physics.set_interest(vec![far]);
        }
        let settings = weightless();
        let nose = nose_of(&in_core_frame(&world, &physics, car, id));
        for _ in 0..120 {
            physics.step(&mut world, &settings, DT);
        }
        let unseen = physics.soft_body(id).unwrap().permanent_strain();
        physics.set_interest(Vec::new());
        for _ in 0..30 {
            physics.step(&mut world, &settings, DT);
        }
        let shape = in_core_frame(&world, &physics, car, id);
        (
            unseen,
            2.2 - shape[nose].x,
            physics.soft_body(id).unwrap().permanent_strain(),
        )
    };
    let (_, live, _) = crash(false);
    let (unseen, replayed, strain) = crash(true);
    assert_eq!(unseen, 0.0, "dented out of sight");
    assert!(strain > 0.0);
    // No less than a solved crash leaves after springing back, and no more than
    // the most a real car crushes at this speed.
    assert!(
        replayed > live && replayed < 0.8,
        "{replayed} m against {live} m"
    );
}

#[test]
fn an_impact_from_elsewhere_dents_a_twin() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let (car, id) = car(&mut world, &mut physics, Vec3::new(0.0, 0.0, 3.0), 0.0);
    let settings = weightless();
    physics.step(&mut world, &settings, DT);
    let shape = in_core_frame(&world, &physics, car, id);
    let nose = nose_of(&shape);
    let before = shape[nose].x;
    physics.soft_body_mut(id).unwrap().apply_impact(Impact {
        point: Vec3::new(2.2, 0.0, 0.0),
        normal: Vec3::NEG_X,
        energy: 50_000.0,
        radius: 1.0,
    });
    for _ in 0..60 {
        physics.step(&mut world, &settings, DT);
    }
    let after = in_core_frame(&world, &physics, car, id)[nose].x;
    assert!(before - after > 0.1, "only {} m", before - after);
}

#[test]
fn unseen_cloth_holds_still_and_moves_again_in_view() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let falling = spawn(
        &mut world,
        &mut physics,
        Vec3::new(9.0, 0.0, 50.0),
        ball(0.2, 100.0),
    );
    let id = physics.add_soft_body(sheet(SoftMaterial::cloth()));
    let at = |physics: &PhysicsWorld| physics.soft_body(id).unwrap().positions()[0];
    let away = Interest {
        center: Vec3::new(1000.0, 0.0, 0.0),
        radius: 10.0,
        frustum: None,
    };
    physics.set_interest(vec![away]);
    run(&mut world, &mut physics, 2);
    let held = at(&physics);
    let height = z(&world, falling);
    run(&mut world, &mut physics, 60);
    assert_eq!(at(&physics), held, "unseen cloth moved");
    assert!(
        z(&world, falling) < height - 1.0,
        "rigid bodies must still step"
    );
    // A view that reaches it counts, far from any player.
    let plane = (Vec3::Z, 1000.0);
    physics.set_interest(vec![Interest {
        frustum: Some([plane; 6]),
        ..away
    }]);
    run(&mut world, &mut physics, 10);
    assert!(at(&physics).z < held.z, "cloth stayed frozen in view");
}

#[test]
fn cars_far_from_the_origin_drive_as_straight_as_those_near_it() {
    let settings = weightless();
    for offset in [0.0, 400.0, 2000.0, -20_000.0] {
        let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
        let start = Vec3::new(offset * 0.5, offset, 3.0);
        let (car, id) = car(&mut world, &mut physics, start, 15.6);
        for _ in 0..120 {
            physics.step(&mut world, &settings, DT);
        }
        let v = physics.get(car).unwrap().velocity.linear;
        assert!(physics.soft_body(id).unwrap().riding(), "{offset} m out");
        assert!(
            v.y.abs() < 0.1 && v.z.abs() < 0.1,
            "{offset} m out it moves sideways at {v}"
        );
    }
}

#[test]
fn a_car_far_from_the_origin_crushes_like_one_near_it() {
    let settings = weightless();
    let crush = |offset: Vec3| {
        let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
        let shape = Collider::Cuboid {
            half_extents: Vec3::new(0.5, 5.0, 5.0),
        };
        let wall = PhysicsBody::new(RigidBody::Static, shape);
        spawn(
            &mut world,
            &mut physics,
            offset + Vec3::new(6.0, 0.0, 3.0),
            wall,
        );
        let start = offset + Vec3::new(0.0, 0.0, 3.0);
        let (car, id) = car(&mut world, &mut physics, start, 15.6);
        let mut crush = 0.0f32;
        for _ in 0..120 {
            physics.step(&mut world, &settings, DT);
            let x = world.get(car).unwrap().transform.translation[0];
            let body = physics.soft_body(id).unwrap();
            let nose = body.positions().iter().fold(f32::MIN, |n, p| n.max(p.x));
            crush = crush.max(2.2 - (nose - x));
        }
        crush
    };
    let (near, far) = (crush(Vec3::ZERO), crush(Vec3::new(3000.0, -4000.0, 0.0)));
    assert!((near - far).abs() < 0.05, "{near} m near, {far} m far");
}

#[test]
fn cloth_hangs_the_same_far_from_the_origin() {
    let hang = |offset: Vec3| {
        let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
        let center = offset + Vec3::new(0.0, 0.0, 4.0);
        let upright = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
        let mut cloth =
            SoftBody::cloth(center, upright, [2.0, 2.0], [12, 12], SoftMaterial::cloth());
        cloth.pin(|p| p.z > center.z + 0.99, f32::INFINITY);
        let id = physics.add_soft_body(cloth);
        run(&mut world, &mut physics, 120);
        let body = physics.soft_body(id).unwrap();
        (body.positions().iter()).fold(f32::MAX, |low, p| low.min(p.z))
    };
    let (near, far) = (hang(Vec3::ZERO), hang(Vec3::new(3000.0, 0.0, 0.0)));
    assert!((near - far).abs() < 0.01, "{near} m near, {far} m far");
}

#[test]
fn embedded_points_past_the_lattice_extrapolate_and_revision_tracks_the_shape() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let start = Vec3::new(0.0, 0.0, 3.0);
    let (_, id) = car(&mut world, &mut physics, start, 5.0);
    let mirror = start + Vec3::new(0.0, 1.3, 0.0);
    let tied = physics.soft_body(id).unwrap().embed(&[mirror]);
    let at_rest = physics.soft_body(id).unwrap().deformed(&tied)[0];
    assert!(at_rest.distance(mirror) < 1.0e-4, "flattened to {at_rest}");
    let settings = weightless();
    for _ in 0..60 {
        physics.step(&mut world, &settings, DT);
    }
    assert!(physics.soft_body(id).unwrap().riding());
    let before = physics.soft_body(id).unwrap().revision();
    for _ in 0..30 {
        physics.step(&mut world, &settings, DT);
    }
    assert_eq!(physics.soft_body(id).unwrap().revision(), before);
}

/// A sheet with a nick at its middle, pulled apart slowly by a bar.
fn pull_apart(cells: u32) -> SoftBody {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let center = Vec3::new(0.0, 0.0, 4.0);
    let material = SoftMaterial {
        weave: 0.0,
        ..SoftMaterial::cloth()
    };
    let mut cloth = SoftBody::cloth(center, Quat::IDENTITY, [2.0, 2.0], [cells, cells], material);
    cloth.pin(|p| p.y > 0.99, f32::INFINITY);
    cloth.nick(center);
    let shape = Collider::Cuboid {
        half_extents: Vec3::new(1.2, 0.02, 0.02),
    };
    let mut bar = PhysicsBody::new(RigidBody::Kinematic, shape);
    bar.velocity.linear = Vec3::NEG_Y * 0.2;
    let bar = spawn(&mut world, &mut physics, Vec3::new(0.0, -1.0, 4.0), bar);
    let id = physics.add_soft_body(cloth);
    physics
        .pin_soft_body(&world, id, |p| p.y < -0.99, bar, f32::INFINITY)
        .unwrap();
    let settings = weightless();
    // Well past parting, so that every bond along the tear has given way.
    for _ in 0..900 {
        physics.step(&mut world, &settings, DT);
    }
    physics.soft_body(id).unwrap().clone()
}

fn tear_length(sheet: &SoftBody) -> f32 {
    (sheet.tear_path().iter())
        .map(|(a, b)| a.distance(*b))
        .sum()
}

#[test]
fn a_tear_stays_a_line_instead_of_shredding_the_sheet() {
    let sheet = pull_apart(24);
    assert!(sheet.pieces() > 1, "never parted");
    // It takes 2 m to cross. Every element overloading at once once cut most of
    // 1,700 edges, some 90 m.
    let length = tear_length(&sheet);
    assert!(length < 16.0, "{length} m of tear");
}

#[test]
fn a_tear_costs_about_the_same_per_metre_at_any_resolution() {
    let per_metre: Vec<f32> = [16, 32]
        .map(|cells| {
            let sheet = pull_apart(cells);
            sheet.tear_work() / tear_length(&sheet)
        })
        .to_vec();
    let (low, high) = (
        per_metre[0].min(per_metre[1]),
        per_metre[0].max(per_metre[1]),
    );
    assert!(low > 0.0, "no work: {per_metre:?}");
    assert!(high < 2.0 * low, "{per_metre:?} J/m");
}

#[test]
fn a_nick_lets_a_sheet_part_sooner() {
    let time_to_part = |nick: bool| {
        let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
        let center = Vec3::new(0.0, 0.0, 4.0);
        let mut cloth = SoftBody::cloth(
            center,
            Quat::IDENTITY,
            [2.0, 2.0],
            [24, 24],
            SoftMaterial::cloth(),
        );
        cloth.pin(|p| p.y > 0.99, f32::INFINITY);
        if nick {
            cloth.nick(center);
        }
        let shape = Collider::Cuboid {
            half_extents: Vec3::new(1.2, 0.02, 0.02),
        };
        let mut bar = PhysicsBody::new(RigidBody::Kinematic, shape);
        bar.velocity.linear = Vec3::NEG_Y * 0.1;
        let bar = spawn(&mut world, &mut physics, Vec3::new(0.0, -1.0, 4.0), bar);
        let id = physics.add_soft_body(cloth);
        physics
            .pin_soft_body(&world, id, |p| p.y < -0.99, bar, f32::INFINITY)
            .unwrap();
        let settings = weightless();
        (0..1500)
            .find(|_| {
                physics.step(&mut world, &settings, DT);
                physics.soft_body(id).unwrap().pieces() > 1
            })
            .unwrap_or(1500)
    };
    let (whole, nicked) = (time_to_part(false), time_to_part(true));
    assert!(nicked <= whole, "{nicked} steps against {whole}");
}

#[test]
fn cloth_pinned_to_a_moving_platform_goes_with_it() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let center = Vec3::new(0.0, 0.0, 4.0);
    let mut platform = PhysicsBody::new(
        RigidBody::Kinematic,
        Collider::Cuboid {
            half_extents: Vec3::new(1.2, 0.02, 0.02),
        },
    );
    platform.velocity.linear = Vec3::X * 2.0;
    let at = Vec3::new(0.0, 1.0, 4.0);
    let platform = spawn(&mut world, &mut physics, at, platform);
    let cloth = SoftBody::cloth(
        center,
        Quat::IDENTITY,
        [2.0, 2.0],
        [8, 8],
        SoftMaterial::cloth(),
    );
    let id = physics.add_soft_body(cloth);
    physics
        .pin_soft_body(&world, id, |p| p.y > 0.99, platform, f32::INFINITY)
        .unwrap();
    let settings = weightless();
    for _ in 0..60 {
        physics.step(&mut world, &settings, DT);
    }
    let x = world.get(platform).unwrap().transform.translation[0];
    let sheet = physics.soft_body(id).unwrap();
    let lag = (sheet.positions().iter())
        .filter(|p| p.y > 0.9)
        .fold(0.0f32, |d, p| d.max((p.x - x).abs() - 1.0));
    assert!(lag < 0.02, "the pinned edge lags by {lag} m");
}

/// A block of tetrahedra hanging from its top under gravity: how far its bottom
/// falls, once settled. A bar's stretch under its own weight is rho g L^2 / 2E.
#[test]
fn a_hanging_tetrahedral_block_stretches_as_much_as_it_should() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let material = SoftMaterial {
        density: 1000.0,
        stiffness: 1.0e6,
        poisson: 0.0,
        damping: 8.0,
        ..SoftMaterial::rubber()
    };
    let center = Vec3::new(0.0, 0.0, 4.0);
    let half = Vec3::new(0.15, 0.15, 0.5);
    let mut block =
        SoftBody::lattice(center, Quat::IDENTITY, half, [3, 3, 10], material).finite_elements();
    block.pin(|p| p.z > 4.49, f32::INFINITY);
    let id = physics.add_soft_body(block);
    run(&mut world, &mut physics, 600);
    let body = physics.soft_body(id).unwrap();
    // Between the middles of its ends: its pins are springs, which each sag by
    // what they carry, so extreme points would count that too.
    let row = |select: fn(f32) -> bool| {
        let rows: Vec<f32> = (body.positions().iter())
            .map(|p| p.z)
            .filter(|&z| select(z))
            .collect();
        rows.iter().sum::<f32>() / rows.len() as f32
    };
    let (high, low) = (row(|z| z > 4.45), row(|z| z < 3.55));
    let (stretch, expected) = (high - low - 1.0, 1000.0 * 9.81 / (2.0 * 1.0e6));
    assert!(
        (stretch / expected - 1.0).abs() < 0.1,
        "{stretch} m, expected {expected} m"
    );
}

/// A cantilever under its own weight droops rho g A L^4 / 8 E I.
#[test]
fn a_tetrahedral_cantilever_droops_about_as_far_as_beam_theory_says() {
    let droop = |cells: [u32; 3]| {
        let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
        let material = SoftMaterial {
            density: 1000.0,
            stiffness: 1.0e7,
            poisson: 0.0,
            damping: 8.0,
            ..SoftMaterial::rubber()
        };
        let center = Vec3::new(0.0, 0.0, 4.0);
        let half = Vec3::new(0.5, 0.05, 0.05);
        let mut beam =
            SoftBody::lattice(center, Quat::IDENTITY, half, cells, material).finite_elements();
        beam.pin(|p| p.x < -0.499, f32::INFINITY);
        let id = physics.add_soft_body(beam);
        run(&mut world, &mut physics, 900);
        let body = physics.soft_body(id).unwrap();
        let tip = (body.positions().iter())
            .filter(|p| p.x > 0.0)
            .fold(f32::MAX, |low, p| low.min(p.z));
        4.0 - 0.05 - tip
    };
    let expected = 1000.0 * 9.81 * 12.0 / (8.0 * 1.0e7 * 0.01);
    let (coarse, fine) = (droop([20, 2, 2]), droop([40, 4, 4]));
    assert!(fine > coarse, "refining must loosen locking");
    assert!(
        fine > 0.6 * expected && fine < 1.2 * expected,
        "{fine} m, expected {expected} m"
    );
}

/// Pull a block apart along Y by a bar until it has torn a while: what lies
/// along the pull, as a share of its torn faces' edges.
fn torn_along_the_pull(material: SoftMaterial) -> f32 {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let center = Vec3::new(0.0, 0.0, 4.0);
    let half = Vec3::new(0.2, 0.3, 0.1);
    let mut block =
        SoftBody::lattice(center, Quat::IDENTITY, half, [6, 9, 3], material).finite_elements();
    block.set_warp(Vec3::Y);
    block.pin(|p| p.y > 0.299, f32::INFINITY);
    let shape = Collider::Cuboid {
        half_extents: Vec3::new(0.3, 0.02, 0.2),
    };
    let mut bar = PhysicsBody::new(RigidBody::Kinematic, shape);
    // Slowly: a fast pull overloads the rows by the bar before load spreads.
    bar.velocity.linear = Vec3::NEG_Y * 0.05;
    let bar = spawn(&mut world, &mut physics, Vec3::new(0.0, -0.3, 4.0), bar);
    let id = physics.add_soft_body(block);
    physics
        .pin_soft_body(&world, id, |p| p.y < -0.299, bar, f32::INFINITY)
        .unwrap();
    let settings = weightless();
    for _ in 0..600 {
        physics.step(&mut world, &settings, DT);
    }
    let path = physics.soft_body(id).unwrap().tear_path();
    assert!(!path.is_empty(), "it never tore");
    let (along, total) = path.iter().fold((0.0, 0.0), |(along, total), (a, b)| {
        (along + (b - a).y.abs(), total + (b - a).length())
    });
    along / total
}

#[test]
fn flesh_splits_along_its_fibres_where_a_plain_solid_tears_across() {
    let plain = SoftMaterial {
        grain: 1.0,
        ..SoftMaterial::flesh()
    };
    let (flesh, plain) = (
        torn_along_the_pull(SoftMaterial::flesh()),
        torn_along_the_pull(plain),
    );
    assert!(
        flesh > 1.3 * plain,
        "{flesh} along the pull against {plain}"
    );
}

#[test]
fn a_hard_blow_never_breaks_a_tetrahedral_body() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    ground(&mut world, &mut physics);
    let center = Vec3::new(0.0, 0.0, 0.31);
    let half = Vec3::new(0.3, 0.3, 0.3);
    let block = SoftBody::lattice(
        center,
        Quat::IDENTITY,
        half,
        [6, 6, 6],
        SoftMaterial::flesh(),
    )
    .finite_elements();
    let id = physics.add_soft_body(block);
    let mut hammer = ball(0.15, 8000.0);
    hammer.velocity.linear = Vec3::NEG_Z * 20.0;
    spawn(&mut world, &mut physics, Vec3::new(0.0, 0.0, 1.0), hammer);
    run(&mut world, &mut physics, 240);
    let body = physics.soft_body(id).unwrap();
    assert!(body.positions().iter().all(|p| p.is_finite()));
    let reach = (body.positions().iter()).fold(0.0f32, |r, p| r.max(p.length()));
    // Nothing leaves faster than the blow, for the 4 s it had.
    assert!(reach < 80.0, "a particle flew {reach} m");
    assert!(body.tears() > 0, "a 113 kg hammer at 20 m/s cut nothing");
}
