//! Rigid and soft bodies on terrain, through the real solver.
use glam::Vec3;
use std::sync::Arc;
use verity_core::{Entity, EntityHandle, Material, Transform, World};
use verity_physics::{
    Bounds, Collider, Ground, PhysicsBody, PhysicsSettings, PhysicsWorld, RigidBody, SoftBody,
    SoftMaterial,
};
use verity_terra::*;

const STEP: f32 = 1.0 / 60.0;

/// Ground with these heights, and no surface noise worth the name.
fn flat(height: f32) -> Terrain {
    Terrain::new(Landscape::flat(height, 385, 1.0))
}

/// A plane tilted by `slope` along x, exact, for what a slope should do.
struct Ramp(f32);

impl Ground for Ramp {
    fn distance(&self, p: Vec3) -> f32 {
        self.normal(p).dot(p)
    }
    fn normal(&self, _: Vec3) -> Vec3 {
        Vec3::new(-self.0, 0.0, 1.0).normalize()
    }
}

fn spawn(world: &mut World, position: Vec3) -> EntityHandle {
    world.spawn(Entity {
        name: "body".into(),
        transform: Transform {
            translation: position.to_array(),
            ..Transform::default()
        },
        material: Material::default(),
        parent: None,
        components: Default::default(),
    })
}

/// A world with a static ground body under `ground`.
fn scene(ground: Arc<dyn Ground>) -> (World, PhysicsWorld) {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let entity = spawn(&mut world, Vec3::ZERO);
    let body = PhysicsBody::new(RigidBody::Static, Collider::Ground);
    physics.insert(&world, entity, body).expect("a live entity");
    physics.set_ground(ground, Bounds::EMPTY);
    (world, physics)
}

fn drop(
    world: &mut World,
    physics: &mut PhysicsWorld,
    collider: Collider,
    at: Vec3,
) -> EntityHandle {
    let entity = spawn(world, at);
    let body = PhysicsBody::new(RigidBody::Dynamic, collider);
    physics.insert(world, entity, body).expect("a live entity");
    entity
}

fn position(world: &World, entity: EntityHandle) -> Vec3 {
    Vec3::from_array(world.get(entity).expect("alive").transform.translation)
}

fn run(world: &mut World, physics: &mut PhysicsWorld, seconds: f32) {
    let settings = PhysicsSettings::default();
    for _ in 0..(seconds / STEP) as usize {
        physics.step(world, &settings, STEP);
    }
}

#[test]
fn bodies_come_to_rest_on_the_surface() {
    let terrain = flat(0.3);
    let surface = |x: f32, y: f32| {
        (terrain.raycast(Vec3::new(x, y, 10.0), -Vec3::Z, 20.0))
            .expect("ground")
            .z
    };
    let (mut world, mut physics) = scene(Arc::new(terrain.clone()));
    let sphere = Collider::Sphere { radius: 0.5 };
    let cuboid = Collider::Cuboid {
        half_extents: Vec3::new(1.0, 0.75, 0.5),
    };
    let a = drop(&mut world, &mut physics, sphere, Vec3::new(0.0, 0.0, 4.0));
    let b = drop(&mut world, &mut physics, cuboid, Vec3::new(6.0, 0.0, 4.0));
    run(&mut world, &mut physics, 4.0);
    let (sphere_at, cuboid_at) = (position(&world, a), position(&world, b));
    // A sphere on a gentle bump touches within a few centimetres of the surface
    // straight below it.
    let sphere_gap = sphere_at.z - 0.5 - surface(sphere_at.x, sphere_at.y);
    assert!(
        sphere_gap.abs() < 0.06,
        "sphere gap {sphere_gap} at {sphere_at}"
    );
    // A cuboid rests on its highest points, so none of it may sink in much, and
    // it must not hover over all of them.
    let lowest = (-2..=2)
        .flat_map(|i| (-1..=1).map(move |j| (i as f32 * 0.5, j as f32 * 0.75)))
        .map(|(dx, dy)| cuboid_at.z - 0.5 - surface(cuboid_at.x + dx, cuboid_at.y + dy))
        .fold(f32::INFINITY, f32::min);
    assert!(lowest.abs() < 0.08, "cuboid gap {lowest} at {cuboid_at}");
    run(&mut world, &mut physics, 1.0);
    assert!(
        position(&world, b).distance(cuboid_at) < 0.01,
        "cuboid creeps"
    );
    assert!(physics.is_sleeping(b), "a body at rest must sleep");
}

#[test]
fn fast_bodies_do_not_tunnel() {
    let (mut world, mut physics) = scene(Arc::new(flat(0.3)));
    let sphere = Collider::Sphere { radius: 0.25 };
    let entity = drop(&mut world, &mut physics, sphere, Vec3::new(0.0, 0.0, 30.0));
    physics.get_mut(entity).expect("just added").velocity.linear = Vec3::new(0.0, 0.0, -200.0);
    run(&mut world, &mut physics, 2.0);
    let at = position(&world, entity);
    assert!(at.z > -1.0, "went through the ground to {at}");
}

#[test]
fn friction_holds_a_box_on_a_gentle_slope_and_not_a_steep_one() {
    let cuboid = Collider::Cuboid {
        half_extents: Vec3::splat(0.5),
    };
    let moved = |slope: f32| {
        let (mut world, mut physics) = scene(Arc::new(Ramp(slope)));
        let normal = Ramp(slope).normal(Vec3::ZERO);
        let tilt = glam::Quat::from_rotation_arc(Vec3::Z, normal);
        let start = normal * 0.51;
        let entity = drop(&mut world, &mut physics, cuboid, start);
        let found = world.get_mut(entity).expect("alive");
        found.transform.rotation = tilt.to_array();
        run(&mut world, &mut physics, 2.0);
        position(&world, entity).distance(start)
    };
    let (gentle, steep) = (moved(0.1), moved(3.0));
    assert!(gentle < 0.1, "slid {gentle} m on a gentle slope");
    assert!(steep > 3.0, "slid only {steep} m on a steep slope");
}

#[test]
fn digging_under_a_sleeping_body_wakes_it_and_it_falls() {
    let mut terrain = flat(0.3);
    terrain.ensure_chunks(&terrain.initial_coords());
    let (mut world, mut physics) = scene(Arc::new(terrain.clone()));
    let cuboid = Collider::Cuboid {
        half_extents: Vec3::splat(0.5),
    };
    let dug = drop(&mut world, &mut physics, cuboid, Vec3::new(0.0, 0.0, 2.0));
    let far = drop(&mut world, &mut physics, cuboid, Vec3::new(40.0, 0.0, 2.0));
    run(&mut world, &mut physics, 8.0);
    assert!(physics.is_sleeping(dug) && physics.is_sleeping(far));
    let rest = position(&world, dug);

    let center = Vec3::new(0.0, 0.0, -2.0);
    assert!(!terrain.brush(center, 5.0, false).is_empty());
    let reach = Vec3::splat(6.0);
    let changed = Bounds {
        min: center - reach,
        max: center + reach,
    };
    physics.set_ground(Arc::new(terrain.clone()), changed);
    run(&mut world, &mut physics, 3.0);
    assert!(
        position(&world, dug).z < rest.z - 2.0,
        "still hanging at {}",
        position(&world, dug)
    );
    assert!(
        physics.is_sleeping(far),
        "an edit far away must not wake bodies"
    );
}

#[test]
fn eroded_landscape_holds_a_pile_of_bodies() {
    let settings = Settings {
        droplets: 500,
        thermal_steps: 3,
        // A 384 m map, as the spots below expect.
        size: 257,
        cell: 1.5,
        mountains: 95.0,
        ..Landform::Mountains.into()
    };
    let terrain = Terrain::new(Landscape::generate(settings));
    let height = |x: f32, y: f32| terrain.landscape.height(x, y);
    let spots: Vec<Vec3> = (0..24)
        .map(|i| {
            let (x, y) = (-60.0 + 5.0 * i as f32, 15.0 - (i % 7) as f32 * 5.0);
            Vec3::new(x, y, height(x, y) + 3.0)
        })
        .collect();
    let (mut world, mut physics) = scene(Arc::new(terrain.clone()));
    let bodies: Vec<_> = spots
        .iter()
        .enumerate()
        .map(|(i, spot)| {
            let collider = if i % 2 == 0 {
                Collider::Sphere { radius: 0.5 }
            } else {
                Collider::Cuboid {
                    half_extents: Vec3::new(0.7, 0.5, 0.4),
                }
            };
            drop(&mut world, &mut physics, collider, *spot)
        })
        .collect();
    run(&mut world, &mut physics, 5.0);
    for body in bodies {
        let at = position(&world, body);
        assert!(at.is_finite(), "exploded");
        assert!(
            terrain.sample(at) > -0.5,
            "buried at {at}, density {}",
            terrain.sample(at)
        );
    }
}

#[test]
fn soft_bodies_land_on_the_ground() {
    let terrain = flat(0.3);
    let (mut world, mut physics) = scene(Arc::new(terrain));
    let cloth = SoftBody::cloth(
        Vec3::new(0.0, 0.0, 3.0),
        glam::Quat::IDENTITY,
        [2.0, 2.0],
        [12, 12],
        SoftMaterial::cloth(),
    );
    let id = physics.add_soft_body(cloth);
    run(&mut world, &mut physics, 4.0);
    let body = physics.soft_body(id).expect("added");
    let (lowest, highest) = body
        .positions()
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), p| {
            (low.min(p.z), high.max(p.z))
        });
    assert!(lowest > -0.5, "fell through to {lowest}");
    assert!(highest < 2.0, "never landed: still at {highest}");
}

/// Contact sets that swap between refreshes keep bodies rocking for ever, which
/// shows only at some spots, so try many.
#[test]
fn bodies_settle_and_sleep_wherever_they_land() {
    for terrain in [
        flat(0.3),
        Terrain::new(Landscape::generate(Settings {
            droplets: 500,
            thermal_steps: 3,
            size: 257,
            cell: 1.5,
            ..Landform::Plains.into()
        })),
    ] {
        let (mut world, mut physics) = scene(Arc::new(terrain.clone()));
        let cuboid = Collider::Cuboid {
            half_extents: Vec3::splat(0.5),
        };
        let bodies: Vec<_> = (0..100)
            .map(|i| {
                let (x, y) = (
                    -90.0 + (i % 10) as f32 * 17.3,
                    -80.0 + (i / 10) as f32 * 16.1,
                );
                let z = terrain.landscape.height(x, y) + 2.0;
                drop(&mut world, &mut physics, cuboid, Vec3::new(x, y, z))
            })
            .collect();
        run(&mut world, &mut physics, 10.0);
        let awake = bodies.iter().filter(|b| !physics.is_sleeping(**b)).count();
        assert!(awake <= 2, "{awake} of 100 bodies never settled");
    }
}

#[test]
fn bodies_land_on_a_planet() {
    let planet = planet::Planet::new(planet::PlanetSettings::default());
    let at = Vec3::new(30.0, 20.0, 0.0);
    let surface = |p: Vec3| p.z - planet.distance(p);
    let ground = surface(at.with_z(200.0)).max(surface(at));
    let (mut world, mut physics) = scene(Arc::new(planet));
    let cube = Collider::Cuboid {
        half_extents: Vec3::splat(0.5),
    };
    let body = drop(&mut world, &mut physics, cube, at.with_z(ground + 6.0));
    run(&mut world, &mut physics, 4.0);
    let rest = position(&world, body);
    assert!(rest.is_finite());
    assert!(
        rest.z > ground - 2.0 && rest.z < ground + 2.5,
        "{rest} over {ground}"
    );
}
