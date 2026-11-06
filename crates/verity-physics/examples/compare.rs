//! verity-physics against Box3D on the same Z-up scenes:
//! `cargo run --release -p verity-physics --example compare [scale]`.
//! Set `RAYON_NUM_THREADS=1` to compare single-threaded (the Box3D bridge uses one worker).
//! Set `VERITY_ITERATIONS` to compare a different AVBD iteration count.
//! Set `BOX3D_SUBSTEPS` to compare a different Box3D substep count.
//! Set `VERITY_SCENE` to `pyramid`, `rain`, `slide`, `free_slide`, or `heavy`.

use glam::{Quat, Vec3};
use std::time::Instant;
use verity_core::{Entity, EntityHandle, Transform, World};

const DT: f32 = 1.0 / 60.0;
const ACTIVE: usize = 60;
const GRAVITY: Vec3 = Vec3::new(0.0, 0.0, -9.81);

/// What both engines are asked to simulate.
enum Kind {
    Floor,
    Cube,
    Ball,
    SlidingCube,
    FreeCube,
    HeavyCube,
}

struct Scene {
    name: &'static str,
    bodies: Vec<(Kind, Vec3)>,
    steps: usize,
}

/// A 2D pyramid of unit cubes on the XZ plane: the classic stacking test.
fn pyramid(rows: usize) -> Scene {
    let mut bodies = vec![(Kind::Floor, Vec3::new(0.0, 0.0, -0.5))];
    for row in 0..rows {
        for i in 0..rows - row {
            let x = i as f32 - (rows - row) as f32 * 0.5;
            bodies.push((Kind::Cube, Vec3::new(x * 1.01, 0.0, 0.5 + row as f32)));
        }
    }
    Scene {
        name: "pyramid",
        bodies,
        steps: 300,
    }
}

/// Deterministic pseudo-random value in [-1, 1).
fn jitter(seed: usize) -> f32 {
    let x = (seed as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .rotate_left(17);
    (x.wrapping_mul(0xBF58_476D_1CE4_E5B9) >> 40) as f32 / (1u64 << 23) as f32 - 1.0
}

/// A block of cubes and balls dropped onto the floor: broadphase and piles.
/// Jitter breaks the grid's symmetry so it collapses into a real pile.
fn rain(side: usize) -> Scene {
    let mut bodies = vec![(Kind::Floor, Vec3::new(0.0, 0.0, -0.5))];
    for z in 0..side {
        for y in 0..side {
            for x in 0..side {
                let n = bodies.len() * 3;
                let p = Vec3::new(x as f32, y as f32, z as f32) * 1.2
                    + Vec3::new(-(side as f32) * 0.6, -(side as f32) * 0.6, 1.0)
                    + Vec3::new(jitter(n), jitter(n + 1), jitter(n + 2)) * 0.08;
                let shape = if (x + y + z) % 2 == 0 {
                    Kind::Cube
                } else {
                    Kind::Ball
                };
                bodies.push((shape, p));
            }
        }
    }
    Scene {
        name: "rain",
        bodies,
        steps: 300,
    }
}

fn slide() -> Scene {
    Scene {
        name: "slide",
        bodies: vec![
            (Kind::Floor, Vec3::new(0.0, 0.0, -0.5)),
            (Kind::SlidingCube, Vec3::new(0.0, 0.0, 0.5)),
        ],
        steps: 300,
    }
}

fn free_slide() -> Scene {
    let mut scene = slide();
    scene.name = "free_slide";
    scene.bodies[1].0 = Kind::FreeCube;
    scene
}

fn heavy() -> Scene {
    let mut bodies = vec![(Kind::Floor, Vec3::new(0.0, 0.0, -0.5))];
    for i in 0..5 {
        bodies.push((Kind::Cube, Vec3::new(0.0, 0.0, 0.5 + i as f32)));
    }
    bodies.push((Kind::HeavyCube, Vec3::new(0.0, 0.0, 5.5)));
    Scene {
        name: "heavy",
        bodies,
        steps: 300,
    }
}

fn spawn(world: &mut World, p: Vec3) -> EntityHandle {
    world.spawn(Entity {
        transform: Transform {
            translation: p.to_array(),
            ..Default::default()
        },
        ..Entity::default()
    })
}

/// Milliseconds per step and the highest body's height afterwards.
struct Result {
    /// The first second, while everything moves.
    active: f64,
    ms: f64,
    top: f32,
    lowest: f32,
    floor_penetration: f32,
    rms_speed: f32,
    furthest_x: f32,
    body_penetration: Option<f32>,
}

fn summarize(mut runs: Vec<Result>) -> (Result, (f64, f64), (f64, f64)) {
    let mut active: Vec<_> = runs.iter().map(|r| r.active).collect();
    active.sort_by(f64::total_cmp);
    runs.sort_by(|a, b| a.ms.total_cmp(&b.ms));
    let active_range = (active[0], active[active.len() - 1]);
    let range = (runs[0].ms, runs[runs.len() - 1].ms);
    let mut median = runs.swap_remove(runs.len() / 2);
    median.active = active[active.len() / 2];
    (median, active_range, range)
}

fn floor_penetration(shape: &Kind, transform: &Transform) -> f32 {
    let extent = match shape {
        Kind::Ball => 0.5,
        Kind::Cube | Kind::SlidingCube | Kind::FreeCube | Kind::HeavyCube => {
            let rotation = Quat::from_array(transform.rotation);
            0.5 * ((rotation * Vec3::X).z.abs()
                + (rotation * Vec3::Y).z.abs()
                + (rotation * Vec3::Z).z.abs())
        }
        Kind::Floor => return 0.0,
    };
    (extent - transform.translation[2]).max(0.0)
}

fn cube_penetration(a: &Transform, b: &Transform) -> f32 {
    let axes = |rotation| {
        let q = Quat::from_array(rotation);
        [q * Vec3::X, q * Vec3::Y, q * Vec3::Z]
    };
    let a_axes = axes(a.rotation);
    let b_axes = axes(b.rotation);
    let offset = Vec3::from_array(a.translation) - Vec3::from_array(b.translation);
    let mut depth = f32::INFINITY;
    let mut test = |axis: Vec3| {
        if axis.length_squared() < 1.0e-10 {
            return;
        }
        let Some(axis) = axis.try_normalize() else {
            return;
        };
        let radius = |axes: &[Vec3; 3]| 0.5 * axes.iter().map(|v| v.dot(axis).abs()).sum::<f32>();
        depth = depth.min(radius(&a_axes) + radius(&b_axes) - offset.dot(axis).abs());
    };
    for axis in a_axes.into_iter().chain(b_axes) {
        test(axis);
    }
    for a in a_axes {
        for b in b_axes {
            test(a.cross(b));
        }
    }
    depth.max(0.0)
}

fn measure(
    world: &World,
    scene: &Scene,
    entities: &[EntityHandle],
    velocity: impl Fn(EntityHandle) -> Vec3,
    start: Instant,
    active: f64,
) -> Result {
    let ms = start.elapsed().as_secs_f64() * 1000.0 / scene.steps as f64;
    assert_eq!(
        world.iter().count(),
        scene.bodies.len(),
        "body count changed"
    );
    let heights = || {
        world.iter().skip(1).map(|e| {
            assert!(
                e.transform
                    .translation
                    .iter()
                    .chain(&e.transform.rotation)
                    .all(|v| v.is_finite()),
                "non-finite body transform"
            );
            let height = e.transform.translation[2];
            assert!(height >= 0.0, "invalid body height: {height}");
            height
        })
    };
    let mut max_penetration: f32 = 0.0;
    let mut speed_squared = 0.0;
    let mut furthest_x: f32 = 0.0;
    for ((kind, _), &entity) in scene.bodies.iter().zip(entities).skip(1) {
        let object = world.get(entity).expect("body entity must remain live");
        max_penetration = max_penetration.max(floor_penetration(kind, object.transform));
        furthest_x = furthest_x.max(object.transform.translation[0].abs());
        let speed = velocity(entity);
        assert!(speed.is_finite(), "non-finite velocity");
        speed_squared += speed.length_squared();
    }
    let body_penetration = if scene.name == "heavy" {
        let mut depth: f32 = 0.0;
        for (i, &a) in entities.iter().enumerate().skip(1) {
            for &b in &entities[i + 1..] {
                depth = depth.max(cube_penetration(
                    world.get(a).unwrap().transform,
                    world.get(b).unwrap().transform,
                ));
            }
        }
        Some(depth)
    } else {
        None
    };
    Result {
        active,
        ms,
        top: heights().fold(f32::MIN, f32::max),
        lowest: heights().fold(f32::MAX, f32::min),
        floor_penetration: max_penetration,
        rms_speed: (speed_squared / (entities.len() - 1) as f32).sqrt(),
        furthest_x,
        body_penetration,
    }
}

fn verity(scene: &Scene) -> Result {
    use verity_physics::*;
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let mut entities = Vec::with_capacity(scene.bodies.len());
    for (shape, p) in &scene.bodies {
        let entity = spawn(&mut world, *p);
        entities.push(entity);
        let mut body = match shape {
            Kind::Floor => PhysicsBody::new(
                RigidBody::Static,
                Collider::Cuboid {
                    half_extents: Vec3::new(100.0, 100.0, 0.5),
                },
            ),
            Kind::Cube | Kind::SlidingCube | Kind::FreeCube | Kind::HeavyCube => PhysicsBody::new(
                RigidBody::Dynamic,
                Collider::Cuboid {
                    half_extents: Vec3::splat(0.5),
                },
            ),
            Kind::Ball => PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius: 0.5 }),
        };
        if matches!(shape, Kind::SlidingCube | Kind::FreeCube) {
            body.velocity.linear = Vec3::new(5.0, 0.0, 0.0);
        }
        if matches!(shape, Kind::FreeCube) {
            body.material.friction = 0.0;
            body.material.static_friction = 0.0;
        }
        if matches!(shape, Kind::HeavyCube) {
            body.material.density *= 100.0;
        }
        physics.insert(&world, entity, body).unwrap();
    }
    let mut settings = PhysicsSettings::default();
    if let Ok(value) = std::env::var("VERITY_ITERATIONS") {
        settings.iterations = value
            .parse()
            .expect("VERITY_ITERATIONS must be a positive integer");
    }
    let start = Instant::now();
    let mut active = 0.0;
    for step in 0..scene.steps {
        physics.step(&mut world, &settings, DT);
        if step == ACTIVE - 1 {
            active = start.elapsed().as_secs_f64() * 1000.0 / ACTIVE as f64;
        }
    }
    measure(
        &world,
        scene,
        &entities,
        |entity| physics.get(entity).unwrap().velocity.linear,
        start,
        active,
    )
}

fn box3d(scene: &Scene) -> Result {
    use verity_box3d::*;
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let mut entities = Vec::with_capacity(scene.bodies.len());
    for (shape, p) in &scene.bodies {
        let entity = spawn(&mut world, *p);
        entities.push(entity);
        let mut body = match shape {
            Kind::Floor => PhysicsBody::new(
                RigidBody::Static,
                Collider::Cuboid {
                    half_extents: Vec3::new(100.0, 100.0, 0.5),
                },
            ),
            Kind::Cube | Kind::SlidingCube | Kind::FreeCube | Kind::HeavyCube => PhysicsBody::new(
                RigidBody::Dynamic,
                Collider::Cuboid {
                    half_extents: Vec3::splat(0.5),
                },
            ),
            Kind::Ball => PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius: 0.5 }),
        };
        if matches!(shape, Kind::SlidingCube | Kind::FreeCube) {
            body.velocity.linear = Vec3::new(5.0, 0.0, 0.0);
        }
        if matches!(shape, Kind::FreeCube) {
            body.material.friction = 0.0;
        }
        if matches!(shape, Kind::HeavyCube) {
            body.material.density *= 100.0;
        }
        physics.insert(&world, entity, body).unwrap();
    }
    let mut settings = Box3dSettings {
        gravity: GRAVITY,
        ..Default::default()
    };
    if let Ok(value) = std::env::var("BOX3D_SUBSTEPS") {
        settings.substeps = value
            .parse()
            .expect("BOX3D_SUBSTEPS must be a positive integer");
        assert!(settings.substeps > 0, "BOX3D_SUBSTEPS must be positive");
    }
    let start = Instant::now();
    let mut active = 0.0;
    for step in 0..scene.steps {
        physics.step(&mut world, &settings, DT);
        if step == ACTIVE - 1 {
            active = start.elapsed().as_secs_f64() * 1000.0 / ACTIVE as f64;
        }
    }
    measure(
        &world,
        scene,
        &entities,
        |entity| physics.get(entity).unwrap().velocity.linear,
        start,
        active,
    )
}

fn main() {
    let scale: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let filter = std::env::var("VERITY_SCENE").ok();
    assert!(
        filter.as_deref().is_none_or(|name| matches!(
            name,
            "pyramid" | "rain" | "slide" | "free_slide" | "heavy"
        )),
        "VERITY_SCENE must be pyramid, rain, slide, free_slide, or heavy"
    );
    for scene in [
        pyramid(20 * scale),
        pyramid(40 * scale),
        rain(8 * scale),
        rain(14 * scale),
        slide(),
        free_slide(),
        heavy(),
    ] {
        if filter.as_deref().is_some_and(|name| name != scene.name) {
            continue;
        }
        // Alternate run order to reduce bias from changing machine load.
        let (mut hs, mut bs) = (Vec::new(), Vec::new());
        for trial in 0..5 {
            if trial % 2 == 0 {
                hs.push(verity(&scene));
                bs.push(box3d(&scene));
            } else {
                bs.push(box3d(&scene));
                hs.push(verity(&scene));
            }
        }
        let ((h, har, hr), (b, bar, br)) = (summarize(hs), summarize(bs));
        println!(
            "{:8} {:5} bodies | active {:7.3} vs {:7.3} ms {:5.2}x | all {:7.3} vs {:7.3} ms {:5.2}x | top {:6.2} vs {:6.2} low {:5.2} vs {:5.2}",
            scene.name,
            scene.bodies.len() - 1,
            h.active,
            b.active,
            b.active / h.active,
            h.ms,
            b.ms,
            b.ms / h.ms,
            h.top,
            b.top,
            h.lowest,
            b.lowest,
        );
        println!(
            "         active range: {:.3}–{:.3} vs {:.3}–{:.3} ms | full-step range: {:.3}–{:.3} vs {:.3}–{:.3} ms",
            har.0, har.1, bar.0, bar.1, hr.0, hr.1, br.0, br.1
        );
        println!(
            "         final floor penetration: {:.4} vs {:.4} m | RMS linear speed: {:.4} vs {:.4} m/s",
            h.floor_penetration, b.floor_penetration, h.rms_speed, b.rms_speed
        );
        println!(
            "         final maximum |x|: {:.4} vs {:.4} m",
            h.furthest_x, b.furthest_x
        );
        if let (Some(a), Some(b)) = (h.body_penetration, b.body_penetration) {
            println!("         final cube penetration: {a:.4} vs {b:.4} m");
        }
        if scene.name == "free_slide" {
            let expected = 5.0 * DT * scene.steps as f32;
            println!(
                "         expected x: {expected:.4} m | position error: {:.4} vs {:.4} m",
                (h.furthest_x - expected).abs(),
                (b.furthest_x - expected).abs()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Kind, Result, box3d, cube_penetration, floor_penetration, free_slide, summarize, verity,
    };
    use glam::{Quat, Vec3};
    use verity_core::Transform;

    #[test]
    fn median_time_uses_its_own_final_state() {
        let runs = [(3.0, 2.0), (1.0, 3.0), (2.0, 1.0)]
            .into_iter()
            .map(|(active, ms)| Result {
                active,
                ms,
                top: ms as f32 * 10.0,
                lowest: ms as f32,
                floor_penetration: 0.0,
                rms_speed: 0.0,
                furthest_x: 0.0,
                body_penetration: None,
            })
            .collect();
        let (median, active_range, range) = summarize(runs);
        assert_eq!((median.active, median.ms), (2.0, 2.0));
        assert_eq!((median.top, median.lowest), (20.0, 2.0));
        assert_eq!((active_range, range), ((1.0, 3.0), (1.0, 3.0)));
    }

    #[test]
    fn floor_penetration_accounts_for_rotation() {
        let mut transform = Transform {
            translation: [0.0, 0.0, 0.5],
            ..Default::default()
        };
        assert_eq!(floor_penetration(&Kind::Ball, &transform), 0.0);
        assert_eq!(floor_penetration(&Kind::Cube, &transform), 0.0);
        transform.rotation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4).to_array();
        assert!(
            (floor_penetration(&Kind::Cube, &transform) - (std::f32::consts::FRAC_1_SQRT_2 - 0.5))
                .abs()
                < 1.0e-6
        );
        transform.translation = Vec3::new(0.0, 0.0, 1.0).to_array();
        assert_eq!(floor_penetration(&Kind::Cube, &transform), 0.0);
    }

    #[test]
    fn cube_penetration_distinguishes_overlap_and_separation() {
        let a = Transform::default();
        let mut b = Transform {
            translation: [0.9, 0.0, 0.0],
            ..Default::default()
        };
        assert!((cube_penetration(&a, &b) - 0.1).abs() < 1.0e-6);
        b.translation[0] = 1.1;
        assert_eq!(cube_penetration(&a, &b), 0.0);
        b.translation[0] = 1.0;
        b.rotation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4).to_array();
        assert!(
            (cube_penetration(&a, &b) - (std::f32::consts::FRAC_1_SQRT_2 - 0.5)).abs() < 1.0e-6
        );
    }

    #[test]
    fn frictionless_sliding_preserves_horizontal_momentum() {
        let scene = free_slide();
        for result in [verity(&scene), box3d(&scene)] {
            assert!((result.furthest_x - 25.0).abs() < 0.01);
            assert!((result.rms_speed - 5.0).abs() < 0.01);
        }
    }
}
