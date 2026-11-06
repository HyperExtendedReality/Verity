//! Physical accuracy against closed-form references, for verity-physics and
//! Box3D: `cargo run --release -p verity-physics --example quality`.
//! Gravity is tilted to make a slope, so every scene has a flat floor at z = 0.
//! Set `VERITY_ITERATIONS` to compare a different AVBD iteration count.

use glam::{Quat, Vec3};
use verity_core::{Entity, EntityHandle, Transform, World};

const DT: f32 = 1.0 / 60.0;
const G: f32 = 9.81;
const MU: f32 = 0.6;

#[derive(Clone, Copy, PartialEq)]
enum Form {
    Cube,
    Ball,
    Slab,
}

/// Half extents of a body with three different principal moments.
const SLAB: Vec3 = Vec3::new(0.5, 0.3, 0.15);

#[derive(Clone, Copy)]
struct Item {
    shape: Form,
    position: Vec3,
    velocity: Vec3,
    spin: Vec3,
    friction: f32,
    density: f32,
}

impl Item {
    fn new(shape: Form, position: Vec3) -> Self {
        Self {
            shape,
            position,
            velocity: Vec3::ZERO,
            spin: Vec3::ZERO,
            friction: MU,
            density: 1000.0,
        }
    }
}

struct Spec {
    items: Vec<Item>,
    gravity: Vec3,
    steps: usize,
    floor_friction: f32,
    /// Build the floor from 1 m tiles along x instead of one slab.
    tiles: bool,
}

impl Spec {
    fn new(items: Vec<Item>, steps: usize) -> Self {
        Self {
            items,
            gravity: Vec3::new(0.0, 0.0, -G),
            steps,
            floor_friction: MU,
            tiles: false,
        }
    }
}

/// What a run produced: final state, plus extremes over every step.
struct Trace {
    position: Vec<Vec3>,
    velocity: Vec<Vec3>,
    angular: Vec<Vec3>,
    rotation: Vec<Quat>,
    /// Deepest floor penetration in any step, and its mean over the last 60.
    peak_floor: f32,
    late_floor: f32,
    /// Largest speed of any body after the first half of the run.
    late_speed: f32,
    /// Deepest overlap between any two bodies in any step, and its mean over
    /// the last 60.
    peak_overlap: f32,
    late_overlap: f32,
}

fn floor_depth(item: &Item, transform: &Transform) -> f32 {
    let extent = match item.shape {
        Form::Ball => 0.5,
        Form::Slab => return 0.0,
        Form::Cube => {
            let q = Quat::from_array(transform.rotation);
            0.5 * ((q * Vec3::X).z.abs() + (q * Vec3::Y).z.abs() + (q * Vec3::Z).z.abs())
        }
    };
    (extent - transform.translation[2]).max(0.0)
}

/// Overlap depth of two bodies; slabs do not meet in these scenes.
fn pair_overlap(sa: Form, a: &Transform, sb: Form, b: &Transform) -> f32 {
    let (pa, pb) = (
        Vec3::from_array(a.translation),
        Vec3::from_array(b.translation),
    );
    if pa.distance_squared(pb) > 3.1 {
        return 0.0;
    }
    let axes = |t: &Transform| {
        let q = Quat::from_array(t.rotation);
        [q * Vec3::X, q * Vec3::Y, q * Vec3::Z]
    };
    match (sa, sb) {
        (Form::Ball, Form::Ball) => (1.0 - pa.distance(pb)).max(0.0),
        (Form::Cube, Form::Ball) => ball_in_cube(pb, pa, axes(a)),
        (Form::Ball, Form::Cube) => ball_in_cube(pa, pb, axes(b)),
        (Form::Cube, Form::Cube) => {
            let (ax, bx) = (axes(a), axes(b));
            let offset = pa - pb;
            let mut depth = f32::INFINITY;
            let mut test = |axis: Vec3| {
                if let Some(axis) = axis.try_normalize() {
                    let radius =
                        |x: &[Vec3; 3]| 0.5 * x.iter().map(|v| v.dot(axis).abs()).sum::<f32>();
                    depth = depth.min(radius(&ax) + radius(&bx) - offset.dot(axis).abs());
                }
            };
            for axis in ax.into_iter().chain(bx) {
                test(axis);
            }
            for x in ax {
                for y in bx {
                    if x.cross(y).length_squared() > 1.0e-6 {
                        test(x.cross(y));
                    }
                }
            }
            depth.max(0.0)
        }
        _ => 0.0,
    }
}

fn ball_in_cube(ball: Vec3, cube: Vec3, axes: [Vec3; 3]) -> f32 {
    let d = ball - cube;
    let local = Vec3::new(d.dot(axes[0]), d.dot(axes[1]), d.dot(axes[2]));
    let clamped = local.clamp(Vec3::splat(-0.5), Vec3::splat(0.5));
    if clamped == local {
        return 0.5 + 0.5 - local.abs().max_element();
    }
    (0.5 - (local - clamped).length()).max(0.0)
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

macro_rules! engine {
    ($name:ident, $module:ident, $configure:expr, $settings:expr) => {
        fn $name(spec: &Spec) -> Trace {
            use $module::*;
            let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
            let configure = $configure;
            let slabs: Vec<(Vec3, Vec3)> = if spec.tiles {
                (-5..60)
                    .map(|i| (Vec3::new(i as f32, 0.0, -0.5), Vec3::new(0.5, 10.0, 0.5)))
                    .collect()
            } else {
                vec![(Vec3::new(0.0, 0.0, -0.5), Vec3::new(500.0, 500.0, 0.5))]
            };
            for (center, half_extents) in slabs {
                let floor = spawn(&mut world, center);
                let collider = Collider::Cuboid { half_extents };
                let mut body = PhysicsBody::new(RigidBody::Static, collider);
                body.material.friction = spec.floor_friction;
                configure(&mut body.material);
                physics.insert(&world, floor, body).unwrap();
            }
            let entities: Vec<_> = spec
                .items
                .iter()
                .map(|item| {
                    let entity = spawn(&mut world, item.position);
                    let collider = match item.shape {
                        Form::Cube => Collider::Cuboid {
                            half_extents: Vec3::splat(0.5),
                        },
                        Form::Ball => Collider::Sphere { radius: 0.5 },
                        Form::Slab => Collider::Cuboid { half_extents: SLAB },
                    };
                    let mut body = PhysicsBody::new(RigidBody::Dynamic, collider);
                    body.velocity.linear = item.velocity;
                    body.velocity.angular = item.spin;
                    body.material.friction = item.friction;
                    body.material.density = item.density;
                    configure(&mut body.material);
                    physics.insert(&world, entity, body).unwrap();
                    entity
                })
                .collect();
            let settings = $settings(spec.gravity);
            let (mut peak_floor, mut late_floor, mut late_speed) = (0.0f32, 0.0f32, 0.0f32);
            let (mut peak_overlap, mut late_overlap) = (0.0f32, 0.0f32);
            for step in 0..spec.steps {
                physics.step(&mut world, &settings, DT);
                let mut deepest = 0.0f32;
                for (item, &entity) in spec.items.iter().zip(&entities) {
                    let transform = world.get(entity).unwrap().transform;
                    deepest = deepest.max(floor_depth(item, transform));
                    if step * 2 >= spec.steps {
                        let speed = physics.get(entity).unwrap().velocity.linear.length();
                        late_speed = late_speed.max(speed);
                    }
                }
                let mut overlap = 0.0f32;
                for (i, &a) in entities.iter().enumerate() {
                    for (j, &b) in entities.iter().enumerate().skip(i + 1) {
                        let (ta, tb) = (
                            world.get(a).unwrap().transform,
                            world.get(b).unwrap().transform,
                        );
                        overlap = overlap.max(pair_overlap(
                            spec.items[i].shape,
                            ta,
                            spec.items[j].shape,
                            tb,
                        ));
                    }
                }
                peak_overlap = peak_overlap.max(overlap);
                if step + 60 >= spec.steps {
                    late_overlap += overlap / 60.0;
                }
                peak_floor = peak_floor.max(deepest);
                if step + 60 >= spec.steps {
                    late_floor += deepest / 60.0;
                }
            }
            let at =
                |e: &EntityHandle| Vec3::from_array(world.get(*e).unwrap().transform.translation);
            Trace {
                position: entities.iter().map(at).collect(),
                velocity: (entities.iter())
                    .map(|e| physics.get(*e).unwrap().velocity.linear)
                    .collect(),
                angular: (entities.iter())
                    .map(|e| physics.get(*e).unwrap().velocity.angular)
                    .collect(),
                rotation: (entities.iter())
                    .map(|e| Quat::from_array(world.get(*e).unwrap().transform.rotation))
                    .collect(),
                peak_floor,
                late_floor,
                late_speed,
                peak_overlap,
                late_overlap,
            }
        }
    };
}

engine!(
    verity,
    verity_physics,
    |m: &mut verity_physics::PhysicsMaterial| m.static_friction = m.friction,
    |gravity| {
        let mut settings = verity_physics::PhysicsSettings {
            gravity,
            ..Default::default()
        };
        if let Ok(value) = std::env::var("VERITY_ITERATIONS") {
            settings.iterations = value.parse().expect("iterations must be a number");
        }
        settings
    }
);
engine!(
    box3d,
    verity_box3d,
    |_: &mut verity_box3d::PhysicsMaterial| {},
    |gravity| verity_box3d::Box3dSettings {
        gravity,
        ..Default::default()
    }
);

/// One measured quantity, its reference, and both engines' values.
fn row(name: &str, unit: &str, reference: f32, h: f32, b: f32) {
    let error = |v: f32| {
        if reference != 0.0 {
            format!("{:+6.1}%", (v - reference) / reference * 100.0)
        } else {
            "      ".to_string()
        }
    };
    println!(
        "{name:34} ref {reference:8.4} {unit:4} | verity {h:8.4} {} | box3d {b:8.4} {}",
        error(h),
        error(b)
    );
}

fn both(spec: &Spec) -> (Trace, Trace) {
    (verity(spec), box3d(spec))
}

fn sliding_stop() {
    for speed in [2.0, 5.0, 10.0] {
        let mut item = Item::new(Form::Cube, Vec3::new(0.0, 0.0, 0.5));
        item.velocity.x = speed;
        let (h, b) = both(&Spec::new(vec![item], 300));
        let reference = speed * speed / (2.0 * MU * G);
        row(
            &format!("slide stop, {speed} m/s"),
            "m",
            reference,
            h.position[0].x,
            b.position[0].x,
        );
    }
}

/// Gravity tilted by `angle`: x is down the slope.
fn tilted(angle: f32) -> Vec3 {
    Vec3::new(angle.sin(), 0.0, -angle.cos()) * G
}

fn tiled_slide() {
    for speed in [2.0, 5.0, 10.0, 20.0] {
        let mut item = Item::new(Form::Cube, Vec3::new(0.0, 0.0, 0.5));
        item.velocity.x = speed;
        let mut spec = Spec::new(vec![item], 420);
        spec.tiles = true;
        let (h, b) = both(&spec);
        let reference = speed * speed / (2.0 * MU * G);
        row(
            &format!("tile slide stop, {speed} m/s"),
            "m",
            reference,
            h.position[0].x,
            b.position[0].x,
        );
    }
}

fn slope_slide() {
    for degrees in [35.0f32, 45.0, 60.0] {
        let angle = degrees.to_radians();
        let mut spec = Spec::new(vec![Item::new(Form::Cube, Vec3::new(0.0, 0.0, 0.5))], 120);
        spec.gravity = tilted(angle);
        let (h, b) = both(&spec);
        let time = 120.0 * DT;
        let reference = 0.5 * G * (angle.sin() - MU * angle.cos()) * time * time;
        row(
            &format!("cube slope {degrees} deg, 2 s"),
            "m",
            reference,
            h.position[0].x,
            b.position[0].x,
        );
    }
}

fn slope_hold() {
    // tan(30 deg) = 0.577 < 0.6: static friction must hold.
    let mut spec = Spec::new(vec![Item::new(Form::Cube, Vec3::new(0.0, 0.0, 0.5))], 300);
    spec.gravity = tilted(30.0f32.to_radians());
    let (h, b) = both(&spec);
    row(
        "cube held at 30 deg, 5 s",
        "m",
        0.0,
        h.position[0].x,
        b.position[0].x,
    );
}

fn rolling() {
    for degrees in [10.0f32, 20.0] {
        let angle = degrees.to_radians();
        let mut spec = Spec::new(vec![Item::new(Form::Ball, Vec3::new(0.0, 0.0, 0.5))], 120);
        spec.gravity = tilted(angle);
        let (h, b) = both(&spec);
        let time = 120.0 * DT;
        let reference = 0.5 * (5.0 / 7.0) * G * angle.sin() * time * time;
        row(
            &format!("ball rolls {degrees} deg, 2 s"),
            "m",
            reference,
            h.position[0].x,
            b.position[0].x,
        );
    }
}

fn slide_to_roll() {
    // A sliding ball ends rolling at 5/7 of its speed, whatever the friction.
    let mut item = Item::new(Form::Ball, Vec3::new(0.0, 0.0, 0.5));
    item.velocity.x = 7.0;
    let (h, b) = both(&Spec::new(vec![item], 180));
    row(
        "ball slide to roll speed",
        "m/s",
        5.0,
        h.velocity[0].x,
        b.velocity[0].x,
    );
    let spin = |t: &Trace| t.angular[0].y * 0.5;
    row("ball spin * radius", "m/s", 5.0, spin(&h), spin(&b));
}

fn stack() {
    for count in [5, 20] {
        let items = (0..count)
            .map(|i| Item::new(Form::Cube, Vec3::new(0.0, 0.0, 0.5 + i as f32)))
            .collect();
        let (h, b) = both(&Spec::new(items, 300));
        let top = |t: &Trace| t.position[count - 1].z;
        row(
            &format!("stack of {count}, top height"),
            "m",
            count as f32 - 0.5,
            top(&h),
            top(&b),
        );
        let drift = |t: &Trace| {
            t.position
                .iter()
                .map(|p| p.x.hypot(p.y))
                .fold(0.0, f32::max)
        };
        row(
            &format!("stack of {count}, sideways drift"),
            "m",
            0.0,
            drift(&h),
            drift(&b),
        );
    }
}

fn heavy_on_light() {
    let mut items: Vec<_> = (0..3)
        .map(|i| Item::new(Form::Cube, Vec3::new(0.0, 0.0, 0.5 + i as f32)))
        .collect();
    items[2].density *= 100.0;
    let (h, b) = both(&Spec::new(items, 300));
    row(
        "100x cube on 2, top",
        "m",
        2.5,
        h.position[2].z,
        b.position[2].z,
    );
}

fn impact() {
    // From 5 m a body arrives at 9.9 m/s, 16 cm per step; from 40 m, 47 cm.
    for height in [5.0f32, 20.0, 40.0] {
        for (name, shape) in [("cube", Form::Cube), ("ball", Form::Ball)] {
            let item = Item::new(shape, Vec3::new(0.0, 0.0, height + 0.5));
            let (h, b) = both(&Spec::new(vec![item], 240));
            let label = format!("drop {name} {height} m");
            row(
                &format!("{label}, peak depth"),
                "m",
                0.0,
                h.peak_floor,
                b.peak_floor,
            );
            row(
                &format!("{label}, late depth"),
                "m",
                0.0,
                h.late_floor,
                b.late_floor,
            );
            row(
                &format!("{label}, late speed"),
                "m/s",
                0.0,
                h.late_speed,
                b.late_speed,
            );
        }
    }
}

/// Body-frame angular velocity after `time`, from Euler's equations by RK4.
fn euler_reference(moments: Vec3, mut w: Vec3, time: f32) -> Vec3 {
    let rate = |w: Vec3| -> Vec3 {
        Vec3::new(
            (moments.y - moments.z) * w.y * w.z / moments.x,
            (moments.z - moments.x) * w.z * w.x / moments.y,
            (moments.x - moments.y) * w.x * w.y / moments.z,
        )
    };
    let h = 1.0e-4;
    for _ in 0..(time / h) as usize {
        let (a, b) = (rate(w), rate(w + rate(w) * (h / 2.0)));
        let c = rate(w + b * (h / 2.0));
        let d = rate(w + c * h);
        w += (a + (b + c) * 2.0 + d) * (h / 6.0);
    }
    w
}

/// Two cubes on a frictionless floor line, no gravity: momentum must survive.
fn collisions() {
    for speed in [5.0f32, 20.0] {
        let mut left = Item::new(Form::Cube, Vec3::new(-3.0, 0.0, 50.0));
        let mut right = Item::new(Form::Cube, Vec3::new(3.0, 0.0, 50.0));
        (left.velocity.x, right.velocity.x) = (speed, -speed);
        let mut spec = Spec::new(vec![left, right], 120);
        spec.gravity = Vec3::ZERO;
        let (h, b) = both(&spec);
        let total = |t: &Trace| t.velocity[0].x + t.velocity[1].x;
        row(
            &format!("head-on {speed} m/s, momentum"),
            "m/s",
            0.0,
            total(&h),
            total(&b),
        );
        row(
            &format!("head-on {speed} m/s, peak overlap"),
            "m",
            0.0,
            h.peak_overlap,
            b.peak_overlap,
        );
        right.velocity.x = 0.0;
        let mut spec = Spec::new(vec![left, right], 120);
        spec.gravity = Vec3::ZERO;
        let (h, b) = both(&spec);
        row(
            &format!("strike {speed} m/s, momentum"),
            "m/s",
            speed,
            total(&h),
            total(&b),
        );
    }
}

fn tumbling() {
    // Spin about the intermediate axis: the body must flip (Dzhanibekov).
    let spin = Vec3::new(0.02, 4.0, 0.02);
    let mut item = Item::new(Form::Slab, Vec3::new(0.0, 0.0, 50.0));
    item.spin = spin;
    let mut spec = Spec::new(vec![item], 300);
    spec.gravity = Vec3::ZERO;
    let (h, b) = both(&spec);
    let square = SLAB * SLAB;
    let moments = Vec3::new(
        square.y + square.z,
        square.x + square.z,
        square.x + square.y,
    ) / 3.0;
    let momentum = |rotation: Quat, w: Vec3| rotation * (moments * (rotation.inverse() * w));
    let start = moments * spin;
    let drift =
        |t: &Trace| (momentum(t.rotation[0], t.angular[0]) - start).length() / start.length();
    row("tumble, momentum drift", "", 0.0, drift(&h), drift(&b));
    let energy = |t: &Trace| {
        let w = t.rotation[0].inverse() * t.angular[0];
        (moments * w).dot(w) / (moments * spin).dot(spin)
    };
    row("tumble, energy ratio", "", 1.0, energy(&h), energy(&b));
    let time = 300.0 * DT;
    let reference = euler_reference(moments, spin, time);
    let body = |t: &Trace| (t.rotation[0].inverse() * t.angular[0]).y;
    row(
        "tumble, body-frame spin y",
        "r/s",
        reference.y,
        body(&h),
        body(&b),
    );
}

fn pile() {
    const SEEDS: usize = 8;
    let (mut verity_sums, mut box3d_sums) = ([0.0f32; 4], [0.0f32; 4]);
    for seed in 0..SEEDS {
        let mut items = Vec::new();
        for z in 0..6 {
            for y in 0..6 {
                for x in 0..6 {
                    let shape = if (x + y + z) % 2 == 0 {
                        Form::Cube
                    } else {
                        Form::Ball
                    };
                    let wobble = ((x * 7 + y * 13 + z * 29 + seed * 5) % 11) as f32 * 0.008;
                    items.push(Item::new(
                        shape,
                        Vec3::new(x as f32, y as f32, z as f32) * 1.2
                            + Vec3::new(wobble - 3.6, 0.04 - wobble - 3.6, 1.0),
                    ));
                }
            }
        }
        let (h, b) = both(&Spec::new(items, 240));
        for (sums, t) in [(&mut verity_sums, &h), (&mut box3d_sums, &b)] {
            let values = [t.peak_floor, t.late_floor, t.peak_overlap, t.late_overlap];
            for (sum, value) in sums.iter_mut().zip(values) {
                *sum += value / SEEDS as f32;
            }
        }
    }
    let names = [
        "pile, peak floor penetration",
        "pile, late floor penetration",
        "pile, peak body overlap",
        "pile, late body overlap",
    ];
    for ((name, h), b) in names.iter().zip(verity_sums).zip(box3d_sums) {
        row(name, "m", 0.0, h, b);
    }
}

/// A ball on a rigid rod: period against the small-angle formula. Verity only:
/// the Box3D bridge has no joints.
fn pendulum() {
    use verity_physics::*;
    let (length, radius, angle) = (2.0f32, 0.1f32, 0.2f32);
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let offset = Vec3::new(angle.sin(), 0.0, -angle.cos()) * length;
    let top = Vec3::new(0.0, 0.0, 10.0);
    let entity = spawn(&mut world, top + offset);
    let ball = PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius });
    physics.insert(&world, entity, ball).unwrap();
    physics
        .add_joint(&world, Joint::new(None, entity, top, -offset))
        .unwrap();
    let settings = PhysicsSettings::default();
    let (mut crossings, mut last, mut peak, mut stretch) = (Vec::new(), offset.x, 0.0f32, 0.0f32);
    for step in 0..1800 {
        physics.step(&mut world, &settings, DT);
        let p = Vec3::from_array(world.get(entity).unwrap().transform.translation);
        let x = p.x;
        if last < 0.0 && x >= 0.0 {
            crossings.push((step as f32 + last / (last - x)) * DT);
        }
        last = x;
        stretch = stretch.max((p.distance(top) - length).abs());
        if step >= 1680 {
            peak = peak.max(x.abs());
        }
    }
    let inertia = 0.4 * radius * radius + length * length;
    let reference =
        std::f32::consts::TAU * (inertia / (G * length)).sqrt() * (1.0 + angle * angle / 16.0);
    let period = (crossings[crossings.len() - 1] - crossings[0]) / (crossings.len() - 1) as f32;
    let start = length * angle.sin();
    println!(
        "pendulum: period {period:.4} s, reference {reference:.4} s ({:+.2}%), \
         amplitude after 30 s {:.1}% of start, worst rod stretch {stretch:.5} m",
        (period - reference) / reference * 100.0,
        peak / start * 100.0,
    );
}

/// Ten hard ball joints released level: a whip with no analytic answer, so
/// this reports mechanical energy left above the hanging rest state.
fn chain() {
    use verity_physics::*;
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let top = Vec3::new(0.0, 0.0, 30.0);
    let half = Vec3::new(0.5, 0.1, 0.1);
    let links: Vec<_> = (0..10)
        .map(|i| {
            let entity = spawn(&mut world, top + Vec3::new(0.5 + i as f32, 0.0, 0.0));
            let bar = PhysicsBody::new(RigidBody::Dynamic, Collider::Cuboid { half_extents: half });
            physics.insert(&world, entity, bar).unwrap();
            entity
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
    let rest: f32 = (0..10).map(|i| G * (top.z - 0.5 - i as f32)).sum();
    let energy = |world: &World, physics: &PhysicsWorld| -> f32 {
        let inertia = (4.0 * half.x * half.x + 4.0 * half.z * half.z) / 12.0;
        (links.iter())
            .map(|&e| {
                let v = physics.get(e).unwrap().velocity;
                let z = world.get(e).unwrap().transform.translation[2];
                G * z + 0.5 * v.linear.length_squared() + 0.5 * inertia * v.angular.length_squared()
            })
            .sum::<f32>()
            - rest
    };
    let start = energy(&world, &physics);
    let settings = PhysicsSettings::default();
    let (mut highest, mut spin, mut marks) = (f32::MIN, 0.0f32, Vec::new());
    for step in 1..=1800 {
        physics.step(&mut world, &settings, DT);
        let e = energy(&world, &physics);
        highest = highest.max(e);
        spin = links.iter().fold(spin, |s, &l| {
            s.max(physics.get(l).unwrap().velocity.angular.length())
        });
        if step % 300 == 0 {
            marks.push(format!("{:.0}%", e / start * 100.0));
        }
    }
    println!(
        "chain: energy each 5 s {}, highest {:.1}% of start, fastest spin {spin:.1} rad/s",
        marks.join(" "),
        highest / start * 100.0,
    );
}

fn main() {
    sliding_stop();
    tiled_slide();
    slope_slide();
    slope_hold();
    rolling();
    slide_to_roll();
    stack();
    heavy_on_light();
    impact();
    collisions();
    tumbling();
    pile();
    pendulum();
    chain();
}
