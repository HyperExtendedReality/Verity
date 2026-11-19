//! Convex hull and triangle mesh colliders, and cooking them from meshes.

use glam::{Quat, Vec3};
use std::sync::Arc;
use verity_core::{Entity, EntityHandle, Material, Transform, World};
use verity_physics::*;

fn spawn(world: &mut World, physics: &mut PhysicsWorld, at: Vec3, body: PhysicsBody) -> EntityHandle {
    spawn_turned(world, physics, at, Quat::IDENTITY, body)
}

fn spawn_turned(
    world: &mut World,
    physics: &mut PhysicsWorld,
    at: Vec3,
    rotation: Quat,
    body: PhysicsBody,
) -> EntityHandle {
    let entity = world.spawn(Entity {
        name: "body".into(),
        transform: Transform {
            translation: at.to_array(),
            rotation: rotation.to_array(),
            ..Default::default()
        },
        material: Material::default(),
        ..Entity::default()
    });
    physics.insert(world, entity, body).unwrap();
    entity
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

fn corners(half: f32) -> Vec<Vec3> {
    (0..8)
        .map(|i| {
            let s = |bit: usize| if i >> bit & 1 == 0 { -half } else { half };
            Vec3::new(s(0), s(1), s(2))
        })
        .collect()
}

fn hull_cube(half: f32) -> Collider {
    Collider::Hull(Arc::new(ConvexHull::new(&corners(half)).unwrap()))
}

fn box_floor(world: &mut World, physics: &mut PhysicsWorld) {
    let floor = Collider::Cuboid {
        half_extents: Vec3::new(20.0, 20.0, 0.5),
    };
    spawn(world, physics, Vec3::new(0.0, 0.0, -0.5), PhysicsBody::new(RigidBody::Static, floor));
}

/// A grid of `n` by `n` quads of `size` metres at height zero, as a mesh.
fn grid(n: usize, size: f32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let mut positions = Vec::new();
    for y in 0..=n {
        for x in 0..=n {
            let half = 0.5 * n as f32 * size;
            positions.push(Vec3::new(x as f32 * size - half, y as f32 * size - half, 0.0));
        }
    }
    let at = |x: usize, y: usize| (y * (n + 1) + x) as u32;
    let mut indices = Vec::new();
    for y in 0..n {
        for x in 0..n {
            indices.push([at(x, y), at(x + 1, y), at(x + 1, y + 1)]);
            indices.push([at(x, y), at(x + 1, y + 1), at(x, y + 1)]);
        }
    }
    (positions, indices)
}

fn mesh_floor(world: &mut World, physics: &mut PhysicsWorld) {
    let (positions, indices) = grid(20, 1.0);
    let mesh = Collider::Mesh(Arc::new(TriMesh::new(&positions, &indices)));
    spawn(world, physics, Vec3::ZERO, PhysicsBody::new(RigidBody::Static, mesh));
}

#[test]
fn a_hull_cube_falls_and_rests_on_a_box_floor() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    box_floor(&mut world, &mut physics);
    let cube = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 2.0),
        PhysicsBody::new(RigidBody::Dynamic, hull_cube(0.5)),
    );
    run(&mut world, &mut physics, 240);
    let z = position(&world, cube).z;
    assert!((z - 0.5).abs() < 0.01, "rests at {z}");
    assert!(physics.is_sleeping(cube));
}

#[test]
fn hull_cubes_stack() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    box_floor(&mut world, &mut physics);
    let cubes: Vec<_> = (0..4)
        .map(|i| {
            let body = PhysicsBody::new(RigidBody::Dynamic, hull_cube(0.5));
            spawn(&mut world, &mut physics, Vec3::new(0.0, 0.0, 0.5 + i as f32 * 1.01), body)
        })
        .collect();
    run(&mut world, &mut physics, 300);
    for (i, &cube) in cubes.iter().enumerate() {
        let p = position(&world, cube);
        let expected = 0.5 + i as f32;
        assert!((p.z - expected).abs() < 0.03, "cube {i} at {p}");
        assert!(p.truncate().length() < 0.02, "cube {i} drifted to {p}");
    }
}

#[test]
fn a_tumbling_hull_settles_flat_on_a_face() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    box_floor(&mut world, &mut physics);
    let turn = Quat::from_euler(glam::EulerRot::XYZ, 0.6, 0.4, 0.2);
    let cube = spawn_turned(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 1.5),
        turn,
        PhysicsBody::new(RigidBody::Dynamic, hull_cube(0.5)),
    );
    run(&mut world, &mut physics, 360);
    let z = position(&world, cube).z;
    assert!((z - 0.5).abs() < 0.01, "rests at {z}");
}

#[test]
fn spheres_and_capsules_rest_on_a_static_hull() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let slab: Vec<Vec3> = corners(1.0).iter().map(|c| *c * Vec3::new(4.0, 4.0, 0.5)).collect();
    let slab = Collider::Hull(Arc::new(ConvexHull::new(&slab).unwrap()));
    spawn(&mut world, &mut physics, Vec3::new(0.0, 0.0, -0.5), PhysicsBody::new(RigidBody::Static, slab));
    let ball = spawn(
        &mut world,
        &mut physics,
        Vec3::new(-1.0, 0.0, 1.0),
        PhysicsBody::new(RigidBody::Dynamic, Collider::Sphere { radius: 0.3 }),
    );
    let capsule = Collider::Capsule {
        radius: 0.2,
        half_height: 0.5,
    };
    let lying = spawn_turned(
        &mut world,
        &mut physics,
        Vec3::new(1.0, 0.0, 1.0),
        Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
        PhysicsBody::new(RigidBody::Dynamic, capsule),
    );
    run(&mut world, &mut physics, 240);
    let (b, c) = (position(&world, ball), position(&world, lying));
    assert!((b.z - 0.3).abs() < 0.01, "ball at {b}");
    assert!((c.z - 0.2).abs() < 0.01, "capsule at {c}");
    assert!(physics.is_sleeping(lying), "a lying capsule rests on both ends");
}

#[test]
fn boxes_hulls_and_balls_rest_on_a_mesh() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    mesh_floor(&mut world, &mut physics);
    let cuboid = Collider::Cuboid {
        half_extents: Vec3::splat(0.5),
    };
    let shapes = [
        (cuboid, 0.5),
        (hull_cube(0.5), 0.5),
        (Collider::Sphere { radius: 0.4 }, 0.4),
    ];
    let bodies: Vec<_> = (shapes.into_iter().enumerate())
        .map(|(i, (shape, rest))| {
            // Off the grid lines, and across them.
            let at = Vec3::new(i as f32 * 3.0 - 3.0 + 0.3, 0.0, 1.5);
            (spawn(&mut world, &mut physics, at, PhysicsBody::new(RigidBody::Dynamic, shape)), rest)
        })
        .collect();
    run(&mut world, &mut physics, 240);
    for (body, rest) in bodies {
        let p = position(&world, body);
        assert!((p.z - rest).abs() < 0.01, "rests at {p}, not {rest}");
        assert!(physics.is_sleeping(body), "{p} still moving");
    }
}

#[test]
fn a_box_slides_over_mesh_seams_without_catching() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    mesh_floor(&mut world, &mut physics);
    let mut body = PhysicsBody::new(
        RigidBody::Dynamic,
        Collider::Cuboid {
            half_extents: Vec3::new(0.5, 0.3, 0.2),
        },
    );
    body.material.friction = 0.0;
    body.material.static_friction = 0.0;
    body.velocity.linear = Vec3::new(4.0, 0.3, 0.0);
    let slider = spawn(&mut world, &mut physics, Vec3::new(-8.0, 0.0, 0.2), body);
    let (mut highest, mut lowest) = (0.0f32, f32::INFINITY);
    for _ in 0..180 {
        run(&mut world, &mut physics, 1);
        let z = position(&world, slider).z;
        (highest, lowest) = (highest.max(z), lowest.min(z));
    }
    let v = physics.get(slider).unwrap().velocity;
    assert!(position(&world, slider).x > 3.0, "did not cross the seams");
    assert!(highest < 0.21 && lowest > 0.19, "between {lowest} and {highest}");
    assert!(v.linear.x > 3.9, "caught on seams: {v:?}");
    assert!(v.angular.length() < 0.05, "tripped on seams: {v:?}");
}

#[test]
fn a_box_pushed_into_a_mesh_corner_holds_both_faces() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    // One mesh: the floor and a wall at x = 1 facing back along -x.
    let (mut positions, mut indices) = grid(6, 1.0);
    let base = positions.len() as u32;
    positions.extend([
        Vec3::new(1.0, -3.0, 0.0),
        Vec3::new(1.0, 3.0, 0.0),
        Vec3::new(1.0, 3.0, 3.0),
        Vec3::new(1.0, -3.0, 3.0),
    ]);
    indices.extend([[base, base + 2, base + 1], [base, base + 3, base + 2]]);
    let mesh = Collider::Mesh(Arc::new(TriMesh::new(&positions, &indices)));
    spawn(&mut world, &mut physics, Vec3::ZERO, PhysicsBody::new(RigidBody::Static, mesh));
    let mut body = PhysicsBody::new(
        RigidBody::Dynamic,
        Collider::Cuboid {
            half_extents: Vec3::splat(0.5),
        },
    );
    // Pushed into the wall as hard as gravity pulls it down.
    body.force.force = Vec3::new(1000.0 * 9.81, 0.0, 0.0);
    let pushed = spawn(&mut world, &mut physics, Vec3::new(-0.2, 0.0, 0.5), body);
    run(&mut world, &mut physics, 240);
    let p = position(&world, pushed);
    assert!((p.z - 0.5).abs() < 0.01, "sank or lifted to {p}");
    assert!((p.x - 0.5).abs() < 0.01, "into or off the wall: {p}");
}

#[test]
fn a_kinematic_mesh_carries_what_rests_on_it() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let (positions, indices) = grid(4, 1.0);
    let mut platform = PhysicsBody::new(
        RigidBody::Kinematic,
        Collider::Mesh(Arc::new(TriMesh::new(&positions, &indices))),
    );
    platform.velocity.linear = Vec3::new(1.0, 0.0, 0.0);
    spawn(&mut world, &mut physics, Vec3::ZERO, platform);
    let rider = spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 0.5),
        PhysicsBody::new(RigidBody::Dynamic, hull_cube(0.5)),
    );
    run(&mut world, &mut physics, 60);
    let p = position(&world, rider);
    assert!((p.x - 1.0).abs() < 0.1, "left behind at {p}");
    assert!((p.z - 0.5).abs() < 0.02, "at {p}");
}

#[test]
fn rays_hit_hull_and_mesh_colliders() {
    let hull = hull_cube(1.0);
    let (t, n) = hull.ray(Vec3::new(0.2, -5.0, 0.1), Vec3::Y, 100.0).unwrap();
    assert!((t - 4.0).abs() < 1.0e-4 && n.abs_diff_eq(-Vec3::Y, 1.0e-5));
    let (positions, indices) = grid(2, 1.0);
    let mesh = Collider::Mesh(Arc::new(TriMesh::new(&positions, &indices)));
    let (t, n) = mesh.ray(Vec3::new(0.3, 0.2, 2.0), -Vec3::Z, 100.0).unwrap();
    assert!((t - 2.0).abs() < 1.0e-4 && n.abs_diff_eq(Vec3::Z, 1.0e-5));
}

/// A cube's surface as six `n` by `n` grids with their own vertices, as a
/// render mesh splits them along its UV seams.
fn split_cube(n: usize) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let (mut positions, mut indices) = (Vec::new(), Vec::new());
    let (flat, flat_indices) = grid(n, 2.0 / n as f32);
    for rotation in [
        Quat::IDENTITY,
        Quat::from_rotation_x(std::f32::consts::PI),
        Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
        Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
        Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
        Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2),
    ] {
        let base = positions.len() as u32;
        // Round so welding sees the shared edges as one.
        positions.extend(flat.iter().map(|&p| (rotation * (p + Vec3::Z)) * 1.0e4).map(|p| p.round() / 1.0e4));
        indices.extend(flat_indices.iter().map(|t| t.map(|i| i + base)));
    }
    (positions, indices)
}

#[test]
fn cooking_simplifies_a_seamed_render_mesh_within_tolerance() {
    let (positions, indices) = split_cube(16);
    let tolerance = 0.005;
    let cooked = TriMesh::collision(&positions, &indices, tolerance);
    let kept = cooked.triangles().len();
    assert!(kept <= 24, "kept {kept} of {}", indices.len());
    // Every input point still lies on the cooked surface.
    let collider = Collider::Mesh(Arc::new(cooked));
    for &p in &positions {
        let inward = -p.normalize();
        let hit = collider.ray(p - inward * 0.1, inward, 1.0);
        assert!(hit.is_some_and(|(t, _)| (t - 0.1).abs() <= tolerance + 1.0e-4), "{p}: {hit:?}");
    }
}

#[test]
fn cooking_a_hull_from_a_render_mesh_keeps_the_vertex_budget() {
    let (positions, _) = split_cube(8);
    let hull = ConvexHull::with_max_vertices(&positions, 16).unwrap();
    assert_eq!(hull.vertices().len(), 8, "a cube needs only its corners");
    assert_eq!(hull.face_count(), 6);
    assert!(hull.error(&positions) < 1.0e-3);
    assert!((hull.volume() - 8.0).abs() < 1.0e-3);
}
