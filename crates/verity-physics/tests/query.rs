use glam::{Affine3A, Quat, Vec3};
use std::sync::Arc;
use verity_core::{Entity, EntityHandle, Transform, World};
use verity_physics::*;

/// A small deterministic generator: tests must not depend on a crate or a seed.
struct Lcg(u32);

impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 8) as f32 / (1 << 24) as f32
    }
    fn vec(&mut self, reach: f32) -> Vec3 {
        (Vec3::new(self.unit(), self.unit(), self.unit()) * 2.0 - 1.0) * reach
    }
}

fn spawn(
    world: &mut World,
    physics: &mut PhysicsWorld,
    at: Vec3,
    body: PhysicsBody,
) -> EntityHandle {
    let entity = world.spawn(Entity {
        transform: Transform {
            translation: at.to_array(),
            ..Default::default()
        },
        ..Entity::default()
    });
    physics.insert(world, entity, body).unwrap();
    entity
}

fn soup(rng: &mut Lcg, count: u32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let mut positions = Vec::new();
    let mut triangles = Vec::new();
    for i in 0..count {
        let center = rng.vec(10.0);
        positions.extend([center, center + rng.vec(1.0), center + rng.vec(1.0)]);
        triangles.push([i * 3, i * 3 + 1, i * 3 + 2]);
    }
    (positions, triangles)
}

fn brute(positions: &[Vec3], triangles: &[[u32; 3]], o: Vec3, d: Vec3, max: f32) -> Option<f32> {
    let mut best: Option<f32> = None;
    for t in triangles {
        let mesh = TriMesh::new(&t.map(|i| positions[i as usize]), &[[0, 1, 2]]);
        if let Some((hit, _)) = mesh.ray(o, d, best.unwrap_or(max)) {
            best = Some(hit);
        }
    }
    best
}

#[test]
fn mesh_tree_finds_what_testing_every_triangle_finds() {
    let mut rng = Lcg(7);
    let (positions, triangles) = soup(&mut rng, 600);
    let mesh = TriMesh::new(&positions, &triangles);
    let mut hits = 0;
    for _ in 0..800 {
        let (o, d) = (rng.vec(14.0), rng.vec(1.0).normalize());
        let found = mesh.ray(o, d, 100.0);
        assert_eq!(
            found.map(|(t, _)| t),
            brute(&positions, &triangles, o, d, 100.0)
        );
        if let Some((_, n)) = found {
            assert!(n.dot(d) <= 0.0, "the normal faces the ray");
            hits += 1;
        }
    }
    assert!(hits > 50, "the test must exercise hits, had {hits}");
}

#[test]
fn every_item_is_in_exactly_one_leaf_and_the_tree_stays_shallow() {
    let mut rng = Lcg(3);
    // Half the items share one point: nothing can split those by position.
    let bounds: Vec<_> = (0..5000)
        .map(|i| {
            let c = if i % 2 == 0 {
                Vec3::ZERO
            } else {
                rng.vec(50.0)
            };
            Bounds {
                min: c - 0.1,
                max: c + 0.1,
            }
        })
        .collect();
    let tree = Bvh::build(&bounds);
    let mut seen = tree.order().to_vec();
    seen.sort_unstable();
    assert_eq!(seen, (0..5000).collect::<Vec<u32>>());
    fn depth(nodes: &[BvhNode], at: usize) -> usize {
        let [lo, hi] = nodes[at];
        if hi[3].to_bits() != 0 {
            return 1;
        }
        1 + depth(nodes, at + 1).max(depth(nodes, lo[3].to_bits() as usize))
    }
    // The GPU walks this with a 32 entry stack.
    assert!(depth(tree.nodes(), 0) <= 32 + 12);
}

#[test]
fn colliders_answer_rays_like_their_closed_forms() {
    let sphere = Collider::Sphere { radius: 2.0 };
    let (t, n) = sphere
        .ray(Vec3::new(-5.0, 0.0, 0.0), Vec3::X, 100.0)
        .unwrap();
    assert!((t - 3.0).abs() < 1e-5 && n.abs_diff_eq(-Vec3::X, 1e-5));
    assert!(
        sphere
            .ray(Vec3::new(-5.0, 2.5, 0.0), Vec3::X, 100.0)
            .is_none()
    );
    assert!(
        sphere
            .ray(Vec3::new(-5.0, 0.0, 0.0), Vec3::X, 2.0)
            .is_none(),
        "beyond max"
    );
    assert!(
        sphere.ray(Vec3::ZERO, Vec3::X, 100.0).is_none(),
        "from inside"
    );

    let capsule = Collider::Capsule {
        radius: 0.5,
        half_height: 1.0,
    };
    let side = capsule.ray(Vec3::new(-5.0, 0.0, 0.8), Vec3::X, 100.0);
    let (t, n) = side.unwrap();
    assert!((t - 4.5).abs() < 1e-5 && n.abs_diff_eq(-Vec3::X, 1e-5));
    let (t, n) = capsule.ray(Vec3::new(0.0, 0.0, 5.0), -Vec3::Z, 100.0).unwrap();
    assert!((t - 3.5).abs() < 1e-5 && n.abs_diff_eq(Vec3::Z, 1e-5), "cap");
    let (t, _) = capsule
        .ray(Vec3::new(-5.0, 0.0, 1.3), Vec3::X, 100.0)
        .unwrap();
    assert!((t - (5.0 - 0.4)).abs() < 1e-4, "cap rim at {t}");
    assert!(
        capsule
            .ray(Vec3::new(-5.0, 0.0, 1.6), Vec3::X, 100.0)
            .is_none()
    );

    let cuboid = Collider::Cuboid {
        half_extents: Vec3::new(1.0, 2.0, 3.0),
    };
    let (t, n) = cuboid
        .ray(Vec3::new(0.0, 7.0, 0.0), -Vec3::Y, 100.0)
        .unwrap();
    assert!((t - 5.0).abs() < 1e-5 && n.abs_diff_eq(Vec3::Y, 1e-5));
    // Axis-parallel, so two of the slabs divide by zero.
    assert!(
        cuboid
            .ray(Vec3::new(1.5, 7.0, 0.0), -Vec3::Y, 100.0)
            .is_none()
    );
}

#[test]
fn a_scene_ray_carries_through_poses_and_scale() {
    let mesh = Arc::new(TriMesh::new(
        &[
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ],
        &[[0, 1, 2]],
    ));
    let pose = Affine3A::from_scale_rotation_translation(
        Vec3::splat(4.0),
        Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
        Vec3::new(0.0, 5.0, 0.0),
    );
    let mut scene = RayScene::default();
    scene.add_statics([(mesh, pose, 11)]);
    let hit = scene.ray(Vec3::ZERO, Vec3::Y, 100.0).unwrap();
    assert!((hit.t - 5.0).abs() < 1e-4, "{}", hit.t);
    assert!(hit.normal.abs_diff_eq(-Vec3::Y, 1e-4));
    assert_eq!(scene.items()[hit.item].source, RaySource::Static(11));
    // Scale widens it: an unscaled triangle would be missed here.
    assert!(
        scene
            .ray(Vec3::new(0.0, 0.0, 2.0), Vec3::Y, 100.0)
            .is_some()
    );
}

#[test]
fn the_scene_tree_matches_brute_force_after_many_refits_and_rebuilds() {
    let mut rng = Lcg(21);
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let bodies: Vec<_> = (0..200)
        .map(|i| {
            let collider = if i % 2 == 0 {
                Collider::Sphere { radius: 0.6 }
            } else {
                Collider::Cuboid {
                    half_extents: Vec3::splat(0.5),
                }
            };
            let body = PhysicsBody::new(RigidBody::Kinematic, collider);
            spawn(&mut world, &mut physics, rng.vec(15.0), body)
        })
        .collect();
    let mut scene = RayScene::default();
    assert!(scene.sync(&physics, &world).restructured);
    for _ in 0..40 {
        for &entity in &bodies {
            let found = world.get_mut(entity).unwrap();
            let at = Vec3::from_array(found.transform.translation) + rng.vec(0.8);
            found.transform.translation = at.to_array();
            found.transform.rotation = Quat::from_rotation_z(at.x).to_array();
        }
        let sync = scene.sync(&physics, &world);
        assert!(sync.changed && !sync.restructured);
        for _ in 0..30 {
            let (o, d) = (rng.vec(20.0), rng.vec(1.0).normalize());
            let every = scene.items().iter().enumerate().filter_map(|(i, item)| {
                let m = item.local_from_world;
                let RayShape::Collider(c) = &item.shape else {
                    unreachable!()
                };
                c.ray(m.transform_point3(o), m.transform_vector3(d), 100.0)
                    .map(|h| (h.0, i))
            });
            let expected = every.min_by(|a, b| a.0.total_cmp(&b.0));
            let got = scene.ray(o, d, 100.0).map(|h| (h.t, h.item));
            assert_eq!(got, expected);
        }
    }
}

#[test]
fn resting_bodies_cost_nothing_and_only_the_mover_is_reported() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let cube = |kind| {
        PhysicsBody::new(
            kind,
            Collider::Cuboid {
                half_extents: Vec3::splat(0.5),
            },
        )
    };
    let still = spawn(
        &mut world,
        &mut physics,
        Vec3::new(-20.0, 0.0, 0.0),
        cube(RigidBody::Static),
    );
    let mover = spawn(
        &mut world,
        &mut physics,
        Vec3::new(20.0, 0.0, 0.0),
        cube(RigidBody::Static),
    );
    let mut scene = RayScene::default();
    scene.sync(&physics, &world);
    assert_eq!(scene.moved().len(), 2);

    assert_eq!(scene.sync(&physics, &world), RaySync::default());
    assert!(scene.moved().is_empty());

    world.get_mut(mover).unwrap().transform.translation = [21.0, 0.0, 0.0];
    let sync = scene.sync(&physics, &world);
    assert!(sync.changed && !sync.restructured);
    let [moved] = scene.moved() else {
        panic!("{:?}", scene.moved())
    };
    assert!(
        (moved.min.x - 19.5).abs() < 1e-5 && (moved.max.x - 21.5).abs() < 1e-5,
        "old and new"
    );

    physics.remove(still);
    let sync = scene.sync(&physics, &world);
    assert!(sync.restructured);
    assert_eq!(scene.items().len(), 1);
    assert!(
        scene
            .ray(Vec3::new(-25.0, 0.0, 0.0), Vec3::X, 10.0)
            .is_none()
    );
    assert!(
        scene
            .ray(Vec3::new(25.0, 0.0, 0.0), -Vec3::X, 10.0)
            .is_some()
    );
}

#[test]
fn entities_without_bodies_trace_as_the_cubes_they_are_drawn_as() {
    let (mut world, physics) = (World::new(), PhysicsWorld::default());
    world.spawn(Entity {
        transform: Transform {
            translation: [0.0, 5.0, 0.0],
            scale: [8.0, 2.0, 8.0],
            ..Default::default()
        },
        ..Entity::default()
    });
    let hidden = world.spawn(Entity {
        transform: Transform {
            translation: [0.0, 2.0, 0.0],
            scale: [0.0; 3],
            ..Default::default()
        },
        ..Entity::default()
    });
    let mut scene = RayScene::default();
    scene.sync(&physics, &world);
    assert!(scene.items().is_empty(), "off by default");

    scene.entity_boxes = true;
    scene.sync(&physics, &world);
    let hit = scene.ray(Vec3::ZERO, Vec3::Y, 100.0).unwrap();
    assert!((hit.t - 4.0).abs() < 1e-5, "{}", hit.t);

    assert_eq!(scene.sync(&physics, &world), RaySync::default(), "resting");
    world.get_mut(hidden).unwrap().transform.scale = [1.0; 3];
    assert!(scene.sync(&physics, &world).restructured);
    assert!((scene.ray(Vec3::ZERO, Vec3::Y, 100.0).unwrap().t - 1.5).abs() < 1e-5);
    world.get_mut(hidden).unwrap().transform.scale = [0.0; 3];
    assert!(scene.sync(&physics, &world).restructured);
    assert_eq!(scene.items().len(), 1);
}

#[test]
fn a_settled_stack_stops_changing_the_scene() {
    let (mut world, mut physics) = (World::new(), PhysicsWorld::default());
    let floor = Collider::Cuboid {
        half_extents: Vec3::new(10.0, 10.0, 0.5),
    };
    spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, -0.5),
        PhysicsBody::new(RigidBody::Static, floor),
    );
    let cube = Collider::Cuboid {
        half_extents: Vec3::splat(0.5),
    };
    spawn(
        &mut world,
        &mut physics,
        Vec3::new(0.0, 0.0, 0.6),
        PhysicsBody::new(RigidBody::Dynamic, cube),
    );
    let (mut scene, settings) = (RayScene::default(), PhysicsSettings::default());
    scene.sync(&physics, &world);
    let mut last_change = 0;
    for step in 0..600 {
        physics.step(&mut world, &settings, 1.0 / 60.0);
        if scene.sync(&physics, &world).changed {
            last_change = step;
        }
    }
    assert!(last_change < 500, "still changing at step {last_change}");
}

#[test]
fn cloth_blocks_rays_where_it_has_fallen_to() {
    let (world, mut physics) = (World::new(), PhysicsWorld::default());
    let cloth = SoftBody::cloth(
        Vec3::new(0.0, 0.0, 5.0),
        Quat::IDENTITY,
        [4.0, 4.0],
        [8, 8],
        SoftMaterial::cloth(),
    );
    physics.add_soft_body(cloth);
    let mut scene = RayScene::default();
    scene.sync(&physics, &world);
    let down = |scene: &RayScene| {
        scene
            .ray(Vec3::new(0.1, 0.1, 20.0), -Vec3::Z, 100.0)
            .map(|h| h.t)
    };
    let start = down(&scene).unwrap();
    assert!((start - 15.0).abs() < 0.01, "{start}");

    let settings = PhysicsSettings::default();
    let mut world = world;
    for _ in 0..20 {
        physics.step(&mut world, &settings, 1.0 / 60.0);
    }
    let sync = scene.sync(&physics, &world);
    assert!(sync.changed && !sync.restructured);
    assert!(!scene.moved().is_empty());
    assert!(down(&scene).unwrap() > start, "it fell");
}
