//! Static scene queries compared with an independent brute-force triangle walk.

use glam::{Mat4, Quat, Vec3};
use std::ops::RangeInclusive;
use verity_lux::cascades::kd::KdTree;
use verity_render::{
    Vertex,
    asset::{Draw, Primitive, SceneData},
};

fn add_primitive(scene: &mut SceneData, triangles: impl IntoIterator<Item = [Vec3; 3]>) -> u32 {
    let base_vertex = scene.vertices.len() as u32;
    let first_index = scene.indices.len() as u32;
    for triangle in triangles {
        for point in triangle {
            scene
                .indices
                .push(scene.vertices.len() as u32 - base_vertex);
            scene.vertices.push(vertex(point));
        }
    }
    let index = scene.primitives.len() as u32;
    scene.primitives.push(Primitive {
        first_index,
        base_vertex,
        index_count: scene.indices.len() as u32 - first_index,
        material: 0,
    });
    index
}

fn vertex(p: Vec3) -> Vertex {
    Vertex {
        position: [p.x, p.y, p.z, 0.0],
        normal: [0.0; 4],
        tangent: [0.0; 4],
        color: [1.0; 4],
    }
}

fn scene() -> SceneData {
    let mut scene = SceneData {
        vertices: vec![vertex(Vec3::splat(98.0))],
        indices: vec![u32::MAX; 2],
        ..SceneData::default()
    };
    let cluster = add_primitive(
        &mut scene,
        (-16..-8).map(|x| {
            let x = x as f32;
            [
                Vec3::new(x, -1.0, 0.0),
                Vec3::new(x + 1.0, -1.0, 0.0),
                Vec3::new(x, 1.0, 0.0),
            ]
        }),
    );
    // This triangle spans both clusters. Its ordinal must not change when
    // the preceding zero-area triangle is omitted.
    let wide = add_primitive(
        &mut scene,
        [
            [Vec3::ZERO; 3],
            [
                Vec3::new(-16.0, -2.0, 0.0),
                Vec3::new(16.0, -2.0, 0.0),
                Vec3::new(0.0, 2.0, 0.0),
            ],
        ],
    );
    let elevated = Mat4::from_scale_rotation_translation(
        Vec3::new(-1.0, 2.0, 0.5),
        Quat::IDENTITY,
        Vec3::Z * 4.0,
    );
    scene.draws.extend([
        Draw {
            primitive: cluster,
            transform: Mat4::IDENTITY,
        },
        Draw {
            primitive: cluster,
            transform: Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0)),
        },
        Draw {
            primitive: wide,
            transform: elevated,
        },
        Draw {
            primitive: wide,
            transform: Mat4::IDENTITY,
        },
        Draw {
            primitive: wide,
            transform: elevated,
        },
    ]);
    scene
}

fn brute_force(
    scene: &SceneData,
    origin: Vec3,
    direction: Vec3,
    range: &RangeInclusive<f64>,
) -> Option<(f64, usize, u32, usize)> {
    let (origin, direction) = (origin.as_dvec3(), direction.as_dvec3());
    let mut best = None;
    for (instance, draw) in scene.draws.iter().enumerate() {
        let primitive = scene.primitives[draw.primitive as usize];
        let start = primitive.first_index as usize;
        let end = start + primitive.index_count as usize;
        for (triangle, indices) in scene.indices[start..end]
            .as_chunks::<3>()
            .0
            .iter()
            .enumerate()
        {
            let [a, b, c] = indices.map(|i| {
                let [x, y, z, _] = scene.vertices[(primitive.base_vertex + i) as usize].position;
                draw.transform
                    .transform_point3(Vec3::new(x, y, z))
                    .as_dvec3()
            });
            let normal = (b - a).cross(c - a);
            let denominator = direction.dot(normal);
            if denominator == 0.0 {
                continue;
            }
            let t = (a - origin).dot(normal) / denominator;
            if !range.contains(&t) {
                continue;
            }
            let point = origin + direction * t;
            let inside = [(a, b), (b, c), (c, a)]
                .into_iter()
                .all(|(from, to)| (to - from).cross(point - from).dot(normal) >= 0.0);
            let candidate = (t, instance, draw.primitive, triangle);
            if inside && best.is_none_or(|hit| candidate < hit) {
                best = Some(candidate);
            }
        }
    }
    best
}

#[test]
fn spatial_queries_match_brute_force() {
    let scene = scene();
    let tree = KdTree::from_scene(&scene).expect("the fixture is valid");
    for x in -17..=17 {
        for y in -4..=4 {
            for (start_z, direction) in [
                (6.0, Vec3::NEG_Z * 2.0),
                (-2.0, Vec3::Z * 2.0),
                (6.0, Vec3::new(2.0, 0.0, -2.0)),
                (6.0, Vec3::new(-2.0, 0.0, -2.0)),
            ] {
                let origin = Vec3::new(x as f32, y as f32, start_z);
                for range in [0.0..=4.0, 0.0..=1.0, 1.25..=3.0, 1.0..=1.0, 3.0..=3.0] {
                    let expected = brute_force(&scene, origin, direction, &range);
                    let actual = tree
                        .nearest_hit(origin, direction, range.clone())
                        .map(|hit| {
                            (
                                hit.t,
                                hit.id.instance_index,
                                hit.id.primitive_index,
                                hit.id.triangle_index,
                            )
                        });
                    assert_eq!(
                        actual, expected,
                        "origin {origin}, direction {direction}, range {range:?}"
                    );
                    assert_eq!(
                        tree.is_occluded(origin, direction, range),
                        expected.is_some()
                    );
                }
            }
        }
    }

    let origin = Vec3::new(0.0, 4.0, 6.0);
    let direction = Vec3::NEG_Z * 2.0;
    let hit = tree
        .nearest_hit(origin, direction, 1.0..=1.0)
        .expect("shared vertex at the endpoint");
    assert_eq!(
        (
            hit.t,
            hit.id.instance_index,
            hit.id.primitive_index,
            hit.id.triangle_index
        ),
        (1.0, 2, 1, 1)
    );
    assert!(!tree.is_occluded(origin, direction, 0.0..=0.999));
    assert!(!tree.is_occluded(Vec3::new(-16.0, 0.0, 4.0), Vec3::X, 0.0..=32.0));
}

#[test]
fn tiny_triangle_and_empty_scene() {
    let mut scene = SceneData::default();
    let empty = KdTree::from_scene(&scene).expect("empty scenes are valid");
    assert!(!empty.is_occluded(Vec3::Z, Vec3::NEG_Z, 0.0..=1.0));

    let side = 2.0_f32.powi(-80);
    let primitive = add_primitive(&mut scene, [[Vec3::ZERO, Vec3::X * side, Vec3::Y * side]]);
    scene.draws.push(Draw {
        primitive,
        transform: Mat4::IDENTITY,
    });
    let tree = KdTree::from_scene(&scene).expect("small finite triangles are valid");
    let origin = Vec3::new(side * 0.25, side * 0.25, 1.0);
    assert!(tree.is_occluded(origin, Vec3::NEG_Z, 0.0..=1.0));
    assert!(!tree.is_occluded(origin, Vec3::NEG_Z, 0.0..=f64::INFINITY));
    assert!(!tree.is_occluded(origin, Vec3::ZERO, 0.0..=1.0));
}

#[test]
fn invalid_index_reports_its_draw() {
    let mut scene = scene();
    scene.draws[2].primitive = u32::MAX;
    let err = KdTree::from_scene(&scene).expect_err("invalid draw primitive");
    assert_eq!(err.instance_index(), 2);
}
