//! Retained meshes, such as terrain chunks, reaching both tracers' scenes.
use glam::{Affine3A, Vec3};
use std::sync::Arc;
use verity_lux::cascades::scene::{GiScene, GiWords};
use verity_physics::{RayScene, RaySource, TriMesh};
use verity_render::{RetainedChange, RetainedMesh, Vertex};

/// A square of two triangles at height `z`, facing up.
fn square(x: f32, z: f32) -> RetainedMesh {
    let corner = |dx: f32, dy: f32| Vertex {
        position: [x + dx, dy, z, 0.0],
        normal: [0.0, 0.0, 1.0, 0.0],
        tangent: [0.0; 4],
        color: [1.0; 4],
    };
    RetainedMesh {
        vertices: vec![
            corner(0.0, 0.0),
            corner(1.0, 0.0),
            corner(1.0, 1.0),
            corner(0.0, 1.0),
        ],
        indices: vec![0, 1, 2, 0, 2, 3],
        bounds: (Vec3::new(x, 0.0, z), Vec3::new(x + 1.0, 1.0, z)),
    }
}

fn change(key: u64, mesh: &RetainedMesh) -> RetainedChange<'_> {
    RetainedChange {
        key,
        mesh: Some(mesh),
    }
}

/// The instances' vertex indices and materials, read back as a shader would.
fn instance_vertices(words: &GiWords) -> Vec<(u32, Vec<u32>)> {
    let [_, base, count, stride] = words.scene_words[0].to_array();
    (0..count)
        .map(|instance| {
            let at = (base + instance * stride) as usize;
            let (material, span) = (words.scene_words[at + 7].w, words.scene_words[at + 8]);
            let triangles = &words.geometry_words[1 + span.y as usize..][..span.z as usize];
            let vertices = triangles.iter().flat_map(|t| [t.x, t.y, t.z]).collect();
            (material, vertices)
        })
        .collect()
}

fn vertex_position(words: &GiWords, vertex: u32) -> Vec3 {
    let header = words.geometry_words[0];
    let word = words.geometry_words[(header.y + vertex * header.z) as usize];
    Vec3::from_array([word.x, word.y, word.z].map(f32::from_bits))
}

#[test]
fn gi_scene_follows_retained_meshes() {
    let (a, b) = (square(0.0, 0.0), square(5.0, 2.0));
    let mut scene = GiScene::default();
    scene.set_retained(&[change(1, &a), change(2, &b)], 7);
    assert!(scene.refresh().is_none(), "new meshes need a layout");
    let words = scene.words();
    assert_eq!((words.triangles, words.instances), (4, 2));

    // Each instance names the material and reaches only its own vertices.
    let instances = instance_vertices(words);
    let mut heights = Vec::new();
    for (material, vertices) in &instances {
        assert_eq!(*material, 7);
        assert_eq!(vertices.len(), 6);
        let z: Vec<f32> = vertices
            .iter()
            .map(|v| vertex_position(words, *v).z)
            .collect();
        assert!(z.iter().all(|h| *h == z[0]), "{z:?}");
        heights.push(z[0]);
    }
    heights.sort_by(f32::total_cmp);
    assert_eq!(heights, [0.0, 2.0]);

    // Removing leaves only what is current, without moving anything else.
    scene.set_retained(&[RetainedChange { key: 2, mesh: None }], 7);
    assert!(scene.refresh().is_some());
    assert_eq!((scene.words().triangles, scene.words().instances), (2, 1));
    scene.set_retained(&[RetainedChange { key: 1, mesh: None }], 7);
    assert!(scene.refresh().is_some());
    assert_eq!(scene.words().instances, 0);
    assert_eq!(scene.words().scene_words[0].z, 0);
}

#[test]
fn gi_scene_rewrites_only_what_changed() {
    let (a, b) = (square(0.0, 0.0), square(5.0, 2.0));
    let mut scene = GiScene::default();
    scene.set_retained(&[change(1, &a), change(2, &b)], 7);
    scene.refresh();
    let before = scene.words().geometry_words.clone();

    // A mesh that fits its slot changes in place, leaving the other's words alone.
    let dug = square(0.0, -4.0);
    scene.set_retained(&[change(1, &dug)], 7);
    let ranges = scene.refresh().expect("it fits its slot");
    let total = scene.words().geometry_words.len();
    assert!(ranges.geometry.iter().map(|r| r.len()).sum::<usize>() < total);
    let after = &scene.words().geometry_words;
    assert_eq!(after.len(), before.len());
    for (word, (old, new)) in before.iter().zip(after).enumerate() {
        if old != new {
            assert!(ranges.geometry.iter().any(|r| r.contains(&word)), "{word}");
        }
    }
    let words = scene.words();
    let mut heights: Vec<f32> = instance_vertices(words)
        .iter()
        .map(|(_, v)| vertex_position(words, v[0]).z)
        .collect();
    heights.sort_by(f32::total_cmp);
    assert_eq!(heights, [-4.0, 2.0]);

    // The TLAS follows: its root must now reach down to the dug mesh.
    let root = words.scene_words[1];
    assert_eq!(f32::from_bits(root.z), -4.0);

    // A mesh that outgrows its slot moves to the free tail, still in place, and
    // a new mesh joins it there.
    let quads = |n: u32, z: f32| {
        let mut big = square(0.0, z);
        for quad in 1..n {
            let more = square(quad as f32, z);
            let base = big.vertices.len() as u32;
            big.vertices.extend(more.vertices);
            big.indices.extend(more.indices.iter().map(|i| i + base));
        }
        big
    };
    let (grown, fresh) = (quads(40, -5.0), square(20.0, 1.0));
    scene.set_retained(&[change(1, &grown), change(3, &fresh)], 7);
    assert!(scene.refresh().is_some());
    let words = scene.words();
    assert_eq!((words.triangles, words.instances), (84, 3));
    let mut sizes: Vec<usize> = instance_vertices(words)
        .iter()
        .map(|(_, v)| v.len())
        .collect();
    sizes.sort_unstable();
    assert_eq!(sizes, [6, 6, 240]);
    for (_, vertices) in instance_vertices(words) {
        let z = vertex_position(words, vertices[0]).z;
        assert!(vertices.iter().all(|v| vertex_position(words, *v).z == z));
    }

    // Past the free space it takes a new layout, and keeps its neighbours.
    let huge = quads(9000, -6.0);
    scene.set_retained(&[change(1, &huge)], 7);
    assert!(scene.refresh().is_none());
    let words = scene.words();
    assert_eq!((words.triangles, words.instances), (18_004, 3));
}

fn tri_mesh(mesh: &RetainedMesh) -> Arc<TriMesh> {
    let positions: Vec<Vec3> = (mesh.vertices.iter())
        .map(|v| Vec3::from_slice(&v.position[..3]))
        .collect();
    Arc::new(TriMesh::new(&positions, mesh.indices.as_chunks::<3>().0))
}

#[test]
fn ray_scene_replaces_and_removes_statics() {
    let pose = Affine3A::IDENTITY;
    let (low, high) = (square(0.0, 0.0), square(0.0, 3.0));
    let other = square(5.0, 0.0);
    let mut scene = RayScene::default();
    scene.add_statics([(tri_mesh(&low), pose, 4), (tri_mesh(&other), pose, 9)]);
    let down = |scene: &RayScene, x: f32| {
        let hit = scene.ray(Vec3::new(x, 0.5, 10.0), -Vec3::Z, 20.0)?;
        Some((10.0 - hit.t, scene.items()[hit.item].source))
    };
    assert_eq!(down(&scene, 0.5), Some((0.0, RaySource::Static(4))));

    // Digging moves the surface; the change spans the old place and the new.
    scene.update_statics([(4, Some((tri_mesh(&high), pose)))]);
    assert_eq!(down(&scene, 0.5), Some((3.0, RaySource::Static(4))));
    assert_eq!(scene.moved().len(), 1);
    assert_eq!(scene.moved()[0].min.z, 0.0);
    assert_eq!(scene.moved()[0].max.z, 3.0);

    // A removal swaps the last item into its place, which must stay reachable.
    scene.update_statics([(4, None)]);
    assert_eq!(down(&scene, 0.5), None);
    assert_eq!(down(&scene, 5.5), Some((0.0, RaySource::Static(9))));
    scene.update_statics([(9, Some((tri_mesh(&high), pose)))]);
    assert_eq!(scene.items().len(), 1);
    scene.update_statics([(4, Some((tri_mesh(&other), pose))), (77, None)]);
    assert_eq!(down(&scene, 5.5), Some((0.0, RaySource::Static(4))));
}
