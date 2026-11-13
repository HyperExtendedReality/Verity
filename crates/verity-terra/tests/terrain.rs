use fast_surface_nets::ndshape::ConstShape;
use glam::{IVec3, Vec3};
use std::collections::HashMap;
use verity_terra::*;

fn flat() -> Landscape {
    Landscape::flat(0.3, 385, 1.0)
}

/// A preset on a 1 km map: tests build this crate unoptimised.
fn small(landform: Landform) -> Settings {
    Settings {
        size: 257,
        ..landform.into()
    }
}

#[test]
fn terrain_generation_excavation_and_chunk_seams() {
    let settings = Settings {
        droplets: 500,
        thermal_steps: 3,
        ..small(Landform::Valley)
    };
    let a = Landscape::generate(settings.clone());
    let b = Landscape::generate(settings.clone());
    assert_eq!(a.heights, b.heights, "seed must reproduce the landscape");
    assert!(a.heights.iter().all(|h| h.is_finite()));
    assert!(a.flow.iter().any(|f| *f > 0.0));
    let raw = Landscape::generate(Settings {
        droplets: 0,
        thermal_steps: 0,
        ..settings.clone()
    });
    let mass_before: f64 = raw.heights.iter().map(|h| *h as f64).sum();
    let mass_after: f64 = a.heights.iter().map(|h| *h as f64).sum();
    assert!(
        (mass_before - mass_after).abs() < 0.5,
        "erosion/deposition must conserve map sediment"
    );

    let mut terrain = Terrain::new(flat());
    let hit = terrain
        .raycast(Vec3::new(0.0, 0.0, 12.0), -Vec3::Z, 30.0)
        .expect("ground hit");
    assert!(hit.z.abs() < 1.0);
    assert!(
        terrain
            .raycast(Vec3::new(0.0, 0.0, 12.0), Vec3::Z, 10.0)
            .is_none()
    );
    assert!(terrain.raycast(Vec3::ZERO, Vec3::ZERO, 10.0).is_none());
    assert!(terrain.brush(Vec3::NAN, 3.0, false).is_empty());
    assert!(terrain.brush(Vec3::ZERO, f32::INFINITY, false).is_empty());

    // Face, edge and corner overlap of signed-coordinate chunks.
    let center = Vec3::new(0.0, 0.0, -3.0);
    let dirty = terrain.brush(center, 5.0, false);
    assert!(dirty.len() >= 4);
    assert!(
        terrain.sample(center) > 4.0,
        "a cavity must be genuinely empty in 3D"
    );
    assert!(
        terrain.sample(center + Vec3::Z * (-8.0)) < 0.0,
        "ground beyond brush must remain"
    );
    let snapshot = terrain.chunks[&IVec3::new(0, 0, -1)].samples.clone();
    let revision = terrain.chunks[&IVec3::new(0, 0, -1)].revision;
    terrain.brush(center, 5.0, true);
    assert!(
        terrain.sample(center) < 0.0,
        "filling must restore solid density"
    );
    assert!(terrain.chunks[&IVec3::new(0, 0, -1)].revision > revision);
    assert_ne!(
        *snapshot,
        *terrain.chunks[&IVec3::new(0, 0, -1)].samples,
        "worker snapshot must be immutable"
    );
    terrain.brush(center, 5.0, false);
    for (&coord, chunk) in &terrain.chunks {
        for i in 0..Shape::SIZE {
            let local = IVec3::from_array(Shape::delinearize(i).map(|v| v as i32));
            let global = coord * CELLS - IVec3::ONE + local;
            assert_eq!(
                chunk.samples[i as usize],
                terrain.lattice(global),
                "halo mismatch at {global}"
            );
        }
    }

    // Weld the eight chunks of a sphere. Every undirected edge must have two incident triangles.
    let mut edges = HashMap::new();
    for z in -1..=0 {
        for y in -1..=0 {
            for x in -1..=0 {
                let coord = IVec3::new(x, y, z);
                let origin = chunk_origin(coord);
                let samples: Vec<f32> = (0..Shape::SIZE)
                    .map(|i| {
                        let p = origin + Vec3::from_array(Shape::delinearize(i).map(|v| v as f32));
                        p.length() - 5.3
                    })
                    .collect();
                let mesh = polygonize(coord, &samples, &flat());
                assert!(
                    mesh.normals
                        .iter()
                        .all(|n| Vec3::from_array(*n).is_finite())
                );
                for face in mesh.indices.as_chunks::<3>().0 {
                    for pair in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
                        let key = |index: u32| {
                            mesh.positions[index as usize].map(|f| (f * 10000.0).round() as i32)
                        };
                        let (a, b) = (key(pair.0), key(pair.1));
                        *edges
                            .entry(if a < b { (a, b) } else { (b, a) })
                            .or_insert(0) += 1;
                    }
                }
            }
        }
    }
    assert!(!edges.is_empty());
    assert!(
        edges.values().all(|n| *n == 2),
        "chunked sphere must be watertight without duplicate faces"
    );
}

#[test]
fn landform_presets() {
    for landform in [
        Landform::Island,
        Landform::Archipelago,
        Landform::Continent,
        Landform::Mountains,
        Landform::Plains,
    ] {
        let settings = Settings {
            droplets: 300,
            thermal_steps: 2,
            ..small(landform)
        };
        let land = Landscape::generate(settings.clone());
        let base = settings.sea_level.unwrap_or(0.0);
        assert!(
            (land.heights.iter()).all(|h| h.is_finite() && *h - base < settings.mountains + 1.0),
            "{landform:?} peaks stay under the mountains setting"
        );
        let Some(sea) = settings.sea_level else {
            continue;
        };
        let dry =
            land.heights.iter().filter(|h| **h > sea).count() as f32 / land.heights.len() as f32;
        assert!(
            (0.05..0.7).contains(&dry),
            "{landform:?} land fraction {dry}"
        );
        if landform == Landform::Island {
            let corner = -land.half() * 0.96;
            assert!(land.height(0.0, 0.0) > sea, "island centre is land");
            assert!(land.height(corner, corner) < sea, "island corners are sea");
        }
    }
}

/// Mean land slope in degrees, and the steepest rise over run to any of the eight neighbours.
fn slopes(land: &Landscape, base: f32) -> (f32, f32) {
    let side = land.side;
    let (mut sum, mut count, mut steepest) = (0.0, 0, 0.0f32);
    for y in 1..side - 1 {
        for x in 1..side - 1 {
            let i = y * side + x;
            for (dx, dy) in [(1, 0), (0, 1), (1, 1), (1, -1)] {
                let j = (y as isize + dy) as usize * side + (x as isize + dx) as usize;
                let run = land.cell * if dx != 0 && dy != 0 { 2f32.sqrt() } else { 1.0 };
                steepest = steepest.max((land.heights[i] - land.heights[j]).abs() / run);
            }
            if land.heights[i] > base + 1.0 {
                let dx = (land.heights[i + 1] - land.heights[i - 1]) / (2.0 * land.cell);
                let dy = (land.heights[i + side] - land.heights[i - side]) / (2.0 * land.cell);
                sum += (dx * dx + dy * dy).sqrt().atan().to_degrees();
                count += 1;
            }
        }
    }
    (sum / count.max(1) as f32, steepest)
}

#[test]
fn erosion_drains_the_land_and_holds_slopes_under_talus() {
    let settings = Settings {
        droplets: 0,
        thermal_steps: 0,
        size: 257,
        cell: 8.0,
        ..Landform::Mountains.into()
    };
    let eroded = Landscape::generate(settings.clone());
    assert!(eroded.heights.iter().all(|h| h.is_finite()));
    let (mean, steepest) = slopes(&eroded, 0.0);
    // Rock stands four times steeper than soil, and no steeper.
    assert!(steepest <= settings.talus * 4.0 + 1.0e-3, "{steepest}");
    assert!(
        (15.0..35.0).contains(&mean),
        "mountain slopes average {mean}°"
    );
    // Lakes fill, so water runs off every cell: none is lower than all its neighbours.
    let side = eroded.side;
    let pits = (0..side * side)
        .filter(|&i| {
            let (x, y) = (i % side, i / side);
            x > 0 && y > 0 && x < side - 1 && y < side - 1 && {
                let h = eroded.heights[i];
                [
                    i - 1,
                    i + 1,
                    i - side,
                    i + side,
                    i - side - 1,
                    i - side + 1,
                    i + side - 1,
                    i + side + 1,
                ]
                .iter()
                .all(|j| eroded.heights[*j] > h)
            }
        })
        .count();
    assert_eq!(pits, 0, "every cell drains");
    assert!(
        eroded.flow.iter().any(|f| *f > 0.5),
        "drainage feeds wetness"
    );
    // Age: unworn uplift is a smooth block; rivers cut it into valleys and ridges.
    let young = Landscape::generate(Settings {
        fluvial_steps: 0,
        ..settings
    });
    let (smooth, _) = slopes(&young, 0.0);
    assert!(mean > smooth + 5.0, "dissected {mean}° vs unworn {smooth}°");
}

#[test]
fn plains_stay_gentle() {
    let plains = Landscape::generate(Settings {
        droplets: 0,
        thermal_steps: 0,
        size: 257,
        cell: 8.0,
        ..Landform::Plains.into()
    });
    let (mean, _) = slopes(&plains, 0.0);
    assert!(mean < 8.0, "plains average {mean}°");
}

#[test]
fn triangles_face_out_of_the_ground() {
    let mut terrain = Terrain::new(flat());
    let coord = IVec3::new(0, 0, 0);
    terrain.ensure_chunk(coord);
    let mesh = polygonize(coord, &terrain.chunks[&coord].samples, &terrain.landscape);
    assert!(!mesh.indices.is_empty());
    // Surface noise folds the odd sliver; back-face culling needs the rest CCW from above.
    let faces = mesh.indices.as_chunks::<3>().0;
    let total = faces.len();
    let outward = faces
        .iter()
        .filter(|face| {
            let [a, b, c] = [0, 1, 2].map(|k| Vec3::from_array(mesh.positions[face[k] as usize]));
            (b - a).cross(c - a).z > 0.0
        })
        .count();
    assert!(outward as f32 > total as f32 * 0.98, "{outward} of {total}");
    for (p, n) in mesh.positions.iter().zip(&mesh.normals) {
        let (p, n) = (Vec3::from_array(*p), Vec3::from_array(*n));
        assert!((n.length() - 1.0).abs() < 1.0e-4);
        assert!(p.cmpge(mesh.bounds.0).all() && p.cmple(mesh.bounds.1).all());
    }
}

#[test]
fn trilinear_sampling_matches_the_lattice() {
    let mut terrain = Terrain::new(flat());
    terrain.ensure_chunk(IVec3::new(-1, 0, -1));
    terrain.brush(Vec3::new(-2.0, 3.0, -1.0), 3.0, false);
    for p in [[-3, 4, -2], [-1, 0, -1], [-24, 23, -24]] {
        let p = IVec3::from_array(p);
        assert_eq!(terrain.sample(p.as_vec3()), terrain.lattice(p));
    }
    // The middle of a cell is the mean of its corners, across a chunk boundary too.
    let lo = IVec3::new(-1, 23, -1);
    let mut mean = 0.0;
    for i in 0..8 {
        mean += terrain.lattice(lo + IVec3::new(i & 1, i >> 1 & 1, i >> 2)) / 8.0;
    }
    assert!((terrain.sample(lo.as_vec3() + 0.5) - mean).abs() < 1.0e-5);
}

#[test]
fn raycast_finds_sub_step_grazing_surfaces() {
    // A small bump barely proud of the line the ray skims.
    let mut terrain = Terrain::new(flat());
    let coords = terrain.initial_coords();
    terrain.ensure_chunks(&coords);
    terrain.brush(Vec3::new(10.0, 0.0, 3.05), 1.0, true);
    let origin = Vec3::new(-20.0, 0.0, 3.9);
    let hit = terrain
        .raycast(origin, Vec3::X, 60.0)
        .expect("a ray skimming the bump must hit it");
    assert!((hit.x - 10.0).abs() < 2.0, "hit at {hit}");
    assert!(terrain.sample(hit).abs() < 0.01);
    // From inside the ground, axis aligned, and starting exactly on a cell corner.
    let up = terrain.raycast(Vec3::new(1.0, 1.0, -5.0), Vec3::Z, 20.0);
    assert!(up.is_some_and(|hit| hit.z.abs() < 1.0));
    let down = terrain.raycast(Vec3::new(0.5, 0.5, 8.0), -Vec3::Z, 20.0);
    assert!(down.is_some_and(|hit| hit.z.abs() < 1.0));
    assert!(terrain.raycast(Vec3::NAN, Vec3::Z, 20.0).is_none());
}

#[test]
fn parallel_chunks_match_serial_ones() {
    let mut serial = Terrain::new(flat());
    let mut parallel = Terrain::new(flat());
    let coords = [IVec3::new(0, 0, 0), IVec3::new(-1, 2, -1)];
    coords.iter().for_each(|c| serial.ensure_chunk(*c));
    parallel.ensure_chunks(&coords);
    for coord in coords {
        assert_eq!(
            serial.chunks[&coord].samples,
            parallel.chunks[&coord].samples
        );
    }
}

#[test]
fn mesher_replaces_results_an_edit_overtook() {
    let mut terrain = Terrain::new(flat());
    let mut mesher = Mesher::default();
    let center = Vec3::new(4.0, 4.0, 0.0);
    let coord = IVec3::new(0, 0, 0);
    mesher.queue([coord]);
    assert!(mesher.update(&mut terrain, center, 8).is_empty());
    // The job is in flight on the old field; this edit must win.
    let dirty = terrain.brush(center, 3.0, false);
    assert!(dirty.contains(&coord));
    mesher.queue(dirty);
    let mut latest = HashMap::new();
    while mesher.outstanding() > 0 {
        latest.extend(mesher.update(&mut terrain, center, 8));
        std::thread::yield_now();
    }
    assert!(mesher.rejected >= 1);
    let current = polygonize(coord, &terrain.chunks[&coord].samples, &terrain.landscape);
    assert_eq!(latest[&coord].positions, current.positions);
}

#[test]
fn generation_does_not_depend_on_thread_count() {
    let settings = Settings {
        droplets: 500,
        fluvial_steps: 8,
        thermal_steps: 4,
        ..small(Landform::Island)
    };
    let serial = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("a one-thread pool")
        .install(|| Landscape::generate(settings.clone()));
    let parallel = Landscape::generate(settings);
    assert_eq!(serial.heights, parallel.heights);
    assert_eq!(serial.flow, parallel.flow);
}

#[test]
fn edits_save_load_and_revert() {
    let mut terrain = Terrain::new(flat());
    let generated = terrain.sample(Vec3::new(4.0, 4.0, 0.0));
    let center = Vec3::new(4.0, 4.0, 0.0);
    terrain.brush(center, 3.0, false);
    let dug = terrain.sample(center);
    assert!(dug > generated);
    let mut saved = Vec::new();
    terrain.save_edits(&mut saved).unwrap();

    // A later edit elsewhere reverts on load, and the saved one comes back.
    let other = Vec3::new(-60.0, 50.0, 0.0);
    let untouched = terrain.sample(other);
    terrain.brush(other, 3.0, false);
    assert_ne!(terrain.sample(other), untouched);
    let mut fresh = Terrain::new(flat());
    fresh.brush(center, 5.0, true);
    for target in [&mut terrain, &mut fresh] {
        let changed = target.load_edits(saved.as_slice()).unwrap();
        assert!(!changed.is_empty());
        assert_eq!(target.sample(center), dug);
        assert_eq!(target.sample(other), untouched);
        let mut again = Vec::new();
        target.save_edits(&mut again).unwrap();
        assert_eq!(again, saved, "a loaded save writes back the same bytes");
    }

    // Bad files leave the terrain alone.
    let mut other_landscape = flat();
    other_landscape.heights[0] = 1.0;
    let mut wrong = Terrain::new(other_landscape);
    assert!(wrong.load_edits(saved.as_slice()).is_err());
    assert!(terrain.load_edits(&saved[..saved.len() - 1]).is_err());
    assert!(terrain.load_edits(&b"nope"[..]).is_err());
    assert_eq!(terrain.sample(center), dug);
}

#[test]
fn streaming_follows_the_eye_and_keeps_edits() {
    let mut terrain = Terrain::new(flat());
    let mut mesher = Mesher::default();
    let mut streamer = Streamer::new(&terrain, 40.0);
    let west = Vec3::new(-150.0, 0.0, 10.0);
    assert!(streamer.update(&mut terrain, &mut mesher, west).is_empty());
    let near = streamer.resident();
    assert!(near > 0 && near < terrain.initial_coords().len());
    let mut meshed = HashMap::new();
    while mesher.outstanding() > 0 {
        meshed.extend(mesher.update(&mut terrain, west, 64));
        std::thread::yield_now();
    }
    assert_eq!(meshed.len(), near);
    let dug = Vec3::new(-150.0, 0.0, 0.0);
    terrain.brush(dug, 3.0, false);
    let value = terrain.sample(dug);

    let east = Vec3::new(150.0, 0.0, 10.0);
    let unloaded = streamer.update(&mut terrain, &mut mesher, east);
    assert_eq!(unloaded.len(), near);
    assert!(unloaded.iter().all(|c| !streamer.is_resident(*c)));
    assert_eq!(terrain.sample(dug), value, "edits survive leaving");
    assert!(
        (terrain.chunks.iter()).all(|(c, chunk)| streamer.is_resident(*c) || chunk.revision > 0),
        "unedited chunks out of reach are dropped"
    );
}

#[test]
fn filters_apply_in_order_and_random_recipes_reproduce() {
    let plain = Settings {
        droplets: 0,
        fluvial_steps: 0,
        thermal_steps: 0,
        ..small(Landform::Mountains)
    };
    let raw = Landscape::generate(plain.clone());
    let terraced = Landscape::generate(Settings {
        filters: vec![Filter::Terraces {
            step: 10.0,
            sharpness: 0.98,
        }],
        ..plain.clone()
    });
    // Sharp terraces leave most ground within a whisker of a 10 m bench.
    let on_bench = (terraced.heights.iter())
        .filter(|h| (**h / 10.0 - (**h / 10.0).round()).abs() < 0.05)
        .count();
    assert!(on_bench * 10 > terraced.heights.len() * 9, "{on_bench}");
    let flattened = Landscape::generate(Settings {
        filters: vec![Filter::Plateau {
            height: 40.0,
            strength: 1.0,
        }],
        ..plain.clone()
    });
    let peak = |l: &Landscape| l.heights.iter().copied().fold(f32::MIN, f32::max);
    assert!(peak(&flattened) < peak(&raw));
    let filled = Landscape::generate(Settings {
        filters: vec![Filter::Sediment { strength: 1.0 }],
        ..plain.clone()
    });
    assert!(raw.heights.iter().zip(&filled.heights).all(|(a, b)| b >= a));
    let a = Settings::random(42);
    assert_eq!(a, Settings::random(42));
    assert_ne!(a, Settings::random(43));
    // Scenes store recipes as JSON.
    let generator = Generator::Landscape(a);
    let json = serde_json::to_string(&generator).unwrap();
    assert_eq!(serde_json::from_str::<Generator>(&json).unwrap(), generator);
}
