use glam::{DVec3, Vec3};
use std::collections::HashSet;
use verity_terra::planet::*;
use verity_terra::*;

fn small() -> Planet {
    // A coarse grid keeps the unoptimised test build quick.
    Planet::new(PlanetSettings {
        resolution: 32,
        ..PlanetSettings::default()
    })
}

/// Patches on show never overlap, and once settled cover all six faces exactly.
fn check(shown: &HashSet<Patch>, settled: bool) {
    for a in shown {
        for b in shown {
            assert!(a == b || !a.contains(*b), "{a:?} overlaps {b:?}");
        }
    }
    let area: f64 = shown.iter().map(|p| 0.25f64.powi(p.level as i32)).sum();
    assert!(area <= 6.0 + 1e-9);
    if settled {
        assert!((area - 6.0).abs() < 1e-9, "covers {area} faces");
    }
}

/// Update until settled, applying each swap to `shown` and checking it as it goes.
fn settle(lod: &mut PlanetLod, shown: &mut HashSet<Patch>, eye: Vec3) -> usize {
    let mut swaps = 0;
    for _ in 0..10_000 {
        let (update, settled) = lod.update(eye);
        swaps += update.added.len() + update.removed.len();
        for patch in update.removed {
            assert!(shown.remove(&patch));
        }
        for (patch, mesh) in update.added {
            assert!(!mesh.indices.is_empty());
            assert!(shown.insert(patch));
        }
        check(shown, settled);
        if settled {
            assert_eq!(shown.len(), lod.shown());
            return swaps;
        }
        std::thread::yield_now();
    }
    panic!("never settled");
}

#[test]
fn lod_swaps_groups_without_holes_or_overlaps() {
    let mut lod = PlanetLod::new(small());
    let mut shown = HashSet::new();
    settle(&mut lod, &mut shown, Vec3::new(0.0, 0.0, 3_000.0));
    let far = shown.len();
    assert!(settle(&mut lod, &mut shown, Vec3::new(0.0, 0.0, 5.0)) > 0);
    assert!(shown.len() > far, "closer means finer");
    // Back out again merges the fine patches into coarse ones.
    settle(&mut lod, &mut shown, Vec3::new(0.0, 0.0, 3_000.0));
    assert_eq!(shown.len(), far);
}

#[test]
fn the_far_side_stays_coarse() {
    let planet = small();
    let near = Vec3::new(0.0, 0.0, 5.0);
    let leaves = planet.leaves(near);
    // The antipode lies far beyond the horizon: its patches never split past the first levels.
    let below = (leaves.iter())
        .filter(|p| p.face == 5 || p.face == 4)
        .map(|p| p.level)
        .collect::<Vec<_>>();
    let deepest = |face| {
        leaves
            .iter()
            .filter(|p| p.face == face)
            .map(|p| p.level)
            .max()
            .unwrap()
    };
    assert!(!below.is_empty());
    assert!(
        deepest(5) <= 1,
        "the hidden face split to level {}",
        deepest(5)
    );
}

#[test]
fn planets_erode_into_drained_ranges() {
    let planet = small();
    let s = &planet.settings;
    let mut highest = f64::MIN;
    let mut land = 0;
    for i in 0..2_000 {
        // A spiral of directions over the sphere.
        let t = i as f64 / 2_000.0;
        let (z, a) = (1.0 - 2.0 * t, i as f64 * 2.399_963);
        let r = (1.0 - z * z).sqrt();
        let h = planet.height(DVec3::new(r * a.cos(), r * a.sin(), z));
        assert!(h.is_finite());
        highest = highest.max(h);
        land += (h > 0.0) as usize;
    }
    assert!(land > 100 && land < 1_900, "{land} of 2000 on land");
    assert!(
        highest < s.mountains as f64 * 1.2,
        "peaks reach {highest} m"
    );
    assert!(
        highest > s.mountains as f64 * 0.3,
        "peaks reach {highest} m"
    );
}

#[test]
fn landscape_far_patches_cover_the_map() {
    let land = std::sync::Arc::new(Landscape::flat(2.0, 257, 4.0));
    let mut lod = PatchLod::new(Far::new(land));
    for _ in 0..10_000 {
        let (_, settled) = lod.update(Vec3::new(0.0, 0.0, 10.0));
        if settled {
            break;
        }
        std::thread::yield_now();
    }
    let leaves = lod.surface.leaves(Vec3::new(0.0, 0.0, 10.0));
    let area: f64 = leaves.iter().map(|p| 0.25f64.powi(p.level as i32)).sum();
    assert!((area - 1.0).abs() < 1e-9);
    assert_eq!(lod.shown(), leaves.len());
    let mesh = lod.surface.mesh(leaves[0]);
    // Below the ground, so the voxel chunks drawn over it win.
    assert!(mesh.positions[0][2] < 2.0);
}
