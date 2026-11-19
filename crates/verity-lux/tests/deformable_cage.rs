use glam::{DVec3, Vec3};
use verity_lux::cage::{CageAsset, CagePose, CageTriangle, InversionPolicy};

const REST: [Vec3; 5] = [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z, Vec3::NEG_Z];
// The cells share a face but deliberately have opposite index winding.
const CELLS: [[u32; 4]; 2] = [[0, 1, 2, 3], [0, 1, 2, 4]];
const TRIANGLE: CageTriangle = CageTriangle {
    primitive_id: 73,
    vertices: [
        Vec3::new(0.125, 0.125, -0.5),
        Vec3::new(0.125, 0.125, 0.5),
        Vec3::new(0.5, 0.125, 0.25),
    ],
};
const UPPER: Vec3 = Vec3::new(0.125, 0.5, 0.375);
const LOWER: Vec3 = Vec3::new(0.75, 0.125, 0.125);
const SEAM: Vec3 = Vec3::new(0.4375, 0.3125, 0.25);

fn rest_point(weights: Vec3) -> Vec3 {
    let [a, b, c] = TRIANGLE.vertices;
    a * weights.x + b * weights.y + c * weights.z
}

fn check_hit(pose: &CagePose, weights: Vec3, positions: &[Vec3; 5], expected_cell: usize) {
    let rest = rest_point(weights);
    let apex = if rest.z < 0.0 { 4 } else { 3 };
    let point = rest + rest.z.abs() * (positions[apex] - REST[apex]);
    let origin = point - Vec3::Y * 3.0;
    let direction = Vec3::Y * 2.0;
    let hit = pose.nearest_hit(origin, direction, 1.0..=2.0).expect("clipped triangle hit");
    assert_eq!(hit.primitive_id, TRIANGLE.primitive_id);
    assert_eq!(hit.cell_index, expected_cell);
    assert!((hit.t - 1.5).abs() < 1e-12, "incorrect world-ray t: {hit:?}");
    assert!(hit.barycentrics.abs_diff_eq(weights.as_dvec3(), 1e-12), "incorrect source weights: {hit:?}");
    assert!(pose.is_occluded(origin, direction, 1.0..=2.0));
    assert!(!pose.is_occluded(origin, direction, 0.0..=1.49));
}

#[test]
fn clips_one_source_triangle_across_two_independently_deformed_cells() {
    let asset = CageAsset::build(&REST, &CELLS, [TRIANGLE]).expect("valid two-cell cage");
    let rest = asset.pose(&REST, InversionPolicy::Reject).expect("original pose");
    let mut posed = asset.pose(&REST, InversionPolicy::Reject).expect("independent pose");
    let mut deformed = REST;
    deformed[3] = Vec3::new(0.5, 0.25, 2.0);
    deformed[4] = Vec3::new(0.25, 0.5, -1.5);
    posed.update(&deformed).expect("first deformation");

    check_hit(&rest, UPPER, &REST, 0);
    check_hit(&rest, LOWER, &REST, 1);
    check_hit(&rest, SEAM, &REST, 0);
    check_hit(&posed, UPPER, &deformed, 0);
    check_hit(&posed, LOWER, &deformed, 1);

    // Change the entire independent instance again, including translation.
    let mut second = REST.map(|point| point + Vec3::new(1.0, 2.0, 4.0));
    second[3].z += 1.0;
    posed.update(&second).expect("second deformation");
    for (cell, weights) in [UPPER, LOWER].into_iter().enumerate() {
        let point = rest_point(weights) + Vec3::new(1.0, 2.0, 4.0 + rest_point(weights).z.max(0.0));
        let hit = posed.nearest_hit(point - Vec3::Y, Vec3::Y, 1.0..=1.0).expect("translated hit");
        assert_eq!(hit.cell_index, cell);
        assert_eq!(hit.t, 1.0);
        assert!(hit.barycentrics.abs_diff_eq(weights.as_dvec3(), 1e-12));
    }
    check_hit(&rest, LOWER, &REST, 1);
}

#[test]
fn rejects_a_bad_pose_atomically_and_can_opt_in_to_inversion() {
    let asset = CageAsset::build(&REST, &CELLS, [TRIANGLE]).expect("valid cage");
    let mut pose = asset.pose(&REST, InversionPolicy::Reject).expect("original pose");
    let mut bad = REST;
    bad[3] *= 2.0;
    bad[4] = Vec3::new(0.25, 0.25, 1.0);
    assert!(asset.pose(&bad, InversionPolicy::Reject).is_err());
    assert!(pose.update(&bad).is_err());
    // The first cell passed validation in the failed update, but neither changed.
    check_hit(&pose, UPPER, &REST, 0);
    check_hit(&pose, LOWER, &REST, 1);

    let inverted = asset.pose(&bad, InversionPolicy::Allow).expect("explicitly allowed inversion");
    let rest = rest_point(LOWER);
    let point = rest + rest.z.abs() * (bad[4] - REST[4]);
    let hit = inverted.nearest_hit(point + Vec3::Y, Vec3::NEG_Y, 0.99..=1.01).expect("inverted fragment hit");
    assert_eq!(hit.cell_index, 1);
    assert!(hit.barycentrics.abs_diff_eq(LOWER.as_dvec3(), 1e-12));

    bad[4].z = 0.0;
    assert!(asset.pose(&bad, InversionPolicy::Allow).is_err());
    assert!(pose.update(&REST[..4]).is_err());
    bad[4].z = f32::NAN;
    assert!(pose.update(&bad).is_err());
    check_hit(&pose, LOWER, &REST, 1);
}

#[test]
fn validates_rest_cell_indices_and_skips_geometry_outside_the_cage() {
    assert!(CageAsset::build(&REST, &[], []).is_err());
    assert!(CageAsset::build(&REST, &[[0, 1, 2, 5]], []).is_err());
    assert!(CageAsset::build(&REST, &[[0, 1, 2, 2]], []).is_err());
    let mut non_finite = TRIANGLE;
    non_finite.vertices[0].y = f32::INFINITY;
    assert!(CageAsset::build(&REST, &CELLS, [non_finite]).is_err());

    let protruding = CageTriangle {
        primitive_id: 1,
        vertices: [
            Vec3::new(0.125, 0.125, 0.125),
            Vec3::new(1.0, 0.125, 0.125),
            Vec3::new(0.125, 0.125, 1.0),
        ],
    };
    let asset = CageAsset::build(&REST, &CELLS, [protruding]).expect("partially covered triangle");
    let pose = asset.pose(&REST, InversionPolicy::Reject).expect("rest pose");
    assert!(pose.is_occluded(Vec3::new(0.25, 0.0, 0.25), Vec3::Y, 0.0..=1.0));
    // This point is in the source triangle and the cell AABB, but not the cell.
    assert!(!pose.is_occluded(Vec3::new(0.5, 0.0, 0.5), Vec3::Y, 0.0..=1.0));
}

#[test]
fn nearest_and_any_hit_search_the_mini_blas_with_closed_range_limits() {
    let triangles = (0..24).map(|i| {
        let y = (i + 1) as f32 / 64.0;
        CageTriangle {
            primitive_id: 100 - i,
            vertices: [Vec3::new(0.125, y, 0.125), Vec3::new(0.25, y, 0.125), Vec3::new(0.125, y, 0.25)],
        }
    });
    let duplicate = CageTriangle {
        primitive_id: 1,
        vertices: [
            Vec3::new(0.125, 1.0 / 64.0, 0.125),
            Vec3::new(0.25, 1.0 / 64.0, 0.125),
            Vec3::new(0.125, 1.0 / 64.0, 0.25),
        ],
    };
    let asset = CageAsset::build(&REST, &CELLS[..1], triangles.chain([duplicate])).expect("valid cage");
    let pose = asset.pose(&REST, InversionPolicy::Reject).expect("rest pose");
    let origin = Vec3::new(0.15625, -1.0, 0.15625);
    let direction = Vec3::Y * 2.0;
    let nearest = pose.nearest_hit(origin, direction, 0.0..=1.0).expect("nearest cell fragment");
    assert_eq!(nearest.t, 65.0 / 128.0);
    assert_eq!(nearest.primitive_id, 1);
    assert_eq!(nearest.barycentrics, DVec3::new(0.5, 0.25, 0.25));
    assert!(pose.is_occluded(origin, direction, 0.6875..=0.6875));
    assert!(!pose.is_occluded(origin, direction, 0.75..=1.0));
    assert!(pose.nearest_hit(origin, Vec3::ZERO, 0.0..=1.0).is_none());
}
