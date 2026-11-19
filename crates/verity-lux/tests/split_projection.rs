//! CPU mirror of Split RC's 6×6 octahedral projection from 8×4 equal-area bins.

fn normal(lane: usize) -> [f32; 3] {
    let x = (lane % 6) as f32;
    let y = (lane / 6) as f32;
    let x = (x + 0.5) / 6.0 * 2.0 - 1.0;
    let y = (y + 0.5) / 6.0 * 2.0 - 1.0;
    let z = 1.0 - x.abs() - y.abs();
    let t = (-z).clamp(0.0, 1.0);
    let x = x - t.copysign(x);
    let y = y - t.copysign(y);
    let length = (x * x + y * y + z * z).sqrt();
    [x / length, y / length, z / length]
}

fn bin_direction(bin: usize) -> [f32; 3] {
    let phi = ((bin % 8) as f32 + 0.5) / 8.0 * std::f32::consts::TAU;
    let y = ((bin / 8) as f32 + 0.5) / 4.0 * 2.0 - 1.0;
    let r = (1.0 - y * y).sqrt();
    [r * phi.cos(), y, r * phi.sin()]
}

fn project(cosines: &[f32; 32], samples: &[(f32, f32); 32]) -> (f32, f32) {
    let nominal: f32 = cosines.iter().sum();
    let mut sum = 0.0;
    let mut observed = 0.0;
    for (&cosine, &(radiance, support)) in cosines.iter().zip(samples) {
        let weight = cosine * support.clamp(0.0, 1.0);
        sum += radiance * weight;
        observed += weight;
    }
    if nominal > 0.0 && observed > 0.0 {
        (sum / observed, (observed / nominal).clamp(0.0, 1.0))
    } else {
        (0.0, 0.0)
    }
}

#[test]
fn split_projection_preserves_dc_and_cosine_coverage() {
    // Analytic black is unsupported if the nominal kernel has no mass.
    assert_eq!(project(&[0.0; 32], &[(0.0, 1.0); 32]), (0.0, 0.0));

    for lane in 0..36 {
        let [x, y, z] = normal(lane);
        let cosines = std::array::from_fn(|bin| {
            let [dx, dy, dz] = bin_direction(bin);
            (x * dx + y * dy + z * dz).max(0.0)
        });
        let nominal = cosines.iter().sum::<f32>();
        assert_eq!(project(&cosines, &[(1.0, 1.0); 32]), (1.0, 1.0));
        // Analytic black is fully supported for inactive sources, and
        // unsupported for active emissive (no payload).
        assert_eq!(project(&cosines, &[(0.0, 1.0); 32]), (0.0, 1.0));
        assert_eq!(project(&cosines, &[(0.0, 0.0); 32]), (0.0, 0.0));
        assert_eq!(project(&cosines, &[(1.0, 0.0); 32]), (0.0, 0.0));

        for (bin, &cosine) in cosines.iter().enumerate() {
            let mut sparse = [(0.0, 0.0); 32];
            sparse[bin] = (1.0, 0.25);
            let expected = (
                if cosine > 0.0 { 1.0 } else { 0.0 },
                0.25 * cosine / nominal,
            );
            assert_eq!(
                project(&cosines, &sparse),
                expected,
                "lane {lane}, bin {bin}"
            );

            let mut mixed = [(1.0, 1.0); 32];
            mixed[bin] = (0.0, 1.0);
            let (value, coverage) = project(&cosines, &mixed);
            assert!((value - (1.0 - cosine / nominal)).abs() < 1e-6);
            assert_eq!(coverage, 1.0);
        }
    }
}

#[test]
fn antipodal_projection_preserves_directional_radiance_and_partial_support() {
    let samples = std::array::from_fn(|bin| {
        // Distinct values and a cold subset distinguish the two hemispheres.
        ((bin + 1) as f32, ((3 * bin) % 7) as f32 / 6.0)
    });
    for lane in 0..36 {
        let [x, y, z] = normal(lane);
        let dot = |bin| {
            let [dx, dy, dz] = bin_direction(bin);
            x * dx + y * dy + z * dz
        };
        let all_bins = std::array::from_fn(|bin| dot(bin).max(0.0));
        let mut paired = [0.0; 32];
        for bin in 0..16 {
            let cosine = dot(bin);
            paired[if cosine >= 0.0 { bin } else { bin ^ 28 }] = cosine.abs();
        }
        let (radiance, support) = project(&paired, &samples);
        let (expected_radiance, expected_support) = project(&all_bins, &samples);
        assert!((radiance - expected_radiance).abs() < 1e-5, "lane {lane}");
        assert!((support - expected_support).abs() < 1e-6, "lane {lane}");
    }
}
