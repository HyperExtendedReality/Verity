//! The 4×2 equal-solid-angle histogram and cosine/guide mixture in world_resolve.slang.

fn pack(mass: [f64; 32]) -> [u32; 2] {
    let mut coarse = [0.0_f64; 8];
    for (i, weight) in mass.into_iter().enumerate() {
        coarse[(i % 8) / 2 + 4 * (i / 16)] += weight;
    }
    let peak = coarse.into_iter().fold(0.0_f64, f64::max);
    let mut words = [0; 2];
    if peak > 0.0 {
        for (i, weight) in coarse.into_iter().enumerate() {
            words[i / 4] |= ((weight / peak * 255.0).round() as u32) << (8 * (i % 4));
        }
    }
    words
}

fn unpack(words: [u32; 2]) -> [u32; 8] {
    std::array::from_fn(|i| (words[i / 4] >> (8 * (i % 4))) & 255)
}

#[test]
fn preprojection_bins_have_disjoint_coarse_cells_and_no_empty_guide() {
    assert_eq!(pack([0.0; 32]), [0; 2]);
    assert_eq!(unpack(pack([1.0; 32])), [255; 8]);
    for i in 0..32 {
        let mut fine = [0.0; 32];
        fine[i] = 1.0;
        let mut expected = [0; 8];
        expected[(i % 8) / 2 + 4 * (i / 16)] = 255;
        assert_eq!(unpack(pack(fine)), expected);
    }
}

#[test]
fn mixture_keeps_cosine_support_and_integrates_bright_and_dark_bins() {
    // Integrate the proposal times its importance weight directly over the
    // sphere (PDF × π, and each of the eight bins has solid angle π/2).
    let masses = [10, 0, 0, 0, 1, 1, 0, 255];
    let total = f64::from(masses.into_iter().sum::<u32>());
    let mut integral = 0.0;
    let mut normalization = 0.0;
    for (bin, mass) in masses.into_iter().enumerate() {
        for step in 0..64 {
            let height = (bin / 4) as f64 - 1.0 + (f64::from(step) + 0.5) / 64.0;
            let cosine = height.max(0.0);
            let pdf = 0.5 * cosine + f64::from(mass) / total;
            normalization += pdf / 128.0;
            if cosine > 0.0 {
                assert!(pdf > 0.0);
                let weight = cosine / pdf;
                assert!(weight <= 2.0);
                let radiance = if bin == 7 { 8.0 } else { 1.0 };
                integral += radiance * weight * pdf / 128.0;
            }
        }
    }
    assert!((normalization - 1.0).abs() < 1e-12);
    assert!((integral - 2.75).abs() < 1e-12);
}
