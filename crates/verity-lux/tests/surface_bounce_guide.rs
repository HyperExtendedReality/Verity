//! Surface bounce guiding: the shared quantized proposal changes directions,
//! not the ray quota or the cosine-weighted lighting target.

const SURFACE: &str = include_str!("../shaders/surface_cache.slang");
// 4 azimuth bins in each y half of the world sphere; most mass faces down.
const PACKED: [u32; 2] = [0x0b04_00ff, 0x0208_0014];

fn mass(bin: usize) -> u32 {
    (PACKED[bin / 4] >> (8 * (bin % 4))) & 255
}

#[test]
fn surface_guiding_leaves_the_existing_cosine_stream_and_quota_as_fallback() {
    let update = SURFACE.split_once("void surfaceUpdate(").unwrap().1;
    assert!(update.contains("quota > 0u && (params.ownerSequence.y & 16u) != 0u"));
    assert!(update.contains("bool guided = any(guide != 0u)"));
    assert!(update.contains("direction = rtgiCosineDirection(normal, u);"));
    assert!(update.contains("for (uint ray = 0u; ray < quota; ++ray)"));
    assert_eq!(update.matches("traceRayTracked(").count(), 1);
    assert!(
        update.find("if (weight <= 0.0) continue;").unwrap()
            < update.find("traceRayTracked(").unwrap()
    );
}

#[test]
fn guide_selection_uses_quantized_mass_including_empty_bins() {
    let total: u32 = (0..8).map(mass).sum();
    let mut selected = [0; 8];
    // One test point per unit of categorical mass, away from CDF boundaries.
    for i in 0..total {
        let selector = (f64::from(i) + 0.5) / f64::from(total);
        let mut target = selector * f64::from(total);
        let mut bin = 0;
        while bin < 7 && target >= f64::from(mass(bin)) {
            target -= f64::from(mass(bin));
            bin += 1;
        }
        selected[bin] += 1;
    }
    assert_eq!(selected, [255, 0, 4, 11, 20, 0, 8, 2]);
}

#[test]
fn mixed_pdf_keeps_cosine_support_and_the_same_rgb_update_target() {
    let total = f64::from((0..8).map(mass).sum::<u32>());
    let history = [2.0, 4.0, 6.0];
    let mut probability = 0.0;
    let mut covered_probability = 0.0;
    let mut delta = [0.0; 3];
    for bin in 0..8 {
        for step in 0..64 {
            // Uniform solid angle in each coarse bin; geometric normal is +Y.
            let y = (bin / 4) as f64 - 1.0 + (f64::from(step) + 0.5) / 64.0;
            let cosine = y.max(0.0);
            let pdf_pi = 0.5 * cosine + f64::from(mass(bin)) / total;
            let p = pdf_pi / 128.0; // solid angle π/2 per bin, then 64 steps
            probability += p;
            let weight = if pdf_pi > 0.0 { cosine / pdf_pi } else { 0.0 };
            if y < 0.0 {
                assert_eq!(weight, 0.0);
                continue;
            }
            assert!(weight > 0.0 && weight <= 2.0);
            if mass(bin) == 0 {
                assert_eq!(weight, 2.0);
            }
            let radiance = match bin {
                4 => [8.0, 4.0, 1.0], // sky sample, background only in the shader
                5 => [4.0, 0.0, 5.0], // valid hit in an empty guide bin
                7 => [0.0, 8.0, 2.0],
                _ => continue, // uncached hit or software reach miss
            };
            covered_probability += p;
            for c in 0..3 {
                delta[c] += p * weight * (radiance[c] - history[c]);
            }
        }
    }
    assert!((probability - 1.0).abs() < 1e-12);
    assert!((covered_probability - 0.75).abs() > 0.25);
    // Each usable bin has 1/4 of the original cosine mass. The missing bin
    // contributes no update, not black; no accepted-ray normalization is used.
    for (actual, expected) in delta.into_iter().zip([1.5, 0.0, -2.5]) {
        assert!((actual - expected).abs() < 1e-12);
    }
}
