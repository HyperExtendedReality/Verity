//! CPU model of world resolve's eight-candidate emissive RIS sampling.

fn hash(mut x: u32) -> u32 {
    x = (x ^ (x >> 16)).wrapping_mul(0x7feb352d);
    x = (x ^ (x >> 15)).wrapping_mul(0x846ca68b);
    x ^ (x >> 16)
}

fn unit(bits: u32) -> f64 {
    f64::from(bits >> 8) / 16_777_216.0
}

fn random(state: &mut u32) -> f64 {
    *state = hash(*state);
    unit(*state)
}

#[test]
fn r3_reduces_joint_candidate_error_and_preserves_the_modeled_ris_mean() {
    // Middle CDF entry, then a triangle patch with sqrt(u) < 3/4 and v < 5/8.
    let expected = 0.5 * 0.75 * 0.75 * 0.625;
    let mut error = [0.0; 2];
    let mut ris_sum = [0.0; 2];
    for frame in 0u32..128 {
        for pixel in 0..64 {
            let seed = hash(pixel ^ hash(frame.wrapping_mul(0x9e3779b9).wrapping_add(0x51ed270b)));
            let shift = [0x68bc21eb, 0x02e5be93, 0x967a889b].map(|salt| hash(seed ^ salt));
            let mut state = seed;
            let mut hits = [0.0; 2];
            let mut total = [0.0; 2];
            let mut chosen_visible = [false; 2];
            for index in 0u32..8 {
                let steps = [3_518_319_155, 2_882_110_345, 2_360_945_575];
                let r3 = std::array::from_fn(|d| {
                    unit(shift[d].wrapping_add(index.wrapping_mul(steps[d])))
                });
                let iid = [(); 3].map(|_| random(&mut state));
                let coins = [unit(hash(seed ^ (0xa511e9b3 + index))), random(&mut state)];
                for (mode, [pick, u, v]) in [r3, iid].into_iter().enumerate() {
                    assert!([pick, u, v].into_iter().all(|x| (0.0..1.0).contains(&x)));
                    let visible = pick > 0.375 && pick <= 0.875 && u.sqrt() < 0.75 && v < 0.625;
                    // Toy p-hat/pdf: brighter on the patch and nonzero everywhere,
                    // but only the patch is visible. f/p-hat is thus 0 or 1.
                    let w = if visible { 8.0 } else { 1.0 };
                    hits[mode] += f64::from(u8::from(visible));
                    total[mode] += w;
                    if coins[mode] * total[mode] < w {
                        chosen_visible[mode] = visible;
                    }
                }
            }
            for mode in 0..2 {
                let balance: f64 = hits[mode] / 8.0 - expected;
                error[mode] += balance * balance;
                if chosen_visible[mode] {
                    ris_sum[mode] += total[mode] / 8.0;
                }
            }
        }
    }
    assert!(error[0] < error[1] * 0.75, "candidate error: {error:?}");
    for sum in ris_sum {
        let mean = sum / (128.0 * 64.0);
        assert!((mean - 8.0 * expected).abs() < 0.05, "RIS mean: {mean}");
    }
}
