//! CPU model of Split RC's surface-cache schedule and cosine bounce sampler.

const SEED: u32 = 0x57face5;

fn hash(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846ca68b);
    value ^ (value >> 16)
}

fn unit(bits: u32) -> f64 {
    f64::from(bits >> 8) / 16_777_216.0
}

fn r2(index: u32) -> [f64; 2] {
    [
        unit(index.wrapping_mul(3_242_174_889)),
        unit(index.wrapping_mul(2_447_445_414)),
    ]
}

#[test]
fn sparse_schedule_keeps_consecutive_samples_per_slot() {
    for (slots, budget) in [(509, 37), (512, 32), (19, 19)] {
        let mut visits = vec![0; slots as usize];
        for frame in 0..(slots * 2) {
            for lane in 0..budget {
                let cursor = frame * budget + lane;
                let visit = cursor / slots;
                let slot = cursor - visit * slots;
                assert_eq!(slot, cursor % slots);
                assert_eq!(visit, visits[slot as usize]);
                visits[slot as usize] += 1;
            }
        }
    }
}

// In the cosine map, z = sqrt(1-u) has mean 2/3. A window that accepts u <
// 0.36 and the first azimuthal quadrant accepts 9% of the cosine-weighted rays.
fn integrands([u, v]: [f64; 2]) -> [f64; 2] {
    [
        (1.0 - u).sqrt(),
        if u < 0.36 && v < 0.25 { 1.0 } else { 0.0 },
    ]
}

#[test]
fn consecutive_r2_directions_reduce_error_at_the_same_ray_budget() {
    for rays in [1, 2, 3, 8] {
        const VISITS: u32 = 16;
        let mut r2_error = [0.0; 2];
        let mut random_error = [0.0; 2];
        for texel in 0..256 {
            let mut stratified_sum = [0.0; 2];
            let mut random_sum = [0.0; 2];
            for visit in 0..VISITS {
                let frame = (texel + visit * 509) / 37;
                let mut state = hash(texel ^ hash(frame ^ SEED));
                for ray in 0..rays {
                    let index = hash(texel ^ SEED).wrapping_add(visit * rays + ray);
                    let stratified = integrands(r2(index));
                    state = hash(state);
                    let u = unit(state);
                    state = hash(state);
                    let random = integrands([u, unit(state)]);
                    for i in 0..2 {
                        stratified_sum[i] += stratified[i];
                        random_sum[i] += random[i];
                    }
                }
            }
            let inv_count = 1.0 / f64::from(VISITS * rays);
            for (i, exact) in [2.0 / 3.0, 0.09].into_iter().enumerate() {
                r2_error[i] += (stratified_sum[i] * inv_count - exact).powi(2);
                random_error[i] += (random_sum[i] * inv_count - exact).powi(2);
            }
        }
        for i in 0..2 {
            assert!(
                r2_error[i] < 0.7 * random_error[i],
                "{rays} rays, integrand {i}"
            );
        }
    }
}
