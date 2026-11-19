//! CPU mirror of voxel.slang's inject schedule and bounce sampler; no GPU or crate needed.

fn hash_u(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^ (x >> 16)
}

fn unit(bits: u32) -> f32 {
    (bits >> 8) as f32 / 16_777_216.0
}

fn bounce_sample(visit: u32, cell: u32, side: u32) -> [f32; 2] {
    let mut y = visit;
    y ^= (y >> 1) & 0x55555555;
    y ^= (y >> 2) & 0x33333333;
    y ^= (y >> 4) & 0x0f0f0f0f;
    y ^= (y >> 8) & 0x00ff00ff;
    y ^= y >> 16;
    let shift = hash_u(cell ^ (side << 31));
    [
        unit(visit.reverse_bits() ^ shift),
        unit(y.reverse_bits() ^ hash_u(shift ^ 0x9e3779b9)),
    ]
}

fn scheduled_visit(frame: u32, index: u32, side: u32) -> Option<u32> {
    let scheduled_index = (index / 4) * 4 + (frame & 3);
    (scheduled_index == index && ((frame >> 2) & 1) == side).then_some(frame >> 3)
}

#[test]
fn visits_stratify_every_dyadic_tiling() {
    const LOG_VISITS: u32 = 6;
    const VISITS: u32 = 1 << LOG_VISITS;
    // The host skips frame zero. Include ordinary windows and the last window before wrap.
    for start in [VISITS, 123 * VISITS, (1 << 29) - VISITS] {
        for index in 300..304 {
            for side in 0..2 {
                let first_frame = start << 3;
                let samples: Vec<_> = (first_frame..=first_frame + (VISITS * 8 - 1))
                    .filter_map(|frame| scheduled_visit(frame, index, side))
                    .enumerate()
                    .map(|(i, visit)| {
                        assert_eq!(visit, start + i as u32);
                        bounce_sample(visit, index * 907, side)
                    })
                    .collect();
                assert_eq!(samples.len(), VISITS as usize);
                for x_bits in 0..=LOG_VISITS {
                    let (cols, rows) = (1 << x_bits, VISITS >> x_bits);
                    let mut counts = [0; VISITS as usize];
                    for [u, v] in &samples {
                        let (x, y) = ((u * cols as f32) as u32, (v * rows as f32) as u32);
                        counts[(y * cols + x) as usize] += 1;
                    }
                    assert_eq!(counts, [1; VISITS as usize]);
                }
            }
        }
    }
}

#[test]
fn late_visits_do_not_repeat_the_low_bits() {
    let sample = |epoch: u32| bounce_sample((epoch << 24) | 73, 435_229, 1);
    for epoch in 0u32..32 {
        // The first coordinate repeats after 2^24 visits, but the full integer
        // visit is mixed into the other coordinate before conversion to float.
        for earlier in 0..epoch {
            assert_ne!(sample(epoch), sample(earlier));
        }
        assert_ne!(sample(epoch), bounce_sample((epoch << 24) | 74, 435_229, 1));
    }
}

fn direction([u, v]: [f32; 2]) -> [f32; 3] {
    let (s, c) = (std::f32::consts::TAU * v).sin_cos();
    [u.sqrt() * c, u.sqrt() * s, (1.0 - u).sqrt()]
}

#[test]
fn cosine_direction_moment_improves_at_fixed_bounce_budget() {
    // With unit incident light the direction moment is (0, 0, 2/3).
    // Measure vector RMSE over voxels, both sides and unaligned visit windows.
    for budget in [8, 64] {
        let (mut stratified, mut jitter) = (0.0, 0.0);
        let indices = 4096..4352;
        for index in indices.clone() {
            for side in 0..2 {
                let first_visit = hash_u(index ^ (side << 31)) >> 4;
                let phase = (side << 2) | (index & 3);
                let (mut net, mut old) = ([0.0; 3], [0.0; 3]);
                for k in 0..budget {
                    let frame = ((first_visit + k) << 3) | phase;
                    let visit = scheduled_visit(frame, index, side).unwrap();
                    let seed = hash_u(index ^ hash_u(frame));
                    let a = direction(bounce_sample(visit, (index - 4096) * 983 + 65536, side));
                    let b = direction([unit(seed), unit(hash_u(seed))]);
                    for axis in 0..3 {
                        net[axis] += a[axis];
                        old[axis] += b[axis];
                    }
                }
                for (axis, exact) in [0.0, 0.0, 2.0 / 3.0].into_iter().enumerate() {
                    stratified += (net[axis] / budget as f32 - exact).powi(2);
                    jitter += (old[axis] / budget as f32 - exact).powi(2);
                }
            }
        }
        let cases = (indices.len() * 2) as f32;
        let (net_rmse, jitter_rmse) = ((stratified / cases).sqrt(), (jitter / cases).sqrt());
        println!("{budget} visits: moment RMSE {net_rmse:.5} vs hash jitter {jitter_rmse:.5}");
        assert!(net_rmse < 0.75 * jitter_rmse);
    }
}
