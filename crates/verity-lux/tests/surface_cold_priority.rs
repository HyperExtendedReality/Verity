//! CPU model of Split's wave-local surface bounce quota (one bit per group lane).

const SETTLED: u32 = 4;
const REFRESH_EVERY: u32 = 4;

fn quotas(rays: u32, opt_in: bool, cold: u64, donor: u64, live: u64) -> [u32; 64] {
    let rays = rays.min(8);
    let cold = cold & live;
    let donor = donor & live;
    std::array::from_fn(|lane| {
        let bit = 1 << lane;
        if live & bit == 0 {
            return 0;
        }
        if !opt_in || rays == 0 || rays == 8 {
            return rays;
        }
        let rank = |mask: u64| (mask & (bit - 1)).count_ones();
        if cold & bit != 0 && rank(cold) < donor.count_ones() {
            rays + 1
        } else if donor & bit != 0 && rank(donor) < cold.count_ones() {
            rays - 1
        } else {
            rays
        }
    })
}

#[test]
fn partial_waves_pair_only_live_lanes_without_exceeding_the_budget_or_caps() {
    // Exhaust the combinations of inactive, cold, warming, and donor lanes.
    // Straddle the two mask words, including their first and last bits.
    for mut state in 0..4_u32.pow(6) {
        let (mut live, mut cold, mut donor) = (0u64, 0u64, 0u64);
        for lane in [0, 7, 31, 32, 49, 63] {
            let bit = 1 << lane;
            match state % 4 {
                1 => cold |= bit,
                3 => donor |= bit,
                _ => (),
            }
            if state % 4 != 0 {
                live |= bit;
            }
            state /= 4;
        }
        for rays in 0u32..=9 {
            for opt_in in [false, true] {
                // Ineligible lanes cannot donate or receive, even if their mass would qualify.
                let q = quotas(rays, opt_in, cold | !live, donor | !live, live);
                let base = rays.min(8);
                assert_eq!(q.iter().sum::<u32>(), base * live.count_ones());
                let (mut gains, mut losses) = (0u64, 0u64);
                for (lane, n) in q.into_iter().enumerate() {
                    let bit = 1 << lane;
                    let before = if live & bit != 0 { base } else { 0 };
                    assert!(n <= 8 && n.abs_diff(before) <= 1);
                    gains |= u64::from(n > before) * bit;
                    losses |= u64::from(n < before) * bit;
                }
                assert_eq!(gains & !cold, 0);
                assert_eq!(losses & !donor, 0);
                let pairs = cold.count_ones().min(donor.count_ones());
                let expected = if opt_in && (1..8).contains(&rays) {
                    pairs
                } else {
                    0
                };
                assert_eq!(gains.count_ones(), expected);
                assert_eq!(losses.count_ones(), expected);
            }
        }
    }
}

#[test]
fn a_permanently_paired_settled_texel_keeps_a_ray_in_every_four_visits() {
    for mass in [SETTLED - 1, SETTLED, SETTLED + 1] {
        for phase in [0u32, 1, 2, 3, u32::MAX] {
            for start in [0u32, 1, 2, 3, u32::MAX - 1] {
                let count = (0..REFRESH_EVERY)
                    .map(|offset| {
                        let visit = start.wrapping_add(offset);
                        let can_donate = mass >= SETTLED
                            && !visit.wrapping_add(phase).is_multiple_of(REFRESH_EVERY);
                        quotas(1, true, 1, u64::from(can_donate) << 63, 1 | (1 << 63))[63]
                    })
                    .sum::<u32>();
                assert_eq!(count, if mass >= SETTLED { 1 } else { REFRESH_EVERY });
            }
        }
    }
}
