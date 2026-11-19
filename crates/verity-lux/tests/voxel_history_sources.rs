//! Standalone CPU model of the voxel temporal source policy; no GPU or crate needed.

fn tag(counts: [u32; 2], rays: u32, proxy: bool) -> u32 {
    let units = 12 / rays;
    1 + (counts[0] * units | ((counts[1] * units) << 4) | (u32::from(proxy) << 8))
}

fn trust(current: u32, previous: u32) -> f32 {
    if current == 0 || previous == 0 {
        return 0.0;
    }
    let (a, b) = (current - 1, previous - 1);
    if (a ^ b) & 256 != 0 {
        return 0.0;
    }
    let diff = |shift| ((a >> shift) & 15_u32) as i32 - ((b >> shift) & 15_u32) as i32;
    let (x, y) = (diff(0), diff(4));
    let delta = x.abs().max(y.abs()).max((x + y).abs()) as f32 / 12.0;
    let t = (2.0 * delta - 1.0).clamp(0.0, 1.0);
    1.0 - 0.75 * (t * t * (3.0 - 2.0 * t))
}

#[test]
fn tags_fit_half_float_and_preserve_the_histogram_across_ray_budgets() {
    for proxy in [false, true] {
        for rays in 1..=4 {
            for a in 0..=rays {
                for b in 0..=rays - a {
                    let packed = tag([a, b], rays, proxy);
                    assert!((1..=2048).contains(&packed)); // exactly representable integers
                    assert_eq!((packed - 1) & 15, a * 12 / rays);
                    assert_eq!(((packed - 1) >> 4) & 15, b * 12 / rays);
                    assert_eq!(((packed - 1) >> 8) != 0, proxy);
                }
            }
        }
        assert_eq!(tag([1, 1], 2, proxy), tag([2, 2], 4, proxy));
    }
}

#[test]
fn stable_sources_retain_history_but_disjoint_histograms_attenuate() {
    for proxy in [false, true] {
        let sources = [[4, 0], [0, 4], [0, 0]].map(|counts| tag(counts, 4, proxy));
        for a in sources {
            for b in sources {
                assert_eq!(trust(a, b), if a == b { 1.0 } else { 0.25 });
            }
        }
        assert_eq!(trust(sources[0], tag([1, 1], 2, proxy)), 1.0);
        assert!(trust(sources[0], tag([1, 3], 4, proxy)) < 1.0);
    }
}

#[test]
fn reject_missing_tags_and_residual_proxy_transitions() {
    let proxy_only = tag([0, 0], 2, true);
    let uncovered = tag([0, 0], 2, false);
    assert_eq!(trust(proxy_only, uncovered), 0.0);
    assert_eq!(trust(uncovered, proxy_only), 0.0);
    assert_eq!(trust(proxy_only, 0), 0.0); // enabled after legacy history
    assert_eq!(trust(0, proxy_only), 0.0); // invalid current sample
}

#[test]
fn reprojected_taps_do_not_mix_in_rejected_light_or_keep_its_history_weight() {
    let proxy = tag([0, 0], 2, true);
    let fallback = tag([0, 0], 2, false);
    let taps = [(proxy, 0.25, 2.0), (fallback, 0.75, 200.0)];
    let retained: f32 = taps.iter().map(|&(tag, w, _)| w * trust(proxy, tag)).sum();
    let light: f32 = taps
        .iter()
        .map(|&(tag, w, rgb)| rgb * w * trust(proxy, tag))
        .sum();
    assert_eq!(light / retained, 2.0);
    let old_frames = 32.0;
    let alpha = 1.0 - retained * (1.0 - 1.0 / old_frames);
    assert!(alpha > 0.75);
    assert!(1.0 / alpha < 2.0);
}

#[test]
fn shader_keeps_the_signed_residual_and_coverage_separate_from_source_tags() {
    let voxel = include_str!("../shaders/voxel.slang");
    let trace = voxel.split_once("void trace(").unwrap().1;
    let trace = trace.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(trace.contains("if (residual) { if (seen.w > 0.5) {"));
    assert!(trace.contains(
        "light += clampFirefly(seen.rgb) - (found != 0u ? voxelSeen(found, dir) : background(dir));"
    ));
    assert!(trace.contains("float4(max(proxy + light / float(rays), 0.0), 1.0)"));
    assert!(trace.contains("float4(light, hits) / float(rays)"));
    assert!(trace.contains("out2.Store(px, float4(source, 0.0, 0.0, 0.0));"));
    let host = include_str!("../src/voxel.rs");
    assert!(host.contains("source_aware_history: false,"));
}
