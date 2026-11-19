use std::{env, fs, path::PathBuf, thread};
use verity_slang::{SlangCompiler, SlangSettings, SlangStage};

/// The Split Radiance Cascades frame's compute passes, all in `cascades.slang`.
const CASCADE_ENTRIES: &[&str] = &[
    "clear",
    "seed",
    "parents",
    "rank",
    "prefix",
    "generateDirections",
    "retain",
    "trace",
    "reduce",
    "merge",
    "seedProbes",
    "packProbes",
    "temporal",
    "denoise",
    "commitCache",
    "remapProbeHeads",
    "worldResolve",
    "guidedResolve",
    "materialResolve",
    "traceDeferred",
    "traceRecord",
    "traceVisibility",
    "traceResolve",
    "surfaceUpdate",
    "reference",
    "materialReference",
];

/// PXTGI's compute passes, all in `pxtgi.slang`.
const PXTGI_ENTRIES: &[&str] = &[
    "cache_gather",
    "cache_shade",
    "trace",
    "temporal",
    "atrous1",
    "atrous2",
    "atrous4",
    "reflect_trace",
    "reflect_temporal",
    "reflect_filter1",
    "reflect_filter2",
    "composite",
    "capture",
    "reference",
];

/// The screen and voxel method's compute passes, all in `voxel.slang`.
const VOXEL_ENTRIES: &[&str] = &[
    "reset",
    "clear",
    "claim",
    "capLarge",
    "claimLarge",
    "accumulate",
    "accumulateLarge",
    "finalize",
    "voxelSdfX",
    "voxelSdfY",
    "voxelSdfZ",
    "romaBuild",
    "inject",
    "trace",
    "temporal",
    "atrous1",
    "atrous2",
    "atrous4",
    "composite",
    "capture",
    "hrcMediumBuild",
    "hrcMeasure",
    "hrcBase",
    "hrcExtend",
    "hrcTop",
    "hrcDown",
    "hrcPull",
];

/// Writes `OUT_DIR/{file}.{entry}.spv` for each entry point, compiling in
/// parallel; the transport shaders are large. The cascades and PXTGI passes also
/// get `{file}.{entry}.rq.spv`, built with `GI_RAY_QUERY` to trace with hardware
/// ray queries, which they use when the GPU supports them and settings allow.
fn main() {
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-changed=../verity-render/shaders/lib");
    println!("cargo:rerun-if-env-changed=SLANGC");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let library = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
        .join("../verity-render/shaders/lib");
    let mut jobs: Vec<(&str, &str, bool)> = Vec::new();
    for ray_query in [false, true] {
        jobs.extend(CASCADE_ENTRIES.iter().map(|&e| ("cascades", e, ray_query)));
    }
    // PXTGI traces the physics scene (verity-physics' `RayScene`), through its own
    // BVH or, built from the same scene, hardware acceleration structures.
    for ray_query in [false, true] {
        jobs.extend(PXTGI_ENTRIES.iter().map(|&e| ("pxtgi", e, ray_query)));
    }
    jobs.extend([
        ("composite", "compositeAnchors", false),
        ("composite", "compositeCompute", false),
        ("composite", "compositeCapture", false),
        ("debug_surface", "debugSurface", false),
    ]);
    jobs.extend(VOXEL_ENTRIES.iter().map(|&e| ("voxel", e, false)));
    let compiler = SlangCompiler::default().with_source_root("shaders");
    // One slangc per job at once runs out of memory; batch them by core count.
    let width = thread::available_parallelism().map_or(4, |n| n.get());
    let failures: Vec<String> = jobs
        .chunks(width)
        .flat_map(|batch| {
            thread::scope(|scope| {
                let handles: Vec<_> = batch
                    .iter()
                    .map(|&(file, entry, ray_query)| {
                        let (compiler, library, out) = (&compiler, &library, &out);
                        scope.spawn(move || {
                            let settings = SlangSettings {
                                entry_points: vec![entry.into()],
                                stage: Some(SlangStage::Compute),
                                include_paths: vec![library.clone()],
                                defines: if ray_query {
                                    vec!["GI_RAY_QUERY".into()]
                                } else {
                                    Vec::new()
                                },
                                ..SlangSettings::default()
                            };
                            let shader = compiler
                                .compile_file(format!("{file}.slang"), &settings)
                                .map_err(|e| format!("{file}.slang:{entry}: {e}"))?;
                            let bytes: Vec<u8> =
                                shader.words.iter().flat_map(|w| w.to_le_bytes()).collect();
                            let variant = if ray_query { ".rq" } else { "" };
                            fs::write(out.join(format!("{file}.{entry}{variant}.spv")), bytes)
                                .map_err(|e| e.to_string())
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .filter_map(|handle| handle.join().unwrap().err())
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    if !failures.is_empty() {
        panic!("slangc failed:\n{}", failures.join("\n"));
    }
    write_blue_noise(&out.join("blue_noise.bin"));
}

/// Side of the blue-noise tile the GI ray generators index by `pixel & 63`.
const NOISE: usize = 64;

/// Four independent blue-noise channels, ranks `0..4096` as little-endian u16,
/// interleaved per texel.
fn write_blue_noise(path: &std::path::Path) {
    let n = NOISE * NOISE;
    let channels: Vec<Vec<u16>> = (1..=4).map(rank_channel).collect();
    for (c, ranks) in channels.iter().enumerate() {
        let mut seen = vec![false; n];
        for &r in ranks {
            assert!(
                !seen[r as usize],
                "blue noise channel {c}: rank {r} repeats"
            );
            seen[r as usize] = true;
        }
        // Neighbouring ranks of white noise differ by n/3 on average; blue by more.
        let mut sum = 0.0;
        for y in 0..NOISE {
            for x in 0..NOISE {
                let a = ranks[x + y * NOISE] as f64;
                sum += (a - ranks[(x + 1) % NOISE + y * NOISE] as f64).abs();
                sum += (a - ranks[x + (y + 1) % NOISE * NOISE] as f64).abs();
            }
        }
        let diff = sum / (2 * n * n) as f64;
        assert!(
            diff > 0.36,
            "blue noise channel {c}: neighbour difference {diff:.3} is white"
        );
    }
    let bytes: Vec<u8> = (0..n)
        .flat_map(|i| channels.iter().map(move |c| c[i]))
        .flat_map(u16::to_le_bytes)
        .collect();
    fs::write(path, bytes).unwrap();
}

/// Void and cluster (Ulichney 1993) on a toroidal tile: each texel's rank in
/// a sequence of points spread as evenly as possible.
fn rank_channel(seed: u32) -> Vec<u16> {
    const R: isize = 6;
    const SIGMA: f32 = 1.9;
    let n = NOISE * NOISE;
    let mut kernel = Vec::new();
    for dy in -R..=R {
        for dx in -R..=R {
            let w = (-((dx * dx + dy * dy) as f32) / (2.0 * SIGMA * SIGMA)).exp();
            kernel.push((dx, dy, w));
        }
    }
    let splat = |energy: &mut [f32], at: usize, sign: f32| {
        let (x, y) = ((at % NOISE) as isize, (at / NOISE) as isize);
        for &(dx, dy, w) in &kernel {
            let i = (x + dx).rem_euclid(NOISE as isize) as usize
                + (y + dy).rem_euclid(NOISE as isize) as usize * NOISE;
            energy[i] += sign * w;
        }
    };
    // The tightest cluster (`want` on, highest energy) or largest void.
    let pick = |energy: &[f32], on: &[bool], want: bool| {
        (0..n)
            .filter(|&i| on[i] == want)
            .reduce(|a, b| {
                let better = if want {
                    energy[b] > energy[a]
                } else {
                    energy[b] < energy[a]
                };
                if better { b } else { a }
            })
            .unwrap()
    };
    let mut energy = vec![0.0f32; n];
    let mut on = vec![false; n];
    let mut state = seed.wrapping_mul(0x9e37_79b9) | 1;
    let initial = n / 10;
    let mut placed = 0;
    while placed < initial {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let i = state as usize % n;
        if !on[i] {
            on[i] = true;
            splat(&mut energy, i, 1.0);
            placed += 1;
        }
    }
    // Move the tightest cluster to the largest void until that changes nothing.
    loop {
        let cluster = pick(&energy, &on, true);
        on[cluster] = false;
        splat(&mut energy, cluster, -1.0);
        let void = pick(&energy, &on, false);
        on[void] = true;
        splat(&mut energy, void, 1.0);
        if void == cluster {
            break;
        }
    }
    let mut rank = vec![0u16; n];
    let (mut e, mut o) = (energy.clone(), on.clone());
    for r in (0..initial).rev() {
        let cluster = pick(&e, &o, true);
        o[cluster] = false;
        splat(&mut e, cluster, -1.0);
        rank[cluster] = r as u16;
    }
    for r in initial..n {
        let void = pick(&energy, &on, false);
        on[void] = true;
        splat(&mut energy, void, 1.0);
        rank[void] = r as u16;
    }
    rank
}
