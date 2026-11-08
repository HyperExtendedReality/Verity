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
    "traceDeferred",
    "traceRecord",
    "traceVisibility",
    "traceResolve",
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
    "inject", "trace", "temporal", "atrous1", "atrous2", "atrous4", "composite", "capture",
];

/// Writes `OUT_DIR/{file}.{entry}.spv` for each entry point, compiling in
/// parallel; the transport shaders are large. The cascades passes also get
/// `cascades.{entry}.rq.spv`, built with `GI_RAY_QUERY` to trace with hardware
/// ray queries, which they pick when the GPU supports them.
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
    // PXTGI traces the physics scene (verity-physics' `RayScene`), never hardware.
    jobs.extend(PXTGI_ENTRIES.iter().map(|&e| ("pxtgi", e, false)));
    jobs.extend([
        ("composite", "compositeAnchors", false),
        ("composite", "compositeCompute", false),
        ("composite", "compositeCapture", false),
        ("debug_surface", "debugSurface", false),
    ]);
    jobs.extend(VOXEL_ENTRIES.iter().map(|&e| ("voxel", e, false)));
    let compiler = SlangCompiler::default().with_source_root("shaders");
    let failures: Vec<String> = thread::scope(|scope| {
        let handles: Vec<_> = jobs
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
            .collect()
    });
    if !failures.is_empty() {
        panic!("slangc failed:\n{}", failures.join("\n"));
    }
}
