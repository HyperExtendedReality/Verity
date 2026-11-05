use hyperon_slang::{SlangCompiler, SlangSettings, SlangStage};
use std::{env, fs, path::PathBuf};

/// Every pipeline entry point, written to `OUT_DIR/{file}.{entry}.spv`.
const ENTRY_POINTS: &[(&str, &str, SlangStage)] = &[
    ("clusters", "clusterMain", SlangStage::Compute),
    ("gpu_culling", "cullMain", SlangStage::Compute),
    ("gpu_culling", "pyramidMain", SlangStage::Compute),
    ("gpu_culling", "occlusionMain", SlangStage::Compute),
    ("visibility", "vertexMain", SlangStage::Vertex),
    ("visibility", "fragmentMain", SlangStage::Fragment),
    ("depth_prepass", "fragmentMain", SlangStage::Fragment),
    ("resolve", "resolveMain", SlangStage::Compute),
    ("composite", "vertexMain", SlangStage::Vertex),
    ("composite", "fragmentMain", SlangStage::Fragment),
    ("shadow", "vertexMain", SlangStage::Vertex),
    ("shadow", "fragmentMain", SlangStage::Fragment),
    ("breaker", "vertexMain", SlangStage::Vertex),
    ("breaker", "fragmentMain", SlangStage::Fragment),
];

fn main() {
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-env-changed=SLANGC");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let compiler = SlangCompiler::default().with_source_root("shaders");
    for &(file, entry, stage) in ENTRY_POINTS {
        let settings = SlangSettings {
            entry_points: vec![entry.into()],
            stage: Some(stage),
            ..SlangSettings::default()
        };
        let shader = compiler
            .compile_file(format!("{file}.slang"), &settings)
            .unwrap_or_else(|error| panic!("compile {file}.slang:{entry} with slangc: {error}"));
        let bytes: Vec<u8> = shader.words.iter().flat_map(|w| w.to_le_bytes()).collect();
        fs::write(out.join(format!("{file}.{entry}.spv")), bytes).unwrap();
    }
}
