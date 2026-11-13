use std::{env, fs, path::PathBuf};
use verity_slang::{SlangCompiler, SlangSettings, SlangStage};

/// Every pipeline entry point, written to `OUT_DIR/{file}.{entry}.spv`.
const ENTRY_POINTS: &[(&str, &str, SlangStage)] = &[
    ("clusters", "clusterMain", SlangStage::Compute),
    ("embedded", "embedMain", SlangStage::Compute),
    ("gpu_culling", "cullMain", SlangStage::Compute),
    ("gpu_culling", "pyramidMain", SlangStage::Compute),
    ("gpu_culling", "occlusionMain", SlangStage::Compute),
    ("visibility", "vertexMain", SlangStage::Vertex),
    ("visibility", "fragmentMain", SlangStage::Fragment),
    ("depth_prepass", "fragmentMain", SlangStage::Fragment),
    ("resolve", "resolveMain", SlangStage::Compute),
    ("resolve", "resolveMsaaMain", SlangStage::Compute),
    ("composite", "vertexMain", SlangStage::Vertex),
    ("composite", "fragmentMain", SlangStage::Fragment),
    ("shadow", "vertexMain", SlangStage::Vertex),
    ("shadow", "fragmentMain", SlangStage::Fragment),
    ("clouds", "shapeMain", SlangStage::Compute),
    ("clouds", "detailMain", SlangStage::Compute),
    ("clouds", "traceMain", SlangStage::Compute),
    ("clouds", "applyMain", SlangStage::Compute),
    ("clouds", "shadowMain", SlangStage::Compute),
    ("taa", "resolveMain", SlangStage::Compute),
    ("taa", "applyMain", SlangStage::Compute),
    ("breaker", "vertexMain", SlangStage::Vertex),
    ("breaker", "fragmentMain", SlangStage::Fragment),
    ("matrix_select", "instanceMain", SlangStage::Compute),
    ("matrix_select", "nodeMain", SlangStage::Compute),
    ("matrix_clusters", "vertexMain", SlangStage::Vertex),
    ("matrix_clusters", "fragmentMain", SlangStage::Fragment),
    ("matrix_clusters_depth", "vertexMain", SlangStage::Vertex),
    ("matrix_clusters_depth", "fragmentMain", SlangStage::Fragment),
    ("matrix_clusters_shadow", "vertexMain", SlangStage::Vertex),
    ("matrix_clusters_shadow", "fragmentMain", SlangStage::Fragment),
    ("matrix_voxels", "vertexMain", SlangStage::Vertex),
    ("matrix_voxels", "fragmentMain", SlangStage::Fragment),
    ("matrix_voxels_depth", "vertexMain", SlangStage::Vertex),
    ("matrix_voxels_depth", "fragmentMain", SlangStage::Fragment),
    ("matrix_voxels_shadow", "vertexMain", SlangStage::Vertex),
    ("matrix_voxels_shadow", "fragmentMain", SlangStage::Fragment),
    ("matrix_imposters", "vertexMain", SlangStage::Vertex),
    ("matrix_imposters", "fragmentMain", SlangStage::Fragment),
    ("matrix_imposters_depth", "vertexMain", SlangStage::Vertex),
    ("matrix_imposters_depth", "fragmentMain", SlangStage::Fragment),
    ("matrix_bake", "vertexMain", SlangStage::Vertex),
    ("matrix_bake", "fragmentMain", SlangStage::Fragment),
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
