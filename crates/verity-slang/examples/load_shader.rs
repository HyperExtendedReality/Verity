//! Compile the bundled example into words suitable for a Vulkan shader module.

use std::path::Path;
use verity_slang::{SlangCompiler, SlangSettings};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let compiler = SlangCompiler::default()
        .with_source_root(Path::new(env!("CARGO_MANIFEST_DIR")).join("assets"));
    let shader = compiler.compile_file("shaders/example.slang", &SlangSettings::default())?;
    if !shader.diagnostics.is_empty() {
        eprintln!("{}", shader.diagnostics);
    }
    println!(
        "Compiled {} SPIR-V words from {}",
        shader.words.len(),
        shader.path
    );
    Ok(())
}
