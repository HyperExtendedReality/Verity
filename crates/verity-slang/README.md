# verity_slang

`verity_slang` compiles trusted [Slang](https://github.com/shader-slang/slang) source to SPIR-V for Verity's Vulkan renderer. It invokes the external `slangc` compiler synchronously, without Bevy or wgpu.

## Requirements

Install a Slang binary distribution with `slangc` on `PATH`, or set `SLANGC` to the executable path. Official Slang releases provide binaries for Windows, Linux and macOS. Runtime invocation is not supported on WebAssembly; compile shaders on a native host instead. A failed launch identifies the configured executable and explains how to install or configure it.

## Compilation

```rust
use verity_slang::{SlangCompiler, SlangSettings, SlangStage};

fn compile() -> Result<(), Box<dyn std::error::Error>> {
    let compiler = SlangCompiler::default()
        .with_source_root("assets")
        .with_include_path("assets/shaders/include");
    let settings = SlangSettings {
        profile: Some("glsl_460+spirv_1_5".into()),
        entry_points: vec!["computeMain".into()],
        stage: Some(SlangStage::Compute),
        defines: vec!["USE_FOG=1".into()],
        debug_info: true,
        ..Default::default()
    };
    let shader = compiler.compile_file("shaders/example.slang", &settings)?;
    if !shader.diagnostics.is_empty() {
        eprintln!("{}", shader.diagnostics);
    }
    // Supply &shader.words to ash::vk::ShaderModuleCreateInfo::default().code(...).
    Ok(())
}
```

`CompiledShader::words` is a `Vec<u32>` suitable for Vulkan shader module creation, `path` is the logical source name, and `diagnostics` contains compiler stdout/stderr even on success. Compilation errors include the path, exit status and compiler output. The compiler checks the SPIR-V header, size and magic; this is not a full SPIR-V or Vulkan validation pass. Slang normally validates its output; only compile trusted shaders and use Vulkan validation as appropriate.

`compile_source(path, bytes, settings)` handles in-memory source. The path is logical, and its parent under `source_root` is searched for sibling includes even if the source itself is not on disk. `compile_bundle(entry, &[(path, source), ...], settings)` writes an isolated temporary source tree, preserving relative includes and imports. Bundle paths must be relative and cannot traverse a parent. The temporary files are removed after compilation. Shared compiler `include_paths` and per-shader `settings.include_paths` are passed to `slangc`; relative include paths are relative to the process working directory. The default source root is `assets`.

Settings expose Slang profiles, stages, entry points, includes, defines, optimization, debug information, matrix layout, named Vulkan entry points, and raw extra arguments. Entry points and stage can normally be omitted when shader functions use `[shader("compute")]` or equivalent attributes. `build_args` exposes the generated command arguments. Compilation blocks while the external process runs, so run it during initialization or on an application-owned worker thread.

The API does not watch source files or track imported dependencies. Applications implementing hot reload must track changes to the entry file and its included/imported files and recompile as needed.

## Descriptor bindings

Vulkan pipeline layouts normally use the descriptor sets and binding numbers emitted by Slang directly. If an application requires different binding numbers, `remap_spirv_bindings(&mut shader, &[SpirvBindingRemap { group, binding, mapped_binding }])` rewrites matching SPIR-V decorations. Unlisted bindings remain unchanged. The caller is responsible for ensuring the resulting shader and pipeline layouts agree.

Run the example with `cargo run -p verity_slang --example load_shader` from the workspace root. The integration test requiring a real Slang installation is ignored by default and can be selected with `cargo test -p verity_slang -- --ignored`.

## License

Licensed under either the Apache License, Version 2.0 or the MIT License.
