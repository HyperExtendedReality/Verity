//! Compile trusted Slang source to SPIR-V for Vulkan with the external `slangc`.
//! Set `SLANGC` or put `slangc` on `PATH`. Compilation is synchronous.

use std::{
    ffi::OsString,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// SPIR-V words ready for `ash::vk::ShaderModuleCreateInfo::code`.
#[derive(Clone, Debug)]
pub struct CompiledShader {
    /// Logical source path, used in diagnostics.
    pub path: String,
    /// Little-endian SPIR-V decoded into native `u32` words.
    pub words: Vec<u32>,
    /// Compiler stdout and stderr, including warnings on successful compilation.
    pub diagnostics: String,
}

/// A descriptor binding relocation for a compiled SPIR-V program.
#[derive(Clone, Copy, Debug)]
pub struct SpirvBindingRemap {
    /// Descriptor set index.
    pub group: u32,
    /// Binding number emitted by Slang.
    pub binding: u32,
    /// Binding number expected by the Vulkan pipeline layout.
    pub mapped_binding: u32,
}

/// Relocates descriptor bindings without changing code, types, or entry points.
///
/// Unlisted bindings remain unchanged. Only decorations are rewritten; the
/// caller must ensure the resulting bindings match the Vulkan pipeline layout.
pub fn remap_spirv_bindings(
    shader: &mut CompiledShader,
    mappings: &[SpirvBindingRemap],
) -> Result<(), SlangLoaderError> {
    let words = &mut shader.words;
    if words.len() < 5 || words[0] != 0x0723_0203 {
        return Err(SlangLoaderError::InvalidSpirV {
            path: shader.path.clone(),
        });
    }
    let mut groups = std::collections::HashMap::new();
    let mut bindings = Vec::new();
    let mut cursor = 5;
    while cursor < words.len() {
        let size = (words[cursor] >> 16) as usize;
        if size == 0 || cursor + size > words.len() {
            return Err(SlangLoaderError::InvalidSpirV {
                path: shader.path.clone(),
            });
        }
        // OpDecorate: DescriptorSet=34, Binding=33. Decoration order is free.
        if words[cursor] & 0xffff == 71 && size == 4 {
            match words[cursor + 2] {
                34 => {
                    groups.insert(words[cursor + 1], words[cursor + 3]);
                }
                33 => bindings.push((words[cursor + 1], cursor + 3)),
                _ => {}
            }
        }
        cursor += size;
    }
    for (id, index) in bindings {
        if let Some(group) = groups.get(&id)
            && let Some(mapping) = mappings
                .iter()
                .find(|m| m.group == *group && m.binding == words[index])
        {
            words[index] = mapping.mapped_binding;
        }
    }
    Ok(())
}

/// A shader stage passed to `slangc` when an entry point does not carry a
/// `[shader(...)]` attribute.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SlangStage {
    /// A vertex shader.
    Vertex,
    /// A fragment/pixel shader.
    #[serde(rename = "fragment")]
    Fragment,
    /// A compute shader.
    Compute,
    /// A mesh shader.
    Mesh,
    /// A task/amplification shader.
    #[serde(rename = "amplification")]
    Amplification,
    /// A ray-generation shader.
    #[serde(rename = "raygeneration")]
    RayGeneration,
    /// A ray-miss shader.
    #[serde(rename = "miss")]
    Miss,
    /// An any-hit shader.
    #[serde(rename = "anyhit")]
    AnyHit,
    /// A closest-hit shader.
    #[serde(rename = "closesthit")]
    ClosestHit,
    /// An intersection shader.
    Intersection,
    /// A callable shader.
    Callable,
}

impl SlangStage {
    fn as_str(self) -> &'static str {
        match self {
            Self::Vertex => "vertex",
            Self::Fragment => "fragment",
            Self::Compute => "compute",
            Self::Mesh => "mesh",
            Self::Amplification => "amplification",
            Self::RayGeneration => "raygeneration",
            Self::Miss => "miss",
            Self::AnyHit => "anyhit",
            Self::ClosestHit => "closesthit",
            Self::Intersection => "intersection",
            Self::Callable => "callable",
        }
    }
}

/// The default matrix memory layout passed to Slang.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum MatrixLayout {
    /// Use column-major layout.
    #[default]
    ColumnMajor,
    /// Use Slang's default row-major layout.
    RowMajor,
}

impl MatrixLayout {
    fn argument(self) -> &'static str {
        match self {
            Self::ColumnMajor => "-matrix-layout-column-major",
            Self::RowMajor => "-matrix-layout-row-major",
        }
    }
}

/// Settings for an individual shader compilation.
///
/// The settings are intentionally close to `slangc`'s command-line vocabulary.
/// A shader can normally rely on `[shader("vertex")]`-style attributes and leave
/// `entry_points` and `stage` empty.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct SlangSettings {
    /// An optional Slang profile such as `glsl_460+spirv_1_5`.
    pub profile: Option<String>,
    /// Entry points to pass as repeated `-entry` arguments.
    pub entry_points: Vec<String>,
    /// A stage to pass to `slangc` with `-stage`.
    pub stage: Option<SlangStage>,
    /// Include paths used for `#include` and `import` resolution.
    pub include_paths: Vec<PathBuf>,
    /// Preprocessor definitions, with or without a value (`NAME` or `NAME=1`).
    pub defines: Vec<String>,
    /// An optional Slang optimization level, for example `0` or `2`.
    pub optimization: Option<u8>,
    /// Emit debug information with `-g1`.
    pub debug_info: bool,
    /// Matrix layout used for generated code.
    pub matrix_layout: MatrixLayout,
    /// Preserve source entry-point names in SPIR-V.
    pub use_entrypoint_name: bool,
    /// Additional raw `slangc` arguments for features not represented above.
    pub extra_args: Vec<String>,
}

impl SlangSettings {
    /// Creates settings with the defaults used by the loader.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for SlangSettings {
    fn default() -> Self {
        Self {
            profile: None,
            entry_points: Vec::new(),
            stage: None,
            include_paths: Vec::new(),
            defines: Vec::new(),
            optimization: None,
            debug_info: false,
            matrix_layout: MatrixLayout::default(),
            use_entrypoint_name: true,
            extra_args: Vec::new(),
        }
    }
}

/// Configuration for the external `slangc` compiler.
#[derive(Clone, Debug)]
pub struct SlangCompiler {
    /// The executable to run. The default is `SLANGC` or `slangc` on `PATH`.
    pub executable: PathBuf,
    /// Directory containing source files. Relative include paths are
    /// interpreted relative to the process working directory.
    pub source_root: PathBuf,
    /// Additional include paths passed to every shader compilation.
    pub include_paths: Vec<PathBuf>,
}

impl Default for SlangCompiler {
    fn default() -> Self {
        let executable = std::env::var_os("SLANGC")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("slangc"));

        Self {
            executable,
            source_root: PathBuf::from("assets"),
            include_paths: Vec::new(),
        }
    }
}

impl SlangCompiler {
    /// Compiles an embedded shader tree, preserving relative includes/imports.
    ///
    /// Each `(path, source)` must have a relative path without parent traversal.
    /// No application assets are written. The temporary tree is removed when
    /// compilation finishes. Settings and configured include paths still apply.
    pub fn compile_bundle(
        &self,
        entry: impl AsRef<Path>,
        sources: &[(&str, &str)],
        settings: &SlangSettings,
    ) -> Result<CompiledShader, SlangLoaderError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let tree = tempfile::tempdir()?;
            let validate = |path: &Path| {
                !path.as_os_str().is_empty()
                    && path.components().all(|component| {
                        matches!(
                            component,
                            std::path::Component::Normal(_) | std::path::Component::CurDir
                        )
                    })
            };
            let entry = entry.as_ref();
            if !validate(entry) || sources.iter().any(|(path, _)| !validate(Path::new(path))) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Slang bundle paths must be relative without parent traversal",
                )
                .into());
            }
            for (path, source) in sources {
                let target = tree.path().join(path);
                if let Some(parent) = target.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(target, source)?;
            }
            let source = std::fs::read(tree.path().join(entry))?;
            self.clone()
                .with_source_root(tree.path())
                .compile_source(entry, source, settings)
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (entry, sources, settings);
            Err(SlangLoaderError::UnsupportedTarget)
        }
    }

    /// Compiles embedded or generated source synchronously to SPIR-V.
    ///
    /// `path` is the logical source filename. Its parent under `source_root` is
    /// searched for relative imports even when the supplied source is not on
    /// disk. Call once during pipeline setup, or from a task pool for hot reload;
    /// this method launches the external compiler and blocks until it exits.
    pub fn compile_source(
        &self,
        path: impl AsRef<Path>,
        source: impl AsRef<[u8]>,
        settings: &SlangSettings,
    ) -> Result<CompiledShader, SlangLoaderError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let path = path.as_ref();
            compile_shader(self, settings, path, &normalize_path(path), source.as_ref())
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (path, source, settings);
            Err(SlangLoaderError::UnsupportedTarget)
        }
    }

    /// Compiles a file relative to `source_root` (or an absolute file path).
    pub fn compile_file(
        &self,
        path: impl AsRef<Path>,
        settings: &SlangSettings,
    ) -> Result<CompiledShader, SlangLoaderError> {
        let path = path.as_ref();
        let source = std::fs::read(self.source_root.join(path))?;
        self.compile_source(path, source, settings)
    }

    /// Creates a compiler configuration for an explicit executable.
    pub fn new(executable: impl Into<PathBuf>) -> Self {
        Self {
            executable: executable.into(),
            ..Self::default()
        }
    }

    /// Sets the directory used to resolve source paths and sibling imports.
    pub fn with_source_root(mut self, source_root: impl Into<PathBuf>) -> Self {
        self.source_root = source_root.into();
        self
    }

    /// Adds an include path shared by all loaded shaders.
    pub fn with_include_path(mut self, include_path: impl Into<PathBuf>) -> Self {
        self.include_paths.push(include_path.into());
        self
    }

    /// Builds the arguments passed to `slangc`.
    ///
    /// The input and output paths are supplied by the caller, so
    /// callers can use this method to inspect or reproduce a failing invocation.
    pub fn build_args(
        &self,
        input_path: &Path,
        output_path: &Path,
        settings: &SlangSettings,
    ) -> Vec<OsString> {
        let mut args = vec![
            OsString::from("-lang"),
            OsString::from("slang"),
            OsString::from("-target"),
            OsString::from("spirv"),
        ];

        if let Some(profile) = &settings.profile {
            args.push(OsString::from("-profile"));
            args.push(OsString::from(profile));
        }

        if let Some(stage) = settings.stage {
            args.push(OsString::from("-stage"));
            args.push(OsString::from(stage.as_str()));
        }

        for entry_point in &settings.entry_points {
            args.push(OsString::from("-entry"));
            args.push(OsString::from(entry_point));
        }

        for include_path in self.include_paths.iter().chain(&settings.include_paths) {
            args.push(OsString::from("-I"));
            args.push(include_path.as_os_str().to_owned());
        }

        for define in &settings.defines {
            args.push(OsString::from("-D"));
            args.push(OsString::from(define));
        }

        args.push(OsString::from(settings.matrix_layout.argument()));

        if settings.use_entrypoint_name {
            args.push(OsString::from("-fvk-use-entrypoint-name"));
        }

        if let Some(level) = settings.optimization {
            args.push(OsString::from(format!("-O{level}")));
        }

        if settings.debug_info {
            args.push(OsString::from("-g1"));
        }

        args.extend(settings.extra_args.iter().map(OsString::from));
        args.push(OsString::from("-o"));
        args.push(output_path.as_os_str().to_owned());
        args.push(input_path.as_os_str().to_owned());
        args
    }
}

/// Errors produced while compiling or loading a Slang shader.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SlangLoaderError {
    /// The source or compiler output could not be read or written.
    #[error("Slang shader I/O failed: {0}")]
    Io(#[from] std::io::Error),
    /// The compiler executable could not be launched.
    #[error(
        "could not launch slangc at {executable:?}: {source}; install slangc and put it on PATH or set SLANGC to its executable path"
    )]
    Launch {
        /// The configured compiler executable.
        executable: PathBuf,
        /// The operating-system error.
        #[source]
        source: std::io::Error,
    },
    /// `slangc` rejected the shader source or compiler arguments.
    #[error("slangc failed for {path} with status {status:?}:\n{diagnostics}")]
    Compilation {
        /// The logical source path being compiled.
        path: String,
        /// The process exit code, if one was available.
        status: Option<i32>,
        /// Combined compiler output.
        diagnostics: String,
    },
    /// The compiler exited successfully but did not create its output file.
    #[error("slangc did not create output for {path}: {output:?}")]
    MissingOutput {
        /// The logical source path being compiled.
        path: String,
        /// The expected output path.
        output: PathBuf,
    },
    /// The SPIR-V output did not begin with the SPIR-V magic number.
    #[error("slangc produced invalid SPIR-V for {path}")]
    InvalidSpirV {
        /// The logical source path being compiled.
        path: String,
    },
    /// Runtime compiler invocation is unavailable on WebAssembly.
    #[error(
        "runtime Slang compilation is not supported on WebAssembly; compile this shader on a native host"
    )]
    UnsupportedTarget,
}

fn normalize_path(path: &Path) -> String {
    path.to_string_lossy()
        .replace(std::path::MAIN_SEPARATOR, "/")
}

#[cfg(not(target_arch = "wasm32"))]
fn compile_shader(
    compiler: &SlangCompiler,
    settings: &SlangSettings,
    asset_path: &Path,
    display_path: &str,
    source: &[u8],
) -> Result<CompiledShader, SlangLoaderError> {
    use std::{fs, process::Command};

    let working_directory = std::env::current_dir()?;
    let source_root = if compiler.source_root.is_absolute() {
        compiler.source_root.clone()
    } else {
        working_directory.join(&compiler.source_root)
    };

    let temporary_directory = tempfile::tempdir()?;
    let source_candidate = source_root.join(asset_path);
    let input_path = if fs::read(&source_candidate)
        .map(|on_disk_source| on_disk_source == source)
        .unwrap_or(false)
    {
        source_candidate.clone()
    } else {
        let file_name = asset_path
            .file_name()
            .unwrap_or_else(|| std::ffi::OsStr::new("shader.slang"));
        let temporary_source = temporary_directory.path().join(file_name);
        fs::write(&temporary_source, source)?;
        temporary_source
    };

    let output_path = temporary_directory.path().join("output.spv");
    let mut args = compiler.build_args(&input_path, &output_path, settings);
    // Preserve sibling include resolution for embedded/generated sources and
    // custom asset readers whose bytes differ from a file on disk.
    if let Some(parent) = source_candidate.parent() {
        args.insert(0, parent.as_os_str().to_owned());
        args.insert(0, OsString::from("-I"));
    }

    let mut command = Command::new(&compiler.executable);
    command.args(&args);
    // Keep the process working directory stable. The source path is absolute,
    // so this makes relative include paths match the app's normal working
    // directory rather than unexpectedly becoming relative to `assets/`.
    command.current_dir(&working_directory);

    let output = command
        .output()
        .map_err(|source| SlangLoaderError::Launch {
            executable: compiler.executable.clone(),
            source,
        })?;

    let diagnostics = command_output(&output.stdout, &output.stderr);
    if !output.status.success() {
        return Err(SlangLoaderError::Compilation {
            path: display_path.to_owned(),
            status: output.status.code(),
            diagnostics,
        });
    }

    let output_bytes = fs::read(&output_path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            SlangLoaderError::MissingOutput {
                path: display_path.to_owned(),
                output: output_path.clone(),
            }
        } else {
            SlangLoaderError::Io(error)
        }
    })?;

    if !is_spirv(&output_bytes) {
        return Err(SlangLoaderError::InvalidSpirV {
            path: display_path.to_owned(),
        });
    }
    let words = output_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect();
    Ok(CompiledShader {
        path: display_path.to_owned(),
        words,
        diagnostics,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn command_output(stdout: &[u8], stderr: &[u8]) -> String {
    let stdout = String::from_utf8_lossy(stdout);
    let stderr = String::from_utf8_lossy(stderr);
    match (stdout.is_empty(), stderr.is_empty()) {
        (true, true) => String::new(),
        (false, true) => stdout.into_owned(),
        (true, false) => stderr.into_owned(),
        (false, false) => format!("{stdout}\n{stderr}"),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn is_spirv(bytes: &[u8]) -> bool {
    bytes.len() >= 20 && bytes.len().is_multiple_of(4) && bytes[..4] == 0x0723_0203u32.to_le_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_binding_relocation_is_scoped_to_its_descriptor_set() {
        let words = [
            0x0723_0203u32,
            0x0001_0500,
            0,
            3,
            0,
            (4 << 16) | 71,
            1,
            33,
            27,
            (4 << 16) | 71,
            2,
            33,
            27,
            (4 << 16) | 71,
            2,
            34,
            1,
            (4 << 16) | 71,
            1,
            34,
            0,
        ];
        let mut shader = CompiledShader {
            words: words.to_vec(),
            path: "test".into(),
            diagnostics: String::new(),
        };
        remap_spirv_bindings(
            &mut shader,
            &[SpirvBindingRemap {
                group: 0,
                binding: 27,
                mapped_binding: 24,
            }],
        )
        .unwrap();
        assert_eq!(shader.words[8], 24);
        assert_eq!(shader.words[12], 27);
        shader.words = vec![0; 5];
        assert!(remap_spirv_bindings(&mut shader, &[]).is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn embedded_bundle_rejects_paths_outside_its_tree() {
        let compiler = SlangCompiler::default();
        for invalid in ["../escape.slang", "/escape.slang"] {
            assert!(
                compiler
                    .compile_bundle(invalid, &[], &SlangSettings::default())
                    .is_err()
            );
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn missing_compiler_is_reported() {
        let compiler =
            SlangCompiler::new(tempfile::tempdir().unwrap().path().join("missing-slangc"));
        let error = compiler
            .compile_source("test.slang", "", &SlangSettings::default())
            .unwrap_err();
        assert!(matches!(error, SlangLoaderError::Launch { .. }));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    #[ignore = "requires slangc on PATH or SLANGC"]
    fn embedded_bundle_preserves_includes_named_entries_and_diagnostics() {
        let compiler = SlangCompiler::default();
        let settings = SlangSettings {
            optimization: Some(2),
            ..SlangSettings::default()
        };
        let shader = compiler
            .compile_bundle(
                "shaders/main.slang",
                &[
                    ("shaders/constants.slang", "static const uint VALUE = 42u;"),
                    (
                        "shaders/main.slang",
                        r#"
                #include "constants.slang"
                [[vk::binding(0, 0)]] RWStructuredBuffer<uint> output;
                [shader("compute")][numthreads(1,1,1)] void first() { output[0] = VALUE; }
                [shader("compute")][numthreads(1,1,1)] void second() { output[1] = VALUE; }
            "#,
                    ),
                ],
                &settings,
            )
            .unwrap();
        let words = shader.words;
        let mut cursor = 5;
        let mut entries = 0;
        while cursor < words.len() {
            if words[cursor] & 0xffff == 15 {
                entries += 1;
            }
            cursor += (words[cursor] >> 16) as usize;
        }
        assert_eq!(entries, 2);
        assert!(
            matches!(compiler.compile_source("bad.slang", "this is invalid", &settings),
            Err(SlangLoaderError::Compilation { diagnostics, .. }) if !diagnostics.is_empty())
        );
    }
}
