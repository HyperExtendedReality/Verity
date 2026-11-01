//! The Split Radiance Cascades frame as a Hyperon render extension.
//!
//! Each frame, after the renderer's resolve: seed receivers from the
//! G-buffer, rank and build the probe hierarchy, trace interval rays through
//! the GI scene, merge them into the world cache, pack directional probes,
//! resolve per-pixel irradiance, filter it temporally and spatially, and add
//! diffuse GI times albedo to the HDR target. Bindings, sizes and pass order
//! follow `legacy/src/render.rs`.

// Every function here is Vulkan calls; an unsafe fn body is one unsafe context.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::{
    scene::{GiScene, LAYER_MASK},
    settings::*,
};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use gpu_allocator::MemoryLocation;
use hyperon_render::{
    FrameContext, Gpu, RenderExtension, Targets,
    asset::SceneData,
    extension::Result,
    gpu::{self, Buffer, Image},
};
use std::{collections::HashMap, ffi::CString, mem, time::Instant};

const SLOTS: usize = 2;
const SOURCE_BACKGROUND: u32 = 1;
const SOURCE_PUNCTUAL: u32 = 4;

macro_rules! spirv {
    ($file:literal, $entry:expr) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $file, ".", $entry, ".spv"))
    };
}

/// Every cascades.slang pass this frame dispatches, with its SPIR-V.
const PASSES: &[(&str, &[u8])] = &[
    ("clear", spirv!("cascades", "clear")),
    ("seed", spirv!("cascades", "seed")),
    ("parents", spirv!("cascades", "parents")),
    ("rank", spirv!("cascades", "rank")),
    ("prefix", spirv!("cascades", "prefix")),
    (
        "generateDirections",
        spirv!("cascades", "generateDirections"),
    ),
    ("retain", spirv!("cascades", "retain")),
    ("trace", spirv!("cascades", "trace")),
    ("reduce", spirv!("cascades", "reduce")),
    ("merge", spirv!("cascades", "merge")),
    ("seedProbes", spirv!("cascades", "seedProbes")),
    ("packProbes", spirv!("cascades", "packProbes")),
    ("temporal", spirv!("cascades", "temporal")),
    ("denoise", spirv!("cascades", "denoise")),
    ("worldResolve", spirv!("cascades", "worldResolve")),
    ("traceDeferred", spirv!("cascades", "traceDeferred")),
    ("traceRecord", spirv!("cascades", "traceRecord")),
    ("traceVisibility", spirv!("cascades", "traceVisibility")),
    ("traceResolve", spirv!("cascades", "traceResolve")),
    ("compositeAnchors", spirv!("composite", "compositeAnchors")),
    ("compositeCompute", spirv!("composite", "compositeCompute")),
];

use vk::DescriptorType as D;
/// The cascades' set 0, as `compute_layout` in the legacy renderer (minus the
/// atlas, the DFG texture and the hardware TLAS).
const MAIN_BINDINGS: &[(u32, D)] = &[
    (0, D::UNIFORM_BUFFER),
    (1, D::STORAGE_BUFFER),
    (2, D::STORAGE_BUFFER),
    (3, D::STORAGE_BUFFER),
    (4, D::SAMPLED_IMAGE),
    (6, D::STORAGE_BUFFER),
    (7, D::STORAGE_BUFFER),
    (8, D::STORAGE_BUFFER),
    (9, D::STORAGE_BUFFER),
    (10, D::STORAGE_IMAGE),
    (11, D::SAMPLED_IMAGE),
    (12, D::STORAGE_BUFFER),
    (13, D::SAMPLED_IMAGE),
    (14, D::STORAGE_IMAGE),
    (15, D::SAMPLED_IMAGE),
    (20, D::SAMPLED_IMAGE),
    (22, D::SAMPLED_IMAGE),
    (23, D::STORAGE_IMAGE),
    (25, D::SAMPLED_IMAGE),
    (26, D::SAMPLED_IMAGE),
    (27, D::SAMPLED_IMAGE),
    (28, D::SAMPLED_IMAGE),
    (29, D::SAMPLED_IMAGE),
    (38, D::STORAGE_IMAGE),
    (39, D::SAMPLED_IMAGE),
    (40, D::SAMPLED_IMAGE),
    (42, D::STORAGE_BUFFER),
    (43, D::STORAGE_BUFFER),
    (44, D::STORAGE_BUFFER),
    (45, D::STORAGE_BUFFER),
    (46, D::STORAGE_BUFFER),
    (47, D::SAMPLED_IMAGE),
];
/// composite.slang's set 0, both passes.
const COMPOSITE_BINDINGS: &[(u32, D)] = &[
    (0, D::UNIFORM_BUFFER),
    (2, D::SAMPLED_IMAGE),
    (3, D::SAMPLED_IMAGE),
    (5, D::SAMPLED_IMAGE),
    (6, D::SAMPLED_IMAGE),
    (7, D::SAMPLED_IMAGE),
    (8, D::SAMPLED_IMAGE),
    (9, D::STORAGE_BUFFER),
    (10, D::STORAGE_BUFFER),
    (11, D::STORAGE_BUFFER),
    (12, D::SAMPLED_IMAGE),
    (13, D::SAMPLED_IMAGE),
    (14, D::STORAGE_IMAGE),
    (15, D::STORAGE_IMAGE),
    (16, D::SAMPLED_IMAGE),
    (17, D::STORAGE_BUFFER),
    (19, D::STORAGE_IMAGE),
];
/// Descriptor sets per slot: compute per level, two clears, temporal, world
/// resolve, denoise per level, then the composite.
const MAIN_SETS: usize = LEVELS + 2 + 2 + LEVELS;

/// `GiLight` in common.slang.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct GiLight {
    /// w = 0: direction toward a directional light; w > 0: position and range.
    position: [f32; 4],
    color: [f32; 4],
    direction: [f32; 4],
    edge1: [f32; 4],
    edge2: [f32; 4],
    source: [u32; 4],
}

/// What a descriptor points at.
#[derive(Clone, Copy)]
enum Res {
    Buffer(vk::Buffer),
    /// A view in `GENERAL` layout, or a depth view in `DEPTH_READ_ONLY_OPTIMAL`.
    View(vk::ImageView, vk::ImageLayout),
}

struct Pipelines {
    set_layout: vk::DescriptorSetLayout,
    composite_set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    composite_layout: vk::PipelineLayout,
    passes: HashMap<&'static str, vk::Pipeline>,
}

/// Everything sized to the render targets.
struct View {
    grid: [u32; 2],
    shading: [u32; 2],
    buckets: u32,
    heads: [Buffer; 2],
    entries: Buffer,
    history: Buffer,
    probes: Buffer,
    /// RTGI buffers the shaders declare; one placeholder each.
    rtgi: [Buffer; 5],
    irradiance: [Image; 4],
    /// 0/1 temporal moments, 2 sparse metadata, 3/4 receiver history.
    metadata: [Image; 5],
    background: [Image; 3],
    denoise_history: Image,
    anchors: [Image; 2],
    /// 1x1 stand-ins: environment, SSAO, sun shadows, RTGI lit history.
    placeholder: Image,
    /// 1x1x1 stand-in for the scene and geometry page arrays.
    pages: Image,
    pages_view: vk::ImageView,
    uniforms: [[Buffer; LEVELS + 2]; SLOTS],
    lights: [Buffer; SLOTS],
    pool: vk::DescriptorPool,
    sets: [[vk::DescriptorSet; MAIN_SETS]; SLOTS],
    composite_sets: [vk::DescriptorSet; SLOTS],
    /// Images and buffers still need clearing to zero.
    fresh: bool,
}

/// Where GI is in getting ready to render.
#[derive(Clone, Debug, PartialEq)]
pub enum GiStatus {
    /// The driver is compiling the transport pipelines, in the background.
    Compiling {
        since: Instant,
    },
    Ready {
        compile_time: std::time::Duration,
    },
    Failed(String),
}

type CompiledPasses = std::result::Result<Vec<(&'static str, vk::Pipeline)>, String>;

/// Where compiled pipelines persist between runs; drivers reject stale data.
fn pipeline_cache_path() -> std::path::PathBuf {
    std::env::temp_dir().join("hyperon-lux-pipelines.bin")
}

/// Compile every pass in parallel through a disk-backed pipeline cache. The
/// transport shaders are large: a cold driver compile takes minutes on one core.
fn compile_passes(
    device: ash::Device,
    layout: vk::PipelineLayout,
    composite_layout: vk::PipelineLayout,
) -> CompiledPasses {
    let initial = std::fs::read(pipeline_cache_path()).unwrap_or_default();
    let cache = unsafe {
        device.create_pipeline_cache(
            &vk::PipelineCacheCreateInfo::default().initial_data(&initial),
            None,
        )
    }
    .or_else(|_| unsafe {
        device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
    })
    .map_err(|e| e.to_string())?;
    let results: Vec<_> = std::thread::scope(|scope| {
        let device = &device;
        let handles: Vec<_> = PASSES
            .iter()
            .map(|&(name, spirv)| {
                scope.spawn(move || {
                    let layout = if name.starts_with("composite") {
                        composite_layout
                    } else {
                        layout
                    };
                    let entry = CString::new(name).map_err(|e| e.to_string())?;
                    // SAFETY: the SPIR-V was validated at build time; the layout outlives the pipeline.
                    unsafe { gpu::compute_pipeline(device, cache, layout, spirv, &entry) }
                        .map(|pipeline| (name, pipeline))
                        .map_err(|e| format!("{name}: {e}"))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err("compile thread panicked".into()))
            })
            .collect()
    });
    unsafe {
        if let Ok(data) = device.get_pipeline_cache_data(cache) {
            // A cache that fails to save only costs the next run a cold compile.
            let _ = std::fs::write(pipeline_cache_path(), data);
        }
        device.destroy_pipeline_cache(cache, None);
    }
    let (ok, failed): (Vec<_>, Vec<_>) = results.into_iter().partition(Result::is_ok);
    let ok: Vec<_> = ok.into_iter().map(Result::unwrap).collect();
    if let Some(Err(error)) = failed.into_iter().next() {
        for (_, pipeline) in ok {
            unsafe { device.destroy_pipeline(pipeline, None) };
        }
        return Err(error);
    }
    Ok(ok)
}

/// Split Radiance Cascades GI. Install with `Renderer::set_extension` before
/// loading scenes. Pipelines compile in the background; until they are ready
/// the renderer keeps its own ambient lighting.
pub struct Lux {
    pub settings: Settings,
    scene: GiScene,
    geometry: Buffer,
    words: Buffer,
    instances: u32,
    pipelines: Option<Pipelines>,
    compile: Option<std::thread::JoinHandle<CompiledPasses>>,
    status: GiStatus,
    view: Option<View>,
    targets: Option<Targets>,
    command_pool: vk::CommandPool,
    schedule: CandidateSchedule,
    history_valid: bool,
    scene_changed: bool,
    previous_clip: Option<Mat4>,
    previous_lights: Vec<GiLight>,
    started: Instant,
    /// Debugging aid: `HYPERON_LUX_PASS_LIMIT=n` dispatches only the first
    /// `n` passes of each frame, to bisect GPU faults.
    pass_limit: Option<usize>,
}

impl Lux {
    pub fn new(settings: Settings) -> Self {
        Self {
            settings,
            scene: GiScene::default(),
            geometry: Buffer::default(),
            words: Buffer::default(),
            instances: 0,
            pipelines: None,
            compile: None,
            status: GiStatus::Compiling {
                since: Instant::now(),
            },
            view: None,
            targets: None,
            command_pool: vk::CommandPool::null(),
            schedule: CandidateSchedule::default(),
            history_valid: false,
            scene_changed: true,
            previous_clip: None,
            previous_lights: Vec::new(),
            started: Instant::now(),
            pass_limit: std::env::var("HYPERON_LUX_PASS_LIMIT")
                .ok()
                .and_then(|v| v.parse().ok()),
        }
    }

    unsafe fn pipelines(gpu: &mut Gpu) -> Result<Pipelines> {
        let device = gpu.device;
        let set_layout = |bindings: &[(u32, D)]| {
            let bindings: Vec<_> = bindings
                .iter()
                .map(|&(binding, kind)| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(binding)
                        .descriptor_type(kind)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        };
        let main = set_layout(MAIN_BINDINGS)?;
        let composite = set_layout(COMPOSITE_BINDINGS)?;
        // Set 1 is the renderer's frame set: materials, textures, DFG table.
        let layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&[main, gpu.frame_set_layout]),
            None,
        )?;
        let composite_layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&[composite, gpu.frame_set_layout]),
            None,
        )?;
        Ok(Pipelines {
            set_layout: main,
            composite_set_layout: composite,
            layout,
            composite_layout,
            passes: HashMap::new(),
        })
    }

    /// Where GI is in getting ready to render.
    pub fn status(&self) -> &GiStatus {
        &self.status
    }

    /// Take the background compile's pipelines once it finishes.
    fn poll_compile(&mut self) {
        if !self.compile.as_ref().is_some_and(|c| c.is_finished()) {
            return;
        }
        let result = self
            .compile
            .take()
            .unwrap()
            .join()
            .unwrap_or_else(|_| Err("compile thread panicked".into()));
        let since = match self.status {
            GiStatus::Compiling { since } => since,
            _ => Instant::now(),
        };
        match (result, self.pipelines.as_mut()) {
            (Ok(passes), Some(pipelines)) => {
                pipelines.passes.extend(passes);
                self.status = GiStatus::Ready {
                    compile_time: since.elapsed(),
                };
            }
            (Ok(_), None) => {}
            (Err(error), _) => self.status = GiStatus::Failed(error),
        }
    }

    unsafe fn destroy_pipelines(device: &ash::Device, pipelines: Pipelines) {
        for pipeline in pipelines.passes.into_values() {
            device.destroy_pipeline(pipeline, None);
        }
        device.destroy_pipeline_layout(pipelines.layout, None);
        device.destroy_pipeline_layout(pipelines.composite_layout, None);
        device.destroy_descriptor_set_layout(pipelines.set_layout, None);
        device.destroy_descriptor_set_layout(pipelines.composite_set_layout, None);
    }

    /// Replace a device-local buffer's contents through a staging copy.
    unsafe fn upload(
        &mut self,
        gpu: &mut Gpu,
        target: &mut Buffer,
        name: &str,
        bytes: &[u8],
    ) -> Result<()> {
        let device = gpu.device;
        if self.command_pool == vk::CommandPool::null() {
            self.command_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(gpu.queue_family)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                None,
            )?;
        }
        let size = bytes.len() as u64;
        let mut staging = gpu::create_buffer(
            device,
            gpu.allocator,
            "gi staging",
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        )?;
        staging.write(bytes);
        gpu::destroy_buffer(device, gpu.allocator, mem::take(target));
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        *target = gpu::create_buffer(
            device,
            gpu.allocator,
            name,
            size,
            usage,
            MemoryLocation::GpuOnly,
        )?;
        let cmd = device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0];
        device.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        device.cmd_copy_buffer(
            cmd,
            staging.buffer,
            target.buffer,
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size,
            }],
        );
        device.end_command_buffer(cmd)?;
        let commands = [cmd];
        let result = device
            .queue_submit(
                gpu.queue,
                &[vk::SubmitInfo::default().command_buffers(&commands)],
                vk::Fence::null(),
            )
            .and_then(|()| device.queue_wait_idle(gpu.queue));
        device.free_command_buffers(self.command_pool, &commands);
        gpu::destroy_buffer(device, gpu.allocator, staging);
        Ok(result?)
    }

    unsafe fn create_view(&self, gpu: &mut Gpu, targets: &Targets) -> Result<View> {
        let device = gpu.device;
        let settings = &self.settings;
        let extent = [targets.extent.width, targets.extent.height];
        let grid = resolution(extent, settings);
        let shading = resolve_size(extent, settings);
        let samples = u64::from(grid[0]) * u64::from(grid[1]);
        let buckets = cache_bucket_count(samples, settings);
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let mut buffer = |name: &str, size: u64, usage, location| {
            gpu::create_buffer(device, gpu.allocator, name, size, usage, location)
        };
        let heads = [
            buffer(
                "gi heads 0",
                cache_head_bytes(buckets, samples),
                storage,
                MemoryLocation::GpuOnly,
            )?,
            buffer(
                "gi heads 1",
                cache_head_bytes(buckets, samples),
                storage,
                MemoryLocation::GpuOnly,
            )?,
        ];
        let entries = buffer(
            "gi entries",
            cache_entry_bytes(samples, buckets),
            storage,
            MemoryLocation::GpuOnly,
        )?;
        let history = buffer(
            "gi history",
            cache_history_bytes(buckets),
            storage,
            MemoryLocation::GpuOnly,
        )?;
        let probes = buffer(
            "gi probes",
            irradiance_probe_bytes(samples),
            storage,
            MemoryLocation::GpuOnly,
        )?;
        let rtgi =
            [(); 5].map(|()| buffer("gi rtgi placeholder", 256, storage, MemoryLocation::GpuOnly));
        let rtgi = rtgi
            .into_iter()
            .collect::<std::result::Result<Vec<_>, _>>()?
            .try_into()
            .ok()
            .unwrap();
        let uniform = vk::BufferUsageFlags::UNIFORM_BUFFER;
        let mut uniforms = Vec::new();
        for _ in 0..SLOTS {
            let slot: Vec<_> = (0..LEVELS + 2)
                .map(|_| {
                    buffer(
                        "gi parameters",
                        mem::size_of::<Parameters>() as u64,
                        uniform,
                        MemoryLocation::CpuToGpu,
                    )
                })
                .collect::<std::result::Result<_, _>>()?;
            uniforms.push(slot.try_into().ok().unwrap());
        }
        let light_bytes =
            (hyperon_render::MAX_LIGHTS as u64 + 2) * mem::size_of::<GiLight>() as u64;
        let lights = [
            buffer(
                "gi lights",
                light_bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::CpuToGpu,
            )?,
            buffer(
                "gi lights",
                light_bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::CpuToGpu,
            )?,
        ];

        use vk::ImageUsageFlags as U;
        let usage = U::STORAGE | U::SAMPLED | U::TRANSFER_SRC | U::TRANSFER_DST;
        let color = vk::ImageAspectFlags::COLOR;
        let rgba32f = vk::Format::R32G32B32A32_SFLOAT;
        let shading_extent = vk::Extent2D {
            width: shading[0],
            height: shading[1],
        };
        let mut image = |name: &str, format, extent| {
            gpu::create_image(device, gpu.allocator, name, format, extent, 1, usage, color)
        };
        let mut images = |name: &str, count: usize| -> Result<Vec<Image>> {
            (0..count)
                .map(|_| image(name, rgba32f, shading_extent))
                .collect()
        };
        let irradiance = images("gi irradiance", 4)?.try_into().ok().unwrap();
        let metadata = images("gi metadata", 5)?.try_into().ok().unwrap();
        let background = images("gi background", 3)?.try_into().ok().unwrap();
        let denoise_history = images("gi denoise history", 1)?.pop().unwrap();
        let anchors = images("gi anchors", 2)?.try_into().ok().unwrap();
        let one = vk::Extent2D {
            width: 1,
            height: 1,
        };
        let placeholder = image("gi placeholder", rgba32f, one)?;
        let pages = image("gi pages", vk::Format::R32G32B32A32_UINT, one)?;
        let pages_view = device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(pages.image)
                .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
                .format(vk::Format::R32G32B32A32_UINT)
                .subresource_range(gpu::subresource(color)),
            None,
        )?;

        let count =
            |list: &[(u32, D)], kind: D| list.iter().filter(|(_, k)| *k == kind).count() as u32;
        let sizes = [
            D::UNIFORM_BUFFER,
            D::STORAGE_BUFFER,
            D::SAMPLED_IMAGE,
            D::STORAGE_IMAGE,
        ]
        .map(|kind| {
            vk::DescriptorPoolSize::default().ty(kind).descriptor_count(
                SLOTS as u32
                    * (MAIN_SETS as u32 * count(MAIN_BINDINGS, kind)
                        + count(COMPOSITE_BINDINGS, kind)),
            )
        });
        let pool = device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets((SLOTS * (MAIN_SETS + 1)) as u32)
                .pool_sizes(&sizes),
            None,
        )?;
        let pipelines = self.pipelines.as_ref().unwrap();
        let mut sets = [[vk::DescriptorSet::null(); MAIN_SETS]; SLOTS];
        let mut composite_sets = [vk::DescriptorSet::null(); SLOTS];
        for slot in 0..SLOTS {
            let allocated = device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&[pipelines.set_layout; MAIN_SETS]),
            )?;
            sets[slot].copy_from_slice(&allocated);
            composite_sets[slot] = device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&[pipelines.composite_set_layout]),
            )?[0];
        }
        let view = View {
            grid,
            shading,
            buckets: buckets as u32,
            heads,
            entries,
            history,
            probes,
            rtgi,
            irradiance,
            metadata,
            background,
            denoise_history,
            anchors,
            placeholder,
            pages,
            pages_view,
            uniforms: uniforms.try_into().ok().unwrap(),
            lights,
            pool,
            sets,
            composite_sets,
            fresh: true,
        };
        Ok(view)
    }

    /// Point every set at this view's resources, the scene and the targets.
    unsafe fn write_sets(&self, device: &ash::Device, targets: &Targets) {
        let view = self.view.as_ref().unwrap();
        let general = vk::ImageLayout::GENERAL;
        let image = |image: &Image| Res::View(image.view, general);
        let depth = Res::View(targets.depth, vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL);
        let placeholder = image(&view.placeholder);
        for slot in 0..SLOTS {
            let (p, q) = (slot, 1 - slot);
            // `bind_compute` in the legacy renderer: (uniform, output, input,
            // sample input, metadata output, metadata input, receiver history
            // output, receiver history input, background output, background sample).
            let variant = |uniform: usize, views: [&Image; 9]| -> Vec<(u32, Res)> {
                let [
                    output,
                    input,
                    sample,
                    metadata_out,
                    metadata_in,
                    history_out,
                    history_in,
                    background_out,
                    background_sample,
                ] = views;
                vec![
                    (0, Res::Buffer(view.uniforms[slot][uniform].buffer)),
                    (1, Res::Buffer(self.geometry.buffer)),
                    (2, Res::Buffer(self.words.buffer)),
                    (3, Res::Buffer(view.lights[slot].buffer)),
                    (4, depth),
                    (6, Res::Buffer(view.heads[p].buffer)),
                    (7, Res::Buffer(view.entries.buffer)),
                    (8, Res::Buffer(view.heads[q].buffer)),
                    (9, Res::Buffer(view.history.buffer)),
                    (10, image(output)),
                    (11, image(input)),
                    (12, Res::Buffer(view.probes.buffer)),
                    (13, image(sample)),
                    (14, image(metadata_out)),
                    (15, image(metadata_in)),
                    (20, Res::View(targets.motion, general)),
                    (22, placeholder),
                    (23, image(history_out)),
                    (25, image(history_in)),
                    (26, Res::View(targets.receiver, general)),
                    (27, Res::View(targets.gbuffer, general)),
                    (28, Res::View(view.pages_view, general)),
                    (29, Res::View(view.pages_view, general)),
                    (38, image(background_out)),
                    (39, image(background_sample)),
                    (40, image(&view.background[1 + q])),
                    (42, Res::Buffer(view.rtgi[0].buffer)),
                    (43, Res::Buffer(view.rtgi[1].buffer)),
                    (44, Res::Buffer(view.rtgi[2].buffer)),
                    (45, Res::Buffer(view.rtgi[3].buffer)),
                    (46, Res::Buffer(view.rtgi[4].buffer)),
                    (47, placeholder),
                ]
            };
            let (irr, meta, bg) = (&view.irradiance, &view.metadata, &view.background);
            let standard = |uniform| {
                variant(
                    uniform,
                    [
                        &irr[p],
                        &irr[q],
                        &irr[q],
                        &meta[p],
                        &meta[q],
                        &meta[3 + p],
                        &meta[3 + q],
                        &bg[1 + p],
                        &bg[0],
                    ],
                )
            };
            let mut contents: Vec<Vec<(u32, Res)>> = (0..LEVELS).map(standard).collect();
            contents.push(standard(LEVELS));
            contents.push(standard(LEVELS + 1));
            contents.push(variant(
                0,
                [
                    &irr[2],
                    &irr[q],
                    &irr[3],
                    &meta[p],
                    &meta[q],
                    &meta[3 + p],
                    &meta[3 + q],
                    &bg[1 + p],
                    &bg[0],
                ],
            ));
            contents.push(variant(
                0,
                [
                    &irr[3],
                    &irr[q],
                    &irr[q],
                    &meta[p],
                    &meta[q],
                    &meta[3 + p],
                    &meta[3 + q],
                    &bg[0],
                    &bg[1 + p],
                ],
            ));
            for level in 0..LEVELS {
                let (input, output) = (&irr[2 + level % 2], &irr[2 + (level + 1) % 2]);
                contents.push(variant(
                    level,
                    [
                        output,
                        input,
                        &meta[p],
                        &meta[q],
                        &meta[p],
                        &view.denoise_history,
                        &meta[3 + p],
                        &bg[1 + p],
                        &bg[0],
                    ],
                ));
            }
            for (set, entries) in view.sets[slot].iter().zip(&contents) {
                write_set(device, *set, MAIN_BINDINGS, entries);
            }
            let display = &irr[composite_display_index(self.settings.spatial_filter_steps, p)];
            let display_history = if self.settings.spatial_filter_steps == 0 {
                &meta[3 + p]
            } else {
                &view.denoise_history
            };
            let composite = [
                (0, Res::Buffer(view.uniforms[slot][0].buffer)),
                (2, image(display)),
                (3, depth),
                (5, Res::View(targets.gbuffer, general)),
                (6, image(display_history)),
                (7, Res::View(targets.receiver, general)),
                (8, placeholder),
                (9, Res::Buffer(view.heads[p].buffer)),
                (10, Res::Buffer(view.entries.buffer)),
                (11, Res::Buffer(view.probes.buffer)),
                (12, image(&view.anchors[0])),
                (13, image(&view.anchors[1])),
                (14, image(&view.anchors[0])),
                (15, image(&view.anchors[1])),
                (16, placeholder),
                (17, Res::Buffer(view.rtgi[2].buffer)),
                (19, Res::View(targets.hdr, general)),
            ];
            write_set(
                device,
                view.composite_sets[slot],
                COMPOSITE_BINDINGS,
                &composite,
            );
        }
    }

    unsafe fn destroy_view(
        device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
        view: View,
    ) {
        device.destroy_descriptor_pool(view.pool, None);
        device.destroy_image_view(view.pages_view, None);
        let buffers = view
            .heads
            .into_iter()
            .chain([view.entries, view.history, view.probes])
            .chain(view.rtgi)
            .chain(view.lights)
            .chain(view.uniforms.into_iter().flatten());
        for buffer in buffers {
            gpu::destroy_buffer(device, allocator, buffer);
        }
        let images = view
            .irradiance
            .into_iter()
            .chain(view.metadata)
            .chain(view.background)
            .chain(view.anchors)
            .chain([view.denoise_history, view.placeholder, view.pages]);
        for image in images {
            gpu::destroy_image(device, allocator, image);
        }
    }

    /// GI lights for the frame: punctual and directional, then the view's
    /// layer-mask record. Returns the lights and the conventional count.
    fn frame_lights(frame: &FrameContext) -> (Vec<GiLight>, u32) {
        let mut lights: Vec<GiLight> = frame
            .lights
            .iter()
            .filter(|l| l.radius > 0.0 && l.intensity > 0.0)
            .map(|l| GiLight {
                position: [l.position[0], l.position[1], l.position[2], l.radius],
                // Hyperon point lights shade as intensity / d², like candela.
                color: (Vec3::from_array(l.color) * l.intensity)
                    .extend(0.0)
                    .to_array(),
                ..GiLight::default()
            })
            .collect();
        if let Some(sun) = frame.sun.filter(|s| s.color.max_element() > 0.0) {
            lights.push(GiLight {
                position: (-sun.direction.normalize()).extend(0.0).to_array(),
                color: sun.color.extend(0.0).to_array(),
                ..GiLight::default()
            });
        }
        let conventional = lights.len() as u32;
        lights.push(GiLight {
            source: LAYER_MASK.to_array(),
            ..GiLight::default()
        });
        (lights, conventional)
    }

    unsafe fn record_frame(&mut self, gpu: &mut Gpu, frame: &FrameContext) -> Result<()> {
        let device = gpu.device;
        let settings = self.settings;
        let view = self.view.as_mut().unwrap();
        let pipelines = self.pipelines.as_ref().unwrap();
        let cmd = frame.cmd;
        let slot = (frame.frame_number & 1) as usize;
        debug_assert_eq!(
            slot, frame.frame_slot,
            "frame parity must match the in-flight slot"
        );

        // Lux's clip space is Y-up (wgpu); the renderer's is flipped for Vulkan.
        let flip = Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0));
        let clip = flip * frame.view_proj;
        let previous_clip = flip * frame.previous_view_proj;
        let camera_changed = self
            .previous_clip
            .is_none_or(|p| !p.abs_diff_eq(clip, 1e-6));
        self.previous_clip = Some(clip);
        let (lights, conventional) = Self::frame_lights(frame);
        let lights_changed = bytemuck::cast_slice::<_, u8>(&lights)
            != bytemuck::cast_slice::<_, u8>(&self.previous_lights);
        let first = view.fresh || self.scene_changed;
        let history_valid = self.history_valid && !first;

        let [width, height] = view.grid;
        let samples = width * height;
        let budget = update_budget(&settings);
        let (phase, quota) = self.schedule.next(samples, budget);
        let active = SOURCE_BACKGROUND | if conventional > 0 { SOURCE_PUNCTUAL } else { 0 };
        let changed = if first {
            active
        } else if lights_changed {
            SOURCE_PUNCTUAL
        } else {
            0
        };
        let [r, g, b] = settings.environment;
        let params = Parameters {
            world_from_clip: clip.inverse().to_cols_array_2d(),
            clip_from_world: clip.to_cols_array_2d(),
            previous_clip_from_world: previous_clip.to_cols_array_2d(),
            previous_world_from_clip: previous_clip.inverse().to_cols_array_2d(),
            camera_position: frame.eye.extend(1.0).to_array(),
            viewport: frame.viewport,
            size: [width, height, samples, view.buckets],
            tuning: [
                settings.probe_spacing,
                settings.interval_length,
                settings.normal_bias,
                settings.intensity,
            ],
            environment: [r, g, b, 1.0],
            history: [
                settings.temporal_blend,
                0.0,
                f32::from(u8::from(history_valid)),
                settings.history_frames as f32,
            ],
            interval_history: [
                settings.split_transmittance_blend,
                1.0,
                0.0,
                traversal_query_caps(settings.sampling.max_traversal_queries).0 as f32,
            ],
            frame: [frame.frame_number as u32, 0, self.instances, conventional],
            light_counts: [conventional, 0, settings.emissive_samples, 0],
            stability: [
                f32::from(u8::from(history_valid)),
                f32::from(u8::from(first || lights_changed)),
                1.0,
                1.0,
            ],
            // The mask record follows the lights; bit 0 uses the G-buffer's occlusion byte.
            layers: [conventional, 1, 1, 0],
            environment_rotation: [0.0, 0.0, 0.0, 1.0],
            environment_scale: [1.0, 0.0, 0.0, 0.0],
            source_history: [active, changed, 0, 0],
            sampling: [
                budget,
                settings.sampling.max_traversal_queries,
                settings.sampling.max_cache_entries,
                resolve_scale(&settings),
            ],
            owner_sequence: [
                resolve_scale(&settings),
                0,
                quota,
                sparse_candidate_phase_word(phase, first || camera_changed, false),
            ],
            cache_time: [self.started.elapsed().as_millis() as u32, u32::MAX, 0, 0],
            ..Parameters::default()
        };
        for (index, uniform) in view.uniforms[slot].iter_mut().enumerate() {
            let mut p = params;
            if index < LEVELS {
                p.frame[1] = index as u32;
            } else {
                // Clear 0 resets root-ray offsets; clear 1 preserves them.
                p.owner_sequence[2] = u32::from(index == LEVELS);
            }
            uniform.write(bytemuck::bytes_of(&p));
        }
        view.lights[slot].write(bytemuck::cast_slice(&lights));
        self.previous_lights = lights;

        use vk::{AccessFlags2 as A, PipelineStageFlags2 as S};
        let compute_barrier = || {
            let barrier = [vk::MemoryBarrier2::default()
                .src_stage_mask(S::COMPUTE_SHADER | S::TRANSFER)
                .src_access_mask(A::SHADER_WRITE | A::TRANSFER_WRITE)
                .dst_stage_mask(S::COMPUTE_SHADER | S::TRANSFER)
                .dst_access_mask(
                    A::SHADER_READ | A::SHADER_WRITE | A::TRANSFER_READ | A::TRANSFER_WRITE,
                )];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&barrier),
            );
        };
        if view.fresh {
            let images = view
                .irradiance
                .iter()
                .chain(&view.metadata)
                .chain(&view.background)
                .chain(&view.anchors)
                .chain([&view.denoise_history, &view.placeholder, &view.pages]);
            let transitions: Vec<_> = images
                .clone()
                .map(|image| {
                    gpu::image_barrier(
                        image.image,
                        vk::ImageAspectFlags::COLOR,
                        S::NONE,
                        A::NONE,
                        S::TRANSFER,
                        A::TRANSFER_WRITE,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::GENERAL,
                    )
                })
                .collect();
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&transitions),
            );
            let range = [gpu::subresource(vk::ImageAspectFlags::COLOR)];
            for image in images {
                // The page stand-in is an integer image; zero is the same bits.
                device.cmd_clear_color_image(
                    cmd,
                    image.image,
                    vk::ImageLayout::GENERAL,
                    &vk::ClearColorValue { uint32: [0; 4] },
                    &range,
                );
            }
            for buffer in view
                .heads
                .iter()
                .chain([&view.entries, &view.history, &view.probes])
                .chain(&view.rtgi)
            {
                device.cmd_fill_buffer(cmd, buffer.buffer, 0, vk::WHOLE_SIZE, 0);
            }
            view.fresh = false;
            compute_barrier();
        }

        let sets = view.sets[slot];
        let bind = |layout, set| {
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[set, frame.frame_set],
                &[],
            );
        };
        let passes = std::cell::Cell::new(0usize);
        let limit = self.pass_limit.unwrap_or(usize::MAX);
        let dispatch = |name: &str, set: vk::DescriptorSet, x: u32, y: u32| {
            passes.set(passes.get() + 1);
            if passes.get() > limit {
                return;
            }
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipelines.passes[name]);
            bind(pipelines.layout, set);
            device.cmd_dispatch(cmd, x.max(1), y.max(1), 1);
            compute_barrier();
        };
        let (compute, clear) = (&sets[..LEVELS], &sets[LEVELS..LEVELS + 2]);
        let (temporal, world_resolve, denoise) =
            (sets[LEVELS + 2], sets[LEVELS + 3], &sets[LEVELS + 4..]);
        let samples = u64::from(samples);
        let image = [width.div_ceil(8), height.div_ceil(8)];
        let [shading_x, shading_y] = view.shading.map(|v| v.div_ceil(8));
        let bucket_groups = view.buckets.div_ceil(64);
        let directory_groups = probe_directory_capacity(samples).div_ceil(64) as u32;
        let clear_groups = u64::from(view.buckets)
            .max(probe_directory_capacity(samples) * LEVELS as u64)
            .div_ceil(64) as u32;
        let ray_groups = samples.div_ceil(64) as u32;
        let trace_groups = ray_groups * TRACE_MAX_LANE_STRIDE;
        let query_lanes = samples.min(DEFERRED_TRACE_RECEIVERS)
            * 2
            * DEFERRED_VISIBILITY_QUERIES
            * u64::from(TRACE_MAX_LANE_STRIDE);
        let work = cache_work_slots(samples);
        let entry_count = work + CACHE_HISTORY_PER_BUCKET * u64::from(view.buckets);
        let (pack_x, pack_y) =
            pack_probe_dispatch_groups(samples).ok_or("receiver grid too large")?;

        // Receivers, probe ranking and the cascade hierarchy.
        dispatch("clear", clear[0], clear_groups, 1);
        dispatch("seed", compute[0], image[0], image[1]);
        dispatch("rank", compute[0], ray_groups, 1);
        for &phase in cold_rank_phases(settings.history_frames, history_valid) {
            dispatch("rank", compute[phase], ray_groups, 1);
        }
        for &set in &compute[1..] {
            dispatch("parents", set, directory_groups, 1);
        }
        for level in (0..LEVELS).rev() {
            dispatch("prefix", compute[level], directory_groups, 1);
        }
        dispatch("generateDirections", compute[0], image[0], image[1]);
        dispatch("clear", clear[1], clear_groups, 1);
        // Transport: closest hits, recorded visibility queries, the queries in
        // parallel, shading and deposit; the immediate kernel takes overflow.
        dispatch("traceDeferred", compute[0], trace_groups, 1);
        dispatch("traceRecord", compute[0], trace_groups, 1);
        dispatch(
            "traceVisibility",
            compute[0],
            query_lanes.div_ceil(64) as u32,
            1,
        );
        dispatch("traceResolve", compute[0], trace_groups, 1);
        dispatch("trace", compute[0], trace_groups, 1);
        // World cache and directional probes.
        dispatch("retain", compute[0], bucket_groups, 1);
        dispatch("reduce", compute[0], bucket_groups, 1);
        for level in (0..LEVELS).rev() {
            dispatch(
                "merge",
                compute[level],
                entry_count.div_ceil(64 * LEVELS as u64) as u32,
                LEVELS as u32,
            );
        }
        dispatch(
            "seedProbes",
            compute[0],
            work.div_ceil(64 * LEVELS as u64) as u32,
            1,
        );
        dispatch("packProbes", compute[0], pack_x, pack_y);
        // Per-pixel irradiance, temporal history, spatial filtering.
        dispatch("worldResolve", world_resolve, shading_x, shading_y);
        dispatch("temporal", temporal, shading_x, shading_y);
        // Only temporal results become history; filtered display values never feed back.
        let layers = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1);
        let region = vk::ImageCopy::default()
            .src_subresource(layers)
            .dst_subresource(layers)
            .extent(vk::Extent3D {
                width: view.shading[0],
                height: view.shading[1],
                depth: 1,
            });
        device.cmd_copy_image(
            cmd,
            view.irradiance[2].image,
            vk::ImageLayout::GENERAL,
            view.irradiance[slot].image,
            vk::ImageLayout::GENERAL,
            &[region],
        );
        compute_barrier();
        for &set in denoise.iter().take(settings.spatial_filter_steps as usize) {
            dispatch("denoise", set, shading_x, shading_y);
        }
        // Add diffuse GI times albedo into the HDR target.
        let composite = view.composite_sets[slot];
        for (name, [x, y]) in [
            ("compositeAnchors", [shading_x, shading_y]),
            (
                "compositeCompute",
                [frame.viewport[2].div_ceil(8), frame.viewport[3].div_ceil(8)],
            ),
        ] {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipelines.passes[name]);
            bind(pipelines.composite_layout, composite);
            device.cmd_dispatch(cmd, x.max(1), y.max(1), 1);
            compute_barrier();
        }

        self.history_valid = true;
        self.scene_changed = false;
        Ok(())
    }
}

unsafe fn write_set(
    device: &ash::Device,
    set: vk::DescriptorSet,
    layout: &[(u32, D)],
    entries: &[(u32, Res)],
) {
    let kinds: HashMap<u32, D> = layout.iter().copied().collect();
    let buffers: Vec<[vk::DescriptorBufferInfo; 1]> = entries
        .iter()
        .map(|(_, res)| match res {
            Res::Buffer(buffer) => [vk::DescriptorBufferInfo::default()
                .buffer(*buffer)
                .range(vk::WHOLE_SIZE)],
            Res::View(..) => [vk::DescriptorBufferInfo::default()],
        })
        .collect();
    let images: Vec<[vk::DescriptorImageInfo; 1]> = entries
        .iter()
        .map(|(_, res)| match res {
            Res::View(view, layout) => [vk::DescriptorImageInfo::default()
                .image_view(*view)
                .image_layout(*layout)],
            Res::Buffer(_) => [vk::DescriptorImageInfo::default()],
        })
        .collect();
    let writes: Vec<_> = entries
        .iter()
        .enumerate()
        .map(|(i, (binding, res))| {
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(*binding)
                .descriptor_type(kinds[binding]);
            match res {
                Res::Buffer(_) => write.buffer_info(&buffers[i]),
                Res::View(..) => write.image_info(&images[i]),
            }
        })
        .collect();
    device.update_descriptor_sets(&writes, &[]);
}

impl RenderExtension for Lux {
    fn active(&self) -> bool {
        matches!(self.status, GiStatus::Ready { .. })
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn scene_loaded(
        &mut self,
        gpu: &mut Gpu,
        scene: &SceneData,
        material_base: u32,
        _draw_base: u32,
    ) -> Result<()> {
        self.scene.add(scene, material_base);
        let words = self.scene.pack();
        self.instances = words.instances;
        unsafe {
            let mut geometry = mem::take(&mut self.geometry);
            let mut scene_words = mem::take(&mut self.words);
            let result = self
                .upload(
                    gpu,
                    &mut geometry,
                    "gi geometry words",
                    bytemuck::cast_slice(&words.geometry_words),
                )
                .and_then(|()| {
                    self.upload(
                        gpu,
                        &mut scene_words,
                        "gi scene words",
                        bytemuck::cast_slice(&words.scene_words),
                    )
                });
            (self.geometry, self.words) = (geometry, scene_words);
            result?;
        }
        self.scene_changed = true;
        // New buffers: every descriptor set must point at them.
        if let Some(targets) = self.targets {
            unsafe { self.write_sets(gpu.device, &targets) };
        }
        Ok(())
    }

    fn targets_changed(&mut self, gpu: &mut Gpu, targets: &Targets) -> Result<()> {
        unsafe {
            if self.pipelines.is_none() {
                let pipelines = Self::pipelines(gpu)?;
                let device = gpu.device.clone();
                let (layout, composite_layout) = (pipelines.layout, pipelines.composite_layout);
                self.compile = Some(std::thread::spawn(move || {
                    compile_passes(device, layout, composite_layout)
                }));
                self.status = GiStatus::Compiling {
                    since: Instant::now(),
                };
                self.pipelines = Some(pipelines);
            }
            if self.geometry.buffer == vk::Buffer::null() {
                // Bindings need buffers before a scene arrives.
                let mut geometry = mem::take(&mut self.geometry);
                let mut words = mem::take(&mut self.words);
                let empty = crate::scene::GiScene::default().pack();
                self.upload(
                    gpu,
                    &mut geometry,
                    "gi geometry words",
                    bytemuck::cast_slice(&empty.geometry_words),
                )?;
                self.upload(
                    gpu,
                    &mut words,
                    "gi scene words",
                    bytemuck::cast_slice(&empty.scene_words),
                )?;
                (self.geometry, self.words) = (geometry, words);
            }
            if let Some(view) = self.view.take() {
                Self::destroy_view(gpu.device, gpu.allocator, view);
            }
            self.view = Some(self.create_view(gpu, targets)?);
            self.targets = Some(*targets);
            self.write_sets(gpu.device, targets);
        }
        self.history_valid = false;
        Ok(())
    }

    fn record(&mut self, gpu: &mut Gpu, frame: &FrameContext) -> Result<()> {
        self.poll_compile();
        if !self.active() || self.view.is_none() || frame.viewport[2] == 0 || frame.viewport[3] == 0
        {
            return Ok(());
        }
        unsafe { self.record_frame(gpu, frame) }
    }

    fn destroy(&mut self, gpu: &mut Gpu) {
        // A compile in flight still owns pipelines: wait, then release them.
        if let Some(compile) = self.compile.take()
            && let Ok(Ok(passes)) = compile.join()
            && let Some(pipelines) = self.pipelines.as_mut()
        {
            pipelines.passes.extend(passes);
        }
        unsafe {
            if let Some(view) = self.view.take() {
                Self::destroy_view(gpu.device, gpu.allocator, view);
            }
            if let Some(pipelines) = self.pipelines.take() {
                Self::destroy_pipelines(gpu.device, pipelines);
            }
            gpu::destroy_buffer(gpu.device, gpu.allocator, mem::take(&mut self.geometry));
            gpu::destroy_buffer(gpu.device, gpu.allocator, mem::take(&mut self.words));
            gpu.device.destroy_command_pool(self.command_pool, None);
            self.command_pool = vk::CommandPool::null();
        }
    }
}
