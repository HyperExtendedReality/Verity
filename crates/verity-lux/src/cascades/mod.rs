//! The Split Radiance Cascades frame as a Verity render extension.
//!
//! Each frame, after the renderer's resolve: seed receivers from the
//! G-buffer, rank and build the probe hierarchy, trace interval rays through
//! the GI scene, merge them into the world cache, pack directional probes,
//! resolve per-pixel irradiance, filter it temporally and spatially, and add
//! diffuse GI times albedo to the HDR target. Bindings, sizes and pass order
//! follow `legacy/src/render.rs`.

// Every function here is Vulkan calls; an unsafe fn body is one unsafe context.
#![allow(unsafe_op_in_unsafe_fn)]

mod accel;
pub mod scene;
pub mod settings;
pub mod surface;

use self::{
    accel::SceneAccel,
    scene::{Emitter, GiScene, LAYER_MASK, MAX_EMITTERS, SurfaceWords},
    settings::*,
};
use crate::{
    GiMethod, GiStatus,
    pxtgi::{Capture, Readback},
    shared::{BLUE_NOISE, PassTimer, Patch, Res, upload, upload_ranges, write_set},
};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use gpu_allocator::MemoryLocation;
use std::ops::Range;
use std::{collections::HashMap, ffi::CString, mem, time::Instant};
use verity_render::{
    FrameContext, Gpu, RenderExtension, RetainedChange, Targets,
    asset::SceneData,
    extension::Result,
    gpu::{self, Buffer, Image},
};

const SLOTS: usize = 2;
/// Frames a hashed surface cell may go unread before `surfaceUpdate` drops it.
const SURFACE_HASH_LIFETIME: u32 = 240;

/// How many `stride`-byte elements a whole-buffer binding held: what
/// `GetDimensions` reported before shaders read buffers through pointers.
fn elements(buffer: &Buffer, stride: u64) -> u32 {
    // `gpu::create_buffer` makes every buffer at least 16 bytes.
    (buffer.size.max(16) / stride) as u32
}
const SOURCE_BACKGROUND: u32 = 1;
const SOURCE_PUNCTUAL: u32 = 4;

macro_rules! spirv {
    ($file:literal, $entry:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $file, ".", $entry, ".spv"))
    };
    ($file:literal, $entry:literal, rq) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $file, ".", $entry, ".rq.spv"))
    };
}

/// Every pass this frame dispatches, with its SPIR-V for the software BVH and
/// for hardware ray queries (`GI_RAY_QUERY`); the composite does not trace.
const PASSES: &[(&str, &[u8], &[u8])] = &[
    ("clear", spirv!("cascades", "clear"), spirv!("cascades", "clear", rq)),
    ("seed", spirv!("cascades", "seed"), spirv!("cascades", "seed", rq)),
    ("parents", spirv!("cascades", "parents"), spirv!("cascades", "parents", rq)),
    ("rank", spirv!("cascades", "rank"), spirv!("cascades", "rank", rq)),
    ("prefix", spirv!("cascades", "prefix"), spirv!("cascades", "prefix", rq)),
    ("generateDirections", spirv!("cascades", "generateDirections"), spirv!("cascades", "generateDirections", rq)),
    ("retain", spirv!("cascades", "retain"), spirv!("cascades", "retain", rq)),
    ("trace", spirv!("cascades", "trace"), spirv!("cascades", "trace", rq)),
    ("reduce", spirv!("cascades", "reduce"), spirv!("cascades", "reduce", rq)),
    ("merge", spirv!("cascades", "merge"), spirv!("cascades", "merge", rq)),
    ("seedProbes", spirv!("cascades", "seedProbes"), spirv!("cascades", "seedProbes", rq)),
    ("packProbes", spirv!("cascades", "packProbes"), spirv!("cascades", "packProbes", rq)),
    ("temporal", spirv!("cascades", "temporal"), spirv!("cascades", "temporal", rq)),
    ("denoise", spirv!("cascades", "denoise"), spirv!("cascades", "denoise", rq)),
    ("commitCache", spirv!("cascades", "commitCache"), spirv!("cascades", "commitCache", rq)),
    ("remapProbeHeads", spirv!("cascades", "remapProbeHeads"), spirv!("cascades", "remapProbeHeads", rq)),
    ("worldResolve", spirv!("cascades", "worldResolve"), spirv!("cascades", "worldResolve", rq)),
    ("traceDeferred", spirv!("cascades", "traceDeferred"), spirv!("cascades", "traceDeferred", rq)),
    ("traceRecord", spirv!("cascades", "traceRecord"), spirv!("cascades", "traceRecord", rq)),
    ("traceVisibility", spirv!("cascades", "traceVisibility"), spirv!("cascades", "traceVisibility", rq)),
    ("traceResolve", spirv!("cascades", "traceResolve"), spirv!("cascades", "traceResolve", rq)),
    ("surfaceUpdate", spirv!("cascades", "surfaceUpdate"), spirv!("cascades", "surfaceUpdate", rq)),
    ("reference", spirv!("cascades", "reference"), spirv!("cascades", "reference", rq)),
    ("compositeAnchors", spirv!("composite", "compositeAnchors"), spirv!("composite", "compositeAnchors")),
    ("compositeCompute", spirv!("composite", "compositeCompute"), spirv!("composite", "compositeCompute")),
    ("compositeCapture", spirv!("composite", "compositeCapture"), spirv!("composite", "compositeCapture")),
];


use vk::DescriptorType as D;
/// The cascades' set 0, as `compute_layout` in the legacy renderer (minus the
/// atlas, the DFG texture and the hardware TLAS).
/// Its buffers are pointers in `Parameters::buffers`.
const MAIN_BINDINGS: &[(u32, D)] = &[
    (0, D::UNIFORM_BUFFER),
    (4, D::SAMPLED_IMAGE),
    (10, D::STORAGE_IMAGE),
    (11, D::SAMPLED_IMAGE),
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
    (47, D::SAMPLED_IMAGE),
];
/// The scene TLAS, in set 0 only for the `GI_RAY_QUERY` passes.
const TLAS_BINDING: (u32, D) = (50, D::ACCELERATION_STRUCTURE_KHR);

fn main_bindings(ray_query: bool) -> Vec<(u32, D)> {
    let mut bindings = MAIN_BINDINGS.to_vec();
    bindings.extend(ray_query.then_some(TLAS_BINDING));
    bindings
}

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
    (17, D::STORAGE_BUFFER),
    (18, D::STORAGE_BUFFER),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum SunChange {
    None,
    /// Small enough that the cache keeps blending its history (time of day).
    Gradual,
    /// Appeared, vanished or jumped: its bounce history is dropped.
    Hard,
}

/// How the sky and ground changed: as with the sun, a step of up to 5% keeps
/// the cache's history blending.
fn ambient_change(now: [Vec3; 2], before: [Vec3; 2]) -> SunChange {
    let step = (now[0] - before[0]).abs().max((now[1] - before[1]).abs()).max_element();
    if step == 0.0 {
        SunChange::None
    } else if step <= 0.05 * before[0].max(before[1]).max_element() {
        SunChange::Gradual
    } else {
        SunChange::Hard
    }
}

/// How the sun and the point lights changed since the previous frame.
fn light_changes(lights: &[GiLight], previous: &[GiLight]) -> (SunChange, bool) {
    // The sun is the one directional record with light; the mask record has none.
    let sun = |ls: &[GiLight]| {
        ls.iter()
            .find(|l| l.position[3] == 0.0 && l.color != [0.0; 4])
            .copied()
    };
    fn points(ls: &[GiLight]) -> impl Iterator<Item = &[u8]> {
        ls.iter().filter(|l| l.position[3] > 0.0).map(bytemuck::bytes_of)
    }
    let sun_change = match (sun(lights), sun(previous)) {
        (None, None) => SunChange::None,
        (Some(a), Some(b)) if bytemuck::bytes_of(&a) == bytemuck::bytes_of(&b) => SunChange::None,
        (Some(a), Some(b)) => {
            let (da, db) = (Vec3::from_slice(&a.position[..3]), Vec3::from_slice(&b.position[..3]));
            let (ca, cb) = (Vec3::from_slice(&a.color[..3]), Vec3::from_slice(&b.color[..3]));
            // About 2.6 degrees and 5% of the brightest channel.
            let small = da.normalize().dot(db.normalize()) > 0.999
                && (ca - cb).abs().max_element() <= 0.05 * cb.max_element();
            if small { SunChange::Gradual } else { SunChange::Hard }
        }
        _ => SunChange::Hard,
    };
    (sun_change, !points(lights).eq(points(previous)))
}

/// Emitter records for `emissiveResponseTracked`: corner, edges and normal,
/// `direction.w` the running CDF of their power, `edge1.w` the chance of
/// picking each and `color.w` its area. `position.w` -2 lights both sides.
fn emitter_lights(emitters: &[Emitter]) -> Vec<GiLight> {
    let total: f32 = emitters.iter().map(|e| e.power).sum();
    let mut cdf = 0.0;
    emitters
        .iter()
        .enumerate()
        .map(|(index, e)| {
            let pmf = e.power / total;
            cdf += pmf;
            GiLight {
                position: e.a.extend(if e.double_sided { -2.0 } else { -1.0 }).to_array(),
                color: [0.0, 0.0, 0.0, e.area],
                // The last bound is exactly 1, so every selection finds an emitter.
                direction: e.normal.extend(if index + 1 == emitters.len() { 1.0 } else { cdf }).to_array(),
                edge1: e.edge1.extend(pmf).to_array(),
                edge2: e.edge2.extend(0.0).to_array(),
                source: [e.instance, e.triangle, 0, 0],
            }
        })
        .collect()
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
    descriptors: Buffer,
    /// The reference's running sum per shaded pixel; 16 bytes when it is off.
    reference_sum: Buffer,
    history: Buffer,
    probes: Buffer,
    merge_list: Buffer,
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

type CompiledPasses = std::result::Result<Vec<(&'static str, vk::Pipeline)>, String>;

/// Where compiled pipelines persist between runs; drivers reject stale data.
fn pipeline_cache_path() -> std::path::PathBuf {
    std::env::temp_dir().join("verity-lux-pipelines.bin")
}

/// Compile every pass in parallel through a disk-backed pipeline cache. The
/// transport shaders are large: a cold driver compile takes minutes on one core.
fn compile_passes(
    device: ash::Device,
    layout: vk::PipelineLayout,
    composite_layout: vk::PipelineLayout,
    ray_query: bool,
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
            .map(|&(name, software, hardware)| {
                let spirv = if ray_query { hardware } else { software };
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
pub struct SplitRadianceCascades {
    pub settings: Settings,
    scene: GiScene,
    geometry: Buffer,
    words: Buffer,
    /// Surface cache: triangle charts, island rects, texel records, and the
    /// texels' radiance (three words each, then the hashed cells'), and the
    /// hashed cells' keys and records (see surface_cache.slang).
    surface: [Buffer; 6],
    /// Texels in the surface atlas; zero when the cache is off.
    surface_texels: u32,
    /// Hashed surface cells; zero when they are off.
    surface_hash_cells: u32,
    blue_noise: Buffer,
    /// The scene for hardware ray queries, on GPUs with them.
    accel: Option<SceneAccel>,
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
    /// Models added since the scene was last uploaded.
    upload_pending: bool,
    previous_clip: Option<Mat4>,
    previous_lights: Vec<GiLight>,
    /// Last frame's sky and ground.
    previous_ambient: [Vec3; 2],
    /// Frames `Settings::reference` has averaged, and the pose it averaged them at.
    reference_frames: u32,
    reference_pose: Option<[glam::Vec4; 3]>,
    started: Instant,
    /// Debugging aid: `VERITY_LUX_PASS_LIMIT=n` dispatches only the first
    /// `n` passes of each frame, to bisect GPU faults.
    pass_limit: Option<usize>,
    timer: Option<PassTimer>,
    capture_requested: bool,
    readback: Option<Readback>,
    capture: Option<Capture>,
}

impl SplitRadianceCascades {
    pub fn new(settings: Settings) -> Self {
        let mut scene = GiScene::default();
        scene.set_surface_texel(if settings.surface_cache { settings.surface_texel } else { 0.0 });
        Self {
            settings,
            scene,
            geometry: Buffer::default(),
            words: Buffer::default(),
            surface: Default::default(),
            surface_texels: 0,
            surface_hash_cells: 0,
            blue_noise: Buffer::default(),
            accel: None,
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
            upload_pending: false,
            previous_clip: None,
            previous_lights: Vec::new(),
            previous_ambient: [Vec3::ZERO; 2],
            reference_frames: 0,
            reference_pose: None,
            started: Instant::now(),
            pass_limit: std::env::var("VERITY_LUX_PASS_LIMIT")
                .ok()
                .and_then(|v| v.parse().ok()),
            timer: None,
            capture_requested: false,
            readback: None,
            capture: None,
        }
    }

    /// Read back the next frame's HDR image; see [`Self::take_capture`].
    pub fn request_capture(&mut self) {
        self.capture_requested = true;
    }

    /// The capture asked for, once the GPU has finished it.
    pub fn take_capture(&mut self) -> Option<Capture> {
        self.capture.take()
    }

    /// Finish a capture whose frame the GPU is done with, then start any
    /// requested one, on this slot since its last frame is done.
    unsafe fn tend_capture(
        &mut self,
        gpu: &mut Gpu,
        frame: &FrameContext,
        slot: usize,
    ) -> Result<()> {
        let set = self.view.as_ref().unwrap().composite_sets[slot];
        if self
            .readback
            .as_ref()
            .is_some_and(|r| frame.frame_number > r.frame && r.frame & 1 == slot as u64)
        {
            let readback = self.readback.take().unwrap();
            let idle = self.view.as_ref().unwrap().rtgi[3].buffer;
            write_set(
                gpu.device,
                set,
                COMPOSITE_BINDINGS,
                &[(18, Res::Buffer(idle))],
            );
            self.capture = Some(Capture {
                width: readback.width,
                height: readback.height,
                pixels: bytemuck::cast_slice(readback.buffer.read()).to_vec(),
            });
            gpu::destroy_buffer(gpu.device, gpu.allocator, readback.buffer);
        }
        if self.capture_requested && self.readback.is_none() {
            self.capture_requested = false;
            let [_, _, width, height] = frame.viewport;
            let buffer = gpu::create_buffer(
                gpu.device,
                gpu.allocator,
                "gi capture",
                u64::from(width) * u64::from(height) * 16,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::GpuToCpu,
            )?;
            write_set(
                gpu.device,
                set,
                COMPOSITE_BINDINGS,
                &[(18, Res::Buffer(buffer.buffer))],
            );
            self.readback = Some(Readback {
                buffer,
                frame: frame.frame_number,
                width,
                height,
            });
        }
        Ok(())
    }

    unsafe fn pipelines(gpu: &mut Gpu, ray_query: bool) -> Result<Pipelines> {
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
        let main = set_layout(&main_bindings(ray_query))?;
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

    unsafe fn upload(
        &mut self,
        gpu: &mut Gpu,
        target: &mut Buffer,
        name: &str,
        bytes: &[u8],
    ) -> Result<()> {
        upload(gpu, &mut self.command_pool, target, name, bytes)
    }

    /// Start over in the other tracing mode: the passes, their descriptor sets
    /// and the acceleration structures all depend on it.
    unsafe fn switch_tracing(&mut self, gpu: &mut Gpu, targets: &Targets) -> Result<()> {
        gpu.device.device_wait_idle()?;
        // A compile in flight still owns pipelines: wait, then release them.
        if let Some(compile) = self.compile.take()
            && let Ok(Ok(passes)) = compile.join()
            && let Some(pipelines) = self.pipelines.as_mut()
        {
            pipelines.passes.extend(passes);
        }
        if let Some(pipelines) = self.pipelines.take() {
            Self::destroy_pipelines(gpu.device, pipelines);
        }
        if let Some(view) = self.view.take() {
            Self::destroy_view(gpu.device, gpu.allocator, view);
        }
        if let Some(accel) = self.accel.take() {
            accel.destroy(gpu);
        }
        self.targets_changed(gpu, targets)
    }

    /// Bring the acceleration structures, if any, up to the packed words; see
    /// [`SceneAccel::update`]. Creates them on the first call on a GPU with
    /// ray queries.
    unsafe fn update_accel(
        &mut self,
        gpu: &mut Gpu,
        changed: Option<&[Range<usize>]>,
    ) -> Result<()> {
        if gpu.ray_query && self.settings.hardware_rays && self.accel.is_none() {
            self.accel = Some(SceneAccel::new(gpu));
        }
        match &mut self.accel {
            Some(accel) => accel.update(gpu, &mut self.command_pool, self.scene.words(), changed),
            None => Ok(()),
        }
    }

    /// Upload models added since the last upload. Earlier frames may still
    /// read the buffers this replaces, so it waits for the GPU first.
    unsafe fn flush_scene(&mut self, gpu: &mut Gpu) -> Result<()> {
        if !self.upload_pending {
            return Ok(());
        }
        self.upload_pending = false;
        gpu.device.device_wait_idle()?;
        self.upload_scene(gpu)
    }

    /// Pack the scene and upload it.
    unsafe fn upload_scene(&mut self, gpu: &mut Gpu) -> Result<()> {
        self.scene.pack();
        self.upload_packed(gpu)
    }

    /// Move the packed scene into new buffers and point every descriptor set at them.
    unsafe fn upload_packed(&mut self, gpu: &mut Gpu) -> Result<()> {
        let mut scene = mem::take(&mut self.scene);
        let words = scene.words();
        self.instances = words.instances;
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
        let result = result.and_then(|()| self.upload_surface(gpu, &scene.words().surface));
        self.scene = mem::take(&mut scene);
        result?;
        self.update_accel(gpu, None)?;
        if let Some(targets) = self.targets {
            self.write_sets(gpu.device, &targets);
        }
        Ok(())
    }

    /// Upload the surface cache's words with every texel's radiance cleared,
    /// since a new atlas moves texels, and every hashed cell empty, since its
    /// records name instances in TLAS order. Each buffer holds at least one word.
    unsafe fn upload_surface(&mut self, gpu: &mut Gpu, surface: &SurfaceWords) -> Result<()> {
        self.surface_texels = surface.texels.len() as u32;
        self.surface_hash_cells = if self.settings.surface_cache {
            self.settings.surface_hash_cells
        } else {
            0
        };
        let cells = self.surface_hash_cells as usize;
        let radiance = vec![glam::UVec4::ZERO; 3 * (surface.texels.len() + cells).max(1)];
        // Two u32 words a cell: key and last read frame.
        let keys = vec![glam::UVec4::ZERO; cells.div_ceil(2).max(1)];
        let records = vec![glam::UVec4::ZERO; cells.max(1)];
        let one = [glam::UVec4::ZERO];
        let words = |w: &[glam::UVec4]| -> Vec<u8> {
            bytemuck::cast_slice(if w.is_empty() { &one } else { w }).to_vec()
        };
        let payloads = [
            ("gi surface triangles", words(&surface.triangles)),
            ("gi surface islands", words(&surface.islands)),
            ("gi surface texels", words(&surface.texels)),
            ("gi surface radiance", words(&radiance)),
            ("gi surface hash keys", words(&keys)),
            ("gi surface hash records", words(&records)),
        ];
        let mut buffers = mem::take(&mut self.surface);
        let mut result = Ok(());
        for (buffer, (name, bytes)) in buffers.iter_mut().zip(&payloads) {
            result = result.and_then(|()| self.upload(gpu, buffer, name, bytes));
        }
        self.surface = buffers;
        result
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
        let descriptors = buffer(
            "gi descriptors",
            descriptor_bytes(samples),
            storage,
            MemoryLocation::GpuOnly,
        )?;
        let reference_sum = buffer(
            "gi reference sum",
            if self.settings.reference {
                u64::from(shading[0]) * u64::from(shading[1]) * 16
            } else {
                16
            },
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
        let merge_list = buffer(
            "gi merge list",
            merge_list_bytes(samples, buckets),
            storage | vk::BufferUsageFlags::INDIRECT_BUFFER,
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
            (verity_render::MAX_LIGHTS + MAX_EMITTERS + 2) as u64 * mem::size_of::<GiLight>() as u64;
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
        let main = main_bindings(self.accel.is_some());
        let sizes: Vec<_> = [
            D::UNIFORM_BUFFER,
            D::STORAGE_BUFFER,
            D::SAMPLED_IMAGE,
            D::STORAGE_IMAGE,
            D::ACCELERATION_STRUCTURE_KHR,
        ]
        .into_iter()
        .map(|kind| {
            vk::DescriptorPoolSize::default().ty(kind).descriptor_count(
                SLOTS as u32
                    * (MAIN_SETS as u32 * count(&main, kind) + count(COMPOSITE_BINDINGS, kind)),
            )
        })
        .filter(|size| size.descriptor_count > 0)
        .collect();
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
            descriptors,
            reference_sum,
            history,
            probes,
            merge_list,
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
                    (4, depth),
                    (10, image(output)),
                    (11, image(input)),
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
            let main = main_bindings(self.accel.is_some());
            for (set, entries) in view.sets[slot].iter().zip(&mut contents) {
                if let Some(accel) = &self.accel {
                    entries.push((TLAS_BINDING.0, Res::Accel(accel.tlas())));
                }
                write_set(device, *set, &main, entries);
            }
            // The reference writes the unfiltered image directly.
            let steps = if self.settings.reference { 0 } else { self.settings.spatial_filter_steps };
            let display = &irr[composite_display_index(steps, p)];
            let display_history = if steps == 0 {
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
                (10, Res::Buffer(view.descriptors.buffer)),
                (11, Res::Buffer(view.probes.buffer)),
                (12, image(&view.anchors[0])),
                (13, image(&view.anchors[1])),
                (14, image(&view.anchors[0])),
                (15, image(&view.anchors[1])),
                (17, Res::Buffer(view.rtgi[2].buffer)),
                (18, Res::Buffer(view.rtgi[3].buffer)),
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
            .chain([view.entries, view.descriptors, view.reference_sum, view.history, view.probes, view.merge_list])
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

    /// GI lights for the frame: punctual and directional, then the emitters
    /// and the view's layer-mask record. Returns the lights and the
    /// conventional count.
    fn frame_lights(frame: &FrameContext, emitters: &[Emitter]) -> (Vec<GiLight>, u32) {
        let mut lights: Vec<GiLight> = frame
            .lights
            .iter()
            .filter(|l| l.radius > 0.0 && l.intensity > 0.0)
            .map(|l| GiLight {
                position: [l.position[0], l.position[1], l.position[2], l.radius],
                // Verity point lights shade as intensity / d², like candela.
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
        lights.extend(emitter_lights(emitters));
        lights.push(GiLight {
            source: LAYER_MASK.to_array(),
            ..GiLight::default()
        });
        (lights, conventional)
    }

    unsafe fn record_frame(&mut self, gpu: &mut Gpu, frame: &FrameContext) -> Result<()> {
        let cmd = frame.cmd;
        let slot = (frame.frame_number & 1) as usize;
        if self.timer.is_none() {
            self.timer = Some(PassTimer::new(gpu)?);
        }
        self.timer.as_mut().unwrap().begin(gpu.device, cmd, slot)?;
        self.tend_capture(gpu, frame, slot)?;
        let device = gpu.device;
        let settings = self.settings;
        let view = self.view.as_mut().unwrap();
        let pipelines = self.pipelines.as_ref().unwrap();
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
        let emitters = &self.scene.words().emitters;
        let (lights, conventional) = Self::frame_lights(frame, emitters);
        let emitter_count = emitters.len() as u32;
        // For MIS, a hit works out its own chance of being picked: from the
        // total power, and whether it made the cut (all did unless the list is full).
        let emitter_power: f32 = emitters.iter().map(|e| e.power).sum();
        let emitter_cutoff = if emitters.len() == MAX_EMITTERS {
            emitters.iter().map(|e| e.power).fold(f32::INFINITY, f32::min) * 0.9999
        } else {
            0.0
        };
        // Emitters change only with the scene, which resets everything anyway;
        // the comparisons below look at the conventional lights alone.
        let conventional_lights = &lights[..conventional as usize];
        // The sky and ground are in the background group, with the sun.
        let ambient = [frame.sky, frame.ground];
        let ambient_change = ambient_change(ambient, self.previous_ambient);
        self.previous_ambient = ambient;
        let lights_changed = ambient_change != SunChange::None
            || bytemuck::cast_slice::<_, u8>(conventional_lights)
                != bytemuck::cast_slice::<_, u8>(&self.previous_lights);
        let first = view.fresh || self.scene_changed;
        let history_valid = self.history_valid && !first;
        // The reference restarts when what it averages changes. TAA jitter moves
        // only the projection's x and y rows, so it compares the rest.
        let pose = [frame.view_proj.row(2), frame.view_proj.row(3), frame.eye.extend(0.0)];
        if first || lights_changed || self.reference_pose != Some(pose) {
            self.reference_frames = 0;
        }
        self.reference_pose = Some(pose);

        let [width, height] = view.grid;
        let samples = width * height;
        let budget = update_budget(&settings);
        let (phase, quota) = self.schedule.next(samples, budget);
        let active = SOURCE_BACKGROUND | if conventional > 0 { SOURCE_PUNCTUAL } else { 0 };
        // Sunlight is stored in the background group, point lights in punctual.
        let (sun_change, points_changed) = light_changes(conventional_lights, &self.previous_lights);
        let sun_change = sun_change.max(ambient_change);
        let changed = if first {
            active
        } else {
            (if points_changed { SOURCE_PUNCTUAL } else { 0 })
                | (if sun_change != SunChange::None { SOURCE_BACKGROUND } else { 0 })
        };
        let gradual_sun = !first && sun_change == SunChange::Gradual;
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
            environment: frame.sky.extend(1.0).to_array(),
            ground: frame.ground.extend(0.0).to_array(),
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
            // Emitters follow the conventional lights.
            light_counts: [conventional, emitter_count, settings.emissive_samples, emitter_power.to_bits()],
            stability: [
                f32::from(u8::from(history_valid)),
                f32::from(u8::from(first || lights_changed)),
                1.0,
                1.0,
            ],
            // The mask record follows the lights; bit 0 uses the G-buffer's occlusion byte.
            layers: [conventional + emitter_count, 1, 1, emitter_cutoff.to_bits()],
            environment_rotation: [0.0, 0.0, 0.0, 1.0],
            environment_scale: [1.0, 0.0, 0.0, 0.0],
            // z bit 0: the sun moved a little, so the shaders blend its history rather than drop it.
            source_history: [active, changed, u32::from(gradual_sun), 0],
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
            cache_time: [
                self.started.elapsed().as_millis() as u32,
                u32::MAX,
                u32::from(settings.multi_bounce),
                // Frames the reference has summed; zero restarts it.
                self.reference_frames,
            ],
            surface: [
                self.surface_texels,
                settings.surface_updates.min(self.surface_texels),
                settings.surface_from_level,
                settings.surface_bounce_rays.min(8)
                    | ((settings.surface_software_reach.max(0.0) * 100.0) as u32).min(0xff_ffff) << 8,
            ],
            surface_hash: [
                self.surface_hash_cells,
                settings.surface_hash_updates.min(self.surface_hash_cells),
                SURFACE_HASH_LIFETIME,
                0,
            ],
            // As `GetDimensions` saw them: the whole buffer, in elements.
            lengths: [
                elements(&view.entries, ENTRY_SIZE),
                elements(&view.history, ENTRY_SIZE),
                elements(&view.heads[slot], 4),
                elements(&view.merge_list, 4),
            ],
            word_lengths: [elements(&self.geometry, 16), elements(&self.words, 16), 0, 0],
            // Uploads and `tend_capture` happen before this; nothing moves until next frame.
            buffers: [
                &self.geometry,
                &self.words,
                &view.lights[slot],
                &view.heads[slot],
                &view.entries,
                &view.heads[1 - slot],
                &view.history,
                &view.probes,
                &view.merge_list,
                &view.rtgi[0],
                &view.rtgi[1],
                &view.rtgi[2],
                &view.rtgi[3],
                &view.rtgi[4],
                &self.blue_noise,
                &self.surface[0],
                &self.surface[1],
                &self.surface[2],
                &self.surface[3],
                &view.descriptors,
                &view.reference_sum,
                &self.surface[4],
                &self.surface[5],
            ]
            .map(|buffer| buffer.address),
            ..Parameters::default()
        };        for (index, uniform) in view.uniforms[slot].iter_mut().enumerate() {
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
        self.previous_lights = lights[..conventional as usize].to_vec();

        use vk::{AccessFlags2 as A, PipelineStageFlags2 as S};
        let compute_barrier = || {
            let barrier = [vk::MemoryBarrier2::default()
                .src_stage_mask(S::COMPUTE_SHADER | S::TRANSFER)
                .src_access_mask(A::SHADER_WRITE | A::TRANSFER_WRITE)
                .dst_stage_mask(S::COMPUTE_SHADER | S::TRANSFER | S::DRAW_INDIRECT)
                .dst_access_mask(
                    A::SHADER_READ
                        | A::SHADER_WRITE
                        | A::TRANSFER_READ
                        | A::TRANSFER_WRITE
                        | A::INDIRECT_COMMAND_READ,
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
                .chain([&view.entries, &view.descriptors, &view.history, &view.probes, &view.merge_list])
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
        let timer = std::cell::RefCell::new(self.timer.as_mut().unwrap());
        let stamp = |name: &'static str| timer.borrow_mut().stamp(device, cmd, slot, name);
        let dispatch = |name: &'static str, set: vk::DescriptorSet, x: u32, y: u32| {
            passes.set(passes.get() + 1);
            if passes.get() > limit {
                return;
            }
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipelines.passes[name]);
            bind(pipelines.layout, set);
            device.cmd_dispatch(cmd, x.max(1), y.max(1), 1);
            compute_barrier();
            stamp(name);
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
        let (pack_x, pack_y) =
            pack_probe_dispatch_groups(samples).ok_or("receiver grid too large")?;

        if settings.reference {
            // Path traced, accumulated in the shader; no cache or filter. The
            // clear empties the probe directory: the composite's fallback walks
            // it, and a stale one made each lookup walk all of it.
            dispatch("clear", clear[0], clear_groups, 1);
            dispatch("reference", compute[0], shading_x, shading_y);
            self.reference_frames += 1;
        } else {
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
            // Surface cache texels, then hashed cells, this frame, before any
            // hit reads them.
            let surface_updates = settings.surface_updates.min(self.surface_texels)
                + settings.surface_hash_updates.min(self.surface_hash_cells);
            if surface_updates > 0 {
                dispatch("surfaceUpdate", compute[0], surface_updates.div_ceil(64), 1);
            }
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
                passes.set(passes.get() + 1);
                if passes.get() > limit {
                    continue;
                }
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    pipelines.passes["merge"],
                );
                bind(pipelines.layout, compute[level]);
                device.cmd_dispatch_indirect(cmd, view.merge_list.buffer, level as u64 * 16);
                compute_barrier();
                stamp("merge");
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
            stamp("copyHistory");
            for &set in denoise.iter().take(settings.spatial_filter_steps as usize) {
                dispatch("denoise", set, shading_x, shading_y);
            }
        }
        // Add diffuse GI times albedo into the HDR target.
        let composite = view.composite_sets[slot];
        let full = [frame.viewport[2].div_ceil(8), frame.viewport[3].div_ceil(8)];
        let capturing = self
            .readback
            .as_ref()
            .is_some_and(|r| r.frame == frame.frame_number);
        let capture = capturing.then_some(("compositeCapture", full));
        for (name, [x, y]) in [
            ("compositeAnchors", [shading_x, shading_y]),
            ("compositeCompute", full),
        ]
        .into_iter()
        .chain(capture)
        {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipelines.passes[name]);
            bind(pipelines.composite_layout, composite);
            device.cmd_dispatch(cmd, x.max(1), y.max(1), 1);
            compute_barrier();
            stamp(name);
        }
        if capturing {
            let host = [vk::MemoryBarrier2::default()
                .src_stage_mask(S::COMPUTE_SHADER)
                .src_access_mask(A::SHADER_WRITE)
                .dst_stage_mask(S::HOST)
                .dst_access_mask(A::HOST_READ)];
            device
                .cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&host));
        }
        // Publish the world cache once composition is done reading it: without
        // this, next frame's history heads point at stale working entries and
        // every interval restarts from a single noisy sample.
        if !settings.reference {
            dispatch("commitCache", compute[0], bucket_groups, 1);
            dispatch("remapProbeHeads", compute[0], directory_groups, 1);
        }

        self.history_valid = true;
        self.scene_changed = false;
        Ok(())
    }
}

impl GiMethod for SplitRadianceCascades {
    fn status(&self) -> GiStatus {
        self.status.clone()
    }

    fn set_hardware_rays(&mut self, on: bool) {
        self.settings.hardware_rays = on;
    }

    fn gpu_time(&self) -> Option<std::time::Duration> {
        self.timer.as_ref().and_then(PassTimer::total)
    }

    fn pass_times(&self) -> &[(&'static str, f32)] {
        self.timer.as_ref().map_or(&[], |t| &t.times)
    }

    fn request_capture(&mut self) {
        SplitRadianceCascades::request_capture(self);
    }

    fn take_capture(&mut self) -> Option<Capture> {
        SplitRadianceCascades::take_capture(self)
    }
}

impl RenderExtension for SplitRadianceCascades {
    fn active(&self) -> bool {
        matches!(self.status, GiStatus::Ready { .. })
    }

    fn sky_reflections(&self) -> bool {
        self.settings.sky_reflections
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn scene_loaded(
        &mut self,
        _gpu: &mut Gpu,
        scene: &SceneData,
        material_base: u32,
        _draw_base: u32,
    ) -> Result<()> {
        self.scene.add(scene, material_base);
        self.scene_changed = true;
        // Packing and uploading covers every model so far: doing it once when
        // GI next needs the scene, not per model, saved seconds on Sponza.
        self.upload_pending = true;
        Ok(())
    }

    fn retained_changed(
        &mut self,
        gpu: &mut Gpu,
        changes: &[RetainedChange],
        material: u32,
    ) -> Result<()> {
        unsafe { self.flush_scene(gpu)? };
        self.scene.set_retained(changes, material);
        // History stays: it refines toward the new geometry, where dropping it
        // would flash the whole view on every brush stroke.
        let Some(ranges) = self.scene.refresh() else {
            return unsafe { self.upload_packed(gpu) };
        };
        let words = self.scene.words();
        self.instances = words.instances;
        let bytes = mem::size_of::<glam::UVec4>();
        let in_bytes = |range: &Range<usize>| range.start * bytes..range.end * bytes;
        unsafe {
            upload_ranges(
                gpu,
                &mut self.command_pool,
                &[
                    Patch {
                        target: &self.geometry,
                        source: bytemuck::cast_slice(&words.geometry_words),
                        ranges: ranges.geometry.iter().map(in_bytes).collect(),
                    },
                    Patch {
                        target: &self.words,
                        source: bytemuck::cast_slice(&words.scene_words),
                        ranges: ranges.scene.iter().map(in_bytes).collect(),
                    },
                ],
            )?;
            if self.accel.is_some() {
                self.update_accel(gpu, Some(&ranges.geometry))?;
                // The TLAS was replaced.
                if let Some(targets) = self.targets {
                    self.write_sets(gpu.device, &targets);
                }
            }
            Ok(())
        }
    }

    fn targets_changed(&mut self, gpu: &mut Gpu, targets: &Targets) -> Result<()> {
        unsafe {
            if self.accel.is_none() {
                self.update_accel(gpu, None)?;
            }
            if self.pipelines.is_none() {
                let ray_query = self.accel.is_some();
                let pipelines = Self::pipelines(gpu, ray_query)?;
                let device = gpu.device.clone();
                let (layout, composite_layout) = (pipelines.layout, pipelines.composite_layout);
                self.compile = Some(std::thread::spawn(move || {
                    compile_passes(device, layout, composite_layout, ray_query)
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
                let mut empty = GiScene::default();
                let empty = empty.pack();
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
                self.upload_surface(gpu, &SurfaceWords::default())?;
            }
            if self.blue_noise.buffer == vk::Buffer::null() {
                let mut blue_noise = mem::take(&mut self.blue_noise);
                self.upload(gpu, &mut blue_noise, "gi blue noise", BLUE_NOISE)?;
                self.blue_noise = blue_noise;
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

    fn prepare(&mut self, gpu: &mut Gpu, viewport: [u32; 4]) -> Result<()> {
        // Frames that draw no 3D view, like the editor's loading frames, leave
        // models to gather so they upload once.
        if viewport[2] == 0 || viewport[3] == 0 {
            return Ok(());
        }
        unsafe { self.flush_scene(gpu) }
    }

    fn record(&mut self, gpu: &mut Gpu, frame: &FrameContext) -> Result<()> {
        let hardware = gpu.ray_query && self.settings.hardware_rays;
        if self.pipelines.is_some()
            && hardware != self.accel.is_some()
            && let Some(targets) = self.targets
        {
            unsafe { self.switch_tracing(gpu, &targets)? };
        }
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
            if let Some(timer) = self.timer.take() {
                timer.destroy(gpu.device);
            }
            if let Some(readback) = self.readback.take() {
                gpu::destroy_buffer(gpu.device, gpu.allocator, readback.buffer);
            }
            if let Some(pipelines) = self.pipelines.take() {
                Self::destroy_pipelines(gpu.device, pipelines);
            }
            gpu::destroy_buffer(gpu.device, gpu.allocator, mem::take(&mut self.geometry));
            gpu::destroy_buffer(gpu.device, gpu.allocator, mem::take(&mut self.words));
            for buffer in &mut self.surface {
                gpu::destroy_buffer(gpu.device, gpu.allocator, mem::take(buffer));
            }
            gpu::destroy_buffer(gpu.device, gpu.allocator, mem::take(&mut self.blue_noise));
            if let Some(accel) = self.accel.take() {
                accel.destroy(gpu);
            }
            gpu.device.destroy_command_pool(self.command_pool, None);
            self.command_pool = vk::CommandPool::null();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emitter_records_pick_in_proportion_to_power() {
        let emitter = |power: f32| Emitter {
            a: Vec3::ZERO,
            edge1: Vec3::X,
            edge2: Vec3::Y,
            normal: Vec3::Z,
            area: 0.5,
            power,
            double_sided: false,
            instance: 0,
            triangle: 0,
        };
        let lights = emitter_lights(&[emitter(1.0), emitter(3.0), emitter(0.1)]);
        let pmf: Vec<f32> = lights.iter().map(|l| l.edge1[3]).collect();
        assert!((pmf[1] - 3.0 * pmf[0]).abs() < 1e-6);
        assert!((pmf.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        let cdf: Vec<f32> = lights.iter().map(|l| l.direction[3]).collect();
        assert!(cdf.is_sorted() && cdf[2] == 1.0);
        assert_eq!((lights[0].position[3], lights[0].color[3]), (-1.0, 0.5));
    }

    fn sun(direction: [f32; 3], color: [f32; 3]) -> GiLight {
        GiLight {
            position: [direction[0], direction[1], direction[2], 0.0],
            color: [color[0], color[1], color[2], 0.0],
            ..GiLight::default()
        }
    }

    #[test]
    fn sun_changes_are_background_changes() {
        let point = GiLight { position: [0.0, 1.0, 0.0, 2.0], color: [1.0; 4], ..GiLight::default() };
        let mask = GiLight { source: [1, 2, 3, 4], ..GiLight::default() };
        let base = [point, sun([0.0, 1.0, 0.0], [3.0; 3]), mask];
        assert_eq!(light_changes(&base, &base), (SunChange::None, false));

        let nudged = [point, sun([0.01, 1.0, 0.0], [3.0; 3]), mask];
        assert_eq!(light_changes(&nudged, &base), (SunChange::Gradual, false));

        let set = [point, sun([1.0, 0.2, 0.0], [3.0; 3]), mask];
        assert_eq!(light_changes(&set, &base), (SunChange::Hard, false));

        let gone = [point, mask];
        assert_eq!(light_changes(&gone, &base), (SunChange::Hard, false));

        let moved = [GiLight { position: [1.0, 1.0, 0.0, 2.0], ..point }, base[1], mask];
        assert_eq!(light_changes(&moved, &base), (SunChange::None, true));
    }

    #[test]
    fn sky_changes_grade_like_the_sun() {
        let day = [Vec3::new(0.3, 0.35, 0.45), Vec3::splat(0.15)];
        assert_eq!(ambient_change(day, day), SunChange::None);
        let later = [day[0] * 0.98, day[1]];
        assert_eq!(ambient_change(later, day), SunChange::Gradual);
        assert_eq!(ambient_change([Vec3::ZERO; 2], day), SunChange::Hard);
        // Out of darkness there is no history worth blending.
        assert_eq!(ambient_change(day, [Vec3::ZERO; 2]), SunChange::Hard);
        assert_eq!(SunChange::Gradual.max(SunChange::Hard), SunChange::Hard);
    }
}
