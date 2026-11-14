//! Vulkan 1.3 renderer: clustered lighting over a hybrid visibility buffer.
//!
//! Each frame a compute pass bins point lights into view-space clusters, a
//! raster pass writes depth and (instance, triangle) IDs, a compute resolve
//! rebuilds every pixel's surface from its triangle and shades it with
//! multiple-scattering GGX and energy-preserving Oren–Nayar (EON) diffuse into
//! an HDR target, and a
//! composite pass tonemaps into the swapchain beneath egui. Two frames are in
//! flight. Shaders live in `shaders/` as Slang; `shaders/lib` is the shared
//! engine library other crates' passes include.
//!
//! Scenes load from `.glb` files with UASTC KTX2 textures ([`Renderer::load_gltf`]),
//! transcoded to BC7, ASTC or ETC2 by what the GPU supports and sampled
//! bindlessly by material index.

pub mod asset;
pub mod extension;
pub mod gpu;
mod embedded;
mod retained;
mod gpu_culling;
mod meshlets;
mod profiler;
mod scene;
mod shadows;
mod upload;
mod virtual_texture;

pub use asset::{Sun, TextureFormat};
pub use matrix::MatrixStats;
pub use extension::{FrameContext, Gpu, RenderExtension, RetainedChange, Targets};
pub use embedded::{Dent, EmbeddedInstanceId, EmbeddedModel, EmbeddedModelId, MAX_DENTS};
pub use retained::RetainedMesh;
pub use scene::{entity_matrix, Camera, DeformingMesh, GpuMaterial, Vertex, UP};
pub use upload::SceneStats;
pub use virtual_texture::{TerrainLayer, TerrainRole, TerrainTextureStats, TERRAIN_ROLES};
pub use profiler::GPU_STAGE_NAMES;

use ash::{vk, Entry};
use gpu::{Buffer, Image};
use gpu_allocator::{
    vulkan::{Allocator, AllocatorCreateDesc},
    AllocationSizes, AllocatorDebugSettings, MemoryLocation,
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::{
    error::Error,
    ffi::{CStr, CString},
    io, mem,
    sync::Arc,
    time::{Duration, Instant},
};
use verity_core::{Material, PointLight, Scene, Transform, World};
use winit::window::Window;

type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;
pub const MAX_LIGHTS: usize = scene::LIGHT_WORDS * 32;
/// Point shadows are rendered for the first four lights (loaded glTF lights
/// first, followed by scene/world lights). Further lights remain unshadowed.
/// Invalid lights still occupy a slot. Each shadow uses six 512×512 depth
/// layers, with a near plane at 0.1% of the light radius. Casters closer than
/// that plane are clipped; filtering does not cross cube-face boundaries.
pub const MAX_POINT_SHADOWS: usize = scene::MAX_POINT_SHADOWS;
const FRAMES: usize = 2;
/// Marks a draw that belongs to no static primitive.
const NO_PRIMITIVE: usize = usize::MAX;
/// Marks a draw of an embedded instance: its bounds are at `this + index`, in
/// `Renderer::embedded_bounds`.
const EMBEDDED_PRIMITIVE: usize = usize::MAX / 2;
const VISIBILITY_FORMAT: vk::Format = vk::Format::R32G32_UINT;
const HDR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
// Neutral gray, matching the editor chrome.
const CLEAR_COLOR: [f32; 4] = [0.086, 0.086, 0.086, 1.0];

/// Exposure state before any metering: no sums, exposure 1, not yet seeded.
const EXPOSURE_INITIAL: [u32; 4] = [0, 0, 0x3f80_0000, 0];

macro_rules! spirv {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))
    };
}

pub mod breaker;
mod clouds;
mod matrix;
mod taa;

pub use taa::AntiAliasing;

/// Tessellated UI and the physical-pixel scene viewport within the swapchain.
pub struct UiFrame {
    pub primitives: Vec<egui::ClippedPrimitive>,
    pub textures_delta: egui::TexturesDelta,
    pub pixels_per_point: f32,
    pub viewport: [u32; 4],
}

/// CPU time spent in the renderer. GPU work runs asynchronously and is not
/// measured by these stages.
pub const RENDER_STAGE_NAMES: [&str; 16] = [
    "Swapchain setup", "Wait for frame", "Acquire image", "Prepare and upload",
    "Record geometry uploads", "Record scene", "Record GI", "Record water",
    "Record composite and UI", "Submit", "Present", "  Scene setup and light bins",
    "  Shadows", "  Depth and occlusion", "  Visibility", "  Resolve",
];

#[derive(Clone, Copy, Default)]
pub struct RenderProfile {
    pub stages: [Duration; RENDER_STAGE_NAMES.len()],
    pub gpu_ms: Option<[f32; GPU_STAGE_NAMES.len()]>,
}

impl RenderProfile {
    fn mark(&mut self, start: &mut Option<Instant>, stage: usize) {
        if let Some(previous) = start {
            let now = Instant::now();
            self.stages[stage] = now.duration_since(*previous);
            *previous = now;
        }
    }
}

enum RenderInput<'a> {
    Scene(&'a Scene),
    World(&'a World),
}

impl RenderInput<'_> {
    fn lights(&self) -> &[PointLight] {
        match self {
            Self::Scene(scene) => &scene.lights,
            Self::World(world) => &world.lights,
        }
    }
    fn sun(&self) -> Option<Sun> {
        match self {
            Self::Scene(scene) => scene.sun,
            Self::World(world) => world.sun,
        }
        .map(Sun::from)
    }
    fn lighting(&self) -> &verity_core::Lighting {
        match self {
            Self::Scene(scene) => &scene.lighting,
            Self::World(world) => &world.lighting,
        }
    }
    fn objects(&self) -> Box<dyn Iterator<Item = (&Transform, &Material)> + '_> {
        match self {
            Self::Scene(scene) => {
                Box::new(scene.entities.iter().map(|e| (&e.transform, &e.material)))
            }
            Self::World(world) => Box::new(world.transforms().iter().zip(world.materials())),
        }
    }
}

/// Back-face culling for one-sided materials; none for two-sided ones.
/// Smallest bounding-sphere radius over distance for a depth prepass
/// occluder: about a tenth of the height of a 60 degree view.
const OCCLUDER_SIZE: f32 = 0.05;

/// Whether an opaque draw with model-space bounds `min`..`max` is large on
/// screen from `eye`. Small or distant draws cost the prepass about what they
/// cost the visibility pass and hide little: a 2x quad-overdraw multiplier.
fn is_occluder(model: glam::Mat4, min: glam::Vec3, max: glam::Vec3, eye: glam::Vec3) -> bool {
    let scale = (model.x_axis.truncate().length())
        .max(model.y_axis.truncate().length())
        .max(model.z_axis.truncate().length());
    let radius = 0.5 * (max - min).length() * scale;
    let distance = model.transform_point3(0.5 * (min + max)).distance(eye);
    distance <= radius || radius >= OCCLUDER_SIZE * distance
}

pub(crate) fn cull_sides(two_sided: bool) -> vk::CullModeFlags {
    if two_sided {
        vk::CullModeFlags::NONE
    } else {
        vk::CullModeFlags::BACK
    }
}

fn failure(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::other(message.into()))
}

fn append_lod_range(
    arena: &mut Vec<u32>,
    ranges: &mut std::collections::HashMap<(usize, usize), (u32, u32)>,
    key: (usize, usize),
    indices: &[u32],
) -> Option<(u32, u32)> {
    if let Some(&range) = ranges.get(&key) {
        return Some(range);
    }
    let first = u32::try_from(arena.len()).ok()?;
    let count = u32::try_from(indices.len()).ok()?;
    first.checked_add(count)?;
    arena.extend_from_slice(indices);
    ranges.insert(key, (first, count));
    Some((first, count))
}

/// One draw's visible meshlet runs, and the primitive and placement they
/// were culled for.
#[derive(Default)]
struct MeshletRuns {
    draw: Option<(usize, [[f32; 4]; 4])>,
    runs: Vec<(u32, u32)>,
}

/// Per-frame CPU-written data and the commands that read it.
#[derive(Default)]
struct Frame {
    command: vk::CommandBuffer,
    fence: vk::Fence,
    acquired: vk::Semaphore,
    set: vk::DescriptorSet,
    globals: Buffer,
    /// Instances, lights, instance motion and last frame's deforming vertex
    /// positions, reached through the pointers in `globals`.
    arena: gpu::FrameArena,
    indices: Buffer,
    /// The `model_indices_key` whose model prefix `indices` already holds.
    indices_prefix: Option<(u64, bool)>,
    materials: Buffer,
    /// This frame's deforming vertices, on their way into `Renderer::vertices`.
    deforming: Buffer,
    /// New retained vertices, on their way into `Renderer::vertices`.
    retained: Buffer,
    /// Terrain layer page table, the pages the resolve wanted, and new pages
    /// on their way to the atlases (see `virtual_texture.rs`).
    layer_table: Buffer,
    layer_feedback: Buffer,
    layer_staging: Buffer,
    /// egui textures to free once this frame's fence signals.
    frees: Vec<egui::TextureId>,
}

/// The swapchain and every render target sized to it.
#[derive(Default)]
struct Swapchain {
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    /// One per image: presentation must not reuse a semaphore still in use.
    rendered: Vec<vk::Semaphore>,
    format: vk::Format,
    extent: vk::Extent2D,
    srgb: bool,
    depth: Option<Image>,
    visibility: Option<Image>,
    /// 4x MSAA visibility and depth, resolved from sample 0 into the two
    /// above; only while an MSAA mode is chosen.
    visibility_msaa: Option<Image>,
    depth_msaa: Option<Image>,
    hdr: Option<Image>,
    gbuffer: Option<Image>,
    receiver: Option<Image>,
    motion: Option<Image>,
}

/// A loaded material and how its draws are issued.
struct LoadedMaterial {
    gpu: scene::GpuMaterial,
    double_sided: bool,
    /// Not drawn: blending needs a forward pass that does not exist yet.
    hidden: bool,
}

/// What [`pick_device`] found and enabled.
struct DeviceChoice {
    physical: vk::PhysicalDevice,
    device: ash::Device,
    queue_family: u32,
    texture_format: TextureFormat,
    /// Maximum sampler anisotropy, or `None` when unsupported.
    anisotropy: Option<f32>,
    /// Acceleration structures, ray queries and buffer device addresses are on.
    ray_query: bool,
}

/// Owns all Vulkan resources and the window for the lifetime of its surface.
pub struct Renderer {
    profiling: bool,
    render_profile: Option<RenderProfile>,
    gpu_timer: Option<profiler::GpuTimer>,
    gpu_timer_attempted: bool,
    /// CPU recording time of `record_scene`'s sub-stages (render stages 11..16).
    scene_stages: [Duration; 5],
    _entry: Entry,
    instance: ash::Instance,
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    _window: Arc<Window>,
    physical: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    ray_query: bool,
    swap_loader: ash::khr::swapchain::Device,
    allocator: Option<Allocator>,
    depth_format: vk::Format,
    swap: Swapchain,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    descriptor_pool: vk::DescriptorPool,
    cluster_pipeline: vk::Pipeline,
    /// Turns the resolve's metering into the exposure; see `shaders/exposure.slang`.
    adapt_pipeline: vk::Pipeline,
    visibility_pipeline: vk::Pipeline,
    depth_prepass: vk::Pipeline,
    resolve_pipeline: vk::Pipeline,
    /// Null when the GPU cannot multisample the visibility pass.
    visibility_msaa_pipeline: vk::Pipeline,
    resolve_msaa_pipeline: vk::Pipeline,
    /// Built for the swapchain format, so rebuilt when that changes.
    composite_pipeline: vk::Pipeline,
    composite_format: vk::Format,
    command_pool: vk::CommandPool,
    frames: Vec<Frame>,
    frame: usize,
    clusters: Buffer,
    /// Auto exposure state, shared across frames: see `shaders/lib/exposure.slang`.
    exposure: Buffer,
    /// When the last frame was presented, for exposure adaptation.
    last_present: Option<Instant>,
    vertices: Buffer,
    brdf_lut: Buffer,
    vertex_count: u32,
    index_total: u32,
    sampler: vk::Sampler,
    /// Sun shadow cascades: an array view for sampling, a view per layer for rendering.
    shadow_map: Option<Image>,
    shadow_layers: Vec<vk::ImageView>,
    point_shadow_map: Option<Image>,
    point_shadow_layers: Vec<vk::ImageView>,
    shadow_pipeline: vk::Pipeline,
    shadow_sampler: vk::Sampler,
    ambient_override: Option<(glam::Vec3, glam::Vec3)>,
    texture_format: TextureFormat,
    anisotropy: Option<f32>,
    textures: Vec<Image>,
    /// Index ranges; primitive 0 is the unit cube world entities draw with.
    primitives: Vec<asset::Primitive>,
    /// Meshlets in primitive order; missing metadata means draw the whole primitive.
    primitive_meshlets: Vec<Option<Vec<meshlets::Meshlet>>>,
    /// Original indices remain available for exact fallback.
    source_indices: Vec<u32>,
    /// Starts with the model draws' indices, built once per `model_indices_key`
    /// and kept across frames; each frame appends its own after them.
    frame_indices: Vec<u32>,
    /// Each model draw's range in `frame_indices`, or `None` when not drawn.
    model_index_ranges: Vec<Option<(u32, u32)>>,
    model_indices_len: usize,
    /// The `model_generation` and Matrix state the model prefix was built for.
    model_indices_key: Option<(u64, bool)>,
    /// Changes whenever model draws or primitives do.
    model_generation: u64,
    /// Each shadow layer's casters, what it was last drawn from, and whether
    /// this frame draws it again.
    shadows: shadows::ShadowLayers,
    /// Meshes drawn as set by `set_deforming_meshes`, and the space reserved
    /// for their vertices at `deforming_base` of the shared vertex buffer.
    deforming: Vec<DeformingMesh>,
    deforming_vertices: Vec<Vertex>,
    deforming_base: u32,
    deforming_capacity: u32,
    retained: retained::Retained,
    clear_color: [f32; 4],
    shadow_distance: f32,
    embedded: Option<embedded::Embedded>,
    terrain_layers: Option<virtual_texture::TerrainLayers>,
    /// This frame's embedded draws, in draw order.
    embedded_bounds: Vec<(glam::Vec3, glam::Vec3)>,
    matrix: matrix::Matrix,
    matrix_enabled: bool,
    matrix_debug: bool,
    model_draws: Vec<asset::Draw>,
    materials: Vec<LoadedMaterial>,
    scene_lights: Vec<PointLight>,
    sun: Option<Sun>,
    sun_is_imported: bool,
    /// Directional light selected for the current frame.
    frame_sun: Option<Sun>,
    /// This frame's ambient hemisphere: sky, ground.
    frame_ambient: (glam::Vec3, glam::Vec3),
    camera: Camera,
    previous_view_proj: Option<glam::Mat4>,
    /// This frame's view-projection and the one before it.
    frame_view_proj: (glam::Mat4, glam::Mat4),
    frame_lights: Vec<PointLight>,
    frame_number: u64,
    extension: Option<Box<dyn RenderExtension>>,
    breakers: breaker::BreakerPass,
    clouds: clouds::CloudPass,
    taa: taa::TaaPass,
    anti_aliasing: AntiAliasing,
    sharpening: f32,
    /// Sunlight let through the clouds, sampled by the resolve.
    cloud_shadow: Option<gpu::Image>,
    /// This frame's cloud layer, from the scene's lighting.
    frame_clouds: Option<verity_core::Clouds>,
    requested_size: (u32, u32),
    recreate: bool,
    /// Present as fast as frames finish instead of waiting for vsync.
    uncapped: bool,
    ui_renderer:
        Option<egui_ash_renderer::Renderer<egui_ash_renderer::allocator::DefaultAllocator>>,
    ui_format: Option<vk::Format>,
    pending_textures: Vec<egui::TexturesDelta>,
    instance_data: Vec<scene::Instance>,
    /// Last frame's model and base vertex per draw index (`Instance::draw[3]`).
    previous_models: std::collections::HashMap<u32, (glam::Mat4, u32)>,
    /// This frame's `Frame::instance_motion`, in `instance_data` order.
    motion_data: Vec<InstanceMotion>,
    /// Each deforming mesh's vertex positions last frame, by slot.
    previous_deforming: Vec<Vec<[f32; 4]>>,
    /// This frame's `Frame::previous_positions`.
    previous_positions: Vec<[f32; 4]>,
    /// Deforming instances: their index in `instance_data` and first vertex
    /// in `previous_positions`.
    deforming_motion: Vec<(usize, u32)>,
    light_data: Vec<scene::Light>,
    material_data: Vec<scene::GpuMaterial>,
    /// Index count, two-sidedness, and primitive index for each instance.
    draw_calls: Vec<(u32, bool, usize)>,
    shadow_calls: Vec<(u32, u32)>,
    culler: Option<gpu_culling::Culler>,
    culling_ranges: Vec<Option<(usize, usize)>>,
    /// Per draw: whether it occludes enough to join the depth prepass.
    occluders: Vec<bool>,
    /// Each draw's meshlet runs that pass the frustum, for the camera and
    /// models in `meshlet_runs_key`; a draw keeps its runs while it stays put.
    meshlet_runs: Vec<MeshletRuns>,
    meshlet_runs_key: Option<(glam::Mat4, u64, bool)>,
    culling_active: bool,
    hiz_active: bool,
    depth_sampled: bool,
}

impl Renderer {
    /// Creates a Vulkan 1.3 instance, device, and window swapchain. Debug
    /// builds enable the Khronos validation layer when it is installed.
    pub fn new(window: Arc<Window>) -> Result<Self> {
        unsafe {
            let entry = Entry::load()?;
            if entry
                .try_enumerate_instance_version()?
                .unwrap_or(vk::API_VERSION_1_0)
                < vk::API_VERSION_1_3
            {
                return Err(failure("Vulkan 1.3 is required"));
            }
            let app = CString::new("Verity")?;
            let display = window.display_handle()?.as_raw();
            let handle = window.window_handle()?.as_raw();
            let mut extensions = ash_window::enumerate_required_extensions(display)?.to_vec();
            let available = entry.enumerate_instance_extension_properties(None)?;
            let has = |name: &CStr| {
                available
                    .iter()
                    .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == name)
            };
            let mut flags = vk::InstanceCreateFlags::empty();
            if has(ash::khr::portability_enumeration::NAME) {
                extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
                flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
            }
            let validation = c"VK_LAYER_KHRONOS_validation";
            let layers: Vec<_> = entry
                .enumerate_instance_layer_properties()?
                .iter()
                .filter(|l| {
                    cfg!(debug_assertions) && CStr::from_ptr(l.layer_name.as_ptr()) == validation
                })
                .map(|_| validation.as_ptr())
                .collect();
            let app_info = vk::ApplicationInfo::default()
                .application_name(&app)
                .api_version(vk::API_VERSION_1_3);
            let instance = entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app_info)
                    .enabled_extension_names(&extensions)
                    .enabled_layer_names(&layers)
                    .flags(flags),
                None,
            )?;
            let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
            let surface = match ash_window::create_surface(&entry, &instance, display, handle, None)
            {
                Ok(surface) => surface,
                Err(e) => {
                    instance.destroy_instance(None);
                    return Err(e.into());
                }
            };
            let choice = match pick_device(&instance, &surface_loader, surface) {
                Ok(picked) => picked,
                Err(e) => {
                    surface_loader.destroy_surface(surface, None);
                    instance.destroy_instance(None);
                    return Err(e);
                }
            };
            let DeviceChoice {
                physical,
                device,
                queue_family,
                texture_format,
                anisotropy,
                ray_query,
            } = choice;
            let size = window.inner_size();
            let mut renderer = Self {
                swap_loader: ash::khr::swapchain::Device::new(&instance, &device),
                queue: device.get_device_queue(queue_family, 0),
                depth_format: depth_format(&instance, physical),
                _entry: entry,
                instance,
                surface_loader,
                surface,
                _window: window,
                physical,
                device,
                queue_family,
                ray_query,
                allocator: None,
                swap: Swapchain::default(),
                set_layout: vk::DescriptorSetLayout::null(),
                pipeline_layout: vk::PipelineLayout::null(),
                descriptor_pool: vk::DescriptorPool::null(),
                cluster_pipeline: vk::Pipeline::null(),
                adapt_pipeline: vk::Pipeline::null(),
                visibility_pipeline: vk::Pipeline::null(),
                depth_prepass: vk::Pipeline::null(),
                resolve_pipeline: vk::Pipeline::null(),
                visibility_msaa_pipeline: vk::Pipeline::null(),
                resolve_msaa_pipeline: vk::Pipeline::null(),
                composite_pipeline: vk::Pipeline::null(),
                composite_format: vk::Format::UNDEFINED,
                command_pool: vk::CommandPool::null(),
                frames: Vec::new(),
                frame: 0,
                profiling: false,
                render_profile: None,
                gpu_timer: None,
                gpu_timer_attempted: false,
                scene_stages: [Duration::ZERO; 5],
                clusters: Buffer::default(),
                exposure: Buffer::default(),
                last_present: None,
                vertices: Buffer::default(),
                brdf_lut: Buffer::default(),
                vertex_count: 0,
                index_total: 0,
                sampler: vk::Sampler::null(),
                shadow_map: None,
                shadow_layers: Vec::new(),
                point_shadow_map: None,
                point_shadow_layers: Vec::new(),
                shadow_pipeline: vk::Pipeline::null(),
                shadow_sampler: vk::Sampler::null(),
                ambient_override: None,
                texture_format,
                anisotropy,
                textures: Vec::new(),
                primitives: Vec::new(),
                primitive_meshlets: Vec::new(),
                source_indices: Vec::new(),
                frame_indices: Vec::new(),
                model_index_ranges: Vec::new(),
                model_indices_len: 0,
                model_indices_key: None,
                model_generation: 0,
                shadows: shadows::ShadowLayers::default(),
                deforming: Vec::new(),
                deforming_vertices: Vec::new(),
                deforming_base: 0,
                deforming_capacity: 0,
                retained: Default::default(),
                clear_color: CLEAR_COLOR,
                shadow_distance: scene::SHADOW_DISTANCE,
                embedded: None,
                terrain_layers: None,
                embedded_bounds: Vec::new(),
                matrix: Default::default(),
                matrix_enabled: false,
                matrix_debug: false,
                model_draws: Vec::new(),
                materials: Vec::new(),
                scene_lights: Vec::new(),
                sun: None,
                sun_is_imported: false,
                frame_sun: None,
                frame_ambient: Default::default(),
                camera: Camera::default(),
                previous_view_proj: None,
                frame_view_proj: Default::default(),
                frame_lights: Vec::new(),
                frame_number: 0,
                extension: None,
                breakers: breaker::BreakerPass::default(),
                clouds: clouds::CloudPass::default(),
                taa: taa::TaaPass::default(),
                anti_aliasing: AntiAliasing::default(),
                sharpening: 0.5,
                cloud_shadow: None,
                frame_clouds: None,
                requested_size: (size.width, size.height),
                recreate: true,
                uncapped: false,
                ui_renderer: None,
                ui_format: None,
                pending_textures: Vec::new(),
                instance_data: Vec::new(),
                previous_models: Default::default(),
                motion_data: Vec::new(),
                previous_deforming: Vec::new(),
                previous_positions: Vec::new(),
                deforming_motion: Vec::new(),
                light_data: Vec::new(),
                material_data: Vec::new(),
                draw_calls: Vec::new(),
                shadow_calls: Vec::new(),
                culler: None,
                culling_ranges: Vec::new(),
                occluders: Vec::new(),
                meshlet_runs: Vec::new(),
                meshlet_runs_key: None,
                hiz_active: false,
                depth_sampled: false,
                culling_active: false,
            };
            // Drop releases whatever a failed step leaves behind.
            renderer.init_resources()?;
            let limits = renderer
                .instance
                .get_physical_device_properties(physical)
                .limits;
            renderer.culler = gpu_culling::Culler::new(&renderer.device, limits).ok();
            renderer.embedded = Some(embedded::Embedded::new(&renderer.device)?);
            renderer.depth_sampled = renderer
                .instance
                .get_physical_device_format_properties(physical, renderer.depth_format)
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE);
            if size.width > 0 && size.height > 0 {
                renderer.rebuild_swapchain()?;
            }
            Ok(renderer)
        }
    }

    unsafe fn init_resources(&mut self) -> Result<()> {
        let device = &self.device;
        self.allocator = Some(Allocator::new(&AllocatorCreateDesc {
            instance: self.instance.clone(),
            device: device.clone(),
            physical_device: self.physical,
            debug_settings: AllocatorDebugSettings::default(),
            buffer_device_address: true,
            allocation_sizes: AllocationSizes::default(),
        })?);

        use vk::DescriptorType as D;
        let all = vk::ShaderStageFlags::COMPUTE
            | vk::ShaderStageFlags::VERTEX
            | vk::ShaderStageFlags::FRAGMENT;
        let kinds = [
            D::UNIFORM_BUFFER,
            D::STORAGE_BUFFER,
            D::STORAGE_BUFFER,
            D::STORAGE_BUFFER,
            D::STORAGE_BUFFER,
            D::STORAGE_BUFFER,
            D::STORAGE_BUFFER,
            D::SAMPLED_IMAGE,
            D::STORAGE_IMAGE,
            D::SAMPLED_IMAGE,
            D::STORAGE_BUFFER,
            D::SAMPLER,
            D::SAMPLED_IMAGE,
            D::STORAGE_IMAGE,
            D::STORAGE_IMAGE,
            D::STORAGE_IMAGE,
            D::SAMPLED_IMAGE,
            D::SAMPLER,
            D::SAMPLED_IMAGE,
            // Matrix (freed below).
            D::STORAGE_BUFFER,
            D::STORAGE_BUFFER,
            D::STORAGE_BUFFER,
            D::STORAGE_BUFFER,
            D::STORAGE_BUFFER,
            D::STORAGE_BUFFER,
            D::STORAGE_BUFFER,
            D::STORAGE_BUFFER,
            D::STORAGE_BUFFER,
            D::STORAGE_BUFFER,
            D::STORAGE_BUFFER,
            // Terrain layers: page table, then feedback.
            D::STORAGE_BUFFER,
            D::STORAGE_BUFFER,
            // Cloud shadows: see `clouds.rs`.
            D::SAMPLED_IMAGE,
            // Scene depth, for `taa.rs`.
            D::SAMPLED_IMAGE,
            // Every visibility sample, for the MSAA resolve; unwritten without MSAA.
            D::SAMPLED_IMAGE,
            // Auto exposure: see `shaders/lib/exposure.slang`.
            D::STORAGE_BUFFER,
        ];
        // Instances, lights, vertices, indices and Matrix's buffers: now
        // pointers in `Globals`. Their slots stay empty so no other binding renumbers.
        let freed = |i: usize| matches!(i, 1 | 2 | 4 | 5 | 19..=29);
        let slots: Vec<_> = (kinds.iter().copied().enumerate())
            .filter(|&(i, _)| !freed(i))
            .collect();
        let textures_binding = 12;
        let count = |i: usize| {
            if i == textures_binding {
                scene::MAX_TEXTURES
            } else {
                1
            }
        };
        let bindings: Vec<_> = slots
            .iter()
            .map(|&(i, kind)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i as u32)
                    .descriptor_type(kind)
                    .descriptor_count(count(i))
                    .stage_flags(all)
            })
            .collect();
        // Texture slots past the loaded ones stay unwritten.
        let flags: Vec<_> = slots
            .iter()
            .map(|&(i, _)| {
                if i == textures_binding {
                    vk::DescriptorBindingFlags::PARTIALLY_BOUND
                } else {
                    vk::DescriptorBindingFlags::empty()
                }
            })
            .collect();
        let mut binding_flags =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
        self.set_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(&bindings)
                .push_next(&mut binding_flags),
            None,
        )?;
        let set_layouts = [self.set_layout];
        let push = [
            vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                .size(8),
            vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(8)
                .size(8),
        ];
        self.pipeline_layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&push),
            None,
        )?;
        let pool_sizes = [
            D::UNIFORM_BUFFER,
            D::STORAGE_BUFFER,
            D::SAMPLED_IMAGE,
            D::STORAGE_IMAGE,
            D::SAMPLER,
        ]
        .map(|ty| {
            let per_set: u32 = (slots.iter())
                .filter(|&&(_, kind)| kind == ty)
                .map(|&(i, _)| count(i))
                .sum();
            vk::DescriptorPoolSize::default()
                .ty(ty)
                .descriptor_count(FRAMES as u32 * per_set)
        });
        self.descriptor_pool = device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(FRAMES as u32)
                .pool_sizes(&pool_sizes),
            None,
        )?;

        let layout = self.pipeline_layout;
        self.cluster_pipeline = gpu::compute_pipeline(
            device,
            vk::PipelineCache::null(),
            layout,
            spirv!("clusters.clusterMain"),
            c"clusterMain",
        )?;
        self.resolve_pipeline = gpu::compute_pipeline(
            device,
            vk::PipelineCache::null(),
            layout,
            spirv!("resolve.resolveMain"),
            c"resolveMain",
        )?;
        self.adapt_pipeline = gpu::compute_pipeline(
            device,
            vk::PipelineCache::null(),
            layout,
            spirv!("exposure.adaptMain"),
            c"adaptMain",
        )?;
        // Cull mode is dynamic: `cull_sides` sets it per draw.
        self.visibility_pipeline = gpu::graphics_pipeline(
            device,
            layout,
            &gpu::Raster {
                vertex: spirv!("visibility.vertexMain"),
                fragment: spirv!("visibility.fragmentMain"),
                color: Some(VISIBILITY_FORMAT),
                depth: Some((self.depth_format, vk::CompareOp::GREATER)),
                cull: None,
                depth_bias: None,
            },
        )?;
        if let Some(samples) = self.msaa_samples() {
            self.resolve_msaa_pipeline = gpu::compute_pipeline(
                device,
                vk::PipelineCache::null(),
                layout,
                spirv!("resolve.resolveMsaaMain"),
                c"resolveMsaaMain",
            )?;
            self.visibility_msaa_pipeline = gpu::graphics_pipeline_samples(
                device,
                layout,
                &gpu::Raster {
                    vertex: spirv!("visibility.vertexMain"),
                    fragment: spirv!("visibility.fragmentMain"),
                    color: Some(VISIBILITY_FORMAT),
                    depth: Some((self.depth_format, vk::CompareOp::GREATER)),
                    cull: None,
                    depth_bias: None,
                },
                samples,
            )?;
        }
        self.depth_prepass = gpu::graphics_pipeline(
            device,
            layout,
            &gpu::Raster {
                vertex: spirv!("visibility.vertexMain"),
                fragment: spirv!("depth_prepass.fragmentMain"),
                color: None,
                depth: Some((self.depth_format, vk::CompareOp::GREATER)),
                cull: None,
                depth_bias: None,
            },
        )?;
        let mut sampler = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .max_lod(vk::LOD_CLAMP_NONE);
        if let Some(max) = self.anisotropy {
            sampler = sampler
                .anisotropy_enable(true)
                .max_anisotropy(max.min(16.0));
        }
        self.sampler = device.create_sampler(&sampler, None)?;
        // Hardware PCF: compare against stored depth; outside the map is lit.
        self.shadow_sampler = device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
                .compare_enable(true)
                .compare_op(vk::CompareOp::LESS_OR_EQUAL),
            None,
        )?;
        let (shadow_map, shadow_layers) = gpu::create_depth_layers(
            device,
            self.allocator.as_mut().unwrap(),
            "sun shadows",
            self.depth_format,
            scene::SHADOW_SIZE,
            scene::CASCADES as u32,
        )?;
        self.shadow_map = Some(shadow_map);
        self.shadow_layers = shadow_layers;
        // New maps hold nothing: every layer draws again.
        self.shadows = shadows::ShadowLayers::default();
        let (point_shadow_map, point_shadow_layers) = gpu::create_depth_layers(
            device,
            self.allocator.as_mut().unwrap(),
            "point shadows",
            self.depth_format,
            scene::POINT_SHADOW_SIZE,
            (scene::MAX_POINT_SHADOWS * scene::POINT_SHADOW_FACES) as u32,
        )?;
        self.point_shadow_map = Some(point_shadow_map);
        self.cloud_shadow = Some(clouds::create_shadow_map(
            device,
            self.allocator.as_mut().unwrap(),
        )?);
        self.point_shadow_layers = point_shadow_layers;
        self.shadow_pipeline = gpu::graphics_pipeline(
            device,
            layout,
            &gpu::Raster {
                vertex: spirv!("shadow.vertexMain"),
                fragment: spirv!("shadow.fragmentMain"),
                color: None,
                depth: Some((self.depth_format, vk::CompareOp::LESS)),
                // Both faces cast: thin and double-sided geometry stays opaque to the sun.
                cull: Some(vk::CullModeFlags::NONE),
                depth_bias: Some((1.25, 1.75)),
            },
        )?;

        self.command_pool = device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(self.queue_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )?;
        let commands = device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(FRAMES as u32),
        )?;
        let sets = device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(&[self.set_layout; FRAMES]),
        )?;

        let msaa_samples = self.msaa_samples();
        let allocator = self.allocator.as_mut().unwrap();
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let lut = scene::brdf_lut();
        self.brdf_lut = gpu::create_buffer(
            device,
            allocator,
            "brdf lut",
            mem::size_of_val(lut.as_slice()) as u64,
            storage,
            MemoryLocation::CpuToGpu,
        )?;
        self.brdf_lut.write(bytemuck::cast_slice(&lut));
        let cluster_bytes = (scene::CLUSTERS * scene::LIGHT_WORDS * 4) as u64;
        self.clusters = gpu::create_buffer(
            device,
            allocator,
            "clusters",
            cluster_bytes,
            storage,
            MemoryLocation::GpuOnly,
        )?;
        self.exposure = gpu::create_buffer(
            device,
            allocator,
            "exposure",
            mem::size_of_val(&EXPOSURE_INITIAL) as u64,
            storage,
            MemoryLocation::CpuToGpu,
        )?;
        self.exposure.write(bytemuck::cast_slice(&EXPOSURE_INITIAL));

        for (command, set) in commands.into_iter().zip(sets) {
            let mut frame = Frame {
                command,
                set,
                ..Frame::default()
            };
            // Push first so Drop owns everything created below.
            self.frames.push(Frame::default());
            frame.fence = device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )?;
            frame.acquired = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
            *self.frames.last_mut().unwrap() = frame;
            let frame = self.frames.last_mut().unwrap();
            let size = mem::size_of::<scene::Globals>() as u64;
            frame.globals = gpu::create_buffer(
                device,
                allocator,
                "globals",
                size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                MemoryLocation::CpuToGpu,
            )?;
            frame.indices = gpu::create_buffer(
                device,
                allocator,
                "frame indices",
                16,
                storage,
                MemoryLocation::CpuToGpu,
            )?;
            frame.materials = gpu::create_buffer(
                device,
                allocator,
                "materials",
                64 * mem::size_of::<scene::GpuMaterial>() as u64,
                storage,
                MemoryLocation::CpuToGpu,
            )?;
            // A zero layer count turns layered terrain off until layers are set.
            for (buffer, name, location) in [
                (
                    &mut frame.layer_table,
                    "terrain layer table",
                    MemoryLocation::CpuToGpu,
                ),
                (
                    &mut frame.layer_feedback,
                    "terrain layer feedback",
                    MemoryLocation::GpuToCpu,
                ),
            ] {
                *buffer = gpu::create_buffer(device, allocator, name, 16, storage, location)?;
                buffer.write(&[0; 16]);
            }
        }
        self.matrix = matrix::Matrix::new(device, allocator, layout, self.depth_format, msaa_samples)?;
        // Appending the cube also writes every frame's buffer descriptors.
        let (vertices, indices) = scene::cube();
        self.append_geometry(&vertices, &indices)?;
        self.primitives.push(asset::Primitive {
            first_index: 0,
            index_count: indices.len() as u32,
            base_vertex: 0,
            material: 0,
        });
        let positions: Vec<_> = vertices
            .iter()
            .map(|v| [v.position[0], v.position[1], v.position[2]])
            .collect();
        self.primitive_meshlets
            .push(meshlets::build(&positions, &indices));
        self.source_indices.extend_from_slice(&indices);
        self.write_texture_descriptors();
        Ok(())
    }

    /// Point every frame's set at the sampler and the loaded textures.
    unsafe fn write_texture_descriptors(&self) {
        let sampler = [vk::DescriptorImageInfo::default().sampler(self.sampler)];
        let shadow_sampler = [vk::DescriptorImageInfo::default().sampler(self.shadow_sampler)];
        let shadow_map = self.shadow_map.as_ref().map(|map| {
            [vk::DescriptorImageInfo::default()
                .image_view(map.view)
                .image_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)]
        });
        let point_shadow_map = self.point_shadow_map.as_ref().map(|map| {
            [vk::DescriptorImageInfo::default()
                .image_view(map.view)
                .image_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)]
        });
        let cloud_shadow = self.cloud_shadow.as_ref().map(|map| {
            [vk::DescriptorImageInfo::default()
                .image_view(map.view)
                .image_layout(vk::ImageLayout::GENERAL)]
        });
        let images: Vec<_> = self
            .textures
            .iter()
            .map(|t| {
                vk::DescriptorImageInfo::default()
                    .image_view(t.view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            })
            .collect();
        for frame in &self.frames {
            let mut writes = vec![vk::WriteDescriptorSet::default()
                .dst_set(frame.set)
                .dst_binding(11)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(&sampler)];
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(frame.set)
                    .dst_binding(17)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&shadow_sampler),
            );
            if let Some(shadow_map) = &shadow_map {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(frame.set)
                        .dst_binding(16)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(shadow_map),
                );
            }
            if let Some(point_shadow_map) = &point_shadow_map {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(frame.set)
                        .dst_binding(18)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(point_shadow_map),
                );
            }
            if let Some(cloud_shadow) = &cloud_shadow {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(frame.set)
                        .dst_binding(32)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(cloud_shadow),
                );
            }
            if !images.is_empty() {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(frame.set)
                        .dst_binding(12)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&images),
                );
            }
            self.device.update_descriptor_sets(&writes, &[]);
        }
    }

    /// Look at `target` from `eye` with a vertical field of view in radians.
    pub fn set_camera(&mut self, eye: glam::Vec3, target: glam::Vec3, vertical_fov: f32) {
        self.camera = Camera {
            eye,
            target,
            vertical_fov,
        };
    }

    /// How far from the camera the sun casts shadows, in metres. A longer reach
    /// spreads the same shadow map over more ground, so close shadows soften.
    pub fn set_shadow_distance(&mut self, distance: f32) {
        self.shadow_distance = distance;
    }

    /// The color behind the scene, in the swapchain's encoding.
    pub fn set_clear_color(&mut self, color: [f32; 4]) {
        self.clear_color = color;
    }

    /// Override the scene's hemisphere colors. Direct-only scenes remain direct-only.
    pub fn set_ambient(&mut self, sky: glam::Vec3, ground: glam::Vec3) {
        self.ambient_override = Some((sky, ground));
    }

    /// Set the fallback directional light when the drawn scene or world has no sun.
    /// A scene/world sun takes precedence over this and any imported glTF sun.
    pub fn set_sun(&mut self, sun: Option<Sun>) {
        self.sun = sun.map(|sun| Sun {
            direction: sun.direction.try_normalize().unwrap_or(glam::Vec3::NEG_Z),
            ..sun
        });
        self.sun_is_imported = false;
    }

    /// Install or replace the shore surface without replacing render extensions.
    /// Grid arrays are copied once and uploaded independently to each in-flight
    /// frame. Update only the time with [`Self::set_breaker_time`] thereafter.
    pub fn set_breakers(
        &mut self,
        grid: breaker::BreakerGrid<'_>,
        profile: breaker::BreakerProfile,
    ) -> Result<()> {
        self.breakers.set(grid, profile)
    }

    /// Set the shore-wave time in seconds without re-uploading its grid.
    pub fn set_breaker_time(&mut self, time: f32) {
        self.breakers.set_time(time);
    }

    /// Stop rendering the shore-wave surface.
    pub fn clear_breakers(&mut self) {
        self.breakers.clear();
    }

    /// Install an extension such as GI, replacing any current one. Install it
    /// before loading scenes: it only learns about scenes loaded afterwards.
    pub fn set_extension(&mut self, mut extension: Box<dyn RenderExtension>) -> Result<()> {
        unsafe { self.device.device_wait_idle()? };
        if let Some(mut old) = self.extension.take() {
            old.destroy(&mut self.gpu());
        }
        if let Some(targets) = self.targets() {
            extension.targets_changed(&mut self.gpu(), &targets)?;
        }
        self.extension = Some(extension);
        self.retained.notify_all();
        Ok(())
    }

    /// The installed extension, if it is a `T`, to change its settings.
    pub fn extension_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.extension.as_mut()?.as_any_mut().downcast_mut()
    }

    /// Remove the installed extension, releasing its GPU resources.
    pub fn remove_extension(&mut self) -> Result<()> {
        unsafe { self.device.device_wait_idle()? };
        if let Some(mut extension) = self.extension.take() {
            extension.destroy(&mut self.gpu());
        }
        Ok(())
    }

    /// Have the extension rebuild its target-sized resources, after settings
    /// that size them change.
    pub fn refresh_extension_targets(&mut self) -> Result<()> {
        unsafe { self.device.device_wait_idle()? };
        if let (Some(targets), Some(mut extension)) = (self.targets(), self.extension.take()) {
            let result = extension.targets_changed(&mut self.gpu(), &targets);
            self.extension = Some(extension);
            result?;
        }
        Ok(())
    }

    /// The device and memory, lent to the extension.
    fn gpu(&mut self) -> Gpu<'_> {
        Gpu {
            instance: &self.instance,
            physical: self.physical,
            device: &self.device,
            queue: self.queue,
            queue_family: self.queue_family,
            ray_query: self.ray_query,
            allocator: self.allocator.as_mut().expect("allocator lives until drop"),
            frame_set_layout: self.set_layout,
            frames_in_flight: FRAMES,
        }
    }

    /// Views of the current render targets, once a swapchain exists.
    fn targets(&self) -> Option<Targets> {
        let swap = &self.swap;
        Some(Targets {
            extent: swap.extent,
            depth: swap.depth.as_ref()?.view,
            gbuffer: swap.gbuffer.as_ref()?.view,
            receiver: swap.receiver.as_ref()?.view,
            motion: swap.motion.as_ref()?.view,
            hdr: swap.hdr.as_ref()?.view,
        })
    }

    /// The block format textures are transcoded to on this GPU.
    pub fn texture_format(&self) -> TextureFormat {
        self.texture_format
    }

    /// Replace the meshes whose vertices change every frame: cloth and other
    /// soft bodies. They are in world space and cast shadows.
    pub fn set_deforming_meshes(&mut self, meshes: Vec<DeformingMesh>) {
        self.deforming = meshes;
    }

    /// Draw cooked meshes through Matrix instead of triangle by triangle.
    pub fn set_matrix_enabled(&mut self, enabled: bool) {
        self.matrix_enabled = enabled;
    }

    /// Stop waiting for vsync, where the surface allows it.
    pub fn set_uncapped(&mut self, uncapped: bool) {
        self.recreate |= uncapped != self.uncapped;
        self.uncapped = uncapped;
    }

    /// Choose how the scene is anti-aliased; [`AntiAliasing::Desktop`] by default.
    pub fn set_anti_aliasing(&mut self, mode: AntiAliasing) {
        // The multisampled targets come and go with the swapchain's.
        self.recreate |= mode.msaa() != self.anti_aliasing.msaa();
        self.anti_aliasing = mode;
    }

    /// How much temporal anti-aliasing sharpens, from 0 (off) to 1; 0.5 by default.
    pub fn set_sharpening(&mut self, amount: f32) {
        self.sharpening = amount.clamp(0.0, 1.0);
    }

    /// Color visible surfaces by the LOD actually selected for that frame.
    pub fn set_matrix_debug(&mut self, enabled: bool) {
        self.matrix_debug = enabled;
    }

    /// Point a frame's descriptor set at its buffers and the shared ones.
    unsafe fn write_buffer_descriptors(&self, index: usize) {
        let frame = &self.frames[index];
        let mut infos: Vec<(u32, [vk::DescriptorBufferInfo; 1])> = vec![
            (0, [frame.globals.info()]),
            (3, [self.clusters.info()]),
            (6, [self.brdf_lut.info()]),
            (10, [frame.materials.info()]),
        ];
        infos.push((30, [frame.layer_table.info()]));
        infos.push((31, [frame.layer_feedback.info()]));
        infos.push((35, [self.exposure.info()]));
        let writes: Vec<_> = infos
            .iter()
            .map(|(binding, info)| {
                let binding = *binding;
                vk::WriteDescriptorSet::default()
                    .dst_set(frame.set)
                    .dst_binding(binding)
                    .descriptor_type(if binding == 0 {
                        vk::DescriptorType::UNIFORM_BUFFER
                    } else {
                        vk::DescriptorType::STORAGE_BUFFER
                    })
                    .buffer_info(info)
            })
            .collect();
        self.device.update_descriptor_sets(&writes, &[]);
    }

    /// 4x, when the GPU can render the visibility pass multisampled and
    /// sample the integer IDs back.
    fn msaa_samples(&self) -> Option<vk::SampleCountFlags> {
        let samples = vk::SampleCountFlags::TYPE_4;
        let limits = unsafe { self.instance.get_physical_device_properties(self.physical) }.limits;
        (limits.framebuffer_color_sample_counts
            & limits.framebuffer_depth_sample_counts
            & limits.sampled_image_integer_sample_counts)
            .contains(samples)
            .then_some(samples)
    }

    /// Point every frame's set at the current render targets.
    unsafe fn write_image_descriptors(&self) {
        let (Some(visibility), Some(hdr), Some(gbuffer), Some(receiver), Some(motion), Some(depth)) = (
            &self.swap.visibility,
            &self.swap.hdr,
            &self.swap.gbuffer,
            &self.swap.receiver,
            &self.swap.motion,
            &self.swap.depth,
        ) else {
            return;
        };
        let storage = |binding, view: vk::ImageView| {
            (
                binding,
                vk::DescriptorType::STORAGE_IMAGE,
                view,
                vk::ImageLayout::GENERAL,
            )
        };
        let images = [
            (
                7,
                vk::DescriptorType::SAMPLED_IMAGE,
                visibility.view,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ),
            (
                8,
                vk::DescriptorType::STORAGE_IMAGE,
                hdr.view,
                vk::ImageLayout::GENERAL,
            ),
            (
                9,
                vk::DescriptorType::SAMPLED_IMAGE,
                hdr.view,
                vk::ImageLayout::GENERAL,
            ),
            storage(13, gbuffer.view),
            storage(14, receiver.view),
            storage(15, motion.view),
            (
                33,
                vk::DescriptorType::SAMPLED_IMAGE,
                depth.view,
                vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL,
            ),
        ];
        let infos = images.map(|(_, _, view, layout)| {
            [vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(layout)]
        });
        let samples = self.swap.visibility_msaa.as_ref().map(|image| {
            [vk::DescriptorImageInfo::default()
                .image_view(image.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]
        });
        for frame in &self.frames {
            let mut writes: Vec<_> = images
                .iter()
                .zip(&infos)
                .map(|(&(binding, kind, _, _), info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(frame.set)
                        .dst_binding(binding)
                        .descriptor_type(kind)
                        .image_info(info)
                })
                .collect();
            if let Some(info) = &samples {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(frame.set)
                        .dst_binding(34)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(info),
                );
            }
            self.device.update_descriptor_sets(&writes, &[]);
        }
    }

    unsafe fn clear_swapchain(&mut self) {
        let device = &self.device;
        let allocator = self.allocator.as_mut().unwrap();
        for image in [
            self.swap.depth.take(),
            self.swap.visibility.take(),
            self.swap.visibility_msaa.take(),
            self.swap.depth_msaa.take(),
            self.swap.hdr.take(),
            self.swap.gbuffer.take(),
            self.swap.receiver.take(),
            self.swap.motion.take(),
        ]
        .into_iter()
        .flatten()
        {
            gpu::destroy_image(device, allocator, image);
        }
        for semaphore in self.swap.rendered.drain(..) {
            device.destroy_semaphore(semaphore, None);
        }
        for view in self.swap.views.drain(..) {
            device.destroy_image_view(view, None);
        }
        self.swap_loader
            .destroy_swapchain(self.swap.swapchain, None);
        self.swap = Swapchain {
            format: self.swap.format,
            ..Swapchain::default()
        };
    }

    unsafe fn rebuild_swapchain(&mut self) -> Result<()> {
        if self.requested_size.0 == 0 || self.requested_size.1 == 0 {
            return Ok(());
        }
        self.device.device_wait_idle()?;
        self.clear_swapchain();
        // Keep partial resources in self so Drop and a subsequent rebuild can release them.
        let caps = self
            .surface_loader
            .get_physical_device_surface_capabilities(self.physical, self.surface)?;
        let formats = self
            .surface_loader
            .get_physical_device_surface_formats(self.physical, self.surface)?;
        let present_modes = self
            .surface_loader
            .get_physical_device_surface_present_modes(self.physical, self.surface)?;
        let srgb_space =
            |f: &vk::SurfaceFormatKHR| f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR;
        let format = formats
            .iter()
            .copied()
            .find(|f| {
                srgb_space(f)
                    && matches!(
                        f.format,
                        vk::Format::B8G8R8A8_UNORM
                            | vk::Format::R8G8B8A8_UNORM
                            | vk::Format::UNDEFINED
                    )
            })
            .or_else(|| {
                formats.iter().copied().find(|f| {
                    srgb_space(f)
                        && matches!(
                            f.format,
                            vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB
                        )
                })
            })
            .ok_or_else(|| failure("surface has no supported sRGB color-space format"))?;
        let chosen = if format.format == vk::Format::UNDEFINED {
            vk::Format::B8G8R8A8_UNORM
        } else {
            format.format
        };
        let srgb = matches!(
            chosen,
            vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB
        );
        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: self
                    .requested_size
                    .0
                    .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: self
                    .requested_size
                    .1
                    .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };
        if extent.width == 0 || extent.height == 0 {
            return Ok(());
        }
        let count = (caps.min_image_count + 1).min(if caps.max_image_count == 0 {
            u32::MAX
        } else {
            caps.max_image_count
        });
        let composite = [
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::INHERIT,
        ]
        .into_iter()
        .find(|f| caps.supported_composite_alpha.contains(*f))
        .ok_or_else(|| failure("surface has no composite alpha mode"))?;
        if !caps
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        {
            return Err(failure("surface images cannot be color attachments"));
        }
        self.swap.swapchain = self.swap_loader.create_swapchain(
            &vk::SwapchainCreateInfoKHR::default()
                .surface(self.surface)
                .min_image_count(count)
                .image_format(chosen)
                .image_color_space(format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(caps.current_transform)
                .composite_alpha(composite)
                .present_mode(gpu::present_mode(&present_modes, self.uncapped))
                .clipped(true),
            None,
        )?;
        self.swap.format = chosen;
        self.swap.extent = extent;
        self.swap.srgb = srgb;
        self.swap.images = self.swap_loader.get_swapchain_images(self.swap.swapchain)?;
        let device = &self.device;
        for &image in &self.swap.images {
            self.swap.views.push(
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(chosen)
                        .subresource_range(gpu::subresource(vk::ImageAspectFlags::COLOR)),
                    None,
                )?,
            );
            self.swap
                .rendered
                .push(device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?);
        }
        let msaa_samples = self.msaa_samples().filter(|_| self.anti_aliasing.msaa());
        let allocator = self.allocator.as_mut().unwrap();
        use vk::ImageUsageFlags as U;
        self.swap.depth = Some(gpu::create_image(
            device,
            allocator,
            "depth",
            self.depth_format,
            extent,
            1,
            U::DEPTH_STENCIL_ATTACHMENT | U::SAMPLED,
            vk::ImageAspectFlags::DEPTH,
        )?);
        self.swap.visibility = Some(gpu::create_image(
            device,
            allocator,
            "visibility",
            VISIBILITY_FORMAT,
            extent,
            1,
            U::COLOR_ATTACHMENT | U::SAMPLED,
            vk::ImageAspectFlags::COLOR,
        )?);
        if let Some(samples) = msaa_samples {
            self.swap.visibility_msaa = Some(gpu::create_multisampled_image(
                device,
                allocator,
                "visibility msaa",
                VISIBILITY_FORMAT,
                extent,
                samples,
                U::COLOR_ATTACHMENT | U::SAMPLED,
                vk::ImageAspectFlags::COLOR,
            )?);
            self.swap.depth_msaa = Some(gpu::create_multisampled_image(
                device,
                allocator,
                "depth msaa",
                self.depth_format,
                extent,
                samples,
                U::DEPTH_STENCIL_ATTACHMENT,
                vk::ImageAspectFlags::DEPTH,
            )?);
        }
        self.swap.hdr = Some(gpu::create_image(
            device,
            allocator,
            "hdr",
            HDR_FORMAT,
            extent,
            1,
            U::STORAGE | U::SAMPLED | U::COLOR_ATTACHMENT,
            vk::ImageAspectFlags::COLOR,
        )?);
        for (target, name, format) in [
            (
                &mut self.swap.gbuffer,
                "gi gbuffer",
                vk::Format::R32G32B32A32_UINT,
            ),
            (
                &mut self.swap.receiver,
                "gi receiver",
                vk::Format::R32G32B32A32_UINT,
            ),
            (&mut self.swap.motion, "motion", vk::Format::R32G32_SFLOAT),
        ] {
            *target = Some(gpu::create_image(
                device,
                allocator,
                name,
                format,
                extent,
                1,
                U::STORAGE | U::SAMPLED,
                vk::ImageAspectFlags::COLOR,
            )?);
        }
        self.write_image_descriptors();

        if self.composite_format != chosen {
            device.destroy_pipeline(self.composite_pipeline, None);
            self.composite_pipeline = vk::Pipeline::null();
            self.composite_pipeline = gpu::graphics_pipeline(
                device,
                self.pipeline_layout,
                &gpu::Raster {
                    vertex: spirv!("composite.vertexMain"),
                    fragment: spirv!("composite.fragmentMain"),
                    color: Some(chosen),
                    depth: None,
                    cull: Some(vk::CullModeFlags::NONE),
                    depth_bias: None,
                },
            )?;
            self.composite_format = chosen;
        }
        let mode =
            egui_ash_renderer::RenderMode::DynamicRendering(egui_ash_renderer::DynamicRendering {
                color_attachment_format: chosen,
                depth_attachment_format: None,
                stencil_attachment_format: None,
            });
        match self.ui_renderer.as_mut() {
            Some(ui) if self.ui_format != Some(chosen) => {
                // egui bakes the framebuffer's sRGB encoding into its textures.
                if self.ui_format.is_some_and(|f| is_srgb(f) != srgb) {
                    return Err(failure(
                        "swapchain color encoding changed while UI textures are active",
                    ));
                }
                ui.set_render_mode(mode)?;
            }
            Some(_) => {}
            None => {
                self.ui_renderer = Some(egui_ash_renderer::Renderer::with_default_allocator(
                    &self.instance,
                    self.physical,
                    device.clone(),
                    mode,
                    egui_ash_renderer::Options {
                        in_flight_frames: FRAMES,
                        enable_depth_test: false,
                        enable_depth_write: false,
                        srgb_framebuffer: srgb,
                    },
                )?);
            }
        }
        self.ui_format = Some(chosen);
        if let (Some(targets), Some(mut extension)) = (self.targets(), self.extension.take()) {
            let result = extension.targets_changed(&mut self.gpu(), &targets);
            self.extension = Some(extension);
            result?;
        }
        self.recreate = false;
        Ok(())
    }

    /// Marks the swapchain for recreation. Zero dimensions suspend rendering.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.requested_size = (width, height);
        self.recreate = true;
    }

    /// Draws one frame. More than [`MAX_LIGHTS`] lights is reported as an error
    /// rather than silently truncated.
    pub fn draw(&mut self, scene: &Scene) -> Result<()> {
        self.draw_inner(RenderInput::Scene(scene), None)
    }

    /// Draws the current runtime world directly, without serializing or copying its components.
    pub fn draw_world(&mut self, world: &World) -> Result<()> {
        self.draw_inner(RenderInput::World(world), None)
    }

    /// Draws the scene in a physical-pixel viewport and composites egui over it.
    pub fn draw_ui(&mut self, scene: &Scene, mut frame: UiFrame) -> Result<()> {
        self.pending_textures
            .push(mem::take(&mut frame.textures_delta));
        self.draw_inner(RenderInput::Scene(scene), Some(frame))
    }

    /// Enable CPU stage timing for subsequent frames.
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling = enabled;
        if !enabled {
            self.render_profile = None;
        }
    }

    /// CPU timings for the last completed profiled frame.
    pub fn profile(&self) -> Option<RenderProfile> {
        self.render_profile
    }

    /// Close GPU stage `i - 1` when profiling with a working timer.
    unsafe fn gpu_stamp(&self, cmd: vk::CommandBuffer, i: u32) {
        if let Some(timer) = self.gpu_timer.as_ref().filter(|_| self.profiling) {
            timer.stamp(&self.device, cmd, self.frame, i);
        }
    }

    fn draw_inner(&mut self, scene: RenderInput<'_>, ui: Option<UiFrame>) -> Result<()> {
        let mut profile = RenderProfile::default();
        let mut timing = self.profiling.then(Instant::now);
        self.render_profile = None;
        if scene.lights().len() + self.scene_lights.len() > MAX_LIGHTS {
            return Err(failure(format!(
                "renderer supports at most {MAX_LIGHTS} point lights"
            )));
        }
        if self.requested_size.0 == 0 || self.requested_size.1 == 0 {
            return Ok(());
        }
        unsafe {
            if self.recreate {
                self.rebuild_swapchain()?;
            }
            if self.swap.swapchain == vk::SwapchainKHR::null() {
                return Ok(());
            }
            profile.mark(&mut timing, 0);
            let device = &self.device;
            let frame_index = self.frame;
            let frame = &mut self.frames[frame_index];
            device.wait_for_fences(&[frame.fence], true, u64::MAX)?;
            for id in frame.frees.drain(..) {
                self.ui_renderer.as_mut().unwrap().free_texture(id)?;
            }
            if self.profiling && !self.gpu_timer_attempted {
                self.gpu_timer_attempted = true;
                self.gpu_timer =
                    profiler::GpuTimer::new(&self.instance, device, self.physical, self.queue_family);
            }
            if let Some(timer) = &mut self.gpu_timer {
                timer.poll(device, frame_index);
                profile.gpu_ms = timer.times;
            }
            profile.mark(&mut timing, 1);
            let (image, suboptimal) = match self.swap_loader.acquire_next_image(
                self.swap.swapchain,
                u64::MAX,
                frame.acquired,
                vk::Fence::null(),
            ) {
                Ok(v) => v,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.recreate = true;
                    return Ok(());
                }
                Err(e) => return Err(e.into()),
            };
            profile.mark(&mut timing, 2);

            self.update_textures(ui.is_some())?;
            let extent = self.swap.extent;
            let viewport = ui
                .as_ref()
                .map_or([0, 0, extent.width, extent.height], |f| f.viewport);
            let x = viewport[0].min(extent.width);
            let y = viewport[1].min(extent.height);
            let viewport = [
                x,
                y,
                viewport[2].min(extent.width - x),
                viewport[3].min(extent.height - y),
            ];
            self.upload_frame(&scene, viewport)?;
            if let Some(mut extension) = self.extension.take() {
                let result = extension.prepare(&mut self.gpu(), viewport);
                self.extension = Some(extension);
                result?;
            }
            profile.mark(&mut timing, 3);

            let (fence, command, set, acquired) = {
                let frame = &self.frames[frame_index];
                (frame.fence, frame.command, frame.set, frame.acquired)
            };
            self.device.reset_fences(&[fence])?;
            self.device
                .reset_command_buffer(command, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            let gpu_timing = self.gpu_timer.as_ref().filter(|_| self.profiling).is_some();
            if gpu_timing {
                self.gpu_timer.as_ref().unwrap().begin(&self.device, command, frame_index);
            }
            if let Some(embedded) = &mut self.embedded {
                embedded.record(&self.device, command, frame_index);
            }
            self.record_vertex_copies(
                command,
                self.frames[self.frame].deforming.buffer,
                &self.deforming_copy(),
            );
            self.record_vertex_copies(
                command,
                self.frames[self.frame].retained.buffer,
                &self.retained.copies,
            );
            self.record_terrain_pages(command);
            self.gpu_stamp(command, 1);
            profile.mark(&mut timing, 4);
            self.clouds.record_shadow(
                &self.device,
                self.allocator.as_mut().unwrap(),
                self.set_layout,
                command,
                set,
                self.cloud_shadow.as_ref().unwrap(),
                self.frame_clouds.filter(|_| self.frame_sun.is_some()),
                self.camera.eye,
                viewport,
            )?;
            self.record_scene(command, set, viewport)?;
            profile.stages[11..16].copy_from_slice(&self.scene_stages);
            profile.mark(&mut timing, 5);
            self.record_extension(command, set, viewport)?;
            self.clouds.record(
                &self.device,
                self.allocator.as_mut().unwrap(),
                command,
                set,
                self.frame_number,
                viewport,
                self.camera.eye,
                self.frame_clouds,
            )?;
            self.gpu_stamp(command, 7);
            profile.mark(&mut timing, 6);
            self.breakers.record(
                &self.device,
                self.allocator.as_mut().unwrap(),
                self.depth_format,
                frame_index,
                command,
                viewport,
                self.frame_view_proj.0,
                self.swap.hdr.as_ref().unwrap(),
                self.swap.depth.as_ref().unwrap(),
            )?;
            self.taa.record(
                &self.device,
                self.allocator.as_mut().unwrap(),
                self.set_layout,
                command,
                set,
                viewport,
                self.anti_aliasing,
                self.swap.visibility_msaa.is_some(),
                self.sharpening,
            )?;
            self.gpu_stamp(command, 8);
            profile.mark(&mut timing, 7);
            self.record_present(command, set, image as usize, viewport, ui.as_ref())?;
            self.gpu_stamp(command, 9);
            profile.mark(&mut timing, 8);
            self.frame_number += 1;
            let device = &self.device;
            device.end_command_buffer(command)?;

            let rendered = self.swap.rendered[image as usize];
            let wait = [vk::SemaphoreSubmitInfo::default()
                .semaphore(acquired)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];
            let signal = [vk::SemaphoreSubmitInfo::default()
                .semaphore(rendered)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(command)];
            device.queue_submit2(
                self.queue,
                &[vk::SubmitInfo2::default()
                    .wait_semaphore_infos(&wait)
                    .command_buffer_infos(&commands)
                    .signal_semaphore_infos(&signal)],
                fence,
            )?;
            if gpu_timing {
                self.gpu_timer.as_mut().unwrap().submitted(frame_index);
            }
            profile.mark(&mut timing, 9);
            let swapchains = [self.swap.swapchain];
            let indices = [image];
            let waits = [rendered];
            let result = self.swap_loader.queue_present(
                self.queue,
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(&waits)
                    .swapchains(&swapchains)
                    .image_indices(&indices),
            );
            profile.mark(&mut timing, 10);
            self.render_profile = self.profiling.then_some(profile);
            self.frame = (self.frame + 1) % FRAMES;
            match result {
                Ok(present_suboptimal) => self.recreate |= suboptimal || present_suboptimal,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.recreate = true,
                Err(e) => return Err(e.into()),
            }
            Ok(())
        }
    }

    /// Apply queued egui texture changes. Frees from the frame being drawn wait
    /// for its fence; earlier skipped frames' frees are safe immediately.
    unsafe fn update_textures(&mut self, drawing_ui: bool) -> Result<()> {
        let mut pending = mem::take(&mut self.pending_textures);
        if pending.iter().any(|delta| !delta.set.is_empty()) {
            // Updates may rewrite textures a frame in flight still samples.
            self.device.device_wait_idle()?;
        }
        let last = pending.len().saturating_sub(1);
        let frame = &mut self.frames[self.frame];
        let painter = self.ui_renderer.as_mut().unwrap();
        let result: Result<()> = (|| {
            for (i, delta) in pending.iter_mut().enumerate() {
                for (id, changes) in delta.set.drain() {
                    for change in changes {
                        painter.set_texture(self.queue, self.command_pool, id, &change)?;
                    }
                }
                for id in delta.free.drain() {
                    if i == last && drawing_ui {
                        frame.frees.push(id);
                    } else {
                        painter.free_texture(id)?;
                    }
                }
            }
            Ok(())
        })();
        if result.is_err() {
            for delta in &mut pending {
                delta.clear();
            }
        }
        result
    }

    /// Map each instance from this frame's world space to last frame's, by
    /// draw index, and remember this frame's models for the next.
    fn track_instance_motion(&mut self) {
        self.motion_data.clear();
        self.previous_models =
            instance_motion(&self.previous_models, &self.instance_data, &mut self.motion_data);
        for &(instance, first) in &self.deforming_motion {
            self.motion_data[instance].info = [1, first, 0, 0];
        }
        self.previous_deforming.clear();
        self.previous_deforming.extend(
            (self.deforming.iter()).map(|mesh| mesh.vertices.iter().map(|v| v.position).collect()),
        );
    }

    /// Build this frame's draws, materials and lights and write them with the
    /// globals, growing buffers as needed.
    unsafe fn upload_frame(&mut self, scene: &RenderInput<'_>, viewport: [u32; 4]) -> Result<()> {
        self.place_retained()?;
        self.stream_terrain_pages()?;
        self.instance_data.clear();
        self.draw_calls.clear();
        self.shadow_calls.clear();
        self.matrix.read_stats(self.frame);
        self.matrix.set_active(self.matrix_enabled);
        self.material_data.clear();
        self.material_data
            .extend(self.materials.iter().map(|m| m.gpu));
        // Model indices only change with the models: copying them every frame
        // cost Sponza 18 ms, so they are built once and kept as the prefix.
        let key = (self.model_generation, self.matrix_enabled);
        if self.model_indices_key != Some(key) {
            self.frame_indices.clear();
            self.model_index_ranges.clear();
            let mut ranges = std::collections::HashMap::<(usize, usize), (u32, u32)>::new();
            for draw in &self.model_draws {
                let primitive = self.primitives[draw.primitive as usize];
                let material = &self.materials[primitive.material as usize];
                let primitive_index = draw.primitive as usize;
                if material.hidden || (self.matrix_enabled && self.matrix.owns(primitive_index)) {
                    self.model_index_ranges.push(None);
                    continue;
                }
                let start = primitive.first_index as usize;
                let end = start + primitive.index_count as usize;
                let range = append_lod_range(
                    &mut self.frame_indices,
                    &mut ranges,
                    (primitive_index, 0),
                    &self.source_indices[start..end],
                )
                .ok_or_else(|| failure("too many frame geometry indices"))?;
                self.model_index_ranges.push(Some(range));
            }
            self.model_indices_len = self.frame_indices.len();
            self.model_indices_key = Some(key);
        }
        self.frame_indices.truncate(self.model_indices_len);
        let mut ranges = std::collections::HashMap::<(usize, usize), (u32, u32)>::new();
        for (index, (draw, range)) in self.model_draws.iter().zip(&self.model_index_ranges).enumerate() {
            let Some((first, index_count)) = *range else {
                continue;
            };
            let primitive = self.primitives[draw.primitive as usize];
            let material = &self.materials[primitive.material as usize];
            let mut instance = scene::instance(
                draw.transform,
                primitive.first_index,
                primitive.base_vertex,
                primitive.material,
            );
            instance.draw[0] = first;
            // The stable draw index extensions know from `scene_loaded`.
            instance.draw[3] = index as u32;
            self.instance_data.push(instance);
            self.draw_calls
                .push((index_count, material.double_sided, draw.primitive as usize));
        }
        // World entities are unit cubes, each with its own plain material.
        let cube = self.primitives[0];
        let model_draws = self.model_draws.len() as u32;
        for (entity, (transform, material)) in scene.objects().enumerate() {
            let Some(model) = scene::entity_matrix(transform) else {
                continue;
            };
            let index = self.material_data.len() as u32;
            self.material_data.push(scene::GpuMaterial::plain(material));
            let start = cube.first_index as usize;
            let end = start + cube.index_count as usize;
            let (first, _) = append_lod_range(
                &mut self.frame_indices,
                &mut ranges,
                (0, 0),
                &self.source_indices[start..end],
            )
            .ok_or_else(|| failure("too many frame geometry indices"))?;
            let mut instance = scene::instance(model, first, cube.base_vertex, index);
            instance.draw[3] = model_draws + entity as u32;
            self.instance_data.push(instance);
            self.draw_calls.push((cube.index_count, false, 0));
        }
        self.deforming_vertices.clear();
        let needed: usize = self.deforming.iter().map(|m| m.vertices.len()).sum();
        if needed > self.deforming_capacity as usize {
            let capacity = needed.next_power_of_two();
            self.device.device_wait_idle()?;
            self.deforming_base = self.vertex_count;
            self.append_geometry(&vec![bytemuck::Zeroable::zeroed(); capacity], &[])?;
            self.deforming_capacity = capacity as u32;
            for index in 0..FRAMES {
                self.write_buffer_descriptors(index);
            }
        }
        let entities = self.instance_data.len() as u32;
        self.previous_positions.clear();
        self.deforming_motion.clear();
        for (slot, mesh) in self.deforming.iter().enumerate() {
            // A mesh whose vertex count changed has no usable last frame: it holds still.
            self.deforming_motion
                .push((self.instance_data.len(), self.previous_positions.len() as u32));
            match self.previous_deforming.get(slot) {
                Some(last) if last.len() == mesh.vertices.len() => {
                    self.previous_positions.extend_from_slice(last)
                }
                _ => self
                    .previous_positions
                    .extend(mesh.vertices.iter().map(|v| v.position)),
            }
            let index = self.material_data.len() as u32;
            let plain = || scene::GpuMaterial::plain(&mesh.material);
            self.material_data.push(mesh.textured.unwrap_or_else(plain));
            let first = u32::try_from(self.frame_indices.len())
                .map_err(|_| failure("too many frame geometry indices"))?;
            let base = self.deforming_base + self.deforming_vertices.len() as u32;
            self.frame_indices.extend_from_slice(&mesh.indices);
            self.deforming_vertices.extend_from_slice(&mesh.vertices);
            let mut instance = scene::instance(glam::Mat4::IDENTITY, first, base, index);
            instance.draw[3] = entities + slot as u32;
            self.instance_data.push(instance);
            self.draw_calls
                .push((mesh.indices.len() as u32, true, NO_PRIMITIVE));
        }
        self.embedded_bounds.clear();
        let mut slot = entities + self.deforming.len() as u32;
        for draw in self.embedded.iter().flat_map(|e| e.draws()) {
            let index = self.material_data.len() as u32;
            self.material_data.push(draw.material);
            // Instances share one copy of the indices: only their vertices differ.
            let first = u32::try_from(self.frame_indices.len())
                .map_err(|_| failure("too many frame geometry indices"))?;
            self.frame_indices.extend_from_slice(draw.indices);
            for (base, transform, bounds) in draw.instances {
                let mut instance = scene::instance(transform, first, base, index);
                instance.draw[3] = slot;
                slot += 1;
                self.instance_data.push(instance);
                let primitive = EMBEDDED_PRIMITIVE + self.embedded_bounds.len();
                self.draw_calls
                    .push((draw.indices.len() as u32, false, primitive));
                self.embedded_bounds.push(bounds);
            }
        }
        self.push_retained_draws(slot, viewport, scene.sun().or(self.sun).as_ref())?;
        // All casters are retained, including those outside the camera.
        self.shadow_calls.extend(
            self.draw_calls
                .iter()
                .enumerate()
                .map(|(index, &(count, _, _))| (index as u32, count)),
        );
        self.frame_lights.clear();
        self.frame_lights
            .extend(self.scene_lights.iter().chain(scene.lights()).cloned());
        self.light_data.clear();
        self.light_data.extend(
            self.scene_lights
                .iter()
                .chain(scene.lights())
                .map(scene::light),
        );
        self.frame_sun = scene.sun().or(self.sun);
        let lighting = scene.lighting();
        self.frame_clouds = lighting.clouds;
        let (sky, ground) = if lighting.method == verity_core::LightingMethod::DirectOnly {
            (glam::Vec3::ZERO, glam::Vec3::ZERO)
        } else {
            self.ambient_override.unwrap_or_else(|| {
                (
                    glam::Vec3::from_array(lighting.sky),
                    glam::Vec3::from_array(lighting.ground),
                )
            })
        };
        self.frame_ambient = (sky, ground);
        let mut globals = scene::globals(
            &self.camera,
            &scene::Lighting {
                sun: self.frame_sun.as_ref(),
                sky,
                ground,
                shadow_distance: self.shadow_distance,
            },
            viewport,
            [
                self.light_data.len() as u32,
                self.instance_data.len() as u32,
                self.extension.as_ref().map_or(0, |e| {
                    u32::from(e.active()) | u32::from(e.active() && e.sky_reflections()) << 1
                }),
            ],
            self.previous_view_proj,
            self.swap.srgb,
        );
        globals.counts[3] = u32::from(self.matrix_debug);
        if self.frame_clouds.is_some() && self.frame_sun.is_some() {
            globals.cloud_shadow = clouds::shadow_region(self.camera.eye);
        }
        for (slot, light) in self.frame_lights.iter().take(MAX_POINT_SHADOWS).enumerate() {
            if light.radius.is_finite()
                && light.radius > 0.0
                && light.position.iter().all(|value| value.is_finite())
            {
                for (face, matrix) in scene::point_shadow_matrices(light).into_iter().enumerate() {
                    globals.point_shadow_view_proj[slot * scene::POINT_SHADOW_FACES + face] =
                        matrix.to_cols_array_2d();
                }
            }
        }
        // Moving geometry has no stable key: every layer it may reach redraws.
        let moving = self.matrix_enabled
            || !self.deforming.is_empty()
            || !self.retained.copies.is_empty()
            || self.embedded.as_ref().is_some_and(|e| e.reshaping());
        self.plan_shadows(&mut globals, viewport, moving);
        let unjittered = glam::Mat4::from_cols_array_2d(&globals.view_proj);
        let jitter = if !self.anti_aliasing.temporal() {
            glam::Mat4::IDENTITY
        } else {
            // Sub-pixel offset in NDC.
            let [x, y] = taa::jitter(self.frame_number, self.anti_aliasing);
            let size = [viewport[2].max(1) as f32, viewport[3].max(1) as f32];
            glam::Mat4::from_translation(glam::Vec3::new(2.0 * x / size[0], 2.0 * y / size[1], 0.0))
        };
        let view_proj = jitter * unjittered;
        // Last frame seen through this frame's jitter: surfaces reconstructed
        // from jittered pixels then get motion without the jitter in it, which
        // otherwise shifts the history by the jitter every frame.
        let previous = jitter * self.previous_view_proj.unwrap_or(unjittered);
        globals.view_proj = view_proj.to_cols_array_2d();
        globals.inverse_view_proj = view_proj.inverse().to_cols_array_2d();
        globals.previous_view_proj = previous.to_cols_array_2d();
        self.frame_view_proj = (view_proj, previous);
        self.previous_view_proj = Some(unjittered);
        self.culling_ranges.clear();
        self.occluders.clear();
        let mut jobs = Vec::new();
        let use_culling = self.matrix_enabled && self.culler.is_some();
        for (instance, &(count, _, primitive)) in
            self.draw_calls.iter().enumerate().filter(|_| use_culling)
        {
            let record = &self.instance_data[instance];
            let model = glam::Mat4::from_cols_array_2d(&record.model);
            let start = jobs.len();
            let masked = self.material_data[record.draw[2] as usize].textures[3] & scene::ALPHA_MASK != 0;
            let mut bounds: Option<(glam::Vec3, glam::Vec3)> = None;
            if primitive == NO_PRIMITIVE {
                // Unculled: its bounds change every frame.
            } else if primitive >= EMBEDDED_PRIMITIVE {
                let (min, max) = self.embedded_bounds[primitive - EMBEDDED_PRIMITIVE];
                bounds = Some((min, max));
                jobs.push(gpu_culling::Job::new(
                    view_proj,
                    model,
                    min.to_array(),
                    max.to_array(),
                    0,
                    count,
                ));
            } else if let Some(meshlets) = &self.primitive_meshlets[primitive] {
                for meshlet in meshlets {
                    let (min, max) = (
                        glam::Vec3::from_array(meshlet.bounds.min),
                        glam::Vec3::from_array(meshlet.bounds.max),
                    );
                    bounds = Some(bounds.map_or((min, max), |(lo, hi)| (lo.min(min), hi.max(max))));
                    jobs.push(gpu_culling::Job::new(
                        view_proj,
                        model,
                        meshlet.bounds.min,
                        meshlet.bounds.max,
                        meshlet.first_triangle * 3,
                        meshlet.triangle_count * 3,
                    ));
                }
            }
            self.culling_ranges
                .push((jobs.len() > start).then_some((start, jobs.len())));
            self.occluders.push(bounds.is_some_and(|(min, max)| {
                !masked && is_occluder(model, min, max, self.camera.eye)
            }));
        }
        self.culling_active = self.matrix_enabled
            && self.culler.as_mut().is_some_and(|culler| {
                culler
                    .prepare(
                        &self.device,
                        self.allocator.as_mut().unwrap(),
                        self.frame,
                        &jobs,
                    )
                    .unwrap_or(false)
            });
        self.hiz_active = self.culling_active
            && self.depth_sampled
            && !jobs.is_empty()
            && viewport[2] > 0
            && viewport[3] > 0
            && self.culler.as_mut().is_some_and(|culler| {
                culler
                    .prepare_hiz(
                        &self.device,
                        self.allocator.as_mut().unwrap(),
                        self.frame,
                        self.swap.depth.as_ref().unwrap().view,
                        self.swap.extent,
                    )
                    .unwrap_or(false)
            });

        self.track_instance_motion();
        let device = &self.device;
        let allocator = self.allocator.as_mut().unwrap();
        let frame = &mut self.frames[self.frame];
        // This frame's fence has signaled, so nothing still reads the arena.
        let needed = gpu::FrameArena::needed([
            mem::size_of_val(self.instance_data.as_slice()),
            mem::size_of_val(self.light_data.as_slice()),
            mem::size_of_val(self.motion_data.as_slice()),
            mem::size_of_val(self.previous_positions.as_slice()),
        ]);
        frame.arena.reset(device, allocator, needed)?;
        let arena = &mut frame.arena;
        let full = || failure("frame arena smaller than `needed`");
        globals.instances = arena.push(&self.instance_data).ok_or_else(full)?;
        globals.lights = arena.push(&self.light_data).ok_or_else(full)?;
        globals.instance_motion = arena.push(&self.motion_data).ok_or_else(full)?;
        globals.previous_positions = arena.push(&self.previous_positions).ok_or_else(full)?;
        let mut grown = false;
        // The model prefix is rewritten only into frames that lack this one.
        let kept = if frame.indices_prefix == self.model_indices_key {
            self.model_indices_len * mem::size_of::<u32>()
        } else {
            0
        };
        frame.indices_prefix = self.model_indices_key;
        for (buffer, name, bytes, mut skip) in [
            (
                &mut frame.indices,
                "frame indices",
                bytemuck::cast_slice::<_, u8>(&self.frame_indices),
                kept,
            ),
            (
                &mut frame.materials,
                "materials",
                bytemuck::cast_slice(&self.material_data),
                0,
            ),
        ] {
            if bytes.len() as u64 > buffer.size {
                skip = 0;
                // This frame's fence has signaled, so nothing still reads the old buffer.
                gpu::destroy_buffer(device, allocator, mem::take(buffer));
                let size = (bytes.len() as u64).next_power_of_two();
                *buffer = gpu::create_buffer(
                    device,
                    allocator,
                    name,
                    size,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    MemoryLocation::CpuToGpu,
                )?;
                grown = true;
            }
            buffer.write_at(skip, &bytes[skip..]);
        }
        // Written last: growing a buffer above moves its address.
        globals.vertices = gpu::DeviceAddress::new(self.vertices.address);
        globals.indices = gpu::DeviceAddress::new(frame.indices.address);
        globals.matrix = self.matrix.addresses();
        frame.globals.write(bytemuck::bytes_of(&globals));
        for (buffer, name, vertices) in [
            (
                &mut frame.deforming,
                "deforming vertices",
                &self.deforming_vertices,
            ),
            (
                &mut frame.retained,
                "retained vertices",
                &self.retained.staging,
            ),
        ] {
            let bytes: &[u8] = bytemuck::cast_slice(vertices);
            if bytes.len() as u64 > buffer.size {
                gpu::destroy_buffer(device, allocator, mem::take(buffer));
                *buffer = gpu::create_buffer(
                    device,
                    allocator,
                    name,
                    (bytes.len() as u64).next_power_of_two(),
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    MemoryLocation::CpuToGpu,
                )?;
            }
            if !bytes.is_empty() {
                buffer.write(bytes);
            }
        }
        if let Some(embedded) = &mut self.embedded {
            embedded.prepare(device, allocator, self.frame, &self.vertices)?;
        }
        if grown {
            self.write_buffer_descriptors(self.frame);
        }
        Ok(())
    }

    /// This frame's deforming vertices as a copy, if there are any.
    fn deforming_copy(&self) -> Vec<vk::BufferCopy> {
        let stride = mem::size_of::<scene::Vertex>() as u64;
        (!self.deforming_vertices.is_empty())
            .then(|| vk::BufferCopy {
                src_offset: 0,
                dst_offset: self.deforming_base as u64 * stride,
                size: self.deforming_vertices.len() as u64 * stride,
            })
            .into_iter()
            .collect()
    }

    /// Copy staged vertices into the shared vertex buffer. The first barrier
    /// also waits for the previous frame, which may still read those ranges.
    unsafe fn record_vertex_copies(
        &self,
        cmd: vk::CommandBuffer,
        source: vk::Buffer,
        copies: &[vk::BufferCopy],
    ) {
        if copies.is_empty() {
            return;
        }
        use vk::AccessFlags2 as A;
        use vk::PipelineStageFlags2 as S;
        let barrier = |src_stage, src_access, dst_stage, dst_access| {
            let (src, dst) = ((src_stage, src_access), (dst_stage, dst_access));
            gpu::memory_barrier(&self.device, cmd, src, dst);
        };
        barrier(S::ALL_COMMANDS, A::MEMORY_READ, S::COPY, A::TRANSFER_WRITE);
        self.device
            .cmd_copy_buffer(cmd, source, self.vertices.buffer, copies);
        barrier(S::COPY, A::TRANSFER_WRITE, S::ALL_COMMANDS, A::MEMORY_READ);
    }

    /// Bring `meshlet_runs` up to this frame's draws and camera. Culling 100k+
    /// meshlets each frame cost Sponza 2-5 ms, so only draws that moved, or a
    /// camera that did, cull again, in chunks across threads.
    fn cull_meshlets(&mut self) {
        // Chunks of meshlets, so a few huge draws still spread across threads.
        const CHUNK: usize = 1024;
        // GPU culling takes over draws, which then keep no runs here.
        let key = (self.frame_view_proj.0, self.model_generation, self.culling_active);
        if self.meshlet_runs_key != Some(key) {
            self.meshlet_runs.clear();
            self.meshlet_runs_key = Some(key);
        }
        self.meshlet_runs
            .resize_with(self.draw_calls.len(), MeshletRuns::default);
        let mut chunks = Vec::new();
        for (instance, cached) in self.meshlet_runs.iter_mut().enumerate() {
            let primitive = self.draw_calls[instance].2;
            let model = self.instance_data[instance].model;
            if cached.draw == Some((primitive, model)) {
                continue;
            }
            cached.draw = Some((primitive, model));
            cached.runs.clear();
            let culled_on_gpu = self.culling_active
                && self.culling_ranges.get(instance).is_some_and(Option::is_some);
            let meshlets = self.primitive_meshlets.get(primitive).and_then(Option::as_ref);
            if let Some(meshlets) = meshlets.filter(|_| !culled_on_gpu) {
                let model = glam::Mat4::from_cols_array_2d(&model);
                chunks.extend(meshlets.chunks(CHUNK).map(|chunk| (instance, chunk, model)));
            }
        }
        let clip = self.frame_view_proj.0;
        use rayon::prelude::*;
        let culled: Vec<Vec<(u32, u32)>> = chunks
            .par_iter()
            .map(|&(_, chunk, model)| {
                let mut runs = Vec::new();
                meshlets::visible_runs(chunk, clip, model, |first, count| {
                    runs.push((first, count));
                });
                runs
            })
            .collect();
        for (&(instance, _, _), runs) in chunks.iter().zip(culled) {
            let all = &mut self.meshlet_runs[instance].runs;
            for (first, count) in runs {
                // Join runs that a chunk boundary split.
                match all.last_mut() {
                    Some((start, length)) if *start + *length == first => *length += count,
                    _ => all.push((first, count)),
                }
            }
        }
    }

    /// Clusters, visibility and resolve: direct light into HDR plus the G-buffer.
    unsafe fn record_scene(
        &mut self,
        cmd: vk::CommandBuffer,
        set: vk::DescriptorSet,
        viewport: [u32; 4],
    ) -> Result<()> {
        use vk::AccessFlags2 as A;
        use vk::ImageLayout as L;
        use vk::PipelineStageFlags2 as S;
        self.record_bakes(cmd, set);
        let culling = Instant::now();
        self.cull_meshlets();
        let culling = culling.elapsed();
        let device = &self.device;
        let extent = self.swap.extent;
        let (depth, visibility, hdr, gbuffer, receiver, motion) = (
            self.swap.depth.as_ref().unwrap(),
            self.swap.visibility.as_ref().unwrap(),
            self.swap.hdr.as_ref().unwrap(),
            self.swap.gbuffer.as_ref().unwrap(),
            self.swap.receiver.as_ref().unwrap(),
            self.swap.motion.as_ref().unwrap(),
        );
        let color = vk::ImageAspectFlags::COLOR;
        let barrier = |images: &[vk::ImageMemoryBarrier2], memory: &[vk::MemoryBarrier2]| {
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(images)
                    .memory_barriers(memory),
            );
        };
        let memory = |src_stage, src_access, dst_stage, dst_access| {
            vk::MemoryBarrier2::default()
                .src_stage_mask(src_stage)
                .src_access_mask(src_access)
                .dst_stage_mask(dst_stage)
                .dst_access_mask(dst_access)
        };
        for bind in [
            vk::PipelineBindPoint::COMPUTE,
            vk::PipelineBindPoint::GRAPHICS,
        ] {
            device.cmd_bind_descriptor_sets(cmd, bind, self.pipeline_layout, 0, &[set], &[]);
        }
        let [x, y, width, height] = viewport;
        let area = vk::Rect2D {
            offset: vk::Offset2D {
                x: x as i32,
                y: y as i32,
            },
            extent: vk::Extent2D { width, height },
        };
        let visible = width > 0 && height > 0;
        let timer = self.gpu_timer.as_ref().filter(|_| self.profiling);
        let mut stages = [Duration::ZERO; 5];
        let mut last = Instant::now();
        // Ends sub-stage `i` (0-based) on the CPU and GPU timelines.
        let mut split = |i: usize| {
            let now = Instant::now();
            stages[i] = now - last;
            last = now;
            if let Some(timer) = timer {
                timer.stamp(device, cmd, self.frame, i as u32 + 2);
            }
        };
        self.matrix
            .record_select(device, cmd, self.pipeline_layout, self.frame);

        // Bin lights. The previous frame's resolve may still read the clusters.
        barrier(
            &[],
            &[memory(
                S::COMPUTE_SHADER,
                A::SHADER_STORAGE_READ,
                S::COMPUTE_SHADER,
                A::SHADER_STORAGE_WRITE,
            )],
        );
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.cluster_pipeline);
        device.cmd_dispatch(
            cmd,
            (scene::CLUSTERS * scene::LIGHT_WORDS).div_ceil(64) as u32,
            1,
            1,
        );
        if self.culling_active {
            self.culler
                .as_ref()
                .unwrap()
                .record(device, cmd, self.frame);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[set],
                &[],
            );
        }
        split(0);

        // Both images are shared across frames. The queue dependency protects the
        // prior resolve's sampled reads before discarding and rewriting a layer.
        // Layers that did not change stay read-only from the frame that drew them.
        for (map, layers, size, first) in [
            (
                self.shadow_map.as_ref().unwrap(),
                &self.shadow_layers,
                scene::SHADOW_SIZE,
                0,
            ),
            (
                self.point_shadow_map.as_ref().unwrap(),
                &self.point_shadow_layers,
                scene::POINT_SHADOW_SIZE,
                scene::CASCADES,
            ),
        ] {
            let redraw: Vec<usize> = (0..layers.len())
                .filter(|&index| self.shadows.redraw(first + index))
                .collect();
            if redraw.is_empty() {
                continue;
            }
            let layer_barriers = |src_stage, src_access, dst_stage, dst_access, old, new| {
                redraw
                    .iter()
                    .map(|&index| {
                        gpu::image_barrier(
                            map.image,
                            vk::ImageAspectFlags::DEPTH,
                            src_stage,
                            src_access,
                            dst_stage,
                            dst_access,
                            old,
                            new,
                        )
                        .subresource_range(
                            gpu::subresource(vk::ImageAspectFlags::DEPTH)
                                .base_array_layer(index as u32)
                                .layer_count(1),
                        )
                    })
                    .collect::<Vec<_>>()
            };
            barrier(
                &layer_barriers(
                    S::COMPUTE_SHADER,
                    A::SHADER_SAMPLED_READ,
                    S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
                    A::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    L::UNDEFINED,
                    L::DEPTH_ATTACHMENT_OPTIMAL,
                ),
                &[],
            );
            let shadow_area = vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: size,
                    height: size,
                },
            };
            for &index in &redraw {
                let layer = first + index;
                let attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(layers[index])
                    .image_layout(L::DEPTH_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    });
                device.cmd_begin_rendering(
                    cmd,
                    &vk::RenderingInfo::default()
                        .render_area(shadow_area)
                        .layer_count(1)
                        .depth_attachment(&attachment),
                );
                if self.shadows.enabled(layer) {
                    set_viewport(device, cmd, shadow_area);
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.shadow_pipeline,
                    );
                    for &(instance, index_count) in self.shadows.calls(layer) {
                        let push = [instance, layer as u32];
                        device.cmd_push_constants(
                            cmd,
                            self.pipeline_layout,
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            0,
                            bytemuck::cast_slice(&push),
                        );
                        device.cmd_draw(cmd, index_count, 1, 0, 0);
                    }
                    self.matrix
                        .record_shadows(device, cmd, self.pipeline_layout, layer as u32);
                }
                device.cmd_end_rendering(cmd);
            }
            barrier(
                &layer_barriers(
                    S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
                    A::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    S::COMPUTE_SHADER,
                    A::SHADER_SAMPLED_READ,
                    L::DEPTH_ATTACHMENT_OPTIMAL,
                    L::DEPTH_READ_ONLY_OPTIMAL,
                ),
                &[],
            );
        }
        split(1);
        if self.hiz_active {
            barrier(
                &[gpu::image_barrier(
                    depth.image,
                    vk::ImageAspectFlags::DEPTH,
                    S::ALL_COMMANDS,
                    A::NONE,
                    S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
                    A::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    L::UNDEFINED,
                    L::DEPTH_ATTACHMENT_OPTIMAL,
                )],
                &[],
            );
            let depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(depth.view)
                .image_layout(L::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 0.0,
                        stencil: 0,
                    },
                });
            device.cmd_begin_rendering(
                cmd,
                &vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent,
                    })
                    .layer_count(1)
                    .depth_attachment(&depth_attachment),
            );
            set_viewport(device, cmd, area);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.depth_prepass);
            let mut bound = None;
            // Only large opaque occluders: Hi-Z culling needs the big shapes,
            // and the visibility pass draws everything anyway.
            let occluders = (self.draw_calls.iter().enumerate())
                .filter(|&(instance, _)| self.occluders.get(instance) == Some(&true));
            for (instance, &(count, two_sided, _)) in occluders {
                if bound != Some(two_sided) {
                    device.cmd_set_cull_mode(cmd, cull_sides(two_sided));
                    bound = Some(two_sided);
                }
                device.cmd_push_constants(
                    cmd,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    &(instance as u32).to_ne_bytes(),
                );
                device.cmd_draw(cmd, count, 1, 0, 0);
            }
            self.matrix.record_draws(
                device,
                cmd,
                self.pipeline_layout,
                matrix::Target::Depth,
            );
            device.cmd_end_rendering(cmd);
            barrier(
                &[gpu::image_barrier(
                    depth.image,
                    vk::ImageAspectFlags::DEPTH,
                    S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
                    A::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    S::COMPUTE_SHADER,
                    A::SHADER_SAMPLED_READ,
                    L::DEPTH_ATTACHMENT_OPTIMAL,
                    L::DEPTH_READ_ONLY_OPTIMAL,
                )],
                &[],
            );
            self.culler
                .as_ref()
                .unwrap()
                .record_occlusion(device, cmd, self.frame, extent, viewport);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[set],
                &[],
            );
            barrier(
                &[gpu::image_barrier(
                    depth.image,
                    vk::ImageAspectFlags::DEPTH,
                    S::COMPUTE_SHADER,
                    A::SHADER_SAMPLED_READ,
                    S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
                    A::DEPTH_STENCIL_ATTACHMENT_READ | A::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    L::DEPTH_READ_ONLY_OPTIMAL,
                    L::DEPTH_ATTACHMENT_OPTIMAL,
                )],
                &[],
            );
        }
        split(2);

        // Visibility: depth plus (instance, triangle) IDs.
        let msaa = self.swap.visibility_msaa.as_ref().zip(self.swap.depth_msaa.as_ref());
        if let Some((ids, depths)) = msaa {
            let attachment = |image, aspect, stage, access, layout| {
                gpu::image_barrier(image, aspect, S::ALL_COMMANDS, A::NONE, stage, access, L::UNDEFINED, layout)
            };
            barrier(
                &[
                    attachment(
                        ids.image,
                        color,
                        S::COLOR_ATTACHMENT_OUTPUT,
                        A::COLOR_ATTACHMENT_WRITE,
                        L::COLOR_ATTACHMENT_OPTIMAL,
                    ),
                    attachment(
                        depths.image,
                        vk::ImageAspectFlags::DEPTH,
                        S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
                        A::DEPTH_STENCIL_ATTACHMENT_READ | A::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        L::DEPTH_ATTACHMENT_OPTIMAL,
                    ),
                ],
                &[],
            );
        }
        barrier(
            &[
                gpu::image_barrier(
                    visibility.image,
                    color,
                    S::COMPUTE_SHADER,
                    A::NONE,
                    S::COLOR_ATTACHMENT_OUTPUT,
                    A::COLOR_ATTACHMENT_WRITE,
                    L::UNDEFINED,
                    L::COLOR_ATTACHMENT_OPTIMAL,
                ),
                gpu::image_barrier(
                    depth.image,
                    vk::ImageAspectFlags::DEPTH,
                    S::LATE_FRAGMENT_TESTS,
                    A::NONE,
                    S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
                    A::DEPTH_STENCIL_ATTACHMENT_READ | A::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    if self.hiz_active {
                        L::DEPTH_ATTACHMENT_OPTIMAL
                    } else {
                        L::UNDEFINED
                    },
                    L::DEPTH_ATTACHMENT_OPTIMAL,
                ),
            ],
            &[],
        );
        let full = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent,
        };
        let mut id_attachment = [vk::RenderingAttachmentInfo::default()
            .image_view(visibility.view)
            .image_layout(L::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    uint32: [u32::MAX; 4],
                },
            })];
        let mut depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(depth.view)
            .image_layout(L::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            });
        if let Some((ids, depths)) = msaa {
            // Sample 0 lands in the single-sample targets every other pass reads.
            let zero = vk::ResolveModeFlags::SAMPLE_ZERO;
            id_attachment[0] = id_attachment[0]
                .image_view(ids.view)
                .resolve_mode(zero)
                .resolve_image_view(visibility.view)
                .resolve_image_layout(L::COLOR_ATTACHMENT_OPTIMAL);
            depth_attachment = depth_attachment
                .image_view(depths.view)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .resolve_mode(zero)
                .resolve_image_view(depth.view)
                .resolve_image_layout(L::DEPTH_ATTACHMENT_OPTIMAL);
        }
        device.cmd_begin_rendering(
            cmd,
            &vk::RenderingInfo::default()
                .render_area(full)
                .layer_count(1)
                .color_attachments(&id_attachment)
                .depth_attachment(&depth_attachment),
        );
        if visible {
            set_viewport(device, cmd, area);
            let pipeline = if msaa.is_some() {
                self.visibility_msaa_pipeline
            } else {
                self.visibility_pipeline
            };
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            let mut bound = None;
            for (instance, &(index_count, two_sided, primitive)) in
                self.draw_calls.iter().enumerate()
            {
                if bound != Some(two_sided) {
                    device.cmd_set_cull_mode(cmd, cull_sides(two_sided));
                    bound = Some(two_sided);
                }
                device.cmd_push_constants(
                    cmd,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    &(instance as u32).to_ne_bytes(),
                );
                if self.culling_active && self.culling_ranges[instance].is_some() {
                    let (start, end) = self.culling_ranges[instance].unwrap();
                    let culler = self.culler.as_ref().unwrap();
                    for job in start..end {
                        culler.draw(device, cmd, self.frame, job);
                    }
                } else if (self.primitive_meshlets.get(primitive)).is_some_and(Option::is_some) {
                    for &(first, count) in &self.meshlet_runs[instance].runs {
                        device.cmd_draw(cmd, count, 1, first, 0);
                    }
                } else {
                    device.cmd_draw(cmd, index_count, 1, 0, 0);
                }
            }
            let target = if msaa.is_some() {
                matrix::Target::VisibilityMsaa
            } else {
                matrix::Target::Visibility
            };
            self.matrix
                .record_draws(device, cmd, self.pipeline_layout, target);
        }
        device.cmd_end_rendering(cmd);
        split(3);

        // Resolve and shade into the HDR target.
        barrier(
            &[
                gpu::image_barrier(
                    visibility.image,
                    color,
                    S::COLOR_ATTACHMENT_OUTPUT,
                    A::COLOR_ATTACHMENT_WRITE,
                    S::COMPUTE_SHADER,
                    A::SHADER_SAMPLED_READ,
                    L::COLOR_ATTACHMENT_OPTIMAL,
                    L::SHADER_READ_ONLY_OPTIMAL,
                ),
                gpu::image_barrier(
                    hdr.image,
                    color,
                    S::FRAGMENT_SHADER,
                    A::NONE,
                    S::COMPUTE_SHADER,
                    A::SHADER_STORAGE_WRITE,
                    L::UNDEFINED,
                    L::GENERAL,
                ),
                // Last frame's extension may still read the G-buffer and depth.
                gpu::image_barrier(
                    gbuffer.image,
                    color,
                    S::COMPUTE_SHADER,
                    A::NONE,
                    S::COMPUTE_SHADER,
                    A::SHADER_STORAGE_WRITE,
                    L::UNDEFINED,
                    L::GENERAL,
                ),
                gpu::image_barrier(
                    receiver.image,
                    color,
                    S::COMPUTE_SHADER,
                    A::NONE,
                    S::COMPUTE_SHADER,
                    A::SHADER_STORAGE_WRITE,
                    L::UNDEFINED,
                    L::GENERAL,
                ),
                gpu::image_barrier(
                    motion.image,
                    color,
                    S::COMPUTE_SHADER,
                    A::NONE,
                    S::COMPUTE_SHADER,
                    A::SHADER_STORAGE_WRITE,
                    L::UNDEFINED,
                    L::GENERAL,
                ),
                gpu::image_barrier(
                    depth.image,
                    vk::ImageAspectFlags::DEPTH,
                    // Multisample resolves write in the color output stage.
                    S::LATE_FRAGMENT_TESTS | S::COLOR_ATTACHMENT_OUTPUT,
                    A::DEPTH_STENCIL_ATTACHMENT_WRITE | A::COLOR_ATTACHMENT_WRITE,
                    S::COMPUTE_SHADER,
                    A::SHADER_SAMPLED_READ,
                    L::DEPTH_ATTACHMENT_OPTIMAL,
                    L::DEPTH_READ_ONLY_OPTIMAL,
                ),
            ],
            // Storage writes too: the exposure sums are atomics after last
            // frame's reset.
            &[memory(
                S::COMPUTE_SHADER,
                A::SHADER_STORAGE_WRITE,
                S::COMPUTE_SHADER,
                A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE,
            )],
        );
        if let Some((ids, _)) = msaa {
            barrier(
                &[gpu::image_barrier(
                    ids.image,
                    color,
                    S::COLOR_ATTACHMENT_OUTPUT,
                    A::COLOR_ATTACHMENT_WRITE,
                    S::COMPUTE_SHADER,
                    A::SHADER_SAMPLED_READ,
                    L::COLOR_ATTACHMENT_OPTIMAL,
                    L::SHADER_READ_ONLY_OPTIMAL,
                )],
                &[],
            );
        }
        if visible {
            let resolve = if msaa.is_some() {
                self.resolve_msaa_pipeline
            } else {
                self.resolve_pipeline
            };
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, resolve);
            device.cmd_dispatch(cmd, width.div_ceil(8), height.div_ceil(8), 1);
            if self.terrain_layers.is_some() {
                // The CPU reads the wanted pages once this frame's fence signals.
                let written = (S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE);
                gpu::memory_barrier(device, cmd, written, (S::HOST, A::HOST_READ));
            }
        }
        if self.extension.is_some() {
            // Extensions read the resolve's outputs and add to HDR in compute.
            let read_write =
                A::SHADER_STORAGE_READ | A::SHADER_SAMPLED_READ | A::SHADER_STORAGE_WRITE;
            let written = (S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE);
            gpu::memory_barrier(device, cmd, written, (S::COMPUTE_SHADER, read_write));
        }
        split(4);
        // Culling ran before the timeline started; it belongs to visibility.
        stages[3] += culling;
        self.scene_stages = stages;
        Ok(())
    }

    /// Run the extension, if any, between the resolve and the composite.
    unsafe fn record_extension(
        &mut self,
        cmd: vk::CommandBuffer,
        set: vk::DescriptorSet,
        viewport: [u32; 4],
    ) -> Result<()> {
        let (Some(targets), Some(mut extension)) = (self.targets(), self.extension.take()) else {
            return Ok(());
        };
        let lights = mem::take(&mut self.frame_lights);
        let frame = FrameContext {
            cmd,
            frame_set: set,
            frame_slot: self.frame,
            frame_number: self.frame_number,
            targets,
            viewport,
            eye: self.camera.eye,
            view_proj: self.frame_view_proj.0,
            previous_view_proj: self.frame_view_proj.1,
            sun: self.frame_sun,
            lights: &lights,
            sky: self.frame_ambient.0,
            ground: self.frame_ambient.1,
            near: scene::NEAR,
        };
        let result = extension.record(&mut self.gpu(), &frame);
        self.frame_lights = lights;
        self.extension = Some(extension);
        result
    }

    /// Tonemap the HDR target and draw the UI into the swapchain image.
    unsafe fn record_present(
        &mut self,
        cmd: vk::CommandBuffer,
        set: vk::DescriptorSet,
        image: usize,
        viewport: [u32; 4],
        ui: Option<&UiFrame>,
    ) -> Result<()> {
        use vk::AccessFlags2 as A;
        use vk::ImageLayout as L;
        use vk::PipelineStageFlags2 as S;
        self.record_bakes(cmd, set);
        let device = &self.device;
        let extent = self.swap.extent;
        let hdr = self.swap.hdr.as_ref().unwrap();
        let color = vk::ImageAspectFlags::COLOR;
        let barrier = |images: &[vk::ImageMemoryBarrier2], memory: &[vk::MemoryBarrier2]| {
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(images)
                    .memory_barriers(memory),
            );
        };
        // An extension may have bound its own graphics state.
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline_layout,
            0,
            &[set],
            &[],
        );
        let [x, y, width, height] = viewport;
        let area = vk::Rect2D {
            offset: vk::Offset2D {
                x: x as i32,
                y: y as i32,
            },
            extent: vk::Extent2D { width, height },
        };
        let visible = width > 0 && height > 0;
        let full = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent,
        };

        // Adapt the exposure to this frame's metering before the composite applies it.
        let now = Instant::now();
        let seconds = self
            .last_present
            .map_or(0.0, |last| now.duration_since(last).as_secs_f32());
        self.last_present = Some(now);
        let read_write = A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE;
        gpu::memory_barrier(
            device,
            cmd,
            (S::COMPUTE_SHADER | S::FRAGMENT_SHADER, read_write),
            (S::COMPUTE_SHADER, read_write),
        );
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline_layout,
            0,
            &[set],
            &[],
        );
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.adapt_pipeline);
        device.cmd_push_constants(
            cmd,
            self.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            8,
            &seconds.to_ne_bytes(),
        );
        device.cmd_dispatch(cmd, 1, 1, 1);

        // Composite and UI into the swapchain image.
        let target = self.swap.images[image];
        barrier(
            &[
                gpu::image_barrier(
                    hdr.image,
                    color,
                    S::COMPUTE_SHADER,
                    A::SHADER_STORAGE_WRITE,
                    S::FRAGMENT_SHADER,
                    A::SHADER_SAMPLED_READ,
                    L::GENERAL,
                    L::GENERAL,
                ),
                gpu::image_barrier(
                    target,
                    color,
                    S::COLOR_ATTACHMENT_OUTPUT,
                    A::NONE,
                    S::COLOR_ATTACHMENT_OUTPUT,
                    A::COLOR_ATTACHMENT_WRITE,
                    L::UNDEFINED,
                    L::COLOR_ATTACHMENT_OPTIMAL,
                ),
            ],
            // The exposure the adapt pass just wrote.
            &[vk::MemoryBarrier2::default()
                .src_stage_mask(S::COMPUTE_SHADER)
                .src_access_mask(A::SHADER_STORAGE_WRITE)
                .dst_stage_mask(S::FRAGMENT_SHADER)
                .dst_access_mask(A::SHADER_STORAGE_READ)],
        );
        let present_attachment = [vk::RenderingAttachmentInfo::default()
            .image_view(self.swap.views[image])
            .image_layout(L::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: self.clear_color,
                },
            })];
        device.cmd_begin_rendering(
            cmd,
            &vk::RenderingInfo::default()
                .render_area(full)
                .layer_count(1)
                .color_attachments(&present_attachment),
        );
        if visible {
            set_viewport(device, cmd, area);
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.composite_pipeline,
            );
            device.cmd_draw(cmd, 3, 1, 0, 0);
        }
        if let Some(frame) = ui {
            self.ui_renderer.as_mut().unwrap().cmd_draw(
                cmd,
                extent,
                frame.pixels_per_point,
                &frame.primitives,
            )?;
        }
        device.cmd_end_rendering(cmd);
        barrier(
            &[gpu::image_barrier(
                target,
                color,
                S::COLOR_ATTACHMENT_OUTPUT,
                A::COLOR_ATTACHMENT_WRITE,
                S::NONE,
                A::NONE,
                L::COLOR_ATTACHMENT_OPTIMAL,
                L::PRESENT_SRC_KHR,
            )],
            &[],
        );
        Ok(())
    }
}

unsafe fn set_viewport(device: &ash::Device, cmd: vk::CommandBuffer, area: vk::Rect2D) {
    device.cmd_set_viewport(
        cmd,
        0,
        &[vk::Viewport {
            x: area.offset.x as f32,
            y: area.offset.y as f32,
            width: area.extent.width as f32,
            height: area.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }],
    );
    device.cmd_set_scissor(cmd, 0, &[area]);
}

fn is_srgb(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB
    )
}

unsafe fn depth_format(instance: &ash::Instance, physical: vk::PhysicalDevice) -> vk::Format {
    // Vulkan guarantees one of these two.
    let d32 = instance
        .get_physical_device_format_properties(physical, vk::Format::D32_SFLOAT)
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT);
    if d32 {
        vk::Format::D32_SFLOAT
    } else {
        vk::Format::X8_D24_UNORM_PACK32
    }
}

/// The first Vulkan 1.3 device with dynamic rendering, synchronization2,
/// draw parameters, non-uniform texture indexing, a Basis transcode target,
/// the visibility format, and one queue for graphics, compute and presentation.
unsafe fn pick_device(
    instance: &ash::Instance,
    surfaces: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<DeviceChoice> {
    for physical in instance.enumerate_physical_devices()? {
        let properties = instance.get_physical_device_properties(physical);
        if properties.api_version < vk::API_VERSION_1_3
            || properties.limits.max_uniform_buffer_range < mem::size_of::<scene::Globals>() as u32
            || properties.limits.max_per_stage_descriptor_sampled_images < scene::MAX_TEXTURES + 4
            || properties.limits.max_descriptor_set_sampled_images < scene::MAX_TEXTURES + 4
        {
            continue;
        }
        let mut v11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut v13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut accel = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
        let mut query = vk::PhysicalDeviceRayQueryFeaturesKHR::default();
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut v11)
            .push_next(&mut v12)
            .push_next(&mut v13)
            .push_next(&mut accel)
            .push_next(&mut query);
        instance.get_physical_device_features2(physical, &mut features);
        let core = features.features;
        // Prefer the highest-quality target the GPU can sample.
        let texture_format = if core.texture_compression_bc != 0 {
            TextureFormat::Bc7
        } else if core.texture_compression_astc_ldr != 0 {
            TextureFormat::Astc4x4
        } else if core.texture_compression_etc2 != 0 {
            TextureFormat::Etc2
        } else {
            continue;
        };
        if v11.shader_draw_parameters == 0
            || v12.shader_sampled_image_array_non_uniform_indexing == 0
            || v12.descriptor_binding_partially_bound == 0
            || v12.buffer_device_address == 0
            || v13.dynamic_rendering == 0
            || v13.synchronization2 == 0
        {
            continue;
        }
        let extensions = instance.enumerate_device_extension_properties(physical)?;
        let has = |name: &CStr| {
            extensions
                .iter()
                .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == name)
        };
        let visibility = instance
            .get_physical_device_format_properties(physical, VISIBILITY_FORMAT)
            .optimal_tiling_features;
        let hdr = instance
            .get_physical_device_format_properties(physical, HDR_FORMAT)
            .optimal_tiling_features;
        if !has(ash::khr::swapchain::NAME)
            || !visibility.contains(
                vk::FormatFeatureFlags::COLOR_ATTACHMENT | vk::FormatFeatureFlags::SAMPLED_IMAGE,
            )
            || !hdr.contains(
                vk::FormatFeatureFlags::COLOR_ATTACHMENT
                    | vk::FormatFeatureFlags::SAMPLED_IMAGE
                    | vk::FormatFeatureFlags::STORAGE_IMAGE,
            )
        {
            continue;
        }
        let families = instance.get_physical_device_queue_family_properties(physical);
        for (index, family) in families.iter().enumerate() {
            let index = index as u32;
            if family.queue_count == 0
                || !family
                    .queue_flags
                    .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                || !surfaces.get_physical_device_surface_support(physical, index, surface)?
                || surfaces
                    .get_physical_device_surface_formats(physical, surface)?
                    .is_empty()
                || surfaces
                    .get_physical_device_surface_present_modes(physical, surface)?
                    .is_empty()
            {
                continue;
            }
            let priorities = [1.0];
            let queues = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(index)
                .queue_priorities(&priorities)];
            let mut names = vec![ash::khr::swapchain::NAME.as_ptr()];
            if has(ash::khr::portability_subset::NAME) {
                names.push(ash::khr::portability_subset::NAME.as_ptr());
            }
            // Hardware ray queries for GI, unless VERITY_NO_RAY_QUERY forces the
            // software BVH (to compare the two).
            let ray_extensions = [
                ash::khr::acceleration_structure::NAME,
                ash::khr::ray_query::NAME,
                ash::khr::deferred_host_operations::NAME,
            ];
            let ray_query = std::env::var_os("VERITY_NO_RAY_QUERY").is_none()
                && ray_extensions.iter().all(|name| has(name))
                && accel.acceleration_structure != 0
                && query.ray_query != 0;
            if ray_query {
                names.extend(ray_extensions.map(CStr::as_ptr));
            }
            let anisotropy =
                (core.sampler_anisotropy != 0).then_some(properties.limits.max_sampler_anisotropy);
            let enabled = vk::PhysicalDeviceFeatures::default()
                .sampler_anisotropy(anisotropy.is_some())
                .texture_compression_bc(texture_format == TextureFormat::Bc7)
                .texture_compression_astc_ldr(texture_format == TextureFormat::Astc4x4)
                .texture_compression_etc2(texture_format == TextureFormat::Etc2);
            let mut v11 =
                vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);
            let mut v12 = vk::PhysicalDeviceVulkan12Features::default()
                .shader_sampled_image_array_non_uniform_indexing(true)
                .descriptor_binding_partially_bound(true)
                .buffer_device_address(true);
            let mut v13 = vk::PhysicalDeviceVulkan13Features::default()
                .dynamic_rendering(true)
                .synchronization2(true);
            let mut accel = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
                .acceleration_structure(true);
            let mut query = vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
            let mut info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queues)
                .enabled_extension_names(&names)
                .enabled_features(&enabled)
                .push_next(&mut v11)
                .push_next(&mut v12)
                .push_next(&mut v13);
            if ray_query {
                info = info.push_next(&mut accel).push_next(&mut query);
            }
            let device = instance.create_device(physical, &info, None)?;
            return Ok(DeviceChoice {
                physical,
                device,
                queue_family: index,
                texture_format,
                anisotropy,
                ray_query,
            });
        }
    }
    Err(failure(
        "no Vulkan 1.3 device supports dynamic rendering, synchronization2, draw parameters, \
         buffer device address, non-uniform texture indexing, BC7/ASTC/ETC2 textures, and graphics, compute and \
         presentation on one queue",
    ))
}

impl Drop for Renderer {
    fn drop(&mut self) {
        // Destroying a null handle is a no-op, so partially built renderers drop cleanly.
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Some(timer) = self.gpu_timer.take() {
                timer.destroy(&self.device);
            }
            self.ui_renderer.take();
            for delta in &mut self.pending_textures {
                delta.clear();
            }
            if let Some(mut extension) = self.extension.take() {
                if self.allocator.is_some() {
                    extension.destroy(&mut self.gpu());
                }
            }
            if let Some(allocator) = self.allocator.as_mut() {
                if let Some(culler) = self.culler.take() {
                    culler.destroy(&self.device, allocator);
                }
                mem::take(&mut self.matrix).destroy(&self.device, allocator);
                self.breakers.destroy(&self.device, allocator);
                self.clouds.destroy(&self.device, allocator);
                self.taa.destroy(&self.device, allocator);
            }
            if self.allocator.is_some() {
                self.clear_swapchain();
            }
            let device = &self.device;
            for view in self
                .shadow_layers
                .drain(..)
                .chain(self.point_shadow_layers.drain(..))
            {
                device.destroy_image_view(view, None);
            }
            if let Some(allocator) = self.allocator.as_mut() {
                if let Some(embedded) = self.embedded.take() {
                    embedded.destroy(device, allocator);
                }
                for frame in self.frames.drain(..) {
                    device.destroy_fence(frame.fence, None);
                    device.destroy_semaphore(frame.acquired, None);
                    for buffer in [
                        frame.globals,
                        frame.arena.buffer,
                        frame.indices,
                        frame.materials,
                        frame.deforming,
                        frame.retained,
                        frame.layer_table,
                        frame.layer_feedback,
                        frame.layer_staging,
                    ] {
                        gpu::destroy_buffer(device, allocator, buffer);
                    }
                }
                for buffer in [
                    mem::take(&mut self.clusters),
                    mem::take(&mut self.exposure),
                    mem::take(&mut self.vertices),
                    mem::take(&mut self.brdf_lut),
                ] {
                    gpu::destroy_buffer(device, allocator, buffer);
                }
                for texture in self
                    .textures
                    .drain(..)
                    .chain(self.shadow_map.take())
                    .chain(self.point_shadow_map.take())
                    .chain(self.cloud_shadow.take())
                {
                    gpu::destroy_image(device, allocator, texture);
                }
            }
            // The allocator's memory blocks must go before the device.
            self.allocator.take();
            for pipeline in [
                self.cluster_pipeline,
                self.adapt_pipeline,
                self.visibility_pipeline,
                self.depth_prepass,
                self.resolve_pipeline,
                self.visibility_msaa_pipeline,
                self.resolve_msaa_pipeline,
                self.composite_pipeline,
            ] {
                device.destroy_pipeline(pipeline, None);
            }
            device.destroy_sampler(self.sampler, None);
            device.destroy_sampler(self.shadow_sampler, None);
            device.destroy_pipeline(self.shadow_pipeline, None);
            device.destroy_command_pool(self.command_pool, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

/// How a surface moved since last frame; mirrors `InstanceMotion` in `common.slang`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceMotion {
    /// This frame's world space to last frame's.
    to_previous: [[f32; 4]; 4],
    /// x: 1 when vertices move on their own (soft bodies, cloth); y: then their
    /// first entry in `Frame::previous_positions`.
    info: [u32; 4],
}

/// Append each instance's map from this frame's world space to last frame's
/// to `motion`, matching draws by draw index; returns this frame's models.
fn instance_motion(
    previous: &std::collections::HashMap<u32, (glam::Mat4, u32)>,
    instances: &[scene::Instance],
    motion: &mut Vec<InstanceMotion>,
) -> std::collections::HashMap<u32, (glam::Mat4, u32)> {
    // shortcut: world entities and embedded draws take their draw index from
    // their order, so adding or removing one gives its neighbours one frame
    // of wrong motion (TAA's clip absorbs it); key them by entity if that shows.
    let mut current = std::collections::HashMap::with_capacity(instances.len());
    for instance in instances {
        let model = glam::Mat4::from_cols_array_2d(&instance.model);
        let [_, base_vertex, _, slot] = instance.draw;
        let to_previous = match previous.get(&slot) {
            // The same mesh in the same slot: the same object.
            Some(&(last, base)) if base == base_vertex && last != model => last * model.inverse(),
            _ => glam::Mat4::IDENTITY,
        };
        motion.push(InstanceMotion {
            to_previous: to_previous.to_cols_array_2d(),
            info: [0; 4],
        });
        current.insert(slot, (model, base_vertex));
    }
    current
}

#[cfg(test)]
mod cascade_tests {
    use super::*;

    #[test]
    fn near_large_draws_occlude_and_far_small_ones_do_not() {
        let unit = (glam::Vec3::splat(-1.0), glam::Vec3::splat(1.0));
        let at = |z: f32, scale: f32| {
            glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::splat(scale),
                glam::Quat::IDENTITY,
                glam::Vec3::new(0.0, 0.0, z),
            )
        };
        let eye = glam::Vec3::ZERO;
        assert!(is_occluder(at(-10.0, 1.0), unit.0, unit.1, eye));
        assert!(!is_occluder(at(-100.0, 1.0), unit.0, unit.1, eye));
        assert!(is_occluder(at(-100.0, 10.0), unit.0, unit.1, eye));
        // The camera inside the bounds.
        assert!(is_occluder(at(0.0, 0.01), unit.0, unit.1, eye));
    }
}

#[cfg(test)]
mod motion_tests {
    use super::*;
    use glam::{Mat4, Vec3};

    #[test]
    fn moved_instance_maps_points_back_to_last_frame() {
        let at = |x: f32, slot: u32| {
            let mut instance = scene::instance(Mat4::from_translation(Vec3::new(x, 0.0, 0.0)), 0, 7, 0);
            instance.draw[3] = slot;
            instance
        };
        let mut motion = Vec::new();
        let first = instance_motion(&Default::default(), &[at(0.0, 0)], &mut motion);
        assert_eq!(motion[0].to_previous, Mat4::IDENTITY.to_cols_array_2d());
        motion.clear();
        instance_motion(&first, &[at(2.0, 0), at(5.0, 1)], &mut motion);
        let back = Mat4::from_cols_array_2d(&motion[0].to_previous)
            .transform_point3(Vec3::new(2.5, 1.0, 0.0));
        assert!(back.abs_diff_eq(Vec3::new(0.5, 1.0, 0.0), 1.0e-5));
        // A slot with no previous frame does not move.
        assert_eq!(motion[1].to_previous, Mat4::IDENTITY.to_cols_array_2d());
    }
}

#[cfg(test)]
mod virtual_geometry_tests {
    use super::*;

    #[test]
    fn selected_ranges_keep_source_and_deduplicate_by_primitive_and_level() {
        let mut arena = vec![0, 1, 2, 2, 3, 0];
        let mut ranges = std::collections::HashMap::new();
        let range = append_lod_range(&mut arena, &mut ranges, (4, 1), &[2, 3, 1]).unwrap();
        assert_eq!(range, (6, 3));
        assert_eq!(
            &arena[range.0 as usize..(range.0 + range.1) as usize],
            &[2, 3, 1]
        );
        assert_eq!(
            append_lod_range(&mut arena, &mut ranges, (4, 1), &[9]),
            Some(range)
        );
        assert_eq!(&arena[..6], &[0, 1, 2, 2, 3, 0]);
        assert_eq!(
            append_lod_range(&mut arena, &mut ranges, (5, 1), &[7, 8, 9]),
            Some((9, 3))
        );
    }
}
