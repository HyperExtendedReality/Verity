//! Vulkan 1.3 renderer: clustered lighting over a hybrid visibility buffer.
//!
//! Each frame a compute pass bins point lights into view-space clusters, a
//! raster pass writes depth and (instance, triangle) IDs, a compute resolve
//! rebuilds every pixel's surface from its triangle and shades it with
//! multiple-scattering GGX and Hammon diffuse into an HDR target, and a
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
mod gpu_culling;
mod meshlets;
mod scene;
mod upload;

pub use asset::{Sun, TextureFormat};
pub use extension::{FrameContext, Gpu, RenderExtension, Targets};
pub use scene::{Camera, Vertex, UP};
pub use upload::SceneStats;

use ash::{vk, Entry};
use gpu::{Buffer, Image};
use gpu_allocator::{
    vulkan::{Allocator, AllocatorCreateDesc},
    AllocationSizes, AllocatorDebugSettings, MemoryLocation,
};
use hyperon_core::{Material, PointLight, Scene, Transform, World};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::{
    error::Error,
    ffi::{CStr, CString},
    io, mem,
    sync::Arc,
};
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
const VISIBILITY_FORMAT: vk::Format = vk::Format::R32G32_UINT;
const HDR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
// Neutral gray, matching the editor chrome.
const CLEAR_COLOR: [f32; 4] = [0.086, 0.086, 0.086, 1.0];

macro_rules! spirv {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))
    };
}

pub mod breaker;

/// Tessellated UI and the physical-pixel scene viewport within the swapchain.
pub struct UiFrame {
    pub primitives: Vec<egui::ClippedPrimitive>,
    pub textures_delta: egui::TexturesDelta,
    pub pixels_per_point: f32,
    pub viewport: [u32; 4],
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
    fn lighting(&self) -> &hyperon_core::Lighting {
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

/// Per-frame CPU-written data and the commands that read it.
#[derive(Default)]
struct Frame {
    command: vk::CommandBuffer,
    fence: vk::Fence,
    acquired: vk::Semaphore,
    set: vk::DescriptorSet,
    globals: Buffer,
    instances: Buffer,
    indices: Buffer,
    lights: Buffer,
    materials: Buffer,
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
}

/// Owns all Vulkan resources and the window for the lifetime of its surface.
pub struct Renderer {
    _entry: Entry,
    instance: ash::Instance,
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    _window: Arc<Window>,
    physical: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    swap_loader: ash::khr::swapchain::Device,
    allocator: Option<Allocator>,
    depth_format: vk::Format,
    swap: Swapchain,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    descriptor_pool: vk::DescriptorPool,
    cluster_pipeline: vk::Pipeline,
    visibility_pipeline: vk::Pipeline,
    /// Same as `visibility_pipeline` without back-face culling.
    visibility_two_sided: vk::Pipeline,
    depth_prepass: vk::Pipeline,
    depth_prepass_two_sided: vk::Pipeline,
    resolve_pipeline: vk::Pipeline,
    /// Built for the swapchain format, so rebuilt when that changes.
    composite_pipeline: vk::Pipeline,
    composite_format: vk::Format,
    command_pool: vk::CommandPool,
    frames: Vec<Frame>,
    frame: usize,
    clusters: Buffer,
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
    virtual_geometry: Vec<Option<hyperon_strata::virtual_geometry::VirtualGeometry>>,
    /// Original indices remain available for exact fallback.
    source_indices: Vec<u32>,
    frame_indices: Vec<u32>,
    strata_enabled: bool,
    strata_debug: bool,
    model_draws: Vec<asset::Draw>,
    materials: Vec<LoadedMaterial>,
    scene_lights: Vec<PointLight>,
    sun: Option<Sun>,
    sun_is_imported: bool,
    /// Directional light selected for the current frame.
    frame_sun: Option<Sun>,
    camera: Camera,
    previous_view_proj: Option<glam::Mat4>,
    /// This frame's view-projection and the one before it.
    frame_view_proj: (glam::Mat4, glam::Mat4),
    frame_lights: Vec<PointLight>,
    frame_number: u64,
    extension: Option<Box<dyn RenderExtension>>,
    breakers: breaker::BreakerPass,
    requested_size: (u32, u32),
    recreate: bool,
    ui_renderer:
        Option<egui_ash_renderer::Renderer<egui_ash_renderer::allocator::DefaultAllocator>>,
    ui_format: Option<vk::Format>,
    pending_textures: Vec<egui::TexturesDelta>,
    instance_data: Vec<scene::Instance>,
    light_data: Vec<scene::Light>,
    material_data: Vec<scene::GpuMaterial>,
    /// Index count, two-sidedness, and primitive index for each instance.
    draw_calls: Vec<(u32, bool, usize)>,
    shadow_calls: Vec<(u32, u32)>,
    culler: Option<gpu_culling::Culler>,
    culling_ranges: Vec<Option<(usize, usize)>>,
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
            let app = CString::new("Hyperon")?;
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
                allocator: None,
                swap: Swapchain::default(),
                set_layout: vk::DescriptorSetLayout::null(),
                pipeline_layout: vk::PipelineLayout::null(),
                descriptor_pool: vk::DescriptorPool::null(),
                cluster_pipeline: vk::Pipeline::null(),
                visibility_pipeline: vk::Pipeline::null(),
                depth_prepass: vk::Pipeline::null(),
                depth_prepass_two_sided: vk::Pipeline::null(),
                visibility_two_sided: vk::Pipeline::null(),
                resolve_pipeline: vk::Pipeline::null(),
                composite_pipeline: vk::Pipeline::null(),
                composite_format: vk::Format::UNDEFINED,
                command_pool: vk::CommandPool::null(),
                frames: Vec::new(),
                frame: 0,
                clusters: Buffer::default(),
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
                virtual_geometry: Vec::new(),
                source_indices: Vec::new(),
                frame_indices: Vec::new(),
                strata_enabled: false,
                strata_debug: false,
                model_draws: Vec::new(),
                materials: Vec::new(),
                scene_lights: Vec::new(),
                sun: None,
                sun_is_imported: false,
                frame_sun: None,
                camera: Camera::default(),
                previous_view_proj: None,
                frame_view_proj: Default::default(),
                frame_lights: Vec::new(),
                frame_number: 0,
                extension: None,
                breakers: breaker::BreakerPass::default(),
                requested_size: (size.width, size.height),
                recreate: true,
                ui_renderer: None,
                ui_format: None,
                pending_textures: Vec::new(),
                instance_data: Vec::new(),
                light_data: Vec::new(),
                material_data: Vec::new(),
                draw_calls: Vec::new(),
                shadow_calls: Vec::new(),
                culler: None,
                culling_ranges: Vec::new(),
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
            buffer_device_address: false,
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
        ];
        let textures_binding = 12;
        let count = |i: usize| {
            if i == textures_binding {
                scene::MAX_TEXTURES
            } else {
                1
            }
        };
        let bindings: Vec<_> = kinds
            .iter()
            .enumerate()
            .map(|(i, &kind)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i as u32)
                    .descriptor_type(kind)
                    .descriptor_count(count(i))
                    .stage_flags(all)
            })
            .collect();
        // Texture slots past the loaded ones stay unwritten.
        let flags: Vec<_> = (0..kinds.len())
            .map(|i| {
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
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .size(8)];
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
            let per_set: u32 = (0..kinds.len())
                .filter(|&i| kinds[i] == ty)
                .map(count)
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
        for (pipeline, cull) in [
            (&mut self.visibility_pipeline, vk::CullModeFlags::BACK),
            (&mut self.visibility_two_sided, vk::CullModeFlags::NONE),
        ] {
            *pipeline = gpu::graphics_pipeline(
                device,
                layout,
                &gpu::Raster {
                    vertex: spirv!("visibility.vertexMain"),
                    fragment: spirv!("visibility.fragmentMain"),
                    color: Some(VISIBILITY_FORMAT),
                    depth: Some((self.depth_format, vk::CompareOp::GREATER)),
                    cull,
                    depth_bias: None,
                },
            )?;
        }
        for (pipeline, cull) in [
            (&mut self.depth_prepass, vk::CullModeFlags::BACK),
            (&mut self.depth_prepass_two_sided, vk::CullModeFlags::NONE),
        ] {
            *pipeline = gpu::graphics_pipeline(
                device,
                layout,
                &gpu::Raster {
                    vertex: spirv!("visibility.vertexMain"),
                    fragment: spirv!("depth_prepass.fragmentMain"),
                    color: None,
                    depth: Some((self.depth_format, vk::CompareOp::GREATER)),
                    cull,
                    depth_bias: None,
                },
            )?;
        }
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
        let (point_shadow_map, point_shadow_layers) = gpu::create_depth_layers(
            device,
            self.allocator.as_mut().unwrap(),
            "point shadows",
            self.depth_format,
            scene::POINT_SHADOW_SIZE,
            (scene::MAX_POINT_SHADOWS * scene::POINT_SHADOW_FACES) as u32,
        )?;
        self.point_shadow_map = Some(point_shadow_map);
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
                cull: vk::CullModeFlags::NONE,
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
            frame.instances = gpu::create_buffer(
                device,
                allocator,
                "instances",
                64 * mem::size_of::<scene::Instance>() as u64,
                storage,
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
            frame.lights = gpu::create_buffer(
                device,
                allocator,
                "lights",
                64 * mem::size_of::<scene::Light>() as u64,
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
        }
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
        self.virtual_geometry.push(None);
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

    /// Enable demand-resolved, per-frame virtual geometry for imported meshes.
    pub fn set_strata_enabled(&mut self, enabled: bool) {
        self.strata_enabled = enabled;
    }

    /// Color visible surfaces by the LOD actually selected for that frame.
    pub fn set_strata_debug(&mut self, enabled: bool) {
        self.strata_debug = enabled;
    }

    /// Point a frame's descriptor set at its buffers and the shared ones.
    unsafe fn write_buffer_descriptors(&self, index: usize) {
        let frame = &self.frames[index];
        let buffers = [
            &frame.globals,
            &frame.instances,
            &frame.lights,
            &self.clusters,
            &self.vertices,
            &frame.indices,
            &self.brdf_lut,
        ];
        let mut infos: Vec<(u32, [vk::DescriptorBufferInfo; 1])> = buffers
            .iter()
            .enumerate()
            .map(|(binding, b)| (binding as u32, [b.info()]))
            .collect();
        infos.push((10, [frame.materials.info()]));
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

    /// Point every frame's set at the current render targets.
    unsafe fn write_image_descriptors(&self) {
        let (Some(visibility), Some(hdr), Some(gbuffer), Some(receiver), Some(motion)) = (
            &self.swap.visibility,
            &self.swap.hdr,
            &self.swap.gbuffer,
            &self.swap.receiver,
            &self.swap.motion,
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
        ];
        let infos = images.map(|(_, _, view, layout)| {
            [vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(layout)]
        });
        for frame in &self.frames {
            let writes: Vec<_> = images
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
            self.device.update_descriptor_sets(&writes, &[]);
        }
    }

    unsafe fn clear_swapchain(&mut self) {
        let device = &self.device;
        let allocator = self.allocator.as_mut().unwrap();
        for image in [
            self.swap.depth.take(),
            self.swap.visibility.take(),
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
                .present_mode(vk::PresentModeKHR::FIFO)
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
                    cull: vk::CullModeFlags::NONE,
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

    fn draw_inner(&mut self, scene: RenderInput<'_>, ui: Option<UiFrame>) -> Result<()> {
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
            let device = &self.device;
            let frame_index = self.frame;
            let frame = &mut self.frames[frame_index];
            device.wait_for_fences(&[frame.fence], true, u64::MAX)?;
            for id in frame.frees.drain(..) {
                self.ui_renderer.as_mut().unwrap().free_texture(id)?;
            }
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
            self.record_scene(command, set, viewport)?;
            self.record_extension(command, set, viewport)?;
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
            self.record_present(command, set, image as usize, viewport, ui.as_ref())?;
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

    /// Build this frame's draws, materials and lights and write them with the
    /// globals, growing buffers as needed.
    unsafe fn upload_frame(&mut self, scene: &RenderInput<'_>, viewport: [u32; 4]) -> Result<()> {
        self.instance_data.clear();
        self.draw_calls.clear();
        self.shadow_calls.clear();
        self.frame_indices.clear();
        let mut ranges = std::collections::HashMap::<(usize, usize), (u32, u32)>::new();
        let mut selected_indices = Vec::new();
        self.material_data.clear();
        self.material_data
            .extend(self.materials.iter().map(|m| m.gpu));
        for (index, draw) in self.model_draws.iter().enumerate() {
            let primitive = self.primitives[draw.primitive as usize];
            let material = &self.materials[primitive.material as usize];
            if material.hidden {
                continue;
            }
            let mut instance = scene::instance(
                draw.transform,
                primitive.first_index,
                primitive.base_vertex,
                primitive.material,
            );
            let primitive_index = draw.primitive as usize;
            let mut level = 0;
            if self.strata_enabled {
                if let Some(geometry) = self.virtual_geometry[primitive_index].as_mut() {
                    let scale = geometry.bounds().map_or(f64::INFINITY, |bounds| {
                        scene::lod_projection_scale(&self.camera, viewport, draw.transform, bounds)
                    });
                    level = geometry
                        .resolve(scale, 1.0, &mut selected_indices)
                        .unwrap_or(0);
                }
            }
            let original_start = primitive.first_index as usize;
            let original_end = original_start + primitive.index_count as usize;
            let indices = if level == 0 {
                &self.source_indices[original_start..original_end]
            } else {
                &selected_indices
            };
            let (first, index_count) = append_lod_range(
                &mut self.frame_indices,
                &mut ranges,
                (primitive_index, level),
                indices,
            )
            .ok_or_else(|| failure("too many frame geometry indices"))?;
            instance.draw[0] = first;
            instance.normal_columns[0][3] = level as f32;
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
        let (sky, ground) = if lighting.method == hyperon_core::LightingMethod::DirectOnly {
            (glam::Vec3::ZERO, glam::Vec3::ZERO)
        } else {
            self.ambient_override.unwrap_or_else(|| {
                (
                    glam::Vec3::from_array(lighting.sky),
                    glam::Vec3::from_array(lighting.ground),
                )
            })
        };
        let mut globals = scene::globals(
            &self.camera,
            &scene::Lighting {
                sun: self.frame_sun.as_ref(),
                sky,
                ground,
            },
            viewport,
            [
                self.light_data.len() as u32,
                self.instance_data.len() as u32,
                u32::from(self.extension.as_ref().is_some_and(|e| e.active())),
            ],
            self.previous_view_proj,
            self.swap.srgb,
        );
        globals.counts[3] = u32::from(self.strata_debug);
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
        let view_proj = glam::Mat4::from_cols_array_2d(&globals.view_proj);
        self.frame_view_proj = (view_proj, self.previous_view_proj.unwrap_or(view_proj));
        self.previous_view_proj = Some(view_proj);
        self.culling_ranges.clear();
        let mut jobs = Vec::new();
        let use_culling = self.strata_enabled && self.culler.is_some();
        for (instance, &(count, _, primitive)) in
            self.draw_calls.iter().enumerate().filter(|_| use_culling)
        {
            let record = &self.instance_data[instance];
            let model = glam::Mat4::from_cols_array_2d(&record.model);
            let start = jobs.len();
            if record.normal_columns[0][3] == 0.0 {
                if let Some(meshlets) = &self.primitive_meshlets[primitive] {
                    for meshlet in meshlets {
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
            } else if let Some(bounds) = self.virtual_geometry[primitive]
                .as_ref()
                .and_then(|geometry| geometry.bounds())
            {
                jobs.push(gpu_culling::Job::new(
                    view_proj, model, bounds.min, bounds.max, 0, count,
                ));
            }
            self.culling_ranges
                .push((jobs.len() > start).then_some((start, jobs.len())));
        }
        self.culling_active = self.strata_enabled
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

        let device = &self.device;
        let allocator = self.allocator.as_mut().unwrap();
        let frame = &mut self.frames[self.frame];
        frame.globals.write(bytemuck::bytes_of(&globals));
        let mut grown = false;
        for (buffer, name, bytes) in [
            (
                &mut frame.instances,
                "instances",
                bytemuck::cast_slice::<_, u8>(&self.instance_data),
            ),
            (
                &mut frame.indices,
                "frame indices",
                bytemuck::cast_slice::<_, u8>(&self.frame_indices),
            ),
            (
                &mut frame.lights,
                "lights",
                bytemuck::cast_slice(&self.light_data),
            ),
            (
                &mut frame.materials,
                "materials",
                bytemuck::cast_slice(&self.material_data),
            ),
        ] {
            if bytes.len() as u64 > buffer.size {
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
            buffer.write(bytes);
        }
        if grown {
            self.write_buffer_descriptors(self.frame);
        }
        Ok(())
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
        let barrier = |images: &[vk::ImageMemoryBarrier2], buffers: &[vk::BufferMemoryBarrier2]| {
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(images)
                    .buffer_memory_barriers(buffers),
            );
        };
        let clusters = |src_stage, src_access, dst_stage, dst_access| {
            vk::BufferMemoryBarrier2::default()
                .buffer(self.clusters.buffer)
                .size(vk::WHOLE_SIZE)
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

        // Bin lights. The previous frame's resolve may still read the clusters.
        barrier(
            &[],
            &[clusters(
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

        // Both images are shared across frames. The queue dependency protects the
        // prior resolve's sampled reads before discarding and rewriting the maps.
        for (map, layers, size, point) in [
            (
                self.shadow_map.as_ref().unwrap(),
                &self.shadow_layers,
                scene::SHADOW_SIZE,
                false,
            ),
            (
                self.point_shadow_map.as_ref().unwrap(),
                &self.point_shadow_layers,
                scene::POINT_SHADOW_SIZE,
                true,
            ),
        ] {
            let all_layers =
                gpu::subresource(vk::ImageAspectFlags::DEPTH).layer_count(layers.len() as u32);
            let shadow_barrier = |src_stage, src_access, dst_stage, dst_access, old, new| {
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
                .subresource_range(all_layers)
            };
            barrier(
                &[shadow_barrier(
                    S::COMPUTE_SHADER,
                    A::SHADER_SAMPLED_READ,
                    S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
                    A::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    L::UNDEFINED,
                    L::DEPTH_ATTACHMENT_OPTIMAL,
                )],
                &[],
            );
            let shadow_area = vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: size,
                    height: size,
                },
            };
            for (index, &layer) in layers.iter().enumerate() {
                let attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(layer)
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
                let enabled = if point {
                    self.frame_lights
                        .get(index / scene::POINT_SHADOW_FACES)
                        .is_some_and(|light| {
                            light.radius.is_finite()
                                && light.radius > 0.0
                                && light.position.iter().all(|value| value.is_finite())
                        })
                } else {
                    self.frame_sun.is_some()
                };
                if enabled {
                    set_viewport(device, cmd, shadow_area);
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.shadow_pipeline,
                    );
                    for &(instance, index_count) in &self.shadow_calls {
                        let matrix = index + if point { scene::CASCADES } else { 0 };
                        let push = [instance, matrix as u32];
                        device.cmd_push_constants(
                            cmd,
                            self.pipeline_layout,
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            0,
                            bytemuck::cast_slice(&push),
                        );
                        device.cmd_draw(cmd, index_count, 1, 0, 0);
                    }
                }
                device.cmd_end_rendering(cmd);
            }
            barrier(
                &[shadow_barrier(
                    S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
                    A::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    S::COMPUTE_SHADER,
                    A::SHADER_SAMPLED_READ,
                    L::DEPTH_ATTACHMENT_OPTIMAL,
                    L::DEPTH_READ_ONLY_OPTIMAL,
                )],
                &[],
            );
        }
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
            let mut bound = None;
            for (instance, &(count, two_sided, _)) in self.draw_calls.iter().enumerate() {
                if bound != Some(two_sided) {
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        if two_sided {
                            self.depth_prepass_two_sided
                        } else {
                            self.depth_prepass
                        },
                    );
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

        // Visibility: depth plus (instance, triangle) IDs.
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
        let id_attachment = [vk::RenderingAttachmentInfo::default()
            .image_view(visibility.view)
            .image_layout(L::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    uint32: [u32::MAX; 4],
                },
            })];
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
                .render_area(full)
                .layer_count(1)
                .color_attachments(&id_attachment)
                .depth_attachment(&depth_attachment),
        );
        if visible {
            set_viewport(device, cmd, area);
            let mut bound = None;
            for (instance, &(index_count, two_sided, primitive)) in
                self.draw_calls.iter().enumerate()
            {
                if bound != Some(two_sided) {
                    let pipeline = if two_sided {
                        self.visibility_two_sided
                    } else {
                        self.visibility_pipeline
                    };
                    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
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
                } else if let Some(meshlets) = self.primitive_meshlets[primitive]
                    .as_ref()
                    .filter(|_| self.instance_data[instance].normal_columns[0][3] == 0.0)
                {
                    let model = glam::Mat4::from_cols_array_2d(&self.instance_data[instance].model);
                    meshlets::visible_runs(
                        meshlets,
                        self.frame_view_proj.0,
                        model,
                        |first, count| {
                            device.cmd_draw(cmd, count, 1, first, 0);
                        },
                    );
                } else {
                    device.cmd_draw(cmd, index_count, 1, 0, 0);
                }
            }
        }
        device.cmd_end_rendering(cmd);

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
                    S::LATE_FRAGMENT_TESTS,
                    A::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    S::COMPUTE_SHADER,
                    A::SHADER_SAMPLED_READ,
                    L::DEPTH_ATTACHMENT_OPTIMAL,
                    L::DEPTH_READ_ONLY_OPTIMAL,
                ),
            ],
            &[clusters(
                S::COMPUTE_SHADER,
                A::SHADER_STORAGE_WRITE,
                S::COMPUTE_SHADER,
                A::SHADER_STORAGE_READ,
            )],
        );
        if visible {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.resolve_pipeline);
            device.cmd_dispatch(cmd, width.div_ceil(8), height.div_ceil(8), 1);
        }
        if self.extension.is_some() {
            // Extensions read the resolve's outputs and add to HDR in compute.
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&[vk::MemoryBarrier2::default()
                    .src_stage_mask(S::COMPUTE_SHADER)
                    .src_access_mask(A::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(S::COMPUTE_SHADER)
                    .dst_access_mask(
                        A::SHADER_STORAGE_READ | A::SHADER_SAMPLED_READ | A::SHADER_STORAGE_WRITE,
                    )]),
            );
        }
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
        let device = &self.device;
        let extent = self.swap.extent;
        let hdr = self.swap.hdr.as_ref().unwrap();
        let color = vk::ImageAspectFlags::COLOR;
        let barrier = |images: &[vk::ImageMemoryBarrier2], buffers: &[vk::BufferMemoryBarrier2]| {
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(images)
                    .buffer_memory_barriers(buffers),
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
            &[],
        );
        let present_attachment = [vk::RenderingAttachmentInfo::default()
            .image_view(self.swap.views[image])
            .image_layout(L::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: CLEAR_COLOR,
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
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut v11)
            .push_next(&mut v12)
            .push_next(&mut v13);
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
                .descriptor_binding_partially_bound(true);
            let mut v13 = vk::PhysicalDeviceVulkan13Features::default()
                .dynamic_rendering(true)
                .synchronization2(true);
            let device = instance.create_device(
                physical,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queues)
                    .enabled_extension_names(&names)
                    .enabled_features(&enabled)
                    .push_next(&mut v11)
                    .push_next(&mut v12)
                    .push_next(&mut v13),
                None,
            )?;
            return Ok(DeviceChoice {
                physical,
                device,
                queue_family: index,
                texture_format,
                anisotropy,
            });
        }
    }
    Err(failure(
        "no Vulkan 1.3 device supports dynamic rendering, synchronization2, draw parameters, \
         non-uniform texture indexing, BC7/ASTC/ETC2 textures, and graphics, compute and \
         presentation on one queue",
    ))
}

impl Drop for Renderer {
    fn drop(&mut self) {
        // Destroying a null handle is a no-op, so partially built renderers drop cleanly.
        unsafe {
            let _ = self.device.device_wait_idle();
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
                self.breakers.destroy(&self.device, allocator);
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
                for frame in self.frames.drain(..) {
                    device.destroy_fence(frame.fence, None);
                    device.destroy_semaphore(frame.acquired, None);
                    for buffer in [
                        frame.globals,
                        frame.instances,
                        frame.indices,
                        frame.lights,
                        frame.materials,
                    ] {
                        gpu::destroy_buffer(device, allocator, buffer);
                    }
                }
                for buffer in [
                    mem::take(&mut self.clusters),
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
                {
                    gpu::destroy_image(device, allocator, texture);
                }
            }
            // The allocator's memory blocks must go before the device.
            self.allocator.take();
            for pipeline in [
                self.cluster_pipeline,
                self.visibility_pipeline,
                self.visibility_two_sided,
                self.depth_prepass,
                self.depth_prepass_two_sided,
                self.resolve_pipeline,
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
