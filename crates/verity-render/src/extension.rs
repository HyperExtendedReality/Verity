//! Render extensions: GPU work recorded into every frame after the resolve has
//! written direct lighting and the thin G-buffer, before tonemapping. Global
//! illumination (verity-lux) is one.

use crate::{asset::SceneData, RetainedMesh, Sun};
use ash::vk;
use glam::{Mat4, Vec3};
use gpu_allocator::vulkan::Allocator;
use std::{any::Any, error::Error};
use verity_core::PointLight;

pub type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;

/// Optional Vulkan ray features enabled on the logical device.
///
/// These gates report availability, not whether an extension has opted in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct RayFeatures {
    /// `VK_EXT_opacity_micromap` with the `micromap` feature.
    pub opacity_micromap: bool,
    /// `VK_NV_cluster_acceleration_structure` with its feature enabled.
    pub cluster_acceleration: bool,
    /// `VK_NV_partitioned_acceleration_structure` with its feature enabled.
    pub partitioned_acceleration: bool,
    /// `VK_AMDX_dense_geometry_format` and its buffer usage flag dependency.
    pub dense_geometry: bool,
}

/// The renderer's device and memory, lent to an extension.
pub struct Gpu<'a> {
    pub instance: &'a ash::Instance,
    pub physical: vk::PhysicalDevice,
    pub device: &'a ash::Device,
    pub queue: vk::Queue,
    /// Family of `queue`, for command pools.
    pub queue_family: u32,
    /// `VK_KHR_acceleration_structure`, `VK_KHR_ray_query` and buffer device
    /// addresses are enabled, so an extension may trace with ray queries.
    pub ray_query: bool,
    /// Enabled optional ray features; all false when `ray_query` is disabled.
    pub ray_features: RayFeatures,
    pub allocator: &'a mut Allocator,
    /// Layout of the renderer's per-frame descriptor set: globals, instances,
    /// lights, geometry, materials, textures and the G-buffer (see
    /// `shaders/lib/common.slang`). Bind it at the set an extension's shaders
    /// expect it, with `MATERIAL_SET` defined to match.
    pub frame_set_layout: vk::DescriptorSetLayout,
    pub frames_in_flight: usize,
}

/// Image views of the current render targets, all `extent` sized.
#[derive(Clone, Copy, Debug)]
pub struct Targets {
    pub extent: vk::Extent2D,
    /// Reverse-Z depth; `DEPTH_READ_ONLY_OPTIMAL` while extensions record.
    pub depth: vk::ImageView,
    /// Thin G-buffer, receiver identity and motion (`shaders/lib/gi_surface.slang`),
    /// in `GENERAL` layout.
    pub gbuffer: vk::ImageView,
    pub receiver: vk::ImageView,
    pub motion: vk::ImageView,
    /// RGBA16F scene radiance in `GENERAL` layout; extensions add to it.
    pub hdr: vk::ImageView,
}

/// What an extension sees of one frame.
pub struct FrameContext<'a> {
    pub cmd: vk::CommandBuffer,
    /// The renderer's descriptor set for this frame, laid out by `Gpu::frame_set_layout`.
    pub frame_set: vk::DescriptorSet,
    /// Which of the frames in flight this is, `0..frames_in_flight`.
    pub frame_slot: usize,
    /// Frames rendered before this one.
    pub frame_number: u64,
    pub targets: Targets,
    /// Scene viewport in target pixels: x, y, width, height.
    pub viewport: [u32; 4],
    pub eye: Vec3,
    pub view_proj: Mat4,
    pub previous_view_proj: Mat4,
    pub sun: Option<Sun>,
    pub lights: &'a [PointLight],
    /// Radiance of the ambient hemisphere from straight up and straight down,
    /// which the resolve leaves out while an extension is [`active`](RenderExtension::active).
    pub sky: Vec3,
    pub ground: Vec3,
    /// The camera's near plane: with reverse-Z infinite depth, view depth is
    /// `near / device depth`.
    pub near: f32,
}

/// A retained mesh that appeared, changed or, with no `mesh`, went away.
#[derive(Clone, Copy)]
pub struct RetainedChange<'a> {
    pub key: u64,
    pub mesh: Option<&'a RetainedMesh>,
}

pub trait RenderExtension: Send {
    /// A scene was loaded: its draws are numbered from `draw_base` in the
    /// receiver image's draw index and its materials from `material_base` in
    /// the material buffer. The GPU is idle.
    fn scene_loaded(
        &mut self,
        gpu: &mut Gpu,
        scene: &SceneData,
        material_base: u32,
        draw_base: u32,
    ) -> Result<()>;

    /// Retained meshes changed, such as terrain being dug. A method that traces
    /// the scene follows them here, or it will not see them. `material` indexes
    /// the renderer's material buffer. Changes arrive batched, a few times a
    /// second at most, and the first call holds every mesh. Vertices are in
    /// world space; a material flagged `TERRAIN_SURFACE` takes its color from
    /// `terrain_albedo` in `shaders/lib/terrain.slang`. The GPU is idle.
    fn retained_changed(
        &mut self,
        _gpu: &mut Gpu,
        _changes: &[RetainedChange],
        _material: u32,
    ) -> Result<()> {
        Ok(())
    }

    /// The render targets were (re)created. The GPU is idle.
    fn targets_changed(&mut self, gpu: &mut Gpu, targets: &Targets) -> Result<()>;

    /// Before this frame records anything, with `viewport` the area it will
    /// draw (empty when it draws no 3D view). Earlier frames may still be on
    /// the GPU, so replacing resources they use means waiting for it first.
    fn prepare(&mut self, _gpu: &mut Gpu, _viewport: [u32; 4]) -> Result<()> {
        Ok(())
    }

    /// Record this frame's commands. Resolve writes to the HDR target and the
    /// G-buffer are visible to compute shaders; leave HDR writes visible to
    /// compute shaders too.
    fn record(&mut self, gpu: &mut Gpu, frame: &FrameContext) -> Result<()>;

    /// Release every GPU resource. The GPU is idle.
    fn destroy(&mut self, gpu: &mut Gpu);

    /// Whether this frame's output replaces the renderer's ambient term, for
    /// example once GI pipelines finish compiling.
    fn active(&self) -> bool {
        true
    }

    /// Whether the resolve keeps the sky's specular reflection while this
    /// extension is [`active`](RenderExtension::active), for one that only adds
    /// diffuse light. The reflection ignores occlusion, so it suits open ground.
    fn sky_reflections(&self) -> bool {
        false
    }

    /// For [`crate::Renderer::extension_mut`].
    fn as_any_mut(&mut self) -> &mut dyn Any;
}
