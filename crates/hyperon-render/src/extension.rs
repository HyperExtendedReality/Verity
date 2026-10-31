//! Render extensions: GPU work recorded into every frame after the resolve has
//! written direct lighting and the thin G-buffer, before tonemapping. Global
//! illumination (hyperon-lux) is one.

use crate::{asset::SceneData, Sun};
use ash::vk;
use glam::{Mat4, Vec3};
use gpu_allocator::vulkan::Allocator;
use hyperon_core::PointLight;
use std::{any::Any, error::Error};

pub type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;

/// The renderer's device and memory, lent to an extension.
pub struct Gpu<'a> {
    pub instance: &'a ash::Instance,
    pub physical: vk::PhysicalDevice,
    pub device: &'a ash::Device,
    pub queue: vk::Queue,
    /// Family of `queue`, for command pools.
    pub queue_family: u32,
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
}

pub trait RenderExtension {
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

    /// The render targets were (re)created. The GPU is idle.
    fn targets_changed(&mut self, gpu: &mut Gpu, targets: &Targets) -> Result<()>;

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

    /// For [`crate::Renderer::extension_mut`].
    fn as_any_mut(&mut self) -> &mut dyn Any;
}
