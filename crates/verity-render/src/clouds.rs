//! Physically based sky and volumetric clouds behind the scene; see
//! `shaders/clouds.slang` for the techniques. Clouds are traced at half
//! resolution into a ping-pong history, then fill the pixels the resolve left
//! empty. A shadow map of sunlight through the layer is drawn before the
//! resolve, which darkens the sun with it. A grid of density summed toward the
//! sun, refreshed a slice in eight per frame, lights the clouds themselves.

use crate::{gpu, Result};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use gpu_allocator::vulkan::Allocator;
use std::time::Instant;
use verity_core::Clouds;

const SHAPE_SIZE: u32 = 128;
const DETAIL_SIZE: u32 = 64;
const LIGHT_SIZE: u32 = 128;
const NOISE_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
const HISTORY_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
const SHADOW_SIZE: u32 = 512;
/// World width of the cloud shadow map, centered on the camera.
const SHADOW_EXTENT: f32 = 20000.0;

/// The renderer's cloud shadow map; `CloudPass::record_shadow` keeps it valid.
pub(crate) unsafe fn create_shadow_map(
    device: &ash::Device,
    allocator: &mut Allocator,
) -> Result<gpu::Image> {
    gpu::create_image(
        device,
        allocator,
        "cloud shadows",
        vk::Format::R32_SFLOAT,
        vk::Extent2D {
            width: SHADOW_SIZE,
            height: SHADOW_SIZE,
        },
        1,
        vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST,
        vk::ImageAspectFlags::COLOR,
    )
}

/// The shadow map's world corner and size, snapped to whole texels so it
/// does not shimmer as the camera moves.
pub(crate) fn shadow_region(eye: Vec3) -> [f32; 4] {
    let texel = SHADOW_EXTENT / SHADOW_SIZE as f32;
    let corner = |v: f32| (v / texel).round() * texel - SHADOW_EXTENT * 0.5;
    [corner(eye.x), corner(eye.y), SHADOW_EXTENT, 1.0]
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    layer: [f32; 4],
    shape: [f32; 4],
    frame: [u32; 4],
    region: [f32; 4],
}

#[derive(Default)]
pub(crate) struct CloudPass {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pool: vk::DescriptorPool,
    samplers: [vk::Sampler; 2],
    shape_pipeline: vk::Pipeline,
    detail_pipeline: vk::Pipeline,
    trace: vk::Pipeline,
    shadow_pipeline: vk::Pipeline,
    light_pipeline: vk::Pipeline,
    /// The shadow map has left `UNDEFINED`; it holds all ones while `shadow_clear`.
    shadow_laid_out: bool,
    shadow_clear: bool,
    apply: vk::Pipeline,
    shape: Option<gpu::Image>,
    detail: Option<gpu::Image>,
    light: Option<gpu::Image>,
    /// `sets[k]` reads `history[k]` and writes the other.
    sets: [vk::DescriptorSet; 2],
    history: Vec<gpu::Image>,
    size: [u32; 2],
    baked: bool,
    /// False until the history images are laid out.
    ready: bool,
    history_valid: bool,
    parity: usize,
    wind: [f32; 2],
    last: Option<Instant>,
}

impl CloudPass {
    unsafe fn init(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        frame_layout: vk::DescriptorSetLayout,
        shadow: vk::ImageView,
    ) -> Result<()> {
        use vk::DescriptorType as D;
        let kinds = [
            D::STORAGE_IMAGE,
            D::STORAGE_IMAGE,
            D::SAMPLED_IMAGE,
            D::SAMPLED_IMAGE,
            D::SAMPLER,
            D::SAMPLER,
            D::SAMPLED_IMAGE,
            D::STORAGE_IMAGE,
            D::STORAGE_IMAGE,
            D::STORAGE_IMAGE,
            D::SAMPLED_IMAGE,
        ];
        let bindings: Vec<_> = (kinds.iter().enumerate())
            .map(|(i, &kind)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i as u32)
                    .descriptor_type(kind)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        self.set_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )?;
        let layouts = [frame_layout, self.set_layout];
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(size_of::<Params>() as u32)];
        self.layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&layouts)
                .push_constant_ranges(&push),
            None,
        )?;
        let sizes = [D::STORAGE_IMAGE, D::SAMPLED_IMAGE, D::SAMPLER].map(|ty| {
            let per_set = kinds.iter().filter(|&&k| k == ty).count() as u32;
            vk::DescriptorPoolSize::default().ty(ty).descriptor_count(2 * per_set)
        });
        self.pool = device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(2)
                .pool_sizes(&sizes),
            None,
        )?;
        let set_layouts = [self.set_layout; 2];
        let sets = device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.pool)
                .set_layouts(&set_layouts),
        )?;
        self.sets = [sets[0], sets[1]];
        for (sampler, mode) in self.samplers.iter_mut().zip([
            vk::SamplerAddressMode::REPEAT,
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
        ]) {
            *sampler = device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(mode)
                    .address_mode_v(mode)
                    .address_mode_w(mode),
                None,
            )?;
        }
        for (pipeline, entry, spirv) in [
            (&mut self.shape_pipeline, c"shapeMain", &spirv!("clouds.shapeMain")[..]),
            (&mut self.detail_pipeline, c"detailMain", &spirv!("clouds.detailMain")[..]),
            (&mut self.trace, c"traceMain", &spirv!("clouds.traceMain")[..]),
            (&mut self.apply, c"applyMain", &spirv!("clouds.applyMain")[..]),
            (&mut self.shadow_pipeline, c"shadowMain", &spirv!("clouds.shadowMain")[..]),
            (&mut self.light_pipeline, c"lightMain", &spirv!("clouds.lightMain")[..]),
        ] {
            *pipeline =
                gpu::compute_pipeline(device, vk::PipelineCache::null(), self.layout, spirv, entry)?;
        }
        let usage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED;
        self.shape = Some(gpu::create_volume(
            device,
            allocator,
            "cloud shape noise",
            NOISE_FORMAT,
            SHAPE_SIZE,
            usage,
        )?);
        self.detail = Some(gpu::create_volume(
            device,
            allocator,
            "cloud detail noise",
            NOISE_FORMAT,
            DETAIL_SIZE,
            usage,
        )?);
        self.light = Some(gpu::create_volume(
            device,
            allocator,
            "cloud light grid",
            HISTORY_FORMAT,
            LIGHT_SIZE,
            usage,
        )?);
        let samplers = self.samplers;
        let shape = self.shape.as_ref().unwrap().view;
        let detail = self.detail.as_ref().unwrap().view;
        let light = self.light.as_ref().unwrap().view;
        for set in self.sets {
            let image = |view, layout| [vk::DescriptorImageInfo::default().image_view(view).image_layout(layout)];
            let general = vk::ImageLayout::GENERAL;
            let infos = [
                (0, D::STORAGE_IMAGE, image(shape, general)),
                (1, D::STORAGE_IMAGE, image(detail, general)),
                (2, D::SAMPLED_IMAGE, image(shape, general)),
                (3, D::SAMPLED_IMAGE, image(detail, general)),
                (4, D::SAMPLER, [vk::DescriptorImageInfo::default().sampler(samplers[0])]),
                (5, D::SAMPLER, [vk::DescriptorImageInfo::default().sampler(samplers[1])]),
                (8, D::STORAGE_IMAGE, image(shadow, general)),
                (9, D::STORAGE_IMAGE, image(light, general)),
                (10, D::SAMPLED_IMAGE, image(light, general)),
            ];
            let writes: Vec<_> = (infos.iter())
                .map(|(binding, ty, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(*binding)
                        .descriptor_type(*ty)
                        .image_info(info)
                })
                .collect();
            device.update_descriptor_sets(&writes, &[]);
        }
        Ok(())
    }

    /// (Re)create the half-resolution history for a viewport of `width` x `height`.
    unsafe fn resize(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let size = [width.div_ceil(2), height.div_ceil(2)];
        if size == self.size && !self.history.is_empty() {
            return Ok(());
        }
        // Earlier frames in flight may still sample the old history.
        device.device_wait_idle()?;
        for image in self.history.drain(..) {
            gpu::destroy_image(device, allocator, image);
        }
        self.size = size;
        for name in ["cloud history a", "cloud history b"] {
            self.history.push(gpu::create_image(
                device,
                allocator,
                name,
                HISTORY_FORMAT,
                vk::Extent2D {
                    width: size[0],
                    height: size[1],
                },
                1,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                vk::ImageAspectFlags::COLOR,
            )?);
        }
        for k in 0..2 {
            let read = [vk::DescriptorImageInfo::default()
                .image_view(self.history[k].view)
                .image_layout(vk::ImageLayout::GENERAL)];
            let write = [vk::DescriptorImageInfo::default()
                .image_view(self.history[1 - k].view)
                .image_layout(vk::ImageLayout::GENERAL)];
            device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.sets[k])
                        .dst_binding(6)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&read),
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.sets[k])
                        .dst_binding(7)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&write),
                ],
                &[],
            );
        }
        self.ready = false;
        self.history_valid = false;
        Ok(())
    }

    fn params(&self, clouds: &Clouds, frame_number: u64, eye: Vec3) -> Params {
        let thickness = clouds.thickness.max(1.0);
        let bottom = clouds.bottom.max(0.0);
        // Each seed moves the sky to another part of the (tiling) noise.
        let seed = |prime: u32| (clouds.seed.wrapping_mul(prime) % 60000) as f32;
        Params {
            layer: [
                bottom,
                bottom + thickness,
                clouds.coverage.clamp(0.0, 1.0),
                clouds.density.max(0.0),
            ],
            shape: [
                clouds.kind.clamp(0.0, 1.0),
                self.wind[0] + seed(7919),
                self.wind[1] + seed(104_729),
                clouds.cirrus.clamp(0.0, 1.0),
            ],
            frame: [
                frame_number as u32,
                u32::from(self.history_valid),
                self.size[0],
                self.size[1],
            ],
            region: shadow_region(eye),
        }
    }

    unsafe fn bind(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame_set: vk::DescriptorSet,
        params: &Params,
    ) {
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.layout,
            0,
            &[frame_set, self.sets[self.parity]],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            self.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(params),
        );
    }

    /// Draw the cloud shadow map, or with no clouds (or no sun) fill it with
    /// ones, before the resolve reads it. Bakes the noise on first use.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn record_shadow(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        frame_layout: vk::DescriptorSetLayout,
        cmd: vk::CommandBuffer,
        frame_set: vk::DescriptorSet,
        shadow: &gpu::Image,
        clouds: Option<Clouds>,
        eye: Vec3,
        viewport: [u32; 4],
    ) -> Result<()> {
        use vk::AccessFlags2 as A;
        use vk::ImageLayout as L;
        use vk::PipelineStageFlags2 as S;
        let color = vk::ImageAspectFlags::COLOR;
        let old_layout = if self.shadow_laid_out { L::GENERAL } else { L::UNDEFINED };
        let Some(clouds) = clouds else {
            if self.shadow_clear {
                return Ok(());
            }
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&[gpu::image_barrier(
                    shadow.image,
                    color,
                    S::COMPUTE_SHADER,
                    A::NONE,
                    S::CLEAR,
                    A::TRANSFER_WRITE,
                    old_layout,
                    L::GENERAL,
                )]),
            );
            device.cmd_clear_color_image(
                cmd,
                shadow.image,
                L::GENERAL,
                &vk::ClearColorValue { float32: [1.0; 4] },
                &[gpu::subresource(color)],
            );
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&[gpu::image_barrier(
                    shadow.image,
                    color,
                    S::CLEAR,
                    A::TRANSFER_WRITE,
                    S::COMPUTE_SHADER,
                    A::SHADER_SAMPLED_READ,
                    L::GENERAL,
                    L::GENERAL,
                )]),
            );
            self.shadow_laid_out = true;
            self.shadow_clear = true;
            return Ok(());
        };
        if self.trace == vk::Pipeline::null() {
            if let Err(error) = self.init(device, allocator, frame_layout, shadow.view) {
                self.destroy(device, allocator);
                return Err(error);
            }
        }
        // Rewriting the history descriptors invalidates any command buffer
        // they are bound in, so resize before this frame binds them.
        if viewport[2] != 0 && viewport[3] != 0 {
            self.resize(device, allocator, viewport[2], viewport[3])?;
        }
        // Waits for last frame's resolve to finish reading the shadow map.
        let mut barriers = vec![gpu::image_barrier(
            shadow.image,
            color,
            S::COMPUTE_SHADER,
            A::NONE,
            S::COMPUTE_SHADER,
            A::SHADER_STORAGE_WRITE,
            old_layout,
            L::GENERAL,
        )];
        if !self.baked {
            barriers.extend([&self.shape, &self.detail, &self.light].map(|image| {
                gpu::image_barrier(
                    image.as_ref().unwrap().image,
                    color,
                    S::COMPUTE_SHADER,
                    A::NONE,
                    S::COMPUTE_SHADER,
                    A::SHADER_STORAGE_WRITE,
                    L::UNDEFINED,
                    L::GENERAL,
                )
            }));
        }
        device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&barriers),
        );
        self.shadow_laid_out = true;
        self.shadow_clear = false;
        let params = self.params(&clouds, 0, eye);
        self.bind(device, cmd, frame_set, &params);
        let memory = |src_access, dst_access| {
            let (src, dst) = ((S::COMPUTE_SHADER, src_access), (S::COMPUTE_SHADER, dst_access));
            gpu::memory_barrier(device, cmd, src, dst);
        };
        if !self.baked {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.shape_pipeline);
            device.cmd_dispatch(cmd, SHAPE_SIZE / 4, SHAPE_SIZE / 4, SHAPE_SIZE / 4);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.detail_pipeline);
            device.cmd_dispatch(cmd, DETAIL_SIZE / 4, DETAIL_SIZE / 4, DETAIL_SIZE / 4);
            memory(A::SHADER_STORAGE_WRITE, A::SHADER_SAMPLED_READ);
            self.baked = true;
        }
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.shadow_pipeline);
        device.cmd_dispatch(cmd, SHADOW_SIZE / 8, SHADOW_SIZE / 8, 1);
        memory(A::SHADER_STORAGE_WRITE, A::SHADER_SAMPLED_READ);
        Ok(())
    }

    /// Trace the clouds and fill the sky, after `record_shadow` this frame.
    /// The resolve's (and any extension's) HDR writes must be visible to
    /// compute shaders; this leaves its own HDR writes as compute-shader
    /// storage writes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn record(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        cmd: vk::CommandBuffer,
        frame_set: vk::DescriptorSet,
        frame_number: u64,
        viewport: [u32; 4],
        eye: Vec3,
        clouds: Option<Clouds>,
    ) -> Result<()> {
        // Without a sun `record_shadow` skipped the setup; the sky stays plain.
        let Some(clouds) = clouds.filter(|_| self.baked) else {
            self.history_valid = false;
            self.last = None;
            return Ok(());
        };
        if viewport[2] == 0 || viewport[3] == 0 {
            return Ok(());
        }
        // A no-op after `record_shadow`; without a sun nothing is bound yet.
        self.resize(device, allocator, viewport[2], viewport[3])?;

        let now = Instant::now();
        let dt = self.last.map_or(0.0, |last| (now - last).as_secs_f32().min(0.1));
        self.last = Some(now);
        for (offset, speed) in self.wind.iter_mut().zip(clouds.wind) {
            // shortcut: unwrapped, so f32 detail drifts after days of wind; wrap at a common noise period if that matters.
            *offset += speed * dt;
        }

        use vk::AccessFlags2 as A;
        use vk::ImageLayout as L;
        use vk::PipelineStageFlags2 as S;
        let memory = |src_access, dst_access| {
            let (src, dst) = ((S::COMPUTE_SHADER, src_access), (S::COMPUTE_SHADER, dst_access));
            gpu::memory_barrier(device, cmd, src, dst);
        };
        let params = self.params(&clouds, frame_number, eye);
        self.bind(device, cmd, frame_set, &params);
        if !self.ready {
            let barriers = [self.history[0].image, self.history[1].image].map(|image| {
                gpu::image_barrier(
                    image,
                    vk::ImageAspectFlags::COLOR,
                    S::COMPUTE_SHADER,
                    A::NONE,
                    S::COMPUTE_SHADER,
                    A::SHADER_STORAGE_WRITE,
                    L::UNDEFINED,
                    L::GENERAL,
                )
            });
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&barriers),
            );
            self.ready = true;
        }
        // Covers last frame's history and this frame's HDR.
        memory(
            A::SHADER_STORAGE_WRITE,
            A::SHADER_SAMPLED_READ | A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE,
        );
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.light_pipeline);
        // The shader redraws every slice when the history restarts.
        let slices = if self.history_valid { LIGHT_SIZE / 8 } else { LIGHT_SIZE };
        device.cmd_dispatch(cmd, LIGHT_SIZE / 8, LIGHT_SIZE / 8, slices);
        memory(A::SHADER_STORAGE_WRITE, A::SHADER_SAMPLED_READ);
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.trace);
        device.cmd_dispatch(cmd, self.size[0].div_ceil(8), self.size[1].div_ceil(8), 1);
        memory(A::SHADER_STORAGE_WRITE, A::SHADER_SAMPLED_READ);
        // The other set samples what this trace wrote.
        self.parity = 1 - self.parity;
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.layout,
            1,
            &[self.sets[self.parity]],
            &[],
        );
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.apply);
        device.cmd_dispatch(cmd, viewport[2].div_ceil(8), viewport[3].div_ceil(8), 1);
        self.history_valid = true;
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        for image in (self.history.drain(..))
            .chain(self.shape.take())
            .chain(self.detail.take())
            .chain(self.light.take())
        {
            gpu::destroy_image(device, allocator, image);
        }
        for pipeline in [
            self.shape_pipeline,
            self.detail_pipeline,
            self.trace,
            self.apply,
            self.shadow_pipeline,
            self.light_pipeline,
        ] {
            device.destroy_pipeline(pipeline, None);
        }
        for sampler in self.samplers {
            device.destroy_sampler(sampler, None);
        }
        device.destroy_descriptor_pool(self.pool, None);
        device.destroy_pipeline_layout(self.layout, None);
        device.destroy_descriptor_set_layout(self.set_layout, None);
        // The shadow map is the renderer's and keeps its layout and contents.
        let (laid_out, clear) = (self.shadow_laid_out, self.shadow_clear);
        *self = Self::default();
        self.shadow_laid_out = laid_out;
        self.shadow_clear = clear;
    }
}
