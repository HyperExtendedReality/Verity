//! Temporal anti-aliasing and SMAA T2x; see `shaders/taa.slang` for the
//! techniques. The renderer jitters the projection by [`jitter`] and this pass
//! accumulates the jittered frames in a ping-pong history the size of the
//! scene viewport; SMAA T2x keeps only the last frame there.

use crate::{gpu, Result};
use ash::vk;
use gpu_allocator::vulkan::Allocator;

const HISTORY_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
/// SMAA's edges and blend weights.
const SMAA_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
/// Halton(2, 3) points per cycle. Eight left low resolutions, where a pixel
/// spans more of each edge, visibly stepped; sixteen fill the pixel evenly.
const JITTER_PHASES: u64 = 16;

/// How the scene is anti-aliased: a player-facing graphics setting.
///
/// The MSAA modes rasterize the visibility buffer with 4 samples per pixel
/// and shade up to two triangles per pixel, weighted by coverage: edges are
/// anti-aliased in the frame itself, at about one extra shade per edge pixel.
/// They fall back to the matching mode without MSAA on GPUs that lack it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AntiAliasing {
    Off,
    /// 4x MSAA visibility only: no history, so no ghosting or temporal blur.
    Msaa,
    /// Temporal, with a 5-tap neighbourhood clamp and one bilinear history
    /// tap: for mobile and integrated GPUs.
    Mobile,
    /// SMAA T2x: morphological edges each frame, blended with only the
    /// previous frame at the other of two jitter positions. No ghosting
    /// trails and sharp textures, but less smoothing of shading than TAA.
    Smaa,
    /// Temporal, with 3x3 variance clipping and a Catmull-Rom history.
    #[default]
    Desktop,
    /// [`Self::Desktop`] over 4x MSAA visibility: TAA then only has shading
    /// to smooth, so it keeps less history.
    Ultra,
}

impl AntiAliasing {
    /// Every choice, cheapest first, for settings menus.
    pub const ALL: [Self; 6] = [
        Self::Off,
        Self::Msaa,
        Self::Mobile,
        Self::Smaa,
        Self::Desktop,
        Self::Ultra,
    ];

    /// A short name for settings menus.
    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::Msaa => "MSAA 4x",
            Self::Mobile => "TAA (fast)",
            Self::Smaa => "SMAA T2x",
            Self::Desktop => "TAA (quality)",
            Self::Ultra => "TAA + MSAA 4x",
        }
    }

    /// Whether the projection jitters and earlier frames are blended in.
    pub fn temporal(self) -> bool {
        matches!(self, Self::Mobile | Self::Smaa | Self::Desktop | Self::Ultra)
    }

    /// Whether the TAA sharpening setting applies.
    pub fn sharpened(self) -> bool {
        matches!(self, Self::Mobile | Self::Desktop | Self::Ultra)
    }

    /// Whether the visibility buffer is multisampled.
    pub fn msaa(self) -> bool {
        matches!(self, Self::Msaa | Self::Ultra)
    }
}

/// Sub-pixel offset for a frame, in pixels within (-0.5, 0.5).
pub(crate) fn jitter(frame_number: u64, mode: AntiAliasing) -> [f32; 2] {
    if mode == AntiAliasing::Smaa {
        // SMAA T2x's two diagonal positions, a half pixel apart.
        return if frame_number.is_multiple_of(2) { [0.25, -0.25] } else { [-0.25, 0.25] };
    }
    if !mode.temporal() {
        return [0.0, 0.0];
    }
    let index = (frame_number % JITTER_PHASES + 1) as u32;
    [halton(index, 2) - 0.5, halton(index, 3) - 0.5]
}

fn halton(mut index: u32, base: u32) -> f32 {
    let (mut result, mut scale) = (0.0, 1.0);
    while index > 0 {
        scale /= base as f32;
        result += scale * (index % base) as f32;
        index /= base;
    }
    result
}

#[derive(Default)]
pub(crate) struct TaaPass {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pool: vk::DescriptorPool,
    sampler: vk::Sampler,
    resolve: vk::Pipeline,
    apply: vk::Pipeline,
    smaa_edges: vk::Pipeline,
    smaa_weights: vk::Pipeline,
    /// `sets[k]` reads `history[k]` and writes the other.
    sets: [vk::DescriptorSet; 2],
    history: Vec<gpu::Image>,
    /// SMAA's edges and blend weights, swapping roles with the sets; the
    /// TAA tiers keep each history texel's length in frames in them.
    smaa: Vec<gpu::Image>,
    size: [u32; 2],
    /// False until the history images are laid out.
    ready: bool,
    history_valid: bool,
    /// Frames accumulated since the history was last reset.
    history_frames: u32,
    parity: usize,
}

impl TaaPass {
    unsafe fn init(&mut self, device: &ash::Device, frame_layout: vk::DescriptorSetLayout) -> Result<()> {
        use vk::DescriptorType as D;
        let kinds = [
            D::SAMPLED_IMAGE,
            D::STORAGE_IMAGE,
            D::SAMPLER,
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
            .size(32)];
        self.layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&layouts)
                .push_constant_ranges(&push),
            None,
        )?;
        let sizes = kinds.map(|ty| vk::DescriptorPoolSize::default().ty(ty).descriptor_count(2));
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
        let mode = vk::SamplerAddressMode::CLAMP_TO_EDGE;
        self.sampler = device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(mode)
                .address_mode_v(mode)
                .address_mode_w(mode),
            None,
        )?;
        let sampler = [vk::DescriptorImageInfo::default().sampler(self.sampler)];
        for set in self.sets {
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(2)
                    .descriptor_type(D::SAMPLER)
                    .image_info(&sampler)],
                &[],
            );
        }
        self.resolve = gpu::compute_pipeline(
            device,
            vk::PipelineCache::null(),
            self.layout,
            &spirv!("taa.resolveMain")[..],
            c"resolveMain",
        )?;
        self.apply = gpu::compute_pipeline(
            device,
            vk::PipelineCache::null(),
            self.layout,
            &spirv!("taa.applyMain")[..],
            c"applyMain",
        )?;
        self.smaa_edges = gpu::compute_pipeline(
            device,
            vk::PipelineCache::null(),
            self.layout,
            &spirv!("taa.smaaEdgesMain")[..],
            c"smaaEdgesMain",
        )?;
        self.smaa_weights = gpu::compute_pipeline(
            device,
            vk::PipelineCache::null(),
            self.layout,
            &spirv!("taa.smaaWeightsMain")[..],
            c"smaaWeightsMain",
        )?;
        Ok(())
    }

    /// (Re)create the history for a viewport of `width` x `height`.
    unsafe fn resize(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let size = [width, height];
        if size == self.size && !self.history.is_empty() {
            return Ok(());
        }
        // Earlier frames in flight may still sample the old history.
        device.device_wait_idle()?;
        for image in self.history.drain(..).chain(self.smaa.drain(..)) {
            gpu::destroy_image(device, allocator, image);
        }
        self.size = size;
        for name in ["taa history a", "taa history b"] {
            self.history.push(gpu::create_image(
                device,
                allocator,
                name,
                HISTORY_FORMAT,
                vk::Extent2D { width, height },
                1,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                vk::ImageAspectFlags::COLOR,
            )?);
        }
        for name in ["smaa edges", "smaa weights"] {
            self.smaa.push(gpu::create_image(
                device,
                allocator,
                name,
                SMAA_FORMAT,
                vk::Extent2D { width, height },
                1,
                vk::ImageUsageFlags::STORAGE,
                vk::ImageAspectFlags::COLOR,
            )?);
        }
        for k in 0..2 {
            let image = |view| {
                [vk::DescriptorImageInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::GENERAL)]
            };
            let (read, write) = (image(self.history[k].view), image(self.history[1 - k].view));
            // Swapped each frame: the TAA tiers write their history lengths through
            // the edges binding and read last frame's through the weights binding.
            let (edges, weights) = (image(self.smaa[k].view), image(self.smaa[1 - k].view));
            device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.sets[k])
                        .dst_binding(3)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&edges),
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.sets[k])
                        .dst_binding(4)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&weights),
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.sets[k])
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&read),
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.sets[k])
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&write),
                    // The same image, sampled: SMAA T2x's last frame.
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.sets[k])
                        .dst_binding(5)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&write),
                ],
                &[],
            );
        }
        self.ready = false;
        self.history_valid = false;
        Ok(())
    }

    /// Resolve this frame into the history and write the result back to HDR,
    /// after every pass that draws into HDR. Leaves HDR as compute-shader
    /// storage writes, which the composite's barrier expects.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn record(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        frame_layout: vk::DescriptorSetLayout,
        cmd: vk::CommandBuffer,
        frame_set: vk::DescriptorSet,
        viewport: [u32; 4],
        mode: AntiAliasing,
        multisampled: bool,
        sharpening: f32,
        jitter: [f32; 2],
    ) -> Result<()> {
        if !mode.temporal() || viewport[2] == 0 || viewport[3] == 0 {
            self.history_valid = false;
            return Ok(());
        }
        if self.resolve == vk::Pipeline::null() {
            if let Err(error) = self.init(device, frame_layout) {
                self.destroy(device, allocator);
                return Err(error);
            }
        }
        self.resize(device, allocator, viewport[2], viewport[3])?;

        use vk::AccessFlags2 as A;
        use vk::ImageLayout as L;
        use vk::PipelineStageFlags2 as S;
        if !self.ready {
            let images = [
                self.history[0].image,
                self.history[1].image,
                self.smaa[0].image,
                self.smaa[1].image,
            ];
            let barriers = images.map(|image| {
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
        let memory = |src_stage, src_access| {
            let read_write =
                A::SHADER_SAMPLED_READ | A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE;
            let dst = (S::COMPUTE_SHADER, read_write);
            gpu::memory_barrier(device, cmd, (src_stage, src_access), dst);
        };
        // HDR from compute passes or the breakers' raster, and last frame's history.
        memory(
            S::COMPUTE_SHADER | S::COLOR_ATTACHMENT_OUTPUT,
            A::SHADER_STORAGE_WRITE | A::COLOR_ATTACHMENT_WRITE,
        );
        let smaa = mode == AntiAliasing::Smaa;
        if !self.history_valid {
            self.history_frames = 0;
        }
        let params = [
            // Bit 0: the desktop tier; bit 1: edges already multisampled; bit 2: SMAA T2x.
            u32::from(mode != AntiAliasing::Mobile)
                | u32::from(multisampled) << 1
                | u32::from(smaa) << 2,
            u32::from(self.history_valid),
            sharpening.to_bits(),
            self.history_frames,
            // This frame's jitter, for the reconstruction filter.
            jitter[0].to_bits(),
            jitter[1].to_bits(),
            0,
            0,
        ];
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
            bytemuck::bytes_of(&params),
        );
        let groups = [viewport[2].div_ceil(8), viewport[3].div_ceil(8)];
        if smaa {
            for pipeline in [self.smaa_edges, self.smaa_weights] {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                device.cmd_dispatch(cmd, groups[0], groups[1], 1);
                memory(S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE);
            }
        }
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.resolve);
        device.cmd_dispatch(cmd, groups[0], groups[1], 1);
        memory(S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE);
        // The other set samples what the resolve wrote.
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
        device.cmd_dispatch(cmd, groups[0], groups[1], 1);
        self.history_valid = true;
        self.history_frames = self.history_frames.saturating_add(1);
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        for image in self.history.drain(..).chain(self.smaa.drain(..)) {
            gpu::destroy_image(device, allocator, image);
        }
        for pipeline in [self.resolve, self.apply, self.smaa_edges, self.smaa_weights] {
            device.destroy_pipeline(pipeline, None);
        }
        device.destroy_sampler(self.sampler, None);
        device.destroy_descriptor_pool(self.pool, None);
        device.destroy_pipeline_layout(self.layout, None);
        device.destroy_descriptor_set_layout(self.set_layout, None);
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jitter_is_sub_pixel_and_covers_the_pixel() {
        let points: Vec<_> = (0..JITTER_PHASES)
            .map(|frame| jitter(frame, AntiAliasing::Desktop))
            .collect();
        assert!(points.iter().flatten().all(|v| v.abs() < 0.5));
        // Halton(2, 3) lands in every quadrant within the cycle.
        for quadrant in [[false, false], [false, true], [true, false], [true, true]] {
            assert!(points.iter().any(|p| [p[0] > 0.0, p[1] > 0.0] == quadrant));
        }
        assert_eq!(
            jitter(0, AntiAliasing::Desktop),
            jitter(JITTER_PHASES, AntiAliasing::Desktop)
        );
    }

    #[test]
    fn smaa_alternates_two_opposite_diagonal_positions() {
        let [a, b] = [0, 1].map(|frame| jitter(frame, AntiAliasing::Smaa));
        assert_eq!([a[0] + b[0], a[1] + b[1]], [0.0, 0.0]);
        assert!(a[0] != 0.0 && a[1] != 0.0 && a[0] == -a[1]);
        assert_eq!(a, jitter(2, AntiAliasing::Smaa));
    }
}
