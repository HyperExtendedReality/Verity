//! Screen-space ambient occlusion with CORTAO; see `shaders/lib/ao.slang`.
//! Runs between the visibility pass and the resolve, over the scene
//! viewport, and leaves its result in the swapchain-sized image the resolve
//! samples to darken the hemisphere ambient.

use crate::{gpu, Result};
use ash::vk;
use gpu_allocator::vulkan::Allocator;

/// Levels of the packed depth and normal chain the shaders bind.
const MIPS: u32 = 8;
/// Largest radius on screen, as a share of the viewport height.
const MAX_SCREEN_RADIUS: f32 = 0.2;
// CORTAOParameters.h's defaults.
const MIP_BIAS: f32 = -4.0;
const MAX_SLOPE_COMPENSATION: f32 = 4.0;
const BACKFACE_THRESHOLD: f32 = 0.5;
const FILTER_DEPTH_SENSITIVITY: f32 = 512.0;
const FILTER_NORMAL_SENSITIVITY: f32 = 128.0;

/// The push constants of each pass: CORTAO's parameter structs, as words.
/// `projection` is the projection's x and y scale (y negative: Vulkan's
/// flipped y), `near` its reverse-Z, infinite-far near plane.
fn params(size: [u32; 2], projection: [f32; 2], near: f32, radius: f32, seed: u32) -> ([u32; 8], [u32; 20], [u32; 4]) {
    let [width, height] = size;
    let resolution = width | height << 16;
    let mips = MIPS.min(32 - width.max(height).leading_zeros());
    // Linear depth = 1 / (depth * scale + offset) = near / depth.
    let prepass = [resolution, (1.0 / near).to_bits(), 0f32.to_bits(), seed, mips, 0, 0, 0];
    let [px, py] = projection;
    // View direction at depth 1 = uv * unproject.xy + unproject.zw, and back.
    let unproject = [2.0 / px, 2.0 / py, -1.0 / px, -1.0 / py];
    let project = [px * 0.5, py * 0.5, -0.5, -0.5];
    let max_radius = MAX_SCREEN_RADIUS * 2.0 / py.abs();
    let f = |v: f32| v.to_bits();
    let trace = [
        resolution,
        seed,
        f(std::f32::consts::SQRT_2 / (px * width as f32)),
        f(radius),
        f(unproject[0]),
        f(unproject[1]),
        f(unproject[2]),
        f(unproject[3]),
        f(project[0]),
        f(project[1]),
        f(project[2]),
        f(project[3]),
        f(max_radius / (max_radius * max_radius + 1.0).sqrt()),
        f((py.abs() * height as f32 * 0.5).log2() + MIP_BIAS),
        f(1.0 / MAX_SLOPE_COMPENSATION),
        f(BACKFACE_THRESHOLD),
        0f32.to_bits(), // backface boost
        0,
        0,
        0,
    ];
    let filter = [resolution, f(FILTER_DEPTH_SENSITIVITY), f(FILTER_NORMAL_SENSITIVITY), 0];
    (prepass, trace, filter)
}

#[derive(Default)]
pub(crate) struct AoPass {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pool: vk::DescriptorPool,
    samplers: [vk::Sampler; 2],
    pipelines: [vk::Pipeline; 3],
    set: vk::DescriptorSet,
    /// Linear depth with a dithered normal in its low bits, mip-mapped.
    packed: Option<gpu::Image>,
    mip_views: Vec<vk::ImageView>,
    /// Occlusion before the filter.
    raw: Option<gpu::Image>,
    /// The viewport size and output view the set was written for.
    key: ([u32; 2], vk::ImageView),
}

impl AoPass {
    unsafe fn init(&mut self, device: &ash::Device, frame_layout: vk::DescriptorSetLayout) -> Result<()> {
        use vk::DescriptorType as D;
        let kinds = [
            (D::STORAGE_IMAGE, MIPS),
            (D::SAMPLED_IMAGE, 1),
            (D::STORAGE_IMAGE, 1),
            (D::SAMPLED_IMAGE, 1),
            (D::STORAGE_IMAGE, 1),
            (D::SAMPLER, 1),
            (D::SAMPLER, 1),
        ];
        let bindings: Vec<_> = (kinds.iter().enumerate())
            .map(|(i, &(kind, count))| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i as u32)
                    .descriptor_type(kind)
                    .descriptor_count(count)
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
            .size(80)];
        self.layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&layouts)
                .push_constant_ranges(&push),
            None,
        )?;
        let sizes = kinds.map(|(ty, count)| vk::DescriptorPoolSize::default().ty(ty).descriptor_count(count));
        self.pool = device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes),
            None,
        )?;
        self.set = device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.pool)
                .set_layouts(&[self.set_layout]),
        )?[0];
        // Nearest with a zero border; integer and float images need their own border color.
        for (sampler, border) in self.samplers.iter_mut().zip([
            vk::BorderColor::INT_TRANSPARENT_BLACK,
            vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
        ]) {
            let mode = vk::SamplerAddressMode::CLAMP_TO_BORDER;
            *sampler = device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(mode)
                    .address_mode_v(mode)
                    .address_mode_w(mode)
                    .max_lod(vk::LOD_CLAMP_NONE)
                    .border_color(border),
                None,
            )?;
        }
        let infos = self.samplers.map(|s| [vk::DescriptorImageInfo::default().sampler(s)]);
        let writes: Vec<_> = (infos.iter().enumerate())
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(self.set)
                    .dst_binding(5 + i as u32)
                    .descriptor_type(D::SAMPLER)
                    .image_info(info)
            })
            .collect();
        device.update_descriptor_sets(&writes, &[]);
        let entries: [(&[u8], &std::ffi::CStr); 3] = [
            (spirv!("ao_prepass.prepassMain"), c"prepassMain"),
            (spirv!("ao_trace.traceMain"), c"traceMain"),
            (spirv!("ao_filter.filterMain"), c"filterMain"),
        ];
        for (pipeline, (code, entry)) in self.pipelines.iter_mut().zip(entries) {
            *pipeline = gpu::compute_pipeline(device, vk::PipelineCache::null(), self.layout, code, entry)?;
        }
        Ok(())
    }

    /// (Re)create the working images for a viewport of `size` and point the
    /// set at them and at `output`.
    unsafe fn resize(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        size: [u32; 2],
        output: vk::ImageView,
    ) -> Result<()> {
        if self.key == (size, output) && self.packed.is_some() {
            return Ok(());
        }
        // Earlier frames in flight may still use the old images and set.
        device.device_wait_idle()?;
        self.destroy_images(device, allocator);
        self.key = (size, output);
        let extent = vk::Extent2D { width: size[0], height: size[1] };
        let levels = MIPS.min(32 - size[0].max(size[1]).leading_zeros());
        let usage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED;
        let color = vk::ImageAspectFlags::COLOR;
        let packed = gpu::create_image(device, allocator, "ao packed", vk::Format::R32_UINT, extent, levels, usage, color)?;
        for level in 0..levels {
            self.mip_views.push(device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(packed.image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R32_UINT)
                    .subresource_range(gpu::subresource(color).base_mip_level(level)),
                None,
            )?);
        }
        let packed_view = packed.view;
        self.packed = Some(packed);
        let raw = gpu::create_image(device, allocator, "ao raw", vk::Format::R32_SFLOAT, extent, 1, usage, color)?;
        let raw_view = raw.view;
        self.raw = Some(raw);

        let info = |view| vk::DescriptorImageInfo::default().image_view(view).image_layout(vk::ImageLayout::GENERAL);
        // Levels past the last are never written; they repeat it to stay valid.
        let mips: Vec<_> = (0..MIPS as usize)
            .map(|i| info(self.mip_views[i.min(self.mip_views.len() - 1)]))
            .collect();
        let single = [packed_view, raw_view, output].map(|view| [info(view)]);
        use vk::DescriptorType as D;
        let set = self.set;
        fn write(set: vk::DescriptorSet, binding: u32, kind: D, infos: &[vk::DescriptorImageInfo]) -> vk::WriteDescriptorSet<'_> {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(binding)
                .descriptor_type(kind)
                .image_info(infos)
        }
        device.update_descriptor_sets(
            &[
                write(set, 0, D::STORAGE_IMAGE, &mips),
                write(set, 1, D::SAMPLED_IMAGE, &single[0]),
                write(set, 2, D::STORAGE_IMAGE, &single[1]),
                write(set, 3, D::SAMPLED_IMAGE, &single[1]),
                write(set, 4, D::STORAGE_IMAGE, &single[2]),
            ],
            &[],
        );
        Ok(())
    }

    /// Write this frame's occlusion into `output` (a swapchain-sized
    /// `R32_SFLOAT` image), or 1 everywhere when `radius` is not positive.
    /// Leaves `output` in `GENERAL`, readable by compute shaders, and the
    /// frame set unbound.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn record(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        frame_layout: vk::DescriptorSetLayout,
        cmd: vk::CommandBuffer,
        frame_set: vk::DescriptorSet,
        viewport: [u32; 4],
        output: (vk::Image, vk::ImageView),
        projection: [f32; 2],
        near: f32,
        radius: f32,
        seed: u32,
    ) -> Result<()> {
        use vk::AccessFlags2 as A;
        use vk::ImageLayout as L;
        use vk::PipelineStageFlags2 as S;
        let color = vk::ImageAspectFlags::COLOR;
        let size = [viewport[2], viewport[3]];
        let barrier = |images: &[vk::ImageMemoryBarrier2]| {
            device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(images));
        };
        let computed = (S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE);
        let read = (S::COMPUTE_SHADER, A::SHADER_SAMPLED_READ | A::SHADER_STORAGE_READ);
        if radius <= 0.0 || size.contains(&0) {
            // Last frame's resolve may still read it.
            barrier(&[gpu::image_barrier(output.0, color, S::COMPUTE_SHADER, A::NONE, S::CLEAR, A::TRANSFER_WRITE, L::UNDEFINED, L::GENERAL)]);
            let one = vk::ClearColorValue { float32: [1.0; 4] };
            device.cmd_clear_color_image(cmd, output.0, L::GENERAL, &one, &[gpu::subresource(color)]);
            gpu::memory_barrier(device, cmd, (S::CLEAR, A::TRANSFER_WRITE), read);
            return Ok(());
        }
        if self.set == vk::DescriptorSet::null() {
            if let Err(error) = self.init(device, frame_layout) {
                self.destroy(device, allocator);
                return Err(error);
            }
        }
        self.resize(device, allocator, size, output.1)?;
        let (packed, raw) = (self.packed.as_ref().unwrap(), self.raw.as_ref().unwrap());
        // Last frame's passes and resolve may still read these; this frame overwrites them.
        let discard = |image, levels| {
            let mut b = gpu::image_barrier(image, color, S::COMPUTE_SHADER, A::NONE, S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE, L::UNDEFINED, L::GENERAL);
            b.subresource_range.level_count = levels;
            b
        };
        // Sky pixels keep stale output, but the resolve never reads them.
        barrier(&[
            discard(packed.image, vk::REMAINING_MIP_LEVELS),
            discard(raw.image, 1),
            discard(output.0, 1),
        ]);
        let (prepass, trace, filter) = params(size, projection, near, radius, seed);
        device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[frame_set, self.set], &[]);
        let [width, height] = size;
        let passes: [(&[u32], [u32; 2]); 3] = [
            (&prepass, [width.div_ceil(8), height.div_ceil(8)]),
            (&trace, [width.div_ceil(16), height.div_ceil(16)]),
            // A thread per 2x2 block, centred on odd pixels up to the far edge.
            (&filter, [(width / 2 + 1).div_ceil(8), (height / 2 + 1).div_ceil(8)]),
        ];
        for (&pipeline, (words, groups)) in self.pipelines.iter().zip(passes) {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::cast_slice(words));
            device.cmd_dispatch(cmd, groups[0], groups[1], 1);
            gpu::memory_barrier(device, cmd, computed, read);
        }
        Ok(())
    }

    unsafe fn destroy_images(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        for view in self.mip_views.drain(..) {
            device.destroy_image_view(view, None);
        }
        for image in [self.packed.take(), self.raw.take()].into_iter().flatten() {
            gpu::destroy_image(device, allocator, image);
        }
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        self.destroy_images(device, allocator);
        for pipeline in self.pipelines {
            device.destroy_pipeline(pipeline, None);
        }
        for sampler in self.samplers {
            device.destroy_sampler(sampler, None);
        }
        device.destroy_descriptor_pool(self.pool, None);
        device.destroy_pipeline_layout(self.layout, None);
        device.destroy_descriptor_set_layout(self.set_layout, None);
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The trace's projection and unprojection undo each other, as CORTAO's
    /// `populateMainPassParameters` arranges, for Vulkan's flipped y.
    #[test]
    fn projection_inverts_unprojection() {
        let (prepass, trace, _) = params([1920, 1080], [0.8, -1.4], 0.1, 1.0, 7);
        let f = |i: usize| f32::from_bits(trace[i]);
        for uv in [[0.0, 0.0], [0.25, 0.9], [1.0, 1.0]] {
            let dir = [uv[0] * f(4) + f(6), uv[1] * f(5) + f(7)];
            // CORTAO projects view (x, y, z) to (px x + pz z, py y + pw z) / -z; here z = -1.
            let back = [f(8) * dir[0] - f(10), f(9) * dir[1] - f(11)];
            assert!((back[0] - uv[0]).abs() < 1e-5 && (back[1] - uv[1]).abs() < 1e-5);
        }
        // Top of the viewport (uv.y 0) looks up in view space, through the flipped y.
        assert!(f(7) > 0.0);
        assert_eq!(prepass[4], 8);
        assert_eq!(f32::from_bits(prepass[1]), 10.0);
    }
}
