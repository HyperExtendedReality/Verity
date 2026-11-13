//! Per-frame GPU meshlet frustum and current-frame reverse-Z occlusion culling.
//!
//! The depth prepass draws all geometry, including alpha-tested materials; the
//! min-depth pyramid reduces subsequent visibility work, not prepass geometry
//! cost. No temporal depth is used. Indirect commands retain primitive-local
//! `firstVertex` for visibility resolve.

use crate::{gpu, Result, FRAMES};
use ash::vk;
use ash::vk::{AccessFlags2 as A, PipelineStageFlags2 as S};
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use gpu_allocator::{vulkan::Allocator, MemoryLocation};
use std::mem;

const WORKGROUP_SIZE: u32 = 64;
const COMMAND_SIZE: u64 = 16;

/// One meshlet of an already selected LOD, in that LOD's local vertex space.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct Job {
    view_proj: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    bounds_min: [f32; 4],
    bounds_max: [f32; 4],
    first_vertex: u32,
    vertex_count: u32,
    padding: [u32; 2],
}

impl Job {
    pub(crate) fn new(
        view_proj: Mat4,
        model: Mat4,
        min: [f32; 3],
        max: [f32; 3],
        first_vertex: u32,
        vertex_count: u32,
    ) -> Self {
        Self {
            view_proj: view_proj.to_cols_array_2d(),
            model: model.to_cols_array_2d(),
            bounds_min: [min[0], min[1], min[2], 0.0],
            bounds_max: [max[0], max[1], max[2], 0.0],
            first_vertex,
            vertex_count,
            padding: [0; 2],
        }
    }
}

#[derive(Default)]
struct Frame {
    set: vk::DescriptorSet,
    jobs: gpu::Buffer,
    commands: gpu::Buffer,
    pyramid: gpu::Buffer,
    pyramid_extent: vk::Extent2D,
    pyramid_levels: u32,
    count: u32,
}

pub(crate) struct Culler {
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pyramid_pipeline: vk::Pipeline,
    occlusion_pipeline: vk::Pipeline,
    frames: Vec<Frame>,
    max_storage_range: u64,
    max_groups: u32,
    max_groups_y: u32,
}

impl Culler {
    /// Creates resources independent of the renderer's material descriptors.
    pub(crate) unsafe fn new(
        device: &ash::Device,
        limits: vk::PhysicalDeviceLimits,
    ) -> Result<Self> {
        let bindings = [
            (0, vk::DescriptorType::STORAGE_BUFFER),
            (1, vk::DescriptorType::STORAGE_BUFFER),
            (2, vk::DescriptorType::SAMPLED_IMAGE),
            (3, vk::DescriptorType::STORAGE_BUFFER),
        ]
        .map(|(binding, ty)| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(ty)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        });
        let set_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )?;
        let mut pool = vk::DescriptorPool::null();
        let mut layout = vk::PipelineLayout::null();
        let mut pipeline = vk::Pipeline::null();
        let mut pyramid_pipeline = vk::Pipeline::null();
        let mut occlusion_pipeline = vk::Pipeline::null();
        let result = (|| -> Result<Self> {
            let pool_sizes = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count((FRAMES * 3) as u32),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(FRAMES as u32),
            ];
            pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(FRAMES as u32)
                    .pool_sizes(&pool_sizes),
                None,
            )?;
            let layouts = vec![set_layout; FRAMES];
            let sets = device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )?;
            let push = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(48)];
            layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[set_layout])
                    .push_constant_ranges(&push),
                None,
            )?;
            pipeline = gpu::compute_pipeline(
                device,
                vk::PipelineCache::null(),
                layout,
                include_bytes!(concat!(env!("OUT_DIR"), "/gpu_culling.cullMain.spv")),
                c"cullMain",
            )?;
            pyramid_pipeline = gpu::compute_pipeline(
                device,
                vk::PipelineCache::null(),
                layout,
                include_bytes!(concat!(env!("OUT_DIR"), "/gpu_culling.pyramidMain.spv")),
                c"pyramidMain",
            )?;
            occlusion_pipeline = gpu::compute_pipeline(
                device,
                vk::PipelineCache::null(),
                layout,
                include_bytes!(concat!(env!("OUT_DIR"), "/gpu_culling.occlusionMain.spv")),
                c"occlusionMain",
            )?;
            Ok(Self {
                set_layout,
                pool,
                layout,
                pipeline,
                pyramid_pipeline,
                occlusion_pipeline,
                frames: sets
                    .into_iter()
                    .map(|set| Frame {
                        set,
                        ..Default::default()
                    })
                    .collect(),
                max_storage_range: limits.max_storage_buffer_range as u64,
                max_groups: limits.max_compute_work_group_count[0],
                max_groups_y: limits.max_compute_work_group_count[1],
            })
        })();
        if result.is_err() {
            device.destroy_pipeline(occlusion_pipeline, None);
            device.destroy_pipeline(pyramid_pipeline, None);
            device.destroy_pipeline(pipeline, None);
            device.destroy_pipeline_layout(layout, None);
            device.destroy_descriptor_pool(pool, None);
            device.destroy_descriptor_set_layout(set_layout, None);
        }
        result
    }

    /// Upload after the frame's fence has signalled. Returns false if the whole
    /// job set exceeds device limits: draw every meshlet directly in that case.
    pub(crate) unsafe fn prepare(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        frame: usize,
        jobs: &[Job],
    ) -> Result<bool> {
        let f = &mut self.frames[frame];
        f.count = 0;
        if jobs.is_empty() {
            return Ok(true);
        }
        let count = match u32::try_from(jobs.len()) {
            Ok(count) => count,
            Err(_) => return Ok(false),
        };
        let input_size = (count as u64) * mem::size_of::<Job>() as u64;
        let output_size = (count as u64) * COMMAND_SIZE;
        if input_size > self.max_storage_range
            || output_size > self.max_storage_range
            || count.div_ceil(WORKGROUP_SIZE) > self.max_groups
        {
            return Ok(false);
        }
        if f.jobs.size < input_size || f.commands.size < output_size {
            // Allocate both before replacing either, so an allocation error leaves
            // the old descriptor set and both old buffers intact.
            let input = gpu::create_buffer(
                device,
                allocator,
                "meshlet culling jobs",
                input_size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::CpuToGpu,
            )?;
            let output = match gpu::create_buffer(
                device,
                allocator,
                "meshlet indirect draws",
                output_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
                MemoryLocation::GpuOnly,
            ) {
                Ok(buffer) => buffer,
                Err(error) => {
                    gpu::destroy_buffer(device, allocator, input);
                    return Err(error);
                }
            };
            let old_input = mem::replace(&mut f.jobs, input);
            let old_output = mem::replace(&mut f.commands, output);
            gpu::destroy_buffer(device, allocator, old_input);
            gpu::destroy_buffer(device, allocator, old_output);
            let input_info = [f.jobs.info()];
            let output_info = [f.commands.info()];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(f.set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&input_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(f.set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&output_info),
            ];
            device.update_descriptor_sets(&writes, &[]);
        }
        f.jobs.write(bytemuck::cast_slice(jobs));
        f.count = count;
        Ok(true)
    }

    /// Prepare a per-frame reverse-Z pyramid after this frame's fence.
    /// Failure or device-limit overflow leaves frustum-only culling available.
    pub(crate) unsafe fn prepare_hiz(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        frame: usize,
        depth_view: vk::ImageView,
        extent: vk::Extent2D,
    ) -> Result<bool> {
        let f = &mut self.frames[frame];
        f.pyramid_levels = 0;
        if f.count == 0
            || extent.width == 0
            || extent.height == 0
            || depth_view == vk::ImageView::null()
            || extent.width.div_ceil(8) > self.max_groups
            || extent.height.div_ceil(8) > self.max_groups_y
        {
            return Ok(false);
        }
        let (mut width, mut height) = (extent.width, extent.height);
        let (mut texels, mut levels) = (0u64, 0u32);
        loop {
            texels = match texels.checked_add(u64::from(width) * u64::from(height)) {
                Some(value) => value,
                None => return Ok(false),
            };
            levels += 1;
            if width == 1 && height == 1 {
                break;
            }
            width = width.div_ceil(2);
            height = height.div_ceil(2);
        }
        let Some(size) = texels.checked_mul(4) else {
            return Ok(false);
        };
        if texels > u32::MAX as u64 || size > self.max_storage_range {
            return Ok(false);
        }
        if f.pyramid.size < size {
            let pyramid = gpu::create_buffer(
                device,
                allocator,
                "reverse-z depth pyramid",
                size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::GpuOnly,
            )?;
            let old = mem::replace(&mut f.pyramid, pyramid);
            gpu::destroy_buffer(device, allocator, old);
        }
        let image = [vk::DescriptorImageInfo::default()
            .image_view(depth_view)
            .image_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)];
        let pyramid = [f.pyramid.info()];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(f.set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&image),
            vk::WriteDescriptorSet::default()
                .dst_set(f.set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&pyramid),
        ];
        device.update_descriptor_sets(&writes, &[]);
        f.pyramid_extent = extent;
        f.pyramid_levels = levels;
        Ok(true)
    }

    /// Build current-frame min depth and replace the frustum-only commands.
    /// The caller transitions the just-rendered depth to read-only first.
    pub(crate) unsafe fn record_occlusion(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame: usize,
        extent: vk::Extent2D,
        viewport: [u32; 4],
    ) {
        let f = &self.frames[frame];
        assert!(
            f.count != 0
                && f.pyramid_levels != 0
                && f.pyramid_extent.width == extent.width
                && f.pyramid_extent.height == extent.height
        );
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.layout,
            0,
            &[f.set],
            &[],
        );
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pyramid_pipeline);
        let (mut width, mut height) = (extent.width, extent.height);
        let (mut offset, mut previous_offset) = (0u32, 0u32);
        let (mut previous_width, mut previous_height) = (width, height);
        for level in 0..f.pyramid_levels {
            let params = [
                f.count,
                level,
                previous_offset,
                offset,
                previous_width,
                previous_height,
                width,
                height,
                0,
                0,
                0,
                0,
            ];
            device.cmd_push_constants(
                cmd,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::cast_slice(&params),
            );
            device.cmd_dispatch(cmd, width.div_ceil(8), height.div_ceil(8), 1);
            gpu::memory_barrier(
                device,
                cmd,
                (S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
                (S::COMPUTE_SHADER, A::SHADER_STORAGE_READ),
            );
            previous_offset = offset;
            offset += width * height;
            previous_width = width;
            previous_height = height;
            width = width.div_ceil(2);
            height = height.div_ceil(2);
        }
        gpu::memory_barrier(
            device,
            cmd,
            (S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
            (S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
        );
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.occlusion_pipeline);
        let params = [
            f.count,
            f.pyramid_levels,
            0,
            0,
            extent.width,
            extent.height,
            0,
            0,
            viewport[0],
            viewport[1],
            viewport[2],
            viewport[3],
        ];
        device.cmd_push_constants(
            cmd,
            self.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&params),
        );
        device.cmd_dispatch(cmd, f.count.div_ceil(WORKGROUP_SIZE), 1, 1);
        gpu::memory_barrier(
            device,
            cmd,
            (S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
            (S::DRAW_INDIRECT, A::INDIRECT_COMMAND_READ),
        );
    }

    /// Record before dynamic rendering. The frame's uploaded jobs must belong
    /// to this submission; the barrier makes every output visible to draws.
    pub(crate) unsafe fn record(&self, device: &ash::Device, cmd: vk::CommandBuffer, frame: usize) {
        let f = &self.frames[frame];
        if f.count == 0 {
            return;
        }
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.layout,
            0,
            &[f.set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            self.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &f.count.to_ne_bytes(),
        );
        device.cmd_dispatch(cmd, f.count.div_ceil(WORKGROUP_SIZE), 1, 1);
        gpu::memory_barrier(
            device,
            cmd,
            (S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE),
            (S::DRAW_INDIRECT, A::INDIRECT_COMMAND_READ),
        );
    }

    /// Draw one job after binding its instance's graphics push constant.
    /// A draw count of one avoids requiring the optional multiDrawIndirect feature.
    pub(crate) unsafe fn draw(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame: usize,
        job_index: usize,
    ) {
        let f = &self.frames[frame];
        assert!(job_index < f.count as usize);
        device.cmd_draw_indirect(
            cmd,
            f.commands.buffer,
            job_index as u64 * COMMAND_SIZE,
            1,
            COMMAND_SIZE as u32,
        );
    }

    /// The GPU must have finished every frame that used this culler.
    pub(crate) unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        for f in self.frames {
            gpu::destroy_buffer(device, allocator, f.jobs);
            gpu::destroy_buffer(device, allocator, f.commands);
            gpu::destroy_buffer(device, allocator, f.pyramid);
        }
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline(self.pyramid_pipeline, None);
        device.destroy_pipeline(self.occlusion_pipeline, None);
        device.destroy_pipeline_layout(self.layout, None);
        device.destroy_descriptor_pool(self.pool, None);
        device.destroy_descriptor_set_layout(self.set_layout, None);
    }
}
