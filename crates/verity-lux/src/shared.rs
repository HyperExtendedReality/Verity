//! Vulkan helpers every GI method uses.

#![allow(unsafe_op_in_unsafe_fn)]

use ash::vk;
use gpu_allocator::MemoryLocation;
use std::{collections::HashMap, mem, ops::Range};
use verity_render::{
    Gpu,
    extension::Result,
    gpu::{self, Buffer},
};
use vk::DescriptorType as D;

/// What a descriptor points at.
#[derive(Clone, Copy)]
pub(crate) enum Res {
    Buffer(vk::Buffer),
    /// A view in `GENERAL` layout, or a depth view in `DEPTH_READ_ONLY_OPTIMAL`.
    View(vk::ImageView, vk::ImageLayout),
    Accel(vk::AccelerationStructureKHR),
}

/// Shader sampling data: a 64×64 tile of four u16 ranks per texel, then Sobol columns.
pub(crate) const SAMPLING_DATA: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/sampling_data.bin"));
/// Blue-noise prefix for methods that do not use the Sobol sampler.
pub(crate) const BLUE_NOISE: &[u8] = SAMPLING_DATA.split_at(64 * 64 * 8).0;

pub(crate) unsafe fn write_set(
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
            _ => [vk::DescriptorBufferInfo::default()],
        })
        .collect();
    let images: Vec<[vk::DescriptorImageInfo; 1]> = entries
        .iter()
        .map(|(_, res)| match res {
            Res::View(view, layout) => [vk::DescriptorImageInfo::default()
                .image_view(*view)
                .image_layout(*layout)],
            _ => [vk::DescriptorImageInfo::default()],
        })
        .collect();
    let mut accels: Vec<_> = entries
        .iter()
        .map(|(_, res)| {
            let handles = match res {
                Res::Accel(handle) => std::slice::from_ref(handle),
                _ => &[],
            };
            vk::WriteDescriptorSetAccelerationStructureKHR::default()
                .acceleration_structures(handles)
        })
        .collect();
    let writes: Vec<_> = entries
        .iter()
        .zip(&mut accels)
        .enumerate()
        .map(|(i, ((binding, res), accel))| {
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(*binding)
                .descriptor_type(kinds[binding]);
            match res {
                Res::Buffer(_) => write.buffer_info(&buffers[i]),
                Res::View(..) => write.image_info(&images[i]),
                Res::Accel(_) => write.descriptor_count(1).push_next(accel),
            }
        })
        .collect();
    device.update_descriptor_sets(&writes, &[]);
}

/// Byte ranges of a device-local buffer to rewrite from `source`, which holds
/// what the whole buffer should.
pub(crate) struct Patch<'a> {
    pub target: &'a Buffer,
    pub source: &'a [u8],
    pub ranges: Vec<Range<usize>>,
}

/// Rewrite byte ranges of device-local buffers from their source bytes, through
/// one staging copy. The GPU must be idle.
pub(crate) unsafe fn upload_ranges(
    gpu: &mut Gpu,
    pool: &mut vk::CommandPool,
    patches: &[Patch],
) -> Result<()> {
    let device = gpu.device;
    if *pool == vk::CommandPool::null() {
        *pool = device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(gpu.queue_family)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT),
            None,
        )?;
    }
    let mut bytes = Vec::new();
    let mut copies = Vec::with_capacity(patches.len());
    for patch in patches {
        let (target, source) = (patch.target, patch.source);
        let regions: Vec<vk::BufferCopy> = (patch.ranges.iter())
            .filter(|range| !range.is_empty())
            .map(|range| {
                let src_offset = bytes.len() as u64;
                bytes.extend_from_slice(&source[range.clone()]);
                vk::BufferCopy {
                    src_offset,
                    dst_offset: range.start as u64,
                    size: range.len() as u64,
                }
            })
            .collect();
        copies.push((target.buffer, regions));
    }
    if bytes.is_empty() {
        return Ok(());
    }
    let mut staging = gpu::create_buffer(
        device,
        gpu.allocator,
        "gi staging",
        bytes.len() as u64,
        vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,
    )?;
    staging.write(&bytes);
    let cmd = device.allocate_command_buffers(
        &vk::CommandBufferAllocateInfo::default()
            .command_pool(*pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1),
    )?[0];
    device.begin_command_buffer(
        cmd,
        &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
    )?;
    for (target, regions) in &copies {
        if !regions.is_empty() {
            device.cmd_copy_buffer(cmd, staging.buffer, *target, regions);
        }
    }
    device.end_command_buffer(cmd)?;
    let commands = [cmd];
    let result = device
        .queue_submit(
            gpu.queue,
            &[vk::SubmitInfo::default().command_buffers(&commands)],
            vk::Fence::null(),
        )
        .and_then(|()| device.queue_wait_idle(gpu.queue));
    device.free_command_buffers(*pool, &commands);
    gpu::destroy_buffer(device, gpu.allocator, staging);
    Ok(result?)
}

/// Replace a device-local buffer's contents through a staging copy, making
/// `pool` on first use.
pub(crate) unsafe fn upload(
    gpu: &mut Gpu,
    pool: &mut vk::CommandPool,
    target: &mut Buffer,
    name: &str,
    bytes: &[u8],
) -> Result<()> {
    let device = gpu.device;
    if *pool == vk::CommandPool::null() {
        *pool = device.create_command_pool(
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
            .command_pool(*pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1),
    )?[0];
    device.begin_command_buffer(
        cmd,
        &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
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
    device.free_command_buffers(*pool, &commands);
    gpu::destroy_buffer(device, gpu.allocator, staging);
    Ok(result?)
}

/// Timestamps a frame slot can hold: one per pass and one to start from.
const TIMESTAMPS: u32 = 128;

/// GPU time per pass, from a timestamp after each.
pub(crate) struct PassTimer {
    pool: vk::QueryPool,
    period: f32,
    /// The passes each slot's last frame stamped, in order.
    stamped: [Vec<&'static str>; 2],
    /// Milliseconds per pass in the latest finished frame, repeats summed.
    pub times: Vec<(&'static str, f32)>,
}

impl PassTimer {
    pub unsafe fn new(gpu: &Gpu) -> Result<Self> {
        let pool = gpu.device.create_query_pool(
            &vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(2 * TIMESTAMPS),
            None,
        )?;
        let properties = gpu.instance.get_physical_device_properties(gpu.physical);
        Ok(Self {
            pool,
            period: properties.limits.timestamp_period,
            stamped: Default::default(),
            times: Vec::new(),
        })
    }

    /// Read this slot's finished frame and start timing its next. The slot's
    /// last frame must be done.
    pub unsafe fn begin(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        slot: usize,
    ) -> Result<()> {
        let first = slot as u32 * TIMESTAMPS;
        let stamped = mem::take(&mut self.stamped[slot]);
        if !stamped.is_empty() {
            let mut ticks = vec![0u64; stamped.len() + 1];
            device.get_query_pool_results(
                self.pool,
                first,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64,
            )?;
            self.times.clear();
            for (name, pair) in stamped.iter().zip(ticks.windows(2)) {
                let ms = pair[1].saturating_sub(pair[0]) as f32 * self.period * 1e-6;
                match self.times.iter_mut().find(|(n, _)| n == name) {
                    Some((_, total)) => *total += ms,
                    None => self.times.push((name, ms)),
                }
            }
        }
        device.cmd_reset_query_pool(cmd, self.pool, first, TIMESTAMPS);
        device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::ALL_COMMANDS, self.pool, first);
        Ok(())
    }

    /// Note that a pass just finished.
    pub unsafe fn stamp(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        slot: usize,
        name: &'static str,
    ) {
        let stamped = &mut self.stamped[slot];
        stamped.push(name);
        debug_assert!((stamped.len() as u32) < TIMESTAMPS);
        device.cmd_write_timestamp2(
            cmd,
            vk::PipelineStageFlags2::ALL_COMMANDS,
            self.pool,
            slot as u32 * TIMESTAMPS + stamped.len() as u32,
        );
    }

    /// The latest frame's whole time, once there is one.
    pub fn total(&self) -> Option<std::time::Duration> {
        let total: f32 = self.times.iter().map(|(_, ms)| ms).sum();
        (total > 0.0).then(|| std::time::Duration::from_secs_f32(total * 1e-3))
    }

    pub unsafe fn destroy(self, device: &ash::Device) {
        device.destroy_query_pool(self.pool, None);
    }
}
