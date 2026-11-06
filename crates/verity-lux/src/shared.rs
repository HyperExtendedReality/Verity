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
}

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
            Res::View(..) => [vk::DescriptorBufferInfo::default()],
        })
        .collect();
    let images: Vec<[vk::DescriptorImageInfo; 1]> = entries
        .iter()
        .map(|(_, res)| match res {
            Res::View(view, layout) => [vk::DescriptorImageInfo::default()
                .image_view(*view)
                .image_layout(*layout)],
            Res::Buffer(_) => [vk::DescriptorImageInfo::default()],
        })
        .collect();
    let writes: Vec<_> = entries
        .iter()
        .enumerate()
        .map(|(i, (binding, res))| {
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(*binding)
                .descriptor_type(kinds[binding]);
            match res {
                Res::Buffer(_) => write.buffer_info(&buffers[i]),
                Res::View(..) => write.image_info(&images[i]),
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
