//! Buffer, image and pipeline helpers over `gpu-allocator`.

use ash::vk;
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
    MemoryLocation,
};
use std::{error::Error, ffi::CStr, marker::PhantomData};

type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;

/// A buffer and its memory. The default value is a null buffer; destroying it is a no-op.
#[derive(Default)]
pub struct Buffer {
    pub buffer: vk::Buffer,
    pub size: u64,
    /// Where shaders reach the buffer through a pointer; 0 for the null buffer.
    pub address: vk::DeviceAddress,
    allocation: Option<Allocation>,
}

/// A GPU pointer to `T`s, laid out as a Slang `T*`.
#[repr(transparent)]
pub struct DeviceAddress<T>(pub vk::DeviceAddress, PhantomData<fn() -> T>);

impl<T> DeviceAddress<T> {
    pub fn new(address: vk::DeviceAddress) -> Self {
        Self(address, PhantomData)
    }
}

impl<T> Clone for DeviceAddress<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for DeviceAddress<T> {}
impl<T> std::fmt::Debug for DeviceAddress<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DeviceAddress({:#x})", self.0)
    }
}
// SAFETY: a plain u64; the marker is zero-sized.
unsafe impl<T: 'static> bytemuck::Zeroable for DeviceAddress<T> {}
unsafe impl<T: 'static> bytemuck::Pod for DeviceAddress<T> {}

/// One host-visible buffer per frame that this frame's CPU-written arrays are
/// packed into; shaders find each array through its address. Reset once the
/// frame's fence has signaled.
#[derive(Default)]
pub struct FrameArena {
    pub buffer: Buffer,
    used: u64,
}

/// Every array starts 16-byte aligned, for 128-bit loads.
const ARENA_ALIGN: u64 = 16;

/// The byte range a `size`-byte array takes after `used` bytes, if it fits in `capacity`.
fn bump(used: u64, size: u64, capacity: u64) -> Option<(u64, u64)> {
    let start = used.checked_next_multiple_of(ARENA_ALIGN)?;
    let end = start.checked_add(size)?;
    (end <= capacity).then_some((start, end))
}

impl FrameArena {
    /// Bytes an arena needs to hold arrays of these sizes.
    pub fn needed(sizes: impl IntoIterator<Item = usize>) -> u64 {
        sizes
            .into_iter()
            .map(|size| (size as u64).next_multiple_of(ARENA_ALIGN))
            .sum()
    }

    /// Empty the arena, growing it first to at least `capacity` bytes.
    ///
    /// # Safety
    /// The GPU must be done with the arena's buffer.
    pub unsafe fn reset(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        capacity: u64,
    ) -> Result<()> {
        self.used = 0;
        if capacity > self.buffer.size {
            destroy_buffer(device, allocator, std::mem::take(&mut self.buffer));
            self.buffer = create_buffer(
                device,
                allocator,
                "frame arena",
                capacity.next_power_of_two(),
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::CpuToGpu,
            )?;
        }
        Ok(())
    }

    /// Copy `items` into the arena; `None` when they do not fit.
    pub fn push<T: bytemuck::Pod>(&mut self, items: &[T]) -> Option<DeviceAddress<T>> {
        let bytes: &[u8] = bytemuck::cast_slice(items);
        let (start, end) = bump(self.used, bytes.len() as u64, self.buffer.size)?;
        // An empty frame leaves the arena unallocated: nothing to map or copy.
        if !bytes.is_empty() {
            self.buffer.write_at(start as usize, bytes);
        }
        self.used = end;
        Some(DeviceAddress(self.buffer.address + start, PhantomData))
    }
}

impl Buffer {
    /// Copy `bytes` to the start of a host-visible buffer.
    pub fn write(&mut self, bytes: &[u8]) {
        self.write_at(0, bytes);
    }

    /// Copy `bytes` to `offset` bytes into a host-visible buffer.
    pub fn write_at(&mut self, offset: usize, bytes: &[u8]) {
        let mapped = self
            .allocation
            .as_mut()
            .and_then(Allocation::mapped_slice_mut)
            .expect("buffer is host visible");
        mapped[offset..offset + bytes.len()].copy_from_slice(bytes);
    }

    /// The contents of a host-visible buffer.
    pub fn read(&self) -> &[u8] {
        self.allocation
            .as_ref()
            .and_then(Allocation::mapped_slice)
            .expect("buffer is host visible")
    }

    pub fn info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::default()
            .buffer(self.buffer)
            .range(vk::WHOLE_SIZE)
    }
}

pub struct Image {
    pub image: vk::Image,
    pub view: vk::ImageView,
    allocation: Allocation,
}

/// # Safety
/// `device` must be the device `allocator` was created for.
pub unsafe fn create_buffer(
    device: &ash::Device,
    allocator: &mut Allocator,
    name: &str,
    size: u64,
    usage: vk::BufferUsageFlags,
    location: MemoryLocation,
) -> Result<Buffer> {
    let buffer = device.create_buffer(
        &vk::BufferCreateInfo::default()
            .size(size.max(16))
            // Every buffer has an address, so any can be read through a pointer.
            .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
            .sharing_mode(vk::SharingMode::EXCLUSIVE),
        None,
    )?;
    let allocation = allocator.allocate(&AllocationCreateDesc {
        name,
        requirements: device.get_buffer_memory_requirements(buffer),
        location,
        linear: true,
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
    });
    let allocation = match allocation {
        Ok(allocation) => allocation,
        Err(error) => {
            device.destroy_buffer(buffer, None);
            return Err(error.into());
        }
    };
    if let Err(error) = device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
    {
        device.destroy_buffer(buffer, None);
        allocator.free(allocation)?;
        return Err(error.into());
    }
    let address =
        device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));
    Ok(Buffer {
        buffer,
        size,
        address,
        allocation: Some(allocation),
    })
}

/// # Safety
/// The GPU must be done with the buffer.
pub unsafe fn destroy_buffer(device: &ash::Device, allocator: &mut Allocator, buffer: Buffer) {
    device.destroy_buffer(buffer.buffer, None);
    if let Some(allocation) = buffer.allocation {
        // Freeing only fails for double frees, which ownership rules out.
        let _ = allocator.free(allocation);
    }
}

/// The present mode to use from those a surface offers. Uncapped, immediate comes
/// first, though it may tear: Windows throttles mailbox in a window (120 FPS on a
/// 60 Hz display, against 617 immediate). FIFO, which every surface has, waits for vsync.
pub fn present_mode(available: &[vk::PresentModeKHR], uncapped: bool) -> vk::PresentModeKHR {
    use vk::PresentModeKHR as P;
    let wanted = [P::IMMEDIATE, P::MAILBOX];
    (wanted.into_iter())
        .find(|mode| uncapped && available.contains(mode))
        .unwrap_or(P::FIFO)
}

/// A device-local 2D image with `levels` mips and a view of all of them.
///
/// # Safety
/// `device` must be the device `allocator` was created for.
#[allow(clippy::too_many_arguments)]
pub unsafe fn create_image(
    device: &ash::Device,
    allocator: &mut Allocator,
    name: &str,
    format: vk::Format,
    extent: vk::Extent2D,
    levels: u32,
    usage: vk::ImageUsageFlags,
    aspect: vk::ImageAspectFlags,
) -> Result<Image> {
    let samples = vk::SampleCountFlags::TYPE_1;
    create_image_of(device, allocator, name, format, extent.into(), levels, usage, aspect, samples)
}

/// A single-level 2D image with `samples` samples per texel.
///
/// # Safety
/// `device` must be the device `allocator` was created for.
#[allow(clippy::too_many_arguments)]
pub unsafe fn create_multisampled_image(
    device: &ash::Device,
    allocator: &mut Allocator,
    name: &str,
    format: vk::Format,
    extent: vk::Extent2D,
    samples: vk::SampleCountFlags,
    usage: vk::ImageUsageFlags,
    aspect: vk::ImageAspectFlags,
) -> Result<Image> {
    create_image_of(device, allocator, name, format, extent.into(), 1, usage, aspect, samples)
}

/// A single-level color volume of `size`³ texels.
///
/// # Safety
/// `device` must be the device `allocator` was created for.
pub unsafe fn create_volume(
    device: &ash::Device,
    allocator: &mut Allocator,
    name: &str,
    format: vk::Format,
    size: u32,
    usage: vk::ImageUsageFlags,
) -> Result<Image> {
    let extent = vk::Extent3D {
        width: size,
        height: size,
        depth: size,
    };
    let samples = vk::SampleCountFlags::TYPE_1;
    create_image_of(device, allocator, name, format, extent, 1, usage, vk::ImageAspectFlags::COLOR, samples)
}

/// A 2D image when `extent.depth` is 1, otherwise a 3D one.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn create_image_of(
    device: &ash::Device,
    allocator: &mut Allocator,
    name: &str,
    format: vk::Format,
    extent: vk::Extent3D,
    levels: u32,
    usage: vk::ImageUsageFlags,
    aspect: vk::ImageAspectFlags,
    samples: vk::SampleCountFlags,
) -> Result<Image> {
    let (image_type, view_type) = if extent.depth > 1 {
        (vk::ImageType::TYPE_3D, vk::ImageViewType::TYPE_3D)
    } else {
        (vk::ImageType::TYPE_2D, vk::ImageViewType::TYPE_2D)
    };
    let image = device.create_image(
        &vk::ImageCreateInfo::default()
            .image_type(image_type)
            .format(format)
            .extent(extent)
            .mip_levels(levels)
            .array_layers(1)
            .samples(samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED),
        None,
    )?;
    let allocation = allocator.allocate(&AllocationCreateDesc {
        name,
        requirements: device.get_image_memory_requirements(image),
        location: MemoryLocation::GpuOnly,
        linear: false,
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
    });
    let allocation = match allocation {
        Ok(allocation) => allocation,
        Err(error) => {
            device.destroy_image(image, None);
            return Err(error.into());
        }
    };
    let view = device
        .bind_image_memory(image, allocation.memory(), allocation.offset())
        .and_then(|()| {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(view_type)
                    .format(format)
                    .subresource_range(subresource(aspect).level_count(levels)),
                None,
            )
        });
    match view {
        Ok(view) => Ok(Image {
            image,
            view,
            allocation,
        }),
        Err(error) => {
            device.destroy_image(image, None);
            allocator.free(allocation)?;
            Err(error.into())
        }
    }
}

/// # Safety
/// The GPU must be done with the image.
pub unsafe fn destroy_image(device: &ash::Device, allocator: &mut Allocator, image: Image) {
    device.destroy_image_view(image.view, None);
    device.destroy_image(image.image, None);
    let _ = allocator.free(image.allocation);
}

pub fn subresource(aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(aspect)
        .level_count(1)
        .layer_count(1)
}

/// One image layout transition, or with equal layouts just an execution and memory dependency.
#[allow(clippy::too_many_arguments)]
pub fn image_barrier(
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> vk::ImageMemoryBarrier2<'static> {
    vk::ImageMemoryBarrier2::default()
        .image(image)
        .subresource_range(subresource(aspect))
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .old_layout(old_layout)
        .new_layout(new_layout)
}

/// # Safety
/// `spirv` must be a valid SPIR-V module.
pub unsafe fn shader_module(device: &ash::Device, spirv: &[u8]) -> Result<vk::ShaderModule> {
    let words = ash::util::read_spv(&mut std::io::Cursor::new(spirv))?;
    Ok(device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?)
}

/// # Safety
/// `spirv` must be a valid compute module with entry point `entry`; `cache`
/// may be null.
pub unsafe fn compute_pipeline(
    device: &ash::Device,
    cache: vk::PipelineCache,
    layout: vk::PipelineLayout,
    spirv: &[u8],
    entry: &CStr,
) -> Result<vk::Pipeline> {
    let module = shader_module(device, spirv)?;
    let info = [vk::ComputePipelineCreateInfo::default()
        .stage(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(entry),
        )
        .layout(layout)];
    let result = device.create_compute_pipelines(cache, &info, None);
    device.destroy_shader_module(module, None);
    Ok(result.map_err(|(_, error)| error)?[0])
}

/// One global memory barrier: orders all memory, not one buffer. Caches are
/// coherent, so naming the resource buys nothing and only has to be tracked.
///
/// # Safety
/// `cmd` must be recording.
pub unsafe fn memory_barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    src: (vk::PipelineStageFlags2, vk::AccessFlags2),
    dst: (vk::PipelineStageFlags2, vk::AccessFlags2),
) {
    let barrier = [vk::MemoryBarrier2::default()
        .src_stage_mask(src.0)
        .src_access_mask(src.1)
        .dst_stage_mask(dst.0)
        .dst_access_mask(dst.1)];
    device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&barrier));
}

/// What a graphics pipeline renders into and how it tests depth.
pub struct Raster<'a> {
    pub vertex: &'a [u8],
    pub fragment: &'a [u8],
    /// `None` for depth-only passes such as shadows.
    pub color: Option<vk::Format>,
    pub depth: Option<(vk::Format, vk::CompareOp)>,
    /// `None`: dynamic, set with `cmd_set_cull_mode` before drawing, so one
    /// pipeline serves both one- and two-sided materials.
    pub cull: Option<vk::CullModeFlags>,
    /// Constant and slope-scaled depth bias, for shadow maps.
    pub depth_bias: Option<(f32, f32)>,
}

/// A triangle-list pipeline for dynamic rendering, with dynamic viewport and scissor.
///
/// # Safety
/// The shaders must be valid SPIR-V with `vertexMain` and `fragmentMain` entry points.
pub unsafe fn graphics_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    raster: &Raster<'_>,
) -> Result<vk::Pipeline> {
    graphics_pipeline_targets(device, layout, raster, &[])
}

/// [`graphics_pipeline`] with `extra_colors` after `raster.color`.
///
/// # Safety
/// As for [`graphics_pipeline`].
pub unsafe fn graphics_pipeline_targets(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    raster: &Raster<'_>,
    extra_colors: &[vk::Format],
) -> Result<vk::Pipeline> {
    pipeline_with(device, layout, raster, extra_colors, vk::SampleCountFlags::TYPE_1)
}

/// [`graphics_pipeline`] for attachments with `samples` samples each.
///
/// # Safety
/// As for [`graphics_pipeline`].
pub unsafe fn graphics_pipeline_samples(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    raster: &Raster<'_>,
    samples: vk::SampleCountFlags,
) -> Result<vk::Pipeline> {
    pipeline_with(device, layout, raster, &[], samples)
}

unsafe fn pipeline_with(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    raster: &Raster<'_>,
    extra_colors: &[vk::Format],
    samples: vk::SampleCountFlags,
) -> Result<vk::Pipeline> {
    let vs = shader_module(device, raster.vertex)?;
    let fs = match shader_module(device, raster.fragment) {
        Ok(fs) => fs,
        Err(error) => {
            device.destroy_shader_module(vs, None);
            return Err(error);
        }
    };
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vs)
            .name(c"vertexMain"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fs)
            .name(c"fragmentMain"),
    ];
    let colors: Vec<vk::Format> = raster
        .color
        .into_iter()
        .chain(extra_colors.iter().copied())
        .collect();
    let mut rendering =
        vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&colors);
    let mut depth_state = vk::PipelineDepthStencilStateCreateInfo::default();
    if let Some((format, compare)) = raster.depth {
        rendering = rendering.depth_attachment_format(format);
        depth_state = depth_state
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(compare);
    }
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let mut rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(raster.cull.unwrap_or_default())
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    if let Some((constant, slope)) = raster.depth_bias {
        rasterization = rasterization
            .depth_bias_enable(true)
            .depth_bias_constant_factor(constant)
            .depth_bias_slope_factor(slope);
    }
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(samples);
    let blend_attachment: Vec<_> = colors
        .iter()
        .map(|_| {
            vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
        })
        .collect();
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachment);
    let dynamic_states = [
        vk::DynamicState::VIEWPORT,
        vk::DynamicState::SCISSOR,
        vk::DynamicState::CULL_MODE,
    ];
    let dynamic_count = if raster.cull.is_some() { 2 } else { 3 };
    let dynamic = vk::PipelineDynamicStateCreateInfo::default()
        .dynamic_states(&dynamic_states[..dynamic_count]);
    let info = [vk::GraphicsPipelineCreateInfo::default()
        .push_next(&mut rendering)
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_state)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(layout)];
    let result = device.create_graphics_pipelines(vk::PipelineCache::null(), &info, None);
    device.destroy_shader_module(vs, None);
    device.destroy_shader_module(fs, None);
    Ok(result.map_err(|(_, error)| error)?[0])
}

/// A device-local depth image with `layers` array layers: an array view of
/// every layer plus one 2D view per layer, for rendering each separately.
///
/// # Safety
/// `device` must be the device `allocator` was created for.
pub unsafe fn create_depth_layers(
    device: &ash::Device,
    allocator: &mut Allocator,
    name: &str,
    format: vk::Format,
    size: u32,
    layers: u32,
) -> Result<(Image, Vec<vk::ImageView>)> {
    let image = device.create_image(
        &vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: size,
                height: size,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED),
        None,
    )?;
    let allocation = match allocator.allocate(&AllocationCreateDesc {
        name,
        requirements: device.get_image_memory_requirements(image),
        location: MemoryLocation::GpuOnly,
        linear: false,
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
    }) {
        Ok(allocation) => allocation,
        Err(error) => {
            device.destroy_image(image, None);
            return Err(error.into());
        }
    };
    if let Err(error) = device.bind_image_memory(image, allocation.memory(), allocation.offset()) {
        device.destroy_image(image, None);
        allocator.free(allocation)?;
        return Err(error.into());
    }
    let view = |kind, first, count| {
        device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(kind)
                .format(format)
                .subresource_range(
                    subresource(vk::ImageAspectFlags::DEPTH)
                        .base_array_layer(first)
                        .layer_count(count),
                ),
            None,
        )
    };
    let mut views = Vec::with_capacity(layers as usize + 1);
    let created = std::iter::once(view(vk::ImageViewType::TYPE_2D_ARRAY, 0, layers))
        .chain((0..layers).map(|layer| view(vk::ImageViewType::TYPE_2D, layer, 1)))
        .try_for_each(|result| result.map(|v| views.push(v)));
    if let Err(error) = created {
        for v in views {
            device.destroy_image_view(v, None);
        }
        device.destroy_image(image, None);
        allocator.free(allocation)?;
        return Err(error.into());
    }
    let array = views.remove(0);
    let layer_views = views;
    Ok((
        Image {
            image,
            view: array,
            allocation,
        },
        layer_views,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vk::PresentModeKHR as P;

    #[test]
    fn present_mode_uncaps_only_when_asked_and_offered() {
        let all = [P::FIFO, P::IMMEDIATE, P::MAILBOX];
        assert_eq!(present_mode(&all, false), P::FIFO);
        assert_eq!(present_mode(&all, true), P::IMMEDIATE);
        assert_eq!(present_mode(&[P::FIFO, P::MAILBOX], true), P::MAILBOX);
        assert_eq!(present_mode(&[P::FIFO], true), P::FIFO);
    }

    #[test]
    fn arena_aligns_every_array_and_refuses_overflow() {
        assert_eq!(bump(0, 12, 64), Some((0, 12)));
        assert_eq!(bump(12, 4, 64), Some((16, 20)));
        assert_eq!(bump(20, 44, 64), None, "48 + 44 > 64");
        assert_eq!(bump(16, 48, 64), Some((16, 64)));
        assert_eq!(bump(u64::MAX - 3, 1, u64::MAX), None);
        // `needed` reserves what `bump` takes for the same arrays.
        assert_eq!(FrameArena::needed([12, 4, 33]), 16 + 16 + 48);
    }

    #[test]
    fn empty_arena_takes_empty_arrays() {
        // A frame with nothing to upload never allocates the arena's buffer.
        let mut arena = FrameArena::default();
        assert_eq!(arena.push::<u32>(&[]).map(|a| a.0), Some(0));
        assert!(arena.push(&[1u32]).is_none());
    }
}
