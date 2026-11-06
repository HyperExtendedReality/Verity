//! Buffer, image and pipeline helpers over `gpu-allocator`.

use ash::vk;
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
    MemoryLocation,
};
use std::{error::Error, ffi::CStr};

type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;

/// A buffer and its memory. The default value is a null buffer; destroying it is a no-op.
#[derive(Default)]
pub struct Buffer {
    pub buffer: vk::Buffer,
    pub size: u64,
    allocation: Option<Allocation>,
}

impl Buffer {
    /// Copy `bytes` to the start of a host-visible buffer.
    pub fn write(&mut self, bytes: &[u8]) {
        let mapped = self
            .allocation
            .as_mut()
            .and_then(Allocation::mapped_slice_mut)
            .expect("buffer is host visible");
        mapped[..bytes.len()].copy_from_slice(bytes);
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
            .usage(usage)
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
    Ok(Buffer {
        buffer,
        size,
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
    let image = device.create_image(
        &vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent.into())
            .mip_levels(levels)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
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
                    .view_type(vk::ImageViewType::TYPE_2D)
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

/// What a graphics pipeline renders into and how it tests depth.
pub struct Raster<'a> {
    pub vertex: &'a [u8],
    pub fragment: &'a [u8],
    /// `None` for depth-only passes such as shadows.
    pub color: Option<vk::Format>,
    pub depth: Option<(vk::Format, vk::CompareOp)>,
    pub cull: vk::CullModeFlags,
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
    let colors: Vec<vk::Format> = raster.color.into_iter().collect();
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
        .cull_mode(raster.cull)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    if let Some((constant, slope)) = raster.depth_bias {
        rasterization = rasterization
            .depth_bias_enable(true)
            .depth_bias_constant_factor(constant)
            .depth_bias_slope_factor(slope);
    }
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let blend_attachment: Vec<_> = colors
        .iter()
        .map(|_| {
            vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
        })
        .collect();
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachment);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
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
