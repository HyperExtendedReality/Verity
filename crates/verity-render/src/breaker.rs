//! GPU shore-wave surface. The supplied cell-centered data and profile are
//! shared with the gameplay wave model; only time needs changing each frame.

use crate::{failure, gpu, Result, FRAMES, HDR_FORMAT};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2};
use gpu_allocator::{vulkan::Allocator, MemoryLocation};

/// Cell-centered shore-wave data, in row-major order. Positions are in metres,
/// phase is in radians, and `time` is in seconds. The grid covers the rectangle
/// from `origin` to `origin + (width, height) * cell`.
/// Depth is the signed difference between sea level and seabed elevation;
/// nonpositive interpolated depth is dry. Each slice contains exactly
/// `width * height` elements.
pub struct BreakerGrid<'a> {
    pub width: usize,
    pub height: usize,
    pub cell: f32,
    pub origin: Vec2,
    pub sea_level: f32,
    pub omega: f32,
    pub time: f32,
    pub lean: f32,
    /// Strength of the wave-group envelope in [0, 1].
    pub groups: f32,
    /// Group phase along the crests, radians per metre.
    pub across: Vec2,
    pub phase: &'a [f32],
    pub direction: &'a [Vec2],
    /// Unbroken height, before the group envelope, in metres.
    pub wave_height: &'a [f32],
    pub wavenumber: &'a [f32],
    /// Signed sea-level-minus-seabed depth in metres.
    pub depth: &'a [f32],
    /// Tallest wave the depth can hold, in metres.
    pub limit: &'a [f32],
    /// When the group front reaches each cell, in seconds.
    pub arrival: &'a [f32],
}

/// Harmonic coefficients of the shared parametric breaking-wave profile.
#[derive(Clone, Copy, Debug)]
pub struct BreakerProfile {
    pub horizontal_second_harmonic: f32,
    pub vertical_second_harmonic: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Cell {
    wave: [f32; 4],
    medium: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Constants {
    view_proj: [[f32; 4]; 4],
    dimensions: [u32; 4],
    domain: [f32; 4],
    animation: [f32; 4],
    profile: [f32; 4],
}

struct Surface {
    cells: Vec<Cell>,
    dimensions: [u32; 4],
    domain: [f32; 4],
    animation: [f32; 4],
    profile: [f32; 4],
    vertex_count: u32,
}

#[derive(Default)]
struct Slot {
    buffer: gpu::Buffer,
    set: vk::DescriptorSet,
    revision: u64,
}

#[derive(Default)]
pub(crate) struct BreakerPass {
    surface: Option<Surface>,
    revision: u64,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pool: vk::DescriptorPool,
    pipeline: vk::Pipeline,
    slots: Vec<Slot>,
}

impl BreakerPass {
    pub fn set(&mut self, grid: BreakerGrid<'_>, profile: BreakerProfile) -> Result<()> {
        const SUBDIVISIONS: u32 = 4;
        let n = grid
            .width
            .checked_mul(grid.height)
            .ok_or_else(|| failure("breaker grid size overflow"))?;
        if grid.width == 0 || grid.height == 0 || !grid.cell.is_finite() || grid.cell <= 0.0 {
            return Err(failure(
                "breaker grid must have nonzero dimensions and positive cell size",
            ));
        }
        if [
            grid.phase.len(),
            grid.direction.len(),
            grid.wave_height.len(),
            grid.wavenumber.len(),
            grid.depth.len(),
            grid.limit.len(),
            grid.arrival.len(),
        ]
        .iter()
        .any(|&len| len != n)
        {
            return Err(failure("breaker grid arrays must match its dimensions"));
        }
        let width = u32::try_from(grid.width)?;
        let height = u32::try_from(grid.height)?;
        let sx = width
            .checked_mul(SUBDIVISIONS)
            .ok_or_else(|| failure("breaker grid is too wide"))?;
        let sy = height
            .checked_mul(SUBDIVISIONS)
            .ok_or_else(|| failure("breaker grid is too tall"))?;
        let vertex_count = sx
            .checked_mul(sy)
            .and_then(|n| n.checked_mul(6))
            .ok_or_else(|| failure("breaker mesh is too large"))?;
        let mut cells = Vec::with_capacity(n);
        for i in 0..n {
            cells.push(Cell {
                wave: [
                    grid.phase[i],
                    grid.direction[i].x,
                    grid.direction[i].y,
                    grid.wave_height[i],
                ],
                medium: [
                    grid.wavenumber[i],
                    grid.depth[i],
                    grid.limit[i],
                    grid.arrival[i],
                ],
            });
        }
        self.surface = Some(Surface {
            cells,
            dimensions: [width, height, sx, sy],
            domain: [grid.origin.x, grid.origin.y, grid.cell, grid.sea_level],
            animation: [
                grid.omega,
                grid.time,
                grid.lean,
                profile.horizontal_second_harmonic,
            ],
            profile: [
                profile.vertical_second_harmonic,
                grid.groups,
                grid.across.x,
                grid.across.y,
            ],
            vertex_count,
        });
        self.revision = self.revision.wrapping_add(1);
        if self.revision == 0 {
            self.revision = 1;
            for slot in &mut self.slots {
                slot.revision = 0;
            }
        }
        Ok(())
    }

    pub fn set_time(&mut self, time: f32) {
        if let Some(surface) = &mut self.surface {
            surface.animation[1] = time;
        }
    }

    pub fn clear(&mut self) {
        self.surface = None;
    }

    unsafe fn init(&mut self, device: &ash::Device, depth_format: vk::Format) -> Result<()> {
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)];
        self.set_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )?;
        let layouts = [self.set_layout];
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .size(std::mem::size_of::<Constants>() as u32)];
        self.layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&layouts)
                .push_constant_ranges(&push),
            None,
        )?;
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(FRAMES as u32)];
        self.pool = device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(FRAMES as u32)
                .pool_sizes(&sizes),
            None,
        )?;
        let layouts = [self.set_layout; FRAMES];
        let sets = device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.pool)
                .set_layouts(&layouts),
        )?;
        self.slots = sets
            .into_iter()
            .map(|set| Slot {
                set,
                ..Default::default()
            })
            .collect();
        self.pipeline = gpu::graphics_pipeline(
            device,
            self.layout,
            &gpu::Raster {
                vertex: spirv!("breaker.vertexMain"),
                fragment: spirv!("breaker.fragmentMain"),
                color: Some(HDR_FORMAT),
                depth: Some((depth_format, vk::CompareOp::GREATER)),
                cull: vk::CullModeFlags::NONE,
                depth_bias: None,
            },
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn record(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        depth_format: vk::Format,
        slot_index: usize,
        cmd: vk::CommandBuffer,
        viewport: [u32; 4],
        view_proj: Mat4,
        hdr: &gpu::Image,
        depth: &gpu::Image,
    ) -> Result<()> {
        if self.surface.is_none() || viewport[2] == 0 || viewport[3] == 0 {
            return Ok(());
        }
        if self.pipeline == vk::Pipeline::null() {
            if let Err(error) = self.init(device, depth_format) {
                self.destroy(device, allocator);
                return Err(error);
            }
        }
        let surface = self.surface.as_ref().unwrap();
        let slot = &mut self.slots[slot_index];
        if slot.revision != self.revision {
            let bytes = bytemuck::cast_slice(&surface.cells);
            if slot.buffer.size < bytes.len() as u64 {
                let replacement = gpu::create_buffer(
                    device,
                    allocator,
                    "breaker cells",
                    bytes.len() as u64,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    MemoryLocation::CpuToGpu,
                )?;
                gpu::destroy_buffer(
                    device,
                    allocator,
                    std::mem::replace(&mut slot.buffer, replacement),
                );
                let infos = [slot.buffer.info()];
                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(slot.set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&infos)],
                    &[],
                );
            }
            slot.buffer.write(bytes);
            slot.revision = self.revision;
        }
        use vk::AccessFlags2 as A;
        use vk::ImageLayout as L;
        use vk::PipelineStageFlags2 as S;
        let color = vk::ImageAspectFlags::COLOR;
        let depth_aspect = vk::ImageAspectFlags::DEPTH;
        let before = [
            gpu::image_barrier(
                hdr.image,
                color,
                S::COMPUTE_SHADER,
                A::SHADER_STORAGE_WRITE,
                S::COLOR_ATTACHMENT_OUTPUT,
                A::COLOR_ATTACHMENT_READ | A::COLOR_ATTACHMENT_WRITE,
                L::GENERAL,
                L::GENERAL,
            ),
            gpu::image_barrier(
                depth.image,
                depth_aspect,
                S::COMPUTE_SHADER | S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
                A::SHADER_SAMPLED_READ | A::DEPTH_STENCIL_ATTACHMENT_WRITE,
                S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
                A::DEPTH_STENCIL_ATTACHMENT_READ | A::DEPTH_STENCIL_ATTACHMENT_WRITE,
                L::DEPTH_READ_ONLY_OPTIMAL,
                L::DEPTH_ATTACHMENT_OPTIMAL,
            ),
        ];
        device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&before),
        );
        let area = vk::Rect2D {
            offset: vk::Offset2D {
                x: viewport[0] as i32,
                y: viewport[1] as i32,
            },
            extent: vk::Extent2D {
                width: viewport[2],
                height: viewport[3],
            },
        };
        let colors = [vk::RenderingAttachmentInfo::default()
            .image_view(hdr.view)
            .image_layout(L::GENERAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(depth.view)
            .image_layout(L::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE);
        device.cmd_begin_rendering(
            cmd,
            &vk::RenderingInfo::default()
                .render_area(area)
                .layer_count(1)
                .color_attachments(&colors)
                .depth_attachment(&depth_attachment),
        );
        crate::set_viewport(device, cmd, area);
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            self.layout,
            0,
            &[slot.set],
            &[],
        );
        let constants = Constants {
            view_proj: view_proj.to_cols_array_2d(),
            dimensions: surface.dimensions,
            domain: surface.domain,
            animation: surface.animation,
            profile: surface.profile,
        };
        device.cmd_push_constants(
            cmd,
            self.layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            bytemuck::bytes_of(&constants),
        );
        device.cmd_draw(cmd, surface.vertex_count, 1, 0, 0);
        device.cmd_end_rendering(cmd);
        let after = [
            gpu::image_barrier(
                hdr.image,
                color,
                S::COLOR_ATTACHMENT_OUTPUT,
                A::COLOR_ATTACHMENT_WRITE,
                S::FRAGMENT_SHADER,
                A::SHADER_SAMPLED_READ,
                L::GENERAL,
                L::GENERAL,
            ),
            gpu::image_barrier(
                depth.image,
                depth_aspect,
                S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
                A::DEPTH_STENCIL_ATTACHMENT_WRITE,
                S::COMPUTE_SHADER,
                A::SHADER_SAMPLED_READ,
                L::DEPTH_ATTACHMENT_OPTIMAL,
                L::DEPTH_READ_ONLY_OPTIMAL,
            ),
        ];
        device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&after),
        );
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        for slot in self.slots.drain(..) {
            gpu::destroy_buffer(device, allocator, slot.buffer);
        }
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_descriptor_pool(self.pool, None);
        device.destroy_pipeline_layout(self.layout, None);
        device.destroy_descriptor_set_layout(self.set_layout, None);
        self.pipeline = vk::Pipeline::null();
        self.pool = vk::DescriptorPool::null();
        self.layout = vk::PipelineLayout::null();
        self.set_layout = vk::DescriptorSetLayout::null();
    }
}
