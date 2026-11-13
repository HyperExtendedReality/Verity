//! Hardware ray tracing: acceleration structures for the passes compiled with
//! `GI_RAY_QUERY`, on devices where [`Gpu::ray_query`] is set. Builds run with
//! the GPU idle, as scene uploads do.

#![allow(unsafe_op_in_unsafe_fn)]

use ash::vk;
use gpu_allocator::MemoryLocation;
use verity_render::{
    Gpu,
    extension::Result,
    gpu::{self, Buffer},
};

/// An acceleration structure and the memory holding it.
pub(crate) struct Accel {
    pub handle: vk::AccelerationStructureKHR,
    pub address: vk::DeviceAddress,
    buffer: Buffer,
}

/// What one acceleration structure is built over; addresses are device addresses.
#[derive(Clone, Copy)]
pub(crate) enum Geometry {
    /// `count` triangles of `u32` index triples into float3 positions `stride`
    /// apart; with `indices` 0, every three positions in a row are one triangle.
    Triangles {
        vertices: vk::DeviceAddress,
        stride: u64,
        max_vertex: u32,
        indices: vk::DeviceAddress,
        count: u32,
        /// Opaque triangles commit without the shader seeing them as candidates.
        opaque: bool,
    },
    /// `count` boxes, six floats each (min then max), for shaders to test.
    Aabbs {
        address: vk::DeviceAddress,
        count: u32,
    },
    /// A TLAS over `count` `vk::AccelerationStructureInstanceKHR`.
    Instances {
        address: vk::DeviceAddress,
        count: u32,
    },
}

impl Geometry {
    fn describe(
        &self,
    ) -> (
        vk::AccelerationStructureGeometryKHR<'static>,
        u32,
        vk::AccelerationStructureTypeKHR,
    ) {
        use vk::GeometryFlagsKHR as F;
        let at = |device_address| vk::DeviceOrHostAddressConstKHR { device_address };
        let geometry = vk::AccelerationStructureGeometryKHR::default();
        match *self {
            Geometry::Triangles {
                vertices,
                stride,
                max_vertex,
                indices,
                count,
                opaque,
            } => (
                geometry
                    .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                    .flags(if opaque {
                        F::OPAQUE
                    } else {
                        F::NO_DUPLICATE_ANY_HIT_INVOCATION
                    })
                    .geometry(vk::AccelerationStructureGeometryDataKHR {
                        triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                            .vertex_format(vk::Format::R32G32B32_SFLOAT)
                            .vertex_data(at(vertices))
                            .vertex_stride(stride)
                            .max_vertex(max_vertex)
                            .index_type(if indices == 0 {
                                vk::IndexType::NONE_KHR
                            } else {
                                vk::IndexType::UINT32
                            })
                            .index_data(at(indices)),
                    }),
                count,
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            ),
            Geometry::Aabbs { address, count } => (
                geometry
                    .geometry_type(vk::GeometryTypeKHR::AABBS)
                    .flags(F::NO_DUPLICATE_ANY_HIT_INVOCATION)
                    .geometry(vk::AccelerationStructureGeometryDataKHR {
                        aabbs: vk::AccelerationStructureGeometryAabbsDataKHR::default()
                            .data(at(address))
                            .stride(24),
                    }),
                count,
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            ),
            Geometry::Instances { address, count } => (
                geometry
                    .geometry_type(vk::GeometryTypeKHR::INSTANCES)
                    .geometry(vk::AccelerationStructureGeometryDataKHR {
                        instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                            .data(at(address)),
                    }),
                count,
                vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            ),
        }
    }
}

/// A TLAS instance: `world_from_local` as three rows, the shader's custom index
/// (24 bits) and the BLAS. Facing culling is left to the shader.
pub(crate) fn instance(
    world_from_local: [f32; 12],
    custom_index: u32,
    blas: vk::DeviceAddress,
) -> vk::AccelerationStructureInstanceKHR {
    let flags = vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8;
    vk::AccelerationStructureInstanceKHR {
        transform: vk::TransformMatrixKHR {
            matrix: world_from_local,
        },
        instance_custom_index_and_mask: vk::Packed24_8::new(custom_index, 0xff),
        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(0, flags),
        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
            device_handle: blas,
        },
    }
}

pub(crate) fn instance_bytes(instances: &[vk::AccelerationStructureInstanceKHR]) -> &[u8] {
    // SAFETY: a plain repr(C) struct of integers and floats, with no padding.
    unsafe {
        std::slice::from_raw_parts(instances.as_ptr().cast(), size_of_val(instances))
    }
}

pub(crate) struct RayTracing {
    loader: ash::khr::acceleration_structure::Device,
    scratch_alignment: u64,
}

impl RayTracing {
    pub unsafe fn new(gpu: &Gpu) -> Self {
        let mut accel = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut accel);
        gpu.instance
            .get_physical_device_properties2(gpu.physical, &mut properties);
        Self {
            loader: ash::khr::acceleration_structure::Device::new(gpu.instance, gpu.device),
            scratch_alignment: u64::from(accel.min_acceleration_structure_scratch_offset_alignment)
                .max(1),
        }
    }

    /// A host-visible buffer of build input, and its address.
    pub unsafe fn input(
        &self,
        gpu: &mut Gpu,
        name: &str,
        bytes: &[u8],
    ) -> Result<(Buffer, vk::DeviceAddress)> {
        let mut buffer = gpu::create_buffer(
            gpu.device,
            gpu.allocator,
            name,
            bytes.len() as u64,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::CpuToGpu,
        )?;
        buffer.write(bytes);
        let address = address(gpu.device, &buffer);
        Ok((buffer, address))
    }

    /// Build one acceleration structure per geometry and wait for them.
    pub unsafe fn build(
        &self,
        gpu: &mut Gpu,
        pool: &mut vk::CommandPool,
        geometries: &[Geometry],
    ) -> Result<Vec<Accel>> {
        let described: Vec<_> = geometries.iter().map(Geometry::describe).collect();
        let mut built = Vec::with_capacity(described.len());
        let mut scratch = Vec::with_capacity(described.len());
        let result = self.record(gpu, pool, &described, &mut built, &mut scratch);
        for buffer in scratch {
            gpu::destroy_buffer(gpu.device, gpu.allocator, buffer);
        }
        match result {
            Ok(()) => Ok(built),
            Err(error) => {
                for accel in built {
                    self.destroy(gpu, accel);
                }
                Err(error)
            }
        }
    }

    unsafe fn record(
        &self,
        gpu: &mut Gpu,
        pool: &mut vk::CommandPool,
        described: &[(
            vk::AccelerationStructureGeometryKHR<'static>,
            u32,
            vk::AccelerationStructureTypeKHR,
        )],
        built: &mut Vec<Accel>,
        scratch: &mut Vec<Buffer>,
    ) -> Result<()> {
        let mut infos = Vec::with_capacity(described.len());
        for (geometry, count, ty) in described {
            let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(*ty)
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(std::slice::from_ref(geometry));
            let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
            self.loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &info,
                &[*count],
                &mut sizes,
            );
            let accel = self.create(gpu, *ty, sizes.acceleration_structure_size)?;
            let handle = accel.handle;
            built.push(accel);
            let buffer = gpu::create_buffer(
                gpu.device,
                gpu.allocator,
                "gi accel scratch",
                sizes.build_scratch_size + self.scratch_alignment,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryLocation::GpuOnly,
            )?;
            let device_address = address(gpu.device, &buffer).next_multiple_of(self.scratch_alignment);
            scratch.push(buffer);
            infos.push(
                info.dst_acceleration_structure(handle)
                    .scratch_data(vk::DeviceOrHostAddressKHR { device_address }),
            );
        }
        let ranges: Vec<[vk::AccelerationStructureBuildRangeInfoKHR; 1]> = (described.iter())
            .map(|(_, count, _)| {
                [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(*count)]
            })
            .collect();
        let ranges: Vec<&[_]> = ranges.iter().map(|r| r.as_slice()).collect();
        submit(gpu, pool, |cmd| {
            self.loader
                .cmd_build_acceleration_structures(cmd, &infos, &ranges);
            let barrier = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ)];
            gpu.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&barrier),
            );
        })
    }

    /// A TLAS with room for `capacity` instances, and scratch to rebuild it in
    /// a frame's commands with [`Self::record_tlas`].
    pub unsafe fn tlas_target(&self, gpu: &mut Gpu, capacity: u32) -> Result<(Accel, Buffer)> {
        let (geometry, _, ty) = Geometry::Instances {
            address: 0,
            count: capacity,
        }
        .describe();
        let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(ty)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(std::slice::from_ref(&geometry));
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        self.loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &info,
            &[capacity],
            &mut sizes,
        );
        let accel = self.create(gpu, ty, sizes.acceleration_structure_size)?;
        let scratch = gpu::create_buffer(
            gpu.device,
            gpu.allocator,
            "gi tlas scratch",
            sizes.build_scratch_size + self.scratch_alignment,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::GpuOnly,
        );
        match scratch {
            Ok(scratch) => Ok((accel, scratch)),
            Err(error) => {
                self.destroy(gpu, accel);
                Err(error)
            }
        }
    }

    /// Rebuild `tlas` over `count` instances at `instances`, in `cmd`, and make
    /// it visible to compute shaders after.
    pub unsafe fn record_tlas(
        &self,
        gpu: &Gpu,
        cmd: vk::CommandBuffer,
        tlas: &Accel,
        scratch: &Buffer,
        instances: vk::DeviceAddress,
        count: u32,
    ) {
        let (geometry, _, ty) = Geometry::Instances {
            address: instances,
            count,
        }
        .describe();
        let scratch = address(gpu.device, scratch).next_multiple_of(self.scratch_alignment);
        let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(ty)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(&geometry))
            .dst_acceleration_structure(tlas.handle)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch,
            });
        let range = [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(count)];
        let before = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
            .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)];
        gpu.device
            .cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&before));
        self.loader
            .cmd_build_acceleration_structures(cmd, &[info], &[&range]);
        let after = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
            .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR)];
        gpu.device
            .cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&after));
    }

    unsafe fn create(
        &self,
        gpu: &mut Gpu,
        ty: vk::AccelerationStructureTypeKHR,
        size: u64,
    ) -> Result<Accel> {
        let buffer = gpu::create_buffer(
            gpu.device,
            gpu.allocator,
            "gi accel",
            size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::GpuOnly,
        )?;
        let handle = self.loader.create_acceleration_structure(
            &vk::AccelerationStructureCreateInfoKHR::default()
                .buffer(buffer.buffer)
                .size(size)
                .ty(ty),
            None,
        );
        let handle = match handle {
            Ok(handle) => handle,
            Err(error) => {
                gpu::destroy_buffer(gpu.device, gpu.allocator, buffer);
                return Err(error.into());
            }
        };
        let address = self.loader.get_acceleration_structure_device_address(
            &vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(handle),
        );
        Ok(Accel {
            handle,
            address,
            buffer,
        })
    }

    /// # Safety
    /// The GPU must be done with it.
    pub unsafe fn destroy(&self, gpu: &mut Gpu, accel: Accel) {
        self.loader
            .destroy_acceleration_structure(accel.handle, None);
        gpu::destroy_buffer(gpu.device, gpu.allocator, accel.buffer);
    }
}

pub(crate) unsafe fn address(device: &ash::Device, buffer: &Buffer) -> vk::DeviceAddress {
    device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer.buffer))
}

/// Record one command buffer, submit it and wait.
unsafe fn submit(
    gpu: &Gpu,
    pool: &mut vk::CommandPool,
    record: impl FnOnce(vk::CommandBuffer),
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
    let cmd = device.allocate_command_buffers(
        &vk::CommandBufferAllocateInfo::default()
            .command_pool(*pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1),
    )?[0];
    let commands = [cmd];
    let result = device
        .begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
        .and_then(|()| {
            record(cmd);
            device.end_command_buffer(cmd)
        })
        .and_then(|()| {
            device.queue_submit(
                gpu.queue,
                &[vk::SubmitInfo::default().command_buffers(&commands)],
                vk::Fence::null(),
            )
        })
        .and_then(|()| device.queue_wait_idle(gpu.queue));
    device.free_command_buffers(*pool, &commands);
    Ok(result?)
}
