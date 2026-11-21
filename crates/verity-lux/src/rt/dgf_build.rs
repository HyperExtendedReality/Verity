//! Device builds of bottom-level acceleration structures from exact DGF1 blocks.

use std::{ffi::c_void, ptr, slice};

use ash::vk;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
};
use verity_render::{Gpu, extension::Result, gpu};

use super::{
    Accel, RayTracing, address,
    dgf::{self, DgfMesh},
    submit,
};

const DGF1_ALIGNMENT: u64 = 128;
const RANGE_BYTES: u32 = size_of::<vk::AccelerationStructureBuildRangeInfoKHR>() as u32;
const COMPRESSED_DATA_DGF1: vk::BufferUsageFlags2KHR = vk::BufferUsageFlags2KHR::from_raw(1 << 33);
const DENSE_GEOMETRY_TRIANGLES: vk::GeometryTypeKHR = vk::GeometryTypeKHR::from_raw(1_000_478_000);
const DENSE_GEOMETRY_TRIANGLES_DATA: vk::StructureType = vk::StructureType::from_raw(1_000_478_001);
const DGF1_FORMAT: i32 = 0;

// VkAccelerationStructureDenseGeometryFormatTrianglesDataAMDX from Vulkan SDK
// 1.4.363 vulkan_beta.h, which is newer than the ash bindings used here.
#[repr(C)]
struct DenseGeometryTrianglesData {
    s_type: vk::StructureType,
    p_next: *const c_void,
    compressed_data: vk::DeviceOrHostAddressConstKHR,
    data_size: vk::DeviceSize,
    num_triangles: u32,
    num_vertices: u32,
    max_primitive_index: u32,
    max_geometry_index: u32,
    format: i32,
}

impl RayTracing {
    /// Build an exact, non-opaque BLAS with mesh-local primitive indices.
    ///
    /// `Ok(None)` selects the conventional path if DGF is unavailable or the
    /// codec cannot encode the mesh exactly. The build completes before return.
    ///
    /// # Errors
    /// Propagates Vulkan and GPU allocation errors without attempting fallback.
    ///
    /// # Safety
    /// `self` and any non-null `pool` must belong to `gpu`. The pool must use
    /// `gpu.queue_family`, and the pool and queue must be externally synchronized.
    pub(crate) unsafe fn try_build_dgf(
        &self,
        gpu: &mut Gpu,
        pool: &mut vk::CommandPool,
        positions: &[[f32; 3]],
        indices: &[u32],
    ) -> Result<Option<Accel>> {
        if !gpu.ray_features.dense_geometry {
            return Ok(None);
        }
        let Ok(mesh) = dgf::encode_exact(positions, indices) else {
            return Ok(None);
        };
        let bytes = mesh.blocks.as_flattened();
        let data_size = bytes.len() as u64;
        // SAFETY: The capability enables DGF and buffer usage flags 2. The
        // caller supplies the owning device and exclusive use of the pool/queue.
        unsafe {
            let input = DgfInput::upload(gpu, bytes, mesh.num_triangles)?;
            let result = self.build_dgf_blocks(gpu, pool, &mesh, input.address, data_size);
            input.destroy(gpu);
            result.map(Some)
        }
    }

    /// # Safety
    /// `device_address` must be a live, aligned DGF1 input address for `mesh`
    /// on the same device as this loader, and remain valid until submit returns.
    unsafe fn build_dgf_blocks(
        &self,
        gpu: &mut Gpu,
        pool: &mut vk::CommandPool,
        mesh: &DgfMesh,
        device_address: vk::DeviceAddress,
        data_size: vk::DeviceSize,
    ) -> Result<Accel> {
        // The codec assigns geometry index 0, non-opaque triangles and the
        // original primitive indices in the blocks, not in the geometry flags.
        let dense = DenseGeometryTrianglesData {
            s_type: DENSE_GEOMETRY_TRIANGLES_DATA,
            p_next: ptr::null(),
            compressed_data: vk::DeviceOrHostAddressConstKHR { device_address },
            data_size,
            num_triangles: mesh.num_triangles,
            num_vertices: mesh.num_vertices,
            max_primitive_index: mesh.max_primitive_index,
            max_geometry_index: 0,
            format: DGF1_FORMAT,
        };
        let mut geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(DENSE_GEOMETRY_TRIANGLES)
            .flags(vk::GeometryFlagsKHR::NO_DUPLICATE_ANY_HIT_INVOCATION);
        geometry.p_next = ptr::from_ref(&dense).cast();
        let ty = vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL;
        let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(ty)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(slice::from_ref(&geometry));
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        let device = gpu.device;

        // SAFETY: The descriptor chain lives through both calls. All GPU
        // resources are from the same device and survive the synchronous build.
        unsafe {
            // Size queries still require one max primitive count per geometry,
            // in addition to the compressed metadata in the pNext chain.
            self.loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &info,
                &[mesh.num_triangles],
                &mut sizes,
            );
            let scratch = if sizes.build_scratch_size == 0 {
                None
            } else {
                Some(gpu::create_buffer(
                    device,
                    gpu.allocator,
                    "gi dgf scratch",
                    sizes.build_scratch_size + (self.scratch_alignment - 1),
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    MemoryLocation::GpuOnly,
                )?)
            };
            let scratch_address = scratch.as_ref().map_or(0, |buffer| {
                buffer.address.next_multiple_of(self.scratch_alignment)
            });
            let result = self
                .create(gpu, ty, sizes.acceleration_structure_size)
                .and_then(|accel| {
                    let info = info.dst_acceleration_structure(accel.handle).scratch_data(
                        vk::DeviceOrHostAddressKHR {
                            device_address: scratch_address,
                        },
                    );
                    let result = submit(gpu, pool, |cmd| {
                        // The current direct command requires a null range for DGF.
                        // Early AMD drivers dereference it. The indirect command
                        // uses a device-side range and the same compressed metadata.
                        self.loader.cmd_build_acceleration_structures_indirect(
                            cmd,
                            &[info],
                            &[device_address + data_size],
                            &[RANGE_BYTES],
                            &[&[mesh.num_triangles]],
                        );
                        gpu::memory_barrier(
                            device,
                            cmd,
                            (
                                vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                                vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
                            ),
                            (
                                vk::PipelineStageFlags2::ALL_COMMANDS,
                                vk::AccessFlags2::MEMORY_READ,
                            ),
                        );
                    });
                    match result {
                        Ok(()) => Ok(accel),
                        Err(error) => {
                            self.destroy(gpu, accel);
                            Err(error)
                        }
                    }
                });
            if let Some(scratch) = scratch {
                gpu::destroy_buffer(device, gpu.allocator, scratch);
            }
            result
        }
    }
}

// The renderer's Buffer helper only takes 32-bit usage flags. Keep the
// usage-flags-2 allocation local to this input, without a staging copy.
struct DgfInput {
    buffer: vk::Buffer,
    allocation: Allocation,
    address: vk::DeviceAddress,
}

impl DgfInput {
    /// # Safety
    /// The device must have DGF and its usage-flags-2 dependency enabled, and
    /// `bytes` must contain a nonempty, complete sequence of DGF1 blocks.
    unsafe fn upload(gpu: &mut Gpu, bytes: &[u8], triangles: u32) -> Result<Self> {
        let device = gpu.device;
        let mut usage = vk::BufferUsageFlags2CreateInfoKHR::default().usage(
            COMPRESSED_DATA_DGF1
                | vk::BufferUsageFlags2KHR::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
                | vk::BufferUsageFlags2KHR::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags2KHR::INDIRECT_BUFFER,
        );
        // SAFETY: The enabled extension permits these usage flags. The
        // allocator was created for this device, and returns coherent mapped
        // memory for CpuToGpu allocations. The extra bytes cover any alignment gap.
        unsafe {
            let buffer = device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(bytes.len() as u64 + u64::from(RANGE_BYTES) + (DGF1_ALIGNMENT - 1))
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .push_next(&mut usage),
                None,
            )?;
            let allocation = gpu.allocator.allocate(&AllocationCreateDesc {
                name: "gi dgf input",
                requirements: device.get_buffer_memory_requirements(buffer),
                location: MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            });
            let mut allocation = match allocation {
                Ok(allocation) => allocation,
                Err(error) => {
                    device.destroy_buffer(buffer, None);
                    return Err(error.into());
                }
            };
            if let Err(error) =
                device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
            {
                device.destroy_buffer(buffer, None);
                let _ = gpu.allocator.free(allocation);
                return Err(error.into());
            }
            // Query the bound handle without transferring ownership to the
            // renderer's 32-bit-usage buffer helper.
            let mut view = gpu::Buffer::default();
            view.buffer = buffer;
            let base = address(device, &view);
            let address = base.next_multiple_of(DGF1_ALIGNMENT);
            let offset = (address - base) as usize;
            let mapped = allocation
                .mapped_slice_mut()
                .expect("DGF input uses host-visible memory");
            let range = offset + bytes.len();
            mapped[offset..range].copy_from_slice(bytes);
            // VkAccelerationStructureBuildRangeInfoKHR: count, then three zero offsets.
            mapped[range..range + RANGE_BYTES as usize]
                .copy_from_slice(bytemuck::bytes_of(&[triangles, 0, 0, 0]));
            Ok(Self {
                buffer,
                allocation,
                address,
            })
        }
    }

    /// # Safety
    /// The owning device must have finished reading this input.
    unsafe fn destroy(self, gpu: &mut Gpu) {
        // SAFETY: The caller waited for the build, and this owns the allocation.
        unsafe { gpu.device.destroy_buffer(self.buffer, None) };
        let _ = gpu.allocator.free(self.allocation);
    }
}
