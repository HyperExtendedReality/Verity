//! Build a cluster BLAS directly from indexed triangle geometry.

use std::{
    collections::{HashMap, hash_map::Entry},
    ffi::c_void,
    mem, ptr,
};

use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;
use verity_render::{
    Gpu,
    extension::Result,
    gpu::{self, Buffer},
};

use super::submit;

const PROPERTIES_NV: vk::StructureType = vk::StructureType::from_raw(1_000_569_001);
const BOTTOM_LEVEL_INPUT_NV: vk::StructureType = vk::StructureType::from_raw(1_000_569_002);
const TRIANGLE_INPUT_NV: vk::StructureType = vk::StructureType::from_raw(1_000_569_003);
const INPUT_INFO_NV: vk::StructureType = vk::StructureType::from_raw(1_000_569_005);
const COMMANDS_INFO_NV: vk::StructureType = vk::StructureType::from_raw(1_000_569_006);

const VERTEX_FORMAT: vk::Format = vk::Format::R32G32B32_SFLOAT;
const INPUT_USAGE: vk::BufferUsageFlags =
    vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;
const STORAGE_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::STORAGE_BUFFER;
const ADDRESS_SIZE: u64 = size_of::<vk::DeviceAddress>() as u64;
const MAX_PACKED_COUNT: u32 = (1 << 9) - 1;
const MAX_PACKED_GEOMETRY_INDEX: u32 = (1 << 24) - 1;
const INDEX_FORMAT_16BIT_NV: u32 = 0x2;
const NO_DUPLICATE_ANYHIT_NV: u32 = 0x2;

// VK_NV_cluster_acceleration_structure is newer than ash 0.38. The private
// definitions below match the Vulkan 1.4 SDK's vulkan_core.h (extension rev 4).
#[repr(C)]
#[derive(Default)]
struct ClusterProperties {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    max_vertices_per_cluster: u32,
    max_triangles_per_cluster: u32,
    cluster_scratch_byte_alignment: u32,
    cluster_byte_alignment: u32,
    cluster_template_byte_alignment: u32,
    cluster_bottom_level_byte_alignment: u32,
    cluster_template_bounds_byte_alignment: u32,
    max_cluster_geometry_index: u32,
}

#[repr(C)]
struct TriangleInput {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    vertex_format: vk::Format,
    max_geometry_index_value: u32,
    max_cluster_unique_geometry_count: u32,
    max_cluster_triangle_count: u32,
    max_cluster_vertex_count: u32,
    max_total_triangle_count: u32,
    max_total_vertex_count: u32,
    min_position_truncate_bit_count: u32,
}

#[repr(C)]
struct BottomLevelInput {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    max_total_cluster_count: u32,
    max_cluster_count_per_acceleration_structure: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
union OpInput {
    clusters_bottom_level: *mut BottomLevelInput,
    triangle_clusters: *mut TriangleInput,
}

#[repr(i32)]
#[derive(Clone, Copy)]
enum OpType {
    BuildBottomLevel = 1,
    BuildTriangleCluster = 2,
}

#[repr(i32)]
#[derive(Clone, Copy)]
enum OpMode {
    ImplicitDestinations = 0,
    ExplicitDestinations = 1,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct InputInfo {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    max_acceleration_structure_count: u32,
    flags: vk::BuildAccelerationStructureFlagsKHR,
    op_type: OpType,
    op_mode: OpMode,
    op_input: OpInput,
}

#[repr(C)]
struct CommandsInfo {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    input: InputInfo,
    dst_implicit_data: vk::DeviceAddress,
    scratch_data: vk::DeviceAddress,
    dst_addresses_array: vk::StridedDeviceAddressRegionKHR,
    dst_sizes_array: vk::StridedDeviceAddressRegionKHR,
    src_infos_array: vk::StridedDeviceAddressRegionKHR,
    src_infos_count: vk::DeviceAddress,
    address_resolution_flags: vk::Flags,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TriangleRecord {
    cluster_id: u32,
    cluster_flags: vk::Flags,
    counts_and_formats: u32,
    geometry_index_and_flags: u32,
    index_buffer_stride: u16,
    vertex_buffer_stride: u16,
    geometry_index_and_flags_buffer_stride: u16,
    opacity_micromap_index_buffer_stride: u16,
    index_buffer: vk::DeviceAddress,
    vertex_buffer: vk::DeviceAddress,
    geometry_index_and_flags_buffer: vk::DeviceAddress,
    opacity_micromap_array: vk::DeviceAddress,
    opacity_micromap_index_buffer: vk::DeviceAddress,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BottomLevelRecord {
    cluster_references_count: u32,
    cluster_references_stride: u32,
    cluster_references: vk::DeviceAddress,
}

#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(size_of::<ClusterProperties>() == 48);
    assert!(size_of::<TriangleInput>() == 48);
    assert!(size_of::<BottomLevelInput>() == 24);
    assert!(size_of::<InputInfo>() == 40);
    assert!(size_of::<CommandsInfo>() == 160);
    assert!(size_of::<TriangleRecord>() == 64);
    assert!(size_of::<BottomLevelRecord>() == 16);
};

type GetBuildSizes = unsafe extern "system" fn(
    vk::Device,
    *const InputInfo,
    *mut vk::AccelerationStructureBuildSizesInfoKHR<'_>,
);
type CmdBuild = unsafe extern "system" fn(vk::CommandBuffer, *const CommandsInfo);

struct Dispatch {
    get_build_sizes: GetBuildSizes,
    cmd_build: CmdBuild,
}

impl Dispatch {
    fn load(gpu: &Gpu) -> Option<Self> {
        // SAFETY: The logical device belongs to this instance. Vulkan returns
        // the exact command signature from the SDK for each non-null entry.
        unsafe {
            let get = gpu.instance.get_device_proc_addr(
                gpu.device.handle(),
                c"vkGetClusterAccelerationStructureBuildSizesNV".as_ptr(),
            )?;
            let cmd = gpu.instance.get_device_proc_addr(
                gpu.device.handle(),
                c"vkCmdBuildClusterAccelerationStructureIndirectNV".as_ptr(),
            )?;
            Some(Self {
                get_build_sizes: mem::transmute::<_, GetBuildSizes>(get),
                cmd_build: mem::transmute::<_, CmdBuild>(cmd),
            })
        }
    }

    fn sizes(&self, gpu: &Gpu, input: &InputInfo) -> vk::AccelerationStructureBuildSizesInfoKHR<'static> {
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        // SAFETY: The cluster feature is enabled and input's op-specific
        // structure is live for the duration of this call.
        unsafe { (self.get_build_sizes)(gpu.device.handle(), input, &mut sizes) };
        sizes
    }
}

struct Cluster {
    first_triangle: u32,
    triangle_count: u32,
    first_vertex: usize,
    vertex_count: u32,
}

struct ClusterMesh {
    vertices: Vec<[f32; 3]>,
    indices: Vec<[u16; 3]>,
    clusters: Vec<Cluster>,
    max_triangles: u32,
    max_vertices: u32,
}

impl ClusterMesh {
    fn compact(
        positions: &[[f32; 3]],
        indices: &[[u32; 3]],
        limits: &ClusterProperties,
    ) -> Result<Self> {
        let triangle_limit = limits.max_triangles_per_cluster.min(MAX_PACKED_COUNT) as usize;
        let vertex_limit = limits.max_vertices_per_cluster.min(MAX_PACKED_COUNT) as usize;
        if indices.is_empty() || triangle_limit == 0 || vertex_limit < 3 {
            return Err("no triangles or invalid Vulkan triangle cluster limits".into());
        }
        let _ = u32::try_from(indices.len())?;
        let mut mesh = Self {
            vertices: Vec::with_capacity(positions.len().min(indices.len().saturating_mul(3))),
            indices: Vec::with_capacity(indices.len()),
            clusters: Vec::with_capacity(indices.len().div_ceil(triangle_limit)),
            max_triangles: 0,
            max_vertices: 0,
        };
        let mut local = HashMap::with_capacity(vertex_limit.min(positions.len()));
        while mesh.indices.len() < indices.len() {
            let first = mesh.indices.len();
            let first_triangle = first as u32;
            if first_triangle >= limits.max_cluster_geometry_index
                || first_triangle > MAX_PACKED_GEOMETRY_INDEX
            {
                return Err("triangle cluster geometry index exceeds the device or ABI limit".into());
            }
            let first_vertex = mesh.vertices.len();
            local.clear();
            for triangle in indices[first..].iter().take(triangle_limit) {
                let available = vertex_limit - local.len();
                if available < 3 {
                    let new_vertices = triangle
                        .iter()
                        .enumerate()
                        .filter(|&(corner, source)| {
                            !local.contains_key(source) && !triangle[..corner].contains(source)
                        })
                        .count();
                    if new_vertices > available {
                        break;
                    }
                }
                let mut remapped = [0; 3];
                for (&source, dst) in triangle.iter().zip(&mut remapped) {
                    let next = local.len() as u16;
                    *dst = match local.entry(source) {
                        Entry::Occupied(entry) => *entry.get(),
                        Entry::Vacant(entry) => {
                            let vertex = positions.get(source as usize).ok_or_else(|| {
                                format!(
                                    "triangle {} references missing vertex {source}",
                                    mesh.indices.len()
                                )
                            })?;
                            mesh.vertices.push(*vertex);
                            *entry.insert(next)
                        }
                    };
                }
                mesh.indices.push(remapped);
            }
            let triangle_count = (mesh.indices.len() - first) as u32;
            let vertex_count = local.len() as u32;
            mesh.max_triangles = mesh.max_triangles.max(triangle_count);
            mesh.max_vertices = mesh.max_vertices.max(vertex_count);
            mesh.clusters.push(Cluster {
                first_triangle,
                triangle_count,
                first_vertex,
                vertex_count,
            });
        }
        Ok(mesh)
    }
}

impl Cluster {
    fn record(
        &self,
        id: u32,
        vertices: vk::DeviceAddress,
        indices: vk::DeviceAddress,
    ) -> TriangleRecord {
        TriangleRecord {
            cluster_id: id,
            // The SDK specifies these as packed fields (9/9/6/4/4 bits).
            // Use no vertex quantization and no opacity micromap array.
            counts_and_formats: self.triangle_count
                | (self.vertex_count << 9)
                | (INDEX_FORMAT_16BIT_NV << 24)
                | (INDEX_FORMAT_16BIT_NV << 28),
            // PrimitiveIndex is local to the cluster. GeometryIndex carries
            // the original first triangle, so their sum indexes the source mesh.
            geometry_index_and_flags: self.first_triangle | (NO_DUPLICATE_ANYHIT_NV << 29),
            index_buffer_stride: size_of::<u16>() as u16,
            vertex_buffer_stride: size_of::<[f32; 3]>() as u16,
            index_buffer: indices + u64::from(self.first_triangle) * size_of::<[u16; 3]>() as u64,
            vertex_buffer: vertices + self.first_vertex as u64 * size_of::<[f32; 3]>() as u64,
            ..TriangleRecord::zeroed()
        }
    }
}

fn align_up(size: u64, alignment: u64) -> Result<u64> {
    if !alignment.is_power_of_two() {
        return Err("invalid Vulkan triangle cluster alignment".into());
    }
    let size = size
        .checked_add(alignment - 1)
        .ok_or("triangle cluster address or size overflow")?;
    Ok(size & !(alignment - 1))
}

fn allocate(
    gpu: &mut Gpu,
    slot: &mut Buffer,
    name: &str,
    size: u64,
    alignment: u64,
    max_buffer_size: u64,
    usage: vk::BufferUsageFlags,
    location: MemoryLocation,
) -> Result<vk::DeviceAddress> {
    if !alignment.is_power_of_two() {
        return Err("invalid Vulkan triangle cluster alignment".into());
    }
    let capacity = size.checked_add(alignment - 1).ok_or("triangle cluster buffer size overflow")?;
    if capacity.max(16) > max_buffer_size {
        return Err(format!("{name} exceeds the Vulkan maximum buffer size").into());
    }
    // SAFETY: The renderer lends out an allocator for this device. The caller
    // holds the allocation through command completion, even when this returns an error.
    *slot = unsafe { gpu::create_buffer(gpu.device, gpu.allocator, name, capacity, usage, location) }?;
    if slot.address == 0 {
        return Err(format!("{name} has no Vulkan device address").into());
    }
    let address = align_up(slot.address, alignment)?;
    address.checked_add(size).ok_or("triangle cluster buffer address overflow")?;
    Ok(address)
}

fn region(address: vk::DeviceAddress, stride: u64, size: u64) -> vk::StridedDeviceAddressRegionKHR {
    vk::StridedDeviceAddressRegionKHR {
        device_address: address,
        stride,
        size,
    }
}

/// GPU storage for both the constituent clusters and the BLAS that references them.
pub(super) struct ClusterBlas {
    storage: Buffer,
    address: vk::DeviceAddress,
}

impl ClusterBlas {
    /// Build non-opaque triangles in mesh order. For a ray hit, the sum of the
    /// geometry index and primitive index is the original mesh triangle index.
    /// The renderer's build pool must belong to the GPU's queue family.
    pub(super) fn build(
        gpu: &mut Gpu,
        pool: &mut vk::CommandPool,
        positions: &[[f32; 3]],
        indices: &[[u32; 3]],
    ) -> Result<Self> {
        if !gpu.ray_features.cluster_acceleration {
            return Err("Vulkan triangle cluster acceleration is not enabled on this device".into());
        }
        let dispatch =
            Dispatch::load(gpu).ok_or("Vulkan triangle cluster commands are unavailable")?;
        let mut allocation_limits = vk::PhysicalDeviceMaintenance3Properties::default();
        let mut buffer_limits = vk::PhysicalDeviceMaintenance4Properties {
            p_next: (&raw mut allocation_limits).cast(),
            ..Default::default()
        };
        let mut limits = ClusterProperties {
            s_type: PROPERTIES_NV,
            p_next: (&raw mut buffer_limits).cast(),
            ..Default::default()
        };
        let mut properties = vk::PhysicalDeviceProperties2::default();
        properties.p_next = (&raw mut limits).cast();
        // SAFETY: The output chain matches the structure tags, and the
        // renderer's physical device belongs to this instance.
        let format = unsafe {
            gpu.instance.get_physical_device_properties2(gpu.physical, &mut properties);
            gpu.instance.get_physical_device_format_properties(gpu.physical, VERTEX_FORMAT)
        };
        if !format
            .buffer_features
            .contains(vk::FormatFeatureFlags::ACCELERATION_STRUCTURE_VERTEX_BUFFER_KHR)
        {
            return Err("Vulkan triangle clusters require R32G32B32_SFLOAT build vertices".into());
        }
        let mesh = ClusterMesh::compact(positions, indices, &limits)?;
        let cluster_count = mesh.clusters.len() as u32;
        let mut triangles = TriangleInput {
            s_type: TRIANGLE_INPUT_NV,
            p_next: ptr::null_mut(),
            vertex_format: VERTEX_FORMAT,
            max_geometry_index_value: mesh.clusters[mesh.clusters.len() - 1].first_triangle,
            max_cluster_unique_geometry_count: 1,
            max_cluster_triangle_count: mesh.max_triangles,
            max_cluster_vertex_count: mesh.max_vertices,
            max_total_triangle_count: mesh.indices.len() as u32,
            max_total_vertex_count: u32::try_from(mesh.vertices.len())?,
            min_position_truncate_bit_count: 0,
        };
        let mut bottom_level = BottomLevelInput {
            s_type: BOTTOM_LEVEL_INPUT_NV,
            p_next: ptr::null_mut(),
            max_total_cluster_count: cluster_count,
            max_cluster_count_per_acceleration_structure: cluster_count,
        };
        let triangles_input = InputInfo {
            s_type: INPUT_INFO_NV,
            p_next: ptr::null_mut(),
            max_acceleration_structure_count: cluster_count,
            flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
            op_type: OpType::BuildTriangleCluster,
            op_mode: OpMode::ImplicitDestinations,
            op_input: OpInput { triangle_clusters: &raw mut triangles },
        };
        let blas_input = InputInfo {
            max_acceleration_structure_count: 1,
            op_type: OpType::BuildBottomLevel,
            op_mode: OpMode::ExplicitDestinations,
            op_input: OpInput { clusters_bottom_level: &raw mut bottom_level },
            ..triangles_input
        };
        let clas_sizes = dispatch.sizes(gpu, &triangles_input);
        let blas_sizes = dispatch.sizes(gpu, &blas_input);
        if clas_sizes.acceleration_structure_size == 0 || blas_sizes.acceleration_structure_size == 0
        {
            return Err("Vulkan returned an empty triangle cluster or cluster BLAS size".into());
        }

        let cluster_alignment = u64::from(limits.cluster_byte_alignment);
        if !cluster_alignment.is_power_of_two() {
            return Err("invalid Vulkan triangle cluster output alignment".into());
        }
        let blas_alignment = u64::from(limits.cluster_bottom_level_byte_alignment);
        let blas_offset = align_up(clas_sizes.acceleration_structure_size, blas_alignment)?;
        let storage_size = blas_offset
            .checked_add(blas_sizes.acceleration_structure_size)
            .ok_or("triangle cluster acceleration structure size overflow")?;
        let storage_alignment = blas_alignment.max(cluster_alignment);
        let vertex_bytes = bytemuck::cast_slice(&mesh.vertices);
        let index_bytes = bytemuck::cast_slice(&mesh.indices);
        let indices_offset = vertex_bytes.len() as u64;
        let triangle_records_offset = align_up(indices_offset + index_bytes.len() as u64, ADDRESS_SIZE)?;
        let triangle_records_size = u64::from(cluster_count) * size_of::<TriangleRecord>() as u64;
        let blas_record_offset = triangle_records_offset + triangle_records_size;
        let blas_address_offset = blas_record_offset + size_of::<BottomLevelRecord>() as u64;
        let upload_size = blas_address_offset + ADDRESS_SIZE;
        let references_size = u64::from(cluster_count) * ADDRESS_SIZE;
        let max_buffer_size = buffer_limits
            .max_buffer_size
            .min(allocation_limits.max_memory_allocation_size);

        let mut storage = Buffer::default();
        let mut scratch = Buffer::default();
        let mut references = Buffer::default();
        let mut upload = Buffer::default();
        let result = (|| {
            let clusters_address = allocate(
                gpu,
                &mut storage,
                "GI cluster BLAS storage",
                storage_size,
                storage_alignment,
                max_buffer_size,
                STORAGE_USAGE | INPUT_USAGE | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
                MemoryLocation::GpuOnly,
            )?;
            let blas_address = clusters_address + blas_offset;
            let scratch_address = allocate(
                gpu,
                &mut scratch,
                "GI triangle cluster scratch",
                clas_sizes.build_scratch_size.max(blas_sizes.build_scratch_size).max(1),
                u64::from(limits.cluster_scratch_byte_alignment),
                max_buffer_size,
                STORAGE_USAGE,
                MemoryLocation::GpuOnly,
            )?;
            let references_address = allocate(
                gpu,
                &mut references,
                "GI triangle cluster references",
                references_size,
                ADDRESS_SIZE,
                max_buffer_size,
                STORAGE_USAGE | INPUT_USAGE,
                MemoryLocation::GpuOnly,
            )?;
            let upload_address = allocate(
                gpu,
                &mut upload,
                "GI triangle cluster build input",
                upload_size,
                ADDRESS_SIZE,
                max_buffer_size,
                STORAGE_USAGE | INPUT_USAGE,
                MemoryLocation::CpuToGpu,
            )?;

            let blas_record = BottomLevelRecord {
                cluster_references_count: cluster_count,
                cluster_references_stride: ADDRESS_SIZE as u32,
                cluster_references: references_address,
            };
            let base = (upload_address - upload.address) as usize;
            upload.write_at(base, vertex_bytes);
            upload.write_at(base + indices_offset as usize, index_bytes);
            let mut record_offset = base + triangle_records_offset as usize;
            for (id, cluster) in mesh.clusters.iter().enumerate() {
                let record =
                    cluster.record(id as u32, upload_address, upload_address + indices_offset);
                upload.write_at(record_offset, bytemuck::bytes_of(&record));
                record_offset += size_of::<TriangleRecord>();
            }
            upload.write_at(base + blas_record_offset as usize, bytemuck::bytes_of(&blas_record));
            upload.write_at(base + blas_address_offset as usize, bytemuck::bytes_of(&blas_address));

            let build_clusters = CommandsInfo {
                s_type: COMMANDS_INFO_NV,
                p_next: ptr::null_mut(),
                input: triangles_input,
                dst_implicit_data: clusters_address,
                scratch_data: scratch_address,
                dst_addresses_array: region(references_address, ADDRESS_SIZE, references_size),
                dst_sizes_array: region(0, size_of::<u32>() as u64, 0),
                // Zero stride means tightly packed in the op-specific record ABI.
                src_infos_array: region(upload_address + triangle_records_offset, 0, triangle_records_size),
                src_infos_count: 0, // use max_acceleration_structure_count
                address_resolution_flags: 0,
            };
            let build_blas = CommandsInfo {
                input: blas_input,
                dst_implicit_data: 0,
                dst_addresses_array: region(
                    upload_address + blas_address_offset,
                    ADDRESS_SIZE,
                    ADDRESS_SIZE,
                ),
                src_infos_array: region(
                    upload_address + blas_record_offset,
                    0,
                    size_of::<BottomLevelRecord>() as u64,
                ),
                ..build_clusters
            };

            // SAFETY: The feature, command pointers, input format, counts and
            // aligned memory ranges are valid. Fresh buffers stay live until
            // submit waits for the renderer's queue. Submission also makes
            // the coherent CpuToGpu writes available to the device.
            unsafe {
                use vk::{AccessFlags2 as A, PipelineStageFlags2 as P};
                let build = P::ACCELERATION_STRUCTURE_BUILD_KHR;
                let as_write = A::ACCELERATION_STRUCTURE_WRITE_KHR;
                submit(gpu, pool, |cmd| {
                    (dispatch.cmd_build)(cmd, &build_clusters);
                    // Publish cluster addresses and AS data, and serialize scratch reuse.
                    gpu::memory_barrier(
                        gpu.device,
                        cmd,
                        (build, as_write),
                        (
                            build,
                            A::ACCELERATION_STRUCTURE_READ_KHR | as_write | A::INDIRECT_COMMAND_READ,
                        ),
                    );
                    (dispatch.cmd_build)(cmd, &build_blas);
                    // Ordinary KHR TLAS builds and ray queries can now read the BLAS.
                    gpu::memory_barrier(
                        gpu.device,
                        cmd,
                        (build, as_write),
                        (P::ALL_COMMANDS, A::ACCELERATION_STRUCTURE_READ_KHR),
                    );
                })?;
            }
            Ok(blas_address)
        })();
        // SAFETY: submit waits for the queue on success; command recording or
        // submission failure leaves no pending work (or the device is lost).
        unsafe {
            for buffer in [upload, references, scratch] {
                gpu::destroy_buffer(gpu.device, gpu.allocator, buffer);
            }
            match result {
                Ok(address) => Ok(Self { storage, address }),
                Err(error) => {
                    gpu::destroy_buffer(gpu.device, gpu.allocator, storage);
                    Err(error)
                }
            }
        }
    }

    /// The address to use as a standard Vulkan TLAS instance's acceleration structure reference.
    pub(super) fn address(&self) -> vk::DeviceAddress {
        self.address
    }

    /// # Safety
    /// `gpu` must own this storage, and no pending build or traversal may reference it.
    pub(super) unsafe fn destroy(self, gpu: &mut Gpu) {
        // SAFETY: Guaranteed by the caller; the cluster and BLAS share this allocation.
        unsafe { gpu::destroy_buffer(gpu.device, gpu.allocator, self.storage) };
    }
}

#[cfg(test)]
mod tests {
    use super::{ClusterMesh, ClusterProperties, MAX_PACKED_GEOMETRY_INDEX};

    #[test]
    fn clusters_preserve_source_triangle_order_and_winding() {
        let positions = core::array::from_fn::<_, 6, _>(|i| [i as f32, 0.0, 0.0]);
        let triangles = [[5, 0, 3], [5, 3, 1], [2, 1, 4], [0, 5, 5]];
        let limits = ClusterProperties {
            max_vertices_per_cluster: 4,
            max_triangles_per_cluster: 2,
            max_cluster_geometry_index: 100,
            ..Default::default()
        };
        let mesh = ClusterMesh::compact(&positions, &triangles, &limits)
            .expect("mesh fits the cluster limits");

        assert_eq!(mesh.clusters.len(), 3);
        assert_eq!(mesh.vertices.len(), 9);
        for (id, cluster) in mesh.clusters.iter().enumerate() {
            let record = cluster.record(id as u32, 0, 0);
            let geometry_index = record.geometry_index_and_flags & MAX_PACKED_GEOMETRY_INDEX;
            let first = geometry_index as usize;
            let count = cluster.triangle_count as usize;
            for (primitive, &local) in mesh.indices[first..first + count].iter().enumerate() {
                assert_eq!(
                    local.map(|v| mesh.vertices[cluster.first_vertex + usize::from(v)]),
                    triangles[first + primitive].map(|v| positions[v as usize]),
                    "cluster {id} primitive {primitive} must retain its input triangle"
                );
            }
        }
    }
}
