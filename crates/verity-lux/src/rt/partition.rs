//! Sparse NVIDIA partitioned TLAS builds over ordinary BLAS device addresses.

use std::{
    collections::{HashMap, HashSet},
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

/// The host descriptor type for a PTLAS; shaders still use `AccelerationStructure`.
pub(crate) const DESCRIPTOR_TYPE: vk::DescriptorType = vk::DescriptorType::from_raw(1_000_570_000);

// Vulkan SDK 1.4.363.0 vulkan_core.h. These declarations are not present in ash 0.38.
const PROPERTIES_NV: vk::StructureType = vk::StructureType::from_raw(1_000_570_001);
const WRITE_DESCRIPTOR_NV: vk::StructureType = vk::StructureType::from_raw(1_000_570_002);
const INSTANCES_INPUT_NV: vk::StructureType = vk::StructureType::from_raw(1_000_570_003);
const BUILD_INFO_NV: vk::StructureType = vk::StructureType::from_raw(1_000_570_004);
const WRITE_INSTANCE_NV: i32 = 0;
const PTLAS_ALIGNMENT: u64 = 256;

#[repr(C)]
struct PartitionProperties {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    max_partition_count: u32,
}

#[repr(C)]
struct WriteDescriptor {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    acceleration_structure_count: u32,
    acceleration_structures: *const vk::DeviceAddress,
}

#[repr(C)]
struct InstancesInput {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    flags: vk::BuildAccelerationStructureFlagsKHR,
    instance_count: u32,
    max_instance_per_partition_count: u32,
    partition_count: u32,
    max_instance_in_global_partition_count: u32,
}

#[repr(C)]
struct BuildInfo {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    input: InstancesInput,
    src_acceleration_structure_data: vk::DeviceAddress,
    dst_acceleration_structure_data: vk::DeviceAddress,
    scratch_data: vk::DeviceAddress,
    src_infos: vk::DeviceAddress,
    src_infos_count: vk::DeviceAddress,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StridedDeviceAddress {
    start_address: vk::DeviceAddress,
    stride_in_bytes: vk::DeviceSize,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct IndirectCommand {
    op_type: i32,
    arg_count: u32,
    arg_data: StridedDeviceAddress,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct WriteInstance {
    transform: [f32; 12],
    explicit_aabb: [f32; 6],
    instance_id: u32,
    instance_mask: u32,
    hit_group_index: u32,
    instance_flags: u32,
    instance_index: u32,
    partition_index: u32,
    acceleration_structure: vk::DeviceAddress,
}

#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(size_of::<PartitionProperties>() == 24);
    assert!(size_of::<WriteDescriptor>() == 32);
    assert!(size_of::<InstancesInput>() == 40);
    assert!(size_of::<BuildInfo>() == 96);
    assert!(size_of::<IndirectCommand>() == 24);
    assert!(size_of::<WriteInstance>() == 104);
};

const COUNT_OFFSET: usize = size_of::<IndirectCommand>();
const WRITES_OFFSET: usize = (COUNT_OFFSET + size_of::<u32>()).next_multiple_of(8);

type GetBuildSizes = unsafe extern "system" fn(
    vk::Device,
    *const InstancesInput,
    *mut vk::AccelerationStructureBuildSizesInfoKHR<'_>,
);
type CmdBuild = unsafe extern "system" fn(vk::CommandBuffer, *const BuildInfo);

struct Commands {
    get_sizes: GetBuildSizes,
    build: CmdBuild,
}

impl Commands {
    fn load(gpu: &Gpu<'_>) -> Result<Self> {
        // SAFETY: These are SDK device commands and the logical device enabled PTLAS.
        let (sizes, build) = unsafe {
            let load = |name: &std::ffi::CStr| {
                gpu.instance
                    .get_device_proc_addr(gpu.device.handle(), name.as_ptr())
            };
            (
                load(c"vkGetPartitionedAccelerationStructuresBuildSizesNV")
                    .ok_or("vkGetPartitionedAccelerationStructuresBuildSizesNV unavailable")?,
                load(c"vkCmdBuildPartitionedAccelerationStructuresNV")
                    .ok_or("vkCmdBuildPartitionedAccelerationStructuresNV unavailable")?,
            )
        };
        // SAFETY: Both non-null addresses came from Vulkan for these exact SDK signatures.
        unsafe {
            Ok(Self {
                get_sizes: mem::transmute::<unsafe extern "system" fn(), GetBuildSizes>(sizes),
                build: mem::transmute::<unsafe extern "system" fn(), CmdBuild>(build),
            })
        }
    }
}

/// A live instance in an authoritative scene snapshot.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PartitionInstance {
    /// Caller-owned identity; survives instance ordering and shader table changes.
    pub stable_id: u64,
    /// The shader's 24-bit custom instance index, not the persistent Vulkan slot.
    pub shader_index: u32,
    /// Row-major world-from-local affine transform.
    pub transform: [f32; 12],
    pub blas: vk::DeviceAddress,
    pub mask: u8,
    /// Only the four facing/opacity flags shared by KHR and NV are accepted.
    pub flags: vk::GeometryInstanceFlagsKHR,
}

impl PartitionInstance {
    fn position(&self) -> [f64; 3] {
        [3, 7, 11].map(|i| f64::from(self.transform[i]))
    }

    fn write(&self, instance_index: u32, partition_index: u32) -> WriteInstance {
        WriteInstance {
            transform: self.transform,
            explicit_aabb: [0.0; 6],
            instance_id: self.shader_index,
            instance_mask: u32::from(self.mask),
            hit_group_index: 0,
            instance_flags: self.flags.as_raw(),
            instance_index,
            partition_index,
            acceleration_structure: self.blas,
        }
    }
}

impl WriteInstance {
    fn disabled(instance_index: u32, partition_index: u32) -> Self {
        Self {
            transform: [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
            instance_index,
            partition_index,
            // VK_NV_partitioned_acceleration_structure disables an instance with a null BLAS.
            ..Self::zeroed()
        }
    }

    fn matches(&self, record: &PartitionInstance, partition_index: u32) -> bool {
        self.instance_id == record.shader_index
            && self.instance_mask == u32::from(record.mask)
            && self.instance_flags == record.flags.as_raw()
            && self.partition_index == partition_index
            && self.acceleration_structure == record.blas
            && self.transform == record.transform
    }
}

#[derive(Clone, Copy)]
struct Axis {
    center: f64,
    cells_per_unit: f64,
    cells: u32,
}

struct Grid {
    axes: [Axis; 3],
    partitions: u32,
}

impl Grid {
    fn fit(records: &[PartitionInstance], count: u32, max_partitions: u32) -> Self {
        let first = records.first().map_or([0.; 3], PartitionInstance::position);
        let (mut low, mut high) = (first, first);
        for record in records.iter().skip(1) {
            for (i, position) in record.position().into_iter().enumerate() {
                low[i] = low[i].min(position);
                high[i] = high[i].max(position);
            }
        }
        let half_extents = [0, 1, 2].map(|i| ((high[i] - low[i]) * 0.625).max(0.5));
        let mut cells = [1; 3];
        let target = count.div_ceil(64).max(2);
        let mut partitions = 1;
        while partitions < target && partitions <= max_partitions / 2 {
            let width = |i: usize| half_extents[i] / f64::from(cells[i]);
            let mut longest = if width(0) >= width(1) { 0 } else { 1 };
            if width(2) > width(longest) {
                longest = 2;
            }
            cells[longest] *= 2;
            partitions *= 2;
        }
        let axes = [0, 1, 2].map(|i| Axis {
            center: (low[i] + high[i]) * 0.5,
            cells_per_unit: f64::from(cells[i]) * 0.5 / half_extents[i],
            cells: cells[i],
        });
        Self { axes, partitions }
    }

    fn locate(&self, record: &PartitionInstance) -> u32 {
        let mut partition = 0;
        for (axis, position) in self.axes.iter().zip(record.position()) {
            if axis.cells == 1 {
                continue;
            }
            let cell =
                ((position - axis.center) * axis.cells_per_unit + f64::from(axis.cells / 2)) as u32;
            // The outer cells are open-ended, so crossing the old scene bounds
            // alone does not discard the existing partition or instance layout.
            partition = partition * axis.cells + cell.min(axis.cells - 1);
        }
        partition
    }
}

#[derive(Clone, Copy)]
struct Limits {
    slots: u32,
    partitions: u32,
    per_partition: u32,
}

impl Limits {
    fn input(self) -> InstancesInput {
        InstancesInput {
            s_type: INSTANCES_INPUT_NV,
            p_next: ptr::null_mut(),
            flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
            instance_count: self.slots,
            max_instance_per_partition_count: self.per_partition,
            partition_count: self.partitions,
            max_instance_in_global_partition_count: 0,
        }
    }
}

fn headroom(count: u32, minimum: u32) -> u32 {
    count.saturating_add(count.div_ceil(4)).max(minimum)
}

struct Retired {
    id: u64,
    index: u32,
    partition: u32,
}

#[derive(Default)]
struct Slots {
    by_id: HashMap<u64, WriteInstance>,
    free: Vec<u32>,
    next: u32,
}

struct SlotChanges {
    free_left: usize,
    retired_used: usize,
    next: u32,
}

#[derive(Default)]
struct Workspace {
    seen: HashSet<u64>,
    bins: Vec<u32>,
    counts: Vec<u32>,
    writes: Vec<WriteInstance>,
    written_ids: Vec<u64>,
    retired: Vec<Retired>,
}

impl Workspace {
    fn validate(&mut self, records: &[PartitionInstance]) -> Result<u32> {
        let count = u32::try_from(records.len())?;
        self.seen.clear();
        self.seen.reserve(records.len());
        for record in records {
            if !self.seen.insert(record.stable_id) {
                return Err(
                    format!("duplicate PTLAS instance identity {}", record.stable_id).into(),
                );
            }
            // NV uses 0x10 for explicit AABBs, not the KHR micromap instance flag.
            let flags = record.flags.as_raw();
            if record.shader_index > 0xff_ffff
                || record.blas == 0
                || flags & !0xf != 0
                || flags & 0xc == 0xc
                || !record.transform.iter().all(|v| v.is_finite())
            {
                return Err(format!("invalid PTLAS instance {}", record.stable_id).into());
            }
        }
        Ok(count)
    }

    fn bin(&mut self, grid: &Grid, records: &[PartitionInstance]) -> u32 {
        self.bins.clear();
        self.bins.reserve(records.len());
        self.counts.clear();
        self.counts.resize(grid.partitions as usize, 0);
        let mut max_count = 0;
        for record in records {
            let index = grid.locate(record);
            self.bins.push(index);
            let count = &mut self.counts[index as usize];
            *count += 1;
            max_count = max_count.max(*count);
        }
        max_count
    }

    fn full_writes(&mut self, records: &[PartitionInstance], count: u32, limits: Limits) {
        self.writes.clear();
        self.writes.reserve(limits.slots as usize);
        self.writes.extend(
            (0..count)
                .zip(records.iter().zip(&self.bins))
                .map(|(slot, (record, &partition))| record.write(slot, partition)),
        );
        // Initialize every reserved Vulkan slot to an explicitly disabled instance.
        // Distribute these writes within the declared partition capacities as well.
        let mut unused = count..limits.slots;
        for (partition, &used) in (0..limits.partitions).zip(&self.counts) {
            self.writes.extend(
                unused
                    .by_ref()
                    .take((limits.per_partition - used) as usize)
                    .map(|slot| WriteInstance::disabled(slot, partition)),
            );
            if unused.is_empty() {
                break;
            }
        }
    }

    fn sparse_writes(&mut self, slots: &Slots, records: &[PartitionInstance]) -> SlotChanges {
        self.writes.clear();
        self.written_ids.clear();
        self.retired.clear();
        self.retired
            .extend(slots.by_id.iter().filter_map(|(&id, old)| {
                (!self.seen.contains(&id)).then_some(Retired {
                    id,
                    index: old.instance_index,
                    partition: old.partition_index,
                })
            }));
        let mut changes = SlotChanges {
            free_left: slots.free.len(),
            retired_used: 0,
            next: slots.next,
        };
        for (record, &partition) in records.iter().zip(&self.bins) {
            let index = if let Some(old) = slots.by_id.get(&record.stable_id) {
                if old.matches(record, partition) {
                    continue;
                }
                old.instance_index
            } else if let Some(retired) = self.retired.get(changes.retired_used) {
                // Overwrite a retired slot once, not both a removal and an insertion.
                changes.retired_used += 1;
                retired.index
            } else if changes.free_left > 0 {
                changes.free_left -= 1;
                slots.free[changes.free_left]
            } else {
                let index = changes.next;
                changes.next += 1;
                index
            };
            self.written_ids.push(record.stable_id);
            self.writes.push(record.write(index, partition));
        }
        self.writes.extend(
            self.retired[changes.retired_used..]
                .iter()
                .map(|old| WriteInstance::disabled(old.index, old.partition)),
        );
        changes
    }

    fn commit(&self, slots: &mut Slots, changes: SlotChanges) {
        for old in &self.retired {
            slots.by_id.remove(&old.id);
        }
        slots.free.truncate(changes.free_left);
        slots.free.extend(
            self.retired[changes.retired_used..]
                .iter()
                .map(|old| old.index),
        );
        slots.next = changes.next;
        for (&id, &write) in self.written_ids.iter().zip(&self.writes) {
            slots.by_id.insert(id, write);
        }
    }
}

#[derive(Default)]
struct Storage {
    accel: Buffer,
    scratch: Buffer,
    upload: Buffer,
}

impl Storage {
    fn allocate(gpu: &mut Gpu<'_>, commands: &Commands, limits: Limits) -> Result<Self> {
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        // SAFETY: PTLAS is enabled and limits conform to the queried device properties.
        unsafe { (commands.get_sizes)(gpu.device.handle(), &limits.input(), &mut sizes) };
        if sizes.acceleration_structure_size == 0 {
            return Err("PTLAS driver returned an empty acceleration structure size".into());
        }
        let accel_size = sizes
            .acceleration_structure_size
            .checked_add(PTLAS_ALIGNMENT - 1)
            .ok_or("PTLAS acceleration structure size overflow")?;
        // NV requires buildScratchSize even when the source PTLAS is non-null.
        let scratch_size = sizes
            .build_scratch_size
            .max(1)
            .checked_add(PTLAS_ALIGNMENT - 1)
            .ok_or("PTLAS scratch buffer size overflow")?;
        let upload_size = (WRITES_OFFSET + 7) as u64
            + u64::from(limits.slots) * size_of::<WriteInstance>() as u64;
        let mut storage = Self::default();
        // SAFETY: The buffers are allocated on the same device as the allocator in Gpu.
        let result: Result<()> = (|| unsafe {
            storage.accel = gpu::create_buffer(
                gpu.device,
                gpu.allocator,
                "gi partitioned tlas",
                accel_size,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
                MemoryLocation::GpuOnly,
            )?;
            storage.scratch = gpu::create_buffer(
                gpu.device,
                gpu.allocator,
                "gi partitioned tlas scratch",
                scratch_size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::GpuOnly,
            )?;
            storage.upload = gpu::create_buffer(
                gpu.device,
                gpu.allocator,
                "gi partitioned tlas ops",
                upload_size,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                MemoryLocation::CpuToGpu,
            )?;
            Ok(())
        })();
        if let Err(error) = result {
            // SAFETY: None of these buffers has been used by the GPU.
            unsafe { storage.destroy(gpu) };
            return Err(error);
        }
        Ok(storage)
    }

    fn address(&self) -> vk::DeviceAddress {
        self.accel.address.next_multiple_of(PTLAS_ALIGNMENT)
    }

    /// # Safety
    /// `upload` must be idle, and all BLAS addresses and GPU objects must be valid.
    unsafe fn write(
        &mut self,
        gpu: &Gpu<'_>,
        pool: &mut vk::CommandPool,
        commands: &Commands,
        limits: Limits,
        src: vk::DeviceAddress,
        writes: &[WriteInstance],
    ) -> Result<()> {
        let base = self.upload.address.next_multiple_of(8);
        let offset = (base - self.upload.address) as usize;
        let cmd = IndirectCommand {
            op_type: WRITE_INSTANCE_NV,
            arg_count: u32::try_from(writes.len())?,
            arg_data: StridedDeviceAddress {
                start_address: base + WRITES_OFFSET as u64,
                stride_in_bytes: size_of::<WriteInstance>() as u64,
            },
        };
        self.upload.write_at(offset, bytemuck::bytes_of(&cmd));
        self.upload
            .write_at(offset + COUNT_OFFSET, bytemuck::bytes_of(&1u32));
        self.upload
            .write_at(offset + WRITES_OFFSET, bytemuck::cast_slice(writes));
        let info = BuildInfo {
            s_type: BUILD_INFO_NV,
            p_next: ptr::null_mut(),
            input: limits.input(),
            src_acceleration_structure_data: src,
            dst_acceleration_structure_data: self.address(),
            scratch_data: self.scratch.address.next_multiple_of(PTLAS_ALIGNMENT),
            src_infos: base,
            src_infos_count: base + COUNT_OFFSET as u64,
        };
        let build = vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR;
        let all = vk::PipelineStageFlags2::ALL_COMMANDS;
        let read = vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR;
        let write = vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR;
        let before = [vk::MemoryBarrier2::default()
            .src_stage_mask(all)
            .src_access_mask(read | write)
            .dst_stage_mask(build)
            .dst_access_mask(read | write)];
        let after = [vk::MemoryBarrier2::default()
            .src_stage_mask(build)
            .src_access_mask(write)
            .dst_stage_mask(all)
            .dst_access_mask(read)];
        // SAFETY: Allocation and the SDK layout above keep all required addresses live,
        // non-overlapping (except in-place source/destination), and aligned. Queue submission
        // makes the coherent host writes visible and waits before the upload can be reused.
        unsafe {
            submit(gpu, pool, |cmd| {
                gpu.device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().memory_barriers(&before),
                );
                (commands.build)(cmd, &info);
                gpu.device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().memory_barriers(&after),
                );
            })
        }
    }

    /// # Safety
    /// The GPU must be done with these buffers.
    unsafe fn destroy(self, gpu: &mut Gpu<'_>) {
        // SAFETY: Owned buffers allocated on this device, and the caller ensured they are idle.
        unsafe {
            gpu::destroy_buffer(gpu.device, gpu.allocator, self.upload);
            gpu::destroy_buffer(gpu.device, gpu.allocator, self.scratch);
            gpu::destroy_buffer(gpu.device, gpu.allocator, self.accel);
        }
    }
}

/// A buffer-backed PTLAS with stable per-instance Vulkan slots between rebuilds.
pub(crate) struct PartitionedTlas {
    commands: Commands,
    max_partitions: u32,
    storage: Storage,
    grid: Grid,
    limits: Limits,
    slots: Slots,
    work: Workspace,
}

impl PartitionedTlas {
    /// Build an initial snapshot, or return `None` if PTLAS was not enabled.
    ///
    /// # Safety
    /// The command pool must belong to this GPU and queue family (or be null).
    /// Input BLASes must be valid on this GPU through all subsequent traversal using them.
    /// The transform's linear part must be invertible for each instance.
    pub(crate) unsafe fn build(
        gpu: &mut Gpu<'_>,
        pool: &mut vk::CommandPool,
        records: &[PartitionInstance],
    ) -> Result<Option<Self>> {
        if !gpu.ray_features.partitioned_acceleration {
            return Ok(None);
        }
        let commands = Commands::load(gpu)?;
        let mut partitions = PartitionProperties {
            s_type: PROPERTIES_NV,
            p_next: ptr::null_mut(),
            max_partition_count: 0,
        };
        let mut properties = vk::PhysicalDeviceProperties2 {
            p_next: ptr::from_mut(&mut partitions).cast(),
            ..Default::default()
        };
        // SAFETY: The enabled extension defines the output chain structure and this physical device.
        unsafe {
            gpu.instance
                .get_physical_device_properties2(gpu.physical, &mut properties)
        };
        let max_partitions = partitions.max_partition_count;
        if max_partitions == 0 {
            return Err("PTLAS driver reported no supported partitions".into());
        }
        let mut work = Workspace::default();
        let count = work.validate(records)?;
        let grid = Grid::fit(records, count, max_partitions);
        let limits = Self::prepare_full(&mut work, &grid, records, count, headroom(count, 16));
        // SAFETY: Validated records and the caller's pool and BLAS validity guarantees.
        let storage = unsafe { Self::build_storage(gpu, pool, &commands, limits, &work.writes) }?;
        let mut slots = Slots::default();
        Self::reset_slots(&mut slots, records, &work.writes, count);
        Ok(Some(Self {
            commands,
            max_partitions,
            storage,
            grid,
            limits,
            slots,
            work,
        }))
    }

    /// The aligned Vulkan device address for an NV PTLAS descriptor write.
    pub(crate) fn address(&self) -> vk::DeviceAddress {
        self.storage.address()
    }

    /// Write this address to the given NV PTLAS descriptor binding.
    ///
    /// # Safety
    /// `device` must own the PTLAS and `set`. The set binding must be idle and of [`DESCRIPTOR_TYPE`].
    pub(crate) unsafe fn write_descriptor(
        &self,
        device: &ash::Device,
        set: vk::DescriptorSet,
        binding: u32,
    ) {
        let address = self.address();
        let extension = WriteDescriptor {
            s_type: WRITE_DESCRIPTOR_NV,
            p_next: ptr::null_mut(),
            acceleration_structure_count: 1,
            acceleration_structures: &address,
        };
        let mut write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(binding)
            .descriptor_type(DESCRIPTOR_TYPE)
            .descriptor_count(1);
        write.p_next = ptr::from_ref(&extension).cast();
        // SAFETY: Caller guarantees a valid, idle binding. The SDK extension and its address
        // remain alive for the duration of this Vulkan call.
        unsafe { device.update_descriptor_sets(&[write], &[]) };
    }

    /// Apply a complete scene snapshot, writing only changed, inserted and retired slots.
    /// Returns `true` if a rebuild changed the descriptor address.
    ///
    /// # Safety
    /// The same GPU must be done traversing the PTLAS. Both the previous and new BLASes
    /// must remain valid until this call completes. The pool and transforms must satisfy
    /// [`Self::build`]'s requirements. Refresh descriptors when the address changes.
    pub(crate) unsafe fn update(
        &mut self,
        gpu: &mut Gpu<'_>,
        pool: &mut vk::CommandPool,
        records: &[PartitionInstance],
    ) -> Result<bool> {
        let count = self.work.validate(records)?;
        if count > self.limits.slots
            || self.work.bin(&self.grid, records) > self.limits.per_partition
        {
            let grid = Grid::fit(records, count, self.max_partitions);
            let capacity = if count > self.limits.slots {
                headroom(count, 16)
            } else {
                self.limits.slots
            };
            let limits = Self::prepare_full(&mut self.work, &grid, records, count, capacity);
            // SAFETY: Pool and BLASes are valid, and a fresh destination is used.
            let storage = unsafe {
                Self::build_storage(gpu, pool, &self.commands, limits, &self.work.writes)
            }?;
            let old = mem::replace(&mut self.storage, storage);
            // SAFETY: The caller guaranteed the old PTLAS was idle, including its backing buffers.
            unsafe { old.destroy(gpu) };
            self.grid = grid;
            self.limits = limits;
            Self::reset_slots(&mut self.slots, records, &self.work.writes, count);
            return Ok(true);
        }
        let changes = self.work.sparse_writes(&self.slots, records);
        if self.work.writes.is_empty() {
            return Ok(false);
        }
        let src = self.address();
        // SAFETY: The current buffer is a previously built PTLAS on this GPU, and writes
        // describe disjoint slots within its unchanged build limits.
        unsafe {
            self.storage.write(
                gpu,
                pool,
                &self.commands,
                self.limits,
                src,
                &self.work.writes,
            )
        }?;
        self.work.commit(&mut self.slots, changes);
        Ok(false)
    }

    /// Release the build buffers and the address backing all descriptor references.
    ///
    /// # Safety
    /// `gpu` must be the original GPU and must be finished with the PTLAS.
    pub(crate) unsafe fn destroy(self, gpu: &mut Gpu<'_>) {
        // SAFETY: Caller ensures that these buffers are idle on the original GPU.
        unsafe { self.storage.destroy(gpu) };
    }

    fn prepare_full(
        work: &mut Workspace,
        grid: &Grid,
        records: &[PartitionInstance],
        count: u32,
        capacity: u32,
    ) -> Limits {
        let max_count = work.bin(grid, records);
        let limits = Limits {
            slots: capacity,
            partitions: grid.partitions,
            per_partition: headroom(max_count, capacity.div_ceil(grid.partitions)).min(capacity),
        };
        work.full_writes(records, count, limits);
        limits
    }

    fn reset_slots(
        slots: &mut Slots,
        records: &[PartitionInstance],
        writes: &[WriteInstance],
        count: u32,
    ) {
        slots.by_id.clear();
        slots.by_id.extend(
            records
                .iter()
                .map(|record| record.stable_id)
                .zip(writes.iter().copied()),
        );
        slots.free.clear();
        slots.next = count;
    }

    /// # Safety
    /// Pool and write addresses must be valid on this GPU, and writes must fit `limits`.
    unsafe fn build_storage(
        gpu: &mut Gpu<'_>,
        pool: &mut vk::CommandPool,
        commands: &Commands,
        limits: Limits,
        writes: &[WriteInstance],
    ) -> Result<Storage> {
        let mut storage = Storage::allocate(gpu, commands, limits)?;
        // SAFETY: Fresh idle buffers; the caller guarantees write and pool validity.
        let result = unsafe { storage.write(gpu, pool, commands, limits, 0, writes) };
        if let Err(error) = result {
            // SAFETY: submit either did not submit the commands, drained the queue, or lost the device.
            unsafe { storage.destroy(gpu) };
            return Err(error);
        }
        Ok(storage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(stable_id: u64, shader_index: u32, x: f32) -> PartitionInstance {
        PartitionInstance {
            stable_id,
            shader_index,
            transform: [1., 0., 0., x, 0., 1., 0., 0., 0., 0., 1., 0.],
            blas: 256,
            mask: 0xff,
            flags: vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE,
        }
    }

    #[test]
    fn packed_order_does_not_move_slots_or_require_unchanged_writes() {
        let first = record(500, 0, -8.);
        let second = record(2, 1, 8.);
        let records = [first, second];
        let grid = Grid::fit(&records, 2, 64);
        let mut work = Workspace::default();
        let limits = PartitionedTlas::prepare_full(&mut work, &grid, &records, 2, 16);
        let mut slots = Slots::default();
        PartitionedTlas::reset_slots(&mut slots, &records, &work.writes, 2);

        let reordered = [second, first];
        work.validate(&reordered).unwrap();
        work.bin(&grid, &reordered);
        work.sparse_writes(&slots, &reordered);
        assert!(work.writes.is_empty());

        let swapped_indices = [record(2, 0, -12.), record(500, 1, -8.)];
        work.validate(&swapped_indices).unwrap();
        assert!(work.bin(&grid, &swapped_indices) <= limits.per_partition);
        let changes = work.sparse_writes(&slots, &swapped_indices);
        assert_eq!(work.writes.len(), 2);
        assert_eq!(work.writes[0].instance_index, 1);
        assert_eq!(work.writes[0].instance_id, 0);
        assert_eq!(
            work.writes[0].partition_index,
            slots.by_id[&500].partition_index
        );
        assert_eq!(work.writes[1].instance_index, 0);
        assert_eq!(work.writes[1].instance_id, 1);
        work.commit(&mut slots, changes);
        assert_eq!(slots.by_id[&2].instance_index, 1);
        assert_eq!(slots.by_id[&500].instance_index, 0);
    }

    #[test]
    fn disabled_and_reused_slots_do_not_rewrite_survivors() {
        let survivor = record(41, 1, 31.);
        let records = [record(30, 0, -31.), survivor];
        let grid = Grid::fit(&records, 2, 64);
        let mut work = Workspace::default();
        let limits = PartitionedTlas::prepare_full(&mut work, &grid, &records, 2, 16);
        assert_eq!(work.writes.len(), 16);
        assert_ne!(
            work.writes[0].partition_index,
            work.writes[1].partition_index
        );
        assert!(
            work.writes[2..]
                .iter()
                .all(|write| write.acceleration_structure == 0)
        );
        assert!(
            work.writes
                .iter()
                .enumerate()
                .all(|(i, write)| write.instance_index as usize == i)
        );
        for partition in 0..limits.partitions {
            let writes = work
                .writes
                .iter()
                .filter(|write| write.partition_index == partition);
            assert!(writes.count() <= limits.per_partition as usize);
        }
        let mut slots = Slots::default();
        PartitionedTlas::reset_slots(&mut slots, &records, &work.writes, 2);

        work.validate(&[survivor]).unwrap();
        work.bin(&grid, &[survivor]);
        let changes = work.sparse_writes(&slots, &[survivor]);
        assert_eq!(work.writes.len(), 1);
        assert_eq!(work.writes[0].instance_index, 0);
        assert_eq!(
            work.writes[0].partition_index,
            slots.by_id[&30].partition_index
        );
        assert_eq!(work.writes[0].acceleration_structure, 0);
        work.commit(&mut slots, changes);
        assert!(!slots.by_id.contains_key(&30));

        let records = [record(65, 0, 31.), survivor];
        work.validate(&records).unwrap();
        work.bin(&grid, &records);
        let changes = work.sparse_writes(&slots, &records);
        assert_eq!(work.writes.len(), 1);
        assert_eq!(work.writes[0].instance_index, 0);
        assert_eq!(
            work.writes[0].partition_index,
            slots.by_id[&41].partition_index
        );
        assert_ne!(work.writes[0].acceleration_structure, 0);
        work.commit(&mut slots, changes);
        assert_eq!(slots.by_id[&65].instance_index, 0);
        assert_eq!(slots.by_id[&41].instance_index, 1);
    }

    #[test]
    fn replacement_writes_the_retired_slot_only_once() {
        let first = record(9, 0, 3.);
        let second = record(11, 0, 3.);
        let grid = Grid::fit(&[first], 1, 1);
        let mut work = Workspace::default();
        PartitionedTlas::prepare_full(&mut work, &grid, &[first], 1, 1);
        let mut slots = Slots::default();
        PartitionedTlas::reset_slots(&mut slots, &[first], &work.writes, 1);

        work.validate(&[second]).unwrap();
        work.bin(&grid, &[second]);
        let changes = work.sparse_writes(&slots, &[second]);
        assert_eq!(work.writes.len(), 1);
        assert_eq!(work.writes[0].instance_index, 0);
        assert_eq!(work.writes[0].acceleration_structure, second.blas);
        work.commit(&mut slots, changes);
        assert!(!slots.by_id.contains_key(&9));
        assert_eq!(slots.by_id[&11].instance_index, 0);
        assert!(slots.free.is_empty());
        assert_eq!(slots.next, 1);
    }
}
