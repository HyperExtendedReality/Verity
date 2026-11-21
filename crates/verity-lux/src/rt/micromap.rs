//! Device-built `VK_EXT_opacity_micromap` inputs for triangle BLAS builds.

use ash::{ext::opacity_micromap, vk};
use gpu_allocator::MemoryLocation;
use verity_render::{
    Gpu,
    extension::Result,
    gpu::{self, Buffer},
};

const FORMAT: vk::OpacityMicromapFormatEXT = vk::OpacityMicromapFormatEXT::TYPE_4_STATE;
const SOURCE_ALIGNMENT: usize = 256; // vkCmdBuildMicromapsEXT-pInfos-07515
const LEVEL_COUNT: usize = usize::BITS as usize / 2; // 4^level must fit in usize.
const INDEX_BYTES: u64 = size_of::<u32>() as u64;
const TRIANGLE_BYTES: u64 = size_of::<vk::MicromapTriangleEXT>() as u64;
const _: () = assert!(TRIANGLE_BYTES == 8);

/// Inclusive, conservative bounds on the final alpha compared by the shader.
///
/// Bounds must include every relevant sample, including texture addressing,
/// filtering, LOD, vertex and material factors, and rounding. Bounds derived
/// only from a downsampled preview or a few samples cannot prove opacity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AlphaBounds {
    pub min: f32,
    pub max: f32,
}

/// The four Vulkan opacity values. Unknown states continue to the existing
/// shader test; never use Vulkan's 2-state override with such a map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum OpacityState {
    Transparent = 0,
    Opaque = 1,
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Lux does not force two-state opacity")
    )]
    UnknownTransparent = 2,
    UnknownOpaque = 3,
}

impl OpacityState {
    /// Classify a binary mask using the shader's `alpha >= cutoff` comparison.
    /// Missing, invalid, or inconclusive bounds require the normal shader test.
    pub(crate) fn classify(bounds: Option<AlphaBounds>, cutoff: f32) -> Self {
        let Some(AlphaBounds { min, max }) = bounds else {
            return Self::UnknownOpaque;
        };
        if !cutoff.is_finite() || !min.is_finite() || !max.is_finite() || min > max {
            Self::UnknownOpaque
        } else if max < cutoff {
            Self::Transparent
        } else if min >= cutoff {
            Self::Opaque
        } else {
            Self::UnknownOpaque
        }
    }

    const fn special_index(self) -> u32 {
        use vk::OpacityMicromapSpecialIndexEXT as Index;
        // The index buffer is UINT32; Vulkan recognizes the unsigned casts of
        // these negative sentinels, not the 0..3 values stored in micromap data.
        let index = match self {
            Self::Transparent => Index::FULLY_TRANSPARENT,
            Self::Opaque => Index::FULLY_OPAQUE,
            Self::UnknownTransparent => Index::FULLY_UNKNOWN_TRANSPARENT,
            Self::UnknownOpaque => Index::FULLY_UNKNOWN_OPAQUE,
        };
        index.as_raw() as u32
    }
}

/// Opacity data for one triangle in geometry primitive order.
#[derive(Clone, Copy, Debug)]
pub(crate) enum TriangleOpacity<'a> {
    /// One state for the whole triangle; encoded as a special index, not an object entry.
    Uniform(OpacityState),
    /// Exactly 4^level states in Vulkan micromap index order for the triangle's
    /// vertex order. Bounds used to generate these must also account for the
    /// difference between hardware and any shader-recomputed barycentrics.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Lux only supplies whole-triangle alpha proofs")
    )]
    Subdivided {
        level: u16,
        states: &'a [OpacityState],
    },
}

/// The micromap and per-primitive index input retained by its owning scene.
#[must_use = "release micromap resources with Micromaps::destroy"]
pub(crate) struct OpacityMicromap {
    storage: Option<Storage>,
    indices: Buffer,
    index_address: vk::DeviceAddress,
    primitive_count: u32,
    usage: Vec<vk::MicromapUsageEXT>,
}

impl OpacityMicromap {
    /// Produce the `pNext` entry for `AccelerationStructureGeometryTrianglesDataKHR`
    /// for both the BLAS size query and the device build. Keep this entry at a
    /// stable address while chained. Use exactly this geometry's original
    /// triangle order with one index per build primitive, no base offset.
    ///
    /// The index buffer must outlive the build; the map must outlive all uses
    /// of its dependent BLASes. For candidate-only shaders that do other tests
    /// as well as alpha, force *non*-opaque traversal even for known opaque
    /// cells, without forcing 2-state OMM. Vulkan drops transparent cells before
    /// the opacity override, leaving every other hit to normal shader testing.
    ///
    /// # Errors
    /// Returns an error if `primitive_count` differs from the map's input length.
    pub(crate) fn attachment(
        &self,
        primitive_count: u32,
    ) -> Result<vk::AccelerationStructureTrianglesOpacityMicromapEXT<'_>> {
        if primitive_count != self.primitive_count {
            return Err("opacity micromap primitive count does not match its BLAS geometry".into());
        }
        let mut attachment = vk::AccelerationStructureTrianglesOpacityMicromapEXT::default()
            .index_type(vk::IndexType::UINT32)
            .index_buffer(at(self.index_address))
            .index_stride(INDEX_BYTES);
        if let Some(storage) = &self.storage {
            attachment = attachment
                .micromap(storage.handle)
                .usage_counts(&self.usage);
        }
        // With only special indices Vulkan expressly permits a null micromap
        // handle and no usage counts. Every index in this case is a sentinel.
        Ok(attachment)
    }
}

struct Storage {
    handle: vk::MicromapEXT,
    buffer: Buffer,
}

/// Vulkan dispatch and limits for a logical device with EXT micromaps enabled.
#[derive(Clone)]
pub(crate) struct Micromaps {
    loader: opacity_micromap::Device,
    scratch_alignment: u64,
    max_subdivision_level: u32,
}

impl Micromaps {
    /// Load entry points and query limits after logical-device negotiation.
    ///
    /// # Safety
    /// `gpu` must identify a live device with `VK_EXT_opacity_micromap`, its
    /// `micromap` feature, and the acceleration structure, buffer address and
    /// synchronization2 features enabled. Use this builder only with that device.
    pub(crate) unsafe fn new(gpu: &Gpu) -> Self {
        let mut mm = vk::PhysicalDeviceOpacityMicromapPropertiesEXT::default();
        let mut accel = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
        let mut properties = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut mm)
            .push_next(&mut accel);
        // SAFETY: The device and extension preconditions apply; output nodes live through the call.
        unsafe {
            gpu.instance
                .get_physical_device_properties2(gpu.physical, &mut properties);
        }
        Self {
            loader: opacity_micromap::Device::new(gpu.instance, gpu.device),
            scratch_alignment: u64::from(accel.min_acceleration_structure_scratch_offset_alignment),
            max_subdivision_level: mm.max_opacity4_state_subdivision_level,
        }
    }

    /// Upload one geometry's opacity inputs and synchronously build the map if needed.
    ///
    /// No resources are created for empty inputs or when all states reduce to
    /// `UnknownOpaque`. Keep the unmodified geometry non-opaque in that case.
    /// Geometries with only uniform states need an index buffer but no micromap
    /// object. The build uses device buffers, not the optional host command feature.
    ///
    /// # Errors
    /// Returns an error for invalid input counts or subdivision levels, oversized
    /// data, allocation failures, or failed Vulkan calls.
    ///
    /// # Safety
    /// `gpu` must be the live device associated with this builder. `pool` must
    /// be null or belong to `gpu.queue_family` on that device. The queue must
    /// support compute and both queue and pool must be externally synchronized.
    /// Consume the map on the same queue, or provide the required queue-to-queue
    /// memory dependency and queue family ownership transfer.
    pub(crate) unsafe fn build<'a>(
        &self,
        gpu: &mut Gpu,
        pool: &mut vk::CommandPool,
        triangles: impl IntoIterator<Item = TriangleOpacity<'a>>,
    ) -> Result<Option<OpacityMicromap>> {
        let Encoded {
            indices,
            triangles,
            data,
            usage,
        } = Encoded::new(triangles, self.max_subdivision_level)?;
        if indices.is_empty() {
            return Ok(None);
        }
        let primitive_count = u32::try_from(indices.len())?;
        let bytes = bytemuck::cast_slice(&indices);
        // SAFETY: The caller guarantees the matching device/allocator. Padding
        // leaves space to align the device address to the UINT32 index type.
        let mut index_buffer = unsafe {
            gpu::create_buffer(
                gpu.device,
                gpu.allocator,
                "gi opacity micromap indices",
                bytes.len() as u64 + INDEX_BYTES - 1,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                MemoryLocation::CpuToGpu,
            )?
        };
        let offset = alignment_padding(index_buffer.address, INDEX_BYTES);
        index_buffer.write_at(offset as usize, bytes);
        drop(indices);

        let storage = if triangles.is_empty() {
            None
        } else {
            // SAFETY: Device, queue and pool satisfy the preconditions of this build;
            // encoding produced a nonempty triangle array and matching usage counts.
            match unsafe { self.build_storage(gpu, pool, &triangles, &data, &usage) } {
                Ok(storage) => Some(storage),
                Err(error) => {
                    // SAFETY: No BLAS build has used the index buffer.
                    unsafe { gpu::destroy_buffer(gpu.device, gpu.allocator, index_buffer) };
                    return Err(error);
                }
            }
        };
        Ok(Some(OpacityMicromap {
            storage,
            index_address: index_buffer.address + offset,
            indices: index_buffer,
            primitive_count,
            usage,
        }))
    }

    unsafe fn build_storage(
        &self,
        gpu: &mut Gpu,
        pool: &mut vk::CommandPool,
        triangles: &[vk::MicromapTriangleEXT],
        data: &[u8],
        usage: &[vk::MicromapUsageEXT],
    ) -> Result<Storage> {
        let info = vk::MicromapBuildInfoEXT::default()
            .ty(vk::MicromapTypeEXT::OPACITY_MICROMAP)
            .flags(vk::BuildMicromapFlagsEXT::PREFER_FAST_TRACE)
            .mode(vk::BuildMicromapModeEXT::BUILD)
            .usage_counts(usage)
            .triangle_array_stride(TRIANGLE_BYTES);
        let fp = self.loader.fp();
        let mut sizes = vk::MicromapBuildSizesInfoEXT::default();
        // SAFETY: The matching EXT device is live and usage describes the encoded triangles.
        unsafe {
            (fp.get_micromap_build_sizes_ext)(
                self.loader.device(),
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &info,
                &mut sizes,
            );
        }
        let scratch_size = sizes
            .build_scratch_size
            .max(1)
            .checked_add(self.scratch_alignment - 1)
            .ok_or("opacity micromap scratch size overflow")?;
        let triangle_offset = data
            .len()
            .checked_next_multiple_of(SOURCE_ALIGNMENT)
            .ok_or("opacity micromap source offset overflow")?;
        let size = triangle_offset
            .checked_add(size_of_val(triangles))
            .and_then(|n| n.checked_add(SOURCE_ALIGNMENT - 1))
            .ok_or("opacity micromap source size overflow")?;
        // SAFETY: The caller guarantees the matching device/allocator. The
        // source will be fully written before submission on the build queue.
        let mut source = unsafe {
            gpu::create_buffer(
                gpu.device,
                gpu.allocator,
                "gi opacity micromap source",
                size as u64,
                vk::BufferUsageFlags::MICROMAP_BUILD_INPUT_READ_ONLY_EXT,
                MemoryLocation::CpuToGpu,
            )?
        };
        let start = alignment_padding(source.address, SOURCE_ALIGNMENT as u64) as usize;
        // SAFETY: vk::MicromapTriangleEXT is repr(C) with a u32 and two u16s
        // and no padding (8 bytes); these initialized bytes borrow `triangles`.
        let descriptors = unsafe {
            std::slice::from_raw_parts(triangles.as_ptr().cast(), size_of_val(triangles))
        };
        source.write_at(start, data);
        source.write_at(start + triangle_offset, descriptors);
        let info = info
            .data(at(source.address + start as u64))
            .triangle_array(at(source.address + (start + triangle_offset) as u64));
        // SAFETY: `source` remains live through the synchronous build. Both
        // addresses are 256-byte aligned and refer to ranges in the buffer.
        let result =
            unsafe { self.build_with_source(gpu, pool, info, sizes.micromap_size, scratch_size) };
        // SAFETY: Submission has finished or failed, so no build uses the source.
        unsafe { gpu::destroy_buffer(gpu.device, gpu.allocator, source) };
        result
    }

    unsafe fn build_with_source(
        &self,
        gpu: &mut Gpu,
        pool: &mut vk::CommandPool,
        info: vk::MicromapBuildInfoEXT<'_>,
        size: u64,
        scratch_size: u64,
    ) -> Result<Storage> {
        // SAFETY: Device and allocator match; scratch is padded for Vulkan alignment.
        let scratch = unsafe {
            gpu::create_buffer(
                gpu.device,
                gpu.allocator,
                "gi opacity micromap scratch",
                scratch_size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::GpuOnly,
            )?
        };
        let result = (|| {
            // SAFETY: Device and allocator match and `size` came from the build query.
            let storage = unsafe { self.create_storage(gpu, size) }?;
            let device_address =
                scratch.address + alignment_padding(scratch.address, self.scratch_alignment);
            let info = info
                .dst_micromap(storage.handle)
                .scratch_data(vk::DeviceOrHostAddressKHR { device_address });
            // SAFETY: Queue and pool are externally synchronized; source, scratch,
            // storage and host build descriptions remain valid through this submission.
            let built = unsafe {
                super::submit(gpu, pool, |cmd| {
                    (self.loader.fp().cmd_build_micromaps_ext)(cmd, 1, &info);
                    let barrier = vk::MemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::MICROMAP_BUILD_EXT)
                        .src_access_mask(vk::AccessFlags2::MICROMAP_WRITE_EXT)
                        .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                        .dst_access_mask(vk::AccessFlags2::MICROMAP_READ_EXT);
                    gpu.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default()
                            .memory_barriers(std::slice::from_ref(&barrier)),
                    );
                })
            };
            if let Err(error) = built {
                // SAFETY: No dependent BLAS exists and submission is no longer pending.
                unsafe { self.destroy_storage(gpu, storage) };
                return Err(error);
            }
            Ok(storage)
        })();
        // SAFETY: The build has completed or no longer uses device resources.
        unsafe { gpu::destroy_buffer(gpu.device, gpu.allocator, scratch) };
        result
    }

    unsafe fn create_storage(&self, gpu: &mut Gpu, size: u64) -> Result<Storage> {
        // SAFETY: Same-device allocator and storage usage required by EXT.
        let buffer = unsafe {
            gpu::create_buffer(
                gpu.device,
                gpu.allocator,
                "gi opacity micromap storage",
                size,
                vk::BufferUsageFlags::MICROMAP_STORAGE_EXT,
                MemoryLocation::GpuOnly,
            )?
        };
        let info = vk::MicromapCreateInfoEXT::default()
            .buffer(buffer.buffer)
            .size(size)
            .ty(vk::MicromapTypeEXT::OPACITY_MICROMAP);
        let mut handle = vk::MicromapEXT::null();
        // SAFETY: `buffer` is bound, the default offset is aligned, info and output are valid.
        let status = unsafe {
            (self.loader.fp().create_micromap_ext)(
                self.loader.device(),
                &info,
                std::ptr::null(),
                &mut handle,
            )
        };
        if let Err(error) = status.result() {
            // SAFETY: No micromap or submission references the buffer.
            unsafe { gpu::destroy_buffer(gpu.device, gpu.allocator, buffer) };
            return Err(error.into());
        }
        Ok(Storage { handle, buffer })
    }

    /// Release the map (if any), its backing allocation and the BLAS index buffer.
    ///
    /// # Safety
    /// Both resources must belong to this builder/device. All commands using
    /// them must have finished, and no dependent BLAS may be used again.
    pub(crate) unsafe fn destroy(&self, gpu: &mut Gpu, map: OpacityMicromap) {
        // SAFETY: Caller guarantees ownership and that no GPU work can use these resources.
        unsafe {
            if let Some(storage) = map.storage {
                self.destroy_storage(gpu, storage);
            }
            gpu::destroy_buffer(gpu.device, gpu.allocator, map.indices);
        }
    }

    unsafe fn destroy_storage(&self, gpu: &mut Gpu, storage: Storage) {
        // SAFETY: Handle and buffer belong to this device and no GPU work uses them.
        unsafe {
            (self.loader.fp().destroy_micromap_ext)(
                self.loader.device(),
                storage.handle,
                std::ptr::null(),
            );
            gpu::destroy_buffer(gpu.device, gpu.allocator, storage.buffer);
        }
    }
}

#[derive(Default)]
struct Encoded {
    indices: Vec<u32>,
    triangles: Vec<vk::MicromapTriangleEXT>,
    data: Vec<u8>,
    usage: Vec<vk::MicromapUsageEXT>,
}

impl Encoded {
    fn new<'a>(
        triangles: impl IntoIterator<Item = TriangleOpacity<'a>>,
        max_level: u32,
    ) -> Result<Self> {
        let mut encoded = Self::default();
        let mut counts = [0u32; LEVEL_COUNT];
        let mut triangles = triangles.into_iter().enumerate();
        let default_index = OpacityState::UnknownOpaque.special_index();
        while let Some((primitive, triangle)) = triangles.next() {
            if primitive >= u32::MAX as usize {
                return Err("too many opacity micromap primitives".into());
            }
            let index = encoded.encode_triangle(triangle, max_level, &mut counts)?;
            if encoded.indices.is_empty() {
                if index == default_index {
                    continue;
                }
                // Do not allocate until there's an effect on traversal. Once
                // there is one, emit the omitted default indices in order.
                let reserve = (primitive + 1).saturating_add(triangles.size_hint().0);
                encoded.indices.try_reserve_exact(reserve)?;
                encoded.indices.resize(primitive, default_index);
            }
            encoded.indices.try_reserve(1)?;
            encoded.indices.push(index);
        }
        encoded.usage = counts
            .into_iter()
            .enumerate()
            .filter(|&(_, count)| count != 0)
            .map(|(level, count)| {
                vk::MicromapUsageEXT::default()
                    .count(count)
                    .subdivision_level(level as u32)
                    .format(FORMAT.as_raw() as u32)
            })
            .collect();
        Ok(encoded)
    }

    fn encode_triangle(
        &mut self,
        triangle: TriangleOpacity<'_>,
        max_level: u32,
        counts: &mut [u32; LEVEL_COUNT],
    ) -> Result<u32> {
        let (level, states) = match triangle {
            TriangleOpacity::Uniform(state) => return Ok(state.special_index()),
            TriangleOpacity::Subdivided { level, states } => (level, states),
        };
        let Some(expected) = 1usize.checked_shl(2 * u32::from(level)) else {
            return Err("opacity micromap subdivision level exceeds addressable storage".into());
        };
        if states.len() != expected {
            return Err(format!(
                "opacity micromap level {level} requires {expected} states, got {}",
                states.len()
            )
            .into());
        }
        let first = states[0];
        if states[1..].iter().all(|&state| state == first) {
            return Ok(first.special_index());
        }
        if u32::from(level) > max_level {
            return Err(
                format!("opacity micromap level {level} exceeds device limit {max_level}").into(),
            );
        }
        let index = u32::try_from(self.triangles.len())?;
        if index >= OpacityState::UnknownOpaque.special_index() {
            return Err("opacity micromap triangle indices would overlap special indices".into());
        }
        let offset = u32::try_from(self.data.len())?;
        self.triangles.try_reserve(1)?;
        self.data.try_reserve(states.len().div_ceil(4))?;
        self.triangles.push(
            vk::MicromapTriangleEXT::default()
                .data_offset(offset)
                .subdivision_level(level)
                .format(FORMAT.as_raw() as u16),
        );
        counts[usize::from(level)] += 1;
        // 4^level states, LSB first, two bits per state. Level zero has
        // already been represented by a special index.
        self.data
            .extend(states.as_chunks::<4>().0.iter().map(|states| {
                states[0] as u8
                    | (states[1] as u8) << 2
                    | (states[2] as u8) << 4
                    | (states[3] as u8) << 6
            }));
        Ok(index)
    }
}

fn alignment_padding(address: vk::DeviceAddress, alignment: u64) -> u64 {
    address.wrapping_neg() & (alignment - 1)
}

fn at(device_address: vk::DeviceAddress) -> vk::DeviceOrHostAddressConstKHR {
    vk::DeviceOrHostAddressConstKHR { device_address }
}

#[cfg(test)]
mod tests {
    use super::{AlphaBounds, Encoded, OpacityState, TriangleOpacity};

    #[test]
    fn classify_inclusive_cutoff_and_inconclusive_bounds() {
        use OpacityState::{Opaque, Transparent, UnknownOpaque};
        let classify = |min, max| OpacityState::classify(Some(AlphaBounds { min, max }), 0.5);
        assert_eq!(classify(0.0, 0.499), Transparent);
        assert_eq!(classify(0.5, 0.5), Opaque);
        assert_eq!(classify(0.0, 0.5), UnknownOpaque);
        assert_eq!(classify(f32::NAN, 0.1), UnknownOpaque);
        assert_eq!(classify(0.6, 0.2), UnknownOpaque);
        assert_eq!(OpacityState::classify(None, 0.5), UnknownOpaque);
    }

    #[test]
    fn packs_four_states_and_preserves_special_indices() {
        use OpacityState::{Opaque, Transparent, UnknownOpaque, UnknownTransparent};
        use TriangleOpacity::{Subdivided, Uniform};
        let states = [Transparent, Opaque, UnknownTransparent, UnknownOpaque];
        let uniform = [Transparent; 4];
        let encoded = Encoded::new(
            [
                Uniform(UnknownOpaque),
                Subdivided {
                    level: 1,
                    states: &states,
                },
                Uniform(Opaque),
                Subdivided {
                    level: 1,
                    states: &uniform,
                },
                Uniform(UnknownTransparent),
                Subdivided {
                    level: 1,
                    states: &states,
                },
            ],
            1,
        )
        .unwrap();
        assert_eq!(
            encoded.indices,
            [u32::MAX - 3, 0, u32::MAX - 1, u32::MAX, u32::MAX - 2, 1]
        );
        assert_eq!(encoded.data, [0b1110_0100, 0b1110_0100]);
        assert_eq!(encoded.triangles[1].data_offset, 1);
        assert_eq!(encoded.usage.len(), 1);
        assert_eq!(encoded.usage[0].count, 2);
        assert_eq!(encoded.usage[0].subdivision_level, 1);
    }

    #[test]
    fn skips_all_default_and_rejects_bad_subdivision() {
        use OpacityState::{Opaque, UnknownOpaque};
        use TriangleOpacity::{Subdivided, Uniform};
        let defaults = Encoded::new([Uniform(UnknownOpaque); 8], 1).unwrap();
        assert!(defaults.indices.is_empty());
        assert!(
            Encoded::new(
                [Subdivided {
                    level: 1,
                    states: &[Opaque]
                }],
                1
            )
            .is_err()
        );
        assert!(
            Encoded::new(
                [Subdivided {
                    level: 1,
                    states: &[Opaque, Opaque, UnknownOpaque, Opaque]
                }],
                0
            )
            .is_err()
        );
    }
}
