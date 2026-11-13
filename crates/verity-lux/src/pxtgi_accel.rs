//! PXTGI's physics scene as hardware acceleration structures, for the passes
//! compiled with `GI_RAY_QUERY`: a BLAS per static mesh over its triangles in
//! the geometry words, one shared unit box for everything else, and a TLAS per
//! frame slot with one instance per packed primitive, whose custom index is
//! that primitive's. The TLAS is rebuilt in the frame's own commands whenever
//! the packed scene changes, so moving bodies cost no stall.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::rt::{Accel, Geometry, RayTracing, instance, instance_bytes};
use ash::vk;
use glam::{Affine3A, Mat4};
use gpu_allocator::MemoryLocation;
use std::{collections::HashMap, ops::Range};
use verity_render::{
    Gpu,
    extension::Result,
    gpu::{self, Buffer},
};

/// A packed primitive's TLAS instance.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Instance {
    pub world_from_local: Affine3A,
    /// A static mesh's BLAS as (first triangle word, triangle count); `None`
    /// is the unit box, which `world_from_local` scales around the primitive.
    pub blas: Option<(i32, u32)>,
}

impl Instance {
    pub fn boxed(world_from_unit: Affine3A) -> Self {
        Self {
            world_from_local: world_from_unit,
            blas: None,
        }
    }
}

#[derive(Default)]
struct Slot {
    tlas: Option<(Accel, Buffer, u32)>,
    instances: Buffer,
    revision: Option<u64>,
}

pub(crate) struct Hardware {
    rt: RayTracing,
    meshes: HashMap<(i32, u32), Accel>,
    unit_box: Accel,
    slots: [Slot; 2],
}

impl Hardware {
    pub unsafe fn new(gpu: &mut Gpu, pool: &mut vk::CommandPool) -> Result<Self> {
        let rt = RayTracing::new(gpu);
        let corners: [f32; 6] = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let (input, address) = rt.input(gpu, "pxtgi unit box", bytemuck::cast_slice(&corners))?;
        let built = rt.build(gpu, pool, &[Geometry::Aabbs { address, count: 1 }]);
        gpu::destroy_buffer(gpu.device, gpu.allocator, input);
        Ok(Self {
            rt,
            meshes: HashMap::new(),
            unit_box: built?.pop().expect("one built"),
            slots: Default::default(),
        })
    }

    /// Keep a BLAS for each wanted mesh. `rewritten` holds the geometry word
    /// ranges replaced since the last call: meshes over them are rebuilt.
    /// The GPU must be idle, as scene uploads leave it.
    pub unsafe fn sync_meshes(
        &mut self,
        gpu: &mut Gpu,
        pool: &mut vk::CommandPool,
        geometry: &[[f32; 4]],
        wanted: impl IntoIterator<Item = (i32, u32)>,
        rewritten: &[Range<usize>],
    ) -> Result<()> {
        let wanted: std::collections::HashSet<_> =
            wanted.into_iter().filter(|&(_, count)| count > 0).collect();
        let stale = |&(at, count): &(i32, u32)| {
            let words = at as usize..at as usize + count as usize * 3;
            !wanted.contains(&(at, count))
                || rewritten.iter().any(|r| r.start < words.end && words.start < r.end)
        };
        let dropped: Vec<_> = self.meshes.keys().filter(|k| stale(k)).copied().collect();
        for key in &dropped {
            let blas = self.meshes.remove(key).expect("listed above");
            self.rt.destroy(gpu, blas);
        }
        let mut missing: Vec<_> = (wanted.into_iter())
            .filter(|key| !self.meshes.contains_key(key))
            .collect();
        if missing.is_empty() && dropped.is_empty() {
            return Ok(());
        }
        // Instances point at BLAS addresses: every slot rebuilds its TLAS.
        for slot in &mut self.slots {
            slot.revision = None;
        }
        if missing.is_empty() {
            return Ok(());
        }
        missing.sort_unstable();
        // The corners of every missing mesh in one input, three to a triangle.
        let mut corners: Vec<[f32; 4]> = Vec::new();
        let mut offsets = Vec::with_capacity(missing.len());
        for &(at, count) in &missing {
            offsets.push(corners.len() as u64);
            corners.extend_from_slice(&geometry[at as usize..at as usize + count as usize * 3]);
        }
        let (input, address) =
            (self.rt).input(gpu, "pxtgi blas corners", bytemuck::cast_slice(&corners))?;
        let geometries: Vec<_> = (missing.iter().zip(&offsets))
            .map(|(&(_, count), &offset)| Geometry::Triangles {
                vertices: address + offset * 16,
                stride: 16,
                max_vertex: count * 3 - 1,
                indices: 0,
                count,
                opaque: true,
            })
            .collect();
        let built = self.rt.build(gpu, pool, &geometries);
        gpu::destroy_buffer(gpu.device, gpu.allocator, input);
        self.meshes.extend(missing.into_iter().zip(built?));
        Ok(())
    }

    pub fn tlas(&self, slot: usize) -> Option<vk::AccelerationStructureKHR> {
        self.slots[slot].tlas.as_ref().map(|(tlas, _, _)| tlas.handle)
    }

    /// Bring a slot's TLAS up to the packed instances, recording the build into
    /// `cmd` ahead of the passes. The slot's last frame must be done. Returns
    /// whether the TLAS was replaced, so its descriptor needs writing.
    pub unsafe fn record(
        &mut self,
        gpu: &mut Gpu,
        cmd: vk::CommandBuffer,
        slot: usize,
        instances: &[Instance],
        revision: u64,
    ) -> Result<bool> {
        let state = &mut self.slots[slot];
        if state.tlas.is_some() && state.revision == Some(revision) {
            return Ok(false);
        }
        let records: Vec<_> = (instances.iter().enumerate())
            .filter_map(|(index, item)| {
                let blas = match item.blas {
                    Some(key) => self.meshes.get(&key)?.address,
                    None => self.unit_box.address,
                };
                let m = Mat4::from(item.world_from_local);
                let rows = [0, 1, 2].map(|i| m.row(i).to_array());
                Some(instance(bytemuck::cast(rows), index as u32, blas))
            })
            .collect();
        let count = records.len() as u32;
        let mut replaced = false;
        if state.tlas.as_ref().is_none_or(|&(_, _, capacity)| capacity < count) {
            if let Some((tlas, scratch, _)) = state.tlas.take() {
                self.rt.destroy(gpu, tlas);
                gpu::destroy_buffer(gpu.device, gpu.allocator, scratch);
            }
            let capacity = (count * 3 / 2).max(64);
            let (tlas, scratch) = self.rt.tlas_target(gpu, capacity)?;
            state.tlas = Some((tlas, scratch, capacity));
            replaced = true;
        }
        let bytes = instance_bytes(&records);
        let size = bytes.len().max(64) as u64;
        if state.instances.buffer == vk::Buffer::null() || state.instances.size < size {
            gpu::destroy_buffer(gpu.device, gpu.allocator, std::mem::take(&mut state.instances));
            state.instances = gpu::create_buffer(
                gpu.device,
                gpu.allocator,
                "pxtgi tlas instances",
                size * 3 / 2,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryLocation::CpuToGpu,
            )?;
        }
        if !bytes.is_empty() {
            state.instances.write(bytes);
        }
        let (tlas, scratch, _) = state.tlas.as_ref().expect("made above");
        let address = crate::rt::address(gpu.device, &state.instances);
        self.rt.record_tlas(gpu, cmd, tlas, scratch, address, count);
        state.revision = Some(revision);
        Ok(replaced)
    }

    /// # Safety
    /// The GPU must be done with them.
    pub unsafe fn destroy(mut self, gpu: &mut Gpu) {
        for (_, blas) in self.meshes.drain() {
            self.rt.destroy(gpu, blas);
        }
        self.rt.destroy(gpu, self.unit_box);
        for slot in self.slots {
            if let Some((tlas, scratch, _)) = slot.tlas {
                self.rt.destroy(gpu, tlas);
                gpu::destroy_buffer(gpu.device, gpu.allocator, scratch);
            }
            gpu::destroy_buffer(gpu.device, gpu.allocator, slot.instances);
        }
    }
}
