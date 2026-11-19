//! The GI scene as hardware acceleration structures, for the passes compiled
//! with `GI_RAY_QUERY`: a BLAS per triangle span the instances use, read from
//! the packed geometry words, and a TLAS over the packed instances, whose
//! custom index is the instance's index in the scene words. `traversal.slang`
//! re-tests every candidate against the words, so the hardware only replaces
//! the BVH walk.

#![allow(unsafe_op_in_unsafe_fn)]

use super::scene::GiWords;
use crate::rt::{Accel, Geometry, RayTracing, instance, instance_bytes};
use ash::vk;
use std::{
    collections::{HashMap, HashSet},
    ops::Range,
};
use verity_render::{Gpu, extension::Result, gpu};

pub(crate) struct SceneAccel {
    rt: RayTracing,
    /// By (first triangle, triangle count): instances of one mesh share it.
    blas: HashMap<(u32, u32), Accel>,
    tlas: Option<Accel>,
}

impl SceneAccel {
    pub unsafe fn new(gpu: &Gpu) -> Self {
        Self {
            rt: RayTracing::new(gpu),
            blas: HashMap::new(),
            tlas: None,
        }
    }

    pub fn tlas(&self) -> vk::AccelerationStructureKHR {
        self.tlas
            .as_ref()
            .map_or(vk::AccelerationStructureKHR::null(), |t| t.handle)
    }

    /// Follow the packed words. `changed` holds the geometry word ranges
    /// rewritten since the last call, or is `None` when all of them were.
    /// Only BLASes over changed triangles are rebuilt; the TLAS always is.
    pub unsafe fn update(
        &mut self,
        gpu: &mut Gpu,
        pool: &mut vk::CommandPool,
        words: &GiWords,
        changed: Option<&[Range<usize>]>,
    ) -> Result<()> {
        let (geometry, scene) = (&words.geometry_words, &words.scene_words);
        // Nothing packed yet is an empty scene.
        let [_, base, count, stride] = scene.first().map_or([0; 4], |w| w.to_array());
        // (first word, BLAS key) per instance.
        let instances: Vec<(usize, (u32, u32))> = (0..count)
            .map(|i| {
                let at = (base + i * stride) as usize;
                (at, (scene[at + 8].y, scene[at + 8].z))
            })
            .collect();
        let used: HashSet<(u32, u32)> = instances.iter().map(|&(_, key)| key).collect();
        let stale = |&(first, count): &(u32, u32)| {
            let triangles = 1 + first as usize..1 + (first + count) as usize;
            changed.is_none_or(|ranges| {
                (ranges.iter()).any(|r| r.start < triangles.end && triangles.start < r.end)
            })
        };
        let dropped: Vec<_> = (self.blas.keys())
            .filter(|key| !used.contains(key) || stale(key))
            .copied()
            .collect();
        for key in dropped {
            let blas = self.blas.remove(&key).expect("listed above");
            self.rt.destroy(gpu, blas);
        }
        let mut missing: Vec<_> = (used.into_iter())
            .filter(|key| !self.blas.contains_key(key))
            .collect();
        missing.sort_unstable();
        if !missing.is_empty() {
            self.build_blas(gpu, pool, geometry, &missing)?;
        }

        let records: Vec<_> = (instances.iter().enumerate())
            .map(|(index, &(at, key))| {
                // Words 3..6 are world_from_local's rows, as the TLAS wants them.
                let rows: Vec<f32> = (scene[at + 3..at + 6].iter())
                    .flat_map(|row| row.to_array().map(f32::from_bits))
                    .collect();
                instance(
                    rows.try_into().expect("three rows"),
                    index as u32,
                    self.blas[&key].address,
                )
            })
            .collect();
        let (input, address) = self
            .rt
            .input(gpu, "gi tlas instances", instance_bytes(&records))?;
        let tlas = self.rt.build(
            gpu,
            pool,
            &[Geometry::Instances {
                address,
                count: records.len() as u32,
            }],
        );
        gpu::destroy_buffer(gpu.device, gpu.allocator, input);
        let tlas = tlas?.pop().expect("one built");
        if let Some(old) = self.tlas.replace(tlas) {
            self.rt.destroy(gpu, old);
        }
        Ok(())
    }

    unsafe fn build_blas(
        &mut self,
        gpu: &mut Gpu,
        pool: &mut vk::CommandPool,
        geometry: &[glam::UVec4],
        spans: &[(u32, u32)],
    ) -> Result<()> {
        let [_, vertex_base, vertex_words, vertices] = geometry[0].to_array();
        // shortcut: sends every position and index for any change; send only
        // the changed spans if retained edits ever make this slow.
        let positions: Vec<[u32; 3]> = (0..vertices)
            .map(|v| {
                geometry[(vertex_base + v * vertex_words) as usize]
                    .truncate()
                    .to_array()
            })
            .collect();
        let indices: Vec<[u32; 3]> = (geometry[1..vertex_base as usize].iter())
            .map(|w| w.truncate().to_array())
            .collect();
        let (positions, vertex_address) =
            self.rt
                .input(gpu, "gi blas positions", bytemuck::cast_slice(&positions))?;
        let built = self
            .rt
            .input(gpu, "gi blas indices", bytemuck::cast_slice(&indices))
            .and_then(|(indices, index_address)| {
                let geometries: Vec<_> = (spans.iter())
                    .map(|&(first, count)| Geometry::Triangles {
                        vertices: vertex_address,
                        stride: 12,
                        max_vertex: vertices.saturating_sub(1),
                        indices: index_address + u64::from(first) * 12,
                        count,
                        opaque: false,
                    })
                    .collect();
                let built = self.rt.build(gpu, pool, &geometries);
                gpu::destroy_buffer(gpu.device, gpu.allocator, indices);
                built
            });
        gpu::destroy_buffer(gpu.device, gpu.allocator, positions);
        self.blas.extend(spans.iter().copied().zip(built?));
        Ok(())
    }

    /// # Safety
    /// The GPU must be done with them.
    pub unsafe fn destroy(mut self, gpu: &mut Gpu) {
        for (_, blas) in self.blas.drain() {
            self.rt.destroy(gpu, blas);
        }
        if let Some(tlas) = self.tlas.take() {
            self.rt.destroy(gpu, tlas);
        }
    }
}
