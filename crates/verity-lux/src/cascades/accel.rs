//! The GI scene as hardware acceleration structures, for the passes compiled
//! with `GI_RAY_QUERY`: a BLAS per triangle span the instances use, read from
//! the packed geometry words, and a TLAS over the packed instances, whose
//! custom index is the instance's index in the scene words. Except for
//! proven opaque hits and transparent micromap cells, `traversal.slang`
//! re-tests every candidate against those words.

#![allow(unsafe_op_in_unsafe_fn)]

use super::scene::{FLAG_DOUBLE_SIDED, GiWords, InstanceIdentity, LAYER_MASK};
use crate::rt::{
    Accel, ClusterBlas, Geometry, RayTracing, instance_bytes, instance_with_flags,
    micromap::{AlphaBounds, Micromaps, OpacityMicromap, OpacityState, TriangleOpacity},
    partition::{self, PartitionInstance, PartitionedTlas},
};
use crate::tier::HardwareProfile;
use ash::vk::{self, GeometryInstanceFlagsKHR as InstanceFlags};
use glam::UVec4;
use std::{collections::HashMap, ops::Range};
use verity_render::{
    Gpu,
    asset::{AlphaMode, SceneData},
    extension::Result,
    gpu,
};

#[derive(Clone, Copy)]
pub(super) struct RayOptions {
    cluster: bool,
    partitioned: bool,
    dgf: bool,
    micromaps: bool,
    opaque: bool,
    diagnostics: bool,
}

impl RayOptions {
    /// The hardware builds a device's profile chose, environment overrides
    /// included.
    pub(super) fn from_profile(profile: &HardwareProfile) -> Self {
        let builds = profile.builds;
        Self {
            cluster: builds.cluster_blas,
            partitioned: builds.partitioned_tlas,
            dgf: builds.dgf_blas,
            micromaps: builds.opacity_micromaps,
            opaque: builds.opaque_rays,
            diagnostics: std::env::var_os("VERITY_LUX_ACCEL_DIAGNOSTICS")
                .is_some_and(|value| value == "1"),
        }
    }

    fn for_gpu(self, gpu: &Gpu) -> Self {
        let features = gpu.ray_features;
        Self {
            cluster: gpu.ray_query && features.cluster_acceleration && self.cluster,
            partitioned: gpu.ray_query && features.partitioned_acceleration && self.partitioned,
            dgf: gpu.ray_query && features.dense_geometry && self.dgf,
            micromaps: gpu.ray_query && features.opacity_micromap && self.micromaps,
            diagnostics: self.diagnostics,
            opaque: gpu.ray_query && self.opaque,
        }
    }
}

#[derive(Clone, Copy)]
struct Mask {
    factor: f32,
    cutoff: f32,
}

impl Mask {
    fn triangle(self, geometry: PackedGeometry<'_>, indices: UVec4) -> OpacityState {
        let [a, b, c] = indices.truncate().to_array().map(|i| geometry.alpha(i));
        if ![a, b, c].into_iter().all(|v| (0.0..=1.0).contains(&v)) {
            return OpacityState::UnknownOpaque;
        }
        // For accepted software intersections the edge weights have one sign
        // and a finite nonzero sum. Bound the convex interpolation, allowing
        // for division, reconstruction of the first weight, and interpolation
        // and factor rounding (all inputs have magnitude at most one).
        const ROUNDING: f32 = 64.0 * f32::EPSILON;
        let bounds = AlphaBounds {
            min: a.min(b).min(c) * self.factor - ROUNDING,
            max: a.max(b).max(c) * self.factor + ROUNDING,
        };
        if OpacityState::classify(Some(bounds), self.cutoff) == OpacityState::Transparent {
            OpacityState::Transparent
        } else {
            // Even known opaque alpha must still go through the shader's other
            // hit checks. Using unknown here works for both KHR TLAS and PTLAS.
            OpacityState::UnknownOpaque
        }
    }
}

#[derive(Clone, Copy)]
enum RayMaterial {
    Opaque,
    Mask(Mask),
}

#[derive(Default)]
pub(super) struct RayMaterials(HashMap<u32, RayMaterial>);

impl RayMaterials {
    pub(super) fn add(
        &mut self,
        gpu: &Gpu,
        options: RayOptions,
        scene: &SceneData,
        material_base: u32,
    ) {
        let options = options.for_gpu(gpu);
        if !(options.opaque || options.micromaps) {
            return;
        }
        for (i, material) in scene.materials.iter().enumerate() {
            let property = match material.alpha_mode {
                AlphaMode::Opaque if options.opaque && material.double_sided => RayMaterial::Opaque,
                AlphaMode::Mask
                    if options.micromaps
                        && material.base_color_texture.is_none()
                        && material.alpha_cutoff.is_finite()
                        && material.alpha_cutoff > 0.0
                        && (0.0..=1.0).contains(&material.base_color[3]) =>
                {
                    RayMaterial::Mask(Mask {
                        factor: material.base_color[3],
                        cutoff: material.alpha_cutoff,
                    })
                }
                _ => continue,
            };
            // Texture previews, blend coverage, and retained materials don't
            // prove a bound for all shader sampling paths.
            self.0.insert(material_base + i as u32, property);
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct BlasKey {
    first: u32,
    count: u32,
    /// Material is part of the key only when it affects the built geometry.
    mask_material: Option<u32>,
}

impl BlasKey {
    fn triangles(self) -> Range<usize> {
        let first = self.first as usize + 1;
        first..first + self.count as usize
    }
}

struct BuildPolicy {
    static_geometry: bool,
    mask: Option<Mask>,
}

enum Blas {
    Standard(Accel),
    Dgf(Accel),
    Cluster(ClusterBlas),
    Micromapped(Accel, Box<OpacityMicromap>),
}

impl Blas {
    fn address(&self) -> vk::DeviceAddress {
        match self {
            Self::Standard(accel) | Self::Dgf(accel) | Self::Micromapped(accel, _) => accel.address,
            Self::Cluster(accel) => accel.address(),
        }
    }
}

#[derive(Clone, Copy)]
struct PackedGeometry<'a> {
    words: &'a [UVec4],
    vertex_base: usize,
    vertex_stride: usize,
    vertex_count: u32,
}

impl<'a> PackedGeometry<'a> {
    fn new(words: &'a [UVec4]) -> Self {
        let [_, base, stride, count] = words[0].to_array();
        Self {
            words,
            vertex_base: base as usize,
            vertex_stride: stride as usize,
            vertex_count: count,
        }
    }

    fn vertex(self, i: u32) -> usize {
        self.vertex_base + i as usize * self.vertex_stride
    }

    fn alpha(self, i: u32) -> f32 {
        // Triangle color is the fifth word of a vertex, matching material.slang.
        f32::from_bits(self.words[self.vertex(i) + 4].w)
    }

    fn triangles(self, key: BlasKey) -> &'a [UVec4] {
        &self.words[key.triangles()]
    }
}

struct SceneInstance<'a> {
    words: &'a [UVec4],
    at: usize,
    index: u32,
    id: InstanceIdentity,
}

impl SceneInstance<'_> {
    fn iter(snapshot: &GiWords) -> impl Iterator<Item = SceneInstance<'_>> + Clone {
        let words = &snapshot.scene_words;
        let [_, base, count, stride] = words.first().map_or([0; 4], |word| word.to_array());
        (0..count).map(move |index| SceneInstance {
            words,
            at: (base + index * stride) as usize,
            index,
            id: snapshot.instance_ids[index as usize],
        })
    }

    fn is_static(&self) -> bool {
        matches!(self.id, InstanceIdentity::Static(_))
    }

    fn material(&self, materials: &RayMaterials) -> Option<RayMaterial> {
        if self.is_static() {
            materials.0.get(&self.words[self.at + 7].w).copied()
        } else {
            None
        }
    }

    fn key(&self, materials: &RayMaterials) -> BlasKey {
        let span = self.words[self.at + 8];
        BlasKey {
            first: span.y,
            count: span.z,
            mask_material: matches!(self.material(materials), Some(RayMaterial::Mask(_)))
                .then(|| self.words[self.at + 7].w),
        }
    }

    fn flags(&self, materials: &RayMaterials, opaque: bool) -> InstanceFlags {
        let record = self.words[self.at + 6];
        // frame_lights and params.layers in cascades/mod.rs always give the
        // shader exactly LAYER_MASK. Don't bypass its visibility or culling
        // checks unless this instance is guaranteed visible and two-sided.
        let eligible = opaque
            && record.w & (FLAG_DOUBLE_SIDED | 3) == FLAG_DOUBLE_SIDED
            && record.y == 1
            && self.words[record.x as usize] == LAYER_MASK
            && matches!(self.material(materials), Some(RayMaterial::Opaque));
        InstanceFlags::TRIANGLE_FACING_CULL_DISABLE
            | if eligible {
                InstanceFlags::FORCE_OPAQUE
            } else {
                InstanceFlags::empty()
            }
    }

    fn transform(&self) -> [f32; 12] {
        // Words 3..6 are world_from_local's rows, as Vulkan expects.
        std::array::from_fn(|i| f32::from_bits(self.words[self.at + 3 + i / 4][i % 4]))
    }
}

struct RetainedId {
    stable: u64,
    shader_index: u32,
}

struct Partitioned {
    tlas: Option<PartitionedTlas>,
    retained: HashMap<u64, RetainedId>,
    next_id: u64,
}

impl Partitioned {
    fn stable_id(&mut self, id: InstanceIdentity, shader_index: u32) -> u64 {
        match id {
            InstanceIdentity::Static(load_index) => u64::from(load_index),
            InstanceIdentity::Retained(key) => {
                let id = self.retained.entry(key).or_insert_with(|| {
                    let stable = self.next_id;
                    self.next_id += 1;
                    RetainedId {
                        stable,
                        shader_index,
                    }
                });
                id.shader_index = shader_index;
                id.stable
            }
        }
    }
}

#[expect(
    clippy::large_enum_variant,
    reason = "one TLAS per scene; storing it inline avoids another allocation"
)]
enum Tlas {
    Khr(Option<Accel>),
    Partitioned(Partitioned),
}

pub(super) struct SceneAccel {
    rt: RayTracing,
    options: RayOptions,
    micromaps: Option<Micromaps>,
    blas: HashMap<BlasKey, Blas>,
    tlas: Tlas,
    opaque_instances: bool,
}

impl SceneAccel {
    pub(super) unsafe fn new(gpu: &Gpu, options: RayOptions) -> Self {
        let options = options.for_gpu(gpu);
        Self {
            rt: RayTracing::new(gpu),
            options,
            micromaps: options.micromaps.then(|| Micromaps::new(gpu)),
            blas: HashMap::new(),
            tlas: if options.partitioned {
                Tlas::Partitioned(Partitioned {
                    tlas: None,
                    retained: HashMap::new(),
                    next_id: 1 << 32,
                })
            } else {
                Tlas::Khr(None)
            },
            opaque_instances: false,
        }
    }

    pub(super) fn descriptor_type(&self) -> vk::DescriptorType {
        match &self.tlas {
            Tlas::Khr(_) => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            Tlas::Partitioned(_) => partition::DESCRIPTOR_TYPE,
        }
    }

    pub(super) fn khr_tlas(&self) -> Option<vk::AccelerationStructureKHR> {
        match &self.tlas {
            Tlas::Khr(Some(tlas)) => Some(tlas.handle),
            _ => None,
        }
    }

    pub(super) unsafe fn write_partitioned(
        &self,
        device: &ash::Device,
        set: vk::DescriptorSet,
        binding: u32,
    ) {
        if let Tlas::Partitioned(partitioned) = &self.tlas {
            partitioned
                .tlas
                .as_ref()
                .expect("partitioned TLAS built before descriptor writes")
                .write_descriptor(device, set, binding);
        }
    }

    pub(super) fn opaque_instances(&self) -> bool {
        self.opaque_instances
    }

    /// Apply the packed scene snapshot, returning whether the descriptor changed.
    /// `changed` holds the geometry word ranges rewritten, or `None` for a full upload.
    /// The queue must be done with the previous snapshot.
    pub(super) unsafe fn update(
        &mut self,
        gpu: &mut Gpu,
        pool: &mut vk::CommandPool,
        words: &GiWords,
        materials: &RayMaterials,
        changed: Option<&[Range<usize>]>,
    ) -> Result<bool> {
        let instances = SceneInstance::iter(words);
        let mut used = HashMap::new();
        for instance in instances.clone() {
            let key = instance.key(materials);
            used.entry(key).or_insert_with(|| BuildPolicy {
                static_geometry: instance.is_static(),
                mask: match instance.material(materials) {
                    Some(RayMaterial::Mask(mask)) => Some(mask),
                    _ => None,
                },
            });
        }
        let mut missing: Vec<_> = used
            .keys()
            .copied()
            .filter(|key| {
                !self.blas.contains_key(key)
                    || changed.is_none_or(|ranges| {
                        let triangles = key.triangles();
                        ranges
                            .iter()
                            .any(|r| r.start < triangles.end && triangles.start < r.end)
                    })
            })
            .collect();
        missing.sort_unstable();

        let mut replacements = HashMap::with_capacity(missing.len());
        let result = (|| {
            if !missing.is_empty() {
                self.build_blas(
                    gpu,
                    pool,
                    PackedGeometry::new(&words.geometry_words),
                    missing,
                    &used,
                    &mut replacements,
                )?;
            }
            let mut opaque_instances = false;
            let records = instances.map(|instance| {
                let key = instance.key(materials);
                let blas = replacements.get(&key).unwrap_or_else(|| &self.blas[&key]);
                let flags = instance.flags(materials, self.options.opaque);
                opaque_instances |= flags.contains(InstanceFlags::FORCE_OPAQUE);
                (
                    instance.id,
                    instance.index,
                    instance.transform(),
                    blas.address(),
                    flags,
                )
            });
            // The old BLASes remain valid through the partitioned update, which
            // may read the previous PTLAS with its old BLAS references.
            let descriptor_changed = match &mut self.tlas {
                Tlas::Khr(target) => {
                    let records: Vec<_> = records
                        .map(|(_, index, transform, blas, flags)| {
                            instance_with_flags(transform, index, blas, flags)
                        })
                        .collect();
                    let (input, address) =
                        self.rt
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
                    let tlas = tlas?.pop().expect("one TLAS geometry submitted");
                    if let Some(old) = target.replace(tlas) {
                        self.rt.destroy(gpu, old);
                    }
                    true
                }
                Tlas::Partitioned(partitioned) => {
                    let records: Vec<_> = records
                        .map(|(id, index, transform, blas, flags)| PartitionInstance {
                            stable_id: partitioned.stable_id(id, index),
                            shader_index: index,
                            transform,
                            blas,
                            mask: 0xff,
                            flags,
                        })
                        .collect();
                    let descriptor_changed = if let Some(tlas) = &mut partitioned.tlas {
                        tlas.update(gpu, pool, &records)?
                    } else {
                        partitioned.tlas = PartitionedTlas::build(gpu, pool, &records)?;
                        assert!(
                            partitioned.tlas.is_some(),
                            "PTLAS capability checked on creation"
                        );
                        true
                    };
                    partitioned.retained.retain(|&key, id| {
                        words.instance_ids.get(id.shader_index as usize)
                            == Some(&InstanceIdentity::Retained(key))
                    });
                    descriptor_changed
                }
            };
            self.opaque_instances = opaque_instances;
            Ok(descriptor_changed)
        })();
        match result {
            Ok(descriptor_changed) => {
                for (_, old) in self
                    .blas
                    .extract_if(|key, _| !used.contains_key(key) || replacements.contains_key(key))
                {
                    Self::destroy_blas(&self.rt, self.micromaps.as_ref(), gpu, old);
                }
                self.blas.extend(replacements);
                if self.options.diagnostics {
                    let [mut triangles, mut dgf, mut clusters, mut micromaps] = [0; 4];
                    for blas in self.blas.values() {
                        match blas {
                            Blas::Standard(_) => triangles += 1,
                            Blas::Dgf(_) => dgf += 1,
                            Blas::Cluster(_) => clusters += 1,
                            Blas::Micromapped(..) => micromaps += 1,
                        }
                    }
                    eprintln!(
                        "Lux acceleration: PTLAS={} descriptor_changed={descriptor_changed} \
                         triangles={triangles} dgf={dgf} clusters={clusters} micromaps={micromaps} \
                         opaque={}",
                        self.options.partitioned, self.opaque_instances,
                    );
                }
                Ok(descriptor_changed)
            }
            Err(error) => {
                for replacement in replacements.into_values() {
                    Self::destroy_blas(&self.rt, self.micromaps.as_ref(), gpu, replacement);
                }
                Err(error)
            }
        }
    }

    unsafe fn build_blas(
        &self,
        gpu: &mut Gpu,
        pool: &mut vk::CommandPool,
        geometry: PackedGeometry<'_>,
        missing: Vec<BlasKey>,
        used: &HashMap<BlasKey, BuildPolicy>,
        built: &mut HashMap<BlasKey, Blas>,
    ) -> Result<()> {
        let positions: Vec<[f32; 3]> = (0..geometry.vertex_count)
            .map(|i| {
                geometry.words[geometry.vertex(i)]
                    .truncate()
                    .to_array()
                    .map(f32::from_bits)
            })
            .collect();
        let conventional_only = !self.options.cluster && !self.options.dgf;
        let mut indices = Vec::with_capacity(if conventional_only {
            missing.iter().map(|key| key.count as usize).sum()
        } else {
            0
        });
        let mut conventional =
            Vec::with_capacity(if conventional_only { missing.len() } else { 0 });
        for key in missing {
            let offset = indices.len();
            let triangles = geometry.triangles(key);
            indices.extend(triangles.iter().map(|word| word.truncate().to_array()));
            let policy = &used[&key];
            let mask = policy.mask.filter(|mask| {
                triangles
                    .iter()
                    .any(|&triangle| mask.triangle(geometry, triangle) == OpacityState::Transparent)
            });
            // Retained meshes are edited repeatedly; only static inputs use
            // the opt-in encoders. A useful micromap takes precedence.
            if policy.static_geometry && mask.is_none() {
                let mesh_indices = &indices[offset..];
                if self.options.dgf
                    && let Some(dgf) =
                        self.rt
                            .try_build_dgf(gpu, pool, &positions, mesh_indices.as_flattened())?
                {
                    built.insert(key, Blas::Dgf(dgf));
                    indices.truncate(offset);
                    continue;
                }
                if self.options.cluster {
                    let cluster = ClusterBlas::build(gpu, pool, &positions, mesh_indices)?;
                    built.insert(key, Blas::Cluster(cluster));
                    indices.truncate(offset);
                    continue;
                }
            }
            conventional.push((key, offset, mask));
        }
        if conventional.is_empty() {
            return Ok(());
        }

        let (positions, vertex_address) =
            self.rt
                .input(gpu, "gi blas positions", bytemuck::cast_slice(&positions))?;
        let result = self
            .rt
            .input(gpu, "gi blas indices", bytemuck::cast_slice(&indices))
            .and_then(|(indices, index_address)| {
                let result = (|| {
                    let mut standard_keys = Vec::new();
                    let mut standard = Vec::new();
                    for (key, offset, mask) in conventional {
                        let triangles = Geometry::Triangles {
                            vertices: vertex_address,
                            stride: 12,
                            max_vertex: geometry.vertex_count - 1,
                            indices: index_address + offset as u64 * 12,
                            count: key.count,
                            opaque: false,
                        };
                        if let (Some(builder), Some(mask)) = (&self.micromaps, mask)
                            && let Some(map) = builder.build(
                                gpu,
                                pool,
                                geometry.triangles(key).iter().map(|&triangle| {
                                    TriangleOpacity::Uniform(mask.triangle(geometry, triangle))
                                }),
                            )?
                        {
                            match map.attachment(key.count).and_then(|attachment| {
                                self.rt.build_micromapped(gpu, pool, triangles, attachment)
                            }) {
                                Ok(accel) => {
                                    built.insert(key, Blas::Micromapped(accel, Box::new(map)));
                                }
                                Err(error) => {
                                    builder.destroy(gpu, map);
                                    return Err(error);
                                }
                            }
                        } else {
                            standard_keys.push(key);
                            standard.push(triangles);
                        }
                    }
                    if !standard.is_empty() {
                        let accels = self.rt.build(gpu, pool, &standard)?;
                        built.extend(
                            standard_keys
                                .into_iter()
                                .zip(accels.into_iter().map(Blas::Standard)),
                        );
                    }
                    Ok(())
                })();
                gpu::destroy_buffer(gpu.device, gpu.allocator, indices);
                result
            });
        gpu::destroy_buffer(gpu.device, gpu.allocator, positions);
        result
    }

    unsafe fn destroy_blas(
        rt: &RayTracing,
        micromaps: Option<&Micromaps>,
        gpu: &mut Gpu,
        blas: Blas,
    ) {
        match blas {
            Blas::Standard(accel) | Blas::Dgf(accel) => rt.destroy(gpu, accel),
            Blas::Cluster(accel) => accel.destroy(gpu),
            Blas::Micromapped(accel, map) => {
                rt.destroy(gpu, accel);
                micromaps
                    .expect("builder owns this micromap")
                    .destroy(gpu, *map);
            }
        }
    }

    /// # Safety
    /// The GPU must be done with the scene.
    pub(super) unsafe fn destroy(self, gpu: &mut Gpu) {
        match self.tlas {
            Tlas::Khr(Some(tlas)) => self.rt.destroy(gpu, tlas),
            Tlas::Partitioned(Partitioned {
                tlas: Some(tlas), ..
            }) => tlas.destroy(gpu),
            _ => {}
        }
        for blas in self.blas.into_values() {
            Self::destroy_blas(&self.rt, self.micromaps.as_ref(), gpu, blas);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opacity_map_keeps_uncertain_candidates_and_respects_vertex_alpha() {
        let mut words = [UVec4::ZERO; 16];
        words[0] = UVec4::new(1, 1, 5, 3);
        for (word, alpha) in [5, 10, 15].into_iter().zip([0.2_f32, 0.3, 0.4]) {
            words[word].w = alpha.to_bits();
        }
        let indices = UVec4::new(0, 1, 2, 0);
        let masked = |cutoff| Mask {
            factor: 1.0,
            cutoff,
        };
        let geometry = PackedGeometry::new(&words);
        assert_eq!(
            masked(0.5).triangle(geometry, indices),
            OpacityState::Transparent
        );
        assert_eq!(
            masked(0.4).triangle(geometry, indices),
            OpacityState::UnknownOpaque
        );
        assert_eq!(
            masked(0.1).triangle(geometry, indices),
            OpacityState::UnknownOpaque
        );
        words[10].w = f32::NAN.to_bits();
        assert_eq!(
            masked(0.5).triangle(PackedGeometry::new(&words), indices),
            OpacityState::UnknownOpaque,
        );
    }

    #[test]
    fn partition_id_is_separate_from_shader_index_and_key_namespace() {
        let mut partitioned = Partitioned {
            tlas: None,
            retained: HashMap::new(),
            next_id: 1 << 32,
        };
        let retained = InstanceIdentity::Retained(7);
        let stable = partitioned.stable_id(retained, 0);
        assert_ne!(
            stable,
            partitioned.stable_id(InstanceIdentity::Static(7), 1)
        );
        assert_ne!(
            stable,
            partitioned.stable_id(InstanceIdentity::Retained(u64::MAX), 2)
        );
        assert_eq!(stable, partitioned.stable_id(retained, 3));
    }
}
