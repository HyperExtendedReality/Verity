//! Moving loaded scenes onto the GPU: vertices in a shared device-local
//! buffer, per-frame indices, and textures with full mip chains.

use crate::{
    asset::{self, AlphaMode, LoadOptions, TextureData, TextureFormat},
    failure, gpu,
    scene::{self, GpuMaterial, Vertex},
    LoadedMaterial, Renderer, Result,
};
use ash::vk;
use gpu_allocator::MemoryLocation;
use rayon::prelude::*;
use std::{
    collections::hash_map::DefaultHasher,
    fs,
    hash::{Hash, Hasher},
    mem,
    path::Path,
};
use verity_matrix::cluster_lod::{ClusterLod, Source, SurfaceSample, COOK_VERSION};

/// Smaller primitives cook faster than a cached file repays.
const CACHED_TRIANGLES: usize = 100_000;

/// What a [`Renderer::load_gltf`] call added.
#[derive(Clone, Copy, Debug, Default)]
pub struct SceneStats {
    pub draws: usize,
    pub triangles: usize,
    pub textures: usize,
    pub lights: usize,
}

/// Build each primitive's Matrix LOD. Hidden materials never draw.
fn cook_primitives(loaded: &asset::SceneData, positions: &[[f32; 3]]) -> Vec<Option<ClusterLod>> {
    let normals: Vec<_> = loaded
        .vertices
        .iter()
        .map(|v| [v.normal[0], v.normal[1], v.normal[2]])
        .collect();
    let uvs: Vec<_> = loaded
        .vertices
        .iter()
        .map(|v| [v.position[3], v.normal[3]])
        .collect();
    (0..loaded.primitives.len())
        .into_par_iter()
        .map(|index| {
            let primitive = &loaded.primitives[index];
            let material = &loaded.materials[primitive.material as usize];
            if material.alpha_mode == AlphaMode::Blend {
                return None;
            }
            let start = primitive.first_index as usize;
            let end = start + primitive.index_count as usize;
            let vertices = primitive.base_vertex as usize
                ..loaded
                    .primitives
                    .get(index + 1)
                    .map_or(positions.len(), |next| next.base_vertex as usize);
            let indices = &loaded.indices[start..end];
            let local = &positions[vertices.clone()];
            let masked = material.alpha_mode == AlphaMode::Mask;
            let preview = material
                .base_color_texture
                .and_then(|t| loaded.textures[t as usize].preview.as_ref());
            let colors = &loaded.vertices[vertices.clone()];
            let surface = |triangle: [u32; 3], weights: [f32; 3]| {
                let mix = |value: &dyn Fn(usize) -> f32| {
                    (0..3)
                        .map(|corner| value(triangle[corner] as usize) * weights[corner])
                        .sum::<f32>()
                };
                let vertex = [0, 1, 2, 3].map(|c| mix(&|v| colors[v].color[c]));
                let uv = [mix(&|v| colors[v].position[3]), mix(&|v| colors[v].normal[3])];
                let texel = preview.map_or([1.0; 4], |preview| preview.sample(uv));
                let alpha = material.base_color[3] * vertex[3] * texel[3];
                SurfaceSample {
                    color: [0, 1, 2].map(|c| material.base_color[c] * vertex[c] * texel[c]),
                    coverage: if !masked || alpha >= material.alpha_cutoff {
                        1.0
                    } else {
                        0.0
                    },
                }
            };
            let source = Source {
                positions: local,
                normals: &normals[vertices.clone()],
                uvs: &uvs[vertices],
                indices,
            };
            let cook = || ClusterLod::cook(source, surface);
            if indices.len() / 3 < CACHED_TRIANGLES {
                return cook();
            }
            let mut key = DefaultHasher::new();
            COOK_VERSION.hash(&mut key);
            bytemuck::cast_slice::<_, u8>(colors).hash(&mut key);
            bytemuck::cast_slice::<_, u8>(indices).hash(&mut key);
            bytemuck::cast_slice::<_, u8>(&material.base_color).hash(&mut key);
            masked.hash(&mut key);
            material.alpha_cutoff.to_bits().hash(&mut key);
            if let Some(preview) = preview {
                preview.rgba.hash(&mut key);
            }
            cached(key.finish(), local.len(), cook)
        })
        .collect()
}

/// The result of `cook`, reusing an earlier run's for the same key. The cache is
/// only a speedup: any failure to read or write it just cooks.
fn cached(
    key: u64,
    vertex_count: usize,
    cook: impl FnOnce() -> Option<ClusterLod>,
) -> Option<ClusterLod> {
    let directory = std::env::temp_dir().join("verity-matrix");
    let path = directory.join(format!("{key:016x}.lod"));
    if let Some(lod) = fs::read(&path)
        .ok()
        .and_then(|bytes| ClusterLod::from_bytes(&bytes, vertex_count))
    {
        return Some(lod);
    }
    let lod = cook()?;
    // Renamed into place, so a reader never sees half a file.
    let partial = path.with_extension(format!("{}.tmp", std::process::id()));
    let written = fs::create_dir_all(&directory)
        .and_then(|()| fs::write(&partial, lod.to_bytes()))
        .and_then(|()| fs::rename(&partial, &path));
    if written.is_err() {
        let _ = fs::remove_file(&partial);
    }
    Some(lod)
}

impl Renderer {
    /// Remove all imported glTF scenes and restore the built-in cube geometry.
    ///
    /// Extensions retain their own scene data, so remove them before clearing.
    pub fn clear_gltf(&mut self) -> Result<()> {
        if self.extension.is_some() {
            return Err(failure(
                "cannot clear imported scenes while a render extension is installed",
            ));
        }
        unsafe {
            self.device.device_wait_idle()?;
            let counts = (self.vertex_count, self.index_total);
            self.deforming_capacity = 0;
            self.retained.clear();
            if let Some(embedded) = &mut self.embedded {
                embedded.clear();
            }
            self.vertex_count = 0;
            self.index_total = 0;
            let (vertices, indices) = scene::cube();
            if let Err(error) = self.append_geometry(&vertices, &indices) {
                (self.vertex_count, self.index_total) = counts;
                return Err(error);
            }
            let allocator = self.allocator.as_mut().unwrap();
            for image in self.textures.drain(..) {
                gpu::destroy_image(&self.device, allocator, image);
            }
            // Terrain layer textures went with the rest: stop the resolve sampling them.
            self.terrain_layers = None;
            for frame in &mut self.frames {
                frame.layer_table.write(&[0; 16]);
            }
            self.primitives.truncate(1);
            self.primitive_meshlets.truncate(1);
            self.matrix.clear();
            self.matrix
                .destroy_bake_targets(&self.device, self.allocator.as_mut().unwrap());
            self.upload_matrix()?;
            self.source_indices.clear();
            self.source_indices.extend_from_slice(&indices);
            self.frame_indices.clear();
            self.model_draws.clear();
            self.model_generation += 1;
            self.materials.clear();
            self.scene_lights.clear();
            if self.sun_is_imported {
                self.sun = None;
                self.sun_is_imported = false;
            }
            self.previous_view_proj = None;
        }
        Ok(())
    }

    /// Load a `.glb` with KTX2 textures (see `scripts/convert_assets.py`) and
    /// draw it every frame from now on. Mip levels above `max_texture_size`
    /// are dropped to bound GPU memory. Its point lights join the frame's
    /// lights and its sun, if any, replaces the current one.
    pub fn load_gltf(
        &mut self,
        path: impl AsRef<Path>,
        max_texture_size: u32,
    ) -> Result<SceneStats> {
        let options = LoadOptions {
            format: self.texture_format,
            max_texture_size,
        };
        let loaded =
            asset::load_gltf(path.as_ref(), options).map_err(|e| failure(e.to_string()))?;
        self.add_scene(loaded)
    }

    /// Draw a scene built or loaded elsewhere every frame from now on, as
    /// [`Renderer::load_gltf`] does.
    pub fn add_scene(&mut self, loaded: asset::SceneData) -> Result<SceneStats> {
        if self.textures.len() + loaded.textures.len() > scene::MAX_TEXTURES as usize {
            return Err(failure(format!(
                "more than {} textures loaded",
                scene::MAX_TEXTURES
            )));
        }
        let positions: Vec<_> = loaded
            .vertices
            .iter()
            .map(|v| [v.position[0], v.position[1], v.position[2]])
            .collect();
        let cooked = cook_primitives(&loaded, &positions);
        let (material_base, draw_base) = unsafe {
            // Frames in flight read the buffers this replaces.
            self.device.device_wait_idle()?;
            let (base_vertex, first_index) =
                self.append_geometry(&loaded.vertices, &loaded.indices)?;
            self.source_indices.extend_from_slice(&loaded.indices);
            let texture_base = self.textures.len() as u32;
            for texture in &loaded.textures {
                let image = self.upload_texture(texture)?;
                self.textures.push(image);
            }
            self.write_texture_descriptors();

            let material_base = self.materials.len() as u32;
            let texture = |slot: Option<u32>| slot.map_or(scene::NO_TEXTURE, |s| texture_base + s);
            self.materials
                .extend(loaded.materials.iter().map(|m| LoadedMaterial {
                    gpu: GpuMaterial {
                        base_color: m.base_color,
                        params: [m.metallic, m.roughness, m.normal_scale, m.alpha_cutoff],
                        textures: [
                            texture(m.base_color_texture),
                            texture(m.metallic_roughness_texture),
                            texture(m.normal_texture),
                            match m.alpha_mode {
                                AlphaMode::Opaque => 0,
                                AlphaMode::Mask => scene::ALPHA_MASK,
                                AlphaMode::Blend => scene::ALPHA_BLEND,
                            },
                        ],
                        transmission: m.transmission,
                    },
                    double_sided: m.double_sided,
                    // shortcut: blended materials need a forward transparency
                    // pass; until it exists they are not drawn.
                    hidden: m.alpha_mode == AlphaMode::Blend,
                }));
            let primitive_base = self.primitives.len() as u32;
            self.primitives
                .extend(loaded.primitives.iter().map(|p| asset::Primitive {
                    first_index: first_index + p.first_index,
                    base_vertex: base_vertex + p.base_vertex,
                    material: material_base + p.material,
                    ..*p
                }));
            for lod in cooked {
                self.matrix.add_primitive(lod);
            }
            let meshlets: Vec<_> = loaded
                .primitives
                .par_iter()
                .map(|p| {
                    let start = p.first_index as usize;
                    let end = start + p.index_count as usize;
                    crate::meshlets::build(
                        &positions[p.base_vertex as usize..],
                        &loaded.indices[start..end],
                    )
                })
                .collect();
            self.primitive_meshlets.extend(meshlets);
            self.model_generation += 1;
            let draw_base = self.model_draws.len() as u32;
            self.model_draws
                .extend(loaded.draws.iter().map(|d| asset::Draw {
                    primitive: primitive_base + d.primitive,
                    ..*d
                }));
            for (index, draw) in loaded.draws.iter().enumerate() {
                self.add_matrix_instance(
                    asset::Draw {
                        primitive: primitive_base + draw.primitive,
                        ..*draw
                    },
                    draw_base + index as u32,
                );
            }
            self.upload_matrix()?;
            (material_base, draw_base)
        };
        if let Some(mut extension) = self.extension.take() {
            let result = extension.scene_loaded(&mut self.gpu(), &loaded, material_base, draw_base);
            self.extension = Some(extension);
            result?;
        }
        self.scene_lights.extend_from_slice(&loaded.lights);
        if loaded.sun.is_some() {
            self.sun = loaded.sun;
            self.sun_is_imported = true;
        }
        Ok(SceneStats {
            draws: loaded.draws.len(),
            triangles: loaded
                .draws
                .iter()
                .map(|d| loaded.primitives[d.primitive as usize].index_count as usize / 3)
                .sum(),
            textures: loaded.textures.len(),
            lights: loaded.lights.len(),
        })
    }

    /// Record and run commands once, waiting for them to finish.
    pub(crate) unsafe fn submit_once(&self, record: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
        let device = &self.device;
        let cmd = device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0];
        let result = (|| -> Result<()> {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            record(cmd);
            device.end_command_buffer(cmd)?;
            let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            device.queue_submit2(
                self.queue,
                &[vk::SubmitInfo2::default().command_buffer_infos(&commands)],
                vk::Fence::null(),
            )?;
            Ok(device.queue_wait_idle(self.queue)?)
        })();
        device.free_command_buffers(self.command_pool, &[cmd]);
        result
    }

    /// Grow the shared vertex buffer and reserve source index offsets.
    /// The GPU must be idle.
    pub(crate) unsafe fn append_geometry(
        &mut self,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> Result<(u32, u32)> {
        let new_vertex_count = self
            .vertex_count
            .checked_add(u32::try_from(vertices.len()).map_err(|_| failure("too many vertices"))?)
            .ok_or_else(|| failure("too many vertices"))?;
        let new_index_total = self
            .index_total
            .checked_add(u32::try_from(indices.len()).map_err(|_| failure("too many indices"))?)
            .ok_or_else(|| failure("too many indices"))?;
        let vertex_bytes: &[u8] = bytemuck::cast_slice(vertices);
        let old_vertex_bytes = self.vertex_count as u64 * mem::size_of::<Vertex>() as u64;
        let device = &self.device;
        let allocator = self.allocator.as_mut().unwrap();
        use vk::BufferUsageFlags as U;
        let mut staging = gpu::create_buffer(
            device,
            allocator,
            "vertex staging",
            vertex_bytes.len() as u64,
            U::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        )?;
        staging.write(vertex_bytes);
        let new_vertices = match gpu::create_buffer(
            device,
            allocator,
            "vertices",
            old_vertex_bytes + vertex_bytes.len() as u64,
            U::STORAGE_BUFFER | U::TRANSFER_SRC | U::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                gpu::destroy_buffer(device, allocator, staging);
                return Err(error);
            }
        };
        let old_vertices = self.vertices.buffer;
        let result = self.submit_once(|cmd| {
            if old_vertex_bytes > 0 {
                device.cmd_copy_buffer(
                    cmd,
                    old_vertices,
                    new_vertices.buffer,
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: old_vertex_bytes,
                    }],
                );
            }
            if !vertex_bytes.is_empty() {
                device.cmd_copy_buffer(
                    cmd,
                    staging.buffer,
                    new_vertices.buffer,
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: old_vertex_bytes,
                        size: vertex_bytes.len() as u64,
                    }],
                );
            }
        });
        let allocator = self.allocator.as_mut().unwrap();
        gpu::destroy_buffer(device, allocator, staging);
        if let Err(error) = result {
            gpu::destroy_buffer(device, allocator, new_vertices);
            return Err(error);
        }
        gpu::destroy_buffer(
            device,
            allocator,
            mem::replace(&mut self.vertices, new_vertices),
        );
        let start = (self.vertex_count, self.index_total);
        self.vertex_count = new_vertex_count;
        self.index_total = new_index_total;
        for frame in 0..self.frames.len() {
            self.write_buffer_descriptors(frame);
        }
        Ok(start)
    }

    /// Copy a transcoded mip chain into a new sampled image.
    pub(crate) unsafe fn upload_texture(&mut self, texture: &TextureData) -> Result<gpu::Image> {
        let format = vk_format(texture.format, texture.srgb);
        let device = &self.device;
        let allocator = self.allocator.as_mut().unwrap();
        let bytes = texture.levels.concat();
        let mut staging = gpu::create_buffer(
            device,
            allocator,
            "texture staging",
            bytes.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        )?;
        staging.write(&bytes);
        let levels = texture.levels.len() as u32;
        let extent = vk::Extent2D {
            width: texture.width,
            height: texture.height,
        };
        let image = match gpu::create_image(
            device,
            allocator,
            "material texture",
            format,
            extent,
            levels,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            vk::ImageAspectFlags::COLOR,
        ) {
            Ok(image) => image,
            Err(e) => {
                gpu::destroy_buffer(device, allocator, staging);
                return Err(e);
            }
        };
        let mut offset = 0;
        let regions: Vec<_> = texture
            .levels
            .iter()
            .enumerate()
            .map(|(level, data)| {
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(offset)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(level as u32)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D {
                        width: (texture.width >> level).max(1),
                        height: (texture.height >> level).max(1),
                        depth: 1,
                    });
                offset += data.len() as u64;
                region
            })
            .collect();
        let range = gpu::subresource(vk::ImageAspectFlags::COLOR).level_count(levels);
        use vk::{AccessFlags2 as A, ImageLayout as L, PipelineStageFlags2 as S};
        let transition = |old, new, src_stage, src_access, dst_stage, dst_access| {
            vk::ImageMemoryBarrier2::default()
                .image(image.image)
                .subresource_range(range)
                .old_layout(old)
                .new_layout(new)
                .src_stage_mask(src_stage)
                .src_access_mask(src_access)
                .dst_stage_mask(dst_stage)
                .dst_access_mask(dst_access)
        };
        let result = self.submit_once(|cmd| {
            let barrier = |b| {
                device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&[b]),
                )
            };
            barrier(transition(
                L::UNDEFINED,
                L::TRANSFER_DST_OPTIMAL,
                S::NONE,
                A::NONE,
                S::COPY,
                A::TRANSFER_WRITE,
            ));
            device.cmd_copy_buffer_to_image(
                cmd,
                staging.buffer,
                image.image,
                L::TRANSFER_DST_OPTIMAL,
                &regions,
            );
            barrier(transition(
                L::TRANSFER_DST_OPTIMAL,
                L::SHADER_READ_ONLY_OPTIMAL,
                S::COPY,
                A::TRANSFER_WRITE,
                S::FRAGMENT_SHADER | S::COMPUTE_SHADER,
                A::SHADER_SAMPLED_READ,
            ));
        });
        let allocator = self.allocator.as_mut().unwrap();
        gpu::destroy_buffer(device, allocator, staging);
        match result {
            Ok(()) => Ok(image),
            Err(e) => {
                gpu::destroy_image(device, allocator, image);
                Err(e)
            }
        }
    }
}

/// The image format a transcoded texture uploads as.
pub(crate) fn vk_format(format: TextureFormat, srgb: bool) -> vk::Format {
    match (format, srgb) {
        (TextureFormat::Bc7, true) => vk::Format::BC7_SRGB_BLOCK,
        (TextureFormat::Bc7, false) => vk::Format::BC7_UNORM_BLOCK,
        (TextureFormat::Astc4x4, true) => vk::Format::ASTC_4X4_SRGB_BLOCK,
        (TextureFormat::Astc4x4, false) => vk::Format::ASTC_4X4_UNORM_BLOCK,
        (TextureFormat::Etc2, true) => vk::Format::ETC2_R8G8B8A8_SRGB_BLOCK,
        (TextureFormat::Etc2, false) => vk::Format::ETC2_R8G8B8A8_UNORM_BLOCK,
    }
}
