//! Physically based sky and volumetric clouds behind the scene; see
//! `shaders/clouds.slang` for the techniques. Clouds are traced at half
//! resolution into a ping-pong history, then fill the pixels the resolve left
//! empty. A shadow map of sunlight through the layer is drawn before the
//! resolve, which darkens the sun with it. A grid of density summed toward the
//! sun, refreshed a slice in eight per frame, lights the clouds themselves.
//!
//! An authored voxel cloudscape (Nubis³) can stand among the layer's clouds.
//! It ships as a `.vqcloud` (VQVDB-style codes for its cloudy bricks only),
//! which the GPU decodes once into a brick atlas when the cloudscape is chosen.

use crate::{asset, gpu, Result};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use gpu_allocator::{vulkan::Allocator, MemoryLocation};
use std::time::Instant;
use verity_core::{Cloudscape, Clouds};

const SHAPE_SIZE: u32 = 128;
const LIGHT_SIZE: u32 = 128;
/// The cloudscape's light grid, stretched over its box (Nubis³: 256x256x32).
const SCAPE_LIGHT_SIZE: [u32; 3] = [128, 128, 64];
const NOISE_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
const HISTORY_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
const SHADOW_SIZE: u32 = 512;
/// World width of the cloud shadow map, centered on the camera.
const SHADOW_EXTENT: f32 = 20000.0;
/// An atlas cell: a brick and a voxel of its neighbours on each side.
const CELL: u32 = 10;
/// Marks an empty brick in the brick table; the low byte is the distance in
/// bricks to the nearest cloudy one.
const EMPTY: u32 = 1 << 31;

/// Guerrilla's curly-alligator and alligator detail noise (Nubis³).
static NUBIS_NOISE: &[u8] = include_bytes!("../../../assets/packed/clouds/nubis_noise.ktx2");

fn cloudscape_file(kind: Cloudscape) -> Option<&'static [u8]> {
    match kind {
        Cloudscape::None => None,
        Cloudscape::Cumulus => Some(include_bytes!("../../../assets/packed/clouds/wdas.vqcloud")),
        Cloudscape::Parkouring => {
            Some(include_bytes!("../../../assets/packed/clouds/parkouring.vqcloud"))
        }
        Cloudscape::Stormbird => {
            Some(include_bytes!("../../../assets/packed/clouds/stormbird.vqcloud"))
        }
    }
}

/// The renderer's cloud shadow map; `CloudPass::record_shadow` keeps it valid.
pub(crate) unsafe fn create_shadow_map(
    device: &ash::Device,
    allocator: &mut Allocator,
) -> Result<gpu::Image> {
    gpu::create_image(
        device,
        allocator,
        "cloud shadows",
        vk::Format::R32_SFLOAT,
        vk::Extent2D {
            width: SHADOW_SIZE,
            height: SHADOW_SIZE,
        },
        1,
        vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST,
        vk::ImageAspectFlags::COLOR,
    )
}

/// The shadow map's world corner and size, snapped to whole texels so it
/// does not shimmer as the camera moves.
pub(crate) fn shadow_region(eye: Vec3) -> [f32; 4] {
    let texel = SHADOW_EXTENT / SHADOW_SIZE as f32;
    let corner = |v: f32| (v / texel).round() * texel - SHADOW_EXTENT * 0.5;
    [corner(eye.x), corner(eye.y), SHADOW_EXTENT, 1.0]
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    layer: [f32; 4],
    shape: [f32; 4],
    frame: [u32; 4],
    region: [f32; 4],
    scape_min: [f32; 4],
    scape_grid: [u32; 4],
    scape_info: [u32; 4],
}

/// Atlas cells along x, y and z for `bricks` bricks: near a cube, so no
/// side passes the 2048 texels every GPU allows.
fn atlas_cells(bricks: u32) -> [u32; 3] {
    let side = (f64::from(bricks).cbrt().ceil() as u32).max(1);
    [side, side, bricks.div_ceil(side * side).max(1)]
}

/// A 3D image of `extent`, at least 2 deep so it stays 3D.
unsafe fn volume(
    device: &ash::Device,
    allocator: &mut Allocator,
    name: &str,
    format: vk::Format,
    [width, height, depth]: [u32; 3],
    levels: u32,
    usage: vk::ImageUsageFlags,
) -> Result<gpu::Image> {
    let extent = vk::Extent3D {
        width,
        height,
        depth: depth.max(2),
    };
    let samples = vk::SampleCountFlags::TYPE_1;
    let aspect = vk::ImageAspectFlags::COLOR;
    gpu::create_image_of(device, allocator, name, format, extent, levels, usage, aspect, samples)
}

unsafe fn host_buffer(
    device: &ash::Device,
    allocator: &mut Allocator,
    name: &str,
    bytes: &[u8],
    usage: vk::BufferUsageFlags,
) -> Result<gpu::Buffer> {
    let mut buffer =
        gpu::create_buffer(device, allocator, name, bytes.len() as u64, usage, MemoryLocation::CpuToGpu)?;
    if !bytes.is_empty() {
        buffer.write(bytes);
    }
    Ok(buffer)
}

/// The chosen cloudscape on the GPU. With none, small placeholders keep the
/// descriptors valid.
#[derive(Default)]
struct Scape {
    kind: Cloudscape,
    atlas: Option<gpu::Image>,
    /// Per brick: its atlas slot, or `EMPTY` | distance.
    leaves: Option<gpu::Image>,
    light: Option<gpu::Image>,
    /// The .vqcloud payload the decode reads.
    payload: gpu::Buffer,
    /// The brick table, copied into `leaves`.
    staging: gpu::Buffer,
    grid: [u32; 3],
    voxel: f32,
    cells: [u32; 3],
    bricks: u32,
    codes: u32,
    /// The brick table still needs copying and the atlas decoding.
    pending: bool,
}

impl Scape {
    unsafe fn new(device: &ash::Device, allocator: &mut Allocator, kind: Cloudscape) -> Result<Self> {
        let cloud = (cloudscape_file(kind).map(asset::decode_vq_cloud).transpose())
            .map_err(|e| e as Box<dyn std::error::Error>)?;
        let mut scape = Self {
            kind,
            grid: [1; 3],
            pending: true,
            ..Self::default()
        };
        let (table, payload) = match &cloud {
            Some(cloud) => {
                let mut table: Vec<u32> = cloud.distance.iter().map(|&d| EMPTY | u32::from(d)).collect();
                let [width, height, _] = cloud.grid;
                for (slot, &brick) in cloud.bricks.iter().enumerate() {
                    let [x, y, z] = [brick & 1023, brick >> 10 & 1023, brick >> 20];
                    table[(x + width * (y + height * z)) as usize] = slot as u32;
                }
                scape.grid = cloud.grid;
                scape.voxel = cloud.voxel;
                scape.bricks = cloud.bricks.len() as u32;
                scape.codes = cloud.codes;
                (table, &cloud.payload[..])
            }
            None => (vec![EMPTY], &[][..]),
        };
        scape.cells = atlas_cells(scape.bricks);
        let atlas_size = scape.cells.map(|n| n * CELL);
        if atlas_size.iter().any(|&n| n > 2048) {
            return Err(format!("cloudscape {kind:?} has too many bricks for one atlas").into());
        }
        let storage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED;
        let result = (|| -> Result<()> {
            scape.payload = host_buffer(
                device,
                allocator,
                "cloudscape codes",
                payload,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )?;
            scape.staging = host_buffer(
                device,
                allocator,
                "cloudscape brick table upload",
                bytemuck::cast_slice(&table),
                vk::BufferUsageFlags::TRANSFER_SRC,
            )?;
            scape.atlas = Some(volume(
                device,
                allocator,
                "cloudscape atlas",
                vk::Format::R8_UNORM,
                atlas_size,
                1,
                storage,
            )?);
            scape.leaves = Some(volume(
                device,
                allocator,
                "cloudscape brick table",
                vk::Format::R32_UINT,
                scape.grid.map(|n| n.max(2)),
                1,
                vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            )?);
            scape.light = Some(volume(
                device,
                allocator,
                "cloudscape light grid",
                HISTORY_FORMAT,
                SCAPE_LIGHT_SIZE,
                1,
                storage,
            )?);
            Ok(())
        })();
        if let Err(error) = result {
            scape.destroy(device, allocator);
            return Err(error);
        }
        Ok(scape)
    }

    unsafe fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        for image in [self.atlas.take(), self.leaves.take(), self.light.take()].into_iter().flatten() {
            gpu::destroy_image(device, allocator, image);
        }
        gpu::destroy_buffer(device, allocator, std::mem::take(&mut self.payload));
        gpu::destroy_buffer(device, allocator, std::mem::take(&mut self.staging));
    }
}

#[derive(Default)]
pub(crate) struct CloudPass {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pool: vk::DescriptorPool,
    samplers: [vk::Sampler; 2],
    shape_pipeline: vk::Pipeline,
    trace: vk::Pipeline,
    shadow_pipeline: vk::Pipeline,
    light_pipeline: vk::Pipeline,
    decode_pipeline: vk::Pipeline,
    scape_light_pipeline: vk::Pipeline,
    /// The shadow map has left `UNDEFINED`; it holds all ones while `shadow_clear`.
    shadow_laid_out: bool,
    shadow_clear: bool,
    apply: vk::Pipeline,
    shape: Option<gpu::Image>,
    /// The Nubis detail noise, with its mips.
    noise: Option<gpu::Image>,
    /// The noise's mips, smallest last, and where each starts in `noise_upload`.
    noise_levels: Vec<(u32, u64)>,
    noise_size: [u32; 3],
    noise_upload: gpu::Buffer,
    light: Option<gpu::Image>,
    scape: Scape,
    /// `sets[k]` reads `history[k]` and writes the other.
    sets: [vk::DescriptorSet; 2],
    history: Vec<gpu::Image>,
    size: [u32; 2],
    baked: bool,
    /// False until the history images are laid out.
    ready: bool,
    history_valid: bool,
    parity: usize,
    wind: [f32; 2],
    last: Option<Instant>,
}

impl CloudPass {
    unsafe fn init(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        frame_layout: vk::DescriptorSetLayout,
        shadow: vk::ImageView,
    ) -> Result<()> {
        use vk::DescriptorType as D;
        let kinds = [
            D::STORAGE_IMAGE,
            D::STORAGE_IMAGE,
            D::SAMPLED_IMAGE,
            D::SAMPLED_IMAGE,
            D::SAMPLER,
            D::SAMPLER,
            D::SAMPLED_IMAGE,
            D::STORAGE_IMAGE,
            D::STORAGE_IMAGE,
            D::STORAGE_IMAGE,
            D::SAMPLED_IMAGE,
            D::SAMPLED_IMAGE,
            D::SAMPLED_IMAGE,
            D::STORAGE_BUFFER,
            D::STORAGE_IMAGE,
            D::SAMPLED_IMAGE,
        ];
        let bindings: Vec<_> = (kinds.iter().enumerate())
            .map(|(i, &kind)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i as u32)
                    .descriptor_type(kind)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        self.set_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )?;
        let layouts = [frame_layout, self.set_layout];
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(size_of::<Params>() as u32)];
        self.layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&layouts)
                .push_constant_ranges(&push),
            None,
        )?;
        let sizes = [D::STORAGE_IMAGE, D::SAMPLED_IMAGE, D::SAMPLER, D::STORAGE_BUFFER].map(|ty| {
            let per_set = kinds.iter().filter(|&&k| k == ty).count() as u32;
            vk::DescriptorPoolSize::default().ty(ty).descriptor_count(2 * per_set)
        });
        self.pool = device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(2)
                .pool_sizes(&sizes),
            None,
        )?;
        let set_layouts = [self.set_layout; 2];
        let sets = device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.pool)
                .set_layouts(&set_layouts),
        )?;
        self.sets = [sets[0], sets[1]];
        for (sampler, mode) in self.samplers.iter_mut().zip([
            vk::SamplerAddressMode::REPEAT,
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
        ]) {
            *sampler = device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .max_lod(vk::LOD_CLAMP_NONE)
                    .address_mode_u(mode)
                    .address_mode_v(mode)
                    .address_mode_w(mode),
                None,
            )?;
        }
        for (pipeline, entry, spirv) in [
            (&mut self.shape_pipeline, c"shapeMain", &spirv!("clouds.shapeMain")[..]),
            (&mut self.trace, c"traceMain", &spirv!("clouds.traceMain")[..]),
            (&mut self.apply, c"applyMain", &spirv!("clouds.applyMain")[..]),
            (&mut self.shadow_pipeline, c"shadowMain", &spirv!("clouds.shadowMain")[..]),
            (&mut self.light_pipeline, c"lightMain", &spirv!("clouds.lightMain")[..]),
            (&mut self.decode_pipeline, c"decodeMain", &spirv!("clouds.decodeMain")[..]),
            (&mut self.scape_light_pipeline, c"scapeLightMain", &spirv!("clouds.scapeLightMain")[..]),
        ] {
            *pipeline =
                gpu::compute_pipeline(device, vk::PipelineCache::null(), self.layout, spirv, entry)?;
        }
        let usage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED;
        self.shape = Some(gpu::create_volume(
            device,
            allocator,
            "cloud shape noise",
            NOISE_FORMAT,
            SHAPE_SIZE,
            usage,
        )?);
        let noise = asset::decode_volume(NUBIS_NOISE).map_err(|e| e as Box<dyn std::error::Error>)?;
        self.noise_size = noise.size;
        let mut upload = Vec::new();
        for (level, data) in noise.levels.iter().enumerate() {
            self.noise_levels.push((level as u32, upload.len() as u64));
            upload.extend_from_slice(data);
        }
        self.noise_upload = host_buffer(
            device,
            allocator,
            "cloud noise upload",
            &upload,
            vk::BufferUsageFlags::TRANSFER_SRC,
        )?;
        self.noise = Some(volume(
            device,
            allocator,
            "cloud detail noise",
            NOISE_FORMAT,
            noise.size,
            noise.levels.len() as u32,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        )?);
        self.light = Some(gpu::create_volume(
            device,
            allocator,
            "cloud light grid",
            HISTORY_FORMAT,
            LIGHT_SIZE,
            usage,
        )?);
        let samplers = self.samplers;
        let shape = self.shape.as_ref().unwrap().view;
        let noise = self.noise.as_ref().unwrap().view;
        let light = self.light.as_ref().unwrap().view;
        for set in self.sets {
            let image = |view, layout| [vk::DescriptorImageInfo::default().image_view(view).image_layout(layout)];
            let general = vk::ImageLayout::GENERAL;
            let infos = [
                (0, D::STORAGE_IMAGE, image(shape, general)),
                (2, D::SAMPLED_IMAGE, image(shape, general)),
                (3, D::SAMPLED_IMAGE, image(noise, general)),
                (4, D::SAMPLER, [vk::DescriptorImageInfo::default().sampler(samplers[0])]),
                (5, D::SAMPLER, [vk::DescriptorImageInfo::default().sampler(samplers[1])]),
                (8, D::STORAGE_IMAGE, image(shadow, general)),
                (9, D::STORAGE_IMAGE, image(light, general)),
                (10, D::SAMPLED_IMAGE, image(light, general)),
            ];
            let writes: Vec<_> = (infos.iter())
                .map(|(binding, ty, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(*binding)
                        .descriptor_type(*ty)
                        .image_info(info)
                })
                .collect();
            device.update_descriptor_sets(&writes, &[]);
        }
        Ok(())
    }

    /// Load cloudscape `kind` if another is loaded, and point the descriptors at it.
    unsafe fn choose_scape(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        kind: Cloudscape,
    ) -> Result<()> {
        if self.scape.atlas.is_some() && self.scape.kind == kind {
            return Ok(());
        }
        if self.scape.atlas.is_some() {
            // Earlier frames in flight may still read the old one.
            device.device_wait_idle()?;
        }
        self.scape.destroy(device, allocator);
        self.scape = Scape::new(device, allocator, kind)?;
        let general = vk::ImageLayout::GENERAL;
        let atlas = self.scape.atlas.as_ref().unwrap().view;
        let leaves = self.scape.leaves.as_ref().unwrap().view;
        let light = self.scape.light.as_ref().unwrap().view;
        let image = |view| [vk::DescriptorImageInfo::default().image_view(view).image_layout(general)];
        let payload = [self.scape.payload.info()];
        use vk::DescriptorType as D;
        for set in self.sets {
            let images = [
                (1, D::STORAGE_IMAGE, image(atlas)),
                (11, D::SAMPLED_IMAGE, image(atlas)),
                (12, D::SAMPLED_IMAGE, image(leaves)),
                (14, D::STORAGE_IMAGE, image(light)),
                (15, D::SAMPLED_IMAGE, image(light)),
            ];
            let mut writes: Vec<_> = (images.iter())
                .map(|(binding, ty, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(*binding)
                        .descriptor_type(*ty)
                        .image_info(info)
                })
                .collect();
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(13)
                    .descriptor_type(D::STORAGE_BUFFER)
                    .buffer_info(&payload),
            );
            device.update_descriptor_sets(&writes, &[]);
        }
        // Both light grids start over, the cloudscape's from scratch.
        self.history_valid = false;
        Ok(())
    }

    /// (Re)create the half-resolution history for a viewport of `width` x `height`.
    unsafe fn resize(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let size = [width.div_ceil(2), height.div_ceil(2)];
        if size == self.size && !self.history.is_empty() {
            return Ok(());
        }
        // Earlier frames in flight may still sample the old history.
        device.device_wait_idle()?;
        for image in self.history.drain(..) {
            gpu::destroy_image(device, allocator, image);
        }
        self.size = size;
        for name in ["cloud history a", "cloud history b"] {
            self.history.push(gpu::create_image(
                device,
                allocator,
                name,
                HISTORY_FORMAT,
                vk::Extent2D {
                    width: size[0],
                    height: size[1],
                },
                1,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                vk::ImageAspectFlags::COLOR,
            )?);
        }
        for k in 0..2 {
            let read = [vk::DescriptorImageInfo::default()
                .image_view(self.history[k].view)
                .image_layout(vk::ImageLayout::GENERAL)];
            let write = [vk::DescriptorImageInfo::default()
                .image_view(self.history[1 - k].view)
                .image_layout(vk::ImageLayout::GENERAL)];
            device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.sets[k])
                        .dst_binding(6)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&read),
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.sets[k])
                        .dst_binding(7)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&write),
                ],
                &[],
            );
        }
        self.ready = false;
        self.history_valid = false;
        Ok(())
    }

    fn params(&self, clouds: &Clouds, frame_number: u64, eye: Vec3) -> Params {
        let thickness = clouds.thickness.max(1.0);
        let bottom = clouds.bottom.max(0.0);
        // Each seed moves the sky to another part of the (tiling) noise.
        let seed = |prime: u32| (clouds.seed.wrapping_mul(prime) % 60000) as f32;
        let scape = &self.scape;
        // Centered over the world origin, its base on the layer's.
        let extent = scape.grid.map(|n| n as f32 * 8.0 * scape.voxel);
        let voxel = if scape.bricks > 0 { scape.voxel } else { 0.0 };
        Params {
            layer: [
                bottom,
                bottom + thickness,
                clouds.coverage.clamp(0.0, 1.0),
                clouds.density.max(0.0),
            ],
            shape: [
                clouds.kind.clamp(0.0, 1.0),
                self.wind[0] + seed(7919),
                self.wind[1] + seed(104_729),
                clouds.cirrus.clamp(0.0, 1.0),
            ],
            frame: [
                frame_number as u32,
                u32::from(self.history_valid),
                self.size[0],
                self.size[1],
            ],
            region: shadow_region(eye),
            scape_min: [-extent[0] * 0.5, -extent[1] * 0.5, bottom, voxel],
            scape_grid: [scape.grid[0], scape.grid[1], scape.grid[2], scape.cells[0]],
            scape_info: [scape.cells[1], scape.bricks, scape.codes, 0],
        }
    }

    unsafe fn bind(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame_set: vk::DescriptorSet,
        params: &Params,
    ) {
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.layout,
            0,
            &[frame_set, self.sets[self.parity]],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            self.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(params),
        );
    }

    /// Copy the detail noise (on first use) and a newly chosen cloudscape's
    /// brick table into their images, and lay out its atlas and light grid.
    /// Leaves the copies visible to compute shaders.
    unsafe fn record_uploads(&mut self, device: &ash::Device, cmd: vk::CommandBuffer) {
        use vk::AccessFlags2 as A;
        use vk::ImageLayout as L;
        use vk::PipelineStageFlags2 as S;
        let color = vk::ImageAspectFlags::COLOR;
        // (image, buffer, regions as (level, offset), extent, mips)
        let mut copies = Vec::new();
        if !self.baked {
            let noise = self.noise.as_ref().unwrap();
            copies.push((noise.image, self.noise_upload.buffer, self.noise_levels.clone(), self.noise_size));
        }
        if self.scape.pending {
            let leaves = self.scape.leaves.as_ref().unwrap();
            copies.push((leaves.image, self.scape.staging.buffer, vec![(0, 0)], self.scape.grid));
        }
        let range = |levels: usize| gpu::subresource(color).level_count(levels as u32);
        let mut before: Vec<_> = (copies.iter())
            .map(|(image, _, levels, _)| {
                gpu::image_barrier(*image, color, S::NONE, A::NONE, S::COPY, A::TRANSFER_WRITE, L::UNDEFINED, L::TRANSFER_DST_OPTIMAL)
                    .subresource_range(range(levels.len()))
            })
            .collect();
        if self.scape.pending {
            for image in [&self.scape.atlas, &self.scape.light] {
                before.push(gpu::image_barrier(
                    image.as_ref().unwrap().image,
                    color,
                    S::NONE,
                    A::NONE,
                    S::COMPUTE_SHADER,
                    A::SHADER_STORAGE_WRITE,
                    L::UNDEFINED,
                    L::GENERAL,
                ));
            }
        }
        if before.is_empty() {
            return;
        }
        device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&before));
        for (image, buffer, levels, [width, height, depth]) in &copies {
            let regions: Vec<_> = (levels.iter())
                .map(|&(level, offset)| {
                    vk::BufferImageCopy::default()
                        .buffer_offset(offset)
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(color)
                                .mip_level(level)
                                .layer_count(1),
                        )
                        .image_extent(vk::Extent3D {
                            width: (width >> level).max(1),
                            height: (height >> level).max(1),
                            depth: (depth >> level).max(1),
                        })
                })
                .collect();
            device.cmd_copy_buffer_to_image(cmd, *buffer, *image, L::TRANSFER_DST_OPTIMAL, &regions);
        }
        let after: Vec<_> = (copies.iter())
            .map(|(image, _, levels, _)| {
                gpu::image_barrier(*image, color, S::COPY, A::TRANSFER_WRITE, S::COMPUTE_SHADER, A::SHADER_SAMPLED_READ, L::TRANSFER_DST_OPTIMAL, L::GENERAL)
                    .subresource_range(range(levels.len()))
            })
            .collect();
        if !after.is_empty() {
            device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&after));
        }
    }

    /// Draw the cloud shadow map, or with no clouds (or no sun) fill it with
    /// ones, before the resolve reads it. Uploads the noise on first use and
    /// decodes a newly chosen cloudscape.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn record_shadow(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        frame_layout: vk::DescriptorSetLayout,
        cmd: vk::CommandBuffer,
        frame_set: vk::DescriptorSet,
        shadow: &gpu::Image,
        clouds: Option<Clouds>,
        eye: Vec3,
        viewport: [u32; 4],
    ) -> Result<()> {
        use vk::AccessFlags2 as A;
        use vk::ImageLayout as L;
        use vk::PipelineStageFlags2 as S;
        let color = vk::ImageAspectFlags::COLOR;
        let old_layout = if self.shadow_laid_out { L::GENERAL } else { L::UNDEFINED };
        let Some(clouds) = clouds else {
            if self.shadow_clear {
                return Ok(());
            }
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&[gpu::image_barrier(
                    shadow.image,
                    color,
                    S::COMPUTE_SHADER,
                    A::NONE,
                    S::CLEAR,
                    A::TRANSFER_WRITE,
                    old_layout,
                    L::GENERAL,
                )]),
            );
            device.cmd_clear_color_image(
                cmd,
                shadow.image,
                L::GENERAL,
                &vk::ClearColorValue { float32: [1.0; 4] },
                &[gpu::subresource(color)],
            );
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&[gpu::image_barrier(
                    shadow.image,
                    color,
                    S::CLEAR,
                    A::TRANSFER_WRITE,
                    S::COMPUTE_SHADER,
                    A::SHADER_SAMPLED_READ,
                    L::GENERAL,
                    L::GENERAL,
                )]),
            );
            self.shadow_laid_out = true;
            self.shadow_clear = true;
            return Ok(());
        };
        if self.trace == vk::Pipeline::null() {
            if let Err(error) = self.init(device, allocator, frame_layout, shadow.view) {
                self.destroy(device, allocator);
                return Err(error);
            }
        }
        self.choose_scape(device, allocator, clouds.cloudscape)?;
        // Rewriting the history descriptors invalidates any command buffer
        // they are bound in, so resize before this frame binds them.
        if viewport[2] != 0 && viewport[3] != 0 {
            self.resize(device, allocator, viewport[2], viewport[3])?;
        }
        self.record_uploads(device, cmd);
        // Waits for last frame's resolve to finish reading the shadow map.
        let mut barriers = vec![gpu::image_barrier(
            shadow.image,
            color,
            S::COMPUTE_SHADER,
            A::NONE,
            S::COMPUTE_SHADER,
            A::SHADER_STORAGE_WRITE,
            old_layout,
            L::GENERAL,
        )];
        if !self.baked {
            barriers.extend([&self.shape, &self.light].map(|image| {
                gpu::image_barrier(
                    image.as_ref().unwrap().image,
                    color,
                    S::COMPUTE_SHADER,
                    A::NONE,
                    S::COMPUTE_SHADER,
                    A::SHADER_STORAGE_WRITE,
                    L::UNDEFINED,
                    L::GENERAL,
                )
            }));
        }
        device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&barriers),
        );
        self.shadow_laid_out = true;
        self.shadow_clear = false;
        let params = self.params(&clouds, 0, eye);
        self.bind(device, cmd, frame_set, &params);
        let memory = |src_access, dst_access| {
            let (src, dst) = ((S::COMPUTE_SHADER, src_access), (S::COMPUTE_SHADER, dst_access));
            gpu::memory_barrier(device, cmd, src, dst);
        };
        if !self.baked {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.shape_pipeline);
            device.cmd_dispatch(cmd, SHAPE_SIZE / 4, SHAPE_SIZE / 4, SHAPE_SIZE / 4);
            memory(A::SHADER_STORAGE_WRITE, A::SHADER_SAMPLED_READ);
            self.baked = true;
        }
        if self.scape.pending {
            if self.scape.bricks > 0 {
                let [x, y, z] = self.scape.cells.map(|n| (n * CELL).div_ceil(4));
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.decode_pipeline);
                device.cmd_dispatch(cmd, x, y, z);
                memory(A::SHADER_STORAGE_WRITE, A::SHADER_SAMPLED_READ);
            }
            self.scape.pending = false;
        }
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.shadow_pipeline);
        device.cmd_dispatch(cmd, SHADOW_SIZE / 8, SHADOW_SIZE / 8, 1);
        memory(A::SHADER_STORAGE_WRITE, A::SHADER_SAMPLED_READ);
        Ok(())
    }

    /// Trace the clouds and fill the sky, after `record_shadow` this frame.
    /// The resolve's (and any extension's) HDR writes must be visible to
    /// compute shaders; this leaves its own HDR writes as compute-shader
    /// storage writes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn record(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        cmd: vk::CommandBuffer,
        frame_set: vk::DescriptorSet,
        frame_number: u64,
        viewport: [u32; 4],
        eye: Vec3,
        clouds: Option<Clouds>,
    ) -> Result<()> {
        // Without a sun `record_shadow` skipped the setup; the sky stays plain.
        let Some(clouds) = clouds.filter(|_| self.baked) else {
            self.history_valid = false;
            self.last = None;
            return Ok(());
        };
        if viewport[2] == 0 || viewport[3] == 0 {
            return Ok(());
        }
        // A no-op after `record_shadow`; without a sun nothing is bound yet.
        self.resize(device, allocator, viewport[2], viewport[3])?;

        let now = Instant::now();
        let dt = self.last.map_or(0.0, |last| (now - last).as_secs_f32().min(0.1));
        self.last = Some(now);
        for (offset, speed) in self.wind.iter_mut().zip(clouds.wind) {
            // shortcut: unwrapped, so f32 detail drifts after days of wind; wrap at a common noise period if that matters.
            *offset += speed * dt;
        }

        use vk::AccessFlags2 as A;
        use vk::ImageLayout as L;
        use vk::PipelineStageFlags2 as S;
        let memory = |src_access, dst_access| {
            let (src, dst) = ((S::COMPUTE_SHADER, src_access), (S::COMPUTE_SHADER, dst_access));
            gpu::memory_barrier(device, cmd, src, dst);
        };
        let params = self.params(&clouds, frame_number, eye);
        self.bind(device, cmd, frame_set, &params);
        if !self.ready {
            let barriers = [self.history[0].image, self.history[1].image].map(|image| {
                gpu::image_barrier(
                    image,
                    vk::ImageAspectFlags::COLOR,
                    S::COMPUTE_SHADER,
                    A::NONE,
                    S::COMPUTE_SHADER,
                    A::SHADER_STORAGE_WRITE,
                    L::UNDEFINED,
                    L::GENERAL,
                )
            });
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&barriers),
            );
            self.ready = true;
        }
        // Covers last frame's history and this frame's HDR.
        memory(
            A::SHADER_STORAGE_WRITE,
            A::SHADER_SAMPLED_READ | A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE,
        );
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.light_pipeline);
        // The shaders redraw every slice when the history restarts.
        let slices = if self.history_valid { LIGHT_SIZE / 8 } else { LIGHT_SIZE };
        device.cmd_dispatch(cmd, LIGHT_SIZE / 8, LIGHT_SIZE / 8, slices);
        if self.scape.bricks > 0 {
            let [x, y, z] = SCAPE_LIGHT_SIZE;
            let slices = if self.history_valid { z / 8 } else { z };
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.scape_light_pipeline);
            device.cmd_dispatch(cmd, x / 8, y / 8, slices);
        }
        memory(A::SHADER_STORAGE_WRITE, A::SHADER_SAMPLED_READ);
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.trace);
        device.cmd_dispatch(cmd, self.size[0].div_ceil(8), self.size[1].div_ceil(8), 1);
        memory(A::SHADER_STORAGE_WRITE, A::SHADER_SAMPLED_READ);
        // The other set samples what this trace wrote.
        self.parity = 1 - self.parity;
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.layout,
            1,
            &[self.sets[self.parity]],
            &[],
        );
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.apply);
        device.cmd_dispatch(cmd, viewport[2].div_ceil(8), viewport[3].div_ceil(8), 1);
        self.history_valid = true;
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        for image in (self.history.drain(..))
            .chain(self.shape.take())
            .chain(self.noise.take())
            .chain(self.light.take())
        {
            gpu::destroy_image(device, allocator, image);
        }
        self.scape.destroy(device, allocator);
        gpu::destroy_buffer(device, allocator, std::mem::take(&mut self.noise_upload));
        for pipeline in [
            self.shape_pipeline,
            self.trace,
            self.apply,
            self.shadow_pipeline,
            self.light_pipeline,
            self.decode_pipeline,
            self.scape_light_pipeline,
        ] {
            device.destroy_pipeline(pipeline, None);
        }
        for sampler in self.samplers {
            device.destroy_sampler(sampler, None);
        }
        device.destroy_descriptor_pool(self.pool, None);
        device.destroy_pipeline_layout(self.layout, None);
        device.destroy_descriptor_set_layout(self.set_layout, None);
        // The shadow map is the renderer's and keeps its layout and contents.
        let (laid_out, clear) = (self.shadow_laid_out, self.shadow_clear);
        *self = Self::default();
        self.shadow_laid_out = laid_out;
        self.shadow_clear = clear;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atlas_holds_every_brick_within_gpu_limits() {
        for bricks in [0, 1, 236, 405, 56_244, 1_000_000] {
            let cells = atlas_cells(bricks);
            assert!(cells.iter().product::<u32>() >= bricks, "{bricks}");
            assert!(cells.iter().all(|&n| n * CELL <= 2048), "{bricks}: {cells:?}");
        }
    }

    #[test]
    fn shipped_cloudscapes_load() {
        for kind in Cloudscape::ALL {
            let Some(file) = cloudscape_file(kind) else {
                continue;
            };
            let cloud = asset::decode_vq_cloud(file).unwrap();
            assert!(cloud.bricks.len() > 100 && cloud.voxel > 1.0, "{kind:?}");
        }
    }
}
