//! Screen and voxel GI: rays follow the depth buffer and take their light from
//! last frame's image; past what the screen shows they walk a voxel copy of the
//! scene, which keeps its own lighting and bounces. It needs no GI scene or
//! physics, and costs the same however many triangles the scene has.
//!
//! Scenes voxelize on the CPU as they load, into one grid fitted to everything
//! loaded so far.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::{
    GiMethod, GiStatus,
    pxtgi::{Capture, FrameData, GpuLight, IO_BINDINGS, Readback},
    shared::{PassTimer, Res, upload, write_set},
};
use ash::vk;
use glam::{Mat4, Vec3};
use gpu_allocator::MemoryLocation;
use rayon::prelude::*;
use std::{
    collections::HashMap, f32::consts::PI, ffi::CString, mem, time::Duration, time::Instant,
};
use verity_render::{
    FrameContext, Gpu, RenderExtension, Targets,
    asset::{AlphaMode, SceneData},
    extension::Result,
    gpu::{self, Buffer, Image},
};

const SLOTS: usize = 2;
/// Cells along each side of the grid.
const GRID: u32 = 256;

macro_rules! spirv {
    ($entry:literal) => {
        (
            $entry,
            include_bytes!(concat!(env!("OUT_DIR"), "/voxel.", $entry, ".spv")) as &[u8],
        )
    };
}

const PASSES: &[(&str, &[u8])] = &[
    spirv!("inject"),
    spirv!("trace"),
    spirv!("temporal"),
    spirv!("atrous1"),
    spirv!("atrous2"),
    spirv!("atrous4"),
    spirv!("composite"),
    spirv!("capture"),
];

use vk::DescriptorType as D;
/// Set 0: the frame and the voxels.
const SCENE_BINDINGS: &[(u32, D)] = &[
    (0, D::UNIFORM_BUFFER),
    (1, D::SAMPLED_IMAGE),
    (2, D::SAMPLED_IMAGE),
    (3, D::SAMPLED_IMAGE),
    (4, D::STORAGE_BUFFER),
    (5, D::STORAGE_BUFFER),
    (6, D::STORAGE_BUFFER),
    (10, D::STORAGE_BUFFER),
    (11, D::STORAGE_IMAGE),
    (12, D::STORAGE_IMAGE),
    (13, D::STORAGE_BUFFER),
];
/// Per slot: trace, temporal, three a-trous, composite.
const IO_SETS: usize = 6;

/// Screen and voxel GI controls. Distances are in metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenVoxelSettings {
    /// How far rays follow what is on screen before walking the voxels.
    pub max_distance: f32,
    /// How far rays walk the voxels. Surfaces further away do not bounce light.
    pub world_ray_distance: f32,
    /// Rays each traced texel sends per frame, clamped to 1..=4.
    pub rays_per_texel: u32,
    /// Scale of the bounce light.
    pub intensity: f32,
}

impl Default for ScreenVoxelSettings {
    fn default() -> Self {
        Self {
            max_distance: 8.0,
            world_ray_distance: 30.0,
            rays_per_texel: 4,
            intensity: 1.0,
        }
    }
}

/// A triangle in world space and the material it wears.
#[derive(Clone, Copy, Debug)]
pub struct Triangle {
    pub corners: [Vec3; 3],
    pub material: u32,
}

/// A scene as the shaders walk it.
#[derive(Debug)]
pub struct Voxels {
    /// The grid's low corner.
    pub origin: Vec3,
    pub cell: f32,
    /// Per cell, x fastest: 0 when empty, else an index into `voxels` + 1.
    pub cells: Vec<u32>,
    /// Cell, material, normal as signed bytes, unused.
    pub voxels: Vec<[u32; 4]>,
}

fn pack_normal(n: Vec3) -> u32 {
    let byte = |v: f32| ((v * 127.0).round() as i32 & 255) as u32;
    byte(n.x) | byte(n.y) << 8 | byte(n.z) << 16
}

/// A cell this full of surface always counts as solid.
const SOLID_COVER: f32 = 1.0;

/// A fixed number in 0..1 for a cell.
fn cell_noise(cell: u32) -> f32 {
    let mut x = cell.wrapping_mul(0x9e37_79b9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    (x >> 8) as f32 / 16_777_216.0
}

/// Mark the cells the triangles fill, in a cubic grid of `grid` cells a side
/// around them. A voxel takes the mean normal of what fell in it, or the first
/// normal where they cancel, and the first material.
///
/// A cell a surface barely clips stays solid only sometimes, in proportion to the
/// surface in it, by a fixed hash so it does not shimmer. Marking all of them
/// fattens everything by up to a cell and plugs railings and narrow openings.
pub fn voxelize(triangles: &[Triangle], grid: u32) -> Voxels {
    let (lo, hi) = triangles
        .iter()
        .flat_map(|t| t.corners)
        .fold((Vec3::MAX, Vec3::MIN), |(lo, hi), p| (lo.min(p), hi.max(p)));
    if triangles.is_empty() {
        return Voxels {
            origin: Vec3::ZERO,
            cell: 1.0,
            cells: vec![0; (grid as usize).pow(3)],
            voxels: Vec::new(),
        };
    }
    let cell = ((hi - lo).max_element() / grid as f32).max(1e-3) * 1.001;
    // Centred, so geometry stays clear of the grid's edges on the shorter axes.
    let origin = (lo + hi) * 0.5 - Vec3::splat(cell * grid as f32 * 0.5);
    let index = |p: Vec3| {
        let c = ((p - origin) / cell).floor().as_ivec3();
        let c = c.clamp(glam::IVec3::ZERO, glam::IVec3::splat(grid as i32 - 1));
        (c.z as u32 * grid + c.y as u32) * grid + c.x as u32
    };
    let mut hits: Vec<(u32, Vec3, u32, f32)> = triangles
        .par_iter()
        .flat_map_iter(|t| {
            let [a, b, c] = t.corners;
            let normal = (b - a).cross(c - a).normalize_or_zero();
            // Half-cell steps across the triangle reach every cell it crosses.
            let longest = (b - a).length().max((c - a).length()).max((c - b).length());
            let steps = (longest / (cell * 0.5)).ceil().max(1.0) as u32;
            let mut touched = Vec::new();
            for i in 0..=steps {
                for j in 0..=steps - i {
                    let (u, v) = (i as f32 / steps as f32, j as f32 / steps as f32);
                    touched.push(index(a + (b - a) * u + (c - a) * v));
                }
            }
            // Each sample stands for an equal share of the triangle's area.
            let share = 0.5 * (b - a).cross(c - a).length() / touched.len() as f32;
            touched.sort_unstable();
            let mut cells = Vec::new();
            for group in touched.chunk_by(|a, b| a == b) {
                cells.push((group[0], normal, t.material, share * group.len() as f32));
            }
            cells
        })
        .collect();
    hits.par_sort_by_key(|hit| hit.0);
    let mut cells = vec![0; (grid as usize).pow(3)];
    let mut voxels = Vec::new();
    for group in hits.chunk_by(|a, b| a.0 == b.0) {
        let (at, first, material, _) = group[0];
        let cover: f32 = group.iter().map(|hit| hit.3).sum::<f32>() / (cell * cell);
        if cell_noise(at) * SOLID_COVER >= cover {
            continue;
        }
        let sum: Vec3 = group.iter().map(|hit| hit.1).sum();
        let normal = sum
            .try_normalize()
            .filter(|_| sum.length() > 0.1)
            .unwrap_or(first);
        voxels.push([at, material, pack_normal(normal), 0]);
        cells[at as usize] = voxels.len() as u32;
    }
    Voxels {
        origin,
        cell,
        cells,
        voxels,
    }
}

/// How many visits a voxel side's arriving light keeps, `age` frames after the
/// lights changed: a side is visited every 8 frames.
fn changed_history(age: u32) -> f32 {
    (2 + age / 8) as f32
}

struct Pipelines {
    scene_layout: vk::DescriptorSetLayout,
    io_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    passes: HashMap<&'static str, vk::Pipeline>,
}

struct View {
    trace_extent: [u32; 2],
    trace: Image,
    accum: [Image; 2],
    accum_depth: [Image; 2],
    moments: [Image; 2],
    variance: Image,
    filtered: [Image; 2],
    filtered_variance: [Image; 2],
    prev_color: Image,
    frame: [Buffer; SLOTS],
    lights: [Buffer; SLOTS],
    pool: vk::DescriptorPool,
    scene_sets: [vk::DescriptorSet; SLOTS],
    io_sets: [[vk::DescriptorSet; IO_SETS]; SLOTS],
    fresh: bool,
}

impl View {
    fn images(&self) -> impl Iterator<Item = &Image> {
        [&self.trace, &self.variance, &self.prev_color]
            .into_iter()
            .chain(&self.accum)
            .chain(&self.accum_depth)
            .chain(&self.moments)
            .chain(&self.filtered)
            .chain(&self.filtered_variance)
    }
}

/// Screen and voxel GI. Install with `Renderer::set_extension` before loading
/// scenes.
// shortcut: retained meshes and moving bodies are not voxelized, so terrain and
// physics only appear on screen. Follow `retained_changed` and `Lux::sync` to add them.
pub struct ScreenVoxel {
    pub settings: ScreenVoxelSettings,
    triangles: Vec<Triangle>,
    origin: Vec3,
    cell: f32,
    count: u32,
    cells: Buffer,
    voxels: Buffer,
    radiance: Buffer,
    no_capture: Buffer,
    pipelines: Option<Pipelines>,
    view: Option<View>,
    targets: Option<Targets>,
    command_pool: vk::CommandPool,
    timer: Option<PassTimer>,
    last_frame: Option<u64>,
    last_viewport: [u32; 4],
    frame: u32,
    /// The lights last frame, and frames since they last changed while that
    /// is recent enough to keep voxel history short.
    previous_lights: Vec<u8>,
    light_age: Option<u32>,
    capture_requested: bool,
    readback: Option<Readback>,
    capture: Option<Capture>,
    started: Instant,
}

impl ScreenVoxel {
    pub fn new(settings: ScreenVoxelSettings) -> Self {
        Self {
            settings,
            triangles: Vec::new(),
            origin: Vec3::ZERO,
            cell: 1.0,
            count: 0,
            cells: Buffer::default(),
            voxels: Buffer::default(),
            radiance: Buffer::default(),
            no_capture: Buffer::default(),
            pipelines: None,
            view: None,
            targets: None,
            command_pool: vk::CommandPool::null(),
            timer: None,
            last_frame: None,
            last_viewport: [0; 4],
            frame: 1,
            previous_lights: Vec::new(),
            light_age: None,
            capture_requested: false,
            readback: None,
            capture: None,
            started: Instant::now(),
        }
    }

    /// Voxelize what has loaded and upload it.
    unsafe fn upload_voxels(&mut self, gpu: &mut Gpu) -> Result<()> {
        let grid = voxelize(&self.triangles, GRID);
        (self.origin, self.cell) = (grid.origin, grid.cell);
        self.count = grid.voxels.len() as u32;
        let pool = &mut self.command_pool;
        upload(
            gpu,
            pool,
            &mut self.cells,
            "voxel cells",
            bytemuck::cast_slice(&grid.cells),
        )?;
        // A buffer may not be empty.
        let voxels: &[[u32; 4]] = if grid.voxels.is_empty() {
            &[[0; 4]]
        } else {
            &grid.voxels
        };
        upload(
            gpu,
            pool,
            &mut self.voxels,
            "voxels",
            bytemuck::cast_slice(voxels),
        )?;
        let dark = vec![0u8; voxels.len() * 64];
        upload(gpu, pool, &mut self.radiance, "voxel radiance", &dark)?;
        if self.view.is_some() {
            for slot in 0..SLOTS {
                self.write_scene_set(gpu.device, slot, self.no_capture.buffer);
            }
        }
        Ok(())
    }

    unsafe fn create_pipelines(gpu: &mut Gpu) -> Result<Pipelines> {
        let device = gpu.device;
        let set_layout = |bindings: &[(u32, D)]| {
            let bindings: Vec<_> = bindings
                .iter()
                .map(|&(binding, kind)| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(binding)
                        .descriptor_type(kind)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        };
        let scene_layout = set_layout(SCENE_BINDINGS)?;
        let io_layout = set_layout(IO_BINDINGS)?;
        // Set 2 is the renderer's frame set: materials and textures.
        let layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&[
                scene_layout,
                io_layout,
                gpu.frame_set_layout,
            ]),
            None,
        )?;
        let mut passes = HashMap::new();
        for &(name, spirv) in PASSES {
            let entry = CString::new(name)?;
            let pipeline =
                gpu::compute_pipeline(device, vk::PipelineCache::null(), layout, spirv, &entry)?;
            passes.insert(name, pipeline);
        }
        Ok(Pipelines {
            scene_layout,
            io_layout,
            layout,
            passes,
        })
    }

    unsafe fn create_view(&self, gpu: &mut Gpu, targets: &Targets) -> Result<View> {
        let device = gpu.device;
        let pipelines = self.pipelines.as_ref().unwrap();
        let extent = targets.extent;
        let trace = vk::Extent2D {
            width: (extent.width / 4).max(1),
            height: (extent.height / 4).max(1),
        };
        use vk::ImageUsageFlags as U;
        let mut image = |name: &str, format, size| {
            gpu::create_image(
                device,
                gpu.allocator,
                name,
                format,
                size,
                1,
                U::STORAGE | U::SAMPLED,
                vk::ImageAspectFlags::COLOR,
            )
        };
        let (rgba, r) = (vk::Format::R16G16B16A16_SFLOAT, vk::Format::R32_SFLOAT);
        let trace_image = image("voxel gi trace", rgba, trace)?;
        let accum = [
            image("voxel gi accum", rgba, trace)?,
            image("voxel gi accum", rgba, trace)?,
        ];
        let accum_depth = [
            image("voxel gi depth", r, trace)?,
            image("voxel gi depth", r, trace)?,
        ];
        let moments = [
            image("voxel gi moments", rgba, trace)?,
            image("voxel gi moments", rgba, trace)?,
        ];
        let variance = image("voxel gi variance", r, trace)?;
        let filtered = [
            image("voxel gi filtered", rgba, trace)?,
            image("voxel gi filtered", rgba, trace)?,
        ];
        let filtered_variance = [
            image("voxel gi variance", r, trace)?,
            image("voxel gi variance", r, trace)?,
        ];
        let prev_color = image("voxel gi previous color", rgba, extent)?;

        let mut host = |name: &str, size: usize, usage| {
            gpu::create_buffer(
                device,
                gpu.allocator,
                name,
                size as u64,
                usage,
                MemoryLocation::CpuToGpu,
            )
        };
        let uniform = vk::BufferUsageFlags::UNIFORM_BUFFER;
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let frame_bytes = mem::size_of::<FrameData>();
        let frame = [
            host("voxel gi frame", frame_bytes, uniform)?,
            host("voxel gi frame", frame_bytes, uniform)?,
        ];
        let light_bytes = verity_render::MAX_LIGHTS * mem::size_of::<GpuLight>();
        let lights = [
            host("voxel gi lights", light_bytes, storage)?,
            host("voxel gi lights", light_bytes, storage)?,
        ];

        let count =
            |list: &[(u32, D)], kind: D| list.iter().filter(|(_, k)| *k == kind).count() as u32;
        let sizes: Vec<_> = [
            D::UNIFORM_BUFFER,
            D::STORAGE_BUFFER,
            D::SAMPLED_IMAGE,
            D::STORAGE_IMAGE,
        ]
        .map(|kind| {
            vk::DescriptorPoolSize::default().ty(kind).descriptor_count(
                SLOTS as u32
                    * (count(SCENE_BINDINGS, kind) + IO_SETS as u32 * count(IO_BINDINGS, kind)),
            )
        })
        .into_iter()
        .filter(|s| s.descriptor_count > 0)
        .collect();
        let pool = device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets((SLOTS * (IO_SETS + 1)) as u32)
                .pool_sizes(&sizes),
            None,
        )?;
        let mut scene_sets = [vk::DescriptorSet::null(); SLOTS];
        let mut io_sets = [[vk::DescriptorSet::null(); IO_SETS]; SLOTS];
        for slot in 0..SLOTS {
            scene_sets[slot] = device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&[pipelines.scene_layout]),
            )?[0];
            io_sets[slot].copy_from_slice(
                &device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&[pipelines.io_layout; IO_SETS]),
                )?,
            );
        }
        Ok(View {
            trace_extent: [trace.width, trace.height],
            trace: trace_image,
            accum,
            accum_depth,
            moments,
            variance,
            filtered,
            filtered_variance,
            prev_color,
            frame,
            lights,
            pool,
            scene_sets,
            io_sets,
            fresh: true,
        })
    }

    unsafe fn write_scene_set(&self, device: &ash::Device, slot: usize, capture: vk::Buffer) {
        let (view, targets) = (self.view.as_ref().unwrap(), self.targets.unwrap());
        let general = vk::ImageLayout::GENERAL;
        let color = Res::View(view.prev_color.view, general);
        write_set(
            device,
            view.scene_sets[slot],
            SCENE_BINDINGS,
            &[
                (0, Res::Buffer(view.frame[slot].buffer)),
                (
                    1,
                    Res::View(targets.depth, vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL),
                ),
                (2, Res::View(targets.gbuffer, general)),
                (3, color),
                (4, Res::Buffer(self.cells.buffer)),
                (5, Res::Buffer(self.voxels.buffer)),
                (6, Res::Buffer(self.radiance.buffer)),
                (10, Res::Buffer(view.lights[slot].buffer)),
                (11, Res::View(targets.hdr, general)),
                (12, color),
                (13, Res::Buffer(capture)),
            ],
        );
    }

    /// Point every pass at this view's images. A slot's frame writes history
    /// `cur` and reads `prev`.
    unsafe fn write_io_sets(&self, device: &ash::Device) {
        let s = self.view.as_ref().unwrap();
        let general = vk::ImageLayout::GENERAL;
        for slot in 0..SLOTS {
            let (cur, prev) = (slot, 1 - slot);
            // Unused slots hold images of the format the binding declares.
            let io = |ins: [&Image; 4], outs: [&Image; 4]| {
                let composite = [
                    &s.filtered[0],
                    &s.accum_depth[cur],
                    &s.filtered[1],
                    &s.accum_depth[prev],
                ];
                ins.into_iter()
                    .chain(outs)
                    .chain(composite)
                    .enumerate()
                    .map(|(i, image)| (i as u32, Res::View(image.view, general)))
                    .collect::<Vec<_>>()
            };
            let atrous = |light: &Image, variance: &Image, i: usize| {
                io(
                    [light, variance, &s.accum_depth[cur], &s.trace],
                    [
                        &s.filtered[i],
                        &s.filtered_variance[i],
                        &s.moments[prev],
                        &s.accum_depth[prev],
                    ],
                )
            };
            let groups = [
                io(
                    [
                        &s.accum[prev],
                        &s.accum_depth[prev],
                        &s.moments[prev],
                        &s.filtered_variance[1],
                    ],
                    [
                        &s.trace,
                        &s.variance,
                        &s.filtered[0],
                        &s.filtered_variance[0],
                    ],
                ),
                io(
                    [
                        &s.trace,
                        &s.accum[prev],
                        &s.accum_depth[prev],
                        &s.moments[prev],
                    ],
                    [
                        &s.accum[cur],
                        &s.accum_depth[cur],
                        &s.moments[cur],
                        &s.variance,
                    ],
                ),
                atrous(&s.accum[cur], &s.variance, 0),
                atrous(&s.filtered[0], &s.filtered_variance[0], 1),
                atrous(&s.filtered[1], &s.filtered_variance[1], 0),
                // The composite reads only bindings 8 and up.
                io(
                    [&s.trace, &s.trace, &s.trace, &s.trace],
                    [
                        &s.trace,
                        &s.variance,
                        &s.filtered[1],
                        &s.filtered_variance[1],
                    ],
                ),
            ];
            for (set, entries) in s.io_sets[slot].iter().zip(&groups) {
                write_set(device, *set, IO_BINDINGS, entries);
            }
        }
    }

    unsafe fn destroy_view(
        device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
        view: View,
    ) {
        device.destroy_descriptor_pool(view.pool, None);
        let images = [view.trace, view.variance, view.prev_color]
            .into_iter()
            .chain(view.accum)
            .chain(view.accum_depth)
            .chain(view.moments)
            .chain(view.filtered)
            .chain(view.filtered_variance);
        for image in images {
            gpu::destroy_image(device, allocator, image);
        }
        for buffer in view.frame.into_iter().chain(view.lights) {
            gpu::destroy_buffer(device, allocator, buffer);
        }
    }

    /// Finish a capture whose frame the GPU is done with, then start any
    /// requested one, on this slot since its last frame is done.
    unsafe fn tend_capture(
        &mut self,
        gpu: &mut Gpu,
        frame: &FrameContext,
        slot: usize,
    ) -> Result<()> {
        if self
            .readback
            .as_ref()
            .is_some_and(|r| frame.frame_number > r.frame && r.frame & 1 == slot as u64)
        {
            let readback = self.readback.take().unwrap();
            self.write_scene_set(gpu.device, slot, self.no_capture.buffer);
            self.capture = Some(Capture {
                width: readback.width,
                height: readback.height,
                pixels: bytemuck::cast_slice(readback.buffer.read()).to_vec(),
            });
            gpu::destroy_buffer(gpu.device, gpu.allocator, readback.buffer);
        }
        if self.capture_requested && self.readback.is_none() {
            self.capture_requested = false;
            let [_, _, width, height] = frame.viewport;
            let buffer = gpu::create_buffer(
                gpu.device,
                gpu.allocator,
                "voxel gi capture",
                u64::from(width) * u64::from(height) * 16,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::GpuToCpu,
            )?;
            self.write_scene_set(gpu.device, slot, buffer.buffer);
            self.readback = Some(Readback {
                buffer,
                frame: frame.frame_number,
                width,
                height,
            });
        }
        Ok(())
    }

    unsafe fn record_frame(&mut self, gpu: &mut Gpu, frame: &FrameContext) -> Result<()> {
        let device = gpu.device;
        let slot = (frame.frame_number & 1) as usize;
        self.tend_capture(gpu, frame, slot)?;
        let lights: Vec<_> = frame
            .lights
            .iter()
            .filter(|l| l.radius > 0.0 && l.intensity > 0.0)
            .map(|l| GpuLight {
                position: [
                    l.position[0],
                    l.position[1],
                    l.position[2],
                    1.0 / (l.radius * l.radius),
                ],
                color: (Vec3::from_array(l.color) * (l.intensity / PI))
                    .extend(0.0)
                    .to_array(),
            })
            .collect();
        let mut key = bytemuck::cast_slice::<_, u8>(&lights).to_vec();
        if let Some(sun) = frame.sun {
            key.extend_from_slice(bytemuck::cast_slice(&[sun.direction, sun.color]));
        }
        if key != self.previous_lights {
            self.previous_lights = key;
            self.light_age = Some(0);
        } else {
            self.light_age = self.light_age.map(|age| age + 1).filter(|&age| changed_history(age) < 64.0);
        }
        let settings = self.settings;
        let [x, y, width, height] = frame.viewport;
        // History needs the frame right before this one, in the same place.
        let unbroken = self.last_frame.is_some_and(|f| f + 1 == frame.frame_number)
            && self.last_viewport == frame.viewport;
        self.last_frame = Some(frame.frame_number);
        self.last_viewport = frame.viewport;
        let view = self.view.as_mut().unwrap();
        let pipelines = self.pipelines.as_ref().unwrap();

        // The shaders' clip space is Y-up; the renderer's is flipped for Vulkan.
        let flip = Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0));
        let clip = flip * frame.view_proj;
        let trace = [
            (width / 4).max(1).min(view.trace_extent[0]),
            (height / 4).max(1).min(view.trace_extent[1]),
        ];
        let sun = frame.sun.filter(|s| s.color.max_element() > 0.0);
        let mut data: FrameData = bytemuck::Zeroable::zeroed();
        data.clip_from_world = clip.to_cols_array_2d();
        data.world_from_clip = clip.inverse().to_cols_array_2d();
        data.prev_clip_from_world = (flip * frame.previous_view_proj).to_cols_array_2d();
        data.camera = frame.eye.extend(frame.near).to_array();
        data.sizes = [width, height, trace[0], trace[1]];
        data.origin = [x, y, 0, 0];
        data.misc = [
            (self.frame % 65536) as f32,
            settings.max_distance.max(0.1),
            settings.intensity,
            0.0,
        ];
        data.sun_direction = sun.map_or([0.0; 4], |s| {
            (-s.direction.normalize()).extend(1.0).to_array()
        });
        data.sun_color = sun.map_or([0.0; 4], |s| (s.color / PI).extend(0.0).to_array());
        data.sky = frame
            .sky
            .extend(settings.world_ray_distance.max(0.1))
            .to_array();
        data.ground = frame.ground.extend(0.0).to_array();
        data.counts = [self.count, GRID, lights.len() as u32, 0];
        data.cache = [self.frame, 0, 0, settings.rays_per_texel.clamp(1, 4)];
        data.extra = [
            0.0,
            self.light_age.map_or(0.0, changed_history),
            f32::from(u8::from(unbroken)),
            0.0,
        ];
        data.reflection = self.origin.extend(self.cell).to_array();
        data.fade = [1.0; 4];
        view.frame[slot].write(bytemuck::bytes_of(&data));
        if !lights.is_empty() {
            view.lights[slot].write(bytemuck::cast_slice(&lights));
        }
        self.frame = self.frame.wrapping_add(1).max(1);

        use vk::{AccessFlags2 as A, PipelineStageFlags2 as S};
        let cmd = frame.cmd;
        let timer = std::cell::RefCell::new(self.timer.as_mut().unwrap());
        timer.borrow_mut().begin(device, cmd, slot)?;
        let barrier = || {
            let barrier = [vk::MemoryBarrier2::default()
                .src_stage_mask(S::COMPUTE_SHADER)
                .src_access_mask(A::SHADER_WRITE)
                .dst_stage_mask(S::COMPUTE_SHADER)
                .dst_access_mask(A::SHADER_READ | A::SHADER_WRITE)];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&barrier),
            );
        };
        if view.fresh {
            let transitions: Vec<_> = view
                .images()
                .map(|image| {
                    gpu::image_barrier(
                        image.image,
                        vk::ImageAspectFlags::COLOR,
                        S::NONE,
                        A::NONE,
                        S::COMPUTE_SHADER,
                        A::SHADER_READ | A::SHADER_WRITE,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::GENERAL,
                    )
                })
                .collect();
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&transitions),
            );
            view.fresh = false;
        }
        barrier();

        let dispatch = |name: &'static str, set: vk::DescriptorSet, [x, y]: [u32; 2]| {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipelines.passes[name]);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipelines.layout,
                0,
                &[view.scene_sets[slot], set, frame.frame_set],
                &[],
            );
            device.cmd_dispatch(cmd, x.max(1), y.max(1), 1);
            barrier();
            timer.borrow_mut().stamp(device, cmd, slot, name);
        };
        let groups = |[x, y]: [u32; 2]| [x.div_ceil(8), y.div_ceil(8)];
        let sets = view.io_sets[slot];
        if self.count > 0 {
            dispatch("inject", sets[0], [self.count.div_ceil(4 * 64), 1]);
        }
        let passes = ["trace", "temporal", "atrous1", "atrous2", "atrous4"];
        for (name, &set) in passes.into_iter().zip(&sets) {
            dispatch(name, set, groups(trace));
        }
        dispatch("composite", sets[5], groups([width, height]));
        if let Some(readback) = &self.readback
            && readback.frame == frame.frame_number
        {
            dispatch("capture", sets[5], groups([width, height]));
            let host = [vk::MemoryBarrier2::default()
                .src_stage_mask(S::COMPUTE_SHADER)
                .src_access_mask(A::SHADER_WRITE)
                .dst_stage_mask(S::HOST)
                .dst_access_mask(A::HOST_READ)];
            device
                .cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&host));
        }
        Ok(())
    }
}

impl GiMethod for ScreenVoxel {
    fn status(&self) -> GiStatus {
        match self.pipelines {
            Some(_) => GiStatus::Ready {
                compile_time: Duration::ZERO,
            },
            None => GiStatus::Compiling {
                since: self.started,
            },
        }
    }

    fn gpu_time(&self) -> Option<Duration> {
        self.timer.as_ref().and_then(PassTimer::total)
    }

    fn pass_times(&self) -> &[(&'static str, f32)] {
        self.timer.as_ref().map_or(&[], |t| &t.times)
    }

    fn request_capture(&mut self) {
        self.capture_requested = true;
    }

    fn take_capture(&mut self) -> Option<Capture> {
        self.capture.take()
    }
}

impl RenderExtension for ScreenVoxel {
    fn active(&self) -> bool {
        self.pipelines.is_some()
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn scene_loaded(
        &mut self,
        gpu: &mut Gpu,
        scene: &SceneData,
        material_base: u32,
        _draw_base: u32,
    ) -> Result<()> {
        let placed = scene.draws.par_iter().flat_map_iter(|draw| {
            let primitive = &scene.primitives[draw.primitive as usize];
            // Blended surfaces are glass and the like: light passes them.
            let solid = scene.materials[primitive.material as usize].alpha_mode != AlphaMode::Blend;
            let first = primitive.first_index as usize;
            let count = if solid {
                primitive.index_count as usize
            } else {
                0
            };
            scene.indices[first..first + count]
                .chunks_exact(3)
                .map(move |corners| {
                    let corner = |i: usize| {
                        let [x, y, z, _] =
                            scene.vertices[(corners[i] + primitive.base_vertex) as usize].position;
                        draw.transform.transform_point3(Vec3::new(x, y, z))
                    };
                    Triangle {
                        corners: [corner(0), corner(1), corner(2)],
                        material: material_base + primitive.material,
                    }
                })
        });
        self.triangles.par_extend(placed);
        unsafe { self.upload_voxels(gpu) }
    }

    fn targets_changed(&mut self, gpu: &mut Gpu, targets: &Targets) -> Result<()> {
        unsafe {
            if self.pipelines.is_none() {
                self.pipelines = Some(Self::create_pipelines(gpu)?);
                self.timer = Some(PassTimer::new(gpu)?);
                self.no_capture = gpu::create_buffer(
                    gpu.device,
                    gpu.allocator,
                    "voxel gi no capture",
                    256,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    MemoryLocation::GpuOnly,
                )?;
            }
            if self.cells.buffer == vk::Buffer::null() {
                self.upload_voxels(gpu)?;
            }
            if let Some(view) = self.view.take() {
                Self::destroy_view(gpu.device, gpu.allocator, view);
            }
            self.view = Some(self.create_view(gpu, targets)?);
            self.targets = Some(*targets);
            self.write_io_sets(gpu.device);
            for slot in 0..SLOTS {
                self.write_scene_set(gpu.device, slot, self.no_capture.buffer);
            }
        }
        self.last_frame = None;
        Ok(())
    }

    fn record(&mut self, gpu: &mut Gpu, frame: &FrameContext) -> Result<()> {
        if self.view.is_none() || frame.viewport[2] == 0 || frame.viewport[3] == 0 {
            return Ok(());
        }
        unsafe { self.record_frame(gpu, frame) }
    }

    fn destroy(&mut self, gpu: &mut Gpu) {
        let device = gpu.device;
        unsafe {
            if let Some(view) = self.view.take() {
                Self::destroy_view(device, gpu.allocator, view);
            }
            if let Some(readback) = self.readback.take() {
                gpu::destroy_buffer(device, gpu.allocator, readback.buffer);
            }
            for buffer in [
                &mut self.cells,
                &mut self.voxels,
                &mut self.radiance,
                &mut self.no_capture,
            ] {
                gpu::destroy_buffer(device, gpu.allocator, mem::take(buffer));
            }
            if let Some(pipelines) = self.pipelines.take() {
                for pipeline in pipelines.passes.into_values() {
                    device.destroy_pipeline(pipeline, None);
                }
                device.destroy_pipeline_layout(pipelines.layout, None);
                device.destroy_descriptor_set_layout(pipelines.scene_layout, None);
                device.destroy_descriptor_set_layout(pipelines.io_layout, None);
            }
            if let Some(timer) = self.timer.take() {
                timer.destroy(device);
            }
            device.destroy_command_pool(self.command_pool, None);
            self.command_pool = vk::CommandPool::null();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn floor(z: f32, material: u32) -> [Triangle; 2] {
        let (a, b) = (Vec3::new(0.0, 0.0, z), Vec3::new(8.0, 0.0, z));
        let (c, d) = (Vec3::new(8.0, 8.0, z), Vec3::new(0.0, 8.0, z));
        [
            Triangle {
                corners: [a, b, c],
                material,
            },
            Triangle {
                corners: [a, c, d],
                material,
            },
        ]
    }

    #[test]
    fn a_floor_fills_one_layer_of_cells() {
        let grid = voxelize(&floor(1.0, 3), 16);
        // Every whole cell; the rim holds only part of one.
        assert!(grid.voxels.len() >= 14 * 14, "{}", grid.voxels.len());
        let layers: std::collections::HashSet<u32> =
            grid.voxels.iter().map(|v| v[0] / (16 * 16)).collect();
        assert_eq!(layers.len(), 1);
        for (i, voxel) in grid.voxels.iter().enumerate() {
            assert_eq!(grid.cells[voxel[0] as usize] as usize, i + 1);
            assert_eq!(voxel[1], 3);
            assert_eq!(voxel[2], pack_normal(Vec3::Z));
        }
    }

    #[test]
    fn opposite_faces_in_a_cell_keep_a_normal() {
        let mut triangles = floor(1.0, 0).to_vec();
        for t in floor(1.0, 0) {
            let [a, b, c] = t.corners;
            triangles.push(Triangle {
                corners: [a, c, b],
                material: 0,
            });
        }
        let grid = voxelize(&triangles, 16);
        assert!(grid.voxels.iter().all(|v| v[2] != 0));
    }

    #[test]
    fn a_cell_barely_clipped_is_often_left_open() {
        // A sliver through many cells: each holds a tiny area of surface.
        let (a, b) = (Vec3::ZERO, Vec3::new(8.0, 0.0, 0.0));
        let sliver = Triangle {
            corners: [a, b, Vec3::new(8.0, 0.01, 0.0)],
            material: 0,
        };
        let wide = Triangle {
            corners: [
                Vec3::new(0.0, 0.0, 8.0),
                Vec3::new(8.0, 8.0, 8.0),
                Vec3::new(0.0, 8.0, 8.0),
            ],
            material: 0,
        };
        let grid = voxelize(&[sliver, wide], 16);
        let open = grid.voxels.iter().filter(|v| v[0] / (16 * 16) < 8).count();
        assert!(open < 8, "{open} cells of 16 kept");
    }

    #[test]
    fn nothing_gives_an_empty_grid() {
        let grid = voxelize(&[], 8);
        assert!(grid.voxels.is_empty());
        assert_eq!(grid.cells.len(), 512);
    }

    #[test]
    fn normals_pack_as_signed_bytes() {
        assert_eq!(pack_normal(Vec3::X), 127);
        assert_eq!(pack_normal(-Vec3::Y), 129 << 8);
    }
}
