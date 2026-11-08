//! Screen and voxel GI: rays follow the depth buffer and take their light from
//! last frame's image; past what the screen shows they walk a voxel copy of the
//! scene, which keeps its own lighting and bounces. It needs no GI scene or
//! physics, and costs the same however many triangles the scene has.
//!
//! The voxels are a clipmap around the camera, voxelized on the GPU from the
//! loaded scenes' and the renderer's retained meshes' triangles: nested levels
//! of `GRID` cells a side, each twice the size of the last. As the camera moves
//! a level shifts by whole cells and rebuilds only the slabs that came in.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::{
    GiMethod, GiStatus,
    pxtgi::{Capture, FrameData, GpuLight, IO_BINDINGS, Readback, pack_ground},
    shared::{BLUE_NOISE, PassTimer, Res, upload, write_set},
};
use ash::vk;
use glam::{IVec3, Mat4, Vec3};
use gpu_allocator::MemoryLocation;
use rayon::prelude::*;
use std::{
    collections::HashMap, f32::consts::PI, ffi::CString, mem, time::Duration, time::Instant,
};
use verity_render::{
    FrameContext, Gpu, RenderExtension, RetainedChange, Targets,
    asset::{AlphaMode, SceneData},
    extension::Result,
    gpu::{self, Buffer, Image},
};

const SLOTS: usize = 2;
/// Cells along each side of a clipmap level. A power of two.
const GRID: i32 = 128;
/// Clipmap levels at most; `MAX_LEVELS` in voxel.slang.
const MAX_LEVELS: usize = 8;
/// Boxes of cells voxelized in a frame at most: three slabs a level.
const MAX_JOBS: usize = 3 * MAX_LEVELS;
/// Voxels the levels share.
const POOL: u32 = 1 << 20;

macro_rules! spirv {
    ($entry:literal) => {
        (
            $entry,
            include_bytes!(concat!(env!("OUT_DIR"), "/voxel.", $entry, ".spv")) as &[u8],
        )
    };
}

const PASSES: &[(&str, &[u8])] = &[
    spirv!("reset"),
    spirv!("clear"),
    spirv!("claim"),
    spirv!("accumulate"),
    spirv!("finalize"),
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
    (7, D::STORAGE_BUFFER),
    (8, D::STORAGE_BUFFER),
    (9, D::STORAGE_BUFFER),
    (10, D::STORAGE_BUFFER),
    (11, D::STORAGE_IMAGE),
    (12, D::STORAGE_IMAGE),
    (13, D::STORAGE_BUFFER),
    (14, D::STORAGE_BUFFER),
    (15, D::STORAGE_BUFFER),
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
    /// Voxel levels around the camera, 1..=8, each twice the size of the last.
    /// Read when the GI is installed.
    pub levels: u32,
    /// The finest level's cell. A level spans 128 cells.
    pub cell_size: f32,
}

impl Default for ScreenVoxelSettings {
    fn default() -> Self {
        Self {
            max_distance: 8.0,
            world_ray_distance: 30.0,
            rays_per_texel: 4,
            intensity: 1.0,
            levels: 5,
            cell_size: 0.125,
        }
    }
}

/// A triangle as the voxelizer reads it: three world-space corners, the first
/// one's w the material and the second's the terrain ground.
type GpuTriangle = [[f32; 4]; 3];

/// Cells a level moves by at least. Voxelizing a slab costs a pass over every
/// triangle, so fewer, thicker slabs are cheaper.
const STEP: i32 = 8;

/// Where a level of `cell`-sized cells starts, in cells, to centre on `eye`
/// within a `STEP`.
fn level_origin(eye: Vec3, cell: f32) -> IVec3 {
    let at = (eye / cell).floor().as_ivec3();
    at.div_euclid(IVec3::splat(STEP)) * STEP - IVec3::splat(GRID / 2)
}

/// A box of cells: low corner, and high corner exclusive.
type CellBox = (IVec3, IVec3);

/// The cells a level moving from `old` to `new` takes in, as boxes that do not
/// overlap: a slab for each axis it moved along.
fn scrolled(old: IVec3, new: IVec3) -> Vec<CellBox> {
    let moved = new - old;
    let (mut low, mut high) = (new, new + GRID);
    if moved.abs().max_element() >= GRID {
        return vec![(low, high)];
    }
    let mut slabs = Vec::new();
    for axis in 0..3 {
        let (mut slab_low, mut slab_high) = (low, high);
        if moved[axis] > 0 {
            slab_low[axis] = high[axis] - moved[axis];
            high[axis] = slab_low[axis];
        } else if moved[axis] < 0 {
            slab_high[axis] = low[axis] - moved[axis];
            low[axis] = slab_high[axis];
        } else {
            continue;
        }
        slabs.push((slab_low, slab_high));
    }
    slabs
}

/// The cells of a level at `origin` that a world box touches, grown by a cell.
fn touched(origin: IVec3, cell: f32, (lo, hi): (Vec3, Vec3)) -> Option<CellBox> {
    let low = ((lo / cell).floor().as_ivec3() - 1).max(origin);
    let high = ((hi / cell).floor().as_ivec3() + 2).min(origin + GRID);
    low.cmplt(high).all().then_some((low, high))
}

fn grow(bounds: &mut Option<(Vec3, Vec3)>, (lo, hi): (Vec3, Vec3)) {
    *bounds = Some(bounds.map_or((lo, hi), |(a, b)| (a.min(lo), b.max(hi))));
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
    /// Per slot: `clip` in voxel.slang.
    clip: [Buffer; SLOTS],
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
// shortcut: moving physics bodies are not voxelized, so they only appear on
// screen. Follow `Lux::sync` to add them.
pub struct ScreenVoxel {
    pub settings: ScreenVoxelSettings,
    /// Loaded scenes' triangles, then each retained mesh's and its bounds.
    triangles: Vec<GpuTriangle>,
    retained: HashMap<u64, (Vec<GpuTriangle>, (Vec3, Vec3))>,
    triangle_count: u32,
    /// Each level's low cell as last voxelized; none to voxelize it whole.
    origins: [Option<IVec3>; MAX_LEVELS],
    /// The finest cell and the levels the voxels were built with.
    built: (f32, usize),
    /// Where the triangles changed since the last voxelizing.
    changed: Option<(Vec3, Vec3)>,
    /// Whether the voxel pool needs setting up.
    reset: bool,
    triangle_buffer: Buffer,
    cells: Buffer,
    voxels: Buffer,
    radiance: Buffer,
    build: Buffer,
    free: Buffer,
    no_capture: Buffer,
    blue_noise: Buffer,
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
            retained: HashMap::new(),
            triangle_count: 0,
            origins: [None; MAX_LEVELS],
            built: (0.0, 0),
            changed: None,
            reset: true,
            triangle_buffer: Buffer::default(),
            cells: Buffer::default(),
            voxels: Buffer::default(),
            radiance: Buffer::default(),
            build: Buffer::default(),
            free: Buffer::default(),
            no_capture: Buffer::default(),
            blue_noise: Buffer::default(),
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

    /// The levels this GI keeps.
    fn levels(&self) -> usize {
        (self.settings.levels as usize).clamp(1, MAX_LEVELS)
    }

    /// Upload every triangle for the voxelizer to read.
    unsafe fn upload_triangles(&mut self, gpu: &mut Gpu) -> Result<()> {
        let retained = self.retained.values().flat_map(|(t, _)| t);
        let mut all: Vec<GpuTriangle> = self.triangles.iter().chain(retained).copied().collect();
        self.triangle_count = all.len() as u32;
        // A buffer may not be empty.
        if all.is_empty() {
            all.push([[0.0; 4]; 3]);
        }
        // shortcut: every change uploads every triangle; upload only the changed
        // meshes' if digging terrain stalls.
        let bytes = bytemuck::cast_slice(&all);
        upload(gpu, &mut self.command_pool, &mut self.triangle_buffer, "voxel triangles", bytes)?;
        if self.view.is_some() {
            for slot in 0..SLOTS {
                self.write_scene_set(gpu.device, slot, self.no_capture.buffer);
            }
        }
        Ok(())
    }

    /// The cells of every level and the voxels they share, set up by the first
    /// frame's `reset` pass.
    unsafe fn create_voxels(&mut self, gpu: &mut Gpu) -> Result<()> {
        let levels = self.levels();
        let cells = levels as u64 * (GRID as u64).pow(3) * 4;
        let pool = u64::from(POOL);
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        for (buffer, name, size) in [
            (&mut self.cells, "voxel cells", cells),
            (&mut self.voxels, "voxels", pool * 16),
            (&mut self.radiance, "voxel radiance", pool * 64),
            (&mut self.build, "voxel build", pool * 16),
            (&mut self.free, "free voxels", (pool + 1) * 4),
        ] {
            gpu::destroy_buffer(gpu.device, gpu.allocator, mem::take(buffer));
            *buffer = gpu::create_buffer(
                gpu.device,
                gpu.allocator,
                name,
                size,
                usage,
                MemoryLocation::GpuOnly,
            )?;
        }
        self.built = (self.settings.cell_size.max(1e-3), levels);
        self.origins = [None; MAX_LEVELS];
        self.reset = true;
        Ok(())
    }

    /// Each level's low cell for a camera at `eye`, and the boxes of cells to
    /// voxelize: what scrolled in, what changed, or all of a level not yet built.
    fn plan(&mut self, eye: Vec3) -> ([[i32; 4]; MAX_LEVELS], Vec<[[i32; 4]; 2]>) {
        let (cell, levels) = self.built;
        let changed = self.changed.take();
        let mut lows = [[0; 4]; MAX_LEVELS];
        let mut jobs = Vec::new();
        for (level, low) in lows.iter_mut().enumerate().take(levels) {
            let size = cell * (1 << level) as f32;
            let new = level_origin(eye, size);
            let mut boxes = match self.origins[level] {
                Some(old) => scrolled(old, new),
                None => vec![(new, new + GRID)],
            };
            if let Some(touched) = changed.and_then(|c| touched(new, size, c)) {
                // Boxes must not overlap, or cells would count triangles twice.
                boxes = if boxes.is_empty() {
                    vec![touched]
                } else {
                    vec![(new, new + GRID)]
                };
            }
            self.origins[level] = Some(new);
            *low = new.extend(0).to_array();
            for (low, high) in boxes {
                let cells = (high - low).element_product();
                jobs.push([low.extend(level as i32).to_array(), high.extend(cells).to_array()]);
            }
        }
        (lows, jobs)
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
        let clip_bytes = (MAX_LEVELS + 2 * MAX_JOBS) * 16;
        let clip = [
            host("voxel gi clipmap", clip_bytes, storage)?,
            host("voxel gi clipmap", clip_bytes, storage)?,
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
            clip,
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
                (7, Res::Buffer(view.clip[slot].buffer)),
                (8, Res::Buffer(self.triangle_buffer.buffer)),
                (9, Res::Buffer(self.build.buffer)),
                (10, Res::Buffer(view.lights[slot].buffer)),
                (11, Res::View(targets.hdr, general)),
                (12, color),
                (13, Res::Buffer(capture)),
                (14, Res::Buffer(self.blue_noise.buffer)),
                (15, Res::Buffer(self.free.buffer)),
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
        for buffer in view.frame.into_iter().chain(view.lights).chain(view.clip) {
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
        if self.built.0 != settings.cell_size.max(1e-3) {
            // Another cell size: build every level afresh.
            self.built.0 = settings.cell_size.max(1e-3);
            self.origins = [None; MAX_LEVELS];
        }
        let (lows, jobs) = self.plan(frame.eye);
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
        data.counts = [POOL, GRID as u32, lights.len() as u32, 0];
        data.cache = [
            self.frame,
            jobs.len() as u32,
            self.triangle_count,
            settings.rays_per_texel.clamp(1, 4),
        ];
        data.extra = [
            0.0,
            self.light_age.map_or(0.0, changed_history),
            f32::from(u8::from(unbroken)),
            0.0,
        ];
        data.reflection = [0.0, 0.0, self.built.1 as f32, self.built.0];
        data.fade = [1.0; 4];
        view.frame[slot].write(bytemuck::bytes_of(&data));
        if !lights.is_empty() {
            view.lights[slot].write(bytemuck::cast_slice(&lights));
        }
        let mut clip = lows.to_vec();
        clip.extend(jobs.iter().flatten());
        view.clip[slot].write(bytemuck::cast_slice(&clip));
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
        if self.reset {
            device.cmd_fill_buffer(cmd, self.cells.buffer, 0, vk::WHOLE_SIZE, 0);
            let filled = [vk::MemoryBarrier2::default()
                .src_stage_mask(S::ALL_TRANSFER)
                .src_access_mask(A::TRANSFER_WRITE)
                .dst_stage_mask(S::COMPUTE_SHADER)
                .dst_access_mask(A::SHADER_READ | A::SHADER_WRITE)];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&filled),
            );
            dispatch("reset", sets[0], [POOL.div_ceil(64), 1]);
            self.reset = false;
        }
        if let Some(most) = jobs.iter().map(|job| job[1][3] as u32).max() {
            let cells = [most.div_ceil(64), jobs.len() as u32];
            // More than a row of groups loop round; see `TRIANGLE_STRIDE`.
            let triangles = [self.triangle_count.div_ceil(64).min(65535), 1];
            dispatch("clear", sets[0], cells);
            if self.triangle_count > 0 {
                dispatch("claim", sets[0], triangles);
                dispatch("accumulate", sets[0], triangles);
            }
            dispatch("finalize", sets[0], cells);
        }
        dispatch("inject", sets[0], [POOL.div_ceil(4 * 64), 1]);
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
                    let corner = |i: usize, w: u32| {
                        let [x, y, z, _] =
                            scene.vertices[(corners[i] + primitive.base_vertex) as usize].position;
                        let p = draw.transform.transform_point3(Vec3::new(x, y, z));
                        p.extend(f32::from_bits(w)).to_array()
                    };
                    let material = material_base + primitive.material;
                    [corner(0, material), corner(1, 0), corner(2, 0)]
                })
        });
        self.triangles.par_extend(placed);
        self.origins = [None; MAX_LEVELS];
        unsafe { self.upload_triangles(gpu) }
    }

    fn retained_changed(
        &mut self,
        gpu: &mut Gpu,
        changes: &[RetainedChange],
        material: u32,
    ) -> Result<()> {
        for change in changes {
            if let Some((_, bounds)) = self.retained.remove(&change.key) {
                grow(&mut self.changed, bounds);
            }
            let Some(mesh) = change.mesh else { continue };
            grow(&mut self.changed, mesh.bounds);
            let triangles = mesh
                .indices
                .par_chunks_exact(3)
                .map(|corners| {
                    let corner = |i: usize, w: f32| {
                        let v = &mesh.vertices[corners[i] as usize];
                        [v.position[0], v.position[1], v.position[2], w]
                    };
                    let first = &mesh.vertices[corners[0] as usize];
                    let ground = pack_ground(first.position[3], first.normal[3]);
                    [
                        corner(0, f32::from_bits(material)),
                        corner(1, ground),
                        corner(2, 0.0),
                    ]
                })
                .collect();
            self.retained.insert(change.key, (triangles, mesh.bounds));
        }
        unsafe { self.upload_triangles(gpu) }
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
                let pool = &mut self.command_pool;
                upload(
                    gpu,
                    pool,
                    &mut self.blue_noise,
                    "voxel blue noise",
                    BLUE_NOISE,
                )?;
            }
            if self.cells.buffer == vk::Buffer::null() {
                self.create_voxels(gpu)?;
            }
            if self.triangle_buffer.buffer == vk::Buffer::null() {
                self.upload_triangles(gpu)?;
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
                &mut self.triangle_buffer,
                &mut self.cells,
                &mut self.voxels,
                &mut self.radiance,
                &mut self.build,
                &mut self.free,
                &mut self.no_capture,
                &mut self.blue_noise,
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

    #[test]
    fn levels_centre_on_the_camera() {
        for eye in [Vec3::new(1.0, -1.0, 0.0), Vec3::splat(-100.3), Vec3::splat(77.7)] {
            let origin = level_origin(eye, 0.5);
            assert_eq!(origin % STEP, IVec3::ZERO);
            let at = (eye / 0.5).floor().as_ivec3() - origin;
            assert!(at.cmpge(IVec3::splat(GRID / 2)).all(), "{eye}");
            assert!(at.cmplt(IVec3::splat(GRID / 2 + STEP)).all(), "{eye}");
        }
    }

    #[test]
    fn a_still_level_rebuilds_nothing() {
        assert!(scrolled(IVec3::ONE, IVec3::ONE).is_empty());
    }

    #[test]
    fn scrolled_slabs_cover_the_new_cells_once() {
        let old = IVec3::new(0, 0, 0);
        for new in [IVec3::new(3, 0, 0), IVec3::new(-2, 5, 1), IVec3::new(200, 0, 0)] {
            let slabs = scrolled(old, new);
            let volume: i32 = slabs.iter().map(|(l, h)| (*h - *l).element_product()).sum();
            // Cells in the new box and not the old one.
            let kept = (old + GRID).min(new + GRID) - old.max(new);
            let overlap = kept.max(IVec3::ZERO).element_product();
            assert_eq!(volume, GRID.pow(3) - overlap, "{new}");
            for (low, high) in &slabs {
                assert!(low.cmpge(new).all() && high.cmple(new + GRID).all());
                // None of a slab was in the old box.
                let inside = (*high).min(old + GRID) - (*low).max(old);
                assert!(inside.cmple(IVec3::ZERO).any(), "{new}: {low} {high}");
            }
        }
    }

    #[test]
    fn a_change_touches_only_its_cells() {
        let origin = IVec3::splat(-64);
        let (low, high) = touched(origin, 1.0, (Vec3::splat(0.5), Vec3::splat(2.5))).unwrap();
        assert_eq!((low, high), (IVec3::splat(-1), IVec3::splat(4)));
        assert!(touched(origin, 1.0, (Vec3::splat(500.0), Vec3::splat(501.0))).is_none());
    }
}
