//! PXTGI: global illumination traced against the physics scene, as a render
//! extension of its own beside [`crate::cascades::SplitRadianceCascades`].
//!
//! A few rays per quarter-resolution texel follow what is on screen, then
//! continue against a [`RayScene`]: the rigid colliders and soft bodies of a
//! [`PhysicsWorld`], plus static meshes from loaded glTF scenes. Off-screen hits
//! light themselves from a hashed radiance cache, which also carries further
//! bounces. Filtering is temporal then a-trous, and reflections run the same way
//! at half resolution.
//!
//! Physics drives more than the geometry. A resting scene uploads nothing and
//! refits nothing. Moving bodies report the region they disturbed, and only
//! there does the GI drop its history and refresh its cache every frame, so a
//! falling box changes the light around it without noise anywhere else.
//!
//! Install with `Renderer::set_extension` before loading scenes, and call
//! [`Pxtgi::sync`] after each physics step. Everything is in `shaders/pxtgi.slang`.

// Every function here is Vulkan calls; an unsafe fn body is one unsafe context.
#![allow(unsafe_op_in_unsafe_fn)]

use crate::{
    GiMethod, GiStatus,
    shared::{PassTimer, Patch, Res, upload, upload_ranges, write_set},
};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Affine3A, Mat4, Vec3};
use gpu_allocator::MemoryLocation;
use rayon::prelude::*;
use std::{
    collections::{HashMap, hash_map::Entry},
    f32::consts::PI,
    ffi::CString,
    mem,
    sync::Arc,
};
use verity_core::World;
use verity_physics::{Bounds, PhysicsWorld, RayScene, RayShape, RaySource, TriMesh};
use verity_render::{
    FrameContext, Gpu, RenderExtension, RetainedChange, RetainedMesh, Targets,
    asset::{AlphaMode, SceneData},
    extension::Result,
    gpu::{self, Buffer, Image},
};

const SLOTS: usize = 2;
/// Radiance cache cells; the shader masks hashes with this less one.
const CACHE_SLOTS: u32 = 1 << 17;
/// Cells a dispatch row covers, as `cache_gather` indexes them.
const CACHE_ROW: u32 = 512;
const CELL_BYTES: u64 = 80;
/// Regions of movement the shaders test each point against.
const MAX_MOVED: usize = 16;
/// Frames of history the GI keeps right after a light changes. It grows back
/// by one a frame up to [`STEADY_HISTORY`], where the cache also calms down.
const CHANGED_HISTORY: u32 = 3;
const STEADY_HISTORY: u32 = 32;
/// How far around itself a moved thing changes the light, as a share of its
/// size, and at least: shadows and bounce light reach past the thing itself.
const MOVED_REACH: f32 = 0.5;
const MOVED_REACH_MIN: f32 = 0.5;
/// The most remembered light may grow in one frame when sources brighten.
const MAX_FADE: f32 = 4.0;
/// Frames a region stays stale after the motion: light settles over several,
/// so one frame's help would barely show.
const MOVED_FRAMES: u32 = 16;
/// Albedo of what has no material of its own, such as a soft body.
const GREY: [f32; 3] = [0.6, 0.6, 0.6];
/// `NO_MATERIAL` in the shader.
const NO_MATERIAL: u32 = u32::MAX;

macro_rules! spirv {
    ($entry:literal) => {
        (
            $entry,
            include_bytes!(concat!(env!("OUT_DIR"), "/pxtgi.", $entry, ".spv")).as_slice(),
        )
    };
}

/// Every pass of a frame, by entry point.
const PASSES: &[(&str, &[u8])] = &[
    spirv!("cache_gather"),
    spirv!("cache_shade"),
    spirv!("trace"),
    spirv!("temporal"),
    spirv!("atrous1"),
    spirv!("atrous2"),
    spirv!("atrous4"),
    spirv!("reflect_trace"),
    spirv!("reflect_temporal"),
    spirv!("reflect_filter1"),
    spirv!("reflect_filter2"),
    spirv!("composite"),
    spirv!("capture"),
    spirv!("reference"),
];

use vk::DescriptorType as D;
/// Set 0: the frame and the scene.
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
];
/// Set 1: one pass's inputs and outputs.
pub(crate) const IO_BINDINGS: &[(u32, D)] = &[
    (0, D::SAMPLED_IMAGE),
    (1, D::SAMPLED_IMAGE),
    (2, D::SAMPLED_IMAGE),
    (3, D::SAMPLED_IMAGE),
    (4, D::STORAGE_IMAGE),
    (5, D::STORAGE_IMAGE),
    (6, D::STORAGE_IMAGE),
    (7, D::STORAGE_IMAGE),
    (8, D::SAMPLED_IMAGE),
    (9, D::SAMPLED_IMAGE),
    (10, D::SAMPLED_IMAGE),
    (11, D::SAMPLED_IMAGE),
];
/// Per slot: trace, temporal, three a-trous, four reflection, composite.
const IO_SETS: usize = 10;

/// How smooth surfaces get their reflections.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PxtgiReflections {
    Off,
    /// What is on screen, the hemisphere otherwise.
    Screen,
    /// What is on screen, then the physics scene.
    #[default]
    Traced,
}

/// What the view shows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PxtgiDebug {
    #[default]
    Off = 0,
    /// Green where a depth round trip lands back on its pixel.
    RoundTrip = 1,
    /// The lit scene with light from the physics scene alone: no screen rays.
    WorldOnly = 2,
    /// The fraction of rays that hit the physics scene.
    WorldHits = 3,
    /// World position in one-metre bands.
    WorldPosition = 4,
    /// The bounce light, before albedo.
    BounceLight = 5,
    Reflections = 6,
    /// Diffuse bounce light by plain path tracing, converging for as long as the
    /// camera stays still, to measure methods against. Scene hits use the same
    /// mean colours as PXTGI, and no cache or filter is involved.
    Reference = 7,
}

/// PXTGI controls. Distances are in metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PxtgiSettings {
    /// How far rays follow what is on screen before continuing against the scene.
    pub max_distance: f32,
    /// How far rays go against the scene. Surfaces further away do not bounce light.
    pub world_ray_distance: f32,
    /// Rays each traced texel sends per frame, clamped to 1..=4. More settle
    /// faster and flicker less, at about proportional cost.
    pub rays_per_texel: u32,
    /// Light bounces more than once through the radiance cache. Off, every ray
    /// that leaves the screen lights what it hits from scratch.
    pub multi_bounce: bool,
    pub reflections: PxtgiReflections,
    /// Surfaces with a perceptual roughness below this get traced reflections,
    /// faded out over the last quarter of it. Clamped to 0..=1.
    pub reflection_roughness_cutoff: f32,
    /// Scale of the bounce light.
    pub intensity: f32,
    pub debug: PxtgiDebug,
}

impl Default for PxtgiSettings {
    fn default() -> Self {
        Self {
            max_distance: 8.0,
            world_ray_distance: 30.0,
            rays_per_texel: 2,
            multi_bounce: true,
            reflections: PxtgiReflections::Traced,
            reflection_roughness_cutoff: 0.5,
            intensity: 1.0,
            debug: PxtgiDebug::Off,
        }
    }
}

/// `Frame` in pxtgi.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct FrameData {
    pub(crate) clip_from_world: [[f32; 4]; 4],
    pub(crate) world_from_clip: [[f32; 4]; 4],
    pub(crate) prev_clip_from_world: [[f32; 4]; 4],
    pub(crate) camera: [f32; 4],
    pub(crate) sizes: [u32; 4],
    pub(crate) origin: [u32; 4],
    pub(crate) misc: [f32; 4],
    pub(crate) sun_direction: [f32; 4],
    pub(crate) sun_color: [f32; 4],
    pub(crate) sky: [f32; 4],
    pub(crate) ground: [f32; 4],
    pub(crate) counts: [u32; 4],
    pub(crate) cache: [u32; 4],
    pub(crate) extra: [f32; 4],
    pub(crate) reflection: [f32; 4],
    pub(crate) fade: [f32; 4],
    pub(crate) moved_min: [[f32; 4]; MAX_MOVED],
    pub(crate) moved_max: [[f32; 4]; MAX_MOVED],
}

/// `Light` in pxtgi.slang.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub(crate) struct GpuLight {
    pub(crate) position: [f32; 4],
    pub(crate) color: [f32; 4],
}

/// `Primitive` in pxtgi.slang.
type GpuPrimitive = [[f32; 4]; 5];

const KIND_BOX: f32 = 0.0;
const KIND_SPHERE: f32 = 1.0;
const KIND_MESH: f32 = 2.0;
const KIND_SOFT: f32 = 3.0;

/// A captured frame: linear RGBA, row by row from the top.
#[derive(Clone, Debug)]
pub struct Capture {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[f32; 4]>,
}

/// What a frame slot's copy of the moving scene holds.
#[derive(Default)]
struct SlotScene {
    primitives: Buffer,
    nodes: Buffer,
    soft: Buffer,
    /// [`Packed::revision`] these hold, once they hold any.
    revision: Option<u64>,
}

/// A capture on its way back from the GPU.
pub(crate) struct Readback {
    pub(crate) buffer: Buffer,
    pub(crate) frame: u64,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

/// The scene as the shaders read it.
#[derive(Default)]
struct Packed {
    primitives: Vec<GpuPrimitive>,
    nodes: Vec<[[f32; 4]; 2]>,
    soft: Vec<[f32; 4]>,
    revision: u64,
}

/// A static item's place in the geometry buffer and its material.
#[derive(Clone, Copy, Default)]
struct StaticMesh {
    nodes_at: i32,
    triangles_at: i32,
    material: u32,
}

/// A retained mesh's static number and the words it may fill in the geometry.
#[derive(Clone, Copy)]
struct RetainedSlot {
    number: u32,
    start: usize,
    capacity: usize,
}

struct Pipelines {
    scene_layout: vk::DescriptorSetLayout,
    io_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    passes: HashMap<&'static str, vk::Pipeline>,
}

/// Everything sized to the render targets.
struct View {
    trace_extent: [u32; 2],
    reflect_extent: [u32; 2],
    trace: Image,
    accum: [Image; 2],
    accum_depth: [Image; 2],
    moments: [Image; 2],
    variance: Image,
    filtered: [Image; 2],
    filtered_variance: [Image; 2],
    reflect_trace: Image,
    reflect_accum: [Image; 2],
    reflect_depth: [Image; 2],
    reflect_filtered: [Image; 2],
    prev_color: Image,
    frame: [Buffer; SLOTS],
    lights: [Buffer; SLOTS],
    pool: vk::DescriptorPool,
    scene_sets: [vk::DescriptorSet; SLOTS],
    io_sets: [[vk::DescriptorSet; IO_SETS]; SLOTS],
    /// Images still need their first layout, and the cache clearing.
    fresh: bool,
}

impl View {
    fn images(&self) -> impl Iterator<Item = &Image> {
        [
            &self.trace,
            &self.variance,
            &self.reflect_trace,
            &self.prev_color,
        ]
        .into_iter()
        .chain(&self.accum)
        .chain(&self.accum_depth)
        .chain(&self.moments)
        .chain(&self.filtered)
        .chain(&self.filtered_variance)
        .chain(&self.reflect_accum)
        .chain(&self.reflect_depth)
        .chain(&self.reflect_filtered)
    }
}

/// See the [module documentation](self).
pub struct Pxtgi {
    pub settings: PxtgiSettings,
    scene: RayScene,
    albedos: HashMap<RaySource, [f32; 3]>,
    statics: Vec<StaticMesh>,
    retained: HashMap<u64, RetainedSlot>,
    /// Mesh nodes and triangles, kept to append later scenes to.
    geometry: Vec<[f32; 4]>,
    /// Where free space for retained meshes starts in `geometry`.
    geometry_free: usize,
    geometry_buffer: Buffer,
    packed: Packed,
    slots: [SlotScene; SLOTS],
    cache_keys: Buffer,
    cells: Buffer,
    /// Bound while no capture is, as the shader declares the buffer.
    no_capture: Buffer,
    pipelines: Option<Pipelines>,
    view: Option<View>,
    targets: Option<Targets>,
    command_pool: vk::CommandPool,
    /// Where things moved since the last frame.
    moved: Vec<Bounds>,
    /// Regions still settling, with their ages in frames.
    recent: Vec<(Bounds, u32)>,
    capture_requested: bool,
    readback: Option<Readback>,
    capture: Option<Capture>,
    timer: Option<PassTimer>,
    /// Frames of the path-traced reference so far, and the view they are of.
    reference_frames: u32,
    reference_view: Option<Mat4>,
    previous_lights: Vec<u8>,
    previous_power: Vec3,
    /// Frames since a light last changed, while that still matters.
    light_age: Option<u32>,
    last_frame: Option<u64>,
    last_viewport: [u32; 4],
    last_reflections: PxtgiReflections,
    frame: u32,
}

impl Pxtgi {
    pub fn new(settings: PxtgiSettings) -> Self {
        let mut scene = RayScene::default();
        scene.entity_boxes = true;
        Self {
            settings,
            scene,
            albedos: HashMap::new(),
            statics: Vec::new(),
            retained: HashMap::new(),
            geometry: Vec::new(),
            geometry_free: 0,
            geometry_buffer: Buffer::default(),
            packed: Packed::default(),
            slots: Default::default(),
            cache_keys: Buffer::default(),
            cells: Buffer::default(),
            no_capture: Buffer::default(),
            pipelines: None,
            view: None,
            targets: None,
            command_pool: vk::CommandPool::null(),
            moved: Vec::new(),
            recent: Vec::new(),
            capture_requested: false,
            readback: None,
            capture: None,
            timer: None,
            reference_frames: 0,
            reference_view: None,
            previous_lights: Vec::new(),
            previous_power: Vec3::ZERO,
            light_age: None,
            last_frame: None,
            last_viewport: [0; 4],
            last_reflections: PxtgiReflections::Off,
            frame: 1,
        }
    }

    /// Follow the physics world. Call after stepping it, before drawing.
    /// Entities without a body trace as the cubes they are drawn as. All take
    /// their albedo from their entity's material.
    pub fn sync(&mut self, physics: &PhysicsWorld, world: &World) {
        let sync = self.scene.sync(physics, world);
        self.moved.extend_from_slice(self.scene.moved());
        let mut repaint = false;
        for item in self.scene.items() {
            let RaySource::Body(entity) = item.source else {
                continue;
            };
            if let Some(found) = world.get(entity) {
                let m = found.material;
                let albedo = m.base_color.map(|c| c * (1.0 - m.metallic));
                repaint |= self.albedos.insert(item.source, albedo) != Some(albedo);
            }
        }
        if sync.changed || repaint {
            self.pack();
        }
    }

    /// Replace the GPU's mesh geometry. The GPU must be idle.
    unsafe fn upload_geometry(&mut self, gpu: &mut Gpu) -> Result<()> {
        let mut buffer = mem::take(&mut self.geometry_buffer);
        let result = upload(
            gpu,
            &mut self.command_pool,
            &mut buffer,
            "pxtgi geometry",
            bytemuck::cast_slice(&self.geometry),
        );
        self.geometry_buffer = buffer;
        result?;
        if self.view.is_some() {
            // New buffer: both slots must point at it, and may have new
            // scene buffers too.
            for slot in 0..SLOTS {
                self.publish(gpu, slot)?;
                self.write_scene_set(gpu.device, slot, self.no_capture.buffer);
            }
        }
        Ok(())
    }

    /// The scene rays run against, for tests and CPU queries.
    pub fn ray_scene(&self) -> &RayScene {
        &self.scene
    }

    /// How long the GPU took over PXTGI's passes in a recent frame.
    pub fn gpu_time(&self) -> Option<std::time::Duration> {
        self.timer.as_ref().and_then(PassTimer::total)
    }

    /// Read back the next frame drawn; see [`Self::take_capture`].
    pub fn request_capture(&mut self) {
        self.capture_requested = true;
    }

    /// The requested frame, once the GPU has finished it, a frame or two later.
    pub fn take_capture(&mut self) -> Option<Capture> {
        self.capture.take()
    }

    /// Lay the scene out for the shaders, in the tree's leaf order.
    fn pack(&mut self) {
        let packed = &mut self.packed;
        packed.primitives.clear();
        packed.soft.clear();
        let tree = self.scene.tree();
        for &item in tree.order() {
            let item = &self.scene.items()[item as usize];
            let rows = Mat4::from(item.local_from_world).transpose();
            let [r0, r1, r2] = [0, 1, 2].map(|i| rows.col(i).to_array());
            let ints = |a: i32, b: i32| (f32::from_bits(a as u32), f32::from_bits(b as u32));
            let [r, g, b] = self.albedos.get(&item.source).copied().unwrap_or(GREY);
            let mut albedo = [r, g, b, f32::from_bits(NO_MATERIAL)];
            let shape = match (&item.shape, item.source) {
                (RayShape::Collider(verity_physics::Collider::Cuboid { half_extents: h }), _) => {
                    [KIND_BOX, h.x, h.y, h.z]
                }
                // Ground never reaches the scene: rays see the mesh drawn for it.
                (RayShape::Collider(verity_physics::Collider::Ground), _) => continue,
                (RayShape::Collider(verity_physics::Collider::Sphere { radius }), _) => {
                    [KIND_SPHERE, *radius, 0.0, 0.0]
                }
                (RayShape::Mesh(_), RaySource::Static(number)) => {
                    let mesh = self.statics[number as usize];
                    albedo[3] = f32::from_bits(mesh.material);
                    let (nodes, triangles) = ints(mesh.nodes_at, mesh.triangles_at);
                    [KIND_MESH, nodes, triangles, 0.0]
                }
                (RayShape::Mesh(mesh), _) => {
                    let at = packed.soft.len() as i32;
                    append_mesh(&mut packed.soft, mesh);
                    let (nodes, triangles) = ints(at, at + mesh.nodes().len() as i32 * 2);
                    [KIND_SOFT, nodes, triangles, 0.0]
                }
            };
            packed.primitives.push([r0, r1, r2, shape, albedo]);
        }
        packed.nodes.clear();
        packed.nodes.extend_from_slice(tree.nodes());
        packed.revision += 1;
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
        // Diffuse traced at a quarter of the resolution, reflections at half, as in Godot.
        let scaled = |by: u32| vk::Extent2D {
            width: (extent.width / by).max(1),
            height: (extent.height / by).max(1),
        };
        let (trace, reflect) = (scaled(4), scaled(2));
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
        let pair = |image: &mut dyn FnMut(&str, vk::Format, vk::Extent2D) -> Result<Image>,
                    name: &str,
                    format,
                    size|
         -> Result<[Image; 2]> {
            Ok([image(name, format, size)?, image(name, format, size)?])
        };
        let trace_image = image("pxtgi trace", rgba, trace)?;
        let accum = pair(&mut image, "pxtgi accum", rgba, trace)?;
        let accum_depth = pair(&mut image, "pxtgi accum depth", r, trace)?;
        let moments = pair(&mut image, "pxtgi moments", rgba, trace)?;
        let variance = image("pxtgi variance", r, trace)?;
        let filtered = pair(&mut image, "pxtgi filtered", rgba, trace)?;
        let filtered_variance = pair(&mut image, "pxtgi filtered variance", r, trace)?;
        let reflect_trace = image("pxtgi reflect trace", rgba, reflect)?;
        let reflect_accum = pair(&mut image, "pxtgi reflect accum", rgba, reflect)?;
        let reflect_depth = pair(&mut image, "pxtgi reflect depth", r, reflect)?;
        let reflect_filtered = pair(&mut image, "pxtgi reflect filtered", rgba, reflect)?;
        let prev_color = image("pxtgi previous color", rgba, extent)?;

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
            host("pxtgi frame", frame_bytes, uniform)?,
            host("pxtgi frame", frame_bytes, uniform)?,
        ];
        let light_bytes = verity_render::MAX_LIGHTS * mem::size_of::<GpuLight>();
        let lights = [
            host("pxtgi lights", light_bytes, storage)?,
            host("pxtgi lights", light_bytes, storage)?,
        ];

        let count =
            |list: &[(u32, D)], kind: D| list.iter().filter(|(_, k)| *k == kind).count() as u32;
        let sizes = [
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
        });
        let sizes: Vec<_> = sizes
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
            reflect_extent: [reflect.width, reflect.height],
            trace: trace_image,
            accum,
            accum_depth,
            moments,
            variance,
            filtered,
            filtered_variance,
            reflect_trace,
            reflect_accum,
            reflect_depth,
            reflect_filtered,
            prev_color,
            frame,
            lights,
            pool,
            scene_sets,
            io_sets,
            fresh: true,
        })
    }

    /// Point a slot's scene set at its buffers. Only that slot's frame may
    /// be in flight no longer.
    unsafe fn write_scene_set(&self, device: &ash::Device, slot: usize, capture: vk::Buffer) {
        let (view, targets) = (self.view.as_ref().unwrap(), self.targets.unwrap());
        let scene = &self.slots[slot];
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
                (4, Res::Buffer(scene.primitives.buffer)),
                (5, Res::Buffer(scene.nodes.buffer)),
                (6, Res::Buffer(self.cache_keys.buffer)),
                (7, Res::Buffer(self.cells.buffer)),
                (8, Res::Buffer(self.geometry_buffer.buffer)),
                (9, Res::Buffer(scene.soft.buffer)),
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
        let view = self.view.as_ref().unwrap();
        let general = vk::ImageLayout::GENERAL;
        let s = view;
        for slot in 0..SLOTS {
            let (cur, prev) = (slot, 1 - slot);
            // Unused slots hold images the pass neither reads nor writes, of
            // the format the binding declares.
            let io = |ins: [&Image; 4], outs: [&Image; 4]| {
                let (a, b) = (&s.filtered[0], &s.accum_depth[cur]);
                let (c, d) = (&s.reflect_filtered[1], &s.reflect_depth[cur]);
                (ins.into_iter().chain(outs).chain([a, b, c, d]))
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
            let spare = [&s.variance, &s.filtered[1], &s.filtered_variance[1]];
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
                io(
                    [
                        &s.accum[prev],
                        &s.accum_depth[prev],
                        &s.moments[prev],
                        &s.trace,
                    ],
                    [&s.reflect_trace, spare[0], spare[1], spare[2]],
                ),
                io(
                    [
                        &s.reflect_trace,
                        &s.reflect_accum[prev],
                        &s.reflect_depth[prev],
                        &s.trace,
                    ],
                    [
                        &s.reflect_accum[cur],
                        &s.reflect_depth[cur],
                        spare[1],
                        spare[2],
                    ],
                ),
                io(
                    [
                        &s.reflect_accum[cur],
                        &s.reflect_depth[cur],
                        &s.trace,
                        &s.trace,
                    ],
                    [&s.reflect_filtered[0], spare[0], spare[1], spare[2]],
                ),
                io(
                    [
                        &s.reflect_filtered[0],
                        &s.reflect_depth[cur],
                        &s.trace,
                        &s.trace,
                    ],
                    [&s.reflect_filtered[1], spare[0], spare[1], spare[2]],
                ),
                // The composite reads only bindings 8 and up.
                io(
                    [&s.trace, &s.trace, &s.trace, &s.trace],
                    [&s.reflect_trace, spare[0], spare[1], spare[2]],
                ),
            ];
            for (set, entries) in view.io_sets[slot].iter().zip(&groups) {
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
        for buffer in view.frame.into_iter().chain(view.lights) {
            gpu::destroy_buffer(device, allocator, buffer);
        }
        let groups = [
            view.accum,
            view.accum_depth,
            view.moments,
            view.filtered,
            view.filtered_variance,
            view.reflect_accum,
            view.reflect_depth,
            view.reflect_filtered,
        ];
        let singles = [
            view.trace,
            view.variance,
            view.reflect_trace,
            view.prev_color,
        ];
        for image in groups.into_iter().flatten().chain(singles) {
            gpu::destroy_image(device, allocator, image);
        }
    }

    /// Bring a slot's copy of the scene up to date. Its last frame is done.
    unsafe fn publish(&mut self, gpu: &mut Gpu, slot: usize) -> Result<()> {
        let (device, packed) = (gpu.device, &self.packed);
        let scene = &mut self.slots[slot];
        if scene.revision == Some(packed.revision) {
            return Ok(());
        }
        let contents: [(&mut Buffer, &str, &[u8]); 3] = [
            (
                &mut scene.primitives,
                "pxtgi primitives",
                bytemuck::cast_slice(&packed.primitives),
            ),
            (
                &mut scene.nodes,
                "pxtgi nodes",
                bytemuck::cast_slice(&packed.nodes),
            ),
            (
                &mut scene.soft,
                "pxtgi soft geometry",
                bytemuck::cast_slice(&packed.soft),
            ),
        ];
        let mut grew = false;
        for (buffer, name, bytes) in contents {
            let size = bytes.len().max(256) as u64;
            if buffer.buffer == vk::Buffer::null() || buffer.size < size {
                gpu::destroy_buffer(device, gpu.allocator, mem::take(buffer));
                // Room to grow, so a scene that fills up settles quickly.
                *buffer = gpu::create_buffer(
                    device,
                    gpu.allocator,
                    name,
                    size * 3 / 2,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    MemoryLocation::CpuToGpu,
                )?;
                grew = true;
            }
            if !bytes.is_empty() {
                buffer.write(bytes);
            }
        }
        scene.revision = Some(packed.revision);
        if grew {
            self.write_scene_set(device, slot, self.no_capture.buffer);
        }
        Ok(())
    }

    /// Finish a capture whose frame the GPU is done with, then start any
    /// requested one, on this slot since its last frame is done.
    unsafe fn tend_capture(
        &mut self,
        gpu: &mut Gpu,
        frame: &FrameContext,
        slot: usize,
    ) -> Result<()> {
        // Its own slot coming round again means that frame is done.
        if self
            .readback
            .as_ref()
            .is_some_and(|r| frame.frame_number > r.frame && r.frame & 1 == slot as u64)
        {
            let readback = self.readback.take().unwrap();
            self.write_scene_set(gpu.device, slot, self.no_capture.buffer);
            let pixels = readback.buffer.read();
            self.capture = Some(Capture {
                width: readback.width,
                height: readback.height,
                pixels: bytemuck::cast_slice(pixels).to_vec(),
            });
            gpu::destroy_buffer(gpu.device, gpu.allocator, readback.buffer);
        }
        if self.capture_requested && self.readback.is_none() {
            self.capture_requested = false;
            let [_, _, width, height] = frame.viewport;
            let buffer = gpu::create_buffer(
                gpu.device,
                gpu.allocator,
                "pxtgi capture",
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

    /// The frame's lights for the shader, in light as the GI holds it:
    /// irradiance / pi. Also notes when they changed.
    fn frame_lights(&mut self, frame: &FrameContext) -> Vec<GpuLight> {
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
            self.light_age = self
                .light_age
                .map(|age| age + 1)
                .filter(|age| CHANGED_HISTORY + age < STEADY_HISTORY);
        }
        lights
    }

    /// How much the light sources' power changed since last frame, per channel.
    // shortcut: a point light counts its whole flux and the sun what crosses
    // the scene's bounds, though walls may hide either from what the camera
    // sees. Exact while one source or all of them change; otherwise history
    // lands near and the short history after a change finishes the job.
    fn power_change(&mut self, frame: &FrameContext) -> Vec3 {
        let points = (frame.lights.iter())
            .filter(|l| l.radius > 0.0)
            .map(|l| Vec3::from_array(l.color) * (l.intensity * 4.0 * PI))
            .sum::<Vec3>();
        let bounds = self
            .scene
            .tree()
            .nodes()
            .first()
            .map_or(Vec3::splat(10.0), |[lo, hi]| {
                (Vec3::new(hi[0], hi[1], hi[2]) - Vec3::new(lo[0], lo[1], lo[2])).max(Vec3::ONE)
            });
        let area = bounds.x * bounds.y;
        let power = points + frame.sun.map_or(Vec3::ZERO, |sun| sun.color * area);
        let ratio = power / self.previous_power.max(Vec3::splat(1e-6));
        // From darkness there is nothing to scale, and a jump up from a faint
        // history would only amplify its noise.
        let fade = Vec3::select(
            self.previous_power.cmpgt(Vec3::splat(1e-6)),
            ratio.min(Vec3::splat(MAX_FADE)),
            Vec3::ONE,
        );
        self.previous_power = power;
        fade
    }

    /// Regions for the shader: grown by how far the change reaches, and merged
    /// when there are more than it takes.
    fn moved_regions(&mut self) -> Vec<Bounds> {
        self.recent.retain_mut(|(_, age)| {
            *age += 1;
            *age < MOVED_FRAMES
        });
        self.recent.extend(self.moved.drain(..).map(|b| {
            let size = (b.max - b.min).length();
            (b.expanded((MOVED_REACH * size).max(MOVED_REACH_MIN)), 0)
        }));
        let mut regions: Vec<_> = self.recent.iter().map(|(region, _)| *region).collect();
        // shortcut: merges the overflow into one box, which refreshes more of
        // the view than needed in a big pile; cluster them if that shows.
        if regions.len() > MAX_MOVED {
            let rest = regions.split_off(MAX_MOVED - 1);
            regions.push(rest.into_iter().fold(Bounds::EMPTY, Bounds::union));
        }
        regions
    }

    unsafe fn record_frame(&mut self, gpu: &mut Gpu, frame: &FrameContext) -> Result<()> {
        let device = gpu.device;
        let slot = (frame.frame_number & 1) as usize;
        self.publish(gpu, slot)?;
        self.tend_capture(gpu, frame, slot)?;
        let lights = self.frame_lights(frame);
        let fade = self.power_change(frame);
        let moved = self.moved_regions();
        let settings = self.settings;
        let [x, y, width, height] = frame.viewport;
        let primitives = self.packed.primitives.len() as u32;

        // History needs the frame right before this one, in the same place.
        let unbroken = self.last_frame.is_some_and(|f| f + 1 == frame.frame_number)
            && self.last_viewport == frame.viewport;
        let reflection_history = unbroken && self.last_reflections == settings.reflections;
        self.last_frame = Some(frame.frame_number);
        self.last_viewport = frame.viewport;
        self.last_reflections = settings.reflections;
        let view = self.view.as_mut().unwrap();
        let pipelines = self.pipelines.as_ref().unwrap();

        // The shaders' clip space is Y-up; the renderer's is flipped for Vulkan.
        let flip = Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0));
        let clip = flip * frame.view_proj;
        let previous_clip = flip * frame.previous_view_proj;
        let trace = [(width / 4).max(1), (height / 4).max(1)];
        let reflect = [(width / 2).max(1), (height / 2).max(1)];
        let trace = [
            trace[0].min(view.trace_extent[0]),
            trace[1].min(view.trace_extent[1]),
        ];
        let reflect = [
            reflect[0].min(view.reflect_extent[0]),
            reflect[1].min(view.reflect_extent[1]),
        ];
        let [mut moved_min, mut moved_max] = [[[0.0; 4]; MAX_MOVED]; 2];
        for (i, region) in moved.iter().enumerate() {
            moved_min[i] = region.min.extend(0.0).to_array();
            moved_max[i] = region.max.extend(0.0).to_array();
        }
        let sun = frame.sun.filter(|s| s.color.max_element() > 0.0);
        let mode = match settings.reflections {
            PxtgiReflections::Off => 0.0,
            PxtgiReflections::Screen => 1.0,
            PxtgiReflections::Traced => 2.0,
        };
        let reference = settings.debug == PxtgiDebug::Reference;
        let cache_on = settings.multi_bounce && primitives > 0 && !reference;
        if self.reference_view != Some(clip) {
            self.reference_view = Some(clip);
            self.reference_frames = 0;
        }
        let reference_frames = self.reference_frames;
        self.reference_frames += 1;
        let data = FrameData {
            clip_from_world: clip.to_cols_array_2d(),
            world_from_clip: clip.inverse().to_cols_array_2d(),
            prev_clip_from_world: previous_clip.to_cols_array_2d(),
            camera: frame.eye.extend(frame.near).to_array(),
            sizes: [width, height, trace[0], trace[1]],
            origin: [x, y, reflect[0], reflect[1]],
            misc: [
                (self.frame % 65536) as f32,
                settings.max_distance.max(0.1),
                settings.intensity,
                settings.debug as u32 as f32,
            ],
            sun_direction: sun.map_or([0.0; 4], |s| {
                (-s.direction.normalize()).extend(1.0).to_array()
            }),
            sun_color: sun.map_or([0.0; 4], |s| (s.color / PI).extend(0.0).to_array()),
            sky: frame
                .sky
                .extend(settings.world_ray_distance.max(0.1))
                .to_array(),
            ground: frame.ground.extend(0.0).to_array(),
            counts: [
                primitives,
                self.packed.nodes.len() as u32,
                lights.len() as u32,
                moved.len() as u32,
            ],
            cache: [
                self.frame,
                CACHE_SLOTS - 1,
                u32::from(cache_on),
                settings.rays_per_texel.clamp(1, 4),
            ],
            extra: [
                0.0,
                self.light_age
                    .map_or(0.0, |age| (CHANGED_HISTORY + age) as f32),
                f32::from(u8::from(unbroken)),
                f32::from(u8::from(reflection_history)),
            ],
            reflection: [
                mode,
                settings.reflection_roughness_cutoff.clamp(0.0, 1.0),
                reference_frames as f32,
                0.0,
            ],
            fade: fade.extend(1.0).to_array(),
            moved_min,
            moved_max,
        };
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
                .src_stage_mask(S::COMPUTE_SHADER | S::TRANSFER)
                .src_access_mask(A::SHADER_WRITE | A::TRANSFER_WRITE)
                .dst_stage_mask(S::COMPUTE_SHADER | S::TRANSFER)
                .dst_access_mask(A::SHADER_READ | A::SHADER_WRITE | A::TRANSFER_WRITE)];
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
            device.cmd_fill_buffer(cmd, self.cache_keys.buffer, 0, vk::WHOLE_SIZE, 0);
            device.cmd_fill_buffer(cmd, self.cells.buffer, 0, vk::WHOLE_SIZE, 0);
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
        if cache_on {
            let cells = [CACHE_ROW / 8, CACHE_SLOTS / CACHE_ROW / 8];
            // Any set whose in1 is last frame's depth, which the cache's rays test.
            dispatch("cache_gather", sets[0], cells);
            dispatch("cache_shade", sets[0], cells);
        }
        let diffuse = ["trace", "temporal", "atrous1", "atrous2", "atrous4"];
        if reference {
            // The composite's light is the depth of `temporal`, then this mean.
            dispatch("temporal", sets[1], groups(trace));
            dispatch("reference", sets[2], groups(trace));
        } else {
            for (name, &set) in diffuse.into_iter().zip(&sets) {
                dispatch(name, set, groups(trace));
            }
        }
        if settings.reflections != PxtgiReflections::Off && !reference {
            let passes = [
                "reflect_trace",
                "reflect_temporal",
                "reflect_filter1",
                "reflect_filter2",
            ];
            for (name, &set) in passes.into_iter().zip(&sets[5..]) {
                dispatch(name, set, groups(reflect));
            }
        }
        dispatch("composite", sets[9], groups([width, height]));
        if let Some(readback) = &self.readback
            && readback.frame == frame.frame_number
        {
            dispatch("capture", sets[9], groups([width, height]));
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

/// A mesh as the shaders walk it: its nodes, then its triangles.
/// A terrain vertex's original ground height, in sixteenths of a metre within
/// ±2 km, and its wetness, as bits for a triangle corner's spare w.
pub(crate) fn pack_ground(height: f32, wet: f32) -> f32 {
    let height = ((height + 2048.0) * 16.0).clamp(0.0, 65535.0) as u32;
    let wet = (wet.clamp(0.0, 1.0) * 65535.0) as u32;
    f32::from_bits(height | wet << 16)
}

/// The words of a retained mesh, its triangle corners carrying what
/// `terrain_albedo` needs, which rays cannot otherwise reach.
fn retained_words(retained: &RetainedMesh, mesh: &TriMesh) -> Vec<[f32; 4]> {
    let key = |p: &[f32]| [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()];
    let ground: HashMap<[u32; 3], f32> = (retained.vertices.iter())
        .map(|v| (key(&v.position), pack_ground(v.position[3], v.normal[3])))
        .collect();
    let mut words = Vec::new();
    append_mesh(&mut words, mesh);
    for corner in &mut words[mesh.nodes().len() * 2..] {
        corner[3] = ground[&key(&corner[..])];
    }
    words
}

fn append_mesh(out: &mut Vec<[f32; 4]>, mesh: &TriMesh) {
    out.extend(mesh.nodes().iter().flatten());
    out.extend(
        mesh.triangles()
            .iter()
            .flatten()
            .map(|p| p.extend(0.0).to_array()),
    );
}

impl GiMethod for Pxtgi {
    fn sync(&mut self, physics: &PhysicsWorld, world: &World) {
        Pxtgi::sync(self, physics, world);
    }

    fn follows_physics(&self) -> bool {
        true
    }

    fn status(&self) -> GiStatus {
        GiStatus::Ready {
            compile_time: std::time::Duration::ZERO,
        }
    }

    fn gpu_time(&self) -> Option<std::time::Duration> {
        Pxtgi::gpu_time(self)
    }

    fn pass_times(&self) -> &[(&'static str, f32)] {
        self.timer.as_ref().map_or(&[], |t| &t.times)
    }

    fn request_capture(&mut self) {
        Pxtgi::request_capture(self);
    }

    fn take_capture(&mut self) -> Option<Capture> {
        Pxtgi::take_capture(self)
    }
}

impl RenderExtension for Pxtgi {
    fn active(&self) -> bool {
        self.view.is_some()
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
        // Blended surfaces are glass and the like: rays pass them. Masked
        // ones count as solid.
        // shortcut: foliage blocks more light than its cut-out shows; sampling
        // alpha at hits needs texture coordinates in the mesh.
        let solid = |primitive: u32| {
            let material = scene.primitives[primitive as usize].material as usize;
            scene.materials[material].alpha_mode != AlphaMode::Blend
        };
        let mut wanted: Vec<u32> = (scene.draws.iter())
            .map(|d| d.primitive)
            .filter(|&p| solid(p))
            .collect();
        wanted.sort_unstable();
        wanted.dedup();
        let built: Vec<(u32, Arc<TriMesh>)> = wanted
            .into_par_iter()
            .map(|primitive| {
                let p = &scene.primitives[primitive as usize];
                let (first, count) = (p.first_index as usize, p.index_count as usize);
                let indices = &scene.indices[first..first + count];
                let triangles: Vec<[u32; 3]> = indices
                    .chunks_exact(3)
                    .map(|t| [t[0], t[1], t[2]])
                    .collect();
                // Only what this primitive uses: vertices are shared per scene.
                let mut local = HashMap::new();
                let mut positions = Vec::new();
                let triangles: Vec<[u32; 3]> = triangles
                    .iter()
                    .map(|t| {
                        t.map(|i| match local.entry(i) {
                            Entry::Occupied(found) => *found.get(),
                            Entry::Vacant(free) => {
                                let [x, y, z, _] =
                                    scene.vertices[(i + p.base_vertex) as usize].position;
                                positions.push(Vec3::new(x, y, z));
                                *free.insert(positions.len() as u32 - 1)
                            }
                        })
                    })
                    .collect();
                (primitive, Arc::new(TriMesh::new(&positions, &triangles)))
            })
            .collect();
        // Where each mesh went in the geometry buffer, by primitive.
        let mut placed = HashMap::new();
        for (primitive, mesh) in &built {
            let nodes_at = self.geometry.len() as i32;
            let triangles_at = nodes_at + mesh.nodes().len() as i32 * 2;
            append_mesh(&mut self.geometry, mesh);
            placed.insert(*primitive, (mesh.clone(), nodes_at, triangles_at));
        }
        // Retained meshes must not grow into what was just appended.
        self.geometry_free = self.geometry.len();
        let mut items = Vec::new();
        for draw in &scene.draws {
            let Some((mesh, nodes_at, triangles_at)) = placed.get(&draw.primitive) else {
                continue;
            };
            let number = self.statics.len() as u32;
            let material = scene.primitives[draw.primitive as usize].material + material_base;
            self.statics.push(StaticMesh {
                nodes_at: *nodes_at,
                triangles_at: *triangles_at,
                material,
            });
            items.push((mesh.clone(), Affine3A::from_mat4(draw.transform), number));
        }
        self.scene.add_statics(items);
        self.pack();
        unsafe { self.upload_geometry(gpu) }
    }

    fn retained_changed(
        &mut self,
        gpu: &mut Gpu,
        changes: &[RetainedChange],
        material: u32,
    ) -> Result<()> {
        let built: Vec<_> = changes
            .par_iter()
            .map(|change| {
                let mesh = change.mesh.map(|retained| {
                    let positions: Vec<Vec3> = (retained.vertices.iter())
                        .map(|v| Vec3::from_slice(&v.position[..3]))
                        .collect();
                    let mesh = TriMesh::new(&positions, retained.indices.as_chunks::<3>().0);
                    let words = retained_words(retained, &mesh);
                    (Arc::new(mesh), words)
                });
                (change.key, mesh)
            })
            .collect();
        let mut items = Vec::with_capacity(built.len());
        let mut written = Vec::with_capacity(built.len());
        for (key, mesh) in built {
            let Some((mesh, words)) = mesh else {
                if let Some(slot) = self.retained.get(&key) {
                    items.push((slot.number, None));
                }
                continue;
            };
            let slot = self.retained.entry(key).or_insert_with(|| {
                self.statics.push(StaticMesh::default());
                RetainedSlot {
                    number: self.statics.len() as u32 - 1,
                    start: 0,
                    capacity: 0,
                }
            });
            if words.len() > slot.capacity {
                // shortcut: the old slot stays a hole. Room doubles, so holes stay
                // under the mesh's own size; reuse them if that ever matters.
                slot.start = self.geometry_free;
                slot.capacity = 2 * words.len();
                self.geometry_free += slot.capacity;
                if self.geometry_free > self.geometry.len() {
                    // Grow well past the need, as growing replaces the whole buffer.
                    (self.geometry).resize(self.geometry_free * 5 / 4 + (1 << 16), [0.0; 4]);
                }
            }
            self.geometry[slot.start..slot.start + words.len()].copy_from_slice(&words);
            written.push(slot.start..slot.start + words.len());
            let nodes_at = slot.start as i32;
            self.statics[slot.number as usize] = StaticMesh {
                nodes_at,
                triangles_at: nodes_at + mesh.nodes().len() as i32 * 2,
                material,
            };
            items.push((slot.number, Some((mesh, Affine3A::IDENTITY))));
        }
        self.scene.update_statics(items);
        // Cache cells near the edit relight; the rest of the view keeps its light.
        self.moved.extend_from_slice(self.scene.moved());
        self.pack();
        let bytes: &[u8] = bytemuck::cast_slice(&self.geometry);
        if bytes.len() as u64 > self.geometry_buffer.size {
            return unsafe { self.upload_geometry(gpu) };
        }
        // Only what changed: the rest of the geometry is already there.
        let word = mem::size_of::<[f32; 4]>();
        let ranges = (written.iter())
            .map(|range| range.start * word..range.end * word)
            .collect();
        unsafe {
            upload_ranges(
                gpu,
                &mut self.command_pool,
                &[Patch {
                    target: &self.geometry_buffer,
                    source: bytes,
                    ranges,
                }],
            )?;
            if self.view.is_some() {
                for slot in 0..SLOTS {
                    self.publish(gpu, slot)?;
                    self.write_scene_set(gpu.device, slot, self.no_capture.buffer);
                }
            }
        }
        Ok(())
    }

    fn targets_changed(&mut self, gpu: &mut Gpu, targets: &Targets) -> Result<()> {
        unsafe {
            if self.pipelines.is_none() {
                self.pipelines = Some(Self::create_pipelines(gpu)?);
                self.timer = Some(PassTimer::new(gpu)?);
            }
            let storage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
            let device = gpu.device;
            if self.cells.buffer == vk::Buffer::null() {
                let mut cache = |name: &str, size| {
                    gpu::create_buffer(
                        device,
                        gpu.allocator,
                        name,
                        size,
                        storage,
                        MemoryLocation::GpuOnly,
                    )
                };
                self.cache_keys = cache("pxtgi cache keys", u64::from(CACHE_SLOTS) * 4)?;
                self.cells = cache("pxtgi cache cells", u64::from(CACHE_SLOTS) * CELL_BYTES)?;
                self.no_capture = cache("pxtgi no capture", 256)?;
                let empty = [[0.0f32; 4]; 16];
                let mut buffer = Buffer::default();
                upload(
                    gpu,
                    &mut self.command_pool,
                    &mut buffer,
                    "pxtgi geometry",
                    bytemuck::cast_slice(&empty),
                )?;
                if self.geometry_buffer.buffer == vk::Buffer::null() {
                    self.geometry_buffer = buffer;
                } else {
                    gpu::destroy_buffer(device, gpu.allocator, buffer);
                }
            }
            if let Some(view) = self.view.take() {
                Self::destroy_view(gpu.device, gpu.allocator, view);
            }
            self.view = Some(self.create_view(gpu, targets)?);
            self.targets = Some(*targets);
            self.write_io_sets(gpu.device);
            for slot in 0..SLOTS {
                // Sets bind real buffers from the first frame on.
                self.slots[slot].revision = None;
                self.publish(gpu, slot)?;
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
            if let Some(pipelines) = self.pipelines.take() {
                for pipeline in pipelines.passes.into_values() {
                    device.destroy_pipeline(pipeline, None);
                }
                device.destroy_pipeline_layout(pipelines.layout, None);
                device.destroy_descriptor_set_layout(pipelines.scene_layout, None);
                device.destroy_descriptor_set_layout(pipelines.io_layout, None);
            }
            let slots = mem::take(&mut self.slots);
            let readback = self.readback.take().map(|r| r.buffer);
            let buffers = slots
                .into_iter()
                .flat_map(|s| [s.primitives, s.nodes, s.soft])
                .chain(readback)
                .chain([
                    mem::take(&mut self.cache_keys),
                    mem::take(&mut self.cells),
                    mem::take(&mut self.no_capture),
                    mem::take(&mut self.geometry_buffer),
                ]);
            for buffer in buffers {
                gpu::destroy_buffer(device, gpu.allocator, buffer);
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
mod ground_tests {
    use super::*;
    use verity_render::Vertex;

    #[test]
    fn triangle_corners_carry_their_ground() {
        let corner = |x: f32, y: f32, height: f32, wet: f32| Vertex {
            position: [x, y, 1.0, height],
            normal: [0.0, 0.0, wet, wet],
            tangent: [0.0; 4],
            color: [1.0; 4],
        };
        let retained = RetainedMesh {
            vertices: vec![
                corner(0.0, 0.0, -40.5, 0.0),
                corner(1.0, 0.0, 12.25, 1.0),
                corner(0.0, 1.0, 130.0, 0.5),
            ],
            indices: vec![0, 1, 2],
            bounds: (Vec3::ZERO, Vec3::ONE),
        };
        let positions = [Vec3::Z, Vec3::new(1.0, 0.0, 1.0), Vec3::new(0.0, 1.0, 1.0)];
        let mesh = TriMesh::new(&positions, &[[0, 1, 2]]);
        let words = retained_words(&retained, &mesh);
        // As the shader unpacks them.
        for word in &words[mesh.nodes().len() * 2..] {
            let bits = word[3].to_bits();
            let height = (bits & 0xffff) as f32 / 16.0 - 2048.0;
            let wet = (bits >> 16) as f32 / 65535.0;
            let want = (retained.vertices.iter())
                .find(|v| v.position[..3] == word[..3])
                .expect("a mesh corner");
            assert!((height - want.position[3]).abs() < 0.07, "{height}");
            assert!((wet - want.normal[3]).abs() < 1e-4, "{wet}");
        }
        assert_eq!(words.len(), mesh.nodes().len() * 2 + 3);
    }
}
