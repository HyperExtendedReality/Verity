//! GPU-driven Matrix virtual geometry: three tiers behind one selection pass.
//!
//! Near geometry is the cluster DAG, disconnected aggregates turn into voxel
//! bricks as they shrink, and copies placed with [`Renderer::place_copies`]
//! become octahedral imposters at the far end. `shaders/matrix_select.slang`
//! picks per instance and per cluster each frame, so the CPU records a fixed
//! handful of indirect draws whatever the scene holds.
//!
//! All geometry is resident. shortcut: lists and queue have fixed capacities and
//! overflow is dropped, visible as holes; grow them, or stream groups, if
//! [`MatrixStats`] shows a scene reaching them.

use crate::{
    asset, failure, gpu,
    gpu::{Buffer, Image},
    scene::{self, Instance, NO_TEXTURE},
    Renderer, Result, FRAMES, VISIBILITY_FORMAT,
};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use gpu_allocator::{vulkan::Allocator, MemoryLocation};
use std::{mem, ops::Range};
use verity_matrix::cluster_lod::{Brick, ClusterLod, Voxel, NO_GROUP};

/// Mirrors `lib/matrix_lists.slang`.
const LIST_ONE_SIDED: usize = 0;
const LIST_TWO_SIDED: usize = 1;
const LIST_VOXELS: usize = 2;
const LIST_IMPOSTER: usize = 3;
const LIST_LAYERS: usize = 4;
const SHADOW_LAYERS: usize = scene::CASCADES + scene::MAX_POINT_SHADOWS * scene::POINT_SHADOW_FACES;
const LISTS: usize = LIST_LAYERS + SHADOW_LAYERS * 2;
const QUEUE: u64 = 1 << 20;
const CLUSTER_LIST: u32 = 1 << 19;
const BRICK_LIST: u32 = 1 << 21;
const CASCADE_CLUSTERS: u32 = 1 << 17;
const CASCADE_BRICKS: u32 = 1 << 19;
const FACE_CLUSTERS: u32 = 1 << 15;
const FACE_BRICKS: u32 = 1 << 16;
const PASS_WORDS: usize = LISTS * 4;
/// Enough for 8^16 groups; the tree depth is checked at load.
const MAX_PASSES: usize = 16;
const COUNTER_WORDS: usize = PASS_WORDS + (MAX_PASSES + 1) * 4;

/// Vertices per entry of a padded cluster and of a brick's faces.
const CLUSTER_VERTICES: u32 = 128 * 3;
const BRICK_VERTICES: u32 = 36;
const CARD_VERTICES: u32 = 6;

fn list_vertices(list: usize) -> u32 {
    match list {
        LIST_ONE_SIDED | LIST_TWO_SIDED => CLUSTER_VERTICES,
        LIST_VOXELS => BRICK_VERTICES,
        LIST_IMPOSTER => CARD_VERTICES,
        _ if list.is_multiple_of(2) => CLUSTER_VERTICES,
        _ => BRICK_VERTICES,
    }
}

/// Mirrors `lib/matrix.slang`.
const IMPOSTER_GRID: u32 = 12;
const IMPOSTER_FRAME: u32 = 96;
const IMPOSTER_SIZE: u32 = IMPOSTER_GRID * IMPOSTER_FRAME;
/// Mips beyond this mix neighboring frames.
const IMPOSTER_LEVELS: u32 = 4;
const ALBEDO_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;
const SURFACE_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

fn list_capacity(list: usize) -> u32 {
    let bricks = list % 2 == 1;
    match list {
        LIST_VOXELS => BRICK_LIST,
        _ if list < LIST_LAYERS => CLUSTER_LIST,
        _ if (list - LIST_LAYERS) / 2 < scene::CASCADES => {
            if bricks {
                CASCADE_BRICKS
            } else {
                CASCADE_CLUSTERS
            }
        }
        _ if bricks => FACE_BRICKS,
        _ => FACE_CLUSTERS,
    }
}

fn list_base(list: usize) -> u32 {
    (0..list).map(list_capacity).sum()
}

/// What selection drew in the last finished frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MatrixStats {
    pub clusters: u32,
    pub bricks: u32,
    pub imposters: u32,
    pub shadow_clusters: u32,
    pub shadow_bricks: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuInstance {
    base: Instance,
    inverse: [[f32; 4]; 4],
    sphere: [f32; 4],
    info: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuAsset {
    sphere: [f32; 4],
    first_root: u32,
    root_count: u32,
    imposter_albedo: u32,
    imposter_surface: u32,
    first_cluster: u32,
    cluster_count: u32,
    padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuNode {
    sphere: [f32; 4],
    error: f32,
    group: u32,
    first_child: u32,
    child_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuGroup {
    sphere: [f32; 4],
    error: f32,
    first_cluster: u32,
    cluster_count: u32,
    padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuCluster {
    sphere: [f32; 4],
    refined: u32,
    first_index: u32,
    triangle_count: u32,
    first_brick: u32,
    brick_count: u32,
    depth: u32,
    padding: [u32; 2],
}

/// Which pass a Matrix draw feeds; see the top of `shaders/matrix.slang`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Target {
    Visibility,
    Depth,
}

struct Bake {
    instance: u32,
    cluster_count: u32,
    /// The atlas textures, and their first-mip views.
    images: [usize; 2],
    views: [vk::ImageView; 2],
}

/// Graphics and compute state, the CPU copy of everything uploaded, and buffers
/// indexed by descriptor binding minus [`FIRST_BINDING`].
#[derive(Default)]
pub(crate) struct Matrix {
    select_instances: vk::Pipeline,
    select_nodes: vk::Pipeline,
    clusters: [vk::Pipeline; 2],
    clusters_depth: [vk::Pipeline; 2],
    clusters_shadow: vk::Pipeline,
    voxels: vk::Pipeline,
    voxels_depth: vk::Pipeline,
    voxels_shadow: vk::Pipeline,
    imposters: vk::Pipeline,
    imposters_depth: vk::Pipeline,
    bake: vk::Pipeline,
    bake_depth: Option<Image>,
    /// Single-mip views of baked atlases, since attachments allow no more.
    bake_views: Vec<vk::ImageView>,
    buffers: [Buffer; BINDINGS],
    readback: Vec<Buffer>,
    instances: Vec<GpuInstance>,
    assets: Vec<GpuAsset>,
    nodes: Vec<GpuNode>,
    groups: Vec<GpuGroup>,
    cluster_data: Vec<GpuCluster>,
    indices: Vec<u32>,
    bricks: Vec<Brick>,
    voxel_data: Vec<Voxel>,
    /// Asset of each renderer primitive, if it was cooked.
    primitive_assets: Vec<Option<u32>>,
    passes: u32,
    pending_bakes: Vec<Bake>,
    stats: MatrixStats,
    /// Whether this frame's selection ran.
    active: bool,
}

pub(crate) const FIRST_BINDING: usize = 19;
const BINDINGS: usize = 11;
const INSTANCES: usize = 0;
const ASSETS: usize = 1;
const NODES: usize = 2;
const GROUPS: usize = 3;
const CLUSTERS: usize = 4;
const INDICES: usize = 5;
const BRICKS: usize = 6;
const VOXELS: usize = 7;
const ENTRIES: usize = 8;
const QUEUE_BUFFER: usize = 9;
const COUNTERS: usize = 10;

impl Matrix {
    /// Create pipelines and placeholder buffers, so descriptors are always valid.
    pub(crate) unsafe fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        layout: vk::PipelineLayout,
        depth_format: vk::Format,
    ) -> Result<Self> {
        let mut matrix = Self::default();
        matrix.create_pipelines(device, layout, depth_format)?;
        use vk::BufferUsageFlags as U;
        for (index, buffer) in matrix.buffers.iter_mut().enumerate() {
            let usage = U::STORAGE_BUFFER | U::TRANSFER_DST;
            *buffer = if index == COUNTERS {
                gpu::create_buffer(
                    device,
                    allocator,
                    "matrix counters",
                    (COUNTER_WORDS * 4) as u64,
                    usage | U::INDIRECT_BUFFER | U::TRANSFER_SRC,
                    MemoryLocation::GpuOnly,
                )?
            } else {
                gpu::create_buffer(
                    device,
                    allocator,
                    "matrix data",
                    16,
                    usage,
                    MemoryLocation::GpuOnly,
                )?
            };
        }
        for _ in 0..FRAMES {
            matrix.readback.push(gpu::create_buffer(
                device,
                allocator,
                "matrix readback",
                (COUNTER_WORDS * 4) as u64,
                U::TRANSFER_DST,
                MemoryLocation::GpuToCpu,
            )?);
        }
        matrix.primitive_assets.push(None);
        Ok(matrix)
    }

    unsafe fn create_pipelines(
        &mut self,
        device: &ash::Device,
        layout: vk::PipelineLayout,
        depth_format: vk::Format,
    ) -> Result<()> {
        let cache = vk::PipelineCache::null();
        self.select_instances = gpu::compute_pipeline(
            device,
            cache,
            layout,
            spirv!("matrix_select.instanceMain"),
            c"instanceMain",
        )?;
        self.select_nodes = gpu::compute_pipeline(
            device,
            cache,
            layout,
            spirv!("matrix_select.nodeMain"),
            c"nodeMain",
        )?;
        let raster = |vertex, fragment, color, compare, cull, depth_bias| gpu::Raster {
            vertex,
            fragment,
            color,
            depth: Some((depth_format, compare)),
            cull,
            depth_bias,
        };
        let greater = vk::CompareOp::GREATER;
        let visible = Some(VISIBILITY_FORMAT);
        use vk::CullModeFlags as Cull;
        for (index, cull) in [Cull::BACK, Cull::NONE].into_iter().enumerate() {
            self.clusters[index] = gpu::graphics_pipeline(
                device,
                layout,
                &raster(
                    spirv!("matrix_clusters.vertexMain"),
                    spirv!("matrix_clusters.fragmentMain"),
                    visible,
                    greater,
                    cull,
                    None,
                ),
            )?;
            self.clusters_depth[index] = gpu::graphics_pipeline(
                device,
                layout,
                &raster(
                    spirv!("matrix_clusters_depth.vertexMain"),
                    spirv!("matrix_clusters_depth.fragmentMain"),
                    None,
                    greater,
                    cull,
                    None,
                ),
            )?;
        }
        let less = vk::CompareOp::LESS;
        self.clusters_shadow = gpu::graphics_pipeline(
            device,
            layout,
            &raster(
                spirv!("matrix_clusters_shadow.vertexMain"),
                spirv!("matrix_clusters_shadow.fragmentMain"),
                None,
                less,
                Cull::NONE,
                Some((1.25, 1.75)),
            ),
        )?;
        // Voxel and imposter vertex shaders choose their own faces.
        self.voxels = gpu::graphics_pipeline(
            device,
            layout,
            &raster(
                spirv!("matrix_voxels.vertexMain"),
                spirv!("matrix_voxels.fragmentMain"),
                visible,
                greater,
                Cull::NONE,
                None,
            ),
        )?;
        self.voxels_depth = gpu::graphics_pipeline(
            device,
            layout,
            &raster(
                spirv!("matrix_voxels_depth.vertexMain"),
                spirv!("matrix_voxels_depth.fragmentMain"),
                None,
                greater,
                Cull::NONE,
                None,
            ),
        )?;
        self.voxels_shadow = gpu::graphics_pipeline(
            device,
            layout,
            &raster(
                spirv!("matrix_voxels_shadow.vertexMain"),
                spirv!("matrix_voxels_shadow.fragmentMain"),
                None,
                less,
                Cull::NONE,
                None,
            ),
        )?;
        self.imposters = gpu::graphics_pipeline(
            device,
            layout,
            &raster(
                spirv!("matrix_imposters.vertexMain"),
                spirv!("matrix_imposters.fragmentMain"),
                visible,
                greater,
                Cull::NONE,
                None,
            ),
        )?;
        self.imposters_depth = gpu::graphics_pipeline(
            device,
            layout,
            &raster(
                spirv!("matrix_imposters_depth.vertexMain"),
                spirv!("matrix_imposters_depth.fragmentMain"),
                None,
                greater,
                Cull::NONE,
                None,
            ),
        )?;
        self.bake = gpu::graphics_pipeline_targets(
            device,
            layout,
            &raster(
                spirv!("matrix_bake.vertexMain"),
                spirv!("matrix_bake.fragmentMain"),
                Some(ALBEDO_FORMAT),
                greater,
                Cull::NONE,
                None,
            ),
            &[SURFACE_FORMAT],
        )?;
        Ok(())
    }

    pub(crate) unsafe fn destroy(mut self, device: &ash::Device, allocator: &mut Allocator) {
        for pipeline in [
            self.select_instances,
            self.select_nodes,
            self.clusters[0],
            self.clusters[1],
            self.clusters_depth[0],
            self.clusters_depth[1],
            self.clusters_shadow,
            self.voxels,
            self.voxels_depth,
            self.voxels_shadow,
            self.imposters,
            self.imposters_depth,
            self.bake,
        ] {
            device.destroy_pipeline(pipeline, None);
        }
        self.destroy_bake_targets(device, allocator);
        for buffer in self.buffers.into_iter().chain(self.readback) {
            gpu::destroy_buffer(device, allocator, buffer);
        }
    }

    /// Release what baking rendered through, once nothing is pending.
    pub(crate) unsafe fn destroy_bake_targets(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
    ) {
        for view in self.bake_views.drain(..) {
            device.destroy_image_view(view, None);
        }
        if let Some(image) = self.bake_depth.take() {
            gpu::destroy_image(device, allocator, image);
        }
    }

    pub(crate) fn buffers(&self) -> impl Iterator<Item = (u32, &Buffer)> {
        self.buffers
            .iter()
            .enumerate()
            .map(|(index, buffer)| ((FIRST_BINDING + index) as u32, buffer))
    }

    /// Whether a primitive's draws go through selection rather than the CPU.
    pub(crate) fn owns(&self, primitive: usize) -> bool {
        self.primitive_assets[primitive].is_some()
    }

    pub(crate) fn clear(&mut self) {
        self.instances.clear();
        self.assets.clear();
        self.nodes.clear();
        self.groups.clear();
        self.cluster_data.clear();
        self.indices.clear();
        self.bricks.clear();
        self.voxel_data.clear();
        self.primitive_assets.truncate(1);
        self.pending_bakes.clear();
        self.passes = 0;
    }

    /// Append one cooked primitive; `None` draws it the ordinary way.
    pub(crate) fn add_primitive(&mut self, lod: Option<ClusterLod>) {
        let Some(lod) = lod else {
            self.primitive_assets.push(None);
            return;
        };
        let offset = |vec_len: usize| vec_len as u32;
        let first_node = offset(self.nodes.len());
        let first_group = offset(self.groups.len());
        let first_cluster = offset(self.cluster_data.len());
        let first_index = offset(self.indices.len());
        let first_brick = offset(self.bricks.len());
        let first_voxel = offset(self.voxel_data.len());
        let sphere = |b: verity_matrix::cluster_lod::Bounds| {
            [b.center[0], b.center[1], b.center[2], b.radius]
        };
        self.nodes.extend(lod.nodes.iter().map(|node| GpuNode {
            sphere: sphere(node.bounds),
            error: node.bounds.error,
            group: if node.group == NO_GROUP {
                NO_GROUP
            } else {
                node.group + first_group
            },
            first_child: node.first_child + first_node,
            child_count: node.child_count,
        }));
        self.groups.extend(lod.groups.iter().map(|group| GpuGroup {
            sphere: sphere(group.simplified),
            error: group.simplified.error,
            first_cluster: group.first_cluster + first_cluster,
            cluster_count: group.cluster_count,
            padding: 0,
        }));
        self.cluster_data
            .extend(lod.clusters.iter().map(|cluster| GpuCluster {
                sphere: sphere(cluster.bounds),
                refined: if cluster.refined == NO_GROUP {
                    NO_GROUP
                } else {
                    cluster.refined + first_group
                },
                first_index: cluster.first_index + first_index,
                triangle_count: cluster.triangle_count,
                first_brick: cluster.first_brick + first_brick,
                brick_count: cluster.brick_count,
                depth: cluster.depth,
                padding: [0; 2],
            }));
        self.indices.extend_from_slice(&lod.indices);
        self.bricks.extend(lod.bricks.iter().map(|brick| Brick {
            first_voxel: brick.first_voxel + first_voxel,
            ..*brick
        }));
        self.voxel_data.extend_from_slice(&lod.voxels);
        self.passes = self.passes.max(lod.tree_depth + 1);
        self.primitive_assets.push(Some(self.assets.len() as u32));
        self.assets.push(GpuAsset {
            sphere: [lod.center[0], lod.center[1], lod.center[2], lod.radius],
            first_root: first_node,
            root_count: lod.root_count,
            imposter_albedo: NO_TEXTURE,
            imposter_surface: NO_TEXTURE,
            first_cluster,
            cluster_count: lod.clusters.len() as u32,
            padding: [0; 2],
        });
    }

    /// Queue a placed draw for selection.
    pub(crate) fn add_instance(
        &mut self,
        transform: Mat4,
        primitive: &asset::Primitive,
        primitive_index: usize,
        two_sided: bool,
        draw_index: u32,
    ) {
        let Some(asset) = self.primitive_assets[primitive_index] else {
            return;
        };
        // A degenerate transform draws nothing either way.
        let inverse = transform.inverse();
        if !inverse.is_finite() {
            return;
        }
        let bounds = self.assets[asset as usize].sphere;
        let center = transform.transform_point3(Vec3::new(bounds[0], bounds[1], bounds[2]));
        let scale = [transform.x_axis, transform.y_axis, transform.z_axis]
            .map(|axis| axis.truncate().length())
            .into_iter()
            .fold(0.0, f32::max);
        let mut base = scene::instance(transform, 0, primitive.base_vertex, primitive.material);
        base.draw[3] = draw_index;
        self.instances.push(GpuInstance {
            base,
            inverse: inverse.to_cols_array_2d(),
            sphere: [center.x, center.y, center.z, bounds[3] * scale],
            info: [asset, u32::from(two_sided), scale.to_bits(), 0],
        });
    }

    pub(crate) fn has_voxels(&self, primitive: usize) -> bool {
        self.primitive_assets[primitive].is_some_and(|asset| {
            let asset = self.assets[asset as usize];
            let clusters = asset.first_cluster as usize;
            self.cluster_data[clusters..clusters + asset.cluster_count as usize]
                .iter()
                .any(|cluster| cluster.brick_count > 0)
        })
    }

    pub(crate) fn has_imposter(&self, primitive: usize) -> bool {
        self.primitive_assets[primitive]
            .is_some_and(|asset| self.assets[asset as usize].imposter_albedo != NO_TEXTURE)
    }

    /// Read last time's counters; the frame's fence has signaled.
    pub(crate) fn read_stats(&mut self, frame: usize) {
        let words: &[u32] = bytemuck::cast_slice(self.readback[frame].read());
        let count = |list: usize| words[list * 4 + 1];
        let layers = |first: usize| (first..LISTS).step_by(2).map(count).sum();
        self.stats = MatrixStats {
            clusters: count(LIST_ONE_SIDED) + count(LIST_TWO_SIDED),
            bricks: count(LIST_VOXELS),
            imposters: count(LIST_IMPOSTER),
            shadow_clusters: layers(LIST_LAYERS),
            shadow_bricks: layers(LIST_LAYERS + 1),
        };
    }

    pub(crate) fn set_active(&mut self, enabled: bool) {
        self.active = enabled && !self.instances.is_empty();
    }

    /// Select this frame's clusters, bricks and imposters from the camera.
    pub(crate) unsafe fn record_select(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        frame: usize,
    ) {
        if !self.active {
            return;
        }
        use vk::AccessFlags2 as A;
        use vk::PipelineStageFlags2 as S;
        let counters = self.buffers[COUNTERS].buffer;
        let barrier = |src_stage, src_access, dst_stage, dst_access| {
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&[vk::MemoryBarrier2::default()
                    .src_stage_mask(src_stage)
                    .src_access_mask(src_access)
                    .dst_stage_mask(dst_stage)
                    .dst_access_mask(dst_access)]),
            );
        };
        // Last frame's draws may still read the lists and commands.
        barrier(
            S::DRAW_INDIRECT | S::VERTEX_SHADER | S::FRAGMENT_SHADER | S::COMPUTE_SHADER,
            A::INDIRECT_COMMAND_READ | A::SHADER_STORAGE_READ,
            S::COPY | S::COMPUTE_SHADER,
            A::TRANSFER_WRITE | A::SHADER_STORAGE_WRITE,
        );
        let mut reset = [0u32; COUNTER_WORDS];
        for list in 0..LISTS {
            reset[list * 4] = list_vertices(list);
        }
        for pass in 0..=MAX_PASSES {
            let arguments = PASS_WORDS + pass * 4;
            reset[arguments + 1] = 1;
            reset[arguments + 2] = 1;
        }
        device.cmd_update_buffer(cmd, counters, 0, bytemuck::cast_slice(&reset));
        barrier(
            S::COPY,
            A::TRANSFER_WRITE,
            S::COMPUTE_SHADER,
            A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE,
        );
        let push = |pass: u32| {
            device.cmd_push_constants(
                cmd,
                layout,
                vk::ShaderStageFlags::COMPUTE,
                8,
                bytemuck::cast_slice(&[pass, self.instances.len() as u32]),
            );
        };
        let compute = vk::PipelineBindPoint::COMPUTE;
        device.cmd_bind_pipeline(cmd, compute, self.select_instances);
        push(0);
        device.cmd_dispatch(cmd, (self.instances.len() as u32).div_ceil(64), 1, 1);
        device.cmd_bind_pipeline(cmd, compute, self.select_nodes);
        for pass in 0..self.passes {
            barrier(
                S::COMPUTE_SHADER,
                A::SHADER_STORAGE_WRITE,
                S::COMPUTE_SHADER | S::DRAW_INDIRECT,
                A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE | A::INDIRECT_COMMAND_READ,
            );
            push(pass);
            let offset = (PASS_WORDS + pass as usize * 4) * 4;
            device.cmd_dispatch_indirect(cmd, counters, offset as u64);
        }
        barrier(
            S::COMPUTE_SHADER,
            A::SHADER_STORAGE_WRITE,
            S::DRAW_INDIRECT | S::VERTEX_SHADER | S::COPY,
            A::INDIRECT_COMMAND_READ | A::SHADER_STORAGE_READ | A::TRANSFER_READ,
        );
        device.cmd_copy_buffer(
            cmd,
            counters,
            self.readback[frame].buffer,
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: (LISTS * 16) as u64,
            }],
        );
    }

    unsafe fn draw_list(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        pipeline: vk::Pipeline,
        list: usize,
        cascade: u32,
    ) {
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            bytemuck::cast_slice(&[list_base(list), cascade]),
        );
        device.cmd_draw_indirect(
            cmd,
            self.buffers[COUNTERS].buffer,
            (list * 16) as u64,
            1,
            16,
        );
    }

    /// Draw every tier into the bound depth, and IDs if the target has them.
    pub(crate) unsafe fn record_draws(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        target: Target,
    ) {
        if !self.active {
            return;
        }
        let (clusters, voxels, imposters) = match target {
            Target::Visibility => (self.clusters, self.voxels, self.imposters),
            Target::Depth => (self.clusters_depth, self.voxels_depth, self.imposters_depth),
        };
        self.draw_list(device, cmd, layout, clusters[0], LIST_ONE_SIDED, 0);
        self.draw_list(device, cmd, layout, clusters[1], LIST_TWO_SIDED, 0);
        self.draw_list(device, cmd, layout, voxels, LIST_VOXELS, 0);
        self.draw_list(device, cmd, layout, imposters, LIST_IMPOSTER, 0);
    }

    /// Draw casters into one shadow layer. `layer` is a cascade, then point faces.
    pub(crate) unsafe fn record_shadows(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        layer: u32,
    ) {
        if !self.active {
            return;
        }
        let list = LIST_LAYERS + layer as usize * 2;
        self.draw_list(device, cmd, layout, self.clusters_shadow, list, layer);
        self.draw_list(device, cmd, layout, self.voxels_shadow, list + 1, layer);
    }
}

impl Renderer {
    /// What selection drew in the last finished frame.
    pub fn matrix_stats(&self) -> MatrixStats {
        self.matrix.stats
    }

    /// Number of model draws loaded so far; a range of them is a model.
    pub fn model_draw_count(&self) -> usize {
        self.model_draws.len()
    }

    /// Place another copy of the model draws in `draws` at each transform, on
    /// top of its placement in the scene. Repeating an object is what pays for
    /// an imposter, so aggregates placed this way get one.
    ///
    /// Copies are not added to a render extension's scene.
    pub fn place_copies(&mut self, draws: Range<usize>, transforms: &[Mat4]) -> Result<()> {
        let sources: Vec<_> = self
            .model_draws
            .get(draws)
            .ok_or_else(|| failure("no such model draws"))?
            .to_vec();
        for &transform in transforms {
            for source in &sources {
                let index = self.model_draws.len() as u32;
                let copy = asset::Draw {
                    transform: transform * source.transform,
                    primitive: source.primitive,
                };
                self.model_draws.push(copy);
                self.add_matrix_instance(copy, index);
            }
        }
        unsafe {
            self.device.device_wait_idle()?;
            for source in &sources {
                let primitive = source.primitive as usize;
                if self.matrix.has_voxels(primitive) && !self.matrix.has_imposter(primitive) {
                    self.add_imposter(primitive)?;
                }
            }
            self.upload_matrix()
        }
    }

    pub(crate) fn add_matrix_instance(&mut self, draw: asset::Draw, index: u32) {
        let primitive = self.primitives[draw.primitive as usize];
        let two_sided = self.materials[primitive.material as usize].double_sided;
        self.matrix.add_instance(
            draw.transform,
            &primitive,
            draw.primitive as usize,
            two_sided,
            index,
        );
    }

    /// Create an atlas for a primitive, to be baked at the start of a frame.
    unsafe fn add_imposter(&mut self, primitive: usize) -> Result<()> {
        if self.textures.len() + 2 > scene::MAX_TEXTURES as usize {
            return Ok(());
        }
        let Some(asset_index) = self.matrix.primitive_assets[primitive] else {
            return Ok(());
        };
        let Some(instance) = self
            .matrix
            .instances
            .iter()
            .position(|instance| instance.info[0] == asset_index)
        else {
            return Ok(());
        };
        use vk::ImageUsageFlags as U;
        let extent = vk::Extent2D {
            width: IMPOSTER_SIZE,
            height: IMPOSTER_SIZE,
        };
        let usage = U::COLOR_ATTACHMENT | U::SAMPLED | U::TRANSFER_SRC | U::TRANSFER_DST;
        let allocator = self.allocator.as_mut().unwrap();
        let albedo = gpu::create_image(
            &self.device,
            allocator,
            "imposter albedo",
            ALBEDO_FORMAT,
            extent,
            IMPOSTER_LEVELS,
            usage,
            vk::ImageAspectFlags::COLOR,
        )?;
        let surface = gpu::create_image(
            &self.device,
            allocator,
            "imposter surface",
            SURFACE_FORMAT,
            extent,
            IMPOSTER_LEVELS,
            usage,
            vk::ImageAspectFlags::COLOR,
        )?;
        if self.matrix.bake_depth.is_none() {
            self.matrix.bake_depth = Some(gpu::create_image(
                &self.device,
                allocator,
                "imposter depth",
                self.depth_format,
                extent,
                1,
                U::DEPTH_STENCIL_ATTACHMENT,
                vk::ImageAspectFlags::DEPTH,
            )?);
        }
        let first_mip = |image: &Image, format| {
            self.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image.image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(gpu::subresource(vk::ImageAspectFlags::COLOR)),
                None,
            )
        };
        let views = [
            first_mip(&albedo, ALBEDO_FORMAT)?,
            first_mip(&surface, SURFACE_FORMAT)?,
        ];
        self.matrix.bake_views.extend(views);
        let albedo_index = self.textures.len();
        self.textures.push(albedo);
        self.textures.push(surface);
        let asset = &mut self.matrix.assets[asset_index as usize];
        asset.imposter_albedo = albedo_index as u32;
        asset.imposter_surface = albedo_index as u32 + 1;
        let cluster_count = asset.cluster_count;
        self.matrix.pending_bakes.push(Bake {
            instance: instance as u32,
            cluster_count,
            images: [albedo_index, albedo_index + 1],
            views,
        });
        self.write_texture_descriptors();
        Ok(())
    }

    /// Replace every Matrix buffer's contents. The GPU must be idle.
    pub(crate) unsafe fn upload_matrix(&mut self) -> Result<()> {
        use vk::BufferUsageFlags as U;
        let usage = U::STORAGE_BUFFER | U::TRANSFER_DST;
        let matrix = &self.matrix;
        let uploads: [(usize, &[u8]); 8] = [
            (INSTANCES, bytemuck::cast_slice(&matrix.instances)),
            (ASSETS, bytemuck::cast_slice(&matrix.assets)),
            (NODES, bytemuck::cast_slice(&matrix.nodes)),
            (GROUPS, bytemuck::cast_slice(&matrix.groups)),
            (CLUSTERS, bytemuck::cast_slice(&matrix.cluster_data)),
            (INDICES, bytemuck::cast_slice(&matrix.indices)),
            (BRICKS, bytemuck::cast_slice(&matrix.bricks)),
            (VOXELS, bytemuck::cast_slice(&matrix.voxel_data)),
        ];
        if matrix.passes as usize > MAX_PASSES {
            return Err(failure("cluster tree too deep"));
        }
        let mut staged = Vec::new();
        let mut result = Ok(());
        for (index, bytes) in uploads {
            if bytes.is_empty() {
                continue;
            }
            let allocator = self.allocator.as_mut().unwrap();
            let staging = gpu::create_buffer(
                &self.device,
                allocator,
                "matrix staging",
                bytes.len() as u64,
                U::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
            );
            let replacement = gpu::create_buffer(
                &self.device,
                allocator,
                "matrix data",
                bytes.len() as u64,
                usage,
                MemoryLocation::GpuOnly,
            );
            match (staging, replacement) {
                (Ok(mut staging), Ok(replacement)) => {
                    staging.write(bytes);
                    staged.push((index, staging, replacement, bytes.len() as u64));
                }
                (staging, replacement) => {
                    for buffer in [staging, replacement] {
                        match buffer {
                            Ok(buffer) => gpu::destroy_buffer(&self.device, allocator, buffer),
                            Err(error) => result = Err(error),
                        }
                    }
                    break;
                }
            }
        }
        if result.is_ok() {
            result = self.submit_once(|cmd| {
                for (_, staging, replacement, size) in &staged {
                    self.device.cmd_copy_buffer(
                        cmd,
                        staging.buffer,
                        replacement.buffer,
                        &[vk::BufferCopy {
                            src_offset: 0,
                            dst_offset: 0,
                            size: *size,
                        }],
                    );
                }
            });
        }
        let device = &self.device;
        let allocator = self.allocator.as_mut().unwrap();
        for (index, staging, replacement, _) in staged {
            gpu::destroy_buffer(device, allocator, staging);
            if result.is_ok() {
                let old = mem::replace(&mut self.matrix.buffers[index], replacement);
                gpu::destroy_buffer(device, allocator, old);
            } else {
                gpu::destroy_buffer(device, allocator, replacement);
            }
        }
        result?;
        // Lists are only needed once something can be selected.
        let entries_bytes = u64::from(list_base(LISTS)) * 8;
        for (index, size) in [(ENTRIES, entries_bytes), (QUEUE_BUFFER, QUEUE * 2 * 8)] {
            if self.matrix.buffers[index].size < size && !self.matrix.instances.is_empty() {
                let buffer = gpu::create_buffer(
                    device,
                    allocator,
                    "matrix lists",
                    size,
                    usage,
                    MemoryLocation::GpuOnly,
                )?;
                let old = mem::replace(&mut self.matrix.buffers[index], buffer);
                gpu::destroy_buffer(device, allocator, old);
            }
        }
        for frame in 0..self.frames.len() {
            self.write_buffer_descriptors(frame);
        }
        Ok(())
    }

    /// Render pending imposters; `set` holds this frame's materials.
    pub(crate) unsafe fn record_bakes(&mut self, cmd: vk::CommandBuffer, set: vk::DescriptorSet) {
        use vk::AccessFlags2 as A;
        use vk::ImageLayout as L;
        use vk::PipelineStageFlags2 as S;
        if self.matrix.pending_bakes.is_empty() || !self.matrix.active {
            return;
        }
        let device = &self.device;
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline_layout,
            0,
            &[set],
            &[],
        );
        let color = vk::ImageAspectFlags::COLOR;
        let area = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: IMPOSTER_SIZE,
                height: IMPOSTER_SIZE,
            },
        };
        let depth = self.matrix.bake_depth.as_ref().unwrap();
        for bake in &self.matrix.pending_bakes {
            let images = bake.images.map(|index| &self.textures[index]);
            let all_levels = gpu::subresource(color).level_count(IMPOSTER_LEVELS);
            let mut barriers: Vec<_> = images
                .iter()
                .map(|image| {
                    gpu::image_barrier(
                        image.image,
                        color,
                        S::ALL_COMMANDS,
                        A::NONE,
                        S::COLOR_ATTACHMENT_OUTPUT,
                        A::COLOR_ATTACHMENT_WRITE,
                        L::UNDEFINED,
                        L::COLOR_ATTACHMENT_OPTIMAL,
                    )
                    .subresource_range(all_levels)
                })
                .collect();
            barriers.push(gpu::image_barrier(
                depth.image,
                vk::ImageAspectFlags::DEPTH,
                S::LATE_FRAGMENT_TESTS,
                A::DEPTH_STENCIL_ATTACHMENT_WRITE,
                S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
                A::DEPTH_STENCIL_ATTACHMENT_READ | A::DEPTH_STENCIL_ATTACHMENT_WRITE,
                L::UNDEFINED,
                L::DEPTH_ATTACHMENT_OPTIMAL,
            ));
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&barriers),
            );
            let colors = bake.views.map(|view| {
                vk::RenderingAttachmentInfo::default()
                    .image_view(view)
                    .image_layout(L::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue { float32: [0.0; 4] },
                    })
            });
            let depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(depth.view)
                .image_layout(L::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 0.0,
                        stencil: 0,
                    },
                });
            device.cmd_begin_rendering(
                cmd,
                &vk::RenderingInfo::default()
                    .render_area(area)
                    .layer_count(1)
                    .color_attachments(&colors)
                    .depth_attachment(&depth_attachment),
            );
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.matrix.bake);
            for frame in 0..IMPOSTER_GRID * IMPOSTER_GRID {
                let tile = vk::Rect2D {
                    offset: vk::Offset2D {
                        x: ((frame % IMPOSTER_GRID) * IMPOSTER_FRAME) as i32,
                        y: ((frame / IMPOSTER_GRID) * IMPOSTER_FRAME) as i32,
                    },
                    extent: vk::Extent2D {
                        width: IMPOSTER_FRAME,
                        height: IMPOSTER_FRAME,
                    },
                };
                crate::set_viewport(device, cmd, tile);
                device.cmd_push_constants(
                    cmd,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::cast_slice(&[bake.instance, frame]),
                );
                device.cmd_draw(cmd, CLUSTER_VERTICES, bake.cluster_count, 0, 0);
            }
            device.cmd_end_rendering(cmd);
            for image in images {
                generate_mips(device, cmd, image.image);
            }
        }
        self.matrix.pending_bakes.clear();
    }
}

/// Fill mips by halving, leaving every level ready to sample.
unsafe fn generate_mips(device: &ash::Device, cmd: vk::CommandBuffer, image: vk::Image) {
    use vk::AccessFlags2 as A;
    use vk::ImageLayout as L;
    use vk::PipelineStageFlags2 as S;
    let color = vk::ImageAspectFlags::COLOR;
    let level = |index: u32| gpu::subresource(color).base_mip_level(index).level_count(1);
    let layers = |index: u32| {
        vk::ImageSubresourceLayers::default()
            .aspect_mask(color)
            .mip_level(index)
            .layer_count(1)
    };
    let barrier = |index, src_stage, src_access, dst_stage, dst_access, old, new| {
        device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&[gpu::image_barrier(
                image, color, src_stage, src_access, dst_stage, dst_access, old, new,
            )
            .subresource_range(level(index))]),
        );
    };
    barrier(
        0,
        S::COLOR_ATTACHMENT_OUTPUT,
        A::COLOR_ATTACHMENT_WRITE,
        S::BLIT,
        A::TRANSFER_READ,
        L::COLOR_ATTACHMENT_OPTIMAL,
        L::TRANSFER_SRC_OPTIMAL,
    );
    for index in 1..IMPOSTER_LEVELS {
        barrier(
            index,
            S::COLOR_ATTACHMENT_OUTPUT,
            A::NONE,
            S::BLIT,
            A::TRANSFER_WRITE,
            L::UNDEFINED,
            L::TRANSFER_DST_OPTIMAL,
        );
        let size = |level: u32| (IMPOSTER_SIZE >> level) as i32;
        let corner = |level: u32| vk::Offset3D {
            x: size(level),
            y: size(level),
            z: 1,
        };
        device.cmd_blit_image(
            cmd,
            image,
            L::TRANSFER_SRC_OPTIMAL,
            image,
            L::TRANSFER_DST_OPTIMAL,
            &[vk::ImageBlit::default()
                .src_subresource(layers(index - 1))
                .src_offsets([vk::Offset3D::default(), corner(index - 1)])
                .dst_subresource(layers(index))
                .dst_offsets([vk::Offset3D::default(), corner(index)])],
            vk::Filter::LINEAR,
        );
        barrier(
            index,
            S::BLIT,
            A::TRANSFER_WRITE,
            S::BLIT,
            A::TRANSFER_READ,
            L::TRANSFER_DST_OPTIMAL,
            L::TRANSFER_SRC_OPTIMAL,
        );
    }
    device.cmd_pipeline_barrier2(
        cmd,
        &vk::DependencyInfo::default().image_memory_barriers(&[gpu::image_barrier(
            image,
            color,
            S::BLIT,
            A::TRANSFER_WRITE,
            S::FRAGMENT_SHADER | S::COMPUTE_SHADER,
            A::SHADER_SAMPLED_READ,
            L::TRANSFER_SRC_OPTIMAL,
            L::SHADER_READ_ONLY_OPTIMAL,
        )
        .subresource_range(gpu::subresource(color).level_count(IMPOSTER_LEVELS))]),
    );
}
