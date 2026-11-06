//! Detailed meshes deformed on the GPU by a soft body's lattice.
//!
//! A model is uploaded once and shared; each instance owns only its output
//! vertices. An instance's compute pass runs when its shape changes, so a car
//! that is only moving costs a transform, and its draw is culled like any other.

use crate::{
    FRAMES, Renderer, Result, failure, gpu,
    gpu::Buffer,
    scene::{GpuMaterial, Vertex},
};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3};
use gpu_allocator::{MemoryLocation, vulkan::Allocator};
use std::mem;

const WORKGROUP: u32 = 64;
const MISSING: u32 = u32::MAX;
/// Most dents kept on a car, newest last: each costs every vertex a test.
pub const MAX_DENTS: usize = 16;

/// A mesh tied to a lattice, in its carrier's frame at rest.
pub struct EmbeddedModel {
    /// `position`, `normal` and `tangent` at rest, with texture coordinates.
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    /// Each vertex's cell corners and weights, as `SoftBody::embed` gives them.
    pub skin: Vec<[(u32, f32); 8]>,
    /// Takes the vertices' frame to the lattice's cell coordinates.
    pub rest_to_cell: Mat3,
    pub material: GpuMaterial,
}

/// A sharp local dent in a mesh, finer than its lattice, in its rest frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dent {
    pub point: Vec3,
    /// Into the body.
    pub normal: Vec3,
    pub depth: f32,
    pub radius: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddedModelId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddedInstanceId(usize);

/// One vertex's corners, as the shader reads them.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Skin {
    nodes: [u32; 8],
    weights: [f32; 8],
}

impl Skin {
    fn new(corners: &[(u32, f32); 8]) -> Self {
        Self {
            nodes: corners.map(|c| c.0),
            weights: corners.map(|c| c.1),
        }
    }
}

/// One instance's dispatch.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Constants {
    basis: [[f32; 4]; 3],
    normal_basis: [[f32; 4]; 3],
    first_rest: u32,
    count: u32,
    first_out: u32,
    shape: u32,
    nodes: u32,
    dents: u32,
    padding: [u32; 2],
}

/// `m`'s rows, padded for the shader.
fn rows(m: Mat3) -> [[f32; 4]; 3] {
    let m = m.transpose();
    [m.x_axis, m.y_axis, m.z_axis].map(|r| [r.x, r.y, r.z, 0.0])
}

/// Lattice nodes, then two entries per dent. Only the newest dents are kept.
fn pack_shape(shape: &mut Vec<[f32; 4]>, nodes: &[[f32; 4]], dents: &[Dent]) -> u32 {
    shape.extend_from_slice(nodes);
    let kept = &dents[dents.len().saturating_sub(MAX_DENTS)..];
    for d in kept {
        let (p, n) = (d.point, d.normal);
        shape.push([p.x, p.y, p.z, d.radius]);
        shape.push([n.x, n.y, n.z, d.depth]);
    }
    kept.len() as u32
}

struct Model {
    first_rest: u32,
    vertices: u32,
    indices: Vec<u32>,
    material: GpuMaterial,
    rest_to_cell: Mat3,
    /// The lattice nodes the skin needs.
    nodes: usize,
    bounds: (Vec3, Vec3),
}

struct Instance {
    model: usize,
    first_out: u32,
    transform: Mat4,
    nodes: Vec<[f32; 4]>,
    dents: Vec<Dent>,
    dirty: bool,
    bounds: (Vec3, Vec3),
}

/// The pipeline and everything drawn through it.
#[derive(Default)]
pub(crate) struct Embedded {
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    sets: Vec<vk::DescriptorSet>,
    /// Lattices and dents changed this frame, one buffer per frame in flight.
    shapes: Vec<Buffer>,
    rest: Buffer,
    skin: Buffer,
    rest_data: Vec<Vertex>,
    skin_data: Vec<Skin>,
    models: Vec<Model>,
    instances: Vec<Instance>,
    shape_data: Vec<[f32; 4]>,
    dispatches: Vec<Constants>,
    dispatched: u64,
}

impl Embedded {
    pub(crate) unsafe fn new(device: &ash::Device) -> Result<Self> {
        let bindings = [0, 1, 2, 3].map(|binding| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        });
        let set_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )?;
        let mut this = Self {
            set_layout,
            ..Self::default()
        };
        let result = (|| -> Result<()> {
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count((FRAMES * 4) as u32)];
            this.pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(FRAMES as u32)
                    .pool_sizes(&sizes),
                None,
            )?;
            let layouts = vec![set_layout; FRAMES];
            this.sets = device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(this.pool)
                    .set_layouts(&layouts),
            )?;
            let push = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(mem::size_of::<Constants>() as u32)];
            this.layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[set_layout])
                    .push_constant_ranges(&push),
                None,
            )?;
            this.pipeline = gpu::compute_pipeline(
                device,
                vk::PipelineCache::null(),
                this.layout,
                include_bytes!(concat!(env!("OUT_DIR"), "/embedded.embedMain.spv")),
                c"embedMain",
            )?;
            Ok(())
        })();
        this.shapes.resize_with(FRAMES, Buffer::default);
        match result {
            Ok(()) => Ok(this),
            Err(error) => {
                this.destroy_objects(device);
                Err(error)
            }
        }
    }

    unsafe fn destroy_objects(&self, device: &ash::Device) {
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.layout, None);
        device.destroy_descriptor_pool(self.pool, None);
        device.destroy_descriptor_set_layout(self.set_layout, None);
    }

    /// The GPU must be idle.
    pub(crate) unsafe fn destroy(mut self, device: &ash::Device, allocator: &mut Allocator) {
        for buffer in mem::take(&mut self.shapes) {
            gpu::destroy_buffer(device, allocator, buffer);
        }
        gpu::destroy_buffer(device, allocator, mem::take(&mut self.rest));
        gpu::destroy_buffer(device, allocator, mem::take(&mut self.skin));
        self.destroy_objects(device);
    }

    /// Forget every model: their vertices went with the shared buffer.
    pub(crate) fn clear(&mut self) {
        self.models.clear();
        self.instances.clear();
        self.rest_data.clear();
        self.skin_data.clear();
        self.dispatches.clear();
    }

    /// Queue this frame's changed instances and upload what they need. The
    /// frame's fence must have signaled.
    pub(crate) unsafe fn prepare(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        frame: usize,
        vertices: &Buffer,
    ) -> Result<()> {
        self.shape_data.clear();
        self.dispatches.clear();
        for instance in self.instances.iter_mut().filter(|i| i.dirty) {
            instance.dirty = false;
            let model = &self.models[instance.model];
            let shape = self.shape_data.len() as u32;
            let dents = pack_shape(&mut self.shape_data, &instance.nodes, &instance.dents);
            self.dispatches.push(Constants {
                basis: rows(model.rest_to_cell),
                normal_basis: rows(model.rest_to_cell.inverse().transpose()),
                first_rest: model.first_rest,
                count: model.vertices,
                first_out: instance.first_out,
                shape,
                nodes: instance.nodes.len() as u32,
                dents,
                padding: [0; 2],
            });
        }
        if self.dispatches.is_empty() {
            return Ok(());
        }
        let bytes: &[u8] = bytemuck::cast_slice(&self.shape_data);
        let buffer = &mut self.shapes[frame];
        if bytes.len() as u64 > buffer.size {
            gpu::destroy_buffer(device, allocator, mem::take(buffer));
            *buffer = gpu::create_buffer(
                device,
                allocator,
                "embedded shapes",
                (bytes.len() as u64).next_power_of_two(),
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::CpuToGpu,
            )?;
        }
        buffer.write(bytes);
        let infos = [&self.rest, &self.skin, &self.shapes[frame], vertices].map(|b| [b.info()]);
        let writes: Vec<_> = (infos.iter().enumerate())
            .map(|(binding, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(self.sets[frame])
                    .dst_binding(binding as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        device.update_descriptor_sets(&writes, &[]);
        Ok(())
    }

    /// Run the queued dispatches, between whatever last read the vertices and
    /// whatever reads them next.
    pub(crate) unsafe fn record(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame: usize,
        vertices: vk::Buffer,
    ) {
        if self.dispatches.is_empty() {
            return;
        }
        use vk::AccessFlags2 as A;
        use vk::PipelineStageFlags2 as S;
        let barrier = |src_stage, src_access, dst_stage, dst_access| {
            let barrier = vk::BufferMemoryBarrier2::default()
                .src_stage_mask(src_stage)
                .src_access_mask(src_access)
                .dst_stage_mask(dst_stage)
                .dst_access_mask(dst_access)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(vertices)
                .size(vk::WHOLE_SIZE);
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().buffer_memory_barriers(&[barrier]),
            );
        };
        barrier(
            S::ALL_COMMANDS,
            A::MEMORY_READ,
            S::COMPUTE_SHADER,
            A::SHADER_STORAGE_WRITE,
        );
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.layout,
            0,
            &[self.sets[frame]],
            &[],
        );
        for constants in &self.dispatches {
            device.cmd_push_constants(
                cmd,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(constants),
            );
            device.cmd_dispatch(cmd, constants.count.div_ceil(WORKGROUP), 1, 1);
        }
        barrier(
            S::COMPUTE_SHADER,
            A::SHADER_STORAGE_WRITE,
            S::ALL_COMMANDS,
            A::MEMORY_READ,
        );
        self.dispatched += self.dispatches.len() as u64;
    }

    /// Each model that has anything to draw: its indices, material and the
    /// instances to draw it with, as output vertex, transform and bounds. An
    /// instance with no shape yet has nothing in its vertices.
    pub(crate) fn draws(&self) -> impl Iterator<Item = EmbeddedDraw<'_>> {
        (self.models.iter().enumerate()).filter_map(|(index, model)| {
            let shaped = |i: &&Instance| i.model == index && !i.nodes.is_empty();
            self.instances.iter().any(|i| shaped(&i)).then(|| {
                let instances = self.instances.iter().filter(shaped);
                EmbeddedDraw {
                    indices: &model.indices,
                    material: model.material,
                    instances: instances
                        .map(|i| (i.first_out, i.transform, i.bounds))
                        .collect(),
                }
            })
        })
    }
}

pub(crate) struct EmbeddedDraw<'a> {
    pub indices: &'a [u32],
    pub material: GpuMaterial,
    pub instances: Vec<(u32, Mat4, (Vec3, Vec3))>,
}

/// The nodes a skin reads: one past the highest it names.
fn nodes_needed(skin: &[[(u32, f32); 8]]) -> usize {
    (skin.iter().flatten())
        .filter(|c| c.0 != MISSING)
        .fold(0, |n, c| n.max(c.0 as usize + 1))
}

impl EmbeddedModel {
    fn validate(&self) -> Result<()> {
        let count = self.vertices.len();
        if count == 0 || self.skin.len() != count {
            return Err(failure("an embedded model needs one skin per vertex"));
        }
        if self.indices.iter().any(|&i| i as usize >= count) {
            return Err(failure("an embedded model's index is out of range"));
        }
        if self.rest_to_cell.determinant().abs() < f32::EPSILON {
            return Err(failure("an embedded model's rest_to_cell is singular"));
        }
        Ok(())
    }
}

impl Renderer {
    /// Upload a mesh for instances to share. This waits for the GPU, so add
    /// models while loading.
    pub fn add_embedded_model(&mut self, model: EmbeddedModel) -> Result<EmbeddedModelId> {
        model.validate()?;
        let bounds = (model.vertices.iter()).fold((Vec3::MAX, Vec3::MIN), |(min, max), v| {
            let p = Vec3::from_slice(&v.position[..3]);
            (min.min(p), max.max(p))
        });
        let embedded = self.embedded.as_mut().expect("created with the renderer");
        let first_rest = embedded.rest_data.len() as u32;
        embedded.rest_data.extend_from_slice(&model.vertices);
        embedded.skin_data.extend(model.skin.iter().map(Skin::new));
        let id = EmbeddedModelId(embedded.models.len());
        embedded.models.push(Model {
            first_rest,
            vertices: model.vertices.len() as u32,
            nodes: nodes_needed(&model.skin),
            indices: model.indices,
            material: model.material,
            rest_to_cell: model.rest_to_cell,
            bounds,
        });
        unsafe {
            self.device.device_wait_idle()?;
            let allocator = self.allocator.as_mut().unwrap();
            let embedded = self.embedded.as_mut().expect("created with the renderer");
            // shortcut: rewrites every model, which is fine for the few a game
            // loads; append in place if models ever stream in.
            for (buffer, name, bytes) in [
                (
                    &mut embedded.rest,
                    "embedded rest vertices",
                    bytemuck::cast_slice::<_, u8>(&embedded.rest_data),
                ),
                (
                    &mut embedded.skin,
                    "embedded skin",
                    bytemuck::cast_slice::<_, u8>(&embedded.skin_data),
                ),
            ] {
                gpu::destroy_buffer(&self.device, allocator, mem::take(buffer));
                *buffer = gpu::create_buffer(
                    &self.device,
                    allocator,
                    name,
                    bytes.len() as u64,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    MemoryLocation::CpuToGpu,
                )?;
                buffer.write(bytes);
            }
        }
        Ok(id)
    }

    /// Reserve `count` instances of a model. Nothing draws until each has a
    /// shape from [`Renderer::set_embedded_shape`]. This waits for the GPU.
    pub fn add_embedded_instances(
        &mut self,
        model: EmbeddedModelId,
        count: usize,
    ) -> Result<Vec<EmbeddedInstanceId>> {
        let embedded = self.embedded.as_ref().expect("created with the renderer");
        let target = (embedded.models.get(model.0)).ok_or_else(|| failure("unknown model"))?;
        let (vertices, bounds) = (target.vertices, target.bounds);
        let zeros = vec![Vertex::zeroed(); vertices as usize * count];
        let (first, _) = unsafe {
            self.device.device_wait_idle()?;
            self.append_geometry(&zeros, &[])?
        };
        let embedded = self.embedded.as_mut().expect("created with the renderer");
        let start = embedded.instances.len();
        embedded
            .instances
            .extend((0..count as u32).map(|i| Instance {
                model: model.0,
                first_out: first + i * vertices,
                transform: Mat4::IDENTITY,
                nodes: Vec::new(),
                dents: Vec::new(),
                dirty: false,
                bounds,
            }));
        Ok((start..start + count).map(EmbeddedInstanceId).collect())
    }

    /// Place an instance: its carrier's transform. This is all a car that is
    /// only moving needs each frame.
    pub fn set_embedded_pose(&mut self, instance: EmbeddedInstanceId, transform: Mat4) {
        let embedded = self.embedded.as_mut().expect("created with the renderer");
        if let Some(instance) = embedded.instances.get_mut(instance.0) {
            instance.transform = transform;
        }
    }

    /// Change an instance's shape: its lattice in its carrier's frame, and the
    /// dents to press in. Call this only when the shape changed.
    pub fn set_embedded_shape(
        &mut self,
        instance: EmbeddedInstanceId,
        nodes: &[Vec3],
        dents: &[Dent],
    ) -> Result<()> {
        let embedded = self.embedded.as_mut().expect("created with the renderer");
        let instance = (embedded.instances.get_mut(instance.0))
            .ok_or_else(|| failure("unknown embedded instance"))?;
        let model = &embedded.models[instance.model];
        if nodes.len() < model.nodes {
            return Err(failure("too few lattice nodes for this model's skin"));
        }
        instance.nodes.clear();
        instance
            .nodes
            .extend(nodes.iter().map(|p| [p.x, p.y, p.z, 0.0]));
        instance.dents.clear();
        instance.dents.extend_from_slice(dents);
        // The mesh stays near its lattice, which is what moves.
        instance.bounds =
            (nodes.iter()).fold(model.bounds, |(min, max), &p| (min.min(p), max.max(p)));
        instance.dirty = true;
        Ok(())
    }

    /// Dispatches run so far: one per instance per change of shape.
    pub fn embedded_dispatches(&self) -> u64 {
        self.embedded.as_ref().map_or(0, |e| e.dispatched)
    }

    /// An instance's deformed vertices, read back from the GPU. For tests: this
    /// waits for the GPU.
    pub fn read_embedded_vertices(&mut self, instance: EmbeddedInstanceId) -> Result<Vec<Vertex>> {
        let embedded = self.embedded.as_ref().expect("created with the renderer");
        let target = (embedded.instances.get(instance.0))
            .ok_or_else(|| failure("unknown embedded instance"))?;
        let count = embedded.models[target.model].vertices as usize;
        let stride = mem::size_of::<Vertex>() as u64;
        let region = vk::BufferCopy {
            src_offset: target.first_out as u64 * stride,
            dst_offset: 0,
            size: count as u64 * stride,
        };
        unsafe {
            self.device.device_wait_idle()?;
            let allocator = self.allocator.as_mut().unwrap();
            let mut readback = gpu::create_buffer(
                &self.device,
                allocator,
                "embedded readback",
                region.size,
                vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuToCpu,
            )?;
            let (source, target) = (self.vertices.buffer, readback.buffer);
            let copied = self.submit_once(|cmd| {
                self.device.cmd_copy_buffer(cmd, source, target, &[region]);
            });
            let vertices = copied.map(|()| {
                let mut vertices = vec![Vertex::zeroed(); count];
                bytemuck::cast_slice_mut(&mut vertices)
                    .copy_from_slice(&readback.read()[..region.size as usize]);
                vertices
            });
            let allocator = self.allocator.as_mut().unwrap();
            gpu::destroy_buffer(&self.device, allocator, mem::take(&mut readback));
            vertices
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dent(depth: f32) -> Dent {
        Dent {
            point: Vec3::X,
            normal: Vec3::NEG_X,
            depth,
            radius: 0.5,
        }
    }

    #[test]
    fn a_shape_is_its_nodes_then_its_newest_dents() {
        let dents: Vec<_> = (0..MAX_DENTS + 3).map(|i| dent(i as f32)).collect();
        let mut shape = vec![[9.0; 4]];
        let kept = pack_shape(&mut shape, &[[1.0; 4], [2.0; 4]], &dents);
        assert_eq!(kept as usize, MAX_DENTS);
        assert_eq!(shape.len(), 3 + 2 * MAX_DENTS);
        assert_eq!(shape[1], [1.0; 4]);
        assert_eq!(shape[3], [1.0, 0.0, 0.0, 0.5]);
        // The oldest three are gone, and the newest is last.
        assert_eq!(shape[4][3], 3.0);
        assert_eq!(shape[shape.len() - 1][3], (MAX_DENTS + 2) as f32);
    }

    #[test]
    fn missing_corners_need_no_node() {
        let mut corners = [(MISSING, 0.0); 8];
        corners[2] = (6, 1.0);
        assert_eq!(nodes_needed(&[corners]), 7);
        assert_eq!(nodes_needed(&[[(MISSING, 0.0); 8]]), 0);
    }

    #[test]
    fn constants_fit_guaranteed_push_constant_space() {
        assert!(mem::size_of::<Constants>() <= 128);
    }

    #[test]
    fn rows_are_a_matrix_as_the_shader_applies_it() {
        let m = Mat3::from_cols(Vec3::X, Vec3::new(2.0, 1.0, 0.0), Vec3::Z);
        let r = rows(m);
        let v = Vec3::new(1.0, 3.0, 5.0);
        let applied = Vec3::new(
            Vec3::from_slice(&r[0][..3]).dot(v),
            Vec3::from_slice(&r[1][..3]).dot(v),
            Vec3::from_slice(&r[2][..3]).dot(v),
        );
        assert_eq!(applied, m * v);
    }
}
