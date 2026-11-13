//! Meshes that stay on the GPU until replaced, such as terrain chunks.
//!
//! Vertices live in ranges of the shared vertex buffer and are uploaded once, so
//! a frame costs only the indices of what it draws. A mesh outside the camera
//! and every sun cascade is skipped; the rest are culled like any other draw.

use crate::{
    asset::Sun,
    extension::RetainedChange,
    scene::{self, GpuMaterial, Vertex},
    LoadedMaterial, Renderer, Result,
};
use ash::vk;
use glam::{Mat4, Vec3};
use std::{
    collections::{BTreeMap, BTreeSet},
    mem,
    time::{Duration, Instant},
};

/// The fewest vertices the shared buffer grows by, since growing stalls the GPU.
const MIN_GROWTH: u32 = 1 << 16;
/// How often extensions hear about changes: each hearing waits for the GPU to
/// idle, and bounced light can trail a brush stroke unnoticed.
const NOTIFY_INTERVAL: Duration = Duration::from_millis(500);

/// A mesh in world space.
#[derive(Clone)]
pub struct RetainedMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    /// Minimum and maximum corners of `vertices`.
    pub bounds: (Vec3, Vec3),
    /// Shade with a plain white material and the vertex colours, instead of
    /// the retained material (the terrain's), as generated cities do.
    pub plain: bool,
}

/// First-fit allocator over vertex ranges, kept sorted and coalesced.
#[derive(Default)]
struct Arena {
    free: Vec<(u32, u32)>,
}

impl Arena {
    fn allocate(&mut self, count: u32) -> Option<u32> {
        let slot = self.free.iter().position(|&(_, size)| size >= count)?;
        let (start, size) = self.free[slot];
        if size == count {
            self.free.remove(slot);
        } else {
            self.free[slot] = (start + count, size - count);
        }
        Some(start)
    }

    fn release(&mut self, start: u32, count: u32) {
        let slot = self.free.partition_point(|&(other, _)| other < start);
        self.free.insert(slot, (start, count));
        if self.free.get(slot + 1).map(|r| r.0) == Some(start + count) {
            self.free[slot].1 += self.free.remove(slot + 1).1;
        }
        if slot > 0 && self.free[slot - 1].0 + self.free[slot - 1].1 == start {
            self.free[slot - 1].1 += self.free.remove(slot).1;
        }
    }
}

struct Entry {
    /// Kept for extensions that join later.
    mesh: RetainedMesh,
    /// Its vertices' range of the shared buffer, once they have one.
    first_vertex: Option<u32>,
}

#[derive(Default)]
pub(crate) struct Retained {
    entries: BTreeMap<u64, Entry>,
    arena: Arena,
    capacity: u32,
    material: Option<GpuMaterial>,
    /// Where `material` sits in `Renderer::materials`, which stays put for
    /// extensions, unlike the per-frame materials after it.
    material_index: Option<u32>,
    /// Where the plain material of `RetainedMesh::plain` meshes sits.
    plain_index: Option<u32>,
    /// Meshes extensions have not heard about, and when they last did.
    notify: BTreeSet<u64>,
    notified: Option<Instant>,
    /// This frame's new vertices, and where each mesh's go.
    pub(crate) staging: Vec<Vertex>,
    pub(crate) copies: Vec<vk::BufferCopy>,
    drawn: (usize, usize),
}

impl Retained {
    /// Forget every mesh, for when the shared vertex buffer is rebuilt.
    pub(crate) fn clear(&mut self) {
        *self = Self {
            material: self.material,
            ..Self::default()
        };
    }

    /// Tell the next extension about everything.
    pub(crate) fn notify_all(&mut self) {
        self.notify = self.entries.keys().copied().collect();
        self.notified = None;
    }
}

/// Whether a box is wholly past one side of a clip volume. Depth is not tested,
/// which keeps a cascade's casters at any distance behind it.
fn outside(clip: &Mat4, (min, max): (Vec3, Vec3)) -> bool {
    let corners: [_; 8] = std::array::from_fn(|i| {
        let pick = |bit: usize, axis: usize| {
            if i >> bit & 1 == 0 {
                min[axis]
            } else {
                max[axis]
            }
        };
        *clip * Vec3::new(pick(0, 0), pick(1, 1), pick(2, 2)).extend(1.0)
    });
    corners.iter().all(|c| c.w <= 0.0)
        || corners.iter().all(|c| c.x < -c.w)
        || corners.iter().all(|c| c.x > c.w)
        || corners.iter().all(|c| c.y < -c.w)
        || corners.iter().all(|c| c.y > c.w)
}

impl Renderer {
    /// The material every retained mesh draws with; nothing draws until it is set.
    pub fn set_retained_material(&mut self, material: GpuMaterial) {
        self.retained.material = Some(material);
        if let Some(index) = self.retained.material_index {
            self.materials[index as usize].gpu = material;
        }
    }

    /// Add, replace or, with `None` or an empty mesh, remove the mesh under `key`.
    /// Its vertices upload with the next frame, all at once, so bound how many
    /// are set per frame.
    pub fn set_retained_mesh(&mut self, key: u64, mesh: Option<RetainedMesh>) {
        let retained = &mut self.retained;
        if let Some(old) = retained.entries.remove(&key) {
            if let Some(first) = old.first_vertex {
                let count = old.mesh.vertices.len() as u32;
                retained.arena.release(first, count);
            }
        }
        if let Some(mesh) = mesh.filter(|mesh| !mesh.indices.is_empty()) {
            let first_vertex = None;
            retained.entries.insert(key, Entry { mesh, first_vertex });
        }
        retained.notify.insert(key);
    }

    /// Meshes and triangles of retained geometry in the last frame.
    pub fn retained_drawn(&self) -> (usize, usize) {
        self.retained.drawn
    }

    /// Give new meshes their vertex ranges and stage their vertices.
    pub(crate) unsafe fn place_retained(&mut self) -> Result<()> {
        self.retained.staging.clear();
        self.retained.copies.clear();
        if let (Some(gpu), None) = (self.retained.material, self.retained.material_index) {
            self.retained.material_index = Some(self.materials.len() as u32);
            self.materials.push(LoadedMaterial {
                gpu,
                double_sided: false,
                hidden: false,
            });
        }
        if self.retained.plain_index.is_none() {
            self.retained.plain_index = Some(self.materials.len() as u32);
            let white = verity_core::Material {
                base_color: [1.0; 3],
                metallic: 0.0,
                roughness: 0.8,
            };
            self.materials.push(LoadedMaterial {
                gpu: GpuMaterial::plain(&white),
                double_sided: false,
                hidden: false,
            });
        }
        let waiting = (self.retained.entries.iter()).filter(|(_, e)| e.first_vertex.is_none());
        let keys: Vec<u64> = waiting.clone().map(|(key, _)| *key).collect();
        let mut remaining: u32 = waiting.map(|(_, e)| e.mesh.vertices.len() as u32).sum();
        let stride = mem::size_of::<Vertex>() as u64;
        for key in keys {
            let count = self.retained.entries[&key].mesh.vertices.len() as u32;
            let first = match self.retained.arena.allocate(count) {
                Some(first) => first,
                None => {
                    // Everything placed so far stays put: growing only appends.
                    let added = remaining.max(self.retained.capacity).max(MIN_GROWTH);
                    self.device.device_wait_idle()?;
                    let zeroes = vec![bytemuck::Zeroable::zeroed(); added as usize];
                    let (start, _) = self.append_geometry(&zeroes, &[])?;
                    self.retained.capacity += added;
                    self.retained.arena.release(start, added);
                    (self.retained.arena.allocate(count))
                        .expect("the range just added holds every waiting mesh")
                }
            };
            remaining -= count;
            let retained = &mut self.retained;
            let entry = retained.entries.get_mut(&key).expect("listed above");
            entry.first_vertex = Some(first);
            retained.copies.push(vk::BufferCopy {
                src_offset: retained.staging.len() as u64 * stride,
                dst_offset: first as u64 * stride,
                size: count as u64 * stride,
            });
            retained.staging.extend_from_slice(&entry.mesh.vertices);
        }
        self.notify_extension()
    }

    /// Hand batched changes to the extension, if one is due.
    unsafe fn notify_extension(&mut self) -> Result<()> {
        let retained = &self.retained;
        let due = retained
            .notified
            .is_none_or(|at| at.elapsed() >= NOTIFY_INTERVAL);
        let (true, false, Some(material)) =
            (due, retained.notify.is_empty(), retained.material_index)
        else {
            return Ok(());
        };
        let Some(mut extension) = self.extension.take() else {
            return Ok(());
        };
        let result = self
            .device
            .device_wait_idle()
            .map_err(Into::into)
            .and_then(|()| {
                let keys = mem::take(&mut self.retained.notify);
                let entries = mem::take(&mut self.retained.entries);
                let changes: Vec<RetainedChange> = keys
                    .iter()
                    .map(|&key| RetainedChange {
                        key,
                        mesh: entries.get(&key).map(|entry| &entry.mesh),
                    })
                    .collect();
                let result = extension.retained_changed(&mut self.gpu(), &changes, material);
                drop(changes);
                self.retained.entries = entries;
                result
            });
        self.extension = Some(extension);
        self.retained.notified = Some(Instant::now());
        result
    }

    /// Append this frame's visible meshes, with draw indices from `slot`.
    pub(crate) fn push_retained_draws(
        &mut self,
        slot: u32,
        viewport: [u32; 4],
        sun: Option<&Sun>,
    ) -> Result<()> {
        self.retained.drawn = (0, 0);
        let camera = scene::view_proj(&self.camera, viewport);
        let aspect = viewport[2].max(1) as f32 / viewport[3].max(1) as f32;
        let cascades = sun
            .map(|sun| scene::cascades(&self.camera, aspect, sun.direction, self.shadow_distance));
        let shadows = cascades.iter().flat_map(|c| &c.view_proj);
        let shadows: Vec<&Mat4> = shadows.collect();
        for (slot, entry) in (slot..).zip(self.retained.entries.values()) {
            let bounds = entry.mesh.bounds;
            let material = if entry.mesh.plain {
                self.retained.plain_index
            } else {
                self.retained.material_index
            };
            let (Some(first_vertex), Some(index)) = (entry.first_vertex, material) else {
                continue;
            };
            if outside(&camera, bounds) && shadows.iter().all(|c| outside(c, bounds)) {
                continue;
            }
            let first = u32::try_from(self.frame_indices.len())
                .map_err(|_| crate::failure("too many frame geometry indices"))?;
            let indices = &entry.mesh.indices;
            self.frame_indices.extend_from_slice(indices);
            let mut instance = scene::instance(Mat4::IDENTITY, first, first_vertex, index);
            instance.draw[3] = slot;
            self.instance_data.push(instance);
            let primitive = crate::EMBEDDED_PRIMITIVE + self.embedded_bounds.len();
            self.draw_calls
                .push((indices.len() as u32, false, primitive));
            self.embedded_bounds.push(bounds);
            self.retained.drawn.0 += 1;
            self.retained.drawn.1 += indices.len() / 3;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_reuses_and_coalesces_ranges() {
        let mut arena = Arena::default();
        arena.release(100, 50);
        let [a, b, c] = [10, 20, 20].map(|n| arena.allocate(n).unwrap());
        assert_eq!((a, b, c), (100, 110, 130));
        assert_eq!(arena.allocate(1), None);
        arena.release(b, 20);
        assert_eq!(arena.allocate(21), None);
        arena.release(c, 20);
        arena.release(a, 10);
        assert_eq!(arena.free, [(100, 50)]);
        assert_eq!(arena.allocate(50), Some(100));
        assert!(arena.free.is_empty());
    }

    #[test]
    fn boxes_are_outside_only_past_a_side() {
        let camera = scene::Camera {
            eye: Vec3::ZERO,
            target: Vec3::X,
            vertical_fov: 1.0,
        };
        let clip = scene::view_proj(&camera, [0, 0, 100, 100]);
        let around = |p: Vec3| (p - 1.0, p + 1.0);
        assert!(!outside(&clip, around(Vec3::X * 50.0)));
        assert!(outside(&clip, around(Vec3::X * -50.0)), "behind the eye");
        assert!(outside(&clip, around(Vec3::new(5.0, 50.0, 0.0))));
        assert!(outside(&clip, around(Vec3::new(5.0, 0.0, 50.0))));
        // Straddling the eye plane, and enclosing the eye.
        assert!(!outside(&clip, (Vec3::splat(-5.0), Vec3::splat(5.0))));
    }
}
