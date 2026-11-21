//! Which shadow layers a frame redraws, and which casters each one draws.
//!
//! Layers `0..CASCADES` are the sun's cascades; the rest are point-light cube
//! faces, six per shadowed light. Each layer draws only the casters whose
//! bounds reach its projection, and keeps what it last drew from as a key:
//! a layer redraws when its key changes, or every frame while geometry moves.
//! A sun cascade's projection already reaches `SHADOW_CASTER_REACH` toward the
//! sun and the shadow pipeline does not clamp depth, so the cull is exact.
//!
//! Outer cascades (from index 2) redraw on a schedule: each pair of cascades
//! refreshes half as often as the pair before it, staggered so every frame
//! draws about the same number of layers. One that waits keeps the projection
//! it was drawn with so the resolve samples it where it was rendered, and
//! waits only while its old map still covers the camera's slice.
//!
//! Casters under half a texel of a cascade are left out of it: their shadow
//! would be smaller than what the cascade can show.

use crate::{
    meshlets, scene, Renderer, EMBEDDED_PRIMITIVE, NO_PRIMITIVE,
};
use glam::{Mat4, Vec3};
use verity_matrix::Bounds;

const LAYERS: usize = scene::CASCADES + scene::MAX_POINT_SHADOWS * scene::POINT_SHADOW_FACES;
/// The first cascade allowed to wait.
const FIRST_LAZY_CASCADE: usize = 2;
/// The longest a cascade waits between redraws, in frames.
const MAX_PERIOD: u64 = 8;
/// A caster whose bounding radius is under this many texels skips a cascade.
const SMALL_CASTER_TEXELS: f32 = 0.5;

/// A cascade's projection and texel size as drawn, and the model generation then.
type Drawn = ([[f32; 4]; 4], f32, u64);

/// How often a cascade redraws while it would every frame, and on which frame
/// of that period: pairs of cascades double the period, the two of a pair
/// take opposite halves, and pairs with the same phase shift by one.
fn schedule(cascade: usize) -> (u64, u64) {
    if cascade < FIRST_LAZY_CASCADE {
        return (1, 0);
    }
    let pair = (cascade / 2) as u32;
    let period = (1u64 << pair).min(MAX_PERIOD);
    let phase = (cascade % 2) as u64 * (period / 2) + u64::from(pair.saturating_sub(2));
    (period, phase % period)
}

#[derive(Default)]
pub(crate) struct ShadowLayers {
    /// Per layer: whether it has a light, its casters as (instance, index
    /// count), what it was last drawn from, and whether this frame draws it.
    enabled: Vec<bool>,
    calls: Vec<Vec<(u32, u32)>>,
    keys: Vec<Vec<u8>>,
    redraw: Vec<bool>,
    /// Per cascade: the projection and texel size its map was drawn with, and
    /// the model generation then.
    drawn: [Option<Drawn>; scene::CASCADES],
    /// Each primitive's bounds, from its meshlets, for `bounds_generation`.
    bounds: Vec<Option<Bounds>>,
    bounds_generation: Option<u64>,
}

impl ShadowLayers {
    pub(crate) fn redraw(&self, layer: usize) -> bool {
        self.redraw.get(layer).copied().unwrap_or(false)
    }

    pub(crate) fn enabled(&self, layer: usize) -> bool {
        self.enabled.get(layer).copied().unwrap_or(false)
    }

    pub(crate) fn calls(&self, layer: usize) -> &[(u32, u32)] {
        self.calls.get(layer).map_or(&[], Vec::as_slice)
    }
}

/// World radius of a caster's bounds as placed by `model`.
fn radius(model: Mat4, bounds: &Bounds) -> f32 {
    let half = (Vec3::from_array(bounds.max) - Vec3::from_array(bounds.min)).length() * 0.5;
    let scale = model
        .x_axis
        .truncate()
        .length()
        .max(model.y_axis.truncate().length())
        .max(model.z_axis.truncate().length());
    half * scale
}

/// Whether the map drawn with `clip` still holds every corner of a slice.
fn covers(clip: Mat4, corners: &[glam::Vec3; 8]) -> bool {
    corners.iter().all(|&corner| {
        let p = clip.project_point3(corner);
        p.is_finite() && p.x.abs() <= 1.0 && p.y.abs() <= 1.0 && (0.0..=1.0).contains(&p.z)
    })
}

impl Renderer {
    /// Cull casters into each layer and decide which layers this frame draws.
    /// A cascade that waits gets its old projection back in `globals`.
    pub(crate) fn plan_shadows(
        &mut self,
        globals: &mut scene::Globals,
        viewport: [u32; 4],
        moving: bool,
    ) {
        let generation = self.model_generation;
        let s = &mut self.shadows;
        if s.bounds_generation != Some(generation) {
            s.bounds = self
                .primitive_meshlets
                .iter()
                .map(|meshlets| {
                    let meshlets = meshlets.as_ref().filter(|m| !m.is_empty())?;
                    Some(meshlets.iter().fold(meshlets[0].bounds, |a, m| Bounds {
                        min: std::array::from_fn(|i| a.min[i].min(m.bounds.min[i])),
                        max: std::array::from_fn(|i| a.max[i].max(m.bounds.max[i])),
                    }))
                })
                .collect();
            s.bounds_generation = Some(generation);
        }
        s.enabled.resize(LAYERS, false);
        s.calls.resize_with(LAYERS, Vec::new);
        s.keys.resize_with(LAYERS, Vec::new);
        s.redraw.resize(LAYERS, false);

        // Placement and object-space bounds of a caster; `None` draws it everywhere.
        let caster = |instance: u32| -> Option<(Mat4, Bounds)> {
            let model = Mat4::from_cols_array_2d(&self.instance_data[instance as usize].model);
            let primitive = self.draw_calls[instance as usize].2;
            let bounds = if primitive == NO_PRIMITIVE {
                return None;
            } else if primitive >= EMBEDDED_PRIMITIVE {
                let (min, max) = self.embedded_bounds[primitive - EMBEDDED_PRIMITIVE];
                Bounds { min: min.to_array(), max: max.to_array() }
            } else {
                (*s.bounds.get(primitive)?)?
            };
            Some((model, bounds))
        };
        let mut key = Vec::new();
        for layer in 0..LAYERS {
            let (enabled, clip) = if layer < scene::CASCADES {
                (self.frame_sun.is_some(), globals.shadow_view_proj[layer])
            } else {
                let point = layer - scene::CASCADES;
                let enabled = self
                    .frame_lights
                    .get(point / scene::POINT_SHADOW_FACES)
                    .is_some_and(|light| {
                        light.radius.is_finite()
                            && light.radius > 0.0
                            && light.position.iter().all(|value| value.is_finite())
                    });
                (enabled, globals.point_shadow_view_proj[point])
            };
            let calls = &mut s.calls[layer];
            calls.clear();
            key.clear();
            key.push(u8::from(enabled));
            if enabled {
                key.extend_from_slice(bytemuck::bytes_of(&clip));
                key.extend_from_slice(&generation.to_ne_bytes());
                let clip = Mat4::from_cols_array_2d(&clip);
                let smallest = if layer < scene::CASCADES {
                    globals.shadow_texels[layer] * SMALL_CASTER_TEXELS
                } else {
                    0.0
                };
                for &(instance, count) in &self.shadow_calls {
                    let placed = caster(instance);
                    if placed.is_some_and(|(model, bounds)| {
                        radius(model, &bounds) < smallest || !meshlets::visible(&bounds, clip, model)
                    }) {
                        continue;
                    }
                    calls.push((instance, count));
                    // What the caster is, not where it sits in this frame's
                    // lists, which shift as other draws come and go.
                    let record = &self.instance_data[instance as usize];
                    let primitive = self.draw_calls[instance as usize].2;
                    let model_primitive = if primitive < EMBEDDED_PRIMITIVE { primitive as u64 } else { u64::MAX };
                    key.extend_from_slice(&model_primitive.to_ne_bytes());
                    key.extend_from_slice(bytemuck::bytes_of(&[count, record.draw[1], record.draw[2]]));
                    key.extend_from_slice(bytemuck::bytes_of(&record.model));
                }
            }
            s.enabled[layer] = enabled;
            s.redraw[layer] = moving || key != s.keys[layer];
            if s.redraw[layer] {
                std::mem::swap(&mut s.keys[layer], &mut key);
            }
        }

        // Outer cascades take turns; the one whose turn it is not may wait.
        let cascades = self.frame_sun.as_ref().map(|sun| {
            let aspect = viewport[2].max(1) as f32 / viewport[3].max(1) as f32;
            scene::cascades(&self.camera, aspect, sun.direction, self.shadow_distance)
        });
        for cascade in 0..scene::CASCADES {
            let (period, phase) = schedule(cascade);
            let turn = (self.frame_number + phase).is_multiple_of(period);
            let wait = s.redraw[cascade]
                && !turn
                && s.drawn[cascade].is_some_and(|(clip, _, drawn_generation)| {
                    drawn_generation == generation
                        && cascades.as_ref().is_some_and(|c| {
                            covers(Mat4::from_cols_array_2d(&clip), &c.corners[cascade])
                        })
                });
            if wait {
                let (clip, texel, _) = s.drawn[cascade].unwrap();
                globals.shadow_view_proj[cascade] = clip;
                globals.shadow_texels[cascade] = texel;
                s.redraw[cascade] = false;
                // Keep the old key so the cascade still differs next frame.
                s.keys[cascade].clear();
            } else if s.redraw[cascade] {
                s.drawn[cascade] = Some((
                    globals.shadow_view_proj[cascade],
                    globals.shadow_texels[cascade],
                    generation,
                ));
            }
        }
        globals.shadow_redraw[0] = (0..LAYERS)
            .filter(|&layer| s.redraw[layer])
            .fold(0, |mask, layer| mask | 1 << layer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn outer_cascades_share_out_their_redraws_evenly() {
        let mut per_frame = [0usize; MAX_PERIOD as usize];
        for cascade in 0..scene::CASCADES {
            let (period, phase) = schedule(cascade);
            assert!(period <= MAX_PERIOD && phase < period, "cascade {cascade}");
            for (frame, count) in per_frame.iter_mut().enumerate() {
                if (frame as u64 + phase).is_multiple_of(period) {
                    *count += 1;
                }
            }
        }
        let (low, high) = (per_frame.iter().min().unwrap(), per_frame.iter().max().unwrap());
        assert!(high - low <= 1, "redraws per frame {per_frame:?}");
        assert!(*high <= 4, "redraws per frame {per_frame:?}");
    }

    #[test]
    fn a_cascade_covers_its_own_slice_but_not_a_distant_one() {
        let camera = scene::Camera::default();
        let sun = Vec3::new(-0.3, -1.0, -0.6);
        let c = scene::cascades(&camera, 16.0 / 9.0, sun, scene::SHADOW_DISTANCE);
        for i in 0..scene::CASCADES {
            assert!(covers(c.view_proj[i], &c.corners[i]), "cascade {i}");
        }
        let moved = scene::Camera {
            eye: camera.eye + Vec3::new(40.0, 0.0, 0.0),
            target: camera.target + Vec3::new(40.0, 0.0, 0.0),
            ..camera
        };
        let m = scene::cascades(&moved, 16.0 / 9.0, sun, scene::SHADOW_DISTANCE);
        assert!(!covers(c.view_proj[2], &m.corners[2]));
    }
}
