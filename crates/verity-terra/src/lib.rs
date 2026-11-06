//! Eroded, smooth, editable voxel terrain, Z-up, in metres.
//!
//! [`Landscape::generate`] shapes a heightfield from a [`Landform`] preset and
//! erodes it. [`Terrain`] caches it as a sparse 3D density field that can be
//! sampled, ray cast and dug into, so cavities and overhangs work. [`Mesher`]
//! turns dirty chunks into [`MeshData`] on worker threads, and drops results an
//! edit has overtaken. Density is negative inside the ground.

use fast_surface_nets::{
    SurfaceNetsBuffer,
    ndshape::{ConstShape, ConstShape3u32},
    surface_nets,
};
use glam::{DVec2, FloatExt, IVec3, Vec2, Vec3};
use noise::{NoiseFn, Perlin};
use rayon::prelude::*;
use std::{
    cell::RefCell,
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet},
    sync::{
        Arc,
        mpsc::{Receiver, Sender, channel},
    },
};
use verity_physics::Ground;

pub const CELLS: i32 = 24;
pub const VOXEL: f32 = 1.0;
pub const HALF_WORLD: i32 = 192;
pub const MIN_Z: i32 = -48;
pub const MAX_Z: i32 = 144;
pub const SIDE: usize = (HALF_WORLD * 2 + 1) as usize;
pub type Shape = ConstShape3u32<26, 26, 26>;
const N: u32 = CELLS as u32 + 2;

/// The large-scale shape of the world. Start from a preset with `Settings::from(Landform::Island)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Landform {
    /// Ridged highlands split by a winding river valley. No sea.
    #[default]
    Valley,
    /// One island with a noisy coastline, beaches and a mountainous interior.
    Island,
    /// A scattered chain of small islands.
    Archipelago,
    /// A continental coast: inland mountain range, coastal lowlands, open sea on one side.
    Continent,
    /// High alpine ridges everywhere. No sea.
    Mountains,
    /// Gentle rolling lowland with a few low ridges. No sea.
    Plains,
}

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub seed: u32,
    pub landform: Landform,
    /// Water height in metres. `None` means a dry world; droplets reaching the sea deposit their load.
    pub sea_level: Option<f32>,
    /// Ridged mountain amplitude in metres.
    pub mountains: f32,
    /// Rolling hill amplitude in metres.
    pub hills: f32,
    /// Iterations of stream-power river incision with tectonic uplift. 0 disables it.
    pub fluvial_steps: usize,
    pub droplets: usize,
    pub thermal_steps: usize,
    pub erosion_rate: f32,
    pub deposition_rate: f32,
    pub evaporation: f32,
    pub talus: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Landform::Valley.into()
    }
}

impl From<Landform> for Settings {
    fn from(landform: Landform) -> Self {
        let (sea_level, mountains, hills) = match landform {
            Landform::Valley => (None, 44.0, 22.0),
            Landform::Island => (Some(0.0), 80.0, 10.0),
            Landform::Archipelago => (Some(0.0), 40.0, 8.0),
            Landform::Continent => (Some(-10.0), 85.0, 12.0),
            Landform::Mountains => (None, 95.0, 14.0),
            Landform::Plains => (None, 24.0, 16.0),
        };
        Self {
            seed: 1337,
            landform,
            sea_level,
            mountains,
            hills,
            fluvial_steps: 60,
            droplets: 4_000,
            thermal_steps: 25,
            erosion_rate: 0.24,
            deposition_rate: 0.18,
            evaporation: 0.025,
            talus: 0.75,
        }
    }
}

#[derive(Clone)]
pub struct Landscape {
    pub heights: Vec<f32>,
    pub flow: Vec<f32>,
    pub noise: Perlin,
}

// Deterministic sampling without an additional random-number dependency.
struct Random(u64);
impl Random {
    fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 40) as f32) / ((1u32 << 24) as f32)
    }
}

impl Landscape {
    pub fn generate(settings: Settings) -> Self {
        let noise = Perlin::new(settings.seed);
        let cells: Vec<(f32, f32)> = (0..SIDE * SIDE)
            .into_par_iter()
            .map(|i| {
                let p = DVec2::new((i % SIDE) as f64, (i / SIDE) as f64) - HALF_WORLD as f64;
                let (height, rock) = relief(&noise, &settings, p);
                let height = height.clamp(MIN_Z as f64 + 4.0, MAX_Z as f64 - 8.0);
                (height as f32, rock as f32)
            })
            .collect();
        let (heights, hardness): (Vec<f32>, Vec<f32>) = cells.into_iter().unzip();
        let mut terrain = Self {
            heights,
            flow: vec![0.0; SIDE * SIDE],
            noise,
        };
        terrain.fluvial(settings, &hardness);
        terrain.erode(settings, &hardness);
        terrain
    }

    fn map_sample(map: &[f32], p: Vec2) -> (f32, Vec2) {
        let p = p.clamp(Vec2::ZERO, Vec2::splat((SIDE - 2) as f32));
        let [x, z] = p.floor().as_uvec2().to_array();
        let f = p.fract();
        let i = z as usize * SIDE + x as usize;
        let [a, b, c, d] = [map[i], map[i + 1], map[i + SIDE], map[i + SIDE + 1]];
        let height = a.lerp(b, f.x).lerp(c.lerp(d, f.x), f.y);
        let gradient = Vec2::new((b - a).lerp(d - c, f.y), (c - a).lerp(d - b, f.x));
        (height, gradient)
    }

    fn deposit(&mut self, p: Vec2, sediment: f32) {
        let [x, z] = p.floor().as_uvec2().to_array();
        let f = p.fract();
        let i = z as usize * SIDE + x as usize;
        for (offset, weight) in [
            (0, (1.0 - f.x) * (1.0 - f.y)),
            (1, f.x * (1.0 - f.y)),
            (SIDE, (1.0 - f.x) * f.y),
            (SIDE + 1, f.x * f.y),
        ] {
            self.heights[i + offset] += sediment * weight;
        }
    }

    /// Landscape evolution: uplift balanced by stream-power incision (Braun & Willett 2013,
    /// implicit O(n) solver). Flow is routed over depressions with an epsilon priority flood
    /// (Barnes et al. 2014), so every cell drains to the sea or the map edge. Only cells above a
    /// large channel-head drainage area incise: a few connected river valleys cut through the
    /// relief while the slopes between them keep their shape.
    fn fluvial(&mut self, s: Settings, hardness: &[f32]) {
        const INCISION: f32 = 0.035; // K·dt, with drainage area in cells (m = 0.5, n = 1).
        const UPLIFT: f32 = 0.015; // Fraction of the deficit to the preset relief restored per step.
        const CHANNEL_HEAD: f32 = 200.0; // Drainage area in cells needed before a channel incises.
        let n = SIDE * SIDE;
        let sea = s.sea_level.unwrap_or(f32::NEG_INFINITY);
        let target = self.heights.clone();
        let mut filled = vec![0.0f32; n];
        let mut receiver = vec![0usize; n];
        let mut area = vec![0.0f32; n];
        let mut done = vec![false; n];
        let mut order = Vec::with_capacity(n);
        let mut heap = BinaryHeap::new();
        // Per route: the uplift rate of unchannelled cells, and each channel cell
        // in downstream order with its receiver and incision factor.
        let mut uplift = vec![0.0f32; n];
        let mut channels: Vec<(u32, u32, f32)> = Vec::new();
        let mut rng = Random(s.seed as u64 + 7);
        for step in 0..s.fluvial_steps {
            // Reroute every few steps: the update below stays valid on a slightly stale graph.
            if step % 4 == 0 {
                // Base level: the sea and the map border.
                done.fill(false);
                order.clear();
                for (i, &h) in self.heights.iter().enumerate() {
                    if h < sea || is_border(i) {
                        done[i] = true;
                        filled[i] = h;
                        heap.push(Reverse(flood_key(h, i)));
                    }
                }
                while let Some(Reverse(key)) = heap.pop() {
                    let i = key as u32 as usize;
                    let level = filled[i];
                    order.push(i);
                    for_neighbours(i, |j, _| {
                        if !done[j] {
                            done[j] = true;
                            // A random epsilon keeps filled basins sloping to their spill point without
                            // the parallel straight channels a constant one carves across flats.
                            filled[j] = self.heights[j].max(level + 1e-3 * (0.1 + rng.unit()));
                            heap.push(Reverse(flood_key(filled[j], j)));
                        }
                    });
                }
                // Steepest descent on the filled surface. A lower neighbour always exists (the cell
                // that flooded it) and was popped earlier, so `order` runs downstream to upstream.
                let (heights, filled) = (&self.heights, &filled);
                receiver.par_iter_mut().enumerate().for_each(|(i, r)| {
                    *r = i;
                    if heights[i] < sea || is_border(i) {
                        return;
                    }
                    let mut best = 0.0;
                    for_neighbours(i, |j, diagonal| {
                        let slope = (filled[i] - filled[j]) / if diagonal { SQRT_2 } else { 1.0 };
                        if slope > best {
                            best = slope;
                            *r = j;
                        }
                    });
                });
                area.fill(1.0);
                for &i in order.iter().rev() {
                    if receiver[i] != i {
                        area[receiver[i]] += area[i];
                    }
                }
                // Below the channel-head area slopes stay unchannelled and keep their relief. A
                // channel's receiver drains more than it does, so it is a channel or a base level:
                // only channels depend on their neighbours, and only they need `order`.
                uplift.fill(0.0);
                channels.clear();
                for &i in &order {
                    let r = receiver[i];
                    if r == i {
                        continue;
                    }
                    let length = if i % SIDE != r % SIDE && i / SIDE != r / SIDE {
                        SQRT_2
                    } else {
                        1.0
                    };
                    let erodibility = INCISION * (1.0 - 0.6 * hardness[i]); // Hard rock forms gorges.
                    let f = erodibility * (area[i] - CHANNEL_HEAD).max(0.0).sqrt() / length;
                    if f > 0.0 {
                        channels.push((i as u32, r as u32, f));
                    } else {
                        uplift[i] = UPLIFT;
                    }
                }
            }
            for ((h, t), rate) in self.heights.iter_mut().zip(&target).zip(&uplift) {
                *h += (t - *h) * rate;
            }
            for &(i, r, f) in &channels {
                let (i, r) = (i as usize, r as usize);
                let raised = self.heights[i] + (target[i] - self.heights[i]) * UPLIFT;
                // Implicit in the downstream height, so large rivers stay stable; lakes don't incise.
                self.heights[i] = if raised > self.heights[r] {
                    (raised + f * self.heights[r]) / (1.0 + f)
                } else {
                    raised
                };
            }
        }
        if s.fluvial_steps > 0 {
            // Drainage area seeds the wetness map so valley floors follow the river network.
            for (f, a) in self.flow.iter_mut().zip(&area) {
                *f = a * 0.05;
            }
        }
    }

    fn erode(&mut self, s: Settings, hardness: &[f32]) {
        let mut rng = Random(s.seed as u64 + 1);
        let sea = s.sea_level.unwrap_or(f32::NEG_INFINITY);
        let mut kernel = [0.0f32; 25];
        for (i, w) in kernel.iter_mut().enumerate() {
            *w = (2.5 - Vec2::new((i % 5) as f32 - 2.0, (i / 5) as f32 - 2.0).length()).max(0.0);
        }
        let total: f32 = kernel.iter().sum();
        kernel.iter_mut().for_each(|w| *w /= total);
        // shortcut: sequential droplets avoid write races, move transport to the GPU for larger maps.
        for _ in 0..s.droplets {
            let mut p = Vec2::new(rng.unit(), rng.unit()) * (SIDE - 5) as f32 + Vec2::splat(2.0);
            let mut direction = Vec2::ZERO;
            let (mut speed, mut water, mut sediment) = (1.0f32, 1.0f32, 0.0f32);
            for _ in 0..70 {
                let (height, gradient) = Self::map_sample(&self.heights, p);
                if height < sea {
                    break; // Rivers drop their load where they meet the sea, building deltas.
                }
                direction = (direction * 0.35 - gradient * 0.65).normalize_or_zero();
                if direction == Vec2::ZERO {
                    break;
                }
                let next = p + direction;
                if next.min_element() < 2.0 || next.max_element() >= (SIDE - 3) as f32 {
                    break;
                }
                let dh = Self::map_sample(&self.heights, next).0 - height;
                let capacity = (-dh).max(0.02) * speed * water * 3.2;
                if dh > 0.0 || sediment > capacity {
                    let amount = if dh > 0.0 {
                        sediment.min(dh)
                    } else {
                        (sediment - capacity) * s.deposition_rate
                    };
                    sediment -= amount;
                    self.deposit(p, amount);
                } else {
                    let amount = ((capacity - sediment) * s.erosion_rate).min(-dh).min(0.65);
                    let corner = p.floor().as_uvec2() - 2;
                    for (k, weight) in kernel.iter().enumerate() {
                        let i = (corner.y as usize + k / 5) * SIDE + corner.x as usize + k % 5;
                        self.heights[i] -= amount * weight;
                    }
                    sediment += amount;
                }
                self.flow[p.y as usize * SIDE + p.x as usize] += water;
                speed = (speed * speed - dh * 4.0).max(0.01).sqrt();
                water *= 1.0 - s.evaporation;
                p = next;
                if water < 0.05 {
                    break;
                }
            }
            // Deposit remaining load at the terminal point to conserve sediment on the map.
            self.deposit(p, sediment);
        }
        // Each cell sheds to its lowest neighbour, then gathers what was shed to it:
        // two passes without write races, and identical on any thread count.
        let mut shed = vec![(0.0f32, usize::MAX); self.heights.len()];
        for _ in 0..s.thermal_steps {
            let heights = &self.heights;
            shed.par_iter_mut().enumerate().for_each(|(i, out)| {
                if is_border(i) {
                    return;
                }
                let lowest = [i - 1, i + 1, i - SIDE, i + SIDE]
                    .into_iter()
                    .min_by(|a, b| heights[*a].total_cmp(&heights[*b]))
                    .expect("four neighbours");
                // Hard rock holds near-vertical cliffs; soil relaxes to the talus angle.
                let talus = s.talus * (1.0 + 3.0 * hardness[i]);
                *out = (
                    ((heights[i] - heights[lowest] - talus) * 0.2).max(0.0),
                    lowest,
                );
            });
            let shed = &shed;
            self.heights.par_iter_mut().enumerate().for_each(|(i, h)| {
                *h -= shed[i].0;
                for j in [i.wrapping_sub(1), i + 1, i.wrapping_sub(SIDE), i + SIDE] {
                    if let Some(&(amount, _)) = shed.get(j).filter(|(_, to)| *to == i) {
                        *h += amount;
                    }
                }
            });
        }
        for value in &mut self.flow {
            *value = (1.0 + *value).ln() / 7.0;
        }
    }

    pub fn height(&self, x: f32, y: f32) -> f32 {
        Self::map_sample(
            &self.heights,
            Vec2::new(x, y) + Vec2::splat(HALF_WORLD as f32),
        )
        .0
    }

    pub fn wetness(&self, x: f32, y: f32) -> f32 {
        Self::map_sample(&self.flow, Vec2::new(x, y) + Vec2::splat(HALF_WORLD as f32))
            .0
            .clamp(0.0, 1.0)
    }

    pub fn density(&self, p: Vec3) -> f32 {
        self.density_over(p, self.height(p.x, p.y))
    }

    fn chunk(&self, coord: IVec3) -> Chunk {
        let origin = chunk_origin(coord);
        let mut samples = vec![0.0; Shape::USIZE];
        // One height lookup per column instead of per sample.
        for y in 0..N {
            for x in 0..N {
                let column = origin + Vec3::new(x as f32, y as f32, 0.0) * VOXEL;
                let height = self.height(column.x, column.y);
                for z in 0..N {
                    let p = origin + Vec3::new(x as f32, y as f32, z as f32) * VOXEL;
                    samples[Shape::linearize([x, y, z]) as usize] = self.density_over(p, height);
                }
            }
        }
        Chunk {
            samples: Arc::new(samples),
            revision: 0,
        }
    }

    fn density_over(&self, p: Vec3, height: f32) -> f32 {
        let base = p.z - height;
        // The detail weight below is exactly zero here; skipping the 3D noise is bit-identical.
        if base.abs() >= 3.0 {
            return base;
        }
        let detail =
            self.noise
                .get([p.x as f64 * 0.14, p.y as f64 * 0.14, p.z as f64 * 0.14]) as f32;
        // A signed scalar field, not an exact Euclidean SDF: never sphere-trace it unchecked.
        base + detail * 0.55 * (1.0 - base.abs() / 3.0).max(0.0)
    }
}

const SQRT_2: f32 = std::f32::consts::SQRT_2;

fn is_border(i: usize) -> bool {
    let (x, z) = (i % SIDE, i / SIDE);
    x == 0 || z == 0 || x == SIDE - 1 || z == SIDE - 1
}

/// Calls `visit(neighbour, is_diagonal)` for the cells around `i` on the map.
#[inline]
fn for_neighbours(i: usize, mut visit: impl FnMut(usize, bool)) {
    const AROUND: [(isize, isize); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];
    let inner = !is_border(i);
    let (x, y) = ((i % SIDE) as isize, (i / SIDE) as isize);
    for (dx, dy) in AROUND {
        let (nx, ny) = (x + dx, y + dy);
        if inner || ((0..SIDE as isize).contains(&nx) && (0..SIDE as isize).contains(&ny)) {
            visit(ny as usize * SIDE + nx as usize, dx != 0 && dy != 0);
        }
    }
}

/// Heap key for the priority flood: orders by height, then index for determinism.
fn flood_key(height: f32, i: usize) -> u64 {
    let bits = height.to_bits();
    let ordered = if bits >> 31 == 1 {
        !bits
    } else {
        bits | 1 << 31
    };
    (ordered as u64) << 32 | i as u64
}

fn smoothstep(lo: f64, hi: f64, x: f64) -> f64 {
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Fractal Brownian motion, roughly in [-1, 1].
fn fbm(noise: &Perlin, p: DVec2, octaves: u32) -> f64 {
    let (mut sum, mut amplitude, mut p) = (0.0, 0.5, p);
    for octave in 0..octaves {
        sum += noise.get([p.x, p.y, octave as f64 * 7.0]) * amplitude;
        amplitude *= 0.5;
        p *= 2.03;
    }
    sum
}

/// Ridged multifractal in [0, 1]: sharp crests, with detail concentrated on the ridges.
fn ridged(noise: &Perlin, p: DVec2, octaves: u32) -> f64 {
    let (mut sum, mut amplitude, mut weight, mut p) = (0.0, 0.6, 1.0, p);
    for octave in 0..octaves {
        let n = (1.0 - noise.get([p.x, p.y, 50.0 + octave as f64 * 7.0]).abs()).powi(2) * weight;
        weight = (n * 1.8).clamp(0.0, 1.0);
        sum += n * amplitude;
        amplitude *= 0.48;
        p *= 2.1;
    }
    sum
}

/// Pre-erosion elevation in metres at world position `p` (x, z), and rock hardness in [0, 1].
fn relief(noise: &Perlin, s: &Settings, p: DVec2) -> (f64, f64) {
    let (mountains, hills) = (s.mountains as f64, s.hills as f64);
    if s.landform == Landform::Valley {
        let warp = noise.get([p.x * 0.007, p.y * 0.007]) * 24.0;
        let mut ridges = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 0.008;
        for octave in 0..5 {
            let n = noise.get([
                (p.x + warp) * frequency,
                p.y * frequency,
                octave as f64 * 7.0,
            ]);
            ridges += (1.0 - n.abs()).powi(3) * amplitude;
            frequency *= 2.1;
            amplitude *= 0.48;
        }
        let river_x = 28.0 * (p.y * 0.016).sin() - 12.0;
        let valley = (-((p.x - river_x) / 39.0).powi(2)).exp();
        let mountain =
            (ridges * mountains + noise.get([p.x * 0.004, p.y * 0.004]) * hills + 20.0).max(10.0);
        return (mountain * (1.0 - valley * 0.83) + 5.0, 0.0);
    }
    // Domain warping bends coastlines and ranges away from the noise grid.
    let warp = DVec2::new(fbm(noise, p * 0.004, 3), fbm(noise, p * 0.004 + 91.7, 3)) * 45.0;
    let q = p + warp;
    let r = q.length() / HALF_WORLD as f64;
    // Signed "continentalness": below zero is sea floor, above zero is land.
    let continent = match s.landform {
        Landform::Island => 0.78 - r * 1.05 + fbm(noise, q * 0.009, 4) * 0.35,
        Landform::Archipelago => fbm(noise, q * 0.013, 4) * 1.3 + 0.12 - r * r * 0.35,
        Landform::Continent => {
            (q.x * 0.8 + q.y * 0.4) / HALF_WORLD as f64 * 0.7 + fbm(noise, q * 0.006, 4) * 0.5 + 0.1
        }
        Landform::Mountains => 0.75 + fbm(noise, q * 0.005, 3) * 0.3,
        Landform::Plains => 0.35 + fbm(noise, q * 0.005, 3) * 0.2,
        Landform::Valley => unreachable!(),
    };
    let base = s.sea_level.unwrap_or(0.0) as f64;
    let land = smoothstep(-0.06, 0.2, continent);
    let floor = -24.0 + fbm(noise, q * 0.02, 3) * 6.0;
    let rolling = (fbm(noise, q * 0.018, 5) * 0.5 + 0.5) * hills;
    // A broad concave dome carries most of the elevation and ridges only modulate it, so land
    // climbs gradually from the coast and peaks sit on massifs instead of rising as spikes.
    let dome = smoothstep(0.12, 0.95, continent);
    let ridges = ridged(noise, q * 0.0085, 5);
    let mut h = base
        + floor * (1.0 - land)
        + land * (1.2 + rolling)
        + mountains * dome * dome * (0.55 + 0.45 * ridges);
    // Hard-rock outcrops in noise-chosen patches step into benches and cliffs. Step height and
    // contour wobble vary, so escarpments look broken rather than like contour lines.
    let rock =
        smoothstep(0.14, 0.3, fbm(noise, q * 0.007 + 37.0, 3)) * smoothstep(1.0, 4.0, h - base);
    if rock > 0.0 {
        let step = 9.0 + 10.0 * (fbm(noise, q * 0.011 + 53.0, 2) * 0.5 + 0.5);
        let wobble = fbm(noise, q * 0.035 + 11.0, 2) * 3.0;
        let t = (h - base + wobble) / step;
        let terraced = base + (t.floor() + smoothstep(0.78, 1.0, t.fract())) * step - wobble;
        h += (terraced - h) * rock;
    }
    // Sea cliffs: along rocky stretches of coast the land rises abruptly at the shoreline.
    let mut hardness = rock;
    if s.sea_level.is_some() {
        let coast = smoothstep(0.0, 0.25, fbm(noise, q * 0.012 + 71.0, 3));
        let rise = smoothstep(0.1, 0.15, continent);
        h += coast * rise * (6.0 + 8.0 * (fbm(noise, q * 0.02 + 19.0, 2) * 0.5 + 0.5));
        hardness = hardness.max(coast * smoothstep(0.06, 0.12, continent));
    }
    (h, hardness)
}

#[derive(Clone)]
pub struct Chunk {
    pub samples: Arc<Vec<f32>>,
    pub revision: u64,
}

/// Clones are cheap, sharing every chunk's samples until one side edits them:
/// physics takes one as its ground, while the original goes on being dug.
#[derive(Clone)]
pub struct Terrain {
    pub landscape: Arc<Landscape>,
    pub chunks: HashMap<IVec3, Chunk>,
}

/// The chunk owning lattice point `p`.
fn chunk_of(p: IVec3) -> IVec3 {
    IVec3::new(
        p.x.div_euclid(CELLS),
        p.y.div_euclid(CELLS),
        p.z.div_euclid(CELLS),
    )
}

impl Ground for Terrain {
    /// The field over its slope: density falls off faster than distance does
    /// on steep ground, so this stays close to the true distance near the
    /// surface. The floor keeps a flat spot in the field from reading as far.
    fn distance(&self, point: Vec3) -> f32 {
        let p = point / VOXEL;
        let lo = p.floor();
        let f = p - lo;
        let c = self.corners(lo.as_ivec3());
        let z0 = c[0].lerp(c[1], f.x).lerp(c[2].lerp(c[3], f.x), f.y);
        let z1 = c[4].lerp(c[5], f.x).lerp(c[6].lerp(c[7], f.x), f.y);
        let dx = (c[1] - c[0])
            .lerp(c[3] - c[2], f.y)
            .lerp((c[5] - c[4]).lerp(c[7] - c[6], f.y), f.z);
        let dy = (c[2] - c[0])
            .lerp(c[3] - c[1], f.x)
            .lerp((c[6] - c[4]).lerp(c[7] - c[5], f.x), f.z);
        let gradient = Vec3::new(dx, dy, z1 - z0).length() / VOXEL;
        z0.lerp(z1, f.z) / gradient.max(0.5)
    }

    fn normal(&self, point: Vec3) -> Vec3 {
        Terrain::normal(self, point)
    }
}

pub fn chunk_origin(coord: IVec3) -> Vec3 {
    (coord * CELLS - IVec3::ONE).as_vec3() * VOXEL
}

pub fn valid_chunk(c: IVec3) -> bool {
    (-HALF_WORLD / CELLS..HALF_WORLD / CELLS).contains(&c.x)
        && (-HALF_WORLD / CELLS..HALF_WORLD / CELLS).contains(&c.y)
        && (MIN_Z / CELLS..MAX_Z / CELLS).contains(&c.z)
}

impl Terrain {
    pub fn new(landscape: Landscape) -> Self {
        Self {
            landscape: Arc::new(landscape),
            chunks: HashMap::new(),
        }
    }

    pub fn initial_coords(&self) -> Vec<IVec3> {
        let mut coords = Vec::new();
        for y in -HALF_WORLD / CELLS..HALF_WORLD / CELLS {
            for x in -HALF_WORLD / CELLS..HALF_WORLD / CELLS {
                let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                for dy in -1..=CELLS + 1 {
                    for dx in -1..=CELLS + 1 {
                        let h = self
                            .landscape
                            .height((x * CELLS + dx) as f32, (y * CELLS + dy) as f32);
                        lo = lo.min(h);
                        hi = hi.max(h);
                    }
                }
                for z in MIN_Z / CELLS..MAX_Z / CELLS {
                    if ((z * CELLS - 1) as f32) < hi + 2.0
                        && (((z + 1) * CELLS + 1) as f32) > lo - 2.0
                    {
                        coords.push(IVec3::new(x, y, z));
                    }
                }
            }
        }
        coords
    }

    pub fn ensure_chunk(&mut self, coord: IVec3) {
        let landscape = &self.landscape;
        self.chunks
            .entry(coord)
            .or_insert_with(|| landscape.chunk(coord));
    }

    /// [`Terrain::ensure_chunk`] for many chunks, in parallel.
    pub fn ensure_chunks(&mut self, coords: &[IVec3]) {
        let built: Vec<_> = coords
            .par_iter()
            .filter(|coord| !self.chunks.contains_key(coord))
            .map(|&coord| (coord, self.landscape.chunk(coord)))
            .collect();
        self.chunks.extend(built);
    }

    pub fn lattice(&self, p: IVec3) -> f32 {
        let coord = chunk_of(p);
        if let Some(chunk) = self.chunks.get(&coord) {
            let q = (p - coord * CELLS + IVec3::ONE).as_uvec3().to_array();
            chunk.samples[Shape::linearize(q) as usize]
        } else {
            self.landscape.density(p.as_vec3() * VOXEL)
        }
    }

    /// The eight lattice values of the cell at `lo`, x fastest. A chunk's halo
    /// holds every corner of its own cells, so this is one chunk lookup.
    fn corners(&self, lo: IVec3) -> [f32; 8] {
        let offset = |i: usize| IVec3::new(i as i32 & 1, i as i32 >> 1 & 1, i as i32 >> 2);
        let coord = chunk_of(lo);
        if let Some(chunk) = self.chunks.get(&coord) {
            let q = (lo - coord * CELLS + IVec3::ONE).as_uvec3();
            std::array::from_fn(|i| {
                let corner = (q + offset(i).as_uvec3()).to_array();
                chunk.samples[Shape::linearize(corner) as usize]
            })
        } else {
            std::array::from_fn(|i| self.landscape.density((lo + offset(i)).as_vec3() * VOXEL))
        }
    }

    pub fn sample(&self, world: Vec3) -> f32 {
        let p = world / VOXEL;
        let lo = p.floor();
        let f = p - lo;
        let c = self.corners(lo.as_ivec3());
        let z0 = c[0].lerp(c[1], f.x).lerp(c[2].lerp(c[3], f.x), f.y);
        let z1 = c[4].lerp(c[5], f.x).lerp(c[6].lerp(c[7], f.x), f.y);
        z0.lerp(z1, f.z)
    }

    pub fn normal(&self, p: Vec3) -> Vec3 {
        let h = VOXEL * 0.5;
        Vec3::new(
            self.sample(p + Vec3::X * h) - self.sample(p - Vec3::X * h),
            self.sample(p + Vec3::Y * h) - self.sample(p - Vec3::Y * h),
            self.sample(p + Vec3::Z * h) - self.sample(p - Vec3::Z * h),
        )
        .normalize_or(Vec3::Z)
    }

    /// The first point along the ray where the field changes sign, inside the
    /// editable volume. Walks the voxel cells the ray crosses: a trilinear cell
    /// cannot change sign unless its corners do, so skipped cells are exact.
    pub fn raycast(&self, origin: Vec3, direction: Vec3, reach: f32) -> Option<Vec3> {
        /// Sign checks along the ray inside a cell with mixed corners.
        const SPLITS: usize = 4;
        let direction = direction.try_normalize()?;
        if !origin.is_finite() || !reach.is_finite() {
            return None;
        }
        let at = |t: f32| origin + direction * t;
        let inside = self.sample(origin) < 0.0;
        let mut cell = (origin / VOXEL).floor().as_ivec3();
        let step = direction.signum().as_ivec3();
        let across = VOXEL / direction.abs();
        // Distance along the ray to the next cell boundary on each axis.
        let far_side = (direction.signum() + 1.0) * 0.5;
        let mut next = ((cell.as_vec3() + far_side) * VOXEL - origin) / direction;
        for axis in 0..3 {
            if direction[axis] == 0.0 {
                next[axis] = f32::INFINITY;
            }
        }
        let mut t = 0.0;
        while t < reach {
            let exit = next.min_element().min(reach);
            if self.corners(cell).iter().any(|v| (*v < 0.0) != inside) {
                let mut a = t;
                for split in 1..=SPLITS {
                    let b = t + (exit - t) * split as f32 / SPLITS as f32;
                    if (self.sample(at(b)) < 0.0) != inside {
                        let (mut a, mut b) = (a, b);
                        for _ in 0..12 {
                            let mid = (a + b) * 0.5;
                            if (self.sample(at(mid)) < 0.0) == inside {
                                a = mid;
                            } else {
                                b = mid;
                            }
                        }
                        let hit = at((a + b) * 0.5);
                        return (hit.x.abs() < HALF_WORLD as f32 - 2.0
                            && hit.y.abs() < HALF_WORLD as f32 - 2.0
                            && hit.z > MIN_Z as f32 + 2.0
                            && hit.z < MAX_Z as f32 - 2.0)
                            .then_some(hit);
                    }
                    a = b;
                }
            }
            t = exit;
            let axis = (0..3).fold(0, |a, b| if next[b] < next[a] { b } else { a });
            cell[axis] += step[axis];
            next[axis] += across[axis];
        }
        None
    }

    /// Update every duplicated halo sample, including corner neighbours, from the same brush.
    pub fn brush(&mut self, center: Vec3, radius: f32, add: bool) -> Vec<IVec3> {
        if !center.is_finite() || !radius.is_finite() || !(1.0..=12.0).contains(&radius) {
            return Vec::new();
        }
        let extent = Vec3::splat(radius + 2.0 * VOXEL);
        let lo = ((center - extent) / (CELLS as f32 * VOXEL))
            .floor()
            .as_ivec3();
        let hi = ((center + extent) / (CELLS as f32 * VOXEL))
            .floor()
            .as_ivec3();
        let mut dirty = Vec::new();
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let coord = IVec3::new(x, y, z);
                    if !valid_chunk(coord) {
                        continue;
                    }
                    self.ensure_chunk(coord);
                    let origin = chunk_origin(coord);
                    // Only visit the brush's bounding box inside this chunk.
                    let reach = Vec3::splat(radius + VOXEL);
                    let box_lo = ((center - reach - origin) / VOXEL).floor().max(Vec3::ZERO);
                    let box_hi = ((center + reach - origin) / VOXEL)
                        .ceil()
                        .min(Vec3::splat((N - 1) as f32));
                    if box_lo.cmpgt(box_hi).any() {
                        continue;
                    }
                    let (box_lo, box_hi) = (box_lo.as_uvec3(), box_hi.as_uvec3());
                    let chunk = self.chunks.get_mut(&coord).expect("just inserted");
                    let samples = Arc::make_mut(&mut chunk.samples);
                    let mut changed = false;
                    for z in box_lo.z..=box_hi.z {
                        for y in box_lo.y..=box_hi.y {
                            for x in box_lo.x..=box_hi.x {
                                let p = origin + Vec3::new(x as f32, y as f32, z as f32) * VOXEL;
                                let sphere = p.distance(center) - radius;
                                // Compact support keeps edits local and all halo duplicates bit-identical.
                                if sphere <= VOXEL {
                                    let d = &mut samples[Shape::linearize([x, y, z]) as usize];
                                    let new = if add { d.min(sphere) } else { d.max(-sphere) };
                                    if new != *d {
                                        *d = new;
                                        changed = true;
                                    }
                                }
                            }
                        }
                    }
                    if changed {
                        chunk.revision += 1;
                        dirty.push(coord);
                    }
                }
            }
        }
        dirty
    }
}

pub struct MeshData {
    pub positions: Vec<[f32; 3]>,
    /// Unit length.
    pub normals: Vec<[f32; 3]>,
    /// The original ground height over each vertex and its wetness: depth below
    /// that height exposes soil and rock in a dug wall.
    pub terrain_uv: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
    /// World-space minimum and maximum corners.
    pub bounds: (Vec3, Vec3),
}

pub fn polygonize(coord: IVec3, samples: &[f32], landscape: &Landscape) -> MeshData {
    thread_local! {
        static BUFFER: RefCell<SurfaceNetsBuffer> = RefCell::default();
    }
    BUFFER.with_borrow_mut(|buffer| {
        surface_nets(samples, &Shape {}, [0; 3], [N - 1; 3], buffer);
        let origin = chunk_origin(coord);
        let mut mesh = MeshData {
            positions: Vec::with_capacity(buffer.positions.len()),
            normals: Vec::with_capacity(buffer.positions.len()),
            terrain_uv: Vec::with_capacity(buffer.positions.len()),
            indices: buffer.indices.clone(),
            bounds: (Vec3::INFINITY, Vec3::NEG_INFINITY),
        };
        for (p, n) in buffer.positions.iter().zip(&buffer.normals) {
            let world = origin + Vec3::from_array(*p) * VOXEL;
            mesh.bounds = (mesh.bounds.0.min(world), mesh.bounds.1.max(world));
            mesh.positions.push(world.to_array());
            let normal = Vec3::from_array(*n).normalize_or(Vec3::Z);
            mesh.normals.push(normal.to_array());
            mesh.terrain_uv.push([
                landscape.height(world.x, world.y),
                landscape.wetness(world.x, world.y),
            ]);
        }
        mesh
    })
}

/// Meshes dirty chunks on worker threads, nearest first, and rejects any result
/// an edit has overtaken. Workers read an immutable snapshot of their chunk.
pub struct Mesher {
    pending: HashSet<IVec3>,
    running: HashMap<IVec3, u64>,
    sender: Sender<(IVec3, MeshData)>,
    receiver: Receiver<(IVec3, MeshData)>,
    /// Results dropped because their chunk changed while they were meshed.
    pub rejected: u64,
}

impl Default for Mesher {
    fn default() -> Self {
        let (sender, receiver) = channel();
        Self {
            pending: HashSet::new(),
            running: HashMap::new(),
            sender,
            receiver,
            rejected: 0,
        }
    }
}

impl Mesher {
    pub fn queue(&mut self, coords: impl IntoIterator<Item = IVec3>) {
        self.pending.extend(coords);
    }

    /// Chunks waiting or being meshed.
    pub fn outstanding(&self) -> usize {
        self.pending.len() + self.running.len()
    }

    /// Up to `budget` current meshes, then start more work near `eye`. An empty
    /// mesh means the chunk no longer has a surface.
    pub fn update(
        &mut self,
        terrain: &mut Terrain,
        eye: Vec3,
        budget: usize,
    ) -> Vec<(IVec3, MeshData)> {
        let mut ready = Vec::new();
        while ready.len() < budget {
            let Ok((coord, mesh)) = self.receiver.try_recv() else {
                break;
            };
            let revision = self.running.remove(&coord);
            if revision == Some(terrain.chunks[&coord].revision) {
                ready.push((coord, mesh));
            } else {
                self.rejected += 1;
                self.pending.insert(coord);
            }
        }
        let room = (rayon::current_num_threads() * 2).saturating_sub(self.running.len());
        // A chunk being meshed waits here until its result returns.
        let mut next: Vec<IVec3> = (self.pending.iter())
            .filter(|coord| !self.running.contains_key(coord))
            .copied()
            .collect();
        // shortcut: a full sort is fine for a finite world, use a heap for streaming ones.
        next.sort_by(|a, b| {
            let distance = |c: &IVec3| chunk_origin(*c).distance_squared(eye);
            distance(a).total_cmp(&distance(b))
        });
        next.truncate(room);
        terrain.ensure_chunks(&next);
        for coord in next {
            self.pending.remove(&coord);
            let chunk = &terrain.chunks[&coord];
            self.running.insert(coord, chunk.revision);
            let (samples, landscape) = (chunk.samples.clone(), terrain.landscape.clone());
            let sender = self.sender.clone();
            rayon::spawn(move || {
                // Only fails once the mesher is gone and nobody wants the result.
                let _ = sender.send((coord, polygonize(coord, &samples, &landscape)));
            });
        }
        ready
    }
}
