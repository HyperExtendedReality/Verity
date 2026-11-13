//! Eroded, smooth, editable voxel terrain, Z-up, in metres.
//!
//! [`Landscape::generate`] raises land with a [`Landform`]'s uplift and lets
//! rivers and landslides wear it down, coarse grid to fine ([`erosion`]), so
//! mountains are what erosion leaves standing. [`Terrain`] caches it as a
//! sparse 3D density field that can be sampled, ray cast and dug into, so
//! cavities and overhangs work. [`Mesher`] turns dirty chunks into
//! [`MeshData`] on worker threads, and drops results an edit has overtaken.
//! Density is negative inside the ground. [`Far`] draws the landscape beyond
//! the voxels as level-of-detail patches.
//!
//! Like World Creator, a landscape is a base shape plus a stack of [`Filter`]s
//! applied in order after erosion, and [`Settings::random`] rolls a whole
//! recipe. [`planet`] builds whole worlds with the same erosion.

mod erosion;
pub mod lod;
pub mod planet;

pub use lod::{Far, LodUpdate, Patch, PatchLod, Surface};

use erosion::{Grid, evolve, upsample};
use fast_surface_nets::{
    SurfaceNetsBuffer,
    ndshape::{ConstShape, ConstShape3u32},
    surface_nets,
};
use glam::{DVec2, FloatExt, IVec2, IVec3, Vec2, Vec3};
use noise::{NoiseFn, Perlin};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    io::{self, Read, Write},
    ops::Range,
    sync::{
        Arc,
        mpsc::{Receiver, Sender, channel},
    },
};
use verity_physics::Ground;

pub const CELLS: i32 = 24;
pub const VOXEL: f32 = 1.0;
pub type Shape = ConstShape3u32<26, 26, 26>;
const N: u32 = CELLS as u32 + 2;
/// The coarsest erosion grid's samples per side.
const COARSEST: usize = 65;

/// The large-scale shape of the world. Start from a preset with `Settings::from(Landform::Island)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
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

impl Landform {
    pub const ALL: [Landform; 6] = [
        Landform::Valley,
        Landform::Island,
        Landform::Archipelago,
        Landform::Continent,
        Landform::Mountains,
        Landform::Plains,
    ];
}

/// A height filter, applied in order after erosion, like World Creator's filter stack.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Filter {
    /// Steps the ground into benches `step` metres tall; `sharpness` in [0, 1] makes the risers abrupt.
    Terraces { step: f32, sharpness: f32 },
    /// Blurs the heights `passes` times, blended in by `strength` in [0, 1].
    Smooth { strength: f32, passes: u32 },
    /// Flattens ground above `height` metres into mesas, by `strength` in [0, 1].
    Plateau { height: f32, strength: f32 },
    /// Fills hollows toward the ground around them, by `strength` in [0, 1].
    Sediment { strength: f32 },
    /// Scales heights about the sea level (or zero) by `scale`, then raises them by `offset` metres.
    Level { scale: f32, offset: f32 },
}

impl Filter {
    /// One of each filter with useful values, to add to a stack.
    pub const PRESETS: [Filter; 5] = [
        Filter::Terraces {
            step: 20.0,
            sharpness: 0.6,
        },
        Filter::Smooth {
            strength: 0.5,
            passes: 2,
        },
        Filter::Plateau {
            height: 400.0,
            strength: 0.8,
        },
        Filter::Sediment { strength: 0.6 },
        Filter::Level {
            scale: 1.0,
            offset: 0.0,
        },
    ];

    pub fn name(&self) -> &'static str {
        match self {
            Filter::Terraces { .. } => "Terraces",
            Filter::Smooth { .. } => "Smooth",
            Filter::Plateau { .. } => "Plateau",
            Filter::Sediment { .. } => "Sediment",
            Filter::Level { .. } => "Level",
        }
    }
}

/// What to generate: a finite eroded landscape or a whole planet.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Generator {
    Landscape(Settings),
    Planet(planet::PlanetSettings),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub seed: u32,
    pub landform: Landform,
    /// Water height in metres. `None` means a dry world; droplets reaching the sea deposit their load.
    pub sea_level: Option<f32>,
    /// Height of the tallest peaks above the sea (or zero), in metres.
    pub mountains: f32,
    /// Rolling relief before erosion, in metres.
    pub hills: f32,
    /// Height samples along each side, rounded up to 2ᵏ + 1.
    pub size: usize,
    /// Metres between height samples.
    pub cell: f32,
    /// How long rivers have worked, in steps on the coarsest grid; each finer
    /// grid adds a third as many. 0 leaves the uplifted land unworn.
    pub fluvial_steps: usize,
    pub droplets: usize,
    pub thermal_steps: usize,
    pub erosion_rate: f32,
    pub deposition_rate: f32,
    pub evaporation: f32,
    /// Steepest slope soil holds, rise over run (0.75 is about 37°). Rock stands four times steeper.
    pub talus: f32,
    /// Applied in order after erosion.
    pub filters: Vec<Filter>,
}

impl Default for Settings {
    fn default() -> Self {
        Landform::Valley.into()
    }
}

impl From<Landform> for Settings {
    fn from(landform: Landform) -> Self {
        let (sea_level, mountains, hills) = match landform {
            Landform::Valley => (None, 450.0, 40.0),
            Landform::Island => (Some(0.0), 420.0, 30.0),
            Landform::Archipelago => (Some(0.0), 160.0, 20.0),
            Landform::Continent => (Some(-10.0), 650.0, 40.0),
            Landform::Mountains => (None, 800.0, 50.0),
            Landform::Plains => (None, 90.0, 30.0),
        };
        Self {
            seed: 1337,
            landform,
            sea_level,
            mountains,
            hills,
            size: 1025,
            cell: 4.0,
            fluvial_steps: 60,
            droplets: 4_000,
            thermal_steps: 25,
            erosion_rate: 0.24,
            deposition_rate: 0.18,
            evaporation: 0.025,
            talus: 0.75,
            filters: Vec::new(),
        }
    }
}

impl Settings {
    /// A whole random recipe from one number: landform, amplitudes, erosion and filters.
    pub fn random(seed: u32) -> Self {
        let mut rng = Random(seed as u64 ^ 0x9e37_79b9_7f4a_7c15);
        let landform = Landform::ALL[(rng.unit() * 6.0) as usize % 6];
        let mut s = Self::from(landform);
        s.seed = seed;
        s.mountains *= 0.6 + rng.unit() * 0.8;
        s.hills *= 0.6 + rng.unit() * 0.8;
        s.fluvial_steps = (20.0 + rng.unit() * 80.0) as usize;
        s.droplets = (1_000.0 + rng.unit() * 8_000.0) as usize;
        s.thermal_steps = (10.0 + rng.unit() * 30.0) as usize;
        s.talus = 0.55 + rng.unit() * 0.35;
        for preset in Filter::PRESETS {
            if rng.unit() < 0.3 {
                s.filters.push(match preset {
                    Filter::Terraces { .. } => Filter::Terraces {
                        step: 8.0 + rng.unit() * 30.0,
                        sharpness: rng.unit(),
                    },
                    Filter::Plateau { .. } => Filter::Plateau {
                        height: s.mountains * (0.4 + rng.unit() * 0.4),
                        strength: 0.5 + rng.unit() * 0.5,
                    },
                    other => other,
                });
            }
        }
        s
    }
}

/// A square heightmap, centred on the origin, with each sample's wetness.
#[derive(Clone)]
pub struct Landscape {
    /// Row-major, `side` × `side`.
    pub heights: Vec<f32>,
    pub flow: Vec<f32>,
    pub noise: Perlin,
    /// Samples along each side.
    pub side: usize,
    /// Metres between samples.
    pub cell: f32,
}

// Deterministic sampling without an additional random-number dependency.
pub(crate) struct Random(pub(crate) u64);
impl Random {
    pub(crate) fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 40) as f32) / ((1u32 << 24) as f32)
    }
}

/// The ground at one sample before erosion, and what erosion works with there.
struct Relief {
    height: f32,
    /// In [0, 1]: where the crust rises, and so where mountains stand.
    uplift: f32,
    /// In [0, 1]: where rock outcrops instead of soil.
    rock: f32,
    /// Thickness of the rock's hard and soft layers, in metres.
    strata: f32,
    /// Shifts the layers, as a fraction of `strata`, so they do not follow contour lines.
    phase: f32,
}

impl Relief {
    /// How hard the ground is at `height`: hard beds in soft ones, so cliffs and benches form.
    fn hardness(&self, height: f32) -> f32 {
        let bed = (height / self.strata + self.phase).rem_euclid(1.0);
        self.rock * smoothstep(0.55, 0.8, bed as f64) as f32
    }
}

impl Landscape {
    /// A level map at `height`, `side` samples `cell` metres apart.
    pub fn flat(height: f32, side: usize, cell: f32) -> Self {
        Self {
            heights: vec![height; side * side],
            flow: vec![0.0; side * side],
            noise: Perlin::new(1337),
            side,
            cell,
        }
    }

    /// Metres from the centre to each edge.
    pub fn half(&self) -> f32 {
        (self.side - 1) as f32 * self.cell * 0.5
    }

    /// Map coordinates, in samples, of world `(x, y)`.
    fn grid(&self, x: f32, y: f32) -> Vec2 {
        Vec2::new(x, y) / self.cell + (self.side - 1) as f32 * 0.5
    }

    pub fn generate(settings: Settings) -> Self {
        let s = &settings;
        let noise = Perlin::new(s.seed);
        let side = (s.size.max(COARSEST) - 1).next_power_of_two() + 1;
        let cell = s.cell.max(0.25);
        let half = (side - 1) as f64 * cell as f64 * 0.5;
        // The noise was laid out for a 384 m map: keep the layout at any size.
        let scale = 192.0 / half;
        let mut sides = vec![side];
        while *sides.last().expect("one side") > COARSEST {
            sides.push((sides.last().expect("one side") - 1) / 2 + 1);
        }
        sides.reverse();
        let sea = s.sea_level.unwrap_or(f32::NEG_INFINITY);
        let base = s.sea_level.unwrap_or(0.0);
        let mut rng = Random(s.seed as u64 + 7);
        // Metres per step at full uplift; rescaled below so the peaks reach `mountains`.
        let mut rate = s.mountains.max(1.0) * 0.012;
        let (mut heights, mut initial, mut area) = (Vec::new(), Vec::new(), Vec::new());
        let mut cells = Vec::new();
        for (level, &n) in sides.iter().enumerate() {
            let c = cell * ((side - 1) / (n - 1)) as f32;
            cells = (0..n * n)
                .into_par_iter()
                .map(|i| {
                    let p = DVec2::new((i % n) as f64, (i / n) as f64) * c as f64 - half;
                    relief(&noise, s, p * scale)
                })
                .collect::<Vec<Relief>>();
            let fresh: Vec<f32> = cells.iter().map(|r| r.height).collect();
            heights = if level == 0 {
                fresh.clone()
            } else {
                // The coarse result plus the detail the coarse grid could not hold.
                let coarse = sides[level - 1];
                let (worn, smooth) = (upsample(&heights, coarse), upsample(&initial, coarse));
                (worn.iter().zip(&smooth).zip(&fresh))
                    .map(|((w, s), f)| w + f - s)
                    .collect()
            };
            initial = fresh;
            let steps = if level == 0 {
                s.fluvial_steps
            } else {
                s.fluvial_steps.div_ceil(3)
            };
            let hard = |i: usize, h: f32| cells[i].hardness(h);
            let grid = Grid { side: n, cell: c };
            let start = heights.clone();
            // Steady-state river relief is proportional to uplift: the coarsest grid runs
            // rivers alone once to measure it, then again with the uplift that brings it
            // to `mountains`. Landslides only ever lower that.
            let measure = level == 0 && steps > 0;
            for pass in 0..1 + measure as usize {
                if measure && pass == 1 {
                    let peak = heights.iter().fold(0.0f32, |m, h| m.max(h - base));
                    // River relief grows with ln(distance / cell): finer grids add some,
                    // and uplift that ignored it would leave every slope at the talus angle.
                    let finer = ((half / cell as f64).ln() / (half / c as f64).ln()) as f32;
                    rate *= (s.mountains.max(0.0) / peak.max(1.0e-3) / finer).clamp(0.1, 10.0);
                    heights.copy_from_slice(&start);
                }
                let talus = if measure && pass == 0 {
                    f32::INFINITY
                } else {
                    s.talus
                };
                let uplift: Vec<f32> = cells.iter().map(|r| r.uplift * rate).collect();
                area = evolve(
                    &grid,
                    &mut heights,
                    &uplift,
                    sea,
                    talus,
                    &hard,
                    steps,
                    &mut rng,
                );
            }
        }
        // Finer grids add relief the coarsest could not hold: bring the peaks back down.
        // Only ever down, which keeps every slope under the talus angle.
        let peak = heights.iter().fold(0.0f32, |m, h| m.max(h - base));
        if peak > s.mountains.max(1.0) {
            let k = s.mountains.max(1.0) / peak;
            (heights.iter_mut().filter(|h| **h > base)).for_each(|h| *h = base + (*h - base) * k);
        }
        let mut landscape = Self {
            heights,
            // Drainage area seeds the wetness map so valley floors follow the river network.
            flow: area.iter().map(|a| a / (cell * cell) * 0.05).collect(),
            noise,
            side,
            cell,
        };
        let hard = |i: usize, h: f32| cells[i].hardness(h);
        landscape.erode(s, &hard);
        landscape.filter(s);
        landscape
    }

    fn filter(&mut self, s: &Settings) {
        let base = s.sea_level.unwrap_or(0.0);
        for filter in &s.filters {
            match *filter {
                Filter::Terraces { step, sharpness } if step > 0.0 => {
                    let lo = sharpness.clamp(0.0, 0.98) as f64;
                    for h in &mut self.heights {
                        let t = ((*h - base) / step) as f64;
                        *h = base + ((t.floor() + smoothstep(lo, 1.0, t.fract())) as f32) * step;
                    }
                }
                Filter::Terraces { .. } => {}
                Filter::Smooth { strength, passes } => {
                    let blurred = blur(&self.heights, self.side, passes.min(64));
                    for (h, b) in self.heights.iter_mut().zip(blurred) {
                        *h = h.lerp(b, strength.clamp(0.0, 1.0));
                    }
                }
                Filter::Plateau { height, strength } => {
                    for h in self.heights.iter_mut().filter(|h| **h > height) {
                        *h = h.lerp(height + (*h - height) * 0.1, strength.clamp(0.0, 1.0));
                    }
                }
                Filter::Sediment { strength } => {
                    let around = blur(&self.heights, self.side, 6);
                    for (h, a) in self.heights.iter_mut().zip(around) {
                        *h = h.max(h.lerp(a, strength.clamp(0.0, 1.0)));
                    }
                }
                Filter::Level { scale, offset } => {
                    for h in &mut self.heights {
                        *h = base + (*h - base) * scale + offset;
                    }
                }
            }
        }
    }

    /// Bilinear height and gradient (metres per sample) at map coordinates `p`.
    fn map_sample(map: &[f32], side: usize, p: Vec2) -> (f32, Vec2) {
        let p = p.clamp(Vec2::ZERO, Vec2::splat((side - 2) as f32));
        let [x, z] = p.floor().as_uvec2().to_array();
        let f = p.fract();
        let i = z as usize * side + x as usize;
        let [a, b, c, d] = [map[i], map[i + 1], map[i + side], map[i + side + 1]];
        let height = a.lerp(b, f.x).lerp(c.lerp(d, f.x), f.y);
        let gradient = Vec2::new((b - a).lerp(d - c, f.y), (c - a).lerp(d - b, f.x));
        (height, gradient)
    }

    fn deposit(&mut self, p: Vec2, sediment: f32) {
        let side = self.side;
        let [x, z] = p.floor().as_uvec2().to_array();
        let f = p.fract();
        let i = z as usize * side + x as usize;
        for (offset, weight) in [
            (0, (1.0 - f.x) * (1.0 - f.y)),
            (1, f.x * (1.0 - f.y)),
            (side, (1.0 - f.x) * f.y),
            (side + 1, f.x * f.y),
        ] {
            self.heights[i + offset] += sediment * weight;
        }
    }

    /// Droplets carve gullies and drop fans, then scree slumps to the talus angle.
    fn erode(&mut self, s: &Settings, hard: &(impl Fn(usize, f32) -> f32 + Sync)) {
        let (side, cell) = (self.side, self.cell);
        let mut rng = Random(s.seed as u64 + 1);
        let sea = s.sea_level.unwrap_or(f32::NEG_INFINITY);
        let mut kernel = [0.0f32; 25];
        for (i, w) in kernel.iter_mut().enumerate() {
            *w = (2.5 - Vec2::new((i % 5) as f32 - 2.0, (i / 5) as f32 - 2.0).length()).max(0.0);
        }
        let total: f32 = kernel.iter().sum();
        kernel.iter_mut().for_each(|w| *w /= total);
        // The recipe's count was for a 385² map; keep the density on any size.
        let droplets = s.droplets * (side * side) / (385 * 385);
        // shortcut: sequential droplets avoid write races, move transport to the GPU for larger maps.
        for _ in 0..droplets {
            let mut p = Vec2::new(rng.unit(), rng.unit()) * (side - 5) as f32 + Vec2::splat(2.0);
            let mut direction = Vec2::ZERO;
            let (mut speed, mut water, mut sediment) = (1.0f32, 1.0f32, 0.0f32);
            for _ in 0..70 {
                let (height, gradient) = Self::map_sample(&self.heights, side, p);
                if height < sea {
                    break; // Rivers drop their load where they meet the sea, building deltas.
                }
                direction = (direction * 0.35 - gradient * 0.65).normalize_or_zero();
                if direction == Vec2::ZERO {
                    break;
                }
                let next = p + direction;
                if next.min_element() < 2.0 || next.max_element() >= (side - 3) as f32 {
                    break;
                }
                let dh = Self::map_sample(&self.heights, side, next).0 - height;
                let slope = dh / cell;
                let capacity = (-slope).max(0.02) * speed * water * 3.2;
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
                        let i = (corner.y as usize + k / 5) * side + corner.x as usize + k % 5;
                        self.heights[i] -= amount * weight;
                    }
                    sediment += amount;
                }
                self.flow[p.y as usize * side + p.x as usize] += water;
                speed = (speed * speed - slope * 4.0).max(0.01).sqrt();
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
                if is_border(side, i) {
                    return;
                }
                let lowest = [i - 1, i + 1, i - side, i + side]
                    .into_iter()
                    .min_by(|a, b| heights[*a].total_cmp(&heights[*b]))
                    .expect("four neighbours");
                // Hard rock holds near-vertical cliffs; soil relaxes to the talus angle.
                let talus = s.talus * cell * (1.0 + 3.0 * hard(i, heights[i]));
                *out = (
                    ((heights[i] - heights[lowest] - talus) * 0.2).max(0.0),
                    lowest,
                );
            });
            let shed = &shed;
            self.heights.par_iter_mut().enumerate().for_each(|(i, h)| {
                *h -= shed[i].0;
                for j in [i.wrapping_sub(1), i + 1, i.wrapping_sub(side), i + side] {
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

    /// Ground height at world `(x, y)`, smooth across samples (Catmull–Rom).
    pub fn height(&self, x: f32, y: f32) -> f32 {
        let side = self.side as isize;
        let p = self
            .grid(x, y)
            .clamp(Vec2::ZERO, Vec2::splat((side - 1) as f32));
        catmull_rom(
            |x, y| self.heights[(y.clamp(0, side - 1) * side + x.clamp(0, side - 1)) as usize],
            p,
        )
    }

    pub fn wetness(&self, x: f32, y: f32) -> f32 {
        Self::map_sample(&self.flow, self.side, self.grid(x, y))
            .0
            .clamp(0.0, 1.0)
    }

    /// Height at `(x, y)`, and how rough the ground is there in metres:
    /// steep dry rock is rough, wet valley floors are smooth.
    fn column(&self, x: f32, y: f32) -> (f32, f32) {
        let gradient = Self::map_sample(&self.heights, self.side, self.grid(x, y)).1;
        let steep = smoothstep(0.3, 1.0, (gradient.length() / self.cell) as f64) as f32;
        let rough = 0.35 + 0.9 * steep * (1.0 - self.wetness(x, y));
        (self.height(x, y), rough)
    }

    pub fn density(&self, p: Vec3) -> f32 {
        let (height, rough) = self.column(p.x, p.y);
        self.density_over(p, height, rough)
    }

    /// FNV-1a over the heights, which identifies the generated landscape.
    pub fn fingerprint(&self) -> u64 {
        (self.heights.iter()).fold(0xcbf2_9ce4_8422_2325, |hash, h| {
            (hash ^ h.to_bits() as u64).wrapping_mul(0x0100_0000_01b3)
        })
    }

    fn chunk(&self, coord: IVec3) -> Chunk {
        let origin = chunk_origin(coord);
        let mut samples = vec![0.0; Shape::USIZE];
        // One height lookup per column instead of per sample.
        for y in 0..N {
            for x in 0..N {
                let column = origin + Vec3::new(x as f32, y as f32, 0.0) * VOXEL;
                let (height, rough) = self.column(column.x, column.y);
                for z in 0..N {
                    let p = origin + Vec3::new(x as f32, y as f32, z as f32) * VOXEL;
                    samples[Shape::linearize([x, y, z]) as usize] =
                        self.density_over(p, height, rough);
                }
            }
        }
        Chunk {
            samples: Arc::new(samples),
            revision: 0,
        }
    }

    fn density_over(&self, p: Vec3, height: f32, rough: f32) -> f32 {
        let base = p.z - height;
        // The detail weight below is exactly zero here; skipping the 3D noise is bit-identical.
        if base.abs() >= 3.0 {
            return base;
        }
        let detail =
            self.noise
                .get([p.x as f64 * 0.14, p.y as f64 * 0.14, p.z as f64 * 0.14]) as f32;
        // A signed scalar field, not an exact Euclidean SDF: never sphere-trace it unchecked.
        base + detail * rough * (1.0 - base.abs() / 3.0).max(0.0)
    }
}

/// Catmull–Rom interpolation of a grid read through `at`, at grid coordinates `p`.
pub(crate) fn catmull_rom(at: impl Fn(isize, isize) -> f32, p: Vec2) -> f32 {
    let weights = |t: f32| {
        let (t2, t3) = (t * t, t * t * t);
        [
            (-t3 + 2.0 * t2 - t) * 0.5,
            (3.0 * t3 - 5.0 * t2 + 2.0) * 0.5,
            (-3.0 * t3 + 4.0 * t2 + t) * 0.5,
            (t3 - t2) * 0.5,
        ]
    };
    let lo = p.floor();
    let (wx, wy) = (weights(p.x - lo.x), weights(p.y - lo.y));
    let (x, y) = (lo.x as isize - 1, lo.y as isize - 1);
    let mut sum = 0.0;
    for (j, wy) in wy.iter().enumerate() {
        for (i, wx) in wx.iter().enumerate() {
            sum += at(x + i as isize, y + j as isize) * wx * wy;
        }
    }
    sum
}

pub(crate) fn is_border(side: usize, i: usize) -> bool {
    let (x, z) = (i % side, i / side);
    x == 0 || z == 0 || x == side - 1 || z == side - 1
}

/// Calls `visit(neighbour, is_diagonal)` for the cells around `i` on a `side`² map.
#[inline]
pub(crate) fn for_neighbours(side: usize, i: usize, mut visit: impl FnMut(usize, bool)) {
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
    let inner = !is_border(side, i);
    let (x, y) = ((i % side) as isize, (i / side) as isize);
    for (dx, dy) in AROUND {
        let (nx, ny) = (x + dx, y + dy);
        if inner || ((0..side as isize).contains(&nx) && (0..side as isize).contains(&ny)) {
            visit(ny as usize * side + nx as usize, dx != 0 && dy != 0);
        }
    }
}

/// `passes` 3×3 box blurs of a map; the border keeps its heights.
fn blur(map: &[f32], side: usize, passes: u32) -> Vec<f32> {
    let mut out = map.to_vec();
    for _ in 0..passes {
        let src = out.clone();
        out.par_iter_mut().enumerate().for_each(|(i, h)| {
            if !is_border(side, i) {
                let mut sum = src[i];
                for_neighbours(side, i, |j, _| sum += src[j]);
                *h = sum / 9.0;
            }
        });
    }
    out
}

pub(crate) fn smoothstep(lo: f64, hi: f64, x: f64) -> f64 {
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

/// The ground before erosion at `p` (x, y), laid out for a 384 m map: low land
/// with gentle noise to seed drainage, plus the uplift that will become
/// mountains. Uplift follows a few long ridged belts, so ranges form, not blobs.
fn relief(noise: &Perlin, s: &Settings, p: DVec2) -> Relief {
    let (mountains, hills) = (s.mountains.max(0.0) as f64, s.hills as f64);
    let base = s.sea_level.unwrap_or(0.0) as f64;
    let rolling = |q: DVec2| (fbm(noise, q * 0.018, 5) * 0.5 + 0.5) * hills;
    let (low, uplift, q) = if s.landform == Landform::Valley {
        let warp = noise.get([p.x * 0.007, p.y * 0.007]) * 24.0;
        let q = DVec2::new(p.x + warp, p.y);
        let river_x = 28.0 * (p.y * 0.016).sin() - 12.0;
        let valley = (-((p.x - river_x) / 39.0).powi(2)).exp();
        let ranges = ridged(noise, q * 0.008, 3);
        let uplift = (1.0 - valley * 0.9) * (0.45 + 0.55 * ranges);
        (base + 5.0 + rolling(q), uplift, q)
    } else {
        // Domain warping bends coastlines and ranges away from the noise grid.
        let warp = DVec2::new(fbm(noise, p * 0.004, 3), fbm(noise, p * 0.004 + 91.7, 3)) * 45.0;
        let q = p + warp;
        let r = q.length() / 192.0;
        // Signed "continentalness": below zero is sea floor, above zero is land.
        let continent = match s.landform {
            Landform::Island => 0.78 - r * 1.05 + fbm(noise, q * 0.009, 4) * 0.35,
            Landform::Archipelago => fbm(noise, q * 0.013, 4) * 1.3 + 0.12 - r * r * 0.35,
            Landform::Continent => {
                (q.x * 0.8 + q.y * 0.4) / 192.0 * 0.7 + fbm(noise, q * 0.006, 4) * 0.5 + 0.1
            }
            Landform::Mountains => 0.75 + fbm(noise, q * 0.005, 3) * 0.3,
            Landform::Plains => 0.35 + fbm(noise, q * 0.005, 3) * 0.2,
            Landform::Valley => unreachable!(),
        };
        let land = smoothstep(-0.06, 0.2, continent);
        let floor = -(10.0 + 0.12 * mountains) + fbm(noise, q * 0.02, 3) * 6.0;
        // Uplift climbs gradually from the coast, so ranges sit inland on massifs.
        let dome = smoothstep(0.12, 0.95, continent);
        let ranges = ridged(noise, q * 0.0085, 3);
        let uplift = dome * dome * (0.35 + 0.65 * ranges);
        let low = base + floor * (1.0 - land) + land * (1.2 + rolling(q));
        (low, uplift, q)
    };
    let height = low + mountains * uplift;
    Relief {
        height: height as f32,
        uplift: uplift as f32,
        rock: (smoothstep(0.1, 0.3, fbm(noise, q * 0.007 + 37.0, 3))
            * smoothstep(1.0, 4.0, height - base)) as f32,
        strata: (20.0 + 40.0 * (fbm(noise, q * 0.011 + 53.0, 2) * 0.5 + 0.5)) as f32,
        phase: (fbm(noise, q * 0.035 + 11.0, 2) * 0.3) as f32,
    }
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
    /// The editable volume's chunks: `lo` inclusive, `hi` exclusive.
    lo: IVec3,
    hi: IVec3,
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

const EDITS_MAGIC: &[u8; 4] = b"VTED";
const EDITS_VERSION: u32 = 1;

impl Terrain {
    /// The editable volume spans the landscape, from below its lowest ground to above its peaks.
    pub fn new(landscape: Landscape) -> Self {
        let size = CELLS as f32 * VOXEL;
        let columns = (landscape.half() / size).floor() as i32;
        let (min, max) = (landscape.heights.iter())
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), h| {
                (lo.min(*h), hi.max(*h))
            });
        Self {
            landscape: Arc::new(landscape),
            chunks: HashMap::new(),
            lo: IVec3::new(-columns, -columns, ((min - 32.0) / size).floor() as i32),
            hi: IVec3::new(columns, columns, ((max + 24.0) / size).ceil() as i32),
        }
    }

    /// Every chunk column in the world.
    fn columns(&self) -> impl Iterator<Item = IVec2> + use<> {
        let (lo, hi) = (self.lo, self.hi);
        (lo.y..hi.y).flat_map(move |y| (lo.x..hi.x).map(move |x| IVec2::new(x, y)))
    }

    pub fn valid_chunk(&self, c: IVec3) -> bool {
        c.cmpge(self.lo).all() && c.cmplt(self.hi).all()
    }

    /// Every chunk the original ground surface passes through.
    pub fn initial_coords(&self) -> Vec<IVec3> {
        self.columns()
            .flat_map(|c| self.column_span(c).map(move |z| c.extend(z)))
            .collect()
    }

    /// The chunk heights in column `c` that the original ground surface passes through.
    fn column_span(&self, c: IVec2) -> Range<i32> {
        let l = &self.landscape;
        let size = CELLS as f32 * VOXEL;
        let last = l.side as f32 - 1.0;
        let from = l.grid(c.x as f32 * size - VOXEL, c.y as f32 * size - VOXEL);
        let to = l.grid(
            (c.x + 1) as f32 * size + VOXEL,
            (c.y + 1) as f32 * size + VOXEL,
        );
        // The samples around the column bound its ground, give or take what interpolation adds.
        let from = (from.floor() - 1.0)
            .clamp(Vec2::ZERO, Vec2::splat(last))
            .as_uvec2();
        let to = (to.ceil() + 1.0)
            .clamp(Vec2::ZERO, Vec2::splat(last))
            .as_uvec2();
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for y in from.y..=to.y {
            for x in from.x..=to.x {
                let h = l.heights[y as usize * l.side + x as usize];
                lo = lo.min(h);
                hi = hi.max(h);
            }
        }
        // Catmull–Rom overshoot, and the surface detail of up to 1.25 m.
        let margin = 4.0 + (hi - lo) * 0.1;
        let mut span = (self.lo.z..self.hi.z).filter(|z| {
            ((z * CELLS - 1) as f32 * VOXEL) < hi + margin
                && (((z + 1) * CELLS + 1) as f32 * VOXEL) > lo - margin
        });
        match span.next() {
            Some(first) => first..span.next_back().unwrap_or(first) + 1,
            None => 0..0,
        }
    }

    /// Chunks dug or filled since generation, or loaded from a save.
    pub fn edited(&self) -> impl Iterator<Item = IVec3> + '_ {
        (self.chunks.iter())
            .filter(|(_, chunk)| chunk.revision > 0)
            .map(|(coord, _)| *coord)
    }

    /// Write every edited chunk. Edits only load onto the landscape they were made on.
    /// shortcut: raw samples, about 70 KB a chunk; compress them if saves grow large.
    pub fn save_edits(&self, mut out: impl Write) -> io::Result<()> {
        let mut coords: Vec<IVec3> = self.edited().collect();
        coords.sort_by_key(|c| c.to_array());
        out.write_all(EDITS_MAGIC)?;
        out.write_all(&EDITS_VERSION.to_le_bytes())?;
        out.write_all(&self.landscape.fingerprint().to_le_bytes())?;
        out.write_all(&(coords.len() as u32).to_le_bytes())?;
        let mut bytes = Vec::with_capacity(Shape::USIZE * 4);
        for coord in coords {
            bytes.clear();
            bytes.extend(coord.to_array().iter().flat_map(|c| c.to_le_bytes()));
            bytes.extend(
                self.chunks[&coord]
                    .samples
                    .iter()
                    .flat_map(|s| s.to_le_bytes()),
            );
            out.write_all(&bytes)?;
        }
        out.flush()
    }

    /// Replace this terrain's edits with saved ones: chunks edited since revert
    /// to the generated ground. Returns every chunk that changed, to remesh. On
    /// an error the terrain is left as it was.
    pub fn load_edits(&mut self, mut input: impl Read) -> io::Result<Vec<IVec3>> {
        let invalid = |message: &str| io::Error::new(io::ErrorKind::InvalidData, message);
        let mut word = [0u8; 4];
        input.read_exact(&mut word)?;
        if &word != EDITS_MAGIC {
            return Err(invalid("not a terrain edits file"));
        }
        input.read_exact(&mut word)?;
        if u32::from_le_bytes(word) != EDITS_VERSION {
            return Err(invalid("unsupported terrain edits version"));
        }
        let mut fingerprint = [0u8; 8];
        input.read_exact(&mut fingerprint)?;
        if u64::from_le_bytes(fingerprint) != self.landscape.fingerprint() {
            return Err(invalid("the edits were made on a different landscape"));
        }
        input.read_exact(&mut word)?;
        let count = u32::from_le_bytes(word) as usize;
        let extent = self.hi - self.lo;
        if count > (extent.x * extent.y * extent.z).max(0) as usize {
            return Err(invalid("too many chunks"));
        }
        let mut loaded = HashMap::with_capacity(count);
        let mut bytes = vec![0u8; Shape::USIZE * 4];
        for _ in 0..count {
            let mut coord = [0u8; 12];
            input.read_exact(&mut coord)?;
            let coord = IVec3::from_array(std::array::from_fn(|i| {
                i32::from_le_bytes(coord[i * 4..i * 4 + 4].try_into().expect("four bytes"))
            }));
            if !self.valid_chunk(coord) {
                return Err(invalid("chunk outside the world"));
            }
            input.read_exact(&mut bytes)?;
            let samples: Vec<f32> = (bytes.as_chunks::<4>().0.iter())
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            if samples.iter().any(|s| !s.is_finite()) {
                return Err(invalid("non-finite sample"));
            }
            if loaded.insert(coord, samples).is_some() {
                return Err(invalid("duplicate chunk"));
            }
        }
        let reverted: Vec<IVec3> = (self.edited())
            .filter(|coord| !loaded.contains_key(coord))
            .collect();
        // Generated ground needs no chunk: sampling falls back to the landscape,
        // and the mesher regenerates it, rejecting meshes of the edits.
        for coord in &reverted {
            self.chunks.remove(coord);
        }
        let mut changed = reverted;
        for (coord, samples) in loaded {
            // A new revision rejects meshes still being made of the old samples.
            let revision = self.chunks.get(&coord).map_or(1, |c| c.revision + 1);
            let samples = Arc::new(samples);
            self.chunks.insert(coord, Chunk { samples, revision });
            changed.push(coord);
        }
        Ok(changed)
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
                        let size = CELLS as f32 * VOXEL;
                        let (lo, hi) = (self.lo.as_vec3() * size, self.hi.as_vec3() * size);
                        return (hit.cmpgt(lo + 2.0).all() && hit.cmplt(hi - 2.0).all())
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
                    if !self.valid_chunk(coord) {
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
    /// Linear vertex colours; empty for voxel chunks, which the terrain material shades.
    pub colors: Vec<[f32; 3]>,
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
            colors: Vec::new(),
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

    /// Stop meshing these chunks: no result comes back for them unless queued again.
    pub fn cancel(&mut self, coords: &[IVec3]) {
        for coord in coords {
            self.pending.remove(coord);
            self.running.remove(coord);
        }
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
            let Some(revision) = self.running.remove(&coord) else {
                continue; // Cancelled.
            };
            // A chunk dropped since was regenerated or reverted: mesh it again.
            if terrain.chunks.get(&coord).map(|c| c.revision) == Some(revision) {
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

/// Keeps the chunks within `radius` of the eye meshed and drops the rest from
/// memory. Edited chunks stay in memory wherever they are, so their edits
/// survive and save; only their meshes go. Unedited ground needs no chunk:
/// sampling, ray casts and physics fall back to the identical landscape.
pub struct Streamer {
    /// Horizontal distance to a chunk column within which it is meshed. A
    /// column unloads a chunk's width further out, so the edge does not churn.
    pub radius: f32,
    /// The ground surface's chunk heights in each of the terrain's columns.
    spans: Vec<(IVec2, Range<i32>)>,
    resident: HashSet<IVec3>,
}

impl Streamer {
    pub fn new(terrain: &Terrain, radius: f32) -> Self {
        let spans = terrain.columns().collect::<Vec<_>>();
        let spans = (spans.into_par_iter())
            .map(|c| (c, terrain.column_span(c)))
            .collect();
        Self {
            radius,
            spans,
            resident: HashSet::new(),
        }
    }

    /// Chunks with a mesh, or one on its way.
    pub fn is_resident(&self, coord: IVec3) -> bool {
        self.resident.contains(&coord)
    }

    pub fn resident(&self) -> usize {
        self.resident.len()
    }

    /// Queue chunks that came within reach of `eye` and cancel those that
    /// left it. Returns the chunks whose meshes should go.
    pub fn update(&mut self, terrain: &mut Terrain, mesher: &mut Mesher, eye: Vec3) -> Vec<IVec3> {
        let distance = |c: IVec2| {
            let center = (c.as_vec2() + 0.5) * (CELLS as f32 * VOXEL);
            center.distance(eye.truncate())
        };
        let (near, keep) = (self.radius, self.radius + CELLS as f32 * VOXEL);
        let mut unloaded = Vec::new();
        self.resident.retain(|c| {
            let stay = distance(c.truncate()) <= keep;
            if !stay {
                unloaded.push(*c);
            }
            stay
        });
        mesher.cancel(&unloaded);
        let surface = (self.spans.iter())
            .filter(|(c, _)| distance(*c) <= near)
            .flat_map(|(c, span)| span.clone().map(|z| c.extend(z)));
        // Digging reaches chunks the original surface never passed through.
        let edited = terrain.edited().filter(|c| distance(c.truncate()) <= near);
        let loaded: Vec<IVec3> = (surface.chain(edited.collect::<Vec<_>>()))
            .filter(|c| self.resident.insert(*c))
            .collect();
        mesher.queue(loaded);
        // Running meshes keep their own copy of the samples.
        let resident = &self.resident;
        (terrain.chunks).retain(|c, chunk| chunk.revision > 0 || resident.contains(c));
        unloaded
    }
}
