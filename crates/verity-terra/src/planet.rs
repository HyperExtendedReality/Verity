//! Whole planets: continents raised by uplift and worn down by rivers and
//! landslides on a cube-sphere grid, coarse to fine, with the same erosion as
//! landscapes. They draw as a quadtree of patches per cube face ([`PlanetLod`])
//! that split near the eye, skip what lies beyond the horizon, and hang
//! skirts below their edges to hide cracks between levels of detail. The sea
//! is the sphere at [`PlanetSettings::radius`], which the ground below it is
//! clamped to and coloured as water. Z is up at the world origin, which sits
//! on the equator; the poles are on the Y axis.

pub use crate::lod::{LodUpdate, Patch};
use crate::{
    MeshData, Random, catmull_rom,
    erosion::{Graph, evolve},
    lod::{GRID, Surface, grid_mesh},
    smoothstep,
};
use glam::{DVec3, Vec2, Vec3};
use noise::{NoiseFn, Perlin};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Patches whose vertices are this far apart, in metres, do not split further.
const FINEST_SPACING: f64 = 0.5;
/// Cells along a cube face's edge on the coarsest erosion grid.
const COARSEST: usize = 16;
/// River steps on the coarsest grid, and on each finer one.
const STEPS: [usize; 2] = [60, 20];
/// Steepest soil slope, rise over run.
const TALUS: f32 = 0.75;

/// Keeps a planet's patches matched to the eye.
pub type PlanetLod = crate::lod::PatchLod<Planet>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PlanetSettings {
    pub seed: u32,
    /// Sea-level radius in metres. The centre sits this far below the origin.
    pub radius: f32,
    /// Height of the tallest peaks above sea level, in metres.
    pub mountains: f32,
    /// Rolling relief over the land before erosion, in metres.
    pub hills: f32,
    /// Ocean floor depth below sea level, in metres.
    pub ocean_depth: f32,
    /// Roughly the share of the surface under water, in [0, 1].
    pub ocean: f32,
    /// How many continents: larger values give more, smaller landmasses.
    pub continents: f32,
    /// Ice caps cover latitudes whose sine exceeds this, in [0, 1].
    pub ice: f32,
    /// Patches split while closer than this many patch widths: more detail draws more triangles.
    pub detail: f32,
    /// Erosion cells along each cube face's edge, rounded up to a power of two.
    /// Finer grids carve finer valleys and take longer to generate.
    pub resolution: usize,
}

impl Default for PlanetSettings {
    fn default() -> Self {
        Self {
            seed: 7,
            radius: 2_000.0,
            mountains: 140.0,
            hills: 18.0,
            ocean_depth: 90.0,
            ocean: 0.55,
            continents: 1.6,
            ice: 0.8,
            detail: 2.0,
            resolution: 256,
        }
    }
}

impl PlanetSettings {
    /// A random world of this size and detail from one number.
    pub fn random(self, seed: u32) -> Self {
        let mut rng = Random(seed as u64 ^ 0x5851_f42d_4c95_7f2d);
        Self {
            seed,
            mountains: self.radius * (0.02 + rng.unit() * 0.1),
            hills: self.radius * (0.003 + rng.unit() * 0.012),
            ocean_depth: self.radius * (0.02 + rng.unit() * 0.05),
            ocean: 0.25 + rng.unit() * 0.6,
            continents: 0.8 + rng.unit() * 2.5,
            ice: 0.6 + rng.unit() * 0.35,
            ..self
        }
    }
}

/// Each face's outward axis and two tangents, with `u × v = normal`. Triangles wind clockwise
/// seen from outside, like the voxel mesher's, which is what the renderer culls by.
const FACES: [[DVec3; 3]; 6] = [
    [DVec3::X, DVec3::Y, DVec3::Z],
    [DVec3::NEG_X, DVec3::Z, DVec3::Y],
    [DVec3::Y, DVec3::Z, DVec3::X],
    [DVec3::NEG_Y, DVec3::X, DVec3::Z],
    [DVec3::Z, DVec3::X, DVec3::Y],
    [DVec3::NEG_Z, DVec3::Y, DVec3::X],
];

impl Patch {
    /// The unit direction at `(s, t)` in [0, 1]² across this patch.
    fn direction(self, s: f64, t: f64) -> DVec3 {
        let [n, u, v] = FACES[self.face as usize];
        let p = self.on_face(s, t) * 2.0 - 1.0;
        let (a, b) = (p.x, p.y);
        // Spherified cube: spreads vertices far more evenly than normalising the cube point.
        let (a2, b2) = (a * a, b * b);
        let along = (1.0 - a2 / 2.0 - b2 / 2.0 + a2 * b2 / 3.0).sqrt();
        (n * along + u * a * (0.5 - b2 / 6.0).sqrt() + v * b * (0.5 - a2 / 6.0).sqrt()).normalize()
    }
}

/// An equal-angle cube-sphere grid: `n + 1` vertices along each face's edge,
/// with the vertices on shared edges and corners merged into one node.
struct Globe {
    n: usize,
    /// The node at each face's vertex `(i, j)`, face by face, rows of `n + 1`.
    face_node: Vec<u32>,
    dirs: Vec<DVec3>,
    /// Up to eight neighbours per node, `u32::MAX` past the last, with arc lengths in metres.
    around: Vec<[(u32, f32); 8]>,
    area: f32,
}

impl Globe {
    fn new(n: usize, radius: f64) -> Self {
        let side = n + 1;
        let angle =
            |i: usize| (std::f64::consts::FRAC_PI_4 * (2.0 * i as f64 / n as f64 - 1.0)).tan();
        let mut index = HashMap::new();
        let mut dirs = Vec::new();
        let mut face_node = Vec::with_capacity(6 * side * side);
        for [normal, u, v] in FACES {
            for j in 0..side {
                for i in 0..side {
                    // Integer cube coordinates are what shared vertices have in common.
                    let lattice = normal.as_ivec3() * n as i32
                        + u.as_ivec3() * (2 * i as i32 - n as i32)
                        + v.as_ivec3() * (2 * j as i32 - n as i32);
                    let node = *index.entry(lattice).or_insert_with(|| {
                        dirs.push((normal + u * angle(i) + v * angle(j)).normalize());
                        dirs.len() as u32 - 1
                    });
                    face_node.push(node);
                }
            }
        }
        let mut around = vec![[(u32::MAX, 0.0f32); 8]; dirs.len()];
        for face in 0..6 {
            for j in 0..side {
                for i in 0..side {
                    let node = face_node[(face * side + j) * side + i] as usize;
                    for (di, dj) in [
                        (-1, -1),
                        (0, -1),
                        (1, -1),
                        (-1, 0),
                        (1, 0),
                        (-1, 1),
                        (0, 1),
                        (1, 1),
                    ] {
                        let (x, y) = (i as isize + di, j as isize + dj);
                        if !(0..side as isize).contains(&x) || !(0..side as isize).contains(&y) {
                            continue;
                        }
                        let other = face_node[(face * side + y as usize) * side + x as usize];
                        let list = &mut around[node];
                        if let Some(slot) = (list.iter_mut())
                            .find(|(j, _)| *j == other || *j == u32::MAX)
                            .filter(|(j, _)| *j == u32::MAX)
                        {
                            let arc = dirs[node].angle_between(dirs[other as usize]) * radius;
                            *slot = (other, arc as f32);
                        }
                    }
                }
            }
        }
        let area = (4.0 * std::f64::consts::PI * radius * radius / dirs.len() as f64) as f32;
        Self {
            n,
            face_node,
            dirs,
            around,
            area,
        }
    }

    /// The face holding `dir`, and its grid coordinates there in [0, n]².
    fn locate(&self, dir: DVec3) -> (usize, Vec2) {
        let face = (0..6)
            .max_by(|a, b| dir.dot(FACES[*a][0]).total_cmp(&dir.dot(FACES[*b][0])))
            .expect("six faces");
        let [normal, u, v] = FACES[face];
        let along = dir.dot(normal);
        let grid = |tangent: DVec3| {
            let a = (dir.dot(tangent) / along).atan() / std::f64::consts::FRAC_PI_4;
            ((a + 1.0) * 0.5 * self.n as f64).clamp(0.0, self.n as f64) as f32
        };
        (face, Vec2::new(grid(u), grid(v)))
    }

    /// `map` at `dir`, smooth across cells and continuous across faces.
    fn sample(&self, map: &[f32], dir: DVec3) -> f32 {
        let (face, p) = self.locate(dir);
        let side = self.n as isize + 1;
        catmull_rom(
            |x, y| {
                let (x, y) = (x.clamp(0, side - 1), y.clamp(0, side - 1));
                map[self.face_node[((face as isize * side + y) * side + x) as usize] as usize]
            },
            p,
        )
    }
}

impl Graph for Globe {
    fn len(&self) -> usize {
        self.dirs.len()
    }

    fn neighbours(&self, i: usize, mut visit: impl FnMut(usize, f32)) {
        for &(j, metres) in self.around[i].iter().take_while(|(j, _)| *j != u32::MAX) {
            visit(j as usize, metres);
        }
    }

    fn is_edge(&self, _: usize) -> bool {
        false
    }

    fn area(&self) -> f32 {
        self.area
    }
}

pub struct Planet {
    pub settings: PlanetSettings,
    noise: Perlin,
    max_level: u8,
    globe: Globe,
    /// Eroded height above sea level at each node.
    heights: Vec<f32>,
    /// Wetness in [0, 1] at each node.
    wet: Vec<f32>,
    /// How rough the ground is at each node, in [0, 1]: steep and dry is rough.
    rough: Vec<f32>,
    /// The highest ground, in metres above the sea.
    peak: f64,
}

/// Fractal noise in 3D, roughly in [-1, 1].
fn fbm(noise: &Perlin, p: DVec3, octaves: u32) -> f64 {
    let (mut sum, mut amplitude, mut p) = (0.0, 0.5, p);
    for octave in 0..octaves {
        sum += noise.get((p + octave as f64 * 17.3).to_array()) * amplitude;
        amplitude *= 0.5;
        p *= 2.03;
    }
    sum
}

/// Ridged multifractal in [0, 1].
fn ridged(noise: &Perlin, p: DVec3, octaves: u32) -> f64 {
    let (mut sum, mut amplitude, mut weight, mut p) = (0.0, 0.6, 1.0, p);
    for octave in 0..octaves {
        let n =
            (1.0 - noise.get((p + 50.0 + octave as f64 * 7.0).to_array()).abs()).powi(2) * weight;
        weight = (n * 1.8).clamp(0.0, 1.0);
        sum += n * amplitude;
        amplitude *= 0.48;
        p *= 2.1;
    }
    sum
}

impl Planet {
    /// Generates and erodes the whole planet: about a second at the default resolution.
    pub fn new(settings: PlanetSettings) -> Self {
        let radius = settings.radius.max(10.0) as f64;
        let spacing = 2.0 * radius / (GRID - 1) as f64;
        let max_level = (spacing / FINEST_SPACING).log2().ceil().clamp(0.0, 22.0) as u8;
        let noise = Perlin::new(settings.seed);
        let finest = settings.resolution.max(COARSEST).next_power_of_two();
        let mut rng = Random(settings.seed as u64 + 7);
        let mountains = settings.mountains.max(0.0);
        let mut rate = mountains.max(1.0) * 0.012;
        let mut level: Option<(Globe, Vec<f32>, Vec<f32>)> = None;
        let mut area = Vec::new();
        let mut n = COARSEST;
        let (globe, heights) = loop {
            let globe = Globe::new(n, radius);
            let fresh: Vec<(f32, f32)> = (globe.dirs.par_iter())
                .map(|dir| relief(&noise, &settings, *dir))
                .collect();
            let initial: Vec<f32> = fresh.iter().map(|(h, _)| *h).collect();
            let mut heights = match &level {
                None => initial.clone(),
                // The coarse result plus the detail the coarse grid could not hold.
                Some((coarse, worn, smooth)) => (globe.dirs.par_iter().zip(&initial))
                    .map(|(dir, f)| coarse.sample(worn, *dir) + f - coarse.sample(smooth, *dir))
                    .collect(),
            };
            let steps = STEPS[level.is_some() as usize];
            let start = heights.clone();
            // Steady-state river relief is proportional to uplift: the coarsest grid runs
            // rivers alone once to measure it, then again with the uplift that brings it
            // to `mountains`. Landslides only ever lower that.
            let measure = level.is_none();
            for pass in 0..1 + measure as usize {
                if measure && pass == 1 {
                    let peak = heights.iter().fold(0.0f32, |m, h| m.max(*h));
                    // River relief grows with ln(distance / cell): finer grids add some,
                    // and uplift that ignored it would leave every slope at the talus angle.
                    let finer = ((finest as f64).ln() / (n as f64).ln()) as f32;
                    rate *= (mountains / peak.max(1.0e-3) / finer).clamp(0.1, 10.0);
                    heights.copy_from_slice(&start);
                }
                let talus = if measure && pass == 0 {
                    f32::INFINITY
                } else {
                    TALUS
                };
                let uplift: Vec<f32> = fresh.iter().map(|(_, u)| u * rate).collect();
                area = evolve(
                    &globe,
                    &mut heights,
                    &uplift,
                    0.0,
                    talus,
                    &|_, _| 0.0,
                    steps,
                    &mut rng,
                );
            }
            if n >= finest {
                // Finer grids add relief the coarsest could not hold: bring the peaks back down.
                // Only ever down, which keeps every slope under the talus angle.
                let peak = heights.iter().fold(0.0f32, |m, h| m.max(*h));
                if peak > mountains.max(1.0) {
                    let k = mountains.max(1.0) / peak;
                    heights
                        .iter_mut()
                        .filter(|h| **h > 0.0)
                        .for_each(|h| *h *= k);
                }
                break (globe, heights);
            }
            level = Some((globe, heights, initial));
            n *= 2;
        };
        let wet: Vec<f32> = (area.iter())
            .map(|a| ((1.0 + a / globe.area * 0.05).ln() / 7.0).clamp(0.0, 1.0))
            .collect();
        let rough = (0..globe.len())
            .into_par_iter()
            .map(|i| {
                let mut steepest = 0.0f32;
                globe.neighbours(i, |j, metres| {
                    steepest = steepest.max((heights[i] - heights[j]).abs() / metres);
                });
                smoothstep(0.3, 1.0, steepest as f64) as f32 * (1.0 - wet[i])
            })
            .collect();
        let peak = heights.iter().fold(0.0f32, |m, h| m.max(*h)) as f64;
        let cell = radius * std::f64::consts::FRAC_PI_2 / globe.n as f64;
        Self {
            noise,
            settings,
            max_level,
            globe,
            heights,
            wet,
            rough,
            // Room for the detail noise on top.
            peak: peak + cell * 0.5,
        }
    }

    pub fn center(&self) -> Vec3 {
        Vec3::new(0.0, 0.0, -self.radius())
    }

    fn radius(&self) -> f32 {
        self.settings.radius.max(10.0)
    }

    /// Ground height above sea level in metres along the unit direction `dir`:
    /// the eroded grid, plus detail below its cells where the ground is rough.
    pub fn height(&self, dir: DVec3) -> f64 {
        let globe = &self.globe;
        let h = globe.sample(&self.heights, dir) as f64;
        let rough = globe.sample(&self.rough, dir).clamp(0.0, 1.0) as f64;
        // Detail at about two cells across, and finer.
        let cell = self.radius() as f64 * std::f64::consts::FRAC_PI_2 / globe.n as f64;
        let frequency = globe.n as f64 / std::f64::consts::PI;
        h + fbm(&self.noise, dir * frequency + 31.0, 6) * cell * 0.25 * (0.3 + rough)
    }

    fn wetness(&self, dir: DVec3) -> f32 {
        self.globe.sample(&self.wet, dir).clamp(0.0, 1.0)
    }

    /// Linear biome colour from height above sea level, slope (the normal's
    /// share along `dir`) and latitude.
    fn color(&self, dir: DVec3, height: f64, up: f64) -> [f32; 3] {
        let s = &self.settings;
        if height <= 0.0 {
            let depth = smoothstep(0.0, s.ocean_depth.max(1.0) as f64 * 0.6, -height) as f32;
            return mix([0.06, 0.24, 0.27], [0.008, 0.035, 0.08], depth);
        }
        let latitude = dir.y.abs();
        let moisture =
            (fbm(&self.noise, dir * 6.0 + 101.0, 3) as f32 * 0.5 + 0.5).max(self.wetness(dir));
        let altitude = (height / s.mountains.max(1.0) as f64) as f32;
        let grass = mix([0.13, 0.20, 0.05], [0.04, 0.09, 0.025], moisture);
        let desert = [0.52, 0.42, 0.26];
        // Deserts form where it is dry in the subtropics.
        let dry = smoothstep(0.55, 0.3, moisture as f64)
            * smoothstep(0.1, 0.3, latitude)
            * smoothstep(0.5, 0.35, latitude);
        let mut color = mix(grass, desert, dry as f32);
        color = mix(
            color,
            [0.62, 0.56, 0.40],
            smoothstep(4.0, 1.0, height) as f32,
        ); // Beach.
        color = mix(
            color,
            [0.25, 0.23, 0.21],
            smoothstep(0.3, 0.7, altitude as f64) as f32,
        );
        color = mix(color, [0.30, 0.28, 0.26], smoothstep(0.85, 0.7, up) as f32); // Cliffs.
        let cold = latitude + altitude as f64 * 0.35;
        let snow = smoothstep(s.ice as f64, s.ice as f64 + 0.05, cold) * smoothstep(0.6, 0.8, up);
        mix(color, [0.85, 0.87, 0.9], snow as f32)
    }

    fn width(&self, patch: Patch) -> f64 {
        2.0 * self.radius() as f64 * 0.5f64.powi(patch.level as i32)
    }
}

impl Surface for Planet {
    /// Each patch splits while closer than `detail` widths, unless it lies
    /// wholly beyond the horizon, where no detail shows.
    fn leaves(&self, eye: Vec3) -> Vec<Patch> {
        let radius = self.radius() as f64;
        let eye = (eye - self.center()).as_dvec3();
        // A point is out of sight once it is further than the horizon plus its own view over the sea.
        let height = eye.length();
        let horizon = (height * height - radius * radius).max(0.0).sqrt();
        let over = ((radius + self.peak).powi(2) - radius * radius).sqrt();
        let mut stack: Vec<Patch> = (0..6).map(Patch::root).collect();
        let mut leaves = Vec::new();
        while let Some(patch) = stack.pop() {
            let width = self.width(patch);
            let dir = patch.direction(0.5, 0.5);
            let center = dir * (radius + self.height(dir).max(0.0));
            let distance = (eye - center).length();
            let hidden = height > radius && distance - width * 0.75 > horizon + over;
            if patch.level < self.max_level
                && !hidden
                && distance < width * self.settings.detail.max(0.5) as f64
            {
                stack.extend(patch.children());
            } else {
                leaves.push(patch);
            }
        }
        leaves
    }

    /// The patch's mesh in world space, with vertex colours and a skirt.
    fn mesh(&self, patch: Patch) -> MeshData {
        let radius = self.radius() as f64;
        let center = self.center().as_dvec3();
        grid_mesh(
            |s, t| {
                let dir = patch.direction(s, t);
                let h = self.height(dir);
                let point = center + dir * (radius + h.max(0.0));
                (point, dir, [h as f32, self.wetness(dir)])
            },
            Some(&(|dir, h, up| self.color(dir, h, up)) as &dyn Fn(DVec3, f64, f64) -> [f32; 3]),
            self.width(patch) * 0.04 + 1.0,
        )
    }

    fn center(&self, patch: Patch) -> Vec3 {
        let dir = patch.direction(0.5, 0.5);
        (self.center().as_dvec3() + dir * self.radius() as f64).as_vec3()
    }
}

/// The ground before erosion along `dir`, and its uplift in [0, 1]. Uplift
/// follows long ridged belts inland, so mountain ranges form, not blobs.
fn relief(noise: &Perlin, s: &PlanetSettings, dir: DVec3) -> (f32, f32) {
    let p = dir * s.continents.max(0.05) as f64;
    // Domain warping bends coasts and ranges away from the noise lattice.
    let warp = DVec3::new(
        fbm(noise, p * 1.3 + 11.0, 3),
        fbm(noise, p * 1.3 + 37.0, 3),
        fbm(noise, p * 1.3 + 71.0, 3),
    );
    let q = p + warp * 0.35;
    let continent = fbm(noise, q, 5) + (0.5 - s.ocean.clamp(0.0, 1.0) as f64) * 0.6;
    if continent < 0.0 {
        let floor = -(s.ocean_depth as f64) * smoothstep(0.0, 0.3, -continent);
        return (floor as f32, 0.0);
    }
    let coast = smoothstep(0.0, 0.03, continent);
    let inland = smoothstep(0.0, 0.35, continent);
    let ranges = ridged(noise, q * 4.0 + 5.0, 3);
    let hills = fbm(noise, q * 12.0 + 3.0, 5) * 0.5 + 0.5;
    let uplift = inland * inland * (0.3 + 0.7 * ranges) * coast;
    let low = (2.0 + hills * s.hills as f64) * coast;
    (
        (low + s.mountains.max(0.0) as f64 * uplift) as f32,
        uplift as f32,
    )
}

impl Planet {
    /// Radial height of `p` over the drawn surface (sea included), in metres.
    fn above(&self, p: Vec3) -> f32 {
        let offset = (p - self.center()).as_dvec3();
        let dir = offset.normalize_or(DVec3::Z);
        (offset.length() - self.radius() as f64 - self.height(dir).max(0.0)) as f32
    }

    fn gradient(&self, p: Vec3) -> Vec3 {
        const H: f32 = 0.25;
        Vec3::new(
            self.above(p + Vec3::X * H) - self.above(p - Vec3::X * H),
            self.above(p + Vec3::Y * H) - self.above(p - Vec3::Y * H),
            self.above(p + Vec3::Z * H) - self.above(p - Vec3::Z * H),
        ) / (2.0 * H)
    }
}

impl verity_physics::Ground for Planet {
    /// Radial height over the slope: on steep ground the height above it
    /// overstates the distance, and the gradient's length corrects that.
    fn distance(&self, p: Vec3) -> f32 {
        self.above(p) / self.gradient(p).length().max(0.5)
    }

    fn normal(&self, p: Vec3) -> Vec3 {
        let up = (p - self.center()).normalize_or(Vec3::Z);
        self.gradient(p).normalize_or(up)
    }
}

fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t)
}
