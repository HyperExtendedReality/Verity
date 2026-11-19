//! Holographic Radiance Cascades in 3D by slicing: a CPU prototype.
//!
//! A ray that lies in a plane meets only what is in that plane, so the sphere of
//! directions at a point is gathered page by page: planes through the vertical
//! axis at a few azimuths φ. A page cuts the volume into parallel 2D slices and
//! each slice runs 2D HRC ([Freeman, Sannikov and Margel 2025][hrc]) unchanged.
//! With α the in-plane angle above the horizontal, dω = |cos α| dα dφ, so each
//! cone counts by its |cos α| measure and each page by its share of φ. In-plane
//! light keeps HRC's exact penumbrae; only φ is sampled coarsely.
//!
//! It is a reference for a GPU version, not a [`crate::GiMethod`]: it gathers
//! light straight from emitters, without bounces.
//!
//! [hrc]: https://arxiv.org/abs/2505.02041

use glam::{IVec3, Vec2, Vec3};
use rayon::prelude::*;
use std::f32::consts::PI;

/// An emitting, absorbing medium. Voxel centres sit at integer positions.
pub struct Volume {
    pub size: IVec3,
    /// What a voxel thick enough to be opaque shows.
    pub emission: Vec<Vec3>,
    /// Extinction per voxel width.
    pub sigma: Vec<f32>,
}

impl Volume {
    pub fn new(size: IVec3) -> Self {
        let count = size.element_product() as usize;
        Self {
            size,
            emission: vec![Vec3::ZERO; count],
            sigma: vec![0.0; count],
        }
    }

    pub fn index(&self, p: IVec3) -> Option<usize> {
        let s = self.size;
        (p.cmpge(IVec3::ZERO).all() && p.cmplt(s).all())
            .then(|| (p.x + s.x * (p.y + s.y * p.z)) as usize)
    }

    /// Fill the voxels from `low` to `high` inclusive.
    pub fn fill(&mut self, low: IVec3, high: IVec3, emission: Vec3, sigma: f32) {
        for z in low.z..=high.z {
            for y in low.y..=high.y {
                for x in low.x..=high.x {
                    if let Some(i) = self.index(IVec3::new(x, y, z)) {
                        self.emission[i] = emission;
                        self.sigma[i] = sigma;
                    }
                }
            }
        }
    }

    fn cell(&self, p: IVec3) -> Cell {
        self.index(p).map_or(Cell::VACUUM, |i| Cell {
            emission: self.emission[i],
            sigma: self.sigma[i],
        })
    }
}

/// What arrives at a point, per colour: fluence ∫L dω and flux ∫L ω dω, with ω
/// pointing towards where the light comes from.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Gather {
    pub fluence: Vec3,
    pub flux: [Vec3; 3],
}

impl Gather {
    /// Irradiance on a surface facing `normal`, for light as linear in ω as
    /// fluence and flux describe.
    pub fn irradiance(&self, normal: Vec3) -> Vec3 {
        let flux = self.flux[0] * normal.x + self.flux[1] * normal.y + self.flux[2] * normal.z;
        self.fluence / 4.0 + flux / 2.0
    }

    fn from_moments(m: Moments) -> Self {
        Self {
            fluence: m[0],
            flux: [m[1], m[2], m[3]],
        }
    }
}

/// Fluence and the three flux axes, as HRC carries them.
type Moments = [Vec3; 4];
/// A cone's weight for each of [`Moments`].
type Measure = [f32; 4];

fn add(a: Moments, b: Moments) -> Moments {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

fn scale(a: Moments, s: f32) -> Moments {
    a.map(|v| v * s)
}

#[derive(Clone, Copy)]
struct Cell {
    emission: Vec3,
    sigma: f32,
}

impl Cell {
    const VACUUM: Cell = Cell {
        emission: Vec3::ZERO,
        sigma: 0.0,
    };
}

/// A segment's light, and how much of what lies past it gets through.
#[derive(Clone, Copy)]
struct Ray {
    radiance: Vec3,
    transmittance: f32,
}

impl Ray {
    const VACUUM: Ray = Ray {
        radiance: Vec3::ZERO,
        transmittance: 1.0,
    };

    fn through(cell: Cell, length: f32) -> Ray {
        let transmittance = (-cell.sigma * length).exp();
        Ray {
            radiance: cell.emission * (1.0 - transmittance),
            transmittance,
        }
    }

    fn then(self, far: Ray) -> Ray {
        Ray {
            radiance: self.radiance + self.transmittance * far.radiance,
            transmittance: self.transmittance * far.transmittance,
        }
    }

    fn mean(self, other: Ray) -> Ray {
        Ray {
            radiance: (self.radiance + other.radiance) * 0.5,
            transmittance: (self.transmittance + other.transmittance) * 0.5,
        }
    }

    /// The cone this ray borders: its own light over `measure`, then `far`.
    fn merge(self, measure: Measure, far: Moments) -> Moments {
        std::array::from_fn(|i| self.radiance * measure[i] + far[i] * self.transmittance)
    }
}

/// Smallest `n` with `2^n >= x`.
fn ceil_log2(x: usize) -> usize {
    x.max(1).next_power_of_two().trailing_zeros() as usize
}

/// Per-cascade storage over probe columns `x = i·2^n`, rows, and `slots` a probe.
struct Grid<T> {
    shift: usize,
    columns: usize,
    height: usize,
    slots: usize,
    data: Vec<T>,
}

impl<T: Copy> Grid<T> {
    fn new(shift: usize, width: usize, height: usize, slots: usize, fill: T) -> Self {
        let columns = width.div_ceil(1 << shift);
        Self {
            shift,
            columns,
            height,
            slots,
            data: vec![fill; columns * height * slots],
        }
    }

    /// Slot `s` of the probe at `(x, y)`, if there is one.
    fn at(&self, x: i64, y: i64, s: usize) -> Option<T> {
        let i = x >> self.shift;
        (x >= 0 && y >= 0 && (i as usize) < self.columns && (y as usize) < self.height)
            .then(|| self.data[(i as usize * self.height + y as usize) * self.slots + s])
    }
}

/// One frustum of 2D HRC: light heading towards −u, gathered from +u within a
/// grid step of 45°, on a `width`×`height` grid of `cells` (row-major in u)
/// a voxel apart. `measure[n][c]` weighs cone `c` of cascade `n`.
fn frustum(cells: &[Cell], width: usize, height: usize, measure: &[Vec<Measure>]) -> Vec<Moments> {
    let top = ceil_log2(width) + 1;
    let cell = |x: i64, y: i64| {
        if x < width as i64 && y >= 0 && y < height as i64 {
            cells[y as usize * width + x as usize]
        } else {
            Cell::VACUUM
        }
    };

    // Rays from each cascade-n probe to the next column, `m` rows across.
    let span = |n: usize| 1i64 << n;
    let mut rays: Vec<Grid<Ray>> = Vec::with_capacity(top + 1);
    let mut base = Grid::new(0, width, height, 3, Ray::VACUUM);
    for x in 0..width as i64 {
        for y in 0..height as i64 {
            for m in -1..=1i64 {
                let half = ((1 + m * m) as f32).sqrt() / 2.0;
                let ray =
                    Ray::through(cell(x, y), half).then(Ray::through(cell(x + 1, y + m), half));
                base.data[((x as usize) * height + y as usize) * 3 + (m + 1) as usize] = ray;
            }
        }
    }
    rays.push(base);
    for n in 0..top {
        let (step, reach) = (span(n), span(n + 1));
        let lower = &rays[n];
        let ray =
            |x: i64, y: i64, m: i64| lower.at(x, y, (m + step) as usize).unwrap_or(Ray::VACUUM);
        let mut grid = Grid::new(n + 1, width, height, (2 * reach + 1) as usize, Ray::VACUUM);
        for i in 0..grid.columns {
            let x = i as i64 * reach;
            for y in 0..height as i64 {
                for m in -reach..=reach {
                    let extended = if m % 2 == 0 {
                        ray(x, y, m / 2).then(ray(x + step, y + m / 2, m / 2))
                    } else {
                        // No probe sits on the midpoint: average the two paths beside it.
                        let (a, b) = ((m - 1) / 2, (m + 1) / 2);
                        let one = ray(x, y, a).then(ray(x + step, y + a, b));
                        one.mean(ray(x, y, b).then(ray(x + step, y + b, a)))
                    };
                    grid.data[(i * height + y as usize) * grid.slots + (m + reach) as usize] =
                        extended;
                }
            }
        }
        rays.push(grid);
    }
    let ray = |n: usize, x: i64, y: i64, m: i64| {
        rays[n]
            .at(x, y, (m + span(n)) as usize)
            .unwrap_or(Ray::VACUUM)
    };

    // Cones, from the top cascade, whose rays leave the grid, down to the base.
    let zero = [Vec3::ZERO; 4];
    let mut upper = Grid::new(top, width, height, 1 << top, zero);
    for i in 0..upper.columns {
        let x = i as i64 * span(top);
        for y in 0..height as i64 {
            for c in 0..upper.slots {
                let low = 2 * c as i64 - span(top);
                let cone = add(
                    ray(top, x, y, low + 2).merge(measure[top + 1][2 * c + 1], zero),
                    ray(top, x, y, low).merge(measure[top + 1][2 * c], zero),
                );
                upper.data[(i * height + y as usize) * upper.slots + c] = cone;
            }
        }
    }
    for n in (0..top).rev() {
        let step = span(n);
        let mut grid = Grid::new(n, width, height, 1 << n, zero);
        let far = |x: i64, y: i64, c: usize| upper.at(x, y, c).unwrap_or(zero);
        for i in 0..grid.columns {
            let x = i as i64 * step;
            for y in 0..height as i64 {
                for c in 0..grid.slots {
                    let low = 2 * c as i64 - step;
                    let mut cone = zero;
                    // The upper and lower half, each a cone of the next cascade.
                    for (child, edge) in [(2 * c + 1, low + 2), (2 * c, low)] {
                        let w = measure[n + 1][child];
                        let half = if i % 2 == 1 {
                            // Follow the cone's edge to the next cascade's column.
                            ray(n, x, y, edge).merge(w, far(x + step, y + edge, child))
                        } else {
                            let along = ray(n + 1, x, y, 2 * edge)
                                .merge(w, far(x + 2 * step, y + 2 * edge, child));
                            scale(add(far(x, y, child), along), 0.5)
                        };
                        cone = add(cone, half);
                    }
                    grid.data[(i * height + y as usize) * grid.slots + c] = cone;
                }
            }
        }
        upper = grid;
    }
    upper.data
}

/// How a frustum's grid lies in a slice: `swap` puts u along the slice's rows
/// (up), `flip` runs u backwards.
#[derive(Clone, Copy)]
struct Frustum {
    swap: bool,
    flip: bool,
}

const FRUSTA: [Frustum; 4] = [
    Frustum {
        swap: false,
        flip: false,
    },
    Frustum {
        swap: false,
        flip: true,
    },
    Frustum {
        swap: true,
        flip: false,
    },
    Frustum {
        swap: true,
        flip: true,
    },
];

impl Frustum {
    /// A step of `(u, v)` grid cells as slice cells `(along, up)`.
    fn to_slice(self, u: f32, v: f32) -> Vec2 {
        let u = if self.flip { -u } else { u };
        if self.swap {
            Vec2::new(v, u)
        } else {
            Vec2::new(u, v)
        }
    }

    /// The slice cell `(t, y)` under grid cell `(u, v)`.
    fn cell(self, u: usize, v: usize, width: usize, height: usize) -> (usize, usize) {
        match (self.swap, self.flip) {
            (false, false) => (u, v),
            (false, true) => (width - 1 - u, v),
            (true, false) => (v, u),
            (true, true) => (v, height - 1 - u),
        }
    }
}

/// Integral of `|cos α|` times `(1, cos α, sin α)` over an arc that
/// stays on one side of the vertical, in either direction.
fn arc_measure(a: Vec2, b: Vec2) -> Vec3 {
    let angle = a.perp_dot(b).atan2(a.dot(b));
    let orientation = if a.x + b.x >= 0.0 { 1.0 } else { -1.0 } * angle.signum();
    orientation
        * Vec3::new(
            b.y - a.y,
            0.5 * (angle + b.x * b.y - a.x * a.y),
            0.5 * (b.y * b.y - a.y * a.y),
        )
}

fn cone_measure(a: Vec2, b: Vec2) -> Vec3 {
    if a.x * b.x < 0.0 {
        let pole = Vec2::new(0.0, a.y.signum());
        arc_measure(a, pole) + arc_measure(pole, b)
    } else {
        arc_measure(a, b)
    }
}

/// Weights of every cone of cascades `0..levels` of a frustum, in a page whose
/// slices run towards `horizontal`.
fn measures(levels: usize, frustum: Frustum, horizontal: Vec3) -> Vec<Vec<Measure>> {
    (0..levels)
        .map(|n| {
            let s = (1i64 << n) as f32;
            (0..1usize << n)
                .map(|c| {
                    let low = 2.0 * c as f32 - s;
                    let a = frustum.to_slice(s, low).normalize();
                    let b = frustum.to_slice(s, low + 2.0).normalize();
                    let m = cone_measure(a, b);
                    let flux = horizontal * m.y + Vec3::Y * m.z;
                    [m.x, flux.x, flux.y, flux.z]
                })
                .collect()
        })
        .collect()
}

/// Gather the page at azimuth `phi` into every voxel, weighted by `weight`.
///
/// The page's slices are a grid turned by `phi` about the volume's middle
/// column, a voxel apart each way. Its cells take the voxel nearest them; each
/// voxel takes the cell nearest it.
fn page(volume: &Volume, phi: f32, weight: f32, out: &mut [Moments]) {
    let size = volume.size;
    let horizontal = Vec3::new(phi.cos(), 0.0, phi.sin());
    let across = Vec3::new(-phi.sin(), 0.0, phi.cos());
    let middle = (size / 2).with_y(0).as_vec3();
    let reach = (size.x.max(size.z) as f32 * std::f32::consts::FRAC_1_SQRT_2).ceil() as i32 + 1;
    let height = size.y as usize;
    let levels = ceil_log2((2 * reach as usize + 1).max(height)) + 3;
    let tables: Vec<_> = FRUSTA
        .iter()
        .map(|&f| measures(levels, f, horizontal))
        .collect();
    let inside = |p: Vec3| {
        p.x > -0.5 && p.z > -0.5 && p.x < size.x as f32 - 0.5 && p.z < size.z as f32 - 0.5
    };

    // Each slice keeps the run of cells over the volume, from `first` on.
    let slices: Vec<Option<(i32, Vec<Moments>, usize)>> = (-reach..=reach)
        .into_par_iter()
        .map(|s| {
            let at = |t: i32| middle + horizontal * t as f32 + across * s as f32;
            let run: Vec<i32> = (-reach..=reach).filter(|&t| inside(at(t))).collect();
            let (&first, width) = (run.first()?, run.len());
            let cells: Vec<Cell> = (0..height)
                .flat_map(|y| (0..width).map(move |t| (t, y)))
                .map(|(t, y)| {
                    volume.cell(
                        (at(first + t as i32) + Vec3::Y * y as f32)
                            .round()
                            .as_ivec3(),
                    )
                })
                .collect();
            let mut sum = vec![[Vec3::ZERO; 4]; width * height];
            for (f, table) in FRUSTA.iter().zip(&tables) {
                let (w, h) = if f.swap {
                    (height, width)
                } else {
                    (width, height)
                };
                let local: Vec<Cell> = (0..h)
                    .flat_map(|v| (0..w).map(move |u| (u, v)))
                    .map(|(u, v)| {
                        let (t, y) = f.cell(u, v, width, height);
                        cells[y * width + t]
                    })
                    .collect();
                // `frustum` lays its output out column by column.
                let result = frustum(&local, w, h, table);
                for u in 0..w {
                    for v in 0..h {
                        let (t, y) = f.cell(u, v, width, height);
                        sum[y * width + t] = add(sum[y * width + t], result[u * h + v]);
                    }
                }
            }
            Some((first, sum, width))
        })
        .collect();
    out.par_chunks_mut(size.x as usize)
        .enumerate()
        .for_each(|(row, out)| {
            let (y, z) = (row % height, (row / height) as i32);
            for (x, out) in out.iter_mut().enumerate() {
                let offset = Vec3::new(x as f32, 0.0, z as f32) - middle;
                let (t, s) = (
                    offset.dot(horizontal).round() as i32,
                    offset.dot(across).round() as i32,
                );
                let Some((first, sum, width)) = &slices[(s + reach) as usize] else {
                    continue;
                };
                if let Some(m) = usize::try_from(t - first)
                    .ok()
                    .filter(|&t| t < *width)
                    .map(|t| sum[y * width + t])
                {
                    *out = add(*out, scale(m, weight));
                }
            }
        });
}

/// What reaches every voxel's centre, gathered over `pages` azimuths spread
/// evenly over the half turn and turned by `turn` of the gap between them.
/// Changing `turn` from frame to frame and averaging hides the gaps.
pub fn gather(volume: &Volume, pages: usize, turn: f32) -> Vec<Gather> {
    let mut out = vec![[Vec3::ZERO; 4]; volume.sigma.len()];
    let gap = PI / pages as f32;
    for k in 0..pages {
        page(volume, (k as f32 + turn) * gap, gap, &mut out);
    }
    out.into_iter().map(Gather::from_moments).collect()
}

/// Brute-force gather at a voxel centre, to check [`gather`] by: `directions`
/// rays spread evenly over the sphere, marched in `step`-long steps through
/// whichever voxel is nearest.
pub fn reference(volume: &Volume, at: IVec3, directions: usize, step: f32) -> Gather {
    let golden = PI * (3.0 - 5f32.sqrt());
    let mut m = [Vec3::ZERO; 4];
    for i in 0..directions {
        let y = 1.0 - 2.0 * (i as f32 + 0.5) / directions as f32;
        let r = (1.0 - y * y).sqrt();
        let phi = i as f32 * golden;
        let dir = Vec3::new(r * phi.cos(), y, r * phi.sin());
        let ray = march(volume, at, dir, step);
        let w = 4.0 * PI / directions as f32;
        m = add(
            m,
            [
                ray.radiance * w,
                ray.radiance * (w * dir.x),
                ray.radiance * (w * dir.y),
                ray.radiance * (w * dir.z),
            ],
        );
    }
    Gather::from_moments(m)
}

/// The light along `dir` from a voxel centre, through the nearest voxel at each step.
fn march(volume: &Volume, at: IVec3, dir: Vec3, step: f32) -> Ray {
    let reach = volume.size.as_vec3().length() + 1.0;
    let mut ray = Ray::VACUUM;
    let mut s = 0.0;
    while s < reach && ray.transmittance > 1e-4 {
        let cell = volume.cell((at.as_vec3() + dir * (s + step / 2.0)).round().as_ivec3());
        ray = ray.then(Ray::through(cell, step));
        s += step;
    }
    ray
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cone_measure_is_exact_on_either_side_of_the_vertical() {
        let q = std::f32::consts::FRAC_1_SQRT_2;
        let right = Vec2::new(q, q);
        let left = Vec2::new(-q, q);
        let arc = cone_measure(right, left);
        let expected = Vec3::new(2.0 * (1.0 - q), 0.0, 0.5);
        assert!((arc - expected).abs().max_element() < 1e-6, "{arc:?}");
        assert!((cone_measure(left, right) - expected).abs().max_element() < 1e-6);
        let down = cone_measure(Vec2::new(q, -q), Vec2::new(-q, -q));
        let expected_down = expected * Vec3::new(1.0, 1.0, -1.0);
        assert!(
            (down - expected_down).abs().max_element() < 1e-6,
            "{down:?}"
        );
    }

    #[test]
    fn cone_measures_cover_the_sphere() {
        // Over a page's four frusta, |cos α| integrates to 4, and the flux to zero.
        for phi in [0.0, 0.4, 2.0] {
            let horizontal = Vec3::new(f32::cos(phi), 0.0, f32::sin(phi));
            for n in 0..5 {
                let mut m = [0.0; 4];
                for f in FRUSTA {
                    for cone in &measures(n + 1, f, horizontal)[n] {
                        (0..4).for_each(|i| m[i] += cone[i]);
                    }
                }
                assert!((m[0] - 4.0).abs() < 1e-3, "{phi} level {n}: {m:?}");
                assert!(m[1..].iter().all(|v| v.abs() < 1e-3), "{m:?}");
            }
        }
    }

    #[test]
    fn a_glowing_fog_reaches_full_fluence_at_its_centre() {
        // Deep enough that every direction sees e: fluence 4π·e, no net flux.
        let mut volume = Volume::new(IVec3::splat(32));
        volume.fill(IVec3::ZERO, IVec3::splat(31), Vec3::ONE, 1.0);
        let g = gather(&volume, 8, 0.5)[volume.index(IVec3::splat(16)).unwrap()];
        assert!((g.fluence.x / (4.0 * PI) - 1.0).abs() < 0.02, "{g:?}");
        assert!(g.flux.iter().all(|f| f.x.abs() < 0.05), "{g:?}");
    }

    /// A light above a floor-wide slab with gaps at its ends, `n` voxels a side
    /// and `depth` deep.
    fn occluded(n: i32, depth: i32) -> Volume {
        let f = |v: i32| v * n / 16;
        let mut volume = Volume::new(IVec3::new(n, n, depth));
        volume.fill(
            IVec3::new(f(6), f(14), 0),
            IVec3::new(f(10) - 1, n - 1, depth),
            Vec3::splat(10.0),
            30.0,
        );
        volume.fill(
            IVec3::new(f(4), f(9), 0),
            IVec3::new(f(12) - 1, f(9), depth),
            Vec3::ZERO,
            30.0,
        );
        volume
    }

    /// Mean relative error of `hrc` against `expected` on every `stride`th voxel
    /// of the middle slice below the light.
    fn error(
        volume: &Volume,
        hrc: &[Gather],
        stride: usize,
        expected: impl Fn(IVec3) -> f32,
    ) -> f32 {
        let (n, z) = (volume.size.x, volume.size.z / 2);
        let (mut error, mut total) = (0.0, 0.0);
        for y in (0..n * 13 / 16).step_by(stride) {
            for x in (0..n).step_by(stride) {
                let p = IVec3::new(x, y, z);
                let e = expected(p);
                error += (hrc[volume.index(p).unwrap()].fluence.x - e).abs();
                total += e;
            }
        }
        error / total
    }

    #[test]
    fn one_page_matches_its_plane() {
        // A one-voxel-deep volume and the page along it: 2D HRC alone, against
        // ∫ L |cos α| dα over the plane, times the page's half turn.
        let volume = occluded(64, 1);
        let hrc = gather(&volume, 1, 0.0);
        let relative = error(&volume, &hrc, 8, |at| {
            let n = 2048;
            (0..n)
                .map(|i| {
                    let a = (i as f32 + 0.5) / n as f32 * 2.0 * PI;
                    let dir = Vec3::new(a.cos(), a.sin(), 0.0);
                    march(&volume, at, dir, 0.25).radiance.x * a.cos().abs()
                })
                .sum::<f32>()
                * 2.0
                * PI
                / n as f32
                * PI
        });
        // HRC converges to it: 10% at 32 voxels, 5% at 64, 2.6% at 128.
        assert!(relative < 0.06, "mean relative error {relative}");
    }

    #[test]
    fn matches_brute_force_around_an_occluder() {
        // What is left is mostly eight azimuths: 16 pages give 6% at 64 voxels.
        let volume = occluded(32, 32);
        let hrc = gather(&volume, 8, 0.5);
        let relative = error(&volume, &hrc, 2, |p| {
            reference(&volume, p, 1024, 0.25).fluence.x
        });
        assert!(relative < 0.12, "mean relative error {relative}");
    }
}
