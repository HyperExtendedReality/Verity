//! Water for Verity: deep and coastal ocean waves, breaking shore waves,
//! dispersive shallow-water flow, cheap character ripples, and buoyancy that
//! couples any of them to `verity-physics` bodies. Z is up, metres and
//! seconds throughout, and every model answers the same [`Water`] query.
//!
//! - [`Ocean`]: FFT ocean (Tessendorf 2001) from empirical spectra as
//!   surveyed by Horvath 2015 — JONSWAP (Hasselmann et al. 1973) with fetch,
//!   TMA finite-depth attenuation (Bouws et al. 1985), Donelan-Banner
//!   spreading with swell narrowing, finite-depth dispersion, several
//!   cascades with disjoint wavelength bands, choppy displacement, and
//!   Jacobian whitecaps that persist and fade.
//! - [`ShoreWaves`]: a swell refracted over a seabed by solving the eikonal
//!   equation |∇φ| = k(h) (geometric optics, so Snell's law falls out),
//!   shoaled by energy-flux conservation, depth-limited by the McCowan and
//!   Miche breaking criteria, and shaped as parametric orbits with
//!   a predefined, adjustable breaking profile. [`ShoreWaves::breaker_grid`]
//!   exposes the same field to the GPU renderer; CPU height queries approximate
//!   one branch when a crest overturns.
//! - [`ShallowWater`]: mass-conserving, wetting-and-drying shallow water on a
//!   staggered grid (Stelling & Duinmeijer 2003) with Manning friction, plus
//!   the dispersive surface-wave layer of Jeschke & Wojtan 2023
//!   ("Generalizing Shallow Water Simulations with Dispersive Surface
//!   Waves"): short waves are split off and propagated exactly by Airy
//!   theory, so they travel at their true depth-dependent speed instead of
//!   the shallow-water one. Nyquist-axis modes are retained but not propagated
//!   by the dispersive layer. Water and momentum modifiers, edge relaxation
//!   zones that generate waves from any other [`Water`] (ocean blending), and
//!   foam advected with the flow.
//! - [`Ripples`]: a small damped wave-equation grid that scrolls with a
//!   character, for wakes and splashes at almost no cost. It tracks vertical
//!   speed explicitly so varying frame durations preserve ripple momentum.
//!
//! Everything runs on CPU threads with fields laid out as the textures a GPU
//! port or the renderer uploads (displacement, slope, foam per texel).

mod fft;
mod ocean;
mod ripples;
mod shallow;
mod shore;

pub use ocean::{Cascade, Ocean, OceanSettings, Spectrum};
pub use ripples::{Ripples, RipplesSettings};
pub use shallow::{ShallowWater, ShallowWaterSettings};
pub use shore::{BreakerGrid, BreakerProfile, BreakerShape, LocalWave, ShoreSettings, ShoreWaves};

use glam::{Mat3, Quat, Vec2, Vec3};
use std::f32::consts::PI;
use std::ops::{Add, Mul};
use verity_core::Transform;
use verity_physics::{Collider, ExternalForce, PhysicsBody};

/// Standard gravity, m/s². Wave dispersion assumes gravity along −Z.
pub const GRAVITY: f32 = 9.81;

/// The water surface above a horizontal point.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WaterSample {
    /// World Z of the surface, m.
    pub height: f32,
    /// Velocity of the water at the surface, m/s.
    pub velocity: Vec3,
}

/// Anything that can report its surface: oceans, shores, simulations, ripples.
pub trait Water {
    fn sample(&self, point: Vec2) -> WaterSample;
}

impl<T: Water + ?Sized> Water for &T {
    fn sample(&self, point: Vec2) -> WaterSample {
        (**self).sample(point)
    }
}

/// Layered water: the first gives the surface, the second adds an offset on
/// top of it (such as [`Ripples`] over an [`Ocean`]).
impl<A: Water, B: Water> Water for (A, B) {
    fn sample(&self, point: Vec2) -> WaterSample {
        let (a, b) = (self.0.sample(point), self.1.sample(point));
        WaterSample {
            height: a.height + b.height,
            velocity: a.velocity + b.velocity,
        }
    }
}

/// Sea bed heightfield: `elevation[x + y * width]` is the bed Z at the center
/// of cell (x, y), whose lower corner is `origin + (x, y) * cell`.
#[derive(Debug, Clone, PartialEq)]
pub struct Seabed {
    pub width: usize,
    pub height: usize,
    pub cell: f32,
    pub origin: Vec2,
    pub elevation: Vec<f32>,
}

impl Seabed {
    pub fn flat(width: usize, height: usize, cell: f32, elevation: f32) -> Self {
        Self {
            width,
            height,
            cell,
            origin: Vec2::ZERO,
            elevation: vec![elevation; width * height],
        }
    }

    pub fn cell_center(&self, x: usize, y: usize) -> Vec2 {
        self.origin + (Vec2::new(x as f32, y as f32) + 0.5) * self.cell
    }

    fn validate(&self) {
        assert!(
            self.width > 0
                && self.height > 0
                && self.cell.is_finite()
                && self.cell > 0.0
                && self.elevation.len() == self.width * self.height
                && self.elevation.iter().all(|e| e.is_finite()),
            "seabed needs a positive size and cell, and one finite elevation per cell"
        );
    }
}

/// Angular frequency ω of a gravity wave of wavenumber `k` (rad/m) in water
/// `depth` deep: ω² = g k tanh(k h). `depth` may be infinite.
pub fn angular_frequency(k: f32, depth: f32) -> f32 {
    if k <= 0.0 {
        return 0.0;
    }
    (GRAVITY * k * (k * depth).min(20.0).tanh()).sqrt()
}

/// Speed at which wave energy travels, m/s.
pub fn group_velocity(k: f32, depth: f32) -> f32 {
    let kh = (k * depth).min(20.0);
    if k <= 0.0 || kh < 1e-4 {
        return (GRAVITY * depth).sqrt();
    }
    0.5 * angular_frequency(k, depth) / k * (1.0 + 2.0 * kh / (2.0 * kh).sinh())
}

/// Wavenumber k solving the dispersion relation for angular frequency
/// `omega` at `depth` (Fenton & McKee 1990 start, then Newton).
pub fn wavenumber(omega: f32, depth: f32) -> f32 {
    assert!(depth > 0.0, "depth must be positive");
    let k0 = omega * omega / GRAVITY;
    if k0 * depth > 10.0 {
        return k0;
    }
    let mut k = k0 / (k0 * depth).powf(0.75).tanh().powf(2.0 / 3.0);
    for _ in 0..3 {
        let t = (k * depth).tanh();
        let f = GRAVITY * k * t - omega * omega;
        let df = GRAVITY * (t + k * depth * (1.0 - t * t));
        k -= f / df;
    }
    k
}

/// Fluid properties for [`buoyancy`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fluid {
    /// kg/m³; 1025 for seawater, 1000 for fresh water.
    pub density: f32,
    /// Linear drag per submerged cubic metre, N·s/m⁴. Damps bobbing and
    /// lets currents carry bodies.
    pub drag: f32,
}

impl Default for Fluid {
    fn default() -> Self {
        Self {
            density: 1025.0,
            drag: 1000.0,
        }
    }
}

/// Buoyancy and water drag on a body, as a force and torque about its center
/// to add to the body's [`ExternalForce`] before each physics step. `gravity`
/// is the physics settings' gravity.
pub fn buoyancy(
    water: &impl Water,
    transform: &Transform,
    body: &PhysicsBody,
    fluid: &Fluid,
    gravity: Vec3,
) -> ExternalForce {
    let center = Vec3::from_array(transform.translation);
    let rotation = Quat::from_array(transform.rotation).normalize();
    let mut out = ExternalForce::default();
    let mut apply = |offset: Vec3, submerged: f32, water_velocity: Vec3| {
        let velocity = body.velocity.linear + body.velocity.angular.cross(offset);
        let force =
            (-gravity * fluid.density - (velocity - water_velocity) * fluid.drag) * submerged;
        out.force += force;
        out.torque += offset.cross(force);
    };
    match body.collider {
        Collider::Sphere { radius } => {
            // Exact cap volume under a surface flat across the sphere.
            let s = water.sample(center.truncate());
            let d = (s.height - (center.z - radius)).clamp(0.0, 2.0 * radius);
            apply(
                Vec3::ZERO,
                PI * d * d * (3.0 * radius - d) / 3.0,
                s.velocity,
            );
        }
        Collider::Cuboid { half_extents } => {
            // 3³ sub-boxes, each submerged by how far the surface covers its
            // world-space vertical extent.
            const N: usize = 3;
            let sub = half_extents / N as f32;
            let reach = Mat3::from_quat(rotation).row(2).abs().dot(sub);
            let volume = 8.0 * sub.x * sub.y * sub.z;
            for i in 0..N * N * N {
                let cell = Vec3::new((i % N) as f32, (i / N % N) as f32, (i / (N * N)) as f32);
                let offset = rotation * ((cell * 2.0 + 1.0 - N as f32) * sub);
                let s = water.sample((center + offset).truncate());
                let covered =
                    ((s.height - center.z - offset.z) / (2.0 * reach) + 0.5).clamp(0.0, 1.0);
                if covered > 0.0 {
                    apply(offset, covered * volume, s.velocity);
                }
            }
        }
    }
    out
}

/// Bilinear sample of a row-major grid at grid coordinates `q` (texel
/// centers on integers), wrapping or clamping at the edges.
pub(crate) fn bilinear<T>(grid: &[T], width: usize, q: Vec2, wrap: bool) -> T
where
    T: Copy + Add<Output = T> + Mul<f32, Output = T>,
{
    let rows = (grid.len() / width) as i64;
    let base = q.floor();
    let f = q - base;
    let at = |dx: i64, dy: i64| {
        let (x, y) = (base.x as i64 + dx, base.y as i64 + dy);
        let (x, y) = if wrap {
            (x.rem_euclid(width as i64), y.rem_euclid(rows))
        } else {
            (x.clamp(0, width as i64 - 1), y.clamp(0, rows - 1))
        };
        grid[x as usize + y as usize * width]
    };
    (at(0, 0) * (1.0 - f.x) + at(1, 0) * f.x) * (1.0 - f.y)
        + (at(0, 1) * (1.0 - f.x) + at(1, 1) * f.x) * f.y
}

/// Finds the rest position whose displaced position lands on `point`, since
/// choppy waves move surface points sideways.
pub(crate) fn undisplace(point: Vec2, displacement: impl Fn(Vec2) -> Vec3) -> Vec2 {
    let mut rest = point;
    for _ in 0..4 {
        rest = point - displacement(rest).truncate();
    }
    rest
}

/// SplitMix64 with Box-Muller normals: deterministic per seed, no dependency.
pub(crate) struct Rng(pub u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1].
    fn uniform(&mut self) -> f32 {
        ((self.next() >> 40) as f32 + 1.0) / (1u64 << 24) as f32
    }

    /// Two independent standard normals.
    pub fn gaussian2(&mut self) -> Vec2 {
        let r = (-2.0 * self.uniform().ln()).sqrt();
        let (s, c) = (std::f32::consts::TAU * self.uniform()).sin_cos();
        Vec2::new(r * c, r * s)
    }
}
