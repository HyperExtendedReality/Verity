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
//!   shoaled and focused by solving energy-flux transport along the rays,
//!   so land casts a lee shadow; arriving in wave groups that travel at the
//!   group speed and move the break point; depth-limited by the McCowan and
//!   Miche breaking criteria, and shaped as parametric orbits with
//!   a predefined, adjustable breaking profile. [`ShoreWaves::breaker_grid`]
//!   exposes the same field to the GPU renderer; CPU height queries approximate
//!   one branch when a crest overturns.
//! - [`Coast`]: an [`Ocean`] running into [`ShoreWaves`], fading as the depth
//!   saturates (Thornton & Guza 1982), and a target for simulated beaches.
//! - [`ShallowWater`]: mass-conserving, wetting-and-drying shallow water on a
//!   staggered grid (Stelling & Duinmeijer 2003) with Manning friction, plus
//!   the dispersive surface-wave layer of Jeschke & Wojtan 2023
//!   ("Generalizing Shallow Water Simulations with Dispersive Surface
//!   Waves"): short waves are split off by a depth-aware filter, propagated
//!   exactly by Airy theory at their true depth-dependent speed, and carried
//!   by the bulk flow, so waves ride a river. Nyquist-axis modes are retained
//!   but not propagated by the dispersive layer. Discharge and open edges for
//!   rivers, water and momentum modifiers, edge relaxation zones that generate
//!   waves from any other [`Water`] (ocean blending), foam made by published
//!   breaking criteria and advected with the flow, and a wetness map of where
//!   water has been.
//! - [`Ripples`]: a small window that scrolls with a character, advanced
//!   exactly by the gravity-capillary dispersion relation (eWave), so wakes
//!   and splashes disperse as real ones do, at almost no cost.
//! - [`stir`]: the other direction, bodies making wakes and splashes in
//!   [`Ripples`] or [`ShallowWater`] from how they move through the water.
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
pub use shallow::{Edge, ShallowWater, ShallowWaterSettings};
pub use shore::{
    BreakerGrid, BreakerProfile, BreakerShape, Coast, LocalWave, ShoreSettings, ShoreWaves,
    WAVES_PER_GROUP, break_wave, group_envelope,
};

use glam::{Mat3, Quat, Vec2, Vec3};
use std::f32::consts::PI;
use std::ops::{Add, Mul};
use verity_core::Transform;
use verity_physics::{Collider, ExternalForce, PhysicsBody};

/// Cells per parallel task: below this, scheduling costs more than the work.
pub(crate) const GRAIN: usize = 2048;

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
    let mut out = ExternalForce::default();
    submerged(water, transform, &body.collider, |offset, volume, s| {
        let velocity = body.velocity.linear + body.velocity.angular.cross(offset);
        let force = (-gravity * fluid.density - (velocity - s.velocity) * fluid.drag) * volume;
        out.force += force;
        out.torque += offset.cross(force);
    });
    out
}

/// Water a body can stir: see [`stir`].
pub trait Stir {
    /// A body whose waterline spans `radius` m around `center`, reaching
    /// `draft` m below the surface on average, moving at `velocity` relative
    /// to the water for `dt` seconds.
    fn stir(&mut self, center: Vec2, radius: f32, draft: f32, velocity: Vec3, dt: f32);
}

/// A hull pushes a share of its draft aside for each footprint it moves, and
/// a falling body presses the surface down as it goes in.
impl Stir for Ripples {
    fn stir(&mut self, center: Vec2, radius: f32, draft: f32, velocity: Vec3, dt: f32) {
        let swept = velocity.truncate().length() * dt / (2.0 * radius);
        self.disturb(center, radius, draft * swept - velocity.z * dt);
    }
}

/// A hull drags the share of the water column it reaches along with it.
impl Stir for ShallowWater {
    fn stir(&mut self, center: Vec2, radius: f32, draft: f32, velocity: Vec3, _: f32) {
        let at = (center - self.origin()) / self.cell() - 0.5;
        let column = bilinear(&self.depth, self.resolution(), at, false);
        if column > 0.0 {
            let target = self.sample(center).velocity.truncate() + velocity.truncate();
            self.push(center, radius, target, (draft / column).min(1.0));
        }
    }
}

/// Wakes and splashes from a body moving through `water`, so they come from
/// physics rather than game code: call once per step next to [`buoyancy`],
/// with the same water, to disturb `target` (such as [`Ripples`] layered over
/// an [`Ocean`], or the [`ShallowWater`] it floats on).
pub fn stir(
    target: &mut impl Stir,
    water: &impl Water,
    transform: &Transform,
    body: &PhysicsBody,
    dt: f32,
) {
    let (mut volume, mut centroid, mut relative) = (0.0, Vec3::ZERO, Vec3::ZERO);
    submerged(water, transform, &body.collider, |offset, v, s| {
        let velocity = body.velocity.linear + body.velocity.angular.cross(offset);
        volume += v;
        centroid += offset * v;
        relative += (velocity - s.velocity) * v;
    });
    if volume <= 0.0 {
        return;
    }
    // shortcut: sizes the waterline as a hemisphere holding the submerged
    // volume; use the hull's real waterplane if long hulls need narrow wakes.
    let radius = (1.5 * volume / PI).cbrt();
    let center = Vec3::from_array(transform.translation) + centroid / volume;
    target.stir(
        center.truncate(),
        radius,
        2.0 * radius / 3.0,
        relative / volume,
        dt,
    );
}

/// Each submerged part of a collider: its offset from the body's center, the
/// volume under water, and the water above it.
fn submerged(
    water: &impl Water,
    transform: &Transform,
    collider: &Collider,
    mut part: impl FnMut(Vec3, f32, WaterSample),
) {
    let center = Vec3::from_array(transform.translation);
    let rotation = Quat::from_array(transform.rotation).normalize();
    let mut apply = |offset: Vec3, volume: f32, s: WaterSample| {
        if volume > 0.0 {
            part(offset, volume, s);
        }
    };
    match *collider {
        Collider::Sphere { radius } => {
            // Exact cap volume under a surface flat across the sphere.
            let s = water.sample(center.truncate());
            let d = (s.height - (center.z - radius)).clamp(0.0, 2.0 * radius);
            apply(Vec3::ZERO, PI * d * d * (3.0 * radius - d) / 3.0, s);
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
                apply(offset, covered * volume, s);
            }
        }
        Collider::Capsule {
            radius,
            half_height,
        } => {
            // Spheres along the core, each with its share of the volume.
            const N: usize = 4;
            let volume = PI * radius * radius * (2.0 * half_height + 4.0 / 3.0 * radius);
            let axis = rotation * Vec3::Z * half_height;
            for i in 0..N {
                let offset = axis * ((2 * i + 1) as f32 / N as f32 - 1.0);
                let s = water.sample((center + offset).truncate());
                let d = (s.height - (center.z + offset.z - radius)).clamp(0.0, 2.0 * radius);
                let share = d * d * (3.0 * radius - d) / (4.0 * radius.powi(3));
                apply(offset, share * volume / N as f32, s);
            }
        }
        // Ground is not something water lifts.
        Collider::Ground => {}
    }
}

/// Bilinear sample of a row-major grid at grid coordinates `q` (texel
/// centers on integers), wrapping or clamping at the edges.
pub(crate) fn bilinear<T>(grid: &[T], width: usize, q: Vec2, wrap: bool) -> T
where
    T: Copy + Add<Output = T> + Mul<f32, Output = T>,
{
    let (columns, rows) = (width as i64, (grid.len() / width) as i64);
    let base = q.floor();
    let f = q - base;
    let (x, y) = (base.x as i64, base.y as i64);
    let (x0, x1, y0, y1) = if wrap {
        let (x0, y0) = (x.rem_euclid(columns), y.rem_euclid(rows));
        let next = |i: i64, size: i64| if i + 1 == size { 0 } else { i + 1 };
        (x0, next(x0, columns), y0, next(y0, rows))
    } else {
        (
            x.clamp(0, columns - 1),
            (x + 1).clamp(0, columns - 1),
            y.clamp(0, rows - 1),
            (y + 1).clamp(0, rows - 1),
        )
    };
    let at = |x: i64, y: i64| grid[x as usize + y as usize * width];
    (at(x0, y0) * (1.0 - f.x) + at(x1, y0) * f.x) * (1.0 - f.y)
        + (at(x0, y1) * (1.0 - f.x) + at(x1, y1) * f.x) * f.y
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
