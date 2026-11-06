use crate::{
    GRAVITY, Ocean, Seabed, Water, WaterSample, bilinear, group_velocity, undisplace, wavenumber,
};
use glam::{Vec2, Vec3};
use rayon::prelude::*;
use std::f32::consts::{PI, TAU};

/// An incoming swell. Several trains are several [`ShoreWaves`], summed.
#[derive(Debug, Clone, PartialEq)]
pub struct ShoreSettings {
    /// Wave period, s.
    pub period: f32,
    /// Deep-water wave height, crest to trough, m.
    pub height: f32,
    /// Offshore travel direction, radians counterclockwise from +X.
    pub direction: f32,
    /// Waves break once taller than this times the depth (McCowan: 0.78).
    pub breaker_index: f32,
    /// Forward tilt of fully breaking crests, radians.
    pub lean: f32,
    /// Wave groups, in [0, 1]: how far sets rise above and lulls fall below
    /// `height`. 0 is an endless identical train. Sets travel at the group
    /// speed and vary along the crests, so the break point moves with them.
    pub groups: f32,
}

impl Default for ShoreSettings {
    fn default() -> Self {
        Self {
            period: 8.0,
            height: 1.0,
            direction: 0.0,
            breaker_index: 0.78,
            lean: 0.6,
            groups: 0.4,
        }
    }
}

/// The wave train's local state at a point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalWave {
    /// Horizontal travel direction, unit length.
    pub direction: Vec2,
    /// Crest-to-trough height, m.
    pub height: f32,
    /// rad/m.
    pub wavenumber: f32,
    /// 0 unbroken, 1 fully breaking: whitewater and foam.
    pub breaking: f32,
}

/// Dimensionless periodic breaker shape shared by the CPU and renderer.
///
/// The second harmonics are blended in with the local breaking factor `b`:
/// `along = -sin(phase) - horizontal_second_harmonic*b*sin(2*phase)` and
/// `up = cos(phase) + vertical_second_harmonic*b*cos(2*phase)`.
/// They sharpen the crest and push it forward; horizontal folds are intentional.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BreakerProfile {
    pub horizontal_second_harmonic: f32,
    pub vertical_second_harmonic: f32,
}

impl Default for BreakerProfile {
    fn default() -> Self {
        Self {
            horizontal_second_harmonic: 1.5,
            vertical_second_harmonic: 0.22,
        }
    }
}

/// Profile displacement and derivatives with respect to phase, not time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BreakerShape {
    pub along: f32,
    pub up: f32,
    pub along_derivative: f32,
    pub up_derivative: f32,
}

impl BreakerProfile {
    /// Evaluate at phase in radians and breaking strength in [0, 1].
    pub fn evaluate(&self, phase: f32, breaking: f32) -> BreakerShape {
        let breaking = breaking.clamp(0.0, 1.0);
        let (s, c) = phase.sin_cos();
        let (s2, c2) = (2.0 * phase).sin_cos();
        let a = self.horizontal_second_harmonic * breaking;
        let b = self.vertical_second_harmonic * breaking;
        BreakerShape {
            along: -s - a * s2,
            up: c + b * c2,
            along_derivative: -c - 2.0 * a * c2,
            up_derivative: -s - 2.0 * b * s2,
        }
    }
}

/// Waves in a typical set, so groups repeat this many times slower than waves.
pub const WAVES_PER_GROUP: f32 = 7.0;
/// Second group component, an irrational ratio so that sets never repeat.
const GROUP_RATIO: f32 = 1.618;
/// Crest length of a set, in offshore wavelengths.
const CRESTS_PER_GROUP: f32 = 8.0;

/// Wave group envelope. `lag` is seconds since the group front passed
/// (time − arrival), `across` the phase along the crest in radians.
pub fn group_envelope(groups: f32, omega: f32, lag: f32, across: f32) -> f32 {
    let t = omega / WAVES_PER_GROUP * lag;
    1.0 + groups * (0.6 * (t + across).sin() + 0.4 * (GROUP_RATIO * t - 0.7 * across).sin())
}

/// Height and breaking strength of a wave of `unbroken` height against the
/// depth-limited height `limit`.
pub fn break_wave(unbroken: f32, limit: f32) -> (f32, f32) {
    if limit <= 0.0 {
        return (0.0, 0.0);
    }
    (
        unbroken.min(limit),
        ((unbroken / limit - 0.9) / 0.3).clamp(0.0, 1.0),
    )
}

/// Borrowed, row-major cell-centered wave fields for rendering.
///
/// For a rest position `p`, interpolate each field at
/// `(p - origin) / cell - 0.5`, clamping at the grid edge, and normalize the
/// interpolated direction (falling back to +X if zero). Interpolated signed
/// depth <= 0 is dry and must be masked; otherwise clamp depth to at least
/// 0.05 m for the orbit. The phase is `phase - omega * time`. Multiply
/// `wave_height` by [`group_envelope`] of `groups`, `omega`,
/// `time - arrival` and `across.dot(p)`, and pass it with `limit` through
/// [`break_wave`] for the height and breaking strength. Evaluate the profile,
/// scale along by `min(height / (2*tanh(k*depth)), 0.9/k)` and up by
/// `height/2`, then rotate by `lean * breaking` as in
/// [`ShoreWaves::displacement_at_rest`]. Heights, signed depth and cell are
/// metres, omega is radians/second, and time and arrival are seconds.
#[derive(Debug, Clone, Copy)]
pub struct BreakerGrid<'a> {
    pub width: usize,
    pub height: usize,
    pub cell: f32,
    pub origin: Vec2,
    pub sea_level: f32,
    pub omega: f32,
    pub time: f32,
    pub lean: f32,
    pub groups: f32,
    /// Radians per metre along the offshore crests.
    pub across: Vec2,
    pub phase: &'a [f32],
    pub direction: &'a [Vec2],
    /// Height the wave would reach unbroken, before the group envelope.
    pub wave_height: &'a [f32],
    pub wavenumber: &'a [f32],
    pub depth: &'a [f32],
    /// Tallest wave the depth can hold.
    pub limit: &'a [f32],
    /// When the group front reaches the cell, s.
    pub arrival: &'a [f32],
}

/// Shallowest depth waves are traced through, so phase stays smooth across
/// the shoreline; amplitude there is zero anyway.
const MIN_DEPTH: f32 = 0.05;
/// Horizontal orbit steepness cap, below looping.
const MAX_STEEPNESS: f32 = 0.9;

/// A swell arriving from deep water, refracting, shoaling and breaking over a
/// seabed. Not simulated: precomputed once, then evaluated in closed form, so
/// it covers kilometres of coast for the cost of a few lookups.
pub struct ShoreWaves {
    settings: ShoreSettings,
    sea_level: f32,
    width: usize,
    cell: f32,
    origin: Vec2,
    omega: f32,
    across: Vec2,
    phase: Vec<f32>,
    direction: Vec<Vec2>,
    wave_height: Vec<f32>,
    wavenumber: Vec<f32>,
    depth: Vec<f32>,
    limit: Vec<f32>,
    arrival: Vec<f32>,
    time: f32,
    profile: BreakerProfile,
}

impl ShoreWaves {
    pub fn new(seabed: &Seabed, sea_level: f32, settings: ShoreSettings) -> Self {
        seabed.validate();
        assert!(
            settings.period > 0.0
                && settings.height >= 0.0
                && settings.breaker_index > 0.0
                && (0.0..=1.0).contains(&settings.groups),
            "period and breaker index must be positive, height nonnegative, groups in [0, 1]"
        );
        let (w, h) = (seabed.width, seabed.height);
        let omega = TAU / settings.period;
        let incoming = Vec2::from_angle(settings.direction);
        let depth: Vec<f32> = seabed.elevation.iter().map(|e| sea_level - e).collect();
        let wavenumber: Vec<f32> = depth
            .iter()
            .map(|d| wavenumber(omega, d.max(MIN_DEPTH)))
            .collect();

        // Sources: deep water, where waves are unrefracted, and the map edge
        // facing the waves most directly. Elsewhere the phase is the first
        // arrival. shortcut: an oblique swell enters from one edge only, so
        // keep the area of interest away from the other upwave edge.
        let along_x = incoming.x.abs() >= incoming.y.abs();
        let mut phase = vec![f32::INFINITY; w * h];
        let mut arrival = vec![f32::INFINITY; w * h];
        let mut fixed = vec![false; w * h];
        let lag: Vec<f32> = wavenumber
            .iter()
            .zip(&depth)
            .map(|(&k, &d)| 1.0 / group_velocity(k, d.max(MIN_DEPTH)))
            .collect();
        for y in 0..h {
            for x in 0..w {
                let i = x + y * w;
                let edge = if along_x {
                    x == if incoming.x > 0.0 { 0 } else { w - 1 }
                } else {
                    y == if incoming.y > 0.0 { 0 } else { h - 1 }
                };
                if depth[i] > 0.0 && (edge || wavenumber[i] * depth[i] > PI) {
                    let along = incoming.dot(seabed.cell_center(x, y));
                    phase[i] = wavenumber[i] * along;
                    arrival[i] = lag[i] * along;
                    fixed[i] = true;
                }
            }
        }
        // Crests follow the phase speed; sets follow the group speed.
        rayon::join(
            || sweep_eikonal(&mut phase, &fixed, &wavenumber, w, h, seabed.cell),
            || sweep_eikonal(&mut arrival, &fixed, &lag, w, h, seabed.cell),
        );

        let deep_group = GRAVITY / (2.0 * omega);
        let mut direction = vec![incoming; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = x + y * w;
                let gradient = |a: Option<usize>, b: Option<usize>| match (a, b) {
                    (Some(a), Some(b)) => (phase[b] - phase[a]) / 2.0,
                    (None, Some(b)) => phase[b] - phase[i],
                    (Some(a), None) => phase[i] - phase[a],
                    (None, None) => 0.0,
                };
                let g = Vec2::new(
                    gradient((x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1)),
                    gradient((y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)),
                );
                if g.is_finite() && g != Vec2::ZERO {
                    direction[i] = g.normalize();
                }
            }
        }
        let flux = energy_flux(&phase, &direction, &depth, w, h);
        let mut wave_height = vec![0.0; w * h];
        let mut limit = vec![0.0; w * h];
        for i in 0..w * h {
            if depth[i] <= 0.0 || phase[i].is_infinite() {
                continue;
            }
            let k = wavenumber[i];
            wave_height[i] =
                settings.height * (flux[i] * deep_group / group_velocity(k, depth[i])).sqrt();
            limit[i] =
                (settings.breaker_index * depth[i]).min(0.142 * TAU / k * (k * depth[i]).tanh());
        }
        // Water no wave reaches stays calm.
        for field in [&mut phase, &mut arrival] {
            field
                .iter_mut()
                .filter(|p| p.is_infinite())
                .for_each(|p| *p = 0.0);
        }
        let offshore = omega * omega / GRAVITY;
        let across = incoming.perp() * offshore / CRESTS_PER_GROUP;
        Self {
            settings,
            sea_level,
            width: w,
            cell: seabed.cell,
            origin: seabed.origin,
            omega,
            across,
            phase,
            direction,
            wave_height,
            wavenumber,
            depth,
            limit,
            arrival,
            time: 0.0,
            profile: BreakerProfile::default(),
        }
    }

    pub fn settings(&self) -> &ShoreSettings {
        &self.settings
    }

    pub fn set_time(&mut self, time: f32) {
        self.time = time;
    }

    /// The predefined profile used by both surface queries and rendering.
    pub fn profile(&self) -> BreakerProfile {
        self.profile
    }

    /// Select the profile used for subsequent surface queries. Re-upload it
    /// to the renderer when changing it so both sides stay in sync.
    pub fn set_profile(&mut self, profile: BreakerProfile) {
        assert!(
            profile.horizontal_second_harmonic.is_finite()
                && profile.vertical_second_harmonic.is_finite(),
            "breaker profile coefficients must be finite"
        );
        self.profile = profile;
    }

    /// Rendering inputs, borrowed without allocating or copying the grid.
    pub fn breaker_grid(&self) -> BreakerGrid<'_> {
        BreakerGrid {
            width: self.width,
            height: self.phase.len() / self.width,
            cell: self.cell,
            origin: self.origin,
            sea_level: self.sea_level,
            omega: self.omega,
            time: self.time,
            lean: self.settings.lean,
            groups: self.settings.groups,
            across: self.across,
            phase: &self.phase,
            direction: &self.direction,
            wave_height: &self.wave_height,
            wavenumber: &self.wavenumber,
            depth: &self.depth,
            limit: &self.limit,
            arrival: &self.arrival,
        }
    }

    /// The wave train at `point` at the current time.
    pub fn local_wave(&self, point: Vec2) -> LocalWave {
        let q = (point - self.origin) / self.cell - 0.5;
        let lag = self.time - bilinear(&self.arrival, self.width, q, false);
        let group = group_envelope(
            self.settings.groups,
            self.omega,
            lag,
            self.across.dot(point),
        );
        let (height, breaking) = break_wave(
            bilinear(&self.wave_height, self.width, q, false) * group,
            bilinear(&self.limit, self.width, q, false),
        );
        LocalWave {
            direction: bilinear(&self.direction, self.width, q, false).normalize_or(Vec2::X),
            height,
            wavenumber: bilinear(&self.wavenumber, self.width, q, false),
            breaking,
        }
    }

    /// Still-water depth at a point, m; negative on land.
    pub fn depth_at(&self, point: Vec2) -> f32 {
        let q = (point - self.origin) / self.cell - 0.5;
        bilinear(&self.depth, self.width, q, false)
    }

    /// Parametric displacement (m) and particle velocity (m/s) at a rest
    /// coordinate. Unlike a heightfield, the rendered surface may overturn.
    pub fn displacement_at_rest(&self, rest: Vec2) -> (Vec3, Vec3) {
        let q = (rest - self.origin) / self.cell - 0.5;
        let wave = self.local_wave(rest);
        let depth = bilinear(&self.depth, self.width, q, false);
        if depth <= 0.0 {
            return (Vec3::ZERO, Vec3::ZERO);
        }
        let depth = depth.max(MIN_DEPTH);
        let theta = bilinear(&self.phase, self.width, q, false) - self.omega * self.time;
        let k = wave.wavenumber;
        // Airy surface orbits: circles in deep water, flattening ellipses in shallow.
        let vertical = 0.5 * wave.height;
        let horizontal = (vertical / (k * depth).tanh()).min(MAX_STEEPNESS / k);
        let shape = self.profile().evaluate(theta, wave.breaking);
        let (sl, cl) = (self.settings.lean * wave.breaking).sin_cos();
        let lean = |along: f32, up: f32| (along * cl + up * sl, up * cl - along * sl);
        let (along, up) = lean(horizontal * shape.along, vertical * shape.up);
        let (along_v, up_v) = lean(
            -horizontal * self.omega * shape.along_derivative,
            -vertical * self.omega * shape.up_derivative,
        );
        let d = wave.direction;
        (
            Vec3::new(d.x * along, d.y * along, up),
            Vec3::new(d.x * along_v, d.y * along_v, up_v),
        )
    }
}

/// Samples one approximate branch of the parametric surface by fixed-point
/// inversion. An overturning crest can have multiple heights at one XY;
/// this inexpensive gameplay query cannot represent all of them and may
/// not converge near a fold. Use the parametric profile for rendering.
impl Water for ShoreWaves {
    fn sample(&self, point: Vec2) -> WaterSample {
        let rest = undisplace(point, |p| self.displacement_at_rest(p).0);
        let (displacement, velocity) = self.displacement_at_rest(rest);
        WaterSample {
            height: self.sea_level + displacement.z,
            velocity,
        }
    }
}

/// Significant wave height surf can hold, over depth: saturated surf keeps
/// its root-mean-square height near 0.42 of the depth (Thornton & Guza 1982).
const SATURATED_SURF: f32 = 0.6;

/// An open ocean running into a coast: the swell of a [`ShoreWaves`] with the
/// sea of an [`Ocean`] on top, whose waves die away as the water grows too
/// shallow to hold them. As a [`Water`] it is also a target for
/// [`ShallowWater::relax_edges`](crate::ShallowWater::relax_edges), for a
/// simulated beach inside it. Update both before sampling.
#[derive(Clone, Copy)]
pub struct Coast<'a> {
    pub ocean: &'a Ocean,
    pub shore: &'a ShoreWaves,
}

impl Coast<'_> {
    /// Share of the open sea's waves the depth at `point` lets through.
    pub fn survival(&self, point: Vec2) -> f32 {
        let held = SATURATED_SURF * self.shore.depth_at(point);
        (held / self.ocean.significant_height().max(1e-6)).clamp(0.0, 1.0)
    }

    /// Whitewater at a point: whitecaps at sea, breakers at the shore.
    pub fn foam(&self, point: Vec2) -> f32 {
        (self.survival(point) * self.ocean.foam(point)).max(self.shore.local_wave(point).breaking)
    }
}

impl Water for Coast<'_> {
    fn sample(&self, point: Vec2) -> WaterSample {
        let (sea, swell) = (self.ocean.sample(point), self.shore.sample(point));
        let survival = self.survival(point);
        WaterSample {
            height: swell.height + survival * sea.height,
            velocity: swell.velocity + survival * sea.velocity,
        }
    }
}

/// Energy flux along the rays, relative to the open sea: the steady transport
/// ∇·(F k̂) = 0 solved upwind in phase order, so diverging rays lose height,
/// converging ones gain it, and land casts a lee shadow. First-order upwinding
/// spreads shadow edges a little, which stands in for diffraction.
fn energy_flux(phase: &[f32], direction: &[Vec2], depth: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut order: Vec<usize> = (0..w * h)
        .filter(|&i| depth[i] > 0.0 && phase[i].is_finite())
        .collect();
    order.par_sort_unstable_by(|&a, &b| phase[a].total_cmp(&phase[b]));
    let mut flux = vec![0.0f32; w * h];
    for i in order {
        let (x, y, d) = (i % w, i / w, direction[i]);
        let upwave_x = if d.x >= 0.0 {
            (x > 0).then(|| i - 1)
        } else {
            (x + 1 < w).then(|| i + 1)
        };
        let upwave_y = if d.y >= 0.0 {
            (y > 0).then(|| i - w)
        } else {
            (y + 1 < h).then(|| i + w)
        };
        // Past the map edge lies open sea.
        let inflow_x = upwave_x.map_or(d.x.abs(), |a| {
            flux[a] * (direction[a].x * d.x.signum()).max(0.0)
        });
        let inflow_y = upwave_y.map_or(d.y.abs(), |a| {
            flux[a] * (direction[a].y * d.y.signum()).max(0.0)
        });
        flux[i] = (inflow_x + inflow_y) / (d.x.abs() + d.y.abs());
    }
    flux
}

/// Fast sweeping (Zhao 2005) for |∇φ| = k, first order Godunov upwind.
fn sweep_eikonal(
    phase: &mut [f32],
    fixed: &[bool],
    slowness: &[f32],
    w: usize,
    h: usize,
    cell: f32,
) {
    let at = |phase: &[f32], x: usize, y: usize, dx: isize, dy: isize| {
        let (nx, ny) = (x as isize + dx, y as isize + dy);
        if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
            f32::INFINITY
        } else {
            phase[nx as usize + ny as usize * w]
        }
    };
    for _ in 0..8 {
        let mut changed = 0.0f32;
        for order in 0..4 {
            for yy in 0..h {
                for xx in 0..w {
                    let x = if order & 1 == 0 { xx } else { w - 1 - xx };
                    let y = if order & 2 == 0 { yy } else { h - 1 - yy };
                    let i = x + y * w;
                    if fixed[i] {
                        continue;
                    }
                    let a = at(phase, x, y, -1, 0).min(at(phase, x, y, 1, 0));
                    let b = at(phase, x, y, 0, -1).min(at(phase, x, y, 0, 1));
                    if a.is_infinite() && b.is_infinite() {
                        continue;
                    }
                    let f = slowness[i] * cell;
                    let next = if (a - b).abs() >= f {
                        a.min(b) + f
                    } else {
                        0.5 * (a + b + (2.0 * f * f - (a - b) * (a - b)).sqrt())
                    };
                    if next < phase[i] {
                        changed = changed.max(if phase[i].is_finite() {
                            phase[i] - next
                        } else {
                            f32::INFINITY
                        });
                        phase[i] = next;
                    }
                }
            }
        }
        if changed < 1e-4 {
            break;
        }
    }
}
