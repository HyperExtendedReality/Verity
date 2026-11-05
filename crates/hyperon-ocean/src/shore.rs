use crate::{
    GRAVITY, Seabed, Water, WaterSample, bilinear, group_velocity, undisplace, wavenumber,
};
use glam::{Vec2, Vec3};
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
}

impl Default for ShoreSettings {
    fn default() -> Self {
        Self {
            period: 8.0,
            height: 1.0,
            direction: 0.0,
            breaker_index: 0.78,
            lean: 0.6,
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

/// Borrowed, row-major cell-centered wave fields for rendering.
///
/// For a rest position `p`, interpolate each field at
/// `(p - origin) / cell - 0.5`, clamping at the grid edge, and normalize the
/// interpolated direction (falling back to +X if zero). Interpolated signed
/// depth <= 0 is dry and must be masked; otherwise clamp depth to at least
/// 0.05 m for the orbit. The phase is `phase - omega * time`. Evaluate the
/// profile, scale along by `min(wave_height / (2*tanh(k*depth)), 0.9/k)` and
/// up by `wave_height/2`, then rotate by `lean * breaking` as in
/// [`ShoreWaves::displacement_at_rest`]. Wave height is in metres, signed
/// depth and cell are metres, omega is radians/second, and time is seconds.
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
    pub phase: &'a [f32],
    pub direction: &'a [Vec2],
    pub wave_height: &'a [f32],
    pub wavenumber: &'a [f32],
    pub depth: &'a [f32],
    pub breaking: &'a [f32],
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
    phase: Vec<f32>,
    direction: Vec<Vec2>,
    wave_height: Vec<f32>,
    wavenumber: Vec<f32>,
    depth: Vec<f32>,
    breaking: Vec<f32>,
    time: f32,
    profile: BreakerProfile,
}

impl ShoreWaves {
    pub fn new(seabed: &Seabed, sea_level: f32, settings: ShoreSettings) -> Self {
        seabed.validate();
        assert!(
            settings.period > 0.0 && settings.height >= 0.0 && settings.breaker_index > 0.0,
            "period and breaker index must be positive, height nonnegative"
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
        let mut fixed = vec![false; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = x + y * w;
                let edge = if along_x {
                    x == if incoming.x > 0.0 { 0 } else { w - 1 }
                } else {
                    y == if incoming.y > 0.0 { 0 } else { h - 1 }
                };
                if depth[i] > 0.0 && (edge || wavenumber[i] * depth[i] > PI) {
                    phase[i] = wavenumber[i] * incoming.dot(seabed.cell_center(x, y));
                    fixed[i] = true;
                }
            }
        }
        sweep_eikonal(&mut phase, &fixed, &wavenumber, w, h, seabed.cell);

        let deep_group = GRAVITY / (2.0 * omega);
        let mut direction = vec![incoming; w * h];
        let mut wave_height = vec![0.0; w * h];
        let mut breaking = vec![0.0; w * h];
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
                if depth[i] <= 0.0 || phase[i].is_infinite() {
                    continue;
                }
                let k = wavenumber[i];
                // shortcut: energy flux conserved along rays without the refraction
                // coefficient (ray spreading); add a transport solve if focusing over
                // reefs and canyons matters.
                let shoaled = settings.height * (deep_group / group_velocity(k, depth[i])).sqrt();
                let limit = (settings.breaker_index * depth[i])
                    .min(0.142 * TAU / k * (k * depth[i]).tanh());
                wave_height[i] = shoaled.min(limit);
                breaking[i] = ((shoaled / limit - 0.9) / 0.3).clamp(0.0, 1.0);
            }
        }
        // Water no wave reaches stays calm.
        phase
            .iter_mut()
            .filter(|p| p.is_infinite())
            .for_each(|p| *p = 0.0);
        Self {
            settings,
            sea_level,
            width: w,
            cell: seabed.cell,
            origin: seabed.origin,
            omega,
            phase,
            direction,
            wave_height,
            wavenumber,
            depth,
            breaking,
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
            phase: &self.phase,
            direction: &self.direction,
            wave_height: &self.wave_height,
            wavenumber: &self.wavenumber,
            depth: &self.depth,
            breaking: &self.breaking,
        }
    }

    pub fn local_wave(&self, point: Vec2) -> LocalWave {
        let q = (point - self.origin) / self.cell - 0.5;
        LocalWave {
            direction: bilinear(&self.direction, self.width, q, false).normalize_or(Vec2::X),
            height: bilinear(&self.wave_height, self.width, q, false),
            wavenumber: bilinear(&self.wavenumber, self.width, q, false),
            breaking: bilinear(&self.breaking, self.width, q, false),
        }
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
