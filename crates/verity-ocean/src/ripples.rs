use crate::fft::{Fft2, conj, negated, signed, times_i};
use crate::{GRAVITY, Water, WaterSample, bilinear};
use glam::{IVec2, Vec2, Vec3};
use std::f32::consts::TAU;

#[derive(Debug, Clone, PartialEq)]
pub struct RipplesSettings {
    /// Cells per side, a power of two.
    pub resolution: usize,
    /// m.
    pub cell: f32,
    /// Water depth, m; `f32::INFINITY` for open water.
    pub depth: f32,
    /// Oscillator damping, 1/s: ripple amplitude fades as exp(−damping·t/2).
    pub damping: f32,
}

impl Default for RipplesSettings {
    fn default() -> Self {
        Self {
            resolution: 64,
            cell: 0.1,
            depth: 1.0,
            damping: 1.5,
        }
    }
}

/// Surface tension over density for clean water, m³/s².
const TENSION: f32 = 7.2e-5;

/// Linear water waves on a small window that scrolls with a character, so
/// ripples stay where they were made. Each Fourier mode advances exactly by
/// the gravity-capillary dispersion relation (Tessendorf's eWave, 2014): any
/// `dt` is stable, a splash disperses into a train of rings with the short
/// ones trailing, and a moving hull draws a true Kelvin wake rather than the
/// Mach cone of a plain wave equation. Its [`Water`] height is an offset:
/// layer it over another water as `(water, &ripples)`.
pub struct Ripples {
    settings: RipplesSettings,
    corner: IVec2,
    height: Vec<f32>,
    rate: Vec<f32>,
    fft: Fft2,
    omega: Vec<f32>,
    /// Absorption by cell, 1/s: the FFT is periodic, so waves must die in
    /// this border before they wrap around the window.
    sponge: Vec<f32>,
    modes: Vec<Vec2>,
}

impl Ripples {
    pub fn new(settings: RipplesSettings) -> Self {
        let n = settings.resolution;
        assert!(
            n >= 16 && settings.cell > 0.0 && settings.depth > 0.0 && settings.damping >= 0.0,
            "ripples need a resolution of at least 16 and positive cell and depth"
        );
        let dk = TAU / (n as f32 * settings.cell);
        let omega = (0..n * n)
            .map(|c| {
                let k = dk * Vec2::new(signed(c % n, n), signed(c / n, n)).length();
                let tanh = (k * settings.depth).min(20.0).tanh();
                ((GRAVITY * k + TENSION * k * k * k) * tanh).sqrt()
            })
            .collect();
        // Strong enough to stop the fastest wave the window holds, its longest.
        let border = n as f32 / 8.0;
        let window = n as f32 * settings.cell;
        let fastest = 0.5 * (GRAVITY * window / TAU).sqrt();
        let peak = 12.0 * fastest / (border * settings.cell);
        let sponge = (0..n * n)
            .map(|c| {
                let (x, y) = (c % n, c / n);
                let edge = x.min(y).min(n - 1 - x).min(n - 1 - y) as f32 + 0.5;
                peak * (1.0 - edge / border).max(0.0).powi(2)
            })
            .collect();
        Self {
            corner: IVec2::splat(-(n as i32) / 2),
            height: vec![0.0; n * n],
            rate: vec![0.0; n * n],
            fft: Fft2::new(n),
            omega,
            sponge,
            modes: vec![Vec2::ZERO; n * n],
            settings,
        }
    }

    /// Center the window on `point`, scrolling by whole cells.
    pub fn follow(&mut self, point: Vec2) {
        let n = self.settings.resolution as i32;
        let corner = (point / self.settings.cell).floor().as_ivec2() - n / 2;
        let shift = corner - self.corner;
        if shift == IVec2::ZERO {
            return;
        }
        for field in [&mut self.height, &mut self.rate] {
            let old = field.clone();
            for y in 0..n {
                for x in 0..n {
                    let (sx, sy) = (x + shift.x, y + shift.y);
                    field[(x + y * n) as usize] = if (0..n).contains(&sx) && (0..n).contains(&sy) {
                        old[(sx + sy * n) as usize]
                    } else {
                        0.0
                    };
                }
            }
        }
        self.corner = corner;
    }

    /// Press the surface down by `depth` m over a disc: footsteps, hulls,
    /// splashes. Negative lifts it.
    pub fn disturb(&mut self, point: Vec2, radius: f32, depth: f32) {
        let n = self.settings.resolution;
        for c in 0..n * n {
            let center = (self.corner.as_vec2() + Vec2::new((c % n) as f32, (c / n) as f32) + 0.5)
                * self.settings.cell;
            let d = center.distance(point) / radius.max(self.settings.cell);
            if d < 1.0 {
                self.height[c] -= depth * (1.0 - d * d).powi(2);
            }
        }
    }

    /// Advance the simulation by `dt` seconds. A zero duration leaves it unchanged.
    pub fn step(&mut self, dt: f32) {
        assert!(
            dt.is_finite() && dt >= 0.0,
            "dt must be finite and nonnegative"
        );
        if dt == 0.0 {
            return;
        }
        let n = self.settings.resolution;
        // Height and speed are real, so one transform carries both: Z = H + iR.
        for (z, (h, r)) in self
            .modes
            .iter_mut()
            .zip(self.height.iter().zip(&self.rate))
        {
            *z = Vec2::new(*h, *r);
        }
        self.fft.run(&mut self.modes, false);
        let keep = (-0.5 * self.settings.damping * dt).exp() / (n * n) as f32;
        for c in 0..n * n {
            let mirror = negated(c, n);
            if mirror < c {
                continue;
            }
            let (a, b) = (self.modes[c], conj(self.modes[mirror]));
            let (h, r) = ((a + b) * 0.5, -times_i((a - b) * 0.5));
            // Above zero, so the still mode's sin(ωt)/ω tends to t.
            let w = self.omega[c].max(1e-6);
            let (s, co) = (w * dt).sin_cos();
            let (h, r) = (h * co + r * (s / w), r * co - h * (w * s));
            self.modes[c] = (h + times_i(r)) * keep;
            self.modes[mirror] = conj(h - times_i(r)) * keep;
        }
        self.fft.run(&mut self.modes, true);
        for (c, z) in self.modes.iter().enumerate() {
            let absorb = (-self.sponge[c] * dt).exp();
            self.height[c] = z.x * absorb;
            self.rate[c] = z.y * absorb;
        }
    }
}

impl Water for Ripples {
    fn sample(&self, point: Vec2) -> WaterSample {
        let n = self.settings.resolution;
        let q = point / self.settings.cell - self.corner.as_vec2() - 0.5;
        let inside = q.cmpge(Vec2::ZERO).all() && q.cmple(Vec2::splat(n as f32 - 1.0)).all();
        if !inside {
            return WaterSample::default();
        }
        let height = bilinear(&self.height, n, q, false);
        // `rate` is the undamped oscillator's; damping adds its own decay.
        let rate = bilinear(&self.rate, n, q, false);
        WaterSample {
            height,
            velocity: Vec3::Z * (rate - 0.5 * self.settings.damping * height),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_oscillate_at_the_gravity_capillary_frequency() {
        let settings = RipplesSettings {
            damping: 0.0,
            depth: f32::INFINITY,
            ..RipplesSettings::default()
        };
        let (n, cell) = (settings.resolution, settings.cell);
        let mut ripples = Ripples::new(settings);
        ripples.sponge.fill(0.0);
        let k = TAU * 16.0 / (n as f32 * cell);
        for c in 0..n * n {
            ripples.height[c] = (k * (c % n) as f32 * cell).cos();
        }
        let omega = (GRAVITY * k + TENSION * k * k * k).sqrt();
        // One long step: stability and phase must not depend on its size.
        let time = 1.37;
        ripples.step(time);
        assert!((ripples.height[0] - (omega * time).cos()).abs() < 1e-3);
    }

    #[test]
    fn changing_frame_duration_preserves_velocity_units() {
        let mut ripples = Ripples::new(RipplesSettings::default());
        let point = Vec2::splat(0.05);
        ripples.disturb(point, 0.2, 1.0);
        ripples.step(0.05);
        let before = ripples.sample(point);
        assert!(before.velocity.z.abs() > 0.1);

        let dt = 0.001;
        ripples.step(dt);
        let after = ripples.sample(point);
        let expected = before.height + before.velocity.z * dt;
        // The neglected acceleration term: ω²·dt²/2 for ω near 17 rad/s.
        assert!((after.height - expected).abs() < 0.0002);
    }

    #[test]
    fn zero_duration_preserves_surface_and_velocity() {
        let mut ripples = Ripples::new(RipplesSettings::default());
        let point = Vec2::splat(0.05);
        ripples.disturb(point, 0.2, 1.0);
        ripples.step(0.05);
        let before = ripples.sample(point);
        ripples.step(0.0);
        let after = ripples.sample(point);
        assert_eq!(before.height, after.height);
        assert_eq!(before.velocity, after.velocity);
    }
}
