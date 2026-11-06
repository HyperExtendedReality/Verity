use crate::{Water, WaterSample, bilinear};
use glam::{IVec2, Vec2, Vec3};

#[derive(Debug, Clone, PartialEq)]
pub struct RipplesSettings {
    /// Cells per side.
    pub resolution: usize,
    /// m.
    pub cell: f32,
    /// m/s; around 0.3 for the short ripples a wading character makes.
    pub wave_speed: f32,
    /// Fraction of ripple speed lost per second.
    pub damping: f32,
}

impl Default for RipplesSettings {
    fn default() -> Self {
        Self {
            resolution: 64,
            cell: 0.1,
            wave_speed: 0.4,
            damping: 1.5,
        }
    }
}

/// Cells near the window edge that absorb outgoing ripples.
const BORDER: usize = 4;

/// Damped wave equation on a small window that scrolls with a character,
/// so ripples stay where they were made. Its [`Water`] height is an offset:
/// layer it over another water as `(water, &ripples)`.
pub struct Ripples {
    settings: RipplesSettings,
    corner: IVec2,
    height: Vec<f32>,
    rate: Vec<f32>,
}

impl Ripples {
    pub fn new(settings: RipplesSettings) -> Self {
        let n = settings.resolution;
        assert!(
            n > 2 * BORDER && settings.cell > 0.0 && settings.wave_speed > 0.0,
            "ripples need a resolution above 8 and positive cell and speed"
        );
        Self {
            corner: IVec2::splat(-(n as i32) / 2),
            height: vec![0.0; n * n],
            rate: vec![0.0; n * n],
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
    ///
    /// Surface velocity is stored in meters per second, so changing the frame
    /// duration does not reinterpret the previous frame's displacement.
    pub fn step(&mut self, dt: f32) {
        assert!(
            dt.is_finite() && dt >= 0.0,
            "dt must be finite and nonnegative"
        );
        if dt == 0.0 {
            return;
        }
        let s = &self.settings;
        let n = s.resolution;
        let substeps = (dt * s.wave_speed / (0.5 * s.cell)).ceil().max(1.0) as usize;
        let h = dt / substeps as f32;
        let c2 = (s.wave_speed * h / s.cell).powi(2);
        let keep = (1.0 - s.damping * h).max(0.0);
        let mut next = vec![0.0; n * n];
        for _ in 0..substeps {
            for y in 0..n {
                for x in 0..n {
                    let i = x + y * n;
                    let at = |dx: isize, dy: isize| {
                        let (nx, ny) = (x as isize + dx, y as isize + dy);
                        if nx < 0 || ny < 0 || nx >= n as isize || ny >= n as isize {
                            0.0
                        } else {
                            self.height[nx as usize + ny as usize * n]
                        }
                    };
                    let laplacian =
                        at(-1, 0) + at(1, 0) + at(0, -1) + at(0, 1) - 4.0 * self.height[i];
                    let edge = x.min(y).min(n - 1 - x).min(n - 1 - y);
                    let absorb = if edge < BORDER {
                        edge as f32 / BORDER as f32
                    } else {
                        1.0
                    };
                    let height = self.height[i];
                    next[i] = (height + self.rate[i] * h * keep + c2 * laplacian) * absorb;
                    self.rate[i] = (next[i] - height) / h;
                }
            }
            std::mem::swap(&mut self.height, &mut next);
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
        WaterSample {
            height: bilinear(&self.height, n, q, false),
            velocity: Vec3::Z * bilinear(&self.rate, n, q, false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!((after.height - expected).abs() < 0.0001);
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
