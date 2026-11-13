use crate::fft::{Fft2, cis, conj, mul, negated, signed, times_i};
use crate::{GRAIN, GRAVITY, Rng, Water, WaterSample, angular_frequency, bilinear, group_velocity};
use glam::{Vec2, Vec3};
use rayon::prelude::*;
use std::f32::consts::{PI, TAU};

/// A JONSWAP spectrum: a wind sea, or a swell from a distant storm.
#[derive(Debug, Clone, PartialEq)]
pub struct Spectrum {
    /// Wind speed 10 m above the sea, m/s.
    pub wind_speed: f32,
    /// Distance the wind has blown over open water, m. Longer fetch grows
    /// taller, longer waves.
    pub fetch: f32,
    /// Direction the waves travel, radians counterclockwise from +X.
    /// A value of zero sends crests toward +X (rather than wind coming from +X).
    pub direction: f32,
    /// Peak enhancement γ: 3.3 is the JONSWAP mean, 1 a fully developed
    /// Pierson-Moskowitz sea.
    pub peak_enhancement: f32,
    /// 0 for a short-crested wind sea; toward 1 narrows spreading into
    /// long-crested swell (Horvath 2015).
    pub swell: f32,
    /// Energy multiplier; 1 is physical.
    pub scale: f32,
}

impl Default for Spectrum {
    fn default() -> Self {
        Self {
            wind_speed: 10.0,
            fetch: 100_000.0,
            direction: 0.0,
            peak_enhancement: 3.3,
            swell: 0.0,
            scale: 1.0,
        }
    }
}

impl Spectrum {
    /// Phillips constant α and peak angular frequency ωp (Hasselmann et al. 1973).
    fn peak(&self) -> (f32, f32) {
        let (u, f, g) = (self.wind_speed, self.fetch, GRAVITY);
        (
            0.076 * (u * u / (f * g)).powf(0.22),
            22.0 * (g * g / (u * f)).powf(1.0 / 3.0),
        )
    }

    /// Frequency spectrum S(ω), m²·s, attenuated by finite depth (TMA).
    pub fn frequency_density(&self, omega: f32, depth: f32) -> f32 {
        if omega <= 0.0 {
            return 0.0;
        }
        let (alpha, wp) = self.peak();
        let sigma = if omega <= wp { 0.07 } else { 0.09 };
        let r = (-(omega - wp).powi(2) / (2.0 * sigma * sigma * wp * wp)).exp();
        let jonswap = alpha * GRAVITY * GRAVITY / omega.powi(5)
            * (-1.25 * (wp / omega).powi(4)).exp()
            * self.peak_enhancement.powf(r);
        jonswap * tma(omega, depth) * self.scale
    }

    /// Wavenumber spectrum S(k), m⁴: integrated over the whole k-plane it
    /// gives the surface height variance.
    pub fn density(&self, k: Vec2, depth: f32) -> f32 {
        let kl = k.length();
        if kl == 0.0 {
            return 0.0;
        }
        let omega = angular_frequency(kl, depth);
        let theta = k.y.atan2(k.x) - self.direction;
        self.frequency_density(omega, depth)
            * self.spreading(omega, theta)
            * group_velocity(kl, depth)
            / kl
    }

    /// Directional spreading, integrating to 1 over θ.
    fn spreading(&self, omega: f32, theta: f32) -> f32 {
        let (_, wp) = self.peak();
        let ratio = omega / wp;
        if self.swell <= 0.0 {
            return donelan_banner(ratio, wrap(theta));
        }
        let s = 16.0 * (wp / omega).tanh() * self.swell * self.swell;
        let f = |t: f32| donelan_banner(ratio, t) * (t * 0.5).cos().abs().powf(2.0 * s);
        // shortcut: normalizes numerically per call (init only); tabulate by ω if startup gets slow.
        const SAMPLES: usize = 64;
        let norm: f32 = (0..SAMPLES)
            .map(|i| f(-PI + (i as f32 + 0.5) * TAU / SAMPLES as f32))
            .sum::<f32>()
            * TAU
            / SAMPLES as f32;
        f(wrap(theta)) / norm
    }
}

/// Kitaigorodskii depth attenuation, as approximated by Thompson & Vincent 1983.
fn tma(omega: f32, depth: f32) -> f32 {
    let wh = omega * (depth / GRAVITY).sqrt();
    if wh <= 1.0 {
        0.5 * wh * wh
    } else if wh < 2.0 {
        1.0 - 0.5 * (2.0 - wh).powi(2)
    } else {
        1.0
    }
}

/// Donelan, Hamilton & Hui 1985 with Banner 1990 above 1.6 ωp.
fn donelan_banner(ratio: f32, theta: f32) -> f32 {
    let beta = if ratio < 0.95 {
        2.61 * ratio.powf(1.3)
    } else if ratio < 1.6 {
        2.28 * ratio.powf(-0.65)
    } else {
        10f32.powf(-0.4 + 0.8393 * (-0.567 * (ratio * ratio).ln()).exp())
    };
    beta / (2.0 * (beta * PI).tanh()) / (beta * theta).cosh().powi(2)
}

fn wrap(theta: f32) -> f32 {
    (theta + PI).rem_euclid(TAU) - PI
}

/// Each cascade starts this many of its own fundamentals above zero, so the
/// next-larger cascade must resolve up to there.
const BAND: f32 = 6.0;

#[derive(Debug, Clone, PartialEq)]
pub struct OceanSettings {
    /// Texels per cascade side, a power of two.
    pub resolution: usize,
    /// Patch size of each cascade in metres, largest first. Each tiles
    /// seamlessly and carries its own wavelength band.
    pub cascades: Vec<f32>,
    /// Sea depth in metres; `f32::INFINITY` for the open ocean.
    pub depth: f32,
    pub spectra: Vec<Spectrum>,
    /// Horizontal displacement λ: 0 gives rounded crests, 1 the sharp
    /// physical ones; above 1 crests fold over.
    pub choppiness: f32,
    /// Surface compression (Jacobian) below which whitecaps form, judged per
    /// cascade and on the summed surface. The default tracks the coverage
    /// Monahan & Muircheartaigh 1980 measured against wind speed, within a
    /// factor of about three.
    pub foam_threshold: f32,
    /// Seconds for whitecaps to fade to 1/e.
    pub foam_lifetime: f32,
    pub seed: u64,
}

impl Default for OceanSettings {
    fn default() -> Self {
        Self {
            resolution: 128,
            // Non-integer ratios, so no two cascades repeat in step.
            cascades: vec![1400.0, 256.0, 48.0, 9.0],
            depth: f32::INFINITY,
            spectra: vec![
                Spectrum::default(),
                Spectrum {
                    wind_speed: 14.0,
                    fetch: 500_000.0,
                    direction: 0.4,
                    swell: 0.8,
                    scale: 0.4,
                    ..Spectrum::default()
                },
            ],
            choppiness: 0.9,
            foam_threshold: 0.76,
            foam_lifetime: 2.0,
            seed: 1,
        }
    }
}

/// One tile of the ocean, sampled at texel (x, y) = world `(x, y) * size /
/// resolution`, modulo `size`. Fields are row-major, `resolution²` long.
#[derive(Debug, Clone)]
pub struct Cascade {
    pub size: f32,
    /// Horizontal displacement in x, y and height in z, m.
    pub displacement: Vec<Vec3>,
    /// (∂z/∂x, ∂z/∂y) for normals: summed over cascades, the normal is
    /// normalize(−sx, −sy, 1).
    pub slope: Vec<Vec2>,
    /// Whitecap coverage in [0, 1].
    pub foam: Vec<f32>,
    /// Surface velocity, m/s.
    velocity: Vec<Vec3>,
    /// Displacement gradient (∂Dx/∂x, ∂Dy/∂y, ∂Dx/∂y): summed over cascades
    /// it gives the whole surface's compression.
    gradient: Vec<Vec3>,
    k: Vec<Vec2>,
    omega: Vec<f32>,
    h0: Vec<Vec2>,
    /// conj(h0(−k)), the mirror term of h(k, t).
    h0_negated: Vec<Vec2>,
    scratch: [Vec<Vec2>; 4],
}

/// Newton steps per surface query; each costs two lookups per cascade.
const NEWTON_STEPS: usize = 3;

/// Jacobian drop below the threshold that takes whitecaps to full strength.
const FOAM_RAMP: f32 = 0.2;

impl Cascade {
    fn update(&mut self, fft: &Fft2, time: f32, lambda: f32, fade: f32, threshold: f32) {
        let [a, b, s, v] = &mut self.scratch;
        // Whole turns go first, in f64: at f32 the phase of the shortest
        // waves turns to noise within hours of `time`.
        let turns = f64::from(time) / std::f64::consts::TAU;
        (
            &mut *a,
            &mut *b,
            &mut *s,
            &mut *v,
            &self.k,
            &self.omega,
            &self.h0,
            &self.h0_negated,
        )
            .into_par_iter()
            .with_min_len(GRAIN)
            .for_each(|(a, b, s, v, &k, &omega, &h0, &h0_negated)| {
                // The inverse FFT uses exp(+i k·x), so exp(-i ωt)
                // propagates the directional spectrum along +k.
                let phase = f64::from(omega) * turns;
                let e = cis(-((phase - phase.floor()) as f32) * TAU);
                let (ahead, mirror) = (mul(h0, e), mul(h0_negated, conj(e)));
                let h = ahead + mirror;
                // ∂h/∂t, exact at any time, so velocity needs no previous frame.
                let dh = times_i(mirror - ahead) * omega;
                let kl = k.length().max(f32::MIN_POSITIVE);
                // Choppy displacement D = i k/|k| h pulls surface points toward
                // crests; it moves at i k/|k| ∂h/∂t.
                let (ih, idh) = (times_i(h), times_i(dh));
                // Real fields packed in pairs: F(A + iB) = F(A) + i F(B).
                *a = ih * (k.x / kl) + times_i(ih * (k.y / kl));
                *b = h + times_i(dh);
                *s = ih * k.x + times_i(ih * k.y);
                *v = idh * (k.x / kl) + times_i(idh * (k.y / kl));
            });
        [&mut *a, &mut *b, &mut *s, &mut *v]
            .into_par_iter()
            .for_each(|field| fft.run(field, true));
        (
            &mut self.displacement,
            &mut self.slope,
            &mut self.velocity,
            &*a,
            &*b,
            &*s,
            &*v,
        )
            .into_par_iter()
            .with_min_len(GRAIN)
            .for_each(|(displacement, slope, velocity, a, b, s, v)| {
                *displacement = Vec3::new(lambda * a.x, lambda * a.y, b.x);
                *slope = *s;
                *velocity = Vec3::new(lambda * v.x, lambda * v.y, b.y);
            });
        // The Jacobian by central differences rather than three more FFTs.
        let n = (self.displacement.len() as f64).sqrt() as usize;
        let scale = n as f32 / (2.0 * self.size);
        let d = &self.displacement;
        self.gradient
            .par_chunks_mut(n)
            .zip(self.foam.par_chunks_mut(n))
            .enumerate()
            .with_min_len(GRAIN / n)
            .for_each(|(y, (gradient, foam))| {
                let (up, down) = ((y + 1) % n * n, (y + n - 1) % n * n);
                for (x, (gradient, foam)) in gradient.iter_mut().zip(foam).enumerate() {
                    let (right, left) = ((x + 1) % n, (x + n - 1) % n);
                    let dx = (d[right + y * n] - d[left + y * n]) * scale;
                    let dy = (d[x + up] - d[x + down]) * scale;
                    *gradient = Vec3::new(dx.x, dy.y, 0.5 * (dy.x + dx.y));
                    *foam = (*foam * fade).max(whitecap(*gradient, threshold));
                }
            });
    }
}

/// Whitecap strength where the displacement has `gradient`.
fn whitecap(gradient: Vec3, threshold: f32) -> f32 {
    let jacobian = (1.0 + gradient.x) * (1.0 + gradient.y) - gradient.z * gradient.z;
    ((threshold - jacobian) / FOAM_RAMP).clamp(0.0, 1.0)
}

pub struct Ocean {
    settings: OceanSettings,
    fft: Fft2,
    cascades: Vec<Cascade>,
    significant_height: f32,
    time: f32,
}

impl Ocean {
    pub fn new(settings: OceanSettings) -> Self {
        let n = settings.resolution;
        let sizes = &settings.cascades;
        assert!(
            !sizes.is_empty()
                && sizes.iter().all(|s| s.is_finite() && *s > 0.0)
                && sizes.windows(2).all(|w| w[0] > w[1]),
            "cascade sizes must be positive and decreasing"
        );
        assert!(settings.depth > 0.0, "depth must be positive");
        assert!(
            settings.foam_lifetime > 0.0,
            "foam lifetime must be positive"
        );
        let fft = Fft2::new(n);
        let mut rng = Rng(settings.seed);
        let cascades: Vec<Cascade> = sizes
            .iter()
            .enumerate()
            .map(|(c, &size)| {
                let lo = if c == 0 { 0.0 } else { BAND * TAU / size };
                // Stopping at the Nyquist circle keeps the spectrum isotropic, and
                // drops the Nyquist bins, whose k has no −k to stay Hermitian with.
                let nyquist = PI * n as f32 / size;
                let hi = sizes.get(c + 1).map_or(nyquist, |next| BAND * TAU / next);
                assert!(
                    hi <= nyquist,
                    "cascade {c} cannot resolve the next cascade's band: raise resolution or bring sizes closer"
                );
                let dk = TAU / size;
                let k: Vec<Vec2> = (0..n * n)
                    .map(|i| Vec2::new(signed(i % n, n), signed(i / n, n)) * dk)
                    .collect();
                // E|h0|² = S Δk² / 2, so h(k, t) carries S Δk² and the field variance is ∫S.
                let noise: Vec<Vec2> = (0..n * n).map(|_| rng.gaussian2()).collect();
                let h0: Vec<Vec2> = k
                    .par_iter()
                    .zip(&noise)
                    .map(|(&k, &xi)| {
                        let kl = k.length();
                        if kl < lo || kl >= hi {
                            return Vec2::ZERO;
                        }
                        let s: f32 = settings.spectra.iter().map(|s| s.density(k, settings.depth)).sum();
                        xi * (s * dk * dk / 4.0).sqrt()
                    })
                    .collect();
                Cascade {
                    size,
                    displacement: vec![Vec3::ZERO; n * n],
                    slope: vec![Vec2::ZERO; n * n],
                    foam: vec![0.0; n * n],
                    velocity: vec![Vec3::ZERO; n * n],
                    gradient: vec![Vec3::ZERO; n * n],
                    omega: k.iter().map(|k| angular_frequency(k.length(), settings.depth)).collect(),
                    k,
                    h0_negated: (0..n * n).map(|i| conj(h0[negated(i, n)])).collect(),
                    h0,
                    scratch: std::array::from_fn(|_| vec![Vec2::ZERO; n * n]),
                }
            })
            .collect();
        // Each bin contributes |h0(k)|² + |h0(−k)|² of height variance.
        let variance: f32 = cascades
            .iter()
            .flat_map(|c| c.h0.iter().chain(&c.h0_negated))
            .map(|h| h.length_squared())
            .sum();
        let mut ocean = Self {
            fft,
            cascades,
            significant_height: 4.0 * variance.sqrt(),
            time: 0.0,
            settings,
        };
        ocean.update(0.0);
        ocean
    }

    pub fn settings(&self) -> &OceanSettings {
        &self.settings
    }

    pub fn cascades(&self) -> &[Cascade] {
        &self.cascades
    }

    pub fn time(&self) -> f32 {
        self.time
    }

    /// Mean height of the highest third of the waves, m: four standard
    /// deviations of the surface.
    pub fn significant_height(&self) -> f32 {
        self.significant_height
    }

    /// Evaluate the surface at absolute `time` in seconds.
    pub fn update(&mut self, time: f32) {
        let lambda = self.settings.choppiness;
        let dt = time - self.time;
        let fade = (-dt.abs() / self.settings.foam_lifetime).exp();
        let threshold = self.settings.foam_threshold;
        let fft = &self.fft;
        self.cascades
            .par_iter_mut()
            .for_each(|c| c.update(fft, time, lambda, fade, threshold));
        self.time = time;
    }

    /// Summed displacement of the surface point resting at `rest`.
    pub fn displacement(&self, rest: Vec2) -> Vec3 {
        self.sum(rest, |c| &c.displacement)
    }

    /// Whitecap coverage at a world point: lingering foam from any cascade,
    /// or a crest that only the cascades together steepen enough to break.
    pub fn foam(&self, point: Vec2) -> f32 {
        let rest = self.rest(point);
        let n = self.settings.resolution;
        let lingering = self
            .cascades
            .iter()
            .map(|c| bilinear(&c.foam, n, rest / c.size * n as f32, true))
            .fold(0.0, f32::max);
        let summed = self.sum(rest, |c| &c.gradient);
        lingering.max(whitecap(summed, self.settings.foam_threshold))
    }

    /// The rest position whose surface point is displaced onto `point`.
    /// Newton's method on the displacement and its gradient converges in a
    /// few steps where plain fixed-point iteration crawls near steep crests;
    /// where the surface folds over, the fixed-point step stands in.
    fn rest(&self, point: Vec2) -> Vec2 {
        let mut rest = point;
        for _ in 0..NEWTON_STEPS {
            let miss = rest + self.displacement(rest).truncate() - point;
            let g = self.sum(rest, |c| &c.gradient);
            let det = (1.0 + g.x) * (1.0 + g.y) - g.z * g.z;
            rest -= if det > 0.1 {
                Vec2::new(
                    (1.0 + g.y) * miss.x - g.z * miss.y,
                    (1.0 + g.x) * miss.y - g.z * miss.x,
                ) / det
            } else {
                miss
            };
        }
        rest
    }

    fn sum(&self, rest: Vec2, field: impl Fn(&Cascade) -> &Vec<Vec3>) -> Vec3 {
        let n = self.settings.resolution;
        self.cascades
            .iter()
            .map(|c| bilinear(field(c), n, rest / c.size * n as f32, true))
            .sum()
    }
}

impl Water for Ocean {
    fn sample(&self, point: Vec2) -> WaterSample {
        let rest = self.rest(point);
        WaterSample {
            height: self.displacement(rest).z,
            velocity: self.sum(rest, |c| &c.velocity),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectral_wave_travels_in_its_configured_direction() {
        let mut ocean = Ocean::new(OceanSettings {
            resolution: 16,
            cascades: vec![100.0],
            spectra: vec![],
            choppiness: 0.0,
            ..OceanSettings::default()
        });
        // A single +X wave makes direction observable without statistical
        // variation from the random spectrum.
        ocean.cascades[0].h0[1] = Vec2::new(0.5, 0.0);
        ocean.cascades[0].h0_negated[15] = Vec2::new(0.5, 0.0);
        let quarter_period = PI / (2.0 * ocean.cascades[0].omega[1]);
        ocean.update(quarter_period);
        let heights = &ocean.cascades[0].displacement;
        assert!(
            heights[4].z > 0.99,
            "crest should travel toward +X: {}",
            heights[4].z
        );
        assert!(
            heights[12].z < -0.99,
            "trough should travel toward +X: {}",
            heights[12].z
        );
    }

    #[test]
    fn shortest_waves_keep_their_phase_after_a_day() {
        let n = 16;
        let mut ocean = Ocean::new(OceanSettings {
            resolution: n,
            cascades: vec![1.0],
            spectra: vec![],
            choppiness: 0.0,
            ..OceanSettings::default()
        });
        let bin = n / 2 - 1;
        ocean.cascades[0].h0[bin] = Vec2::new(0.5, 0.0);
        ocean.cascades[0].h0_negated[n - bin] = Vec2::new(0.5, 0.0);
        let time = 86_400.0f32;
        ocean.update(time);
        let omega = f64::from(ocean.cascades[0].omega[bin]);
        let expected = (omega * f64::from(time)).cos() as f32;
        let got = ocean.cascades[0].displacement[0].z;
        assert!((got - expected).abs() < 1e-3, "{got} vs {expected}");
    }

    #[test]
    fn spectrum_stops_at_the_nyquist_circle() {
        let ocean = Ocean::new(OceanSettings {
            resolution: 32,
            cascades: vec![10.0],
            ..OceanSettings::default()
        });
        let c = &ocean.cascades[0];
        let nyquist = PI * 32.0 / c.size;
        assert!(
            c.k.iter()
                .zip(&c.h0)
                .all(|(k, h)| k.length() < nyquist || *h == Vec2::ZERO)
        );
        assert!(c.h0.iter().any(|h| *h != Vec2::ZERO));
    }
}
