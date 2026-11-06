use crate::fft::{Fft2, cis, conj, mul, negated, signed, times_i};
use crate::{
    GRAIN, GRAVITY, Rng, Water, WaterSample, angular_frequency, bilinear, group_velocity,
    undisplace,
};
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
    /// Surface compression (Jacobian) of a single cascade below which whitecaps
    /// form. The default tracks the coverage Monahan & Muircheartaigh 1980
    /// measured against wind speed, within a factor of about three.
    pub foam_threshold: f32,
    /// Seconds for whitecaps to fade to 1/e.
    pub foam_lifetime: f32,
    pub seed: u64,
}

impl Default for OceanSettings {
    fn default() -> Self {
        Self {
            resolution: 128,
            cascades: vec![256.0, 48.0, 9.0],
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
            foam_threshold: 0.745,
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
    previous: Vec<Vec3>,
    k: Vec<Vec2>,
    omega: Vec<f32>,
    h0: Vec<Vec2>,
    /// conj(h0(−k)), the mirror term of h(k, t).
    h0_negated: Vec<Vec2>,
    scratch: [Vec<Vec2>; 4],
}

/// Jacobian drop below the threshold that takes whitecaps to full strength.
const FOAM_RAMP: f32 = 0.2;

impl Cascade {
    fn update(&mut self, fft: &Fft2, time: f32, lambda: f32, fade: f32, threshold: f32) {
        let [a, b, s, j] = &mut self.scratch;
        // Whole turns go first, in f64: at f32 the phase of the shortest
        // waves turns to noise within hours of `time`.
        let turns = f64::from(time) / std::f64::consts::TAU;
        (
            &mut *a,
            &mut *b,
            &mut *s,
            &mut *j,
            &self.k,
            &self.omega,
            &self.h0,
            &self.h0_negated,
        )
            .into_par_iter()
            .with_min_len(GRAIN)
            .for_each(|(a, b, s, j, &k, &omega, &h0, &h0_negated)| {
                // The inverse FFT uses exp(+i k·x), so exp(-i ωt)
                // propagates the directional spectrum along +k.
                let phase = f64::from(omega) * turns;
                let e = cis(-((phase - phase.floor()) as f32) * TAU);
                let h = mul(h0, e) + mul(h0_negated, conj(e));
                let kl = k.length().max(f32::MIN_POSITIVE);
                // Choppy displacement D = i k/|k| h pulls surface points toward crests.
                let ih = times_i(h);
                let (dx, dy) = (ih * (k.x / kl), ih * (k.y / kl));
                let (jxx, jyy, jxy) = (
                    h * (-k.x * k.x / kl),
                    h * (-k.y * k.y / kl),
                    h * (-k.x * k.y / kl),
                );
                // Real fields packed in pairs: F(A + iB) = F(A) + i F(B).
                *a = dx + times_i(dy);
                *b = h + times_i(jxy);
                *s = ih * k.x + times_i(ih * k.y);
                *j = jxx + times_i(jyy);
            });
        [&mut *a, &mut *b, &mut *s, &mut *j]
            .into_par_iter()
            .for_each(|field| fft.run(field, true));
        std::mem::swap(&mut self.previous, &mut self.displacement);
        (
            &mut self.displacement,
            &mut self.slope,
            &mut self.foam,
            &*a,
            &*b,
            &*s,
            &*j,
        )
            .into_par_iter()
            .with_min_len(GRAIN)
            .for_each(|(displacement, slope, foam, a, b, s, j)| {
                *displacement = Vec3::new(lambda * a.x, lambda * a.y, b.x);
                *slope = *s;
                let jacobian = (1.0 + lambda * j.x) * (1.0 + lambda * j.y) - (lambda * b.y).powi(2);
                *foam = (*foam * fade).max(((threshold - jacobian) / FOAM_RAMP).clamp(0.0, 1.0));
            });
    }
}

pub struct Ocean {
    settings: OceanSettings,
    fft: Fft2,
    cascades: Vec<Cascade>,
    significant_height: f32,
    time: f32,
    dt: f32,
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
                    previous: vec![Vec3::ZERO; n * n],
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
            dt: 0.0,
            settings,
        };
        ocean.update(0.0);
        ocean.dt = 0.0;
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
        self.dt = dt;
        self.time = time;
    }

    /// Summed displacement of the surface point resting at `rest`.
    pub fn displacement(&self, rest: Vec2) -> Vec3 {
        self.sum(rest, |c| &c.displacement)
    }

    /// Whitecap coverage at a world point.
    pub fn foam(&self, point: Vec2) -> f32 {
        let rest = undisplace(point, |p| self.displacement(p));
        let n = self.settings.resolution;
        self.cascades
            .iter()
            .map(|c| bilinear(&c.foam, n, rest / c.size * n as f32, true))
            .fold(0.0, f32::max)
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
        let rest = undisplace(point, |p| self.displacement(p));
        let now = self.displacement(rest);
        let velocity = if self.dt != 0.0 {
            (now - self.sum(rest, |c| &c.previous)) / self.dt
        } else {
            Vec3::ZERO
        };
        WaterSample {
            height: now.z,
            velocity,
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
