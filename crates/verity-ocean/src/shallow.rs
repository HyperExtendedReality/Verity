use crate::fft::{Fft2, cis, conj, mul, negated, signed, times_i};
use crate::{GRAIN, GRAVITY, Seabed, Water, WaterSample, angular_frequency, bilinear};
use glam::Vec2;
use rayon::prelude::*;
use std::f32::consts::TAU;

/// What happens at one edge of the domain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Edge {
    /// Nothing crosses.
    Wall,
    /// Water leaves freely, as at the downstream end of a river.
    Open,
    /// A steady inflow: `rate` m³/s entering over `span`, the metres along
    /// the edge from the domain's lower corner.
    Discharge { rate: f32, span: [f32; 2] },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShallowWaterSettings {
    /// Manning bed roughness n, s/m^⅓: 0.012 concrete, 0.03 natural streams.
    pub manning: f32,
    /// Carry short waves with Airy dispersion (Jeschke & Wojtan 2023) rather
    /// than at the shallow-water speed √(g h), which is wrong once waves are
    /// shorter than a few depths.
    pub dispersion: bool,
    /// Depths the Airy step is solved at and interpolated between; each
    /// costs two FFTs per substep.
    pub depth_levels: usize,
    /// Diffusion passes splitting bulk flow from surface waves. More moves
    /// longer waves into the dispersive layer, up to wavelengths of about
    /// 2π·depth, where the shallow-water speed becomes the right one.
    pub split_iterations: usize,
    /// Courant number bounding substeps, in (0, 1].
    pub courant: f32,
    /// Seconds for foam to fade to 1/e.
    pub foam_lifetime: f32,
    /// Seconds for exposed ground to dry to 1/e.
    pub wetness_lifetime: f32,
    /// The domain's −X, +X, −Y and +Y edges.
    pub edges: [Edge; 4],
}

impl Default for ShallowWaterSettings {
    fn default() -> Self {
        Self {
            manning: 0.02,
            dispersion: true,
            depth_levels: 4,
            split_iterations: 16,
            courant: 0.4,
            foam_lifetime: 3.0,
            wetness_lifetime: 30.0,
            edges: [Edge::Wall; 4],
        }
    }
}

/// Columns shallower than this are dry, m.
const DRY: f32 = 1e-3;

/// Strides of the split's diffusion. Each pass reaches further for the same
/// cost, so filtering out to several depths stays affordable.
const STRIDES: [usize; 4] = [1, 2, 4, 8];

/// Jeschke & Wojtan's gradient penalty δ = 1/100, in the form that fits
/// `split`: steep fronts stay in the bulk rather than exploding into ripples.
const STEEPNESS_PENALTY: f32 = 0.64;

/// Share of wave growth kept where the bulk flow converges (Jeschke & Wojtan's
/// γ); Airy waves cannot break, so full shoaling would overshoot.
const CONVERGENCE_GROWTH: f32 = 0.25;

/// Surface-wave height (m) and speed (m/s) below which the dispersive layer
/// is not worth its FFTs.
const QUIET: f32 = 1e-5;

/// Elements per task for passes doing a few flops each: they stream memory,
/// so splitting them finer costs more in hand-offs than it saves.
const STREAM: usize = 8 * GRAIN;

/// Cells over which the edge sponge ramps up.
const SPONGE: f32 = 4.0;

/// Surface-wave amplitude over depth above which waves break (McCowan: height
/// 0.78 of the depth).
const BREAKING_AMPLITUDE: f32 = 0.39;

/// What a Fourier mode needs that never changes.
struct Mode {
    /// Length of k; negative on the Nyquist axes, which are not propagated.
    length: f32,
    direction: Vec2,
    /// Half-cell shifts along x and y: e^(i k·x dx/2), e^(i k·y dx/2).
    shift: [Vec2; 2],
}

/// Buffers kept across substeps, because a fresh allocation this size costs
/// more in page faults than the arithmetic done with it.
#[derive(Default)]
struct Scratch {
    height: Vec<Vec2>,
    flux: Vec<Vec2>,
    /// Height, x-flux and y-flux coefficients at the origin.
    origin: Vec<[Vec2; 3]>,
    levels: Vec<(Vec<Vec2>, Vec<Vec2>)>,
    /// Spare fields, handed out and given back within a step.
    spare: Vec<Vec<f32>>,
}

impl Scratch {
    fn zeroed(&mut self, len: usize) -> Vec<f32> {
        let mut field = self.spare.pop().unwrap_or_default();
        field.clear();
        field.resize(len, 0.0);
        field
    }

    fn copy(&mut self, from: &[f32]) -> Vec<f32> {
        let mut field = self.spare.pop().unwrap_or_default();
        field.clear();
        field.extend_from_slice(from);
        field
    }

    fn give<const N: usize>(&mut self, fields: [Vec<f32>; N]) {
        self.spare.extend(fields);
    }
}

/// Shallow water over a square power-of-two [`Seabed`]. Depth and foam live
/// at cell centers; `u` on the (n+1)×n x-faces (face (i, j) at x = i·cell),
/// `v` on the n×(n+1) y-faces. [`ShallowWaterSettings::edges`] says what
/// crosses the domain edges, and [`relax_edges`](Self::relax_edges) can drive
/// them toward another water.
pub struct ShallowWater {
    n: usize,
    cell: f32,
    origin: Vec2,
    pub bed: Vec<f32>,
    pub depth: Vec<f32>,
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    /// Whitewater in [0, 1], made where flow is supercritical or the surface
    /// steep, carried with the flow.
    pub foam: Vec<f32>,
    /// Ground dampness in [0, 1]: 1 under water, drying after it recedes, so
    /// a beach shows the reach of the last waves.
    pub wetness: Vec<f32>,
    /// How fast each column's surface rises, m/s.
    rise: Vec<f32>,
    fft: Fft2,
    modes: Vec<Mode>,
    scratch: Scratch,
}

impl ShallowWater {
    /// Still water filled to `surface` Z over the seabed.
    pub fn new(seabed: &Seabed, surface: f32) -> Self {
        seabed.validate();
        let n = seabed.width;
        assert!(
            seabed.height == n && n.is_power_of_two() && n >= 4,
            "shallow water needs a square, power-of-two seabed"
        );
        let dk = TAU / (n as f32 * seabed.cell);
        let modes = (0..n * n)
            .map(|c| {
                let (i, j) = (c % n, c / n);
                let k = Vec2::new(signed(i, n), signed(j, n)) * dk;
                let nyquist = i == n / 2 || j == n / 2;
                Mode {
                    length: if nyquist { -1.0 } else { k.length() },
                    direction: k.normalize_or_zero(),
                    shift: [cis(k.x * seabed.cell / 2.0), cis(k.y * seabed.cell / 2.0)],
                }
            })
            .collect();
        Self {
            n,
            cell: seabed.cell,
            origin: seabed.origin,
            bed: seabed.elevation.clone(),
            depth: seabed
                .elevation
                .iter()
                .map(|e| (surface - e).max(0.0))
                .collect(),
            u: vec![0.0; (n + 1) * n],
            v: vec![0.0; n * (n + 1)],
            foam: vec![0.0; n * n],
            wetness: seabed
                .elevation
                .iter()
                .map(|e| if *e < surface { 1.0 } else { 0.0 })
                .collect(),
            rise: vec![0.0; n * n],
            fft: Fft2::new(n),
            modes,
            scratch: Scratch::default(),
        }
    }

    pub fn resolution(&self) -> usize {
        self.n
    }

    pub fn cell(&self) -> f32 {
        self.cell
    }

    pub fn origin(&self) -> Vec2 {
        self.origin
    }

    /// Total water, m³.
    pub fn volume(&self) -> f32 {
        self.depth.iter().sum::<f32>() * self.cell * self.cell
    }

    fn xu(&self, i: usize, j: usize) -> usize {
        i + j * (self.n + 1)
    }

    fn center(&self, i: usize, j: usize) -> Vec2 {
        self.origin + (Vec2::new(i as f32, j as f32) + 0.5) * self.cell
    }

    /// Advance `dt` seconds in as many stable substeps as needed.
    pub fn step(&mut self, dt: f32, settings: &ShallowWaterSettings) {
        assert!(
            dt.is_finite() && dt >= 0.0,
            "dt must be finite and nonnegative"
        );
        assert!(
            settings.courant > 0.0
                && settings.courant <= 1.0
                && settings.depth_levels > 0
                && settings.manning >= 0.0,
            "courant must be in (0, 1], depth levels positive, manning nonnegative"
        );
        if dt == 0.0 {
            return;
        }
        let mut scratch = std::mem::take(&mut self.scratch);
        if settings.dispersion && self.depth.par_iter().with_min_len(STREAM).any(|&d| d > DRY) {
            self.dispersive_step(dt, settings, &mut scratch);
        } else {
            let mut left = dt;
            while left > 0.0 {
                let h = self.substep(left, settings);
                let before = scratch.copy(&self.depth);
                self.bulk(h, settings, &mut scratch);
                self.settle(h, before, settings, &mut scratch);
                left -= h;
            }
        }
        self.scratch = scratch;
    }

    /// Length of the next stable bulk substep with `left` seconds to go.
    fn substep(&self, left: f32, settings: &ShallowWaterSettings) -> f32 {
        let fastest = |field: &[f32]| {
            field
                .par_iter()
                .with_min_len(STREAM)
                .map(|v| v.abs())
                .reduce(|| 0.0, f32::max)
        };
        let flow = fastest(&self.u).max(fastest(&self.v));
        let deepest = fastest(&self.depth);
        let limit = settings.courant * self.cell / (flow + (GRAVITY * deepest).sqrt()).max(1e-6);
        // Even shares, or the remainder pays for a whole substep of its own.
        left / (left / limit).ceil().max(1.0)
    }

    /// Jeschke & Wojtan 2023: bulk flow stays shallow water, the
    /// high-frequency remainder is propagated by Airy theory and carried by
    /// the bulk flow. Airy propagation is exact for any `dt` and transport is
    /// semi-Lagrangian, so only the bulk needs stable substeps.
    fn dispersive_step(&mut self, dt: f32, settings: &ShallowWaterSettings, scratch: &mut Scratch) {
        let before = scratch.copy(&self.depth);
        let (hb, ub, vb) = self.split(settings.split_iterations, scratch);
        // The surface layer is what the split took out, made in the old fields.
        let mut hs = std::mem::replace(&mut self.depth, hb);
        let mut us = std::mem::replace(&mut self.u, ub);
        let mut vs = std::mem::replace(&mut self.v, vb);
        let subtract = |s: &mut [f32], b: &[f32]| {
            s.par_iter_mut()
                .with_min_len(STREAM)
                .zip(b)
                .for_each(|(s, b)| *s -= b)
        };
        subtract(&mut hs, &self.depth);
        subtract(&mut us, &self.u);
        subtract(&mut vs, &self.v);
        let volume: f32 = hs.par_iter().with_min_len(STREAM).sum();
        let largest = hs
            .par_iter()
            .with_min_len(STREAM)
            .chain(&us)
            .chain(&vs)
            .map(|s| s.abs())
            .reduce(|| 0.0, f32::max);
        // Still lakes and steady rivers carry no surface waves: skip the FFTs.
        let waves = largest > QUIET;
        if waves {
            self.absorb_at_edges(&settings.edges, &mut hs, &mut us, &mut vs, dt);
            self.airy(
                scratch,
                &mut hs,
                &mut us,
                &mut vs,
                dt,
                settings.depth_levels,
            );
        }
        let mut left = dt;
        while left > 0.0 {
            let h = self.substep(left, settings);
            self.bulk(h, settings, scratch);
            left -= h;
        }
        if waves {
            self.transport(scratch, &mut hs, &mut us, &mut vs, dt);
        }
        hs.par_iter_mut()
            .with_min_len(STREAM)
            .zip(&self.depth)
            .for_each(|(s, h)| {
                let limit = BREAKING_AMPLITUDE * h;
                *s = s.clamp(-limit, limit);
            });
        // Blending levels, masking land, advection, breaking and the edge
        // sponge all move a little water; give it back.
        let wet = self
            .depth
            .par_iter()
            .with_min_len(STREAM)
            .filter(|&&h| h > DRY)
            .count();
        if wet > 0 {
            let fix = (volume - hs.par_iter().with_min_len(STREAM).sum::<f32>()) / wet as f32;
            hs.par_iter_mut()
                .with_min_len(STREAM)
                .zip(&self.depth)
                .filter(|(_, h)| **h > DRY)
                .for_each(|(s, _)| *s += fix);
        }
        self.depth
            .par_iter_mut()
            .with_min_len(STREAM)
            .zip(&hs)
            .for_each(|(h, s)| *h = (*h + s).max(0.0));
        let add = |f: &mut [f32], s: &[f32]| {
            f.par_iter_mut()
                .with_min_len(STREAM)
                .zip(s)
                .for_each(|(f, s)| *f += s)
        };
        add(&mut self.u, &us);
        add(&mut self.v, &vs);
        scratch.give([hs, us, vs]);
        self.settle(dt, before, settings, scratch);
    }

    /// The surface rise since `before`, then foam, after `dt` seconds.
    fn settle(
        &mut self,
        dt: f32,
        before: Vec<f32>,
        settings: &ShallowWaterSettings,
        scratch: &mut Scratch,
    ) {
        (&mut self.rise, &self.depth, &before)
            .into_par_iter()
            .with_min_len(STREAM)
            .for_each(|(rise, now, was)| *rise = (now - was) / dt);
        scratch.give([before]);
        self.update_foam(dt, settings, scratch);
    }

    /// Sponge over the cells nearest the edges water can leave by. The Airy
    /// step's FFT is periodic, so without it surface waves leaving one edge
    /// would come back in at the opposite one.
    /// shortcut: walls keep that wrap, since a sponge would stop them
    /// reflecting; mirror the domain in the FFT if waves must bounce off walls.
    fn absorb_at_edges(
        &self,
        edges: &[Edge; 4],
        hs: &mut [f32],
        us: &mut [f32],
        vs: &mut [f32],
        dt: f32,
    ) {
        let open = edges.map(|e| e != Edge::Wall);
        if !open.contains(&true) {
            return;
        }
        let n = self.n as f32;
        // Fastest a surface wave can travel, in cells per second.
        let deepest = self
            .depth
            .par_iter()
            .with_min_len(STREAM)
            .copied()
            .reduce(|| 0.0, f32::max);
        let speed = (GRAVITY * deepest).sqrt() / self.cell;
        let keep = |x: f32, y: f32| {
            let edge = [x, n - x, y, n - y]
                .into_iter()
                .zip(open)
                .filter_map(|(d, open)| open.then_some(d))
                .fold(f32::MAX, f32::min);
            (-(1.0 - edge / SPONGE).max(0.0) * speed * dt).exp()
        };
        let sponge = |field: &mut [f32], width: usize, offset: Vec2| {
            field
                .par_chunks_mut(width)
                .with_min_len(STREAM / width)
                .enumerate()
                .for_each(|(j, row)| {
                    for (i, s) in row.iter_mut().enumerate() {
                        *s *= keep(i as f32 + offset.x, j as f32 + offset.y);
                    }
                });
        };
        sponge(hs, self.n, Vec2::splat(0.5));
        sponge(us, self.n + 1, Vec2::new(0.0, 0.5));
        sponge(vs, self.n, Vec2::new(0.5, 0.0));
    }

    /// Smooth the free surface and velocities where water is deep, giving the
    /// bulk part. The filter's width follows the depth, so waves shorter than
    /// about 2π·depth, which shallow water would send too fast, become
    /// surface waves. Shallow and dry faces don't diffuse, so shorelines and
    /// nonlinear bores stay in the bulk, and still water over any bed splits
    /// into no surface waves at all.
    fn split(&self, iterations: usize, scratch: &mut Scratch) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = self.n;
        let mut eta = scratch.zeroed(n * n);
        (&mut eta, &self.bed, &self.depth)
            .into_par_iter()
            .with_min_len(STREAM)
            .for_each(|(e, b, h)| *e = b + h);
        // Filter variance wanted at each cell, in cells².
        let mut variance = scratch.zeroed(n * n);
        variance
            .par_iter_mut()
            .enumerate()
            .with_min_len(GRAIN)
            .for_each(|(c, out)| {
                let (i, j) = (c % n, c / n);
                let slope = Vec2::new(
                    eta[(i + 1).min(n - 1) + j * n] - eta[i.saturating_sub(1) + j * n],
                    eta[i + (j + 1).min(n - 1) * n] - eta[i + j.saturating_sub(1) * n],
                ) / (2.0 * self.cell);
                let depth = self.depth[c] / self.cell;
                *out = (depth * depth - STEEPNESS_PENALTY * slope.length_squared()).max(0.0);
            });
        let (fx, fy) = self.face_depths(&variance, f32::min, scratch);
        let (mut ub, mut vb) = (scratch.copy(&self.u), scratch.copy(&self.v));
        rayon::join(
            || diffuse(&mut eta, n, &variance, iterations),
            || {
                rayon::join(
                    || diffuse(&mut ub, n + 1, &fx, iterations),
                    || diffuse(&mut vb, n, &fy, iterations),
                )
            },
        );
        scratch.give([variance, fx, fy]);
        eta.par_iter_mut()
            .with_min_len(STREAM)
            .zip(&self.bed)
            .for_each(|(e, b)| *e = (*e - b).max(0.0));
        (eta, ub, vb)
    }

    /// Per-face depths combining the two cells each interior face separates;
    /// zero on the boundary.
    fn face_depths(
        &self,
        h: &[f32],
        combine: fn(f32, f32) -> f32,
        scratch: &mut Scratch,
    ) -> (Vec<f32>, Vec<f32>) {
        let n = self.n;
        let mut fx = scratch.zeroed((n + 1) * n);
        let mut fy = scratch.zeroed(n * (n + 1));
        fx.par_chunks_mut(n + 1)
            .with_min_len(STREAM / (n + 1))
            .enumerate()
            .for_each(|(j, row)| {
                for i in 1..n {
                    row[i] = combine(h[i - 1 + j * n], h[i + j * n]);
                }
            });
        fy[n..n * n]
            .par_chunks_mut(n)
            .with_min_len(STREAM / n)
            .enumerate()
            .for_each(|(row, out)| {
                let i = row + 1;
                for (j, out) in out.iter_mut().enumerate() {
                    *out = combine(h[j + (i - 1) * n], h[j + i * n]);
                }
            });
        (fx, fy)
    }

    /// Exact linear propagation of surface height and flux in Fourier space
    /// with ω² = g k tanh(k h), solved at a few depths and blended per cell
    /// by local bulk depth.
    /// Modes on either Nyquist axis are retained without propagation: their
    /// staggered-grid phase is ambiguous, but discarding them would change the
    /// state even in the limit of a zero-length step.
    fn airy(
        &self,
        scratch: &mut Scratch,
        hs: &mut [f32],
        us: &mut [f32],
        vs: &mut [f32],
        dt: f32,
        levels: usize,
    ) {
        let n = self.n;
        let hb = &self.depth;
        let (fx, fy) = self.face_depths(hb, |a, b| 0.5 * (a + b), scratch);
        let lo = hb
            .iter()
            .copied()
            .filter(|&h| h > DRY)
            .fold(f32::INFINITY, f32::min);
        let hi = hb.iter().copied().fold(0.0, f32::max);
        if hi <= DRY {
            scratch.give([fx, fy]);
            return;
        }
        let (mut out_h, mut out_qx, mut out_qy) = (
            scratch.zeroed(n * n),
            scratch.zeroed(n * n),
            scratch.zeroed(n * n),
        );

        let Scratch {
            height: hk,
            flux: qk,
            origin,
            levels: planes,
            ..
        } = scratch;
        hk.clear();
        hk.extend(hs.iter().map(|&h| Vec2::new(h, 0.0)));
        qk.clear();
        qk.extend((0..n * n).map(|c| {
            let x = self.xu(c % n, c / n);
            Vec2::new(us[x] * fx[x], vs[c] * fy[c])
        }));
        rayon::join(|| self.fft.run(hk, false), || self.fft.run(qk, false));
        let (hk, qk) = (&*hk, &*qk);
        // Coefficients of the continuous fields at the origin: heights sit at
        // cell centers, x-fluxes half a cell lower in y, y-fluxes in x.
        (0..n * n)
            .into_par_iter()
            .with_min_len(GRAIN)
            .map(|c| {
                let mode = &self.modes[c];
                if mode.length < 0.0 {
                    return [Vec2::ZERO; 3];
                }
                let [sx, sy] = mode.shift;
                let (p, m) = (qk[c], conj(qk[negated(c, n)]));
                let (qx, qy) = ((p + m) * 0.5, -times_i((p - m) * 0.5));
                [
                    mul(hk[c], conj(mul(sx, sy))),
                    mul(qx, conj(sy)),
                    mul(qy, conj(sx)),
                ]
            })
            .collect_into_vec(origin);
        let origin = &*origin;

        let count = if hi / lo < 1.01 { 1 } else { levels };
        // Fractional level of a depth; each level weighs in by its distance.
        let level = |d: f32| {
            if count == 1 {
                return 0.0;
            }
            (d.clamp(lo, hi) / lo).ln() / (hi / lo).ln() * (count - 1) as f32
        };
        let weight = |level: f32, l: usize| (1.0 - (level - l as f32).abs()).max(0.0);
        planes.resize_with(count, Default::default);
        planes.par_iter_mut().enumerate().for_each(|(l, (ph, pq))| {
            let depth = if count == 1 {
                hi
            } else {
                lo * (hi / lo).powf(l as f32 / (count - 1) as f32)
            };
            ph.resize(n * n, Vec2::ZERO);
            pq.resize(n * n, Vec2::ZERO);
            (&mut *ph, &mut *pq, &self.modes, origin, hk, qk)
                .into_par_iter()
                .with_min_len(GRAIN)
                .for_each(|(ph, pq, mode, &[h, qx, qy], &grid_h, &grid_q)| {
                    // Keep the original grid coefficients for ambiguous
                    // Nyquist modes rather than filtering them on every step.
                    if mode.length < 0.0 {
                        (*ph, *pq) = (grid_h, grid_q);
                        return;
                    }
                    let (h, qx, qy) = if mode.length > 0.0 {
                        let (kl, khat) = (mode.length, mode.direction);
                        let w = angular_frequency(kl, depth);
                        let ql = qx * khat.x + qy * khat.y;
                        let (s, c) = (w * dt).sin_cos();
                        let h2 = h * c - times_i(ql) * (kl / w * s);
                        let dq = ql * (c - 1.0) - times_i(h) * (w / kl * s);
                        (h2, qx + dq * khat.x, qy + dq * khat.y)
                    } else {
                        (h, qx, qy)
                    };
                    let [sx, sy] = mode.shift;
                    *ph = mul(h, mul(sx, sy));
                    *pq = mul(qx, sy) + times_i(mul(qy, sx));
                });
        });
        // Each level's height spectrum is Hermitian, so its surface is real:
        // pairs of levels share one transform as real and imaginary parts.
        planes.par_chunks_mut(2).for_each(|pair| {
            if let [(even, _), (odd, _)] = pair {
                even.par_iter_mut()
                    .with_min_len(STREAM)
                    .zip(&*odd)
                    .for_each(|(e, o)| *e += times_i(*o));
            }
        });
        planes
            .par_iter_mut()
            .enumerate()
            .flat_map_iter(|(l, (ph, pq))| {
                [(l % 2 == 0).then_some(ph), Some(pq)].into_iter().flatten()
            })
            .for_each(|field| self.fft.run(field, true));
        let levels = &*planes;
        let scale = 1.0 / (n * n) as f32;
        (&mut out_h, &mut out_qx, &mut out_qy, 0..n * n)
            .into_par_iter()
            .with_min_len(GRAIN)
            .for_each(|(h, qx, qy, c)| {
                let x = self.xu(c % n, c / n);
                let (th, tx, ty) = (level(hb[c]), level(fx[x]), level(fy[c]));
                for (l, (_, pq)) in levels.iter().enumerate() {
                    let packed = levels[l & !1].0[c];
                    let height = if l % 2 == 0 { packed.x } else { packed.y };
                    *h += weight(th, l) * height * scale;
                    *qx += weight(tx, l) * pq[c].x * scale;
                    *qy += weight(ty, l) * pq[c].y * scale;
                }
            });

        hs.par_iter_mut()
            .with_min_len(STREAM)
            .zip(&out_h)
            .zip(hb)
            .for_each(|((s, &o), &h)| *s = if h > DRY { o } else { 0.0 });
        us.par_chunks_mut(n + 1)
            .with_min_len(STREAM / (n + 1))
            .enumerate()
            .for_each(|(j, row)| {
                for (i, s) in row.iter_mut().enumerate() {
                    let x = i + j * (n + 1);
                    *s = if i > 0 && i < n && fx[x] > DRY {
                        out_qx[i + j * n] / fx[x]
                    } else {
                        0.0
                    };
                }
            });
        vs.par_chunks_mut(n)
            .with_min_len(STREAM / n)
            .enumerate()
            .for_each(|(i, row)| {
                for (j, s) in row.iter_mut().enumerate() {
                    let y = j + i * n;
                    *s = if i > 0 && i < n && fy[y] > DRY {
                        out_qy[y] / fy[y]
                    } else {
                        0.0
                    };
                }
            });
        scratch.give([fx, fy, out_h, out_qx, out_qy]);
    }

    /// Carry surface waves on the bulk flow (Jeschke & Wojtan's section 4.4):
    /// waves ride a current, and steepen or relax where it converges or
    /// diverges. Cubic, since linear interpolation would smear short waves
    /// away. Still water needs none of this and is left bit-exact.
    fn transport(
        &self,
        scratch: &mut Scratch,
        hs: &mut Vec<f32>,
        us: &mut [f32],
        vs: &mut [f32],
        dt: f32,
    ) {
        let (n, dx) = (self.n, self.cell);
        let flow = self
            .u
            .par_iter()
            .with_min_len(STREAM)
            .chain(&self.v)
            .map(|v| v.abs())
            .reduce(|| 0.0, f32::max);
        if flow * dt / dx < 1e-3 {
            return;
        }
        let mut growth = scratch.zeroed(n * n);
        growth
            .par_iter_mut()
            .with_min_len(STREAM)
            .enumerate()
            .for_each(|(c, out)| {
                *out = {
                    let (i, j) = (c % n, c / n);
                    let divergence = self.u[self.xu(i + 1, j)] - self.u[self.xu(i, j)]
                        + self.v[i + (j + 1) * n]
                        - self.v[i + j * n];
                    let rate = -divergence * dt / dx;
                    rate.min(CONVERGENCE_GROWTH * rate).exp()
                };
            });
        let back = |p: Vec2| p - self.velocity_at(p) * (dt / dx);
        let mut heights = scratch.copy(hs);
        heights
            .par_iter_mut()
            .with_min_len(STREAM)
            .zip(&growth)
            .for_each(|(h, g)| *h *= g);
        (0..n * n)
            .into_par_iter()
            .with_min_len(GRAIN)
            .map(|c| {
                let p = Vec2::new((c % n) as f32 + 0.5, (c / n) as f32 + 0.5);
                cubic(&heights, n, back(p) - 0.5)
            })
            .collect_into_vec(hs);
        let old = scratch.copy(us);
        us.par_chunks_mut(n + 1)
            .enumerate()
            .with_min_len(GRAIN / n)
            .for_each(|(j, row)| {
                for (i, out) in row.iter_mut().enumerate().take(n).skip(1) {
                    let at = back(Vec2::new(i as f32, j as f32 + 0.5)) - Vec2::new(0.0, 0.5);
                    let g = 0.5 * (growth[i - 1 + j * n] + growth[i + j * n]);
                    *out = g * cubic(&old, n + 1, at);
                }
            });
        scratch.give([heights, old]);
        let old = scratch.copy(vs);
        vs[n..n * n]
            .par_chunks_mut(n)
            .enumerate()
            .with_min_len(GRAIN / n)
            .for_each(|(row, out)| {
                let j = row + 1;
                for (i, out) in out.iter_mut().enumerate() {
                    let at = back(Vec2::new(i as f32 + 0.5, j as f32)) - Vec2::new(0.5, 0.0);
                    let g = 0.5 * (growth[i + (j - 1) * n] + growth[i + j * n]);
                    *out = g * cubic(&old, n, at);
                }
            });
        scratch.give([growth, old]);
    }

    /// One shallow-water step: semi-Lagrangian momentum advection, surface
    /// pressure, Manning friction, then upwind fluxes limited so no cell
    /// drains below empty (Stelling & Duinmeijer 2003). Mass is conserved
    /// exactly and still water over any bed stays still.
    fn bulk(&mut self, dt: f32, settings: &ShallowWaterSettings, scratch: &mut Scratch) {
        let manning = settings.manning;
        let (n, dx) = (self.n, self.cell);
        let Self {
            bed, depth, u, v, ..
        } = self;
        let (u0, v0) = (scratch.copy(u), scratch.copy(v));
        let mut eta = scratch.zeroed(n * n);
        (&mut eta, &*bed, &*depth)
            .into_par_iter()
            .with_min_len(STREAM)
            .for_each(|(e, b, h)| *e = b + h);
        let limit = 0.5 * dx / dt;
        // Pressure and friction on an advected face velocity.
        let accelerate = |vel: f32, l: usize, r: usize| {
            // Water can only cross a face where it stands above both beds.
            let hf = eta[l].max(eta[r]) - bed[l].max(bed[r]);
            if hf <= DRY {
                return 0.0;
            }
            let w = vel - GRAVITY * dt * (eta[r] - eta[l]) / dx;
            let w = w / (1.0 + dt * GRAVITY * manning * manning * w.abs() / hf.powf(4.0 / 3.0));
            w.clamp(-limit, limit)
        };
        let upwind = |vel: f32, l: usize, r: usize| vel * depth[if vel > 0.0 { l } else { r }];
        let (mut fu, mut fv) = (scratch.zeroed(u.len()), scratch.zeroed(v.len()));
        u.par_chunks_mut(n + 1)
            .zip(fu.par_chunks_mut(n + 1))
            .enumerate()
            .with_min_len(GRAIN / n)
            .for_each(|(j, (u, f))| {
                for i in 1..n {
                    let p = Vec2::new(i as f32, j as f32 + 0.5);
                    let back = p - Vec2::new(u0[i + j * (n + 1)], sample_v(&v0, n, p)) * (dt / dx);
                    let (l, r) = (i - 1 + j * n, i + j * n);
                    u[i] = accelerate(sample_u(&u0, n, back), l, r);
                    f[i] = upwind(u[i], l, r);
                }
            });
        // The first and last rows of v are the walls.
        v[n..n * n]
            .par_chunks_mut(n)
            .zip(fv[n..n * n].par_chunks_mut(n))
            .enumerate()
            .with_min_len(GRAIN / n)
            .for_each(|(row, (v, f))| {
                let i = row + 1;
                for j in 0..n {
                    let p = Vec2::new(j as f32 + 0.5, i as f32);
                    let back = p - Vec2::new(sample_u(&u0, n, p), v0[j + i * n]) * (dt / dx);
                    let (l, r) = (j + (i - 1) * n, j + i * n);
                    v[j] = accelerate(sample_v(&v0, n, back), l, r);
                    f[j] = upwind(v[j], l, r);
                }
            });
        for (side, edge) in settings.edges.iter().enumerate() {
            for t in 0..n {
                // Velocities and fluxes, the edge face, the face just inside
                // it, the cell it bounds, and which way is inward.
                let (velocity, flux, face, inner, c, inward) = match side {
                    0 => (&mut *u, &mut fu, t * (n + 1), t * (n + 1) + 1, t * n, 1.0),
                    1 => {
                        let face = n + t * (n + 1);
                        (&mut *u, &mut fu, face, face - 1, n - 1 + t * n, -1.0)
                    }
                    2 => (&mut *v, &mut fv, t, t + n, t, 1.0),
                    _ => {
                        let c = t + (n - 1) * n;
                        (&mut *v, &mut fv, t + n * n, c, c, -1.0)
                    }
                };
                let along = (t as f32 + 0.5) * dx;
                let (speed, through) = match *edge {
                    Edge::Wall => (0.0, 0.0),
                    Edge::Open => {
                        let outward = (-inward * velocity[inner]).max(0.0);
                        (-inward * outward, -inward * outward * depth[c])
                    }
                    Edge::Discharge { rate, span } if (span[0]..span[1]).contains(&along) => {
                        let q = inward * rate / (span[1] - span[0]);
                        ((q / depth[c].max(0.05)).clamp(-limit, limit), q)
                    }
                    Edge::Discharge { .. } => (0.0, 0.0),
                };
                velocity[face] = speed;
                flux[face] = through;
            }
        }
        // Shrink what leaves each cell so that none drains below empty.
        let mut scale = scratch.zeroed(n * n);
        scale
            .par_iter_mut()
            .enumerate()
            .with_min_len(GRAIN)
            .for_each(|(c, scale)| {
                let (i, j) = (c % n, c / n);
                let out = fu[i + 1 + j * (n + 1)].max(0.0)
                    + (-fu[i + j * (n + 1)]).max(0.0)
                    + fv[i + (j + 1) * n].max(0.0)
                    + (-fv[i + j * n]).max(0.0);
                *scale = if out * dt / dx > depth[c] {
                    depth[c] / (out * dt / dx)
                } else {
                    1.0
                };
            });
        fu.par_chunks_mut(n + 1)
            .enumerate()
            .with_min_len(GRAIN / n)
            .for_each(|(j, row)| {
                for i in 1..n {
                    row[i] *= scale[if row[i] > 0.0 {
                        i - 1 + j * n
                    } else {
                        i + j * n
                    }];
                }
            });
        fv[n..n * n]
            .par_chunks_mut(n)
            .enumerate()
            .with_min_len(GRAIN / n)
            .for_each(|(row, f)| {
                let i = row + 1;
                for j in 0..n {
                    f[j] *= scale[if f[j] > 0.0 {
                        j + (i - 1) * n
                    } else {
                        j + i * n
                    }];
                }
            });
        // Outflow through the domain's edges obeys the same limit.
        for t in 0..n {
            let west = t * (n + 1);
            fu[west] *= if fu[west] < 0.0 { scale[t * n] } else { 1.0 };
            let east = west + n;
            fu[east] *= if fu[east] > 0.0 {
                scale[n - 1 + t * n]
            } else {
                1.0
            };
            fv[t] *= if fv[t] < 0.0 { scale[t] } else { 1.0 };
            let north = t + n * n;
            fv[north] *= if fv[north] > 0.0 {
                scale[t + (n - 1) * n]
            } else {
                1.0
            };
        }
        depth
            .par_chunks_mut(n)
            .enumerate()
            .with_min_len(GRAIN / n)
            .for_each(|(j, row)| {
                for i in 0..n {
                    let divergence = fu[i + 1 + j * (n + 1)] - fu[i + j * (n + 1)]
                        + fv[i + (j + 1) * n]
                        - fv[i + j * n];
                    row[i] = (row[i] - dt / dx * divergence).max(0.0);
                }
            });
        scratch.give([u0, v0, eta, fu, fv, scale]);
    }

    fn update_foam(&mut self, dt: f32, settings: &ShallowWaterSettings, scratch: &mut Scratch) {
        let (n, dx) = (self.n, self.cell);
        let fade = (-dt / settings.foam_lifetime.max(f32::MIN_POSITIVE)).exp();
        let drying = (-dt / settings.wetness_lifetime.max(f32::MIN_POSITIVE)).exp();
        let eta = |i: usize, j: usize| self.bed[i + j * n] + self.depth[i + j * n];
        let mut foam = scratch.copy(&self.foam);
        let mut wetness = std::mem::take(&mut self.wetness);
        foam.par_chunks_mut(n)
            .zip(wetness.par_chunks_mut(n))
            .enumerate()
            .with_min_len(GRAIN / n)
            .for_each(|(j, (row, wet))| {
                for (i, (foam, wet)) in row.iter_mut().zip(wet).enumerate() {
                    let c = i + j * n;
                    if self.depth[c] <= DRY {
                        *foam = 0.0;
                        *wet *= drying;
                        continue;
                    }
                    *wet = 1.0;
                    let flow = self.velocity_at(Vec2::new(i as f32 + 0.5, j as f32 + 0.5));
                    let celerity = (GRAVITY * self.depth[c]).sqrt();
                    let slope = Vec2::new(
                        eta((i + 1).min(n - 1), j) - eta(i.saturating_sub(1), j),
                        eta(i, (j + 1).min(n - 1)) - eta(i, j.saturating_sub(1)),
                    ) / (2.0 * dx);
                    // Breaking onset (Bacigaluppi et al. 2019): a surface that
                    // rises faster than 0.35 to 0.65 of the wave speed or is
                    // too steep to hold, and for steady hydraulic jumps, the
                    // Froude number.
                    let source = ((flow.length() / celerity - 0.8) / 0.5)
                        .max((slope.length() - 0.25) / 0.25)
                        .max((self.rise[c] / celerity - 0.35) / 0.3)
                        .clamp(0.0, 1.0);
                    *foam = (*foam * fade).max(source);
                }
            });
        self.wetness = wetness;
        let mut advected = std::mem::take(&mut self.foam);
        advected
            .par_chunks_mut(n)
            .enumerate()
            .with_min_len(GRAIN / n)
            .for_each(|(j, row)| {
                for (i, out) in row.iter_mut().enumerate() {
                    let p = Vec2::new(i as f32 + 0.5, j as f32 + 0.5);
                    let back = p - self.velocity_at(p) * (dt / dx);
                    *out = bilinear(&foam, n, back - 0.5, false);
                }
            });
        self.foam = advected;
        scratch.give([foam]);
    }

    /// Flow velocity at a point in cell units.
    fn velocity_at(&self, p: Vec2) -> Vec2 {
        Vec2::new(sample_u(&self.u, self.n, p), sample_v(&self.v, self.n, p))
    }

    /// Ground dampness at a world point, in [0, 1].
    pub fn wetness_at(&self, point: Vec2) -> f32 {
        bilinear(
            &self.wetness,
            self.n,
            (point - self.origin) / self.cell - 0.5,
            false,
        )
    }

    /// Whitewater at a world point.
    pub fn foam_at(&self, point: Vec2) -> f32 {
        bilinear(
            &self.foam,
            self.n,
            (point - self.origin) / self.cell - 0.5,
            false,
        )
    }

    /// Add `volume` m³ of water over a disc, or drain it when negative.
    pub fn add_water(&mut self, center: Vec2, radius: f32, volume: f32) {
        let weights: Vec<(usize, f32)> = self.disc(center, radius).collect();
        let total: f32 = weights.iter().map(|(_, w)| w).sum();
        if total == 0.0 {
            return;
        }
        for (c, w) in weights {
            self.depth[c] = (self.depth[c] + volume * w / (total * self.cell * self.cell)).max(0.0);
        }
    }

    /// Pull the flow within a disc toward `velocity`; `strength` 1 sets it.
    /// Paddles, boat wakes and river currents.
    pub fn push(&mut self, center: Vec2, radius: f32, velocity: Vec2, strength: f32) {
        let n = self.n;
        for (c, w) in self.disc(center, radius).collect::<Vec<_>>() {
            let (i, j) = (c % n, c / n);
            let a = (strength * w).clamp(0.0, 1.0);
            for x in [self.xu(i, j), self.xu(i + 1, j)] {
                self.u[x] += (velocity.x - self.u[x]) * a;
            }
            for y in [i + j * n, i + (j + 1) * n] {
                self.v[y] += (velocity.y - self.v[y]) * a;
            }
        }
    }

    /// Relaxation zone (Mayer et al. 1998): within `width` metres of the
    /// domain edges, blend the surface and flow toward `target`. This
    /// generates the target's waves and absorbs outgoing ones; with an
    /// [`Ocean`](crate::Ocean) or [`ShoreWaves`](crate::ShoreWaves) target it
    /// blends the open sea into the simulation. Call once per step.
    pub fn relax_edges(&mut self, target: &(impl Water + Sync), width: f32) {
        let (n, cell) = (self.n, self.cell);
        // Blend weight at a point in cell units.
        let ramp = |x: f32, y: f32| {
            let edge = x.min(y).min(n as f32 - x).min(n as f32 - y) * cell;
            (1.0 - edge / width).max(0.0).powi(2)
        };
        // The target, sampled once per cell inside the zone: the costly part.
        let samples: Vec<Option<WaterSample>> = (0..n * n)
            .into_par_iter()
            .with_min_len(GRAIN / 8)
            .map(|c| {
                let (i, j) = (c % n, c / n);
                (ramp(i as f32 + 0.5, j as f32 + 0.5) > 0.0)
                    .then(|| target.sample(self.center(i, j)))
            })
            .collect();
        let bed = &self.bed;
        self.depth
            .par_iter_mut()
            .enumerate()
            .with_min_len(STREAM)
            .for_each(|(c, depth)| {
                if let Some(s) = samples[c] {
                    let w = ramp((c % n) as f32 + 0.5, (c / n) as f32 + 0.5);
                    *depth += ((s.height - bed[c]).max(0.0) - *depth) * w;
                }
            });
        // Each face is pulled by the cells on either side, lower one first.
        // shortcut: drives depth-averaged flow with the target's surface
        // velocity, which overstates it in deep water.
        self.u
            .par_chunks_mut(n + 1)
            .enumerate()
            .with_min_len(STREAM / (n + 1))
            .for_each(|(j, row)| {
                for (i, u) in row.iter_mut().enumerate() {
                    let w = ramp(i as f32, j as f32 + 0.5);
                    for c in [i.checked_sub(1), (i < n).then_some(i)]
                        .into_iter()
                        .flatten()
                    {
                        if let Some(s) = samples[c + j * n] {
                            *u += (s.velocity.x - *u) * w;
                        }
                    }
                }
            });
        self.v
            .par_chunks_mut(n)
            .enumerate()
            .with_min_len(STREAM / n)
            .for_each(|(j, row)| {
                for (i, v) in row.iter_mut().enumerate() {
                    let w = ramp(i as f32 + 0.5, j as f32);
                    for r in [j.checked_sub(1), (j < n).then_some(j)]
                        .into_iter()
                        .flatten()
                    {
                        if let Some(s) = samples[i + r * n] {
                            *v += (s.velocity.y - *v) * w;
                        }
                    }
                }
            });
    }

    /// Cells within a disc with smooth falloff weights.
    fn disc(&self, center: Vec2, radius: f32) -> impl Iterator<Item = (usize, f32)> + '_ {
        let (n, radius) = (self.n, radius.max(self.cell * 0.5));
        // Only the cells under the disc's bounding square.
        let cell =
            |p: Vec2| ((p - self.origin) / self.cell).clamp(Vec2::ZERO, Vec2::splat(n as f32));
        let (lo, hi) = (cell(center - radius).floor(), cell(center + radius).ceil());
        (lo.y as usize..hi.y as usize)
            .flat_map(move |j| (lo.x as usize..hi.x as usize).map(move |i| i + j * n))
            .filter_map(move |c| {
                let d = self.center(c % n, c / n).distance(center) / radius;
                (d < 1.0).then(|| (c, (1.0 - d * d).powi(2)))
            })
    }
}

impl Water for ShallowWater {
    /// Over dry ground the surface is the bed.
    fn sample(&self, point: Vec2) -> WaterSample {
        let p = (point - self.origin) / self.cell;
        let q = p - 0.5;
        WaterSample {
            height: bilinear(&self.bed, self.n, q, false) + bilinear(&self.depth, self.n, q, false),
            velocity: self
                .velocity_at(p)
                .extend(bilinear(&self.rise, self.n, q, false)),
        }
    }
}

/// x-face velocity at `p` in cell units.
fn sample_u(u: &[f32], n: usize, p: Vec2) -> f32 {
    bilinear(u, n + 1, Vec2::new(p.x, p.y - 0.5), false)
}

fn sample_v(v: &[f32], n: usize, p: Vec2) -> f32 {
    bilinear(v, n, Vec2::new(p.x - 0.5, p.y), false)
}

/// Catmull-Rom sample of a row-major grid, kept within the four surrounding
/// values so it cannot overshoot. Coordinates clamp at the edges.
fn cubic(grid: &[f32], width: usize, q: Vec2) -> f32 {
    let rows = (grid.len() / width) as i64;
    let base = q.floor();
    let f = q - base;
    let (x, y) = (base.x as i64, base.y as i64);
    let at = |dx: i64, dy: i64| {
        let (x, y) = (
            (x + dx).clamp(0, width as i64 - 1),
            (y + dy).clamp(0, rows - 1),
        );
        grid[x as usize + y as usize * width]
    };
    let spline = |a: f32, b: f32, c: f32, d: f32, t: f32| {
        b + 0.5 * t * (c - a + t * (2.0 * a - 5.0 * b + 4.0 * c - d + t * (3.0 * (b - c) + d - a)))
    };
    let row = |dy: i64| spline(at(-1, dy), at(0, dy), at(1, dy), at(2, dy), f.x);
    let value = spline(row(-1), row(0), row(1), row(2), f.y);
    let corners = [at(0, 0), at(1, 0), at(0, 1), at(1, 1)];
    let (lo, hi) = corners
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &c| (lo.min(c), hi.max(c)));
    value.clamp(lo, hi)
}

/// A per-cell table, with zero where there is no pair.
fn pairs(len: usize, value: impl Fn(usize) -> Option<f32> + Sync) -> Vec<f32> {
    (0..len)
        .into_par_iter()
        .with_min_len(GRAIN)
        .map(|i| value(i).unwrap_or(0.0))
        .collect()
}

/// Depth-aware Gaussian filter by explicit diffusion, in [`STRIDES`]: each
/// stride takes an equal share of the wanted `variance` (cells², per cell),
/// and the widest strides reach several times further than plain neighbors
/// could for the same passes. Pair weights are symmetric, so the field's sum
/// is conserved, and stable since they total ≤ 1. A pair is bounded by the
/// lowest variance between its cells, so nothing diffuses across dry land.
fn diffuse(field: &mut Vec<f32>, width: usize, variance: &[f32], iterations: usize) {
    let len = field.len();
    let rows = len / width;
    let passes = (iterations / STRIDES.len()).max(1);
    // Lowest variance from each cell to its partner along x and along y.
    let mut along_x = pairs(len, |i| {
        (i % width + 1 < width).then(|| variance[i].min(variance[i + 1]))
    });
    let mut along_y = pairs(len, |i| {
        (i + width < len).then(|| variance[i].min(variance[i + width]))
    });
    let mut next = field.clone();
    for (index, &stride) in STRIDES.iter().enumerate() {
        if index > 0 {
            // Doubling: a pair spans two pairs of the previous stride.
            let half = stride / 2;
            along_x = pairs(len, |i| {
                (i % width + stride < width).then(|| along_x[i].min(along_x[i + half]))
            });
            along_y = pairs(len, |i| {
                (i + stride * width < len).then(|| along_y[i].min(along_y[i + half * width]))
            });
        }
        let share = 1.0 / (2.0 * (passes * stride * stride * STRIDES.len()) as f32);
        let weight = |v: f32| (v * share).min(0.25);
        let reach = stride * width;
        // The domain edge mirrors the field: a pair that would cross it folds
        // back onto a cell inside, or edge cells would be filtered less than
        // the rest and their waves would keep the shallow-water speed.
        let mut folded: Vec<(usize, usize, f32)> = vec![];
        let mut fold = |a: usize, b: usize, step: usize| {
            let lowest = (a..=b)
                .step_by(step)
                .map(|i| variance[i])
                .fold(f32::MAX, f32::min);
            if lowest > 0.0 {
                folded.push((a, b, weight(lowest)));
            }
        };
        for t in 0..stride / 2 {
            if stride < width {
                for row in (0..len).step_by(width) {
                    fold(row + t, row + stride - 1 - t, 1);
                    fold(row + width - stride + t, row + width - 1 - t, 1);
                }
            }
            if stride < rows {
                for x in 0..width {
                    fold(x + t * width, x + (stride - 1 - t) * width, width);
                    let top = x + (rows - 1) * width;
                    fold(top - (stride - 1 - t) * width, top - t * width, width);
                }
            }
        }
        for _ in 0..passes {
            let source = &*field;
            next.par_chunks_mut(width)
                .enumerate()
                .with_min_len(GRAIN / width)
                .for_each(|(y, out)| {
                    let row = y * width;
                    let below = row.checked_sub(reach).unwrap_or(row);
                    let above = if row + reach < len { row + reach } else { row };
                    let center = &source[row..row + width];
                    // Rows past the edge pair with themselves: a zero difference.
                    let (down, up) = (&source[below..below + width], &source[above..above + width]);
                    // Weights come from the pair variances inline: a table
                    // of them costs a pass of its own.
                    let (w_down, w_up) =
                        (&along_y[below..below + width], &along_y[row..row + width]);
                    for (((((out, &c), &d), &u), &wd), &wu) in out
                        .iter_mut()
                        .zip(center)
                        .zip(down)
                        .zip(up)
                        .zip(w_down)
                        .zip(w_up)
                    {
                        *out = c + weight(wd) * (d - c) + weight(wu) * (u - c);
                    }
                    let span = width.saturating_sub(stride);
                    let w_right = &along_x[row..row + span];
                    let (near, far) = (&center[..span], &center[width - span..]);
                    for (((out, &w), &a), &b) in out.iter_mut().zip(w_right).zip(near).zip(far) {
                        *out += weight(w) * (b - a);
                    }
                    let shifted = &mut out[width - span..];
                    for (((out, &w), &a), &b) in shifted.iter_mut().zip(w_right).zip(near).zip(far)
                    {
                        *out -= weight(w) * (b - a);
                    }
                });
            for &(a, b, w) in &folded {
                let t = w * (field[b] - field[a]);
                next[a] += t;
                next[b] -= t;
            }
            std::mem::swap(field, &mut next);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diffusion_filters_by_depth_and_conserves_the_sum() {
        // 65 wide, 64 tall: non-square, like the u field.
        let (width, rows) = (65, 64);
        let start: Vec<f32> = (0..width * rows)
            .map(|i| (TAU * 10.0 * ((i % width) as f32 + 0.5) / width as f32).cos())
            .collect();
        let filtered = |variance: f32| {
            let mut field = start.clone();
            diffuse(&mut field, width, &vec![variance; start.len()], 32);
            field
        };
        let deep = filtered(100.0);
        let total = |f: &[f32]| f.iter().sum::<f32>();
        assert!((total(&deep) - total(&start)).abs() < 0.1);
        let swing = |f: &[f32]| f.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(swing(&deep) < 0.01 * swing(&start), "{}", swing(&deep));
        assert_eq!(filtered(0.0), start, "no depth, no filtering");
    }

    #[test]
    fn diffusion_does_not_leak_across_a_dry_strip() {
        let n = 32;
        let mut field = vec![0.0; n * n];
        let mut variance = vec![400.0; n * n];
        for y in 0..n {
            for x in 0..n {
                field[x + y * n] = if x < 15 { 1.0 } else { 0.0 };
                variance[x + y * n] = if x == 15 { 0.0 } else { 400.0 };
            }
        }
        diffuse(&mut field, n, &variance, 32);
        assert!(
            field
                .iter()
                .enumerate()
                .all(|(i, v)| i % n <= 15 || *v == 0.0)
        );
    }

    #[test]
    fn cubic_is_exact_on_linear_fields_and_cannot_overshoot() {
        let grid: Vec<f32> = (0..64)
            .map(|i| (i % 8) as f32 * 2.0 + (i / 8) as f32)
            .collect();
        assert!((cubic(&grid, 8, Vec2::new(3.3, 4.6)) - (6.6 + 4.6)).abs() < 1e-4);
        let step: Vec<f32> = (0..64).map(|i| if i % 8 < 4 { 0.0 } else { 1.0 }).collect();
        for t in 0..20 {
            let v = cubic(&step, 8, Vec2::new(t as f32 * 0.35, 3.0));
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn dispersive_step_preserves_short_waves_as_dt_approaches_zero() {
        let n = 8;
        let seabed = Seabed::flat(n, n, 1.0, 0.0);
        let settings = ShallowWaterSettings {
            manning: 0.0,
            split_iterations: 1,
            ..Default::default()
        };
        // Exercise each Nyquist axis and their intersection.
        for (x_wave, y_wave) in [(true, false), (false, true), (true, true)] {
            let mut water = ShallowWater::new(&seabed, 1.0);
            for j in 0..n {
                for i in 0..n {
                    let parity = (if x_wave { i } else { 0 }) + (if y_wave { j } else { 0 });
                    water.depth[i + j * n] += if parity % 2 == 0 { 0.1 } else { -0.1 };
                }
            }
            let before = water.depth.clone();
            let volume = water.volume();
            water.step(1e-6, &settings);
            for (actual, initial) in water.depth.iter().zip(&before) {
                assert!(
                    (actual - initial).abs() < 1e-4,
                    "{actual} changed from {initial}"
                );
            }
            assert!((water.volume() - volume).abs() < 1e-4);
        }
    }
}
