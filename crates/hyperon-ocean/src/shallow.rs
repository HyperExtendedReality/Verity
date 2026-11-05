use crate::fft::{Fft2, cis, conj, mul, negated, signed, times_i};
use crate::{GRAVITY, Seabed, Water, WaterSample, angular_frequency, bilinear};
use glam::Vec2;
use std::f32::consts::TAU;

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
    /// longer waves into the dispersive layer.
    pub split_iterations: usize,
    /// Courant number bounding substeps, in (0, 1].
    pub courant: f32,
    /// Seconds for foam to fade to 1/e.
    pub foam_lifetime: f32,
}

impl Default for ShallowWaterSettings {
    fn default() -> Self {
        Self {
            manning: 0.02,
            dispersion: true,
            depth_levels: 4,
            split_iterations: 32,
            courant: 0.4,
            foam_lifetime: 3.0,
        }
    }
}

/// Columns shallower than this are dry, m.
const DRY: f32 = 1e-3;

/// Shallow water over a square power-of-two [`Seabed`]. Depth and foam live
/// at cell centers; `u` on the (n+1)×n x-faces (face (i, j) at x = i·cell),
/// `v` on the n×(n+1) y-faces. Domain edges are walls unless
/// [`relax_edges`](Self::relax_edges) drives them.
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
    fft: Fft2,
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
            fft: Fft2::new(n),
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
        let mut left = dt;
        while left > 0.0 {
            let flow = self
                .u
                .iter()
                .chain(&self.v)
                .fold(0.0f32, |m, v| m.max(v.abs()));
            let deepest = self.depth.iter().fold(0.0f32, |m, &d| m.max(d));
            let limit =
                settings.courant * self.cell / (flow + (GRAVITY * deepest).sqrt()).max(1e-6);
            let h = if limit >= left { left } else { limit };
            self.substep(h, settings);
            left -= h;
        }
    }

    fn substep(&mut self, dt: f32, settings: &ShallowWaterSettings) {
        if settings.dispersion && self.depth.iter().any(|&d| d > DRY) {
            // Jeschke & Wojtan 2023: bulk flow stays shallow water, the
            // high-frequency remainder is propagated by Airy theory.
            let (hb, ub, vb) = self.split(settings.split_iterations);
            let mut hs: Vec<f32> = self.depth.iter().zip(&hb).map(|(h, b)| h - b).collect();
            let mut us: Vec<f32> = self.u.iter().zip(&ub).map(|(u, b)| u - b).collect();
            let mut vs: Vec<f32> = self.v.iter().zip(&vb).map(|(v, b)| v - b).collect();
            (self.depth, self.u, self.v) = (hb, ub, vb);
            self.airy(&mut hs, &mut us, &mut vs, dt, settings.depth_levels);
            self.bulk(dt, settings.manning);
            self.depth
                .iter_mut()
                .zip(&hs)
                .for_each(|(h, s)| *h = (*h + s).max(0.0));
            self.u.iter_mut().zip(&us).for_each(|(u, s)| *u += s);
            self.v.iter_mut().zip(&vs).for_each(|(v, s)| *v += s);
        } else {
            self.bulk(dt, settings.manning);
        }
        self.update_foam(dt, settings.foam_lifetime);
    }

    /// Smooth the free surface and velocities where water is deep, giving the
    /// bulk part. Shallow and dry faces don't diffuse, so shorelines and
    /// nonlinear bores stay in the bulk, and still water over any bed splits
    /// into no surface waves at all.
    fn split(&self, iterations: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = self.n;
        let mut eta: Vec<f32> = self
            .bed
            .iter()
            .zip(&self.depth)
            .map(|(b, h)| b + h)
            .collect();
        diffuse(&mut eta, n, &self.depth, self.cell, iterations);
        let hb = eta
            .iter()
            .zip(&self.bed)
            .map(|(e, b)| (e - b).max(0.0))
            .collect();
        let (fx, fy) = self.face_depths(&self.depth, f32::min);
        let (mut ub, mut vb) = (self.u.clone(), self.v.clone());
        diffuse(&mut ub, n + 1, &fx, self.cell, iterations);
        diffuse(&mut vb, n, &fy, self.cell, iterations);
        (hb, ub, vb)
    }

    /// Per-face depths combining the two cells each interior face separates;
    /// zero on the boundary.
    fn face_depths(&self, h: &[f32], combine: fn(f32, f32) -> f32) -> (Vec<f32>, Vec<f32>) {
        let n = self.n;
        let mut fx = vec![0.0; (n + 1) * n];
        let mut fy = vec![0.0; n * (n + 1)];
        for j in 0..n {
            for i in 1..n {
                fx[self.xu(i, j)] = combine(h[i - 1 + j * n], h[i + j * n]);
                fy[j + i * n] = combine(h[j + (i - 1) * n], h[j + i * n]);
            }
        }
        (fx, fy)
    }

    /// Exact linear propagation of surface height and flux in Fourier space
    /// with ω² = g k tanh(k h), solved at a few depths and blended per cell
    /// by local bulk depth.
    /// Modes on either Nyquist axis are retained without propagation: their
    /// staggered-grid phase is ambiguous, but discarding them would change the
    /// state even in the limit of a zero-length step.
    fn airy(&self, hs: &mut [f32], us: &mut [f32], vs: &mut [f32], dt: f32, levels: usize) {
        let (n, dx) = (self.n, self.cell);
        let hb = &self.depth;
        let (fx, fy) = self.face_depths(hb, |a, b| 0.5 * (a + b));
        let wet: Vec<f32> = hb.iter().copied().filter(|&h| h > DRY).collect();
        let lo = wet.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = wet.iter().copied().fold(0.0, f32::max);
        if wet.is_empty() {
            return;
        }
        let before: f32 = hs.iter().sum();

        let mut hk: Vec<Vec2> = hs.iter().map(|&h| Vec2::new(h, 0.0)).collect();
        let mut qk: Vec<Vec2> = (0..n * n)
            .map(|c| {
                let x = self.xu(c % n, c / n);
                Vec2::new(us[x] * fx[x], vs[c] * fy[c])
            })
            .collect();
        self.fft.run(&mut hk, false);
        self.fft.run(&mut qk, false);
        // Coefficients of the continuous fields at the origin: heights sit at
        // cell centers, x-fluxes half a cell lower in y, y-fluxes in x.
        let dk = TAU / (n as f32 * dx);
        let modes: Vec<[Vec2; 4]> = (0..n * n)
            .map(|c| {
                let (i, j) = (c % n, c / n);
                if i == n / 2 || j == n / 2 {
                    return [Vec2::ZERO; 4];
                }
                let k = Vec2::new(signed(i, n), signed(j, n)) * dk;
                let (p, m) = (qk[c], conj(qk[negated(c, n)]));
                let (qx, qy) = ((p + m) * 0.5, -times_i((p - m) * 0.5));
                [
                    k,
                    mul(hk[c], cis(-(k.x + k.y) * dx / 2.0)),
                    mul(qx, cis(-k.y * dx / 2.0)),
                    mul(qy, cis(-k.x * dx / 2.0)),
                ]
            })
            .collect();

        let count = if hi / lo < 1.01 { 1 } else { levels };
        let weight = |d: f32, l: usize| {
            if count == 1 {
                return 1.0;
            }
            let t = (d.clamp(lo, hi) / lo).ln() / (hi / lo).ln() * (count - 1) as f32;
            (1.0 - (t - l as f32).abs()).max(0.0)
        };
        let (mut out_h, mut out_qx, mut out_qy) =
            (vec![0.0; n * n], vec![0.0; n * n], vec![0.0; n * n]);
        let scale = 1.0 / (n * n) as f32;
        for l in 0..count {
            let depth = if count == 1 {
                hi
            } else {
                lo * (hi / lo).powf(l as f32 / (count - 1) as f32)
            };
            let (mut ph, mut pq): (Vec<Vec2>, Vec<Vec2>) = modes
                .iter()
                .enumerate()
                .map(|(c, &[k, h, qx, qy])| {
                    // Keep the original grid coefficients for ambiguous
                    // Nyquist modes rather than filtering them on every step.
                    if c % n == n / 2 || c / n == n / 2 {
                        return (hk[c], qk[c]);
                    }
                    let kl = k.length();
                    let (h, qx, qy) = if kl > 0.0 {
                        let w = angular_frequency(kl, depth);
                        let khat = k / kl;
                        let ql = qx * khat.x + qy * khat.y;
                        let (s, c) = (w * dt).sin_cos();
                        let h2 = h * c - times_i(ql) * (kl / w * s);
                        let dq = ql * (c - 1.0) - times_i(h) * (w / kl * s);
                        (h2, qx + dq * khat.x, qy + dq * khat.y)
                    } else {
                        (h, qx, qy)
                    };
                    (
                        mul(h, cis((k.x + k.y) * dx / 2.0)),
                        mul(qx, cis(k.y * dx / 2.0)) + times_i(mul(qy, cis(k.x * dx / 2.0))),
                    )
                })
                .unzip();
            self.fft.run(&mut ph, true);
            self.fft.run(&mut pq, true);
            for c in 0..n * n {
                let x = self.xu(c % n, c / n);
                out_h[c] += weight(hb[c], l) * ph[c].x * scale;
                out_qx[c] += weight(fx[x], l) * pq[c].x * scale;
                out_qy[c] += weight(fy[c], l) * pq[c].y * scale;
            }
        }

        for c in 0..n * n {
            hs[c] = if hb[c] > DRY { out_h[c] } else { 0.0 };
        }
        // Blending levels and masking land move a little water; give it back.
        // shortcut: the FFT is periodic, so surface waves wrap across domain
        // edges and pass under thin walls; keep a margin or relax the edges.
        let wet_cells = hb.iter().filter(|&&h| h > DRY).count() as f32;
        let fix = (before - hs.iter().sum::<f32>()) / wet_cells;
        hs.iter_mut()
            .zip(hb)
            .filter(|(_, h)| **h > DRY)
            .for_each(|(s, _)| *s += fix);
        for j in 0..n {
            for i in 0..=n {
                let x = self.xu(i, j);
                us[x] = if i > 0 && i < n && fx[x] > DRY {
                    out_qx[i + j * n] / fx[x]
                } else {
                    0.0
                };
                let y = j + i * n;
                vs[y] = if i > 0 && i < n && fy[y] > DRY {
                    out_qy[y] / fy[y]
                } else {
                    0.0
                };
            }
        }
    }

    /// One shallow-water step: semi-Lagrangian momentum advection, surface
    /// pressure, Manning friction, then upwind fluxes limited so no cell
    /// drains below empty (Stelling & Duinmeijer 2003). Mass is conserved
    /// exactly and still water over any bed stays still.
    fn bulk(&mut self, dt: f32, manning: f32) {
        let (n, dx) = (self.n, self.cell);
        let Self {
            bed, depth, u, v, ..
        } = self;
        let (u0, v0) = (u.clone(), v.clone());
        for j in 0..n {
            for i in 1..n {
                let p = Vec2::new(i as f32, j as f32 + 0.5);
                let back = p - Vec2::new(u0[i + j * (n + 1)], sample_v(&v0, n, p)) * (dt / dx);
                u[i + j * (n + 1)] = sample_u(&u0, n, back);
                let p = Vec2::new(j as f32 + 0.5, i as f32);
                let back = p - Vec2::new(sample_u(&u0, n, p), v0[j + i * n]) * (dt / dx);
                v[j + i * n] = sample_v(&v0, n, back);
            }
        }

        let eta: Vec<f32> = bed.iter().zip(depth.iter()).map(|(b, h)| b + h).collect();
        let limit = 0.5 * dx / dt;
        let accelerate = |vel: &mut f32, l: usize, r: usize| {
            // Water can only cross a face where it stands above both beds.
            let hf = eta[l].max(eta[r]) - bed[l].max(bed[r]);
            if hf <= DRY {
                *vel = 0.0;
                return;
            }
            let w = *vel - GRAVITY * dt * (eta[r] - eta[l]) / dx;
            let w = w / (1.0 + dt * GRAVITY * manning * manning * w.abs() / hf.powf(4.0 / 3.0));
            *vel = w.clamp(-limit, limit);
        };
        let (mut fu, mut fv) = (vec![0.0; u.len()], vec![0.0; v.len()]);
        let mut out = vec![0.0; n * n];
        let mut flux = |vel: f32, l: usize, r: usize, f: &mut f32| {
            *f = vel * if vel > 0.0 { depth[l] } else { depth[r] };
            out[if vel > 0.0 { l } else { r }] += f.abs();
        };
        for j in 0..n {
            for i in 1..n {
                let (l, r) = (i - 1 + j * n, i + j * n);
                accelerate(&mut u[i + j * (n + 1)], l, r);
                flux(u[i + j * (n + 1)], l, r, &mut fu[i + j * (n + 1)]);
                let (l, r) = (j + (i - 1) * n, j + i * n);
                accelerate(&mut v[j + i * n], l, r);
                flux(v[j + i * n], l, r, &mut fv[j + i * n]);
            }
        }
        let scale: Vec<f32> = out
            .iter()
            .zip(depth.iter())
            .map(|(o, h)| {
                if o * dt / dx > *h {
                    h / (o * dt / dx)
                } else {
                    1.0
                }
            })
            .collect();
        for j in 0..n {
            for i in 1..n {
                let f = &mut fu[i + j * (n + 1)];
                *f *= scale[if *f > 0.0 { i - 1 + j * n } else { i + j * n }];
                let f = &mut fv[j + i * n];
                *f *= scale[if *f > 0.0 { j + (i - 1) * n } else { j + i * n }];
            }
        }
        for j in 0..n {
            for i in 0..n {
                let divergence = fu[i + 1 + j * (n + 1)] - fu[i + j * (n + 1)]
                    + fv[i + (j + 1) * n]
                    - fv[i + j * n];
                let h = &mut depth[i + j * n];
                *h = (*h - dt / dx * divergence).max(0.0);
            }
        }
    }

    fn update_foam(&mut self, dt: f32, lifetime: f32) {
        let (n, dx) = (self.n, self.cell);
        let fade = (-dt / lifetime.max(f32::MIN_POSITIVE)).exp();
        let eta = |i: usize, j: usize| self.bed[i + j * n] + self.depth[i + j * n];
        let mut foam = self.foam.clone();
        for j in 0..n {
            for i in 0..n {
                let c = i + j * n;
                if self.depth[c] <= DRY {
                    foam[c] = 0.0;
                    continue;
                }
                let flow = self.velocity_at(Vec2::new(i as f32 + 0.5, j as f32 + 0.5));
                let froude = flow.length() / (GRAVITY * self.depth[c]).sqrt();
                let slope = Vec2::new(
                    eta((i + 1).min(n - 1), j) - eta(i.saturating_sub(1), j),
                    eta(i, (j + 1).min(n - 1)) - eta(i, j.saturating_sub(1)),
                ) / (2.0 * dx);
                // Hydraulic jumps and bores, or crests too steep to hold.
                let source = ((froude - 0.8) / 0.5)
                    .clamp(0.0, 1.0)
                    .max(((slope.length() - 0.25) / 0.25).clamp(0.0, 1.0));
                foam[c] = (foam[c] * fade).max(source);
            }
        }
        for j in 0..n {
            for i in 0..n {
                let p = Vec2::new(i as f32 + 0.5, j as f32 + 0.5);
                let back = p - self.velocity_at(p) * (dt / dx);
                self.foam[i + j * n] = bilinear(&foam, n, back - 0.5, false);
            }
        }
    }

    /// Flow velocity at a point in cell units.
    fn velocity_at(&self, p: Vec2) -> Vec2 {
        Vec2::new(sample_u(&self.u, self.n, p), sample_v(&self.v, self.n, p))
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
    pub fn relax_edges(&mut self, target: &impl Water, width: f32) {
        let n = self.n;
        let (size, origin) = (n as f32 * self.cell, self.origin);
        let ramp = |p: Vec2| {
            let local = p - origin;
            let edge = local.x.min(local.y).min(size - local.x).min(size - local.y);
            (1.0 - edge / width).max(0.0).powi(2)
        };
        for j in 0..n {
            for i in 0..n {
                let p = self.center(i, j);
                let w = ramp(p);
                if w == 0.0 {
                    continue;
                }
                let s = target.sample(p);
                let c = i + j * n;
                self.depth[c] += ((s.height - self.bed[c]).max(0.0) - self.depth[c]) * w;
                // shortcut: drives depth-averaged flow with the target's surface
                // velocity, which overstates it in deep water.
                let corner = Vec2::new(i as f32, j as f32);
                for (x, at) in [
                    (self.xu(i, j), Vec2::new(0.0, 0.5)),
                    (self.xu(i + 1, j), Vec2::new(1.0, 0.5)),
                ] {
                    self.u[x] +=
                        (s.velocity.x - self.u[x]) * ramp(origin + (corner + at) * self.cell);
                }
                for (y, at) in [
                    (c, Vec2::new(0.5, 0.0)),
                    (i + (j + 1) * n, Vec2::new(0.5, 1.0)),
                ] {
                    self.v[y] +=
                        (s.velocity.y - self.v[y]) * ramp(origin + (corner + at) * self.cell);
                }
            }
        }
    }

    /// Cells within a disc with smooth falloff weights.
    fn disc(&self, center: Vec2, radius: f32) -> impl Iterator<Item = (usize, f32)> + '_ {
        (0..self.n * self.n).filter_map(move |c| {
            let d =
                self.center(c % self.n, c / self.n).distance(center) / radius.max(self.cell * 0.5);
            (d < 1.0).then(|| (c, (1.0 - d * d).powi(2)))
        })
    }
}

impl Water for ShallowWater {
    /// Over dry ground the surface is the bed.
    fn sample(&self, point: Vec2) -> WaterSample {
        let p = (point - self.origin) / self.cell;
        let q = p - 0.5;
        // shortcut: no vertical velocity; buoyancy drag ignores it.
        WaterSample {
            height: bilinear(&self.bed, self.n, q, false) + bilinear(&self.depth, self.n, q, false),
            velocity: self.velocity_at(p).extend(0.0),
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

/// Explicit diffusion with per-pair weight ¼·min(depth/cell, 1): symmetric,
/// so it conserves the field's sum, and stable since weights total ≤ 1.
fn diffuse(field: &mut [f32], width: usize, depth: &[f32], cell: f32, iterations: usize) {
    let rows = field.len() / width;
    let weight = |a: usize, b: usize| 0.25 * (depth[a].min(depth[b]) / cell).min(1.0);
    let mut next = field.to_vec();
    for _ in 0..iterations {
        for y in 0..rows {
            for x in 0..width {
                let i = x + y * width;
                let neighbors = [
                    (x > 0).then(|| i - 1),
                    (x + 1 < width).then(|| i + 1),
                    (y > 0).then(|| i - width),
                    (y + 1 < rows).then(|| i + width),
                ];
                next[i] = field[i]
                    + neighbors
                        .into_iter()
                        .flatten()
                        .map(|j| weight(i, j) * (field[j] - field[i]))
                        .sum::<f32>();
            }
        }
        field.copy_from_slice(&next);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
