//! Radix-2 complex FFT over square power-of-two grids. Complex numbers are
//! `Vec2` (re, im), the layout the GPU port uploads unchanged.

use glam::Vec2;

pub(crate) fn mul(a: Vec2, b: Vec2) -> Vec2 {
    Vec2::new(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x)
}

pub(crate) fn conj(a: Vec2) -> Vec2 {
    Vec2::new(a.x, -a.y)
}

/// i·a
pub(crate) fn times_i(a: Vec2) -> Vec2 {
    Vec2::new(-a.y, a.x)
}

/// e^(i·angle)
pub(crate) fn cis(angle: f32) -> Vec2 {
    let (s, c) = angle.sin_cos();
    Vec2::new(c, s)
}

/// Signed frequency index of DFT bin `i`: 0, 1, …, n/2 − 1, −n/2, …, −1.
pub(crate) fn signed(i: usize, n: usize) -> f32 {
    if i < n / 2 {
        i as f32
    } else {
        i as f32 - n as f32
    }
}

/// Index of the bin holding −k for the bin at `index` in an n×n grid.
pub(crate) fn negated(index: usize, n: usize) -> usize {
    (n - index % n) % n + (n - index / n) % n * n
}

/// Columns per pass of the column transform, sized so a strip stays in L1.
const STRIP: usize = 32;

/// Single-threaded by design: callers run independent fields side by side,
/// which outscales splitting a transform this small across threads.
pub(crate) struct Fft2 {
    n: usize,
    /// Forward and inverse twiddles, each stage contiguous: the stage of
    /// half-length h starts at h − 1.
    twiddles: [Vec<Vec2>; 2],
    reverse: Vec<usize>,
}

impl Fft2 {
    pub fn new(n: usize) -> Self {
        assert!(
            n.is_power_of_two() && n >= 2,
            "FFT size must be a power of two"
        );
        let bits = n.trailing_zeros();
        let forward: Vec<Vec2> = (0..bits)
            .flat_map(|stage| {
                let half = 1usize << stage;
                (0..half).map(move |j| {
                    let a = -std::f64::consts::PI * j as f64 / half as f64;
                    Vec2::new(a.cos() as f32, a.sin() as f32)
                })
            })
            .collect();
        Self {
            n,
            twiddles: [forward.iter().copied().map(conj).collect(), forward],
            reverse: (0..n)
                .map(|i| i.reverse_bits() >> (usize::BITS - bits))
                .collect(),
        }
    }

    fn row(&self, a: &mut [Vec2], twiddles: &[Vec2]) {
        for (i, &j) in self.reverse.iter().enumerate() {
            if i < j {
                a.swap(i, j);
            }
        }
        // The first two stages' twiddles are 1 and ∓i: no multiplications.
        let mut half = 1;
        if self.n >= 4 {
            let quarter = twiddles[2]; // −i forward, +i inverse
            for block in a.as_chunks_mut::<4>().0 {
                let (s0, d0) = (block[0] + block[1], block[0] - block[1]);
                let (s1, d1) = (block[2] + block[3], block[2] - block[3]);
                let turned = Vec2::new(-quarter.y * d1.y, quarter.y * d1.x);
                block[0] = s0 + s1;
                block[2] = s0 - s1;
                block[1] = d0 + turned;
                block[3] = d0 - turned;
            }
            half = 4;
        }
        while half < self.n {
            let w = &twiddles[half - 1..2 * half - 1];
            for block in a.chunks_exact_mut(2 * half) {
                let (lo, hi) = block.split_at_mut(half);
                for ((u, v), &w) in lo.iter_mut().zip(hi).zip(w) {
                    butterfly(u, v, w);
                }
            }
            half *= 2;
        }
    }

    /// Transforms every column at once: each butterfly pairs two whole rows
    /// under one twiddle, so it vectorizes and needs no transpose.
    fn columns(&self, data: &mut [Vec2], twiddles: &[Vec2]) {
        let n = self.n;
        for (i, &j) in self.reverse.iter().enumerate() {
            if i < j {
                let (lo, hi) = data.split_at_mut(j * n);
                lo[i * n..(i + 1) * n].swap_with_slice(&mut hi[..n]);
            }
        }
        let width = STRIP.min(n);
        for strip in (0..n).step_by(width) {
            let mut half = 1;
            while half < n {
                let w = &twiddles[half - 1..2 * half - 1];
                for block in data.chunks_exact_mut(2 * half * n) {
                    let (lo, hi) = block.split_at_mut(half * n);
                    for (r, &w) in w.iter().enumerate() {
                        let span = r * n + strip..r * n + strip + width;
                        for (u, v) in lo[span.clone()].iter_mut().zip(&mut hi[span]) {
                            butterfly(u, v, w);
                        }
                    }
                }
                half *= 2;
            }
        }
    }

    /// Unnormalized 2D DFT of a row-major n×n grid: forward sums with
    /// e^(−i k·x), inverse with e^(+i k·x).
    pub fn run(&self, data: &mut [Vec2], inverse: bool) {
        let n = self.n;
        assert_eq!(data.len(), n * n, "FFT grid must be n×n");
        let twiddles = &self.twiddles[usize::from(!inverse)];
        for row in data.chunks_exact_mut(n) {
            self.row(row, twiddles);
        }
        self.columns(data, twiddles);
    }
}

#[inline(always)]
fn butterfly(u: &mut Vec2, v: &mut Vec2, w: Vec2) {
    let t = mul(*v, w);
    *v = *u - t;
    *u += t;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    #[test]
    fn matches_naive_dft_and_round_trips() {
        // 8 fits inside one column strip, 64 spans two.
        for n in [8, 64] {
            let input: Vec<Vec2> = (0..n * n)
                .map(|i| Vec2::new((i as f32 * 0.7).sin(), (i as f32 * 1.3).cos()))
                .collect();
            let mut data = input.clone();
            let fft = Fft2::new(n);
            fft.run(&mut data, false);
            for k in (0..n * n).step_by(n * n / 64) {
                let (kx, ky) = (k % n, k / n);
                let want = input.iter().enumerate().fold(Vec2::ZERO, |sum, (x, &v)| {
                    let turns = (kx * (x % n) + ky * (x / n)) % n;
                    sum + mul(v, cis(-TAU * turns as f32 / n as f32))
                });
                let got = data[k];
                assert!((got - want).length() < 1e-2, "bin {k}: {got} vs {want}");
            }
            fft.run(&mut data, true);
            for (a, b) in data.iter().zip(&input) {
                assert!((*a / (n * n) as f32 - *b).length() < 1e-4);
            }
        }
    }
}
