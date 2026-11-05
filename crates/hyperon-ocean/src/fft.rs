//! Radix-2 complex FFT over square power-of-two grids. Complex numbers are
//! `Vec2` (re, im), the layout the GPU port uploads unchanged.

use glam::Vec2;
use rayon::prelude::*;

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

pub(crate) struct Fft2 {
    n: usize,
    twiddles: Vec<Vec2>,
    reverse: Vec<usize>,
}

impl Fft2 {
    pub fn new(n: usize) -> Self {
        assert!(
            n.is_power_of_two() && n >= 2,
            "FFT size must be a power of two"
        );
        let bits = n.trailing_zeros();
        Self {
            n,
            twiddles: (0..n / 2)
                .map(|i| {
                    let a = -std::f64::consts::TAU * i as f64 / n as f64;
                    Vec2::new(a.cos() as f32, a.sin() as f32)
                })
                .collect(),
            reverse: (0..n)
                .map(|i| i.reverse_bits() >> (usize::BITS - bits))
                .collect(),
        }
    }

    fn line(&self, a: &mut [Vec2], inverse: bool) {
        let n = self.n;
        for i in 0..n {
            let j = self.reverse[i];
            if i < j {
                a.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let step = n / len;
            for start in (0..n).step_by(len) {
                for j in 0..len / 2 {
                    let w = self.twiddles[j * step];
                    let w = if inverse { conj(w) } else { w };
                    let (u, v) = (a[start + j], mul(a[start + j + len / 2], w));
                    a[start + j] = u + v;
                    a[start + j + len / 2] = u - v;
                }
            }
            len <<= 1;
        }
    }

    /// Unnormalized 2D DFT of a row-major n×n grid: forward sums with
    /// e^(−i k·x), inverse with e^(+i k·x).
    pub fn run(&self, data: &mut [Vec2], inverse: bool) {
        let n = self.n;
        assert_eq!(data.len(), n * n, "FFT grid must be n×n");
        data.par_chunks_mut(n)
            .for_each(|row| self.line(row, inverse));
        transpose(data, n);
        data.par_chunks_mut(n)
            .for_each(|row| self.line(row, inverse));
        transpose(data, n);
    }
}

fn transpose(data: &mut [Vec2], n: usize) {
    for y in 0..n {
        for x in y + 1..n {
            data.swap(x + y * n, y + x * n);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    #[test]
    fn matches_naive_dft_and_round_trips() {
        let n = 8;
        let input: Vec<Vec2> = (0..n * n)
            .map(|i| Vec2::new((i as f32 * 0.7).sin(), (i as f32 * 1.3).cos()))
            .collect();
        let mut data = input.clone();
        let fft = Fft2::new(n);
        fft.run(&mut data, false);
        for (k, &got) in data.iter().enumerate() {
            let (kx, ky) = (k % n, k / n);
            let want = input.iter().enumerate().fold(Vec2::ZERO, |sum, (x, &v)| {
                let phase = -TAU * ((kx * (x % n) + ky * (x / n)) as f32) / n as f32;
                sum + mul(v, cis(phase))
            });
            assert!((got - want).length() < 1e-3, "bin {k}: {got} vs {want}");
        }
        fft.run(&mut data, true);
        for (a, b) in data.iter().zip(&input) {
            assert!((*a / (n * n) as f32 - *b).length() < 1e-5);
        }
    }
}
