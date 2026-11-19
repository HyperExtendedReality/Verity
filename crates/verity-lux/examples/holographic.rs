//! Sliced 3D Holographic Radiance Cascades against brute force, on a 64³ box
//! with a ceiling light, a blue glowing cube, a hanging block and smoke.
//!
//! Writes the middle slice's fluence from each, and their difference, to
//! `target/holographic/` (or the folder given), and prints timings and error.
//!
//! `--pages=K` sets the azimuths a frame gathers (8), and `--frames=F` averages
//! F frames whose pages each turn by a fraction of the gap between them (1).
//!
//! `cargo run -p verity-lux --release --example holographic -- [--pages=8] [--frames=1] [out]`

use glam::{IVec3, Vec3};
use rayon::prelude::*;
use std::{error::Error, f32::consts::PI, fs, path::Path, time::Instant};
use verity_lux::holographic::{Gather, Volume, gather, reference};

const N: i32 = 64;
const ZOOM: usize = 6;

fn scene() -> Volume {
    let wall = 40.0;
    let mut v = Volume::new(IVec3::splat(N));
    let (lo, hi) = (IVec3::ZERO, IVec3::splat(N - 1));
    for axis in 0..3 {
        let (mut near, mut far) = (hi, lo);
        near[axis] = 0;
        far[axis] = N - 1;
        v.fill(lo, near, Vec3::ZERO, wall);
        v.fill(far, hi, Vec3::ZERO, wall);
    }
    v.fill(
        IVec3::new(24, 62, 24),
        IVec3::new(39, 62, 39),
        Vec3::new(9.0, 8.5, 7.5),
        wall,
    );
    v.fill(
        IVec3::new(8, 1, 28),
        IVec3::new(13, 6, 35),
        Vec3::new(0.5, 1.5, 8.0),
        wall,
    );
    v.fill(
        IVec3::new(28, 36, 24),
        IVec3::new(43, 39, 39),
        Vec3::ZERO,
        wall,
    );
    v.fill(
        IVec3::new(48, 1, 8),
        IVec3::new(60, 30, 56),
        Vec3::ZERO,
        0.15,
    );
    v
}

fn write_png(
    path: &Path,
    size: usize,
    pixel: impl Fn(usize, usize) -> Vec3,
) -> Result<(), Box<dyn Error>> {
    let side = size * ZOOM;
    let mut bytes = Vec::with_capacity(side * side * 3);
    for row in 0..side {
        for col in 0..side {
            // Rows run down the image, y up the volume.
            let c = pixel(col / ZOOM, size - 1 - row / ZOOM);
            bytes.extend(
                c.to_array()
                    .map(|v| (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8),
            );
        }
    }
    let mut encoder = png::Encoder::new(fs::File::create(path)?, side as u32, side as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&bytes)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str, default: usize| -> Result<usize, Box<dyn Error>> {
        let prefix = format!("--{name}=");
        Ok(
            match args.iter().find_map(|a| a.strip_prefix(prefix.as_str())) {
                Some(v) => v.parse()?,
                None => default,
            },
        )
    };
    let (pages, frames) = (flag("pages", 8)?.max(1), flag("frames", 1)?.max(1));
    let out = Path::new(
        args.iter()
            .find(|a| !a.starts_with("--"))
            .map_or("target/holographic", |s| s),
    );
    fs::create_dir_all(out)?;

    let volume = scene();
    let start = Instant::now();
    let mut hrc = vec![Gather::default(); volume.sigma.len()];
    for frame in 0..frames {
        let turn = (frame as f32 + 0.5) / frames as f32;
        for (sum, g) in hrc.iter_mut().zip(gather(&volume, pages, turn)) {
            sum.fluence += g.fluence / frames as f32;
        }
    }
    let hrc_time = start.elapsed() / frames as u32;

    let z = N / 2;
    let start = Instant::now();
    let slice: Vec<_> = (0..N * N)
        .into_par_iter()
        .map(|i| reference(&volume, IVec3::new(i % N, i / N, z), 16384, 0.5))
        .collect();
    let reference_time = start.elapsed();

    let at = |x: usize, y: usize| IVec3::new(x as i32, y as i32, z);
    let solid = |x, y| volume.sigma[volume.index(at(x, y)).unwrap()] > 1.0;
    let tone = |f: Vec3| {
        let v = f / (4.0 * PI);
        v / (Vec3::ONE + v)
    };
    let hrc_at = |x, y| hrc[volume.index(at(x, y)).unwrap()].fluence;
    let ref_at = |x: usize, y: usize| slice[y * N as usize + x].fluence;
    let shown = |f: Vec3, x, y| {
        if solid(x, y) {
            Vec3::splat(0.03)
        } else {
            tone(f)
        }
    };

    let size = N as usize;
    write_png(&out.join("hrc.png"), size, |x, y| shown(hrc_at(x, y), x, y))?;
    write_png(&out.join("reference.png"), size, |x, y| {
        shown(ref_at(x, y), x, y)
    })?;
    write_png(&out.join("difference.png"), size, |x, y| {
        // Red where HRC is brighter, blue where darker, against the reference.
        let d = (hrc_at(x, y) - ref_at(x, y)).element_sum() / ref_at(x, y).element_sum().max(1e-3);
        Vec3::new(d.max(0.0), 0.0, (-d).max(0.0))
    })?;

    let (mut error, mut total) = (0.0, 0.0);
    for y in 0..size {
        for x in 0..size {
            if !solid(x, y) {
                error += (hrc_at(x, y) - ref_at(x, y)).abs().element_sum();
                total += ref_at(x, y).element_sum();
            }
        }
    }
    println!("{pages} pages, {frames} frames");
    println!(
        "HRC, all {} voxels: {:.2} s a frame",
        N * N * N,
        hrc_time.as_secs_f32()
    );
    println!(
        "brute force, one slice: {:.2} s",
        reference_time.as_secs_f32()
    );
    println!(
        "mean relative error on the slice: {:.1}%",
        100.0 * error / total
    );
    println!("images in {}", out.display());
    Ok(())
}
