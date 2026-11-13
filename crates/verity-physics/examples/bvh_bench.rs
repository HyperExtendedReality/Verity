//! Temporary: tree build time and CPU ray speed for TriMesh, before/after builder changes.
use glam::Vec3;
use std::time::Instant;
use verity_physics::{Bounds, Bvh, TriMesh};

fn rng(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    (*seed >> 40) as f32 / (1u64 << 24) as f32
}

fn bench(name: &str, positions: &[Vec3], tris: &[[u32; 3]], lo: Vec3, hi: Vec3) {
    let start = Instant::now();
    let mut mesh = None;
    for _ in 0..5 {
        mesh = Some(TriMesh::new(positions, tris));
    }
    let build = start.elapsed().as_secs_f64() * 1000.0 / 5.0;
    let mesh = mesh.unwrap();
    let mut seed = 0x1234_5678u64;
    let rays: Vec<(Vec3, Vec3)> = (0..400_000)
        .map(|_| {
            let o = lo + (hi - lo) * Vec3::new(rng(&mut seed), rng(&mut seed), rng(&mut seed));
            let d = Vec3::new(rng(&mut seed) - 0.5, rng(&mut seed) - 0.5, rng(&mut seed) - 0.5)
                .normalize_or(Vec3::Z);
            (o, d)
        })
        .collect();
    let start = Instant::now();
    let (mut hits, mut sum) = (0u32, 0.0f64);
    for &(o, d) in &rays {
        if let Some((t, _)) = mesh.ray(o, d, 1e6) {
            hits += 1;
            sum += t as f64;
        }
    }
    let ns = start.elapsed().as_secs_f64() * 1e9 / rays.len() as f64;
    println!("{name:8} tris {:7} build {build:8.2} ms  ray {ns:7.1} ns  hits {hits} sum {sum:.3} nodes {}",
        tris.len(), mesh.nodes().len());
}

fn small(count: usize) {
    let mut seed = 7u64;
    let bounds: Vec<Bounds> = (0..count)
        .map(|_| {
            let c = Vec3::new(rng(&mut seed), rng(&mut seed), rng(&mut seed)) * 50.0;
            Bounds { min: c - 0.5, max: c + 0.5 }
        })
        .collect();
    let reps = 2_000_000 / count;
    let start = Instant::now();
    let mut nodes = 0;
    for _ in 0..reps {
        nodes += Bvh::build(&bounds).nodes().len();
    }
    let us = start.elapsed().as_secs_f64() * 1e6 / reps as f64;
    println!("bodies {count:6} build {us:9.2} us  nodes {}", nodes / reps);
}

fn main() {
    for count in [50, 300, 2000] {
        small(count);
    }
    // A noisy height field, 400 x 400 cells.
    let n = 401u32;
    let positions: Vec<Vec3> = (0..n * n)
        .map(|i| {
            let (x, y) = ((i % n) as f32, (i / n) as f32);
            Vec3::new(x, y, (x * 0.13).sin() * 4.0 + (y * 0.07).cos() * 6.0)
        })
        .collect();
    let mut tris = Vec::new();
    for y in 0..n - 1 {
        for x in 0..n - 1 {
            let i = y * n + x;
            tris.push([i, i + 1, i + n]);
            tris.push([i + 1, i + n + 1, i + n]);
        }
    }
    bench("terrain", &positions, &tris, Vec3::new(0.0, 0.0, 10.0), Vec3::new(400.0, 400.0, 30.0));
    // Clustered soup: small triangles in a few dense clumps plus long thin ones.
    let mut seed = 99u64;
    let mut positions = Vec::new();
    let mut tris = Vec::new();
    for k in 0..200_000u32 {
        let clump = Vec3::new((k % 7) as f32 * 9.0, (k % 5) as f32 * 11.0, (k % 3) as f32 * 6.0);
        let c = clump + Vec3::new(rng(&mut seed), rng(&mut seed), rng(&mut seed)) * 8.0;
        let s = if k % 50 == 0 { 20.0 } else { 0.3 };
        for _ in 0..3 {
            positions.push(c + (Vec3::new(rng(&mut seed), rng(&mut seed), rng(&mut seed)) - 0.5) * s);
        }
        tris.push([3 * k, 3 * k + 1, 3 * k + 2]);
    }
    bench("soup", &positions, &tris, Vec3::splat(-5.0), Vec3::new(70.0, 55.0, 25.0));
}
