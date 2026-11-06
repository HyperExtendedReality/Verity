//! Generation, field caching, meshing and editing timings on the CPU.
use fast_surface_nets::ndshape::ConstShape;
use glam::Vec3;
use rayon::prelude::*;
use std::time::Instant;
use verity_terra::*;

fn main() {
    let start = Instant::now();
    let mut terrain = Terrain::new(Landscape::generate(Settings::default()));
    println!("Generation: {:.3}s", start.elapsed().as_secs_f64());
    let coords = terrain.initial_coords();
    let start = Instant::now();
    terrain.ensure_chunks(&coords);
    let triangles: usize = coords
        .par_iter()
        .map(|coord| {
            let mesh = polygonize(*coord, &terrain.chunks[coord].samples, &terrain.landscape);
            mesh.indices.len() / 3
        })
        .sum();
    println!(
        "Initial cache + meshes: {:.3}s; {} chunks, {} triangles, {:.1} MiB samples",
        start.elapsed().as_secs_f64(),
        coords.len(),
        triangles,
        terrain.chunks.len() as f64 * Shape::USIZE as f64 * 4.0 / 1048576.0
    );
    let center = Vec3::new(0.0, 0.0, terrain.landscape.height(0.0, 0.0) - 2.0);
    let start = Instant::now();
    let dirty = terrain.brush(center, 7.0, false);
    let brush_time = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    for coord in &dirty {
        std::hint::black_box(polygonize(
            *coord,
            &terrain.chunks[coord].samples,
            &terrain.landscape,
        ));
    }
    println!(
        "7m excavation: {:.3}ms field update; {} chunks; {:.3}ms total CPU meshing",
        brush_time,
        dirty.len(),
        start.elapsed().as_secs_f64() * 1000.0
    );
    let eye = Vec3::new(50.0, 100.0, 70.0);
    let target = Vec3::new(0.0, 20.0, terrain.landscape.height(0.0, 20.0));
    let start = Instant::now();
    for i in 0..1000 {
        let offset = Vec3::new((i % 40) as f32 - 20.0, (i / 40) as f32 - 12.0, 0.0);
        std::hint::black_box(terrain.raycast(eye, target + offset - eye, 90.0));
    }
    println!(
        "Raycast: {:.1}us each",
        start.elapsed().as_secs_f64() * 1000.0
    );
}
