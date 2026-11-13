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
    // What the editor streams: the surface chunks within 250 m of the centre.
    let coords: Vec<_> = (terrain.initial_coords().into_iter())
        .filter(|c| chunk_origin(*c).truncate().length() < 250.0)
        .collect();
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
    let start = Instant::now();
    let far = Far::new(terrain.landscape.clone());
    let leaves = far.leaves(eye);
    let patches: usize = (leaves.par_iter())
        .map(|p| far.mesh(*p).indices.len() / 3)
        .sum();
    println!(
        "Far patches: {:.3}s; {} patches, {} triangles",
        start.elapsed().as_secs_f64(),
        leaves.len(),
        patches
    );
    let start = Instant::now();
    let planet = planet::Planet::new(planet::PlanetSettings::default());
    println!("Planet generation: {:.3}s", start.elapsed().as_secs_f64());
    let start = Instant::now();
    let leaves = planet.leaves(Vec3::new(0.0, 0.0, 20.0));
    let triangles: usize = (leaves.par_iter())
        .map(|p| planet.mesh(*p).indices.len() / 3)
        .sum();
    println!(
        "Planet patches near the ground: {:.3}s; {} patches, {} triangles",
        start.elapsed().as_secs_f64(),
        leaves.len(),
        triangles
    );
}
