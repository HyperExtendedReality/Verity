//! Times the GI scene build for the converted Sponza in `assets/packed`.
//! `cargo run --release -p verity-lux --example build_sponza`
use std::{path::Path, time::Instant};
use verity_lux::cascades::scene::GiScene;
use verity_render::asset::{LoadOptions, TextureFormat, load_gltf};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let packed = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/packed");
    // Textures are irrelevant here; the smallest mip keeps loading fast.
    let options = LoadOptions {
        format: TextureFormat::Bc7,
        max_texture_size: 1,
    };
    for name in [
        "NewSponza_Main.glb",
        "NewSponza_Curtains.glb",
        "NewSponza_IvyGrowth.glb",
    ] {
        let scene = load_gltf(&packed.join(name), options)?;
        let started = Instant::now();
        let gi = GiScene::build(&scene, 0);
        let bytes = (gi.geometry_words.len() + gi.scene_words.len()) * 16;
        println!(
            "{name}: {} triangles, {} instances, {:.1} MB of words in {:.2?}",
            gi.triangles,
            gi.instances,
            bytes as f64 / 1e6,
            started.elapsed()
        );
    }
    Ok(())
}
