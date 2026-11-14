//! The scene's generated terrain: an eroded landscape, as voxel chunks streamed
//! around the camera and level-of-detail patches beyond them, or a planet
//! whose patches split and merge as it moves. All draw as the renderer's
//! retained meshes. Either generates on a worker thread, one at a time; the
//! old ground stays up until the new one has meshed.

use std::{collections::HashSet, sync::Arc, thread::JoinHandle};

use verity_physics::Ground;

use glam::{IVec3, Vec3};
use verity_core::{Entity, Material, Scene, Transform};
use verity_render::{GpuMaterial, Renderer, RetainedMesh, Vertex};
use verity_terra::{
    Far, Generator, Landscape, LodUpdate, MeshData, Mesher, PatchLod, Settings, Streamer, Terrain,
    planet::{Planet, PlanetLod},
};

/// Finished chunk meshes uploaded per frame.
const UPLOAD_BUDGET: usize = 64;
/// Voxel chunks within this distance can be dug; far patches draw the rest.
const STREAM_RADIUS: f32 = 250.0;
/// Sets planet patch keys apart from landscape chunk keys.
const PLANET: u64 = 1 << 63;
/// Sets far landscape patch keys apart from chunk keys.
const FAR: u64 = 1 << 62;

/// The scene's terrain recipe; one that does not parse counts as none.
pub fn generator(scene: &Scene) -> Option<Generator> {
    serde_json::from_value(scene.terrain.clone()?).ok()
}

/// A flat sea out to the horizon, drawn with the scene but never saved in it.
pub fn sea(level: f32) -> Entity {
    Entity {
        name: "Sea".into(),
        transform: Transform {
            translation: [0.0, 0.0, level - 0.5],
            scale: [8000.0, 8000.0, 1.0],
            ..Transform::default()
        },
        material: Material {
            base_color: [0.05, 0.19, 0.24],
            metallic: 0.0,
            roughness: 0.06,
        },
        components: Default::default(),
        parent: None,
    }
}

// One source exists at a time, so the landscape's size costs nothing.
#[allow(clippy::large_enum_variant)]
#[derive(Default)]
enum Source {
    #[default]
    None,
    Landscape {
        settings: Settings,
        terrain: Terrain,
        mesher: Mesher,
        streamer: Streamer,
        far: PatchLod<Far>,
    },
    Planet(PlanetLod),
}

impl Source {
    fn shows(&self, recipe: &Generator) -> bool {
        match (self, recipe) {
            (Source::Landscape { settings, .. }, Generator::Landscape(wanted)) => {
                settings == wanted
            }
            (Source::Planet(lod), Generator::Planet(wanted)) => &lod.surface.settings == wanted,
            _ => false,
        }
    }
}

/// What a worker thread generates.
enum Built {
    Landscape(Landscape),
    Planet(Planet),
}

#[derive(Default)]
pub struct TerrainView {
    /// The recipe the scene asked for last.
    applied: Option<Generator>,
    source: Source,
    generating: Option<(Generator, JoinHandle<Built>)>,
    /// Retained mesh keys on the GPU.
    shown: HashSet<u64>,
    /// Keys of a replaced source, removed once the new one has settled.
    stale: HashSet<u64>,
}

fn key(coord: IVec3) -> u64 {
    let [x, y, z] = coord.to_array().map(|c| c as u16 as u64);
    x | y << 16 | z << 32
}

fn retained(mesh: MeshData) -> Option<RetainedMesh> {
    let vertices = (mesh
        .positions
        .iter()
        .zip(&mesh.normals)
        .zip(&mesh.terrain_uv))
    .enumerate()
    .map(|(i, ((p, n), uv))| {
        let [r, g, b] = mesh.colors.get(i).copied().unwrap_or([1.0; 3]);
        Vertex {
            position: [p[0], p[1], p[2], uv[0]],
            normal: [n[0], n[1], n[2], uv[1]],
            tangent: [0.0; 4],
            color: [r, g, b, 1.0],
        }
    })
    .collect();
    (!mesh.indices.is_empty()).then_some(RetainedMesh {
        vertices,
        indices: mesh.indices,
        bounds: mesh.bounds,
        plain: false,
    })
}

fn set(
    shown: &mut HashSet<u64>,
    stale: &mut HashSet<u64>,
    renderer: &mut Renderer,
    key: u64,
    mesh: Option<RetainedMesh>,
) {
    stale.remove(&key);
    if mesh.is_some() {
        shown.insert(key);
    } else {
        shown.remove(&key);
    }
    renderer.set_retained_mesh(key, mesh);
}

/// Swap a patch update's meshes in, keyed apart by `tag`.
fn swap(
    shown: &mut HashSet<u64>,
    stale: &mut HashSet<u64>,
    renderer: &mut Renderer,
    update: LodUpdate,
    tag: u64,
) {
    for patch in update.removed {
        set(shown, stale, renderer, patch.key() | tag, None);
    }
    for (patch, mesh) in update.added {
        set(shown, stale, renderer, patch.key() | tag, retained(mesh));
    }
}

impl TerrainView {
    /// Follow the scene's recipe and keep the meshes near `eye` current.
    /// Returns whether work remains, so the caller keeps drawing.
    pub fn update(
        &mut self,
        wanted: Option<Generator>,
        renderer: &mut Renderer,
        eye: Vec3,
    ) -> bool {
        if wanted != self.applied {
            if wanted.is_none() {
                self.replace(Source::None);
            }
            self.applied = wanted;
        }
        if let Some((recipe, handle)) = self.generating.take_if(|(_, h)| h.is_finished()) {
            match (handle.join(), recipe) {
                (Ok(_), recipe) if self.applied.as_ref() != Some(&recipe) => {} // Out of date.
                (Ok(Built::Landscape(landscape)), Generator::Landscape(settings)) => {
                    let terrain = Terrain::new(landscape);
                    let streamer = Streamer::new(&terrain, STREAM_RADIUS);
                    let far = PatchLod::new(Far::new(terrain.landscape.clone()));
                    let sea = settings.sea_level.unwrap_or(-1000.0);
                    renderer.set_retained_material(GpuMaterial::terrain(0.19, sea));
                    self.replace(Source::Landscape {
                        settings,
                        terrain,
                        mesher: Mesher::default(),
                        streamer,
                        far,
                    });
                }
                (Ok(Built::Planet(planet)), _) => {
                    self.replace(Source::Planet(PlanetLod::new(planet)));
                    renderer.set_retained_material(GpuMaterial::plain(&Material {
                        base_color: [1.0; 3],
                        metallic: 0.0,
                        roughness: 0.85,
                    }));
                }
                (Ok(Built::Landscape(_)), _) => {
                    unreachable!("a landscape recipe builds a landscape")
                }
                (Err(_), _) => eprintln!("Terrain generation panicked"),
            }
        }
        if let Some(recipe) = &self.applied
            && self.generating.is_none()
            && !self.source.shows(recipe)
        {
            let build = recipe.clone();
            let handle = std::thread::spawn(move || match build {
                Generator::Landscape(settings) => Built::Landscape(Landscape::generate(settings)),
                Generator::Planet(settings) => Built::Planet(Planet::new(settings)),
            });
            self.generating = Some((recipe.clone(), handle));
        }
        let (shown, stale) = (&mut self.shown, &mut self.stale);
        let settled = match &mut self.source {
            Source::None => true,
            Source::Landscape {
                terrain,
                mesher,
                streamer,
                far,
                ..
            } => {
                for coord in streamer.update(terrain, mesher, eye) {
                    set(shown, stale, renderer, key(coord), None);
                }
                for (coord, mesh) in mesher.update(terrain, eye, UPLOAD_BUDGET) {
                    set(shown, stale, renderer, key(coord), retained(mesh));
                }
                let (update, far_settled) = far.update(eye);
                swap(shown, stale, renderer, update, FAR);
                mesher.outstanding() == 0 && far_settled
            }
            Source::Planet(lod) => {
                let (update, settled) = lod.update(eye);
                swap(shown, stale, renderer, update, PLANET);
                settled
            }
        };
        if settled {
            for key in self.stale.drain() {
                self.shown.remove(&key);
                renderer.set_retained_mesh(key, None);
            }
        }
        self.generating.is_some() || !settled
    }

    fn replace(&mut self, source: Source) {
        self.source = source;
        self.stale.extend(self.shown.iter().copied());
    }

    /// The ground on show, for play mode's physics to collide with.
    pub fn ground(&self) -> Option<Arc<dyn Ground>> {
        match &self.source {
            Source::None => None,
            // Clones share the chunks, so this costs nothing.
            Source::Landscape { terrain, .. } => Some(Arc::new(terrain.clone())),
            Source::Planet(lod) => Some(lod.surface.clone()),
        }
    }

    /// The landscape on show, whose heights cities and crowds stand on.
    pub fn landscape(&self) -> Option<Arc<verity_terra::Landscape>> {
        match &self.source {
            Source::Landscape { terrain, .. } => Some(terrain.landscape.clone()),
            _ => None,
        }
    }

    /// The sea level of the landscape on show.
    pub fn sea_level(&self) -> Option<f32> {
        match &self.source {
            Source::Landscape { settings, .. } => settings.sea_level,
            _ => None,
        }
    }

    pub fn status(&self) -> Option<String> {
        let generating = (self.generating.as_ref()).map(|(recipe, _)| match recipe {
            Generator::Landscape(_) => "Generating landscape… ",
            Generator::Planet(_) => "Generating planet… ",
        });
        let shown = match &self.source {
            Source::None => String::new(),
            Source::Landscape {
                mesher,
                streamer,
                far,
                ..
            } => format!(
                "{} chunks, {} meshing, {} far patches",
                streamer.resident(),
                mesher.outstanding(),
                far.shown()
            ),
            Source::Planet(lod) => format!("{} patches", lod.shown()),
        };
        (generating.is_some() || !shown.is_empty())
            .then(|| format!("{}{shown}", generating.unwrap_or_default()))
    }
}
