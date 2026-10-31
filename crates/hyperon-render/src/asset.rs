//! glTF 2.0 binary scenes (`.glb`) with KTX2 textures, decoded on the CPU into
//! GPU-ready geometry, materials and transcoded mip chains.
//!
//! Textures must be Basis Universal UASTC (`KHR_texture_basisu`), as written
//! by `scripts/convert_assets.py`. They are transcoded to BC7 on desktop GPUs
//! or ASTC 4x4 on mobile, with ETC2 for GPUs that have neither. Every format
//! is 16 bytes per 4x4 block.

use crate::scene::Vertex;
use basis_universal::{
    DecodeFlags, LowLevelUastcTranscoder, SliceParametersUastc, TranscoderBlockFormat,
};
use glam::{Mat4, Vec3};
use hyperon_core::PointLight;
use rayon::prelude::*;
use std::{
    collections::HashMap,
    error::Error,
    fs, io,
    path::{Path, PathBuf},
};

type Result<T, E = Box<dyn Error + Send + Sync>> = std::result::Result<T, E>;

fn invalid(message: impl Into<String>) -> Box<dyn Error + Send + Sync> {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}

/// GPU block format every texture is transcoded to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureFormat {
    Bc7,
    Astc4x4,
    Etc2,
}

impl TextureFormat {
    fn block(self) -> TranscoderBlockFormat {
        match self {
            Self::Bc7 => TranscoderBlockFormat::BC7,
            Self::Astc4x4 => TranscoderBlockFormat::ASTC_4x4,
            Self::Etc2 => TranscoderBlockFormat::ETC2_RGBA,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LoadOptions {
    pub format: TextureFormat,
    /// Mip levels larger than this are skipped, bounding memory on small GPUs.
    pub max_texture_size: u32,
}

pub struct TextureData {
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub srgb: bool,
    /// Transcoded mip levels, largest first.
    pub levels: Vec<Vec<u8>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlphaMode {
    Opaque,
    Mask,
    Blend,
}

#[derive(Clone, Debug)]
pub struct MaterialData {
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub normal_scale: f32,
    pub base_color_texture: Option<u32>,
    pub metallic_roughness_texture: Option<u32>,
    pub normal_texture: Option<u32>,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
}

impl Default for MaterialData {
    fn default() -> Self {
        Self {
            base_color: [1.0; 4],
            metallic: 1.0,
            roughness: 1.0,
            normal_scale: 1.0,
            base_color_texture: None,
            metallic_roughness_texture: None,
            normal_texture: None,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
        }
    }
}

/// A triangle list within [`SceneData::indices`]; indices are relative to `base_vertex`.
#[derive(Clone, Copy, Debug)]
pub struct Primitive {
    pub first_index: u32,
    pub index_count: u32,
    pub base_vertex: u32,
    pub material: u32,
}

/// One primitive placed in the world.
#[derive(Clone, Copy, Debug)]
pub struct Draw {
    pub transform: Mat4,
    pub primitive: u32,
}

/// A directional light: unit direction the light travels, linear color times illuminance (lux).
#[derive(Clone, Copy, Debug)]
pub struct Sun {
    pub direction: Vec3,
    pub color: Vec3,
}

impl From<hyperon_core::Sun> for Sun {
    fn from(sun: hyperon_core::Sun) -> Self {
        Self {
            direction: Vec3::from_array(sun.direction)
                .try_normalize()
                .unwrap_or(Vec3::NEG_Y),
            color: Vec3::from_array(sun.color) * sun.intensity,
        }
    }
}

#[derive(Default)]
pub struct SceneData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub primitives: Vec<Primitive>,
    pub draws: Vec<Draw>,
    pub materials: Vec<MaterialData>,
    pub textures: Vec<TextureData>,
    pub lights: Vec<PointLight>,
    pub sun: Option<Sun>,
}

/// Load a `.glb` and every KTX2 texture it references.
pub fn load_gltf(path: &Path, options: LoadOptions) -> Result<SceneData> {
    let bytes = fs::read(path)?;
    let gltf = gltf::Gltf::from_slice(&bytes)?;
    let blob = gltf.blob.as_deref();
    let directory = path.parent().unwrap_or(Path::new("."));
    let mut scene = SceneData::default();

    // Textures: glTF texture index -> loaded image slot, decoding each image once.
    let mut image_slots: HashMap<usize, u32> = HashMap::new();
    let mut image_paths: Vec<PathBuf> = Vec::new();
    let mut texture_slots: Vec<u32> = Vec::new();
    for texture in gltf.textures() {
        let image = texture
            .extension_value("KHR_texture_basisu")
            .and_then(|basisu| basisu.get("source")?.as_u64())
            .ok_or_else(|| {
                invalid(format!(
                    "texture {} is not KHR_texture_basisu KTX2",
                    texture.index()
                ))
            })? as usize;
        let slot = *image_slots.entry(image).or_insert_with(|| {
            let uri = match gltf.images().nth(image).map(|i| i.source()) {
                Some(gltf::image::Source::Uri { uri, .. }) => uri.to_string(),
                _ => String::new(),
            };
            image_paths.push(directory.join(percent_decode(&uri)));
            image_paths.len() as u32 - 1
        });
        texture_slots.push(slot);
    }
    scene.textures = image_paths
        .par_iter()
        .map(|path| {
            let bytes = fs::read(path).map_err(|e| invalid(format!("{}: {e}", path.display())))?;
            decode_ktx2(&bytes, options).map_err(|e| invalid(format!("{}: {e}", path.display())))
        })
        .collect::<Result<_>>()?;

    let slot = |info: Option<gltf::texture::Texture<'_>>| info.map(|t| texture_slots[t.index()]);
    for material in gltf.materials() {
        let pbr = material.pbr_metallic_roughness();
        scene.materials.push(MaterialData {
            base_color: pbr.base_color_factor(),
            metallic: pbr.metallic_factor(),
            roughness: pbr.roughness_factor(),
            normal_scale: material.normal_texture().map_or(1.0, |n| n.scale()),
            base_color_texture: slot(pbr.base_color_texture().map(|i| i.texture())),
            metallic_roughness_texture: slot(pbr.metallic_roughness_texture().map(|i| i.texture())),
            normal_texture: slot(material.normal_texture().map(|n| n.texture())),
            alpha_mode: match material.alpha_mode() {
                gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
                gltf::material::AlphaMode::Mask => AlphaMode::Mask,
                gltf::material::AlphaMode::Blend => AlphaMode::Blend,
            },
            alpha_cutoff: material.alpha_cutoff().unwrap_or(0.5),
            double_sided: material.double_sided(),
        });
    }
    // Primitives without a material use the glTF default, appended last.
    let default_material = scene.materials.len() as u32;
    scene.materials.push(MaterialData::default());

    let mut primitive_slots: HashMap<(usize, usize), Vec<u32>> = HashMap::new();
    let Some(root) = gltf.default_scene().or_else(|| gltf.scenes().next()) else {
        return Ok(scene);
    };
    let mut stack: Vec<_> = root.nodes().map(|n| (n, Mat4::IDENTITY)).collect();
    while let Some((node, parent)) = stack.pop() {
        let transform = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        stack.extend(node.children().map(|child| (child, transform)));
        if let Some(light) = node.light() {
            add_light(&mut scene, &light, transform);
        }
        let Some(mesh) = node.mesh() else { continue };
        let slots = match primitive_slots.get(&(mesh.index(), 0)) {
            Some(slots) => slots.clone(),
            None => {
                let mut slots = Vec::new();
                for primitive in mesh.primitives() {
                    if primitive.mode() != gltf::mesh::Mode::Triangles {
                        continue;
                    }
                    let material = primitive
                        .material()
                        .index()
                        .map_or(default_material, |m| m as u32);
                    if let Some(id) = read_primitive(&mut scene, &primitive, blob, material)? {
                        slots.push(id);
                    }
                }
                primitive_slots.insert((mesh.index(), 0), slots.clone());
                slots
            }
        };
        for primitive in slots {
            scene.draws.push(Draw {
                transform,
                primitive,
            });
        }
    }
    Ok(scene)
}

fn add_light(scene: &mut SceneData, light: &gltf::khr_lights_punctual::Light<'_>, transform: Mat4) {
    let color = Vec3::from_array(light.color()) * light.intensity();
    match light.kind() {
        gltf::khr_lights_punctual::Kind::Point => {
            // Range where intensity / d² falls to 0.01 when glTF gives none.
            let radius = light
                .range()
                .unwrap_or_else(|| (color.max_element() / 0.01).sqrt());
            if radius > 0.0 && color.max_element() > 0.0 {
                scene.lights.push(PointLight {
                    position: transform.w_axis.truncate().to_array(),
                    color: light.color(),
                    intensity: light.intensity(),
                    radius,
                });
            }
        }
        gltf::khr_lights_punctual::Kind::Directional => {
            // glTF directional lights shine along the node's local -Z.
            let direction = transform.transform_vector3(Vec3::NEG_Z).normalize_or_zero();
            if direction != Vec3::ZERO && scene.sun.is_none() {
                scene.sun = Some(Sun { direction, color });
            }
        }
        // shortcut: spot lights are not shaded yet; import them once the
        // clustered pass has cone data.
        gltf::khr_lights_punctual::Kind::Spot { .. } => {}
    }
}

/// Append one primitive's vertices and indices; `None` when it has no positions.
fn read_primitive(
    scene: &mut SceneData,
    primitive: &gltf::Primitive<'_>,
    blob: Option<&[u8]>,
    material: u32,
) -> Result<Option<u32>> {
    let reader = primitive.reader(|buffer| match buffer.source() {
        gltf::buffer::Source::Bin => blob,
        gltf::buffer::Source::Uri(_) => None,
    });
    let Some(positions) = reader.read_positions() else {
        return Ok(None);
    };
    let positions: Vec<[f32; 3]> = positions.collect();
    let count = positions.len();
    let normals: Vec<[f32; 3]> = reader
        .read_normals()
        .map_or_else(|| vec![[0.0, 1.0, 0.0]; count], |n| n.collect());
    let uvs: Vec<[f32; 2]> = reader
        .read_tex_coords(0)
        .map_or_else(|| vec![[0.0; 2]; count], |t| t.into_f32().collect());
    // A zero tangent tells the shader to derive one from the triangle.
    let tangents: Vec<[f32; 4]> = reader
        .read_tangents()
        .map_or_else(|| vec![[0.0; 4]; count], |t| t.collect());
    let colors: Vec<[f32; 4]> = reader
        .read_colors(0)
        .map_or_else(|| vec![[1.0; 4]; count], |c| c.into_rgba_f32().collect());
    if [normals.len(), uvs.len(), tangents.len(), colors.len()]
        .iter()
        .any(|&n| n != count)
    {
        return Err(invalid("vertex attributes have different lengths"));
    }
    let indices: Vec<u32> = reader
        .read_indices()
        .map_or_else(|| (0..count as u32).collect(), |i| i.into_u32().collect());
    if indices.iter().any(|&i| i as usize >= count) || !indices.len().is_multiple_of(3) {
        return Err(invalid(
            "primitive indices are out of range or not whole triangles",
        ));
    }

    let base_vertex = scene.vertices.len() as u32;
    scene.vertices.extend((0..count).map(|i| {
        let ([px, py, pz], [nx, ny, nz], [u, v]) = (positions[i], normals[i], uvs[i]);
        Vertex {
            position: [px, py, pz, u],
            normal: [nx, ny, nz, v],
            tangent: tangents[i],
            color: colors[i],
        }
    }));
    let first_index = scene.indices.len() as u32;
    scene.indices.extend_from_slice(&indices);
    scene.primitives.push(Primitive {
        first_index,
        index_count: indices.len() as u32,
        base_vertex,
        material,
    });
    Ok(Some(scene.primitives.len() as u32 - 1))
}

/// Transcode a UASTC KTX2 file to `options.format`, skipping levels above the size cap.
pub fn decode_ktx2(bytes: &[u8], options: LoadOptions) -> Result<TextureData> {
    let reader = ktx2::Reader::new(bytes).map_err(|e| invalid(format!("not KTX2: {e:?}")))?;
    let header = reader.header();
    let dfd = reader
        .basic_dfd()
        .ok_or_else(|| invalid("KTX2 has no basic data format descriptor"))?;
    if header.format.is_some() || dfd.color_model != Some(ktx2::ColorModel::UASTC) {
        return Err(invalid(
            "texture must be Basis UASTC (re-run scripts/convert_assets.py)",
        ));
    }
    if header.layer_count > 1 || header.face_count != 1 || header.pixel_depth > 1 {
        return Err(invalid("only single 2D textures are supported"));
    }
    let zstd = match header.supercompression_scheme {
        None => false,
        Some(ktx2::SupercompressionScheme::Zstandard) => true,
        Some(other) => return Err(invalid(format!("unsupported supercompression {other:?}"))),
    };
    // UASTC channel ids 3 (RGBA) and 5 (RRRG) carry alpha.
    let has_alpha = dfd
        .sample_information
        .iter()
        .any(|s| matches!(s.channel_type, 3 | 5));
    let srgb = dfd.transfer_function == Some(ktx2::TransferFunction::SRGB);
    let size = |level: usize| {
        (
            (header.pixel_width >> level).max(1),
            (header.pixel_height >> level).max(1),
        )
    };
    let levels = reader.levels().count();
    // Keep at least the smallest level whatever the cap.
    let skip = (0..levels)
        .find(|&l| size(l).0.max(size(l).1) <= options.max_texture_size.max(1))
        .unwrap_or(levels - 1);

    let transcoder = LowLevelUastcTranscoder::new();
    let mut decoder = ruzstd::decoding::FrameDecoder::new();
    let mut out = Vec::with_capacity(levels - skip);
    for (index, level) in reader.levels().enumerate().skip(skip) {
        let data = if zstd {
            let mut raw = Vec::with_capacity(level.uncompressed_byte_length as usize);
            decoder
                .decode_all_to_vec(level.data, &mut raw)
                .map_err(|e| invalid(format!("zstd: {e}")))?;
            raw
        } else {
            level.data.to_vec()
        };
        let (width, height) = size(index);
        let blocks = (width.div_ceil(4), height.div_ceil(4));
        if data.len() != (blocks.0 * blocks.1 * 16) as usize {
            return Err(invalid(format!(
                "level {index} has {} bytes, expected UASTC 4x4 blocks",
                data.len()
            )));
        }
        let params = SliceParametersUastc {
            num_blocks_x: blocks.0,
            num_blocks_y: blocks.1,
            has_alpha,
            original_width: width,
            original_height: height,
        };
        out.push(
            transcoder
                .transcode_slice(
                    &data,
                    params,
                    DecodeFlags::HIGH_QUALITY,
                    options.format.block(),
                )
                .map_err(|e| invalid(format!("transcode level {index}: {e:?}")))?,
        );
    }
    let (width, height) = size(skip);
    Ok(TextureData {
        width,
        height,
        format: options.format,
        srgb,
        levels: out,
    })
}

fn percent_decode(uri: &str) -> String {
    let bytes = uri.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let hex = bytes
            .get(i + 1..i + 3)
            .and_then(|h| u8::from_str_radix(std::str::from_utf8(h).ok()?, 16).ok());
        match (bytes[i], hex) {
            (b'%', Some(byte)) => {
                out.push(byte);
                i += 3;
            }
            (byte, _) => {
                out.push(byte);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_sun_normalizes_direction_and_handles_zero() {
        let source = hyperon_core::Sun {
            direction: [0.0, -2.0, 0.0],
            color: [1.0, 0.5, 0.25],
            intensity: 4.0,
        };
        let sun = Sun::from(source);
        assert_eq!(sun.direction, Vec3::NEG_Y);
        assert_eq!(sun.color, Vec3::new(4.0, 2.0, 1.0));
        let zero = Sun::from(hyperon_core::Sun {
            direction: [0.0; 3],
            ..source
        });
        assert_eq!(zero.direction, Vec3::NEG_Y);
    }
    fn packed(name: &str) -> Option<PathBuf> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/packed")
            .join(name);
        path.exists().then_some(path)
    }

    #[test]
    fn percent_decoding_keeps_plain_text() {
        assert_eq!(
            percent_decode("textures/Fortress-Ka%C5%A1tel.ktx2"),
            "textures/Fortress-Kaštel.ktx2"
        );
        assert_eq!(percent_decode("100%"), "100%");
    }

    #[test]
    fn curtains_load_with_capped_bc7_textures() {
        // The converted assets are local only; skip when they are absent.
        let Some(path) = packed("NewSponza_Curtains.glb") else {
            return;
        };
        let options = LoadOptions {
            format: TextureFormat::Bc7,
            max_texture_size: 256,
        };
        let scene = load_gltf(&path, options).unwrap();
        assert_eq!(scene.textures.len(), 8);
        for texture in &scene.textures {
            assert_eq!((texture.width, texture.height), (256, 256));
            assert_eq!(texture.levels.len(), 9);
            // BC7: 16 bytes per 4x4 block.
            assert_eq!(texture.levels[0].len(), 64 * 64 * 16);
            assert_eq!(texture.levels.last().unwrap().len(), 16);
        }
        assert!(scene.textures.iter().any(|t| t.srgb) && scene.textures.iter().any(|t| !t.srgb));
        assert_eq!(scene.draws.len(), 8);
        for primitive in &scene.primitives {
            let end = primitive.first_index + primitive.index_count;
            let indices = &scene.indices[primitive.first_index as usize..end as usize];
            assert!(indices
                .iter()
                .all(|&i| ((primitive.base_vertex + i) as usize) < scene.vertices.len()));
            let material = &scene.materials[primitive.material as usize];
            assert!(material.base_color_texture.is_some() && material.normal_texture.is_some());
        }
    }

    #[test]
    fn astc_and_etc2_decode_the_same_blocks() {
        let Some(dir) = packed("textures") else {
            return;
        };
        let bytes = fs::read(dir.join("curtain_fabric_red_BaseColor.ktx2"))
            .or_else(|_| fs::read(fs::read_dir(&dir).unwrap().next().unwrap().unwrap().path()))
            .unwrap();
        for format in [TextureFormat::Astc4x4, TextureFormat::Etc2] {
            let options = LoadOptions {
                format,
                max_texture_size: 16,
            };
            let texture = decode_ktx2(&bytes, options).unwrap();
            assert_eq!((texture.width, texture.levels[0].len()), (16, 4 * 4 * 16));
        }
    }
}
