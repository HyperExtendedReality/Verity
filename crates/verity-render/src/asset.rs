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
use rayon::prelude::*;
use std::{
    collections::HashMap,
    error::Error,
    fs, io,
    path::{Path, PathBuf},
};
use verity_core::PointLight;

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
    pub(crate) fn block(self) -> TranscoderBlockFormat {
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
    /// A small uncompressed copy for baking on the CPU, when one could be decoded.
    pub preview: Option<Preview>,
}

/// A low mip level, uncompressed.
pub struct Preview {
    pub width: u32,
    pub height: u32,
    /// Encoded as in the file, four bytes per texel.
    pub rgba: Vec<u8>,
    pub srgb: bool,
}

impl Preview {
    /// Linear color and alpha at a wrapped texture coordinate, nearest texel.
    pub fn sample(&self, uv: [f32; 2]) -> [f32; 4] {
        let texel = |value: f32, size: u32| {
            ((value - value.floor()) * size as f32).min(size as f32 - 1.0) as usize
        };
        let (x, y) = (texel(uv[0], self.width), texel(uv[1], self.height));
        let at = (y * self.width as usize + x) * 4;
        let decode = |byte: u8| {
            let value = f32::from(byte) / 255.0;
            match self.srgb {
                true if value <= 0.04045 => value / 12.92,
                true => ((value + 0.055) / 1.055).powf(2.4),
                false => value,
            }
        };
        [
            decode(self.rgba[at]),
            decode(self.rgba[at + 1]),
            decode(self.rgba[at + 2]),
            f32::from(self.rgba[at + 3]) / 255.0,
        ]
    }
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
    /// Emitted radiance: glTF's emissive factor times `KHR_materials_emissive_strength`.
    pub emission: [f32; 3],
    pub emission_texture: Option<u32>,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
    /// Diffuse transmission color (rgb) and factor (w), as in
    /// `GpuMaterial::transmission`.
    pub transmission: [f32; 4],
    /// Specular color (rgb) and IOR (w), as in `GpuMaterial::specular`.
    pub specular: [f32; 4],
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
            emission: [0.0; 3],
            emission_texture: None,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
            transmission: [1.0, 1.0, 1.0, 0.0],
            specular: crate::scene::DEFAULT_SPECULAR,
        }
    }
}

/// OpenPBR specular color (rgb) and IOR (w) from KHR_materials_specular and
/// KHR_materials_ior. glTF's specularFactor scales the color.
// shortcut: specularTexture and specularColorTexture are ignored; sample them in the shaders when assets use them.
fn specular(
    specular: Option<&gltf::json::Value>,
    ior: Option<&gltf::json::Value>,
) -> [f32; 4] {
    let number = |value: &gltf::json::Value| value.as_f64().map(|v| v as f32);
    let [r, g, b, default_ior] = crate::scene::DEFAULT_SPECULAR;
    // Below 1 (glTF allows exactly 0) has no opaque-surface meaning here.
    let ior = ior
        .and_then(|e| e.get("ior"))
        .and_then(number)
        .filter(|n| *n >= 1.0)
        .unwrap_or(default_ior);
    let factor = specular
        .and_then(|e| e.get("specularFactor"))
        .and_then(number)
        .unwrap_or(1.0)
        .clamp(0.0, 1.0);
    let color = specular
        .and_then(|e| e.get("specularColorFactor"))
        .and_then(|c| c.as_array())
        .filter(|c| c.len() == 3)
        .map_or([r, g, b], |c| std::array::from_fn(|i| number(&c[i]).unwrap_or(1.0).max(0.0)));
    [color[0] * factor, color[1] * factor, color[2] * factor, ior]
}

/// KHR_materials_diffuse_transmission's color (rgb) and factor (w); no
/// extension means none passes.
// shortcut: diffuseTransmissionTexture and diffuseTransmissionColorTexture are ignored; sample them in the shaders when assets use them.
fn diffuse_transmission(extension: Option<&gltf::json::Value>) -> [f32; 4] {
    let Some(extension) = extension else {
        return [1.0, 1.0, 1.0, 0.0];
    };
    let number = |value: &gltf::json::Value| value.as_f64().map(|v| v as f32);
    let factor = extension.get("diffuseTransmissionFactor").and_then(number).unwrap_or(0.0);
    let color = extension
        .get("diffuseTransmissionColorFactor")
        .and_then(|c| c.as_array())
        .filter(|c| c.len() == 3)
        .map_or([1.0; 3], |c| std::array::from_fn(|i| number(&c[i]).unwrap_or(1.0)));
    [color[0], color[1], color[2], factor.clamp(0.0, 1.0)]
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

impl From<verity_core::Sun> for Sun {
    fn from(sun: verity_core::Sun) -> Self {
        Self {
            direction: Vec3::from_array(sun.direction)
                .try_normalize()
                .unwrap_or(Vec3::NEG_Z),
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
            emission: material
                .emissive_factor()
                .map(|c| c * material.emissive_strength().unwrap_or(1.0)),
            emission_texture: slot(material.emissive_texture().map(|i| i.texture())),
            alpha_mode: match material.alpha_mode() {
                gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
                gltf::material::AlphaMode::Mask => AlphaMode::Mask,
                gltf::material::AlphaMode::Blend => AlphaMode::Blend,
            },
            alpha_cutoff: material.alpha_cutoff().unwrap_or(0.5),
            double_sided: material.double_sided(),
            transmission: diffuse_transmission(
                material.extension_value("KHR_materials_diffuse_transmission"),
            ),
            specular: specular(
                material.extension_value("KHR_materials_specular"),
                material.extension_value("KHR_materials_ior"),
            ),
        });
    }
    // Primitives without a material use the glTF default, appended last.
    let default_material = scene.materials.len() as u32;
    scene.materials.push(MaterialData::default());

    let mut primitive_slots: HashMap<(usize, usize), Vec<u32>> = HashMap::new();
    let Some(root) = gltf.default_scene().or_else(|| gltf.scenes().next()) else {
        return Ok(scene);
    };
    // glTF is Y-up; a quarter turn about X makes it Z-up.
    let z_up = Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2);
    let mut stack: Vec<_> = root.nodes().map(|n| (n, z_up)).collect();
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
    // The largest level at most this size, or the last one if none is.
    const PREVIEW_SIZE: u32 = 128;
    let preview_level = (0..levels)
        .find(|&l| size(l).0.max(size(l).1) <= PREVIEW_SIZE)
        .unwrap_or(levels - 1);
    let mut preview = None;
    for (index, level) in reader.levels().enumerate().skip(skip.min(preview_level)) {
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
        if index == preview_level {
            // The binding sizes pixel formats' output in blocks and overruns it,
            // so go through a block format and decode that here.
            let blocks = transcoder.transcode_slice(
                &data,
                SliceParametersUastc { ..params },
                DecodeFlags::HIGH_QUALITY,
                TranscoderBlockFormat::BC3,
            );
            preview = blocks.ok().map(|blocks| Preview {
                width,
                height,
                rgba: decode_bc3(&blocks, width, height),
                srgb,
            });
        }
        if index < skip {
            continue;
        }
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
        preview,
    })
}

/// A 3D RGBA8 volume from `scripts/convert_clouds.py`.
pub struct VolumeData {
    /// Width, height and depth of the largest level; z is up.
    pub size: [u32; 3],
    /// Mip levels, largest first, x fastest then y then z.
    pub levels: Vec<Vec<u8>>,
    /// The file's "verity.cloud" JSON: what the channels hold and where the
    /// volume sits.
    pub metadata: String,
}

/// Read an uncompressed (optionally Zstd-supercompressed) R8G8B8A8_UNORM 3D KTX2.
pub fn decode_volume(bytes: &[u8]) -> Result<VolumeData> {
    let reader = ktx2::Reader::new(bytes).map_err(|e| invalid(format!("not KTX2: {e:?}")))?;
    let header = reader.header();
    if header.format != Some(ktx2::Format::R8G8B8A8_UNORM) || header.pixel_depth == 0 {
        return Err(invalid(
            "volume must be 3D R8G8B8A8_UNORM (re-run scripts/convert_clouds.py)",
        ));
    }
    let zstd = match header.supercompression_scheme {
        None => false,
        Some(ktx2::SupercompressionScheme::Zstandard) => true,
        Some(other) => return Err(invalid(format!("unsupported supercompression {other:?}"))),
    };
    let size = [header.pixel_width, header.pixel_height, header.pixel_depth];
    let mut decoder = ruzstd::decoding::FrameDecoder::new();
    let mut levels = Vec::with_capacity(reader.levels().len());
    for (index, level) in reader.levels().enumerate() {
        let data = if zstd {
            let mut raw = Vec::with_capacity(level.uncompressed_byte_length as usize);
            decoder
                .decode_all_to_vec(level.data, &mut raw)
                .map_err(|e| invalid(format!("zstd: {e}")))?;
            raw
        } else {
            level.data.to_vec()
        };
        let texels: usize = size.iter().map(|&n| (n >> index).max(1) as usize).product();
        if data.len() != texels * 4 {
            return Err(invalid(format!(
                "level {index} has {} bytes, expected {}",
                data.len(),
                texels * 4
            )));
        }
        levels.push(data);
    }
    let metadata = reader
        .key_value_data()
        .find(|(key, _)| *key == "verity.cloud")
        .map(|(_, value)| String::from_utf8_lossy(value).trim_end_matches(' ').to_owned())
        .ok_or_else(|| invalid("volume has no verity.cloud metadata"))?;
    Ok(VolumeData {
        size,
        levels,
        metadata,
    })
}

/// A sparse voxel cloudscape from `scripts/convert_clouds.py`: its cloudy 8^3
/// bricks, each 64 one-byte codes into a shared codebook of 2^3 blocks.
pub struct VqCloud {
    /// Bricks along x, y and z; z is up.
    pub grid: [u32; 3],
    /// Metres per voxel.
    pub voxel: f32,
    /// Each cloudy brick's coordinates, x | y << 10 | z << 20.
    pub bricks: Vec<u32>,
    pub codes: u32,
    /// What the GPU decodes: codebook, bricks, gains, attributes and codes,
    /// as the file lays them out.
    pub payload: Vec<u8>,
    /// Per brick, x fastest: bricks to the nearest cloudy one, rounded down.
    pub distance: Vec<u8>,
    /// The file's JSON, with the source credit.
    pub metadata: String,
}

impl VqCloud {
    fn word(&self, index: usize) -> u32 {
        u32::from_le_bytes(self.payload[index * 4..index * 4 + 4].try_into().unwrap())
    }

    /// The dimensional profile at a voxel; what the GPU decode writes.
    pub fn profile(&self, [x, y, z]: [u32; 3]) -> f32 {
        let key = (x / 8) | ((y / 8) << 10) | ((z / 8) << 20);
        // Bricks are listed z, y, x major to minor, so their keys ascend.
        let Ok(slot) = self.bricks.binary_search(&key) else {
            return 0.0;
        };
        let (codes, count) = (self.codes as usize, self.bricks.len());
        let (local, block) = ([x % 8, y % 8, z % 8], |a: u32, b: u32, c: u32| (a + 4 * b + 16 * c) as usize);
        let code_index = (codes * 8 + 3 * count) * 4 + slot * 64 + block(local[0] / 2, local[1] / 2, local[2] / 2);
        let code = self.payload[code_index] as usize;
        let entry = (local[0] % 2 + 2 * (local[1] % 2) + 4 * (local[2] % 2)) as usize;
        f32::from_bits(self.word(code * 8 + entry)) * f32::from_bits(self.word(codes * 8 + count + slot))
    }
}

/// Read a `.vqcloud`.
pub fn decode_vq_cloud(bytes: &[u8]) -> Result<VqCloud> {
    let word = |at: usize| -> Result<u32> {
        bytes
            .get(at..at + 4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .ok_or_else(|| invalid("vqcloud is truncated"))
    };
    if bytes.get(..8) != Some(b"VQCLOUD1") {
        return Err(invalid("not a vqcloud (re-run scripts/convert_clouds.py)"));
    }
    let grid = [word(8)?, word(12)?, word(16)?];
    let (count, codes) = (word(20)? as usize, word(24)?);
    let voxel = f32::from_bits(word(28)?);
    let json = word(32)? as usize;
    let start = 36 + json;
    let payload_size = (codes as usize * 8 + 3 * count) * 4 + count * 64;
    let cells = grid.iter().map(|&n| n as usize).product::<usize>();
    let end = start + payload_size;
    if bytes.len() < end + cells || grid.iter().any(|&n| n == 0 || n >= 1024) {
        return Err(invalid("vqcloud is truncated or its grid is out of range"));
    }
    let payload = bytes[start..end].to_vec();
    let bricks: Vec<u32> = (0..count).map(|i| word(start + (codes as usize * 8 + i) * 4)).collect::<Result<_>>()?;
    Ok(VqCloud {
        grid,
        voxel,
        bricks,
        codes,
        payload,
        distance: bytes[end..end + cells].to_vec(),
        metadata: String::from_utf8_lossy(&bytes[36..start]).trim_end().to_owned(),
    })
}

/// Expand BC3 blocks, row by row, to RGBA8.
fn decode_bc3(blocks: &[u8], width: u32, height: u32) -> Vec<u8> {
    let (width, height) = (width as usize, height as usize);
    let mut rgba = vec![0; width * height * 4];
    for (index, block) in blocks.as_chunks::<16>().0.iter().enumerate() {
        let (block_x, block_y) = (index % width.div_ceil(4) * 4, index / width.div_ceil(4) * 4);
        let (a0, a1) = (u32::from(block[0]), u32::from(block[1]));
        let alpha_bits = block[2..8]
            .iter()
            .rev()
            .fold(0u64, |bits, &byte| bits << 8 | u64::from(byte));
        let alpha = |code: u32| match code {
            0 => a0,
            1 => a1,
            _ if a0 > a1 => ((8 - code) * a0 + (code - 1) * a1) / 7,
            6 => 0,
            7 => 255,
            _ => ((6 - code) * a0 + (code - 1) * a1) / 5,
        };
        let endpoint = |low: u8, high: u8| {
            let packed = u32::from(low) | u32::from(high) << 8;
            [
                (packed >> 11 & 31) * 255 / 31,
                (packed >> 5 & 63) * 255 / 63,
                (packed & 31) * 255 / 31,
            ]
        };
        let (c0, c1) = (endpoint(block[8], block[9]), endpoint(block[10], block[11]));
        let color = |code: u32, channel: usize| match code {
            0 => c0[channel],
            1 => c1[channel],
            2 => (2 * c0[channel] + c1[channel]) / 3,
            _ => (c0[channel] + 2 * c1[channel]) / 3,
        };
        let color_bits = u32::from_le_bytes([block[12], block[13], block[14], block[15]]);
        for texel in 0..16 {
            let (x, y) = (block_x + texel % 4, block_y + texel / 4);
            if x >= width || y >= height {
                continue;
            }
            let code = color_bits >> (texel * 2) & 3;
            let at = (y * width + x) * 4;
            for channel in 0..3 {
                rgba[at + channel] = color(code, channel) as u8;
            }
            rgba[at + 3] = alpha((alpha_bits >> (texel * 3) & 7) as u32) as u8;
        }
    }
    rgba
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
    fn diffuse_transmission_reads_factor_and_color_with_defaults() {
        let value = |text: &str| serde_json::from_str::<gltf::json::Value>(text).unwrap();
        assert_eq!(diffuse_transmission(None), [1.0, 1.0, 1.0, 0.0]);
        assert_eq!(diffuse_transmission(Some(&value("{}"))), [1.0, 1.0, 1.0, 0.0]);
        let leaf = value(
            r#"{"diffuseTransmissionFactor": 0.4, "diffuseTransmissionColorFactor": [0.5, 0.75, 0.25]}"#,
        );
        assert_eq!(diffuse_transmission(Some(&leaf)), [0.5, 0.75, 0.25, 0.4]);
        // Out-of-range factors clamp; a malformed color falls back to white.
        let odd = value(r#"{"diffuseTransmissionFactor": 3, "diffuseTransmissionColorFactor": [1]}"#);
        assert_eq!(diffuse_transmission(Some(&odd)), [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn specular_reads_color_factor_and_ior_with_defaults() {
        let value = |text: &str| serde_json::from_str::<gltf::json::Value>(text).unwrap();
        assert_eq!(specular(None, None), crate::scene::DEFAULT_SPECULAR);
        let tinted = value(r#"{"specularFactor": 0.5, "specularColorFactor": [1.0, 0.5, 0.0]}"#);
        let glass = value(r#"{"ior": 1.33}"#);
        assert_eq!(specular(Some(&tinted), Some(&glass)), [0.5, 0.25, 0.0, 1.33]);
        // An IOR below 1 falls back to the default.
        assert_eq!(specular(None, Some(&value(r#"{"ior": 0}"#)))[3], 1.5);
    }

    #[test]
    fn scene_sun_normalizes_direction_and_handles_zero() {
        let source = verity_core::Sun {
            direction: [0.0, -2.0, 0.0],
            color: [1.0, 0.5, 0.25],
            intensity: 4.0,
        };
        let sun = Sun::from(source);
        assert_eq!(sun.direction, Vec3::NEG_Y);
        assert_eq!(sun.color, Vec3::new(4.0, 2.0, 1.0));
        let zero = Sun::from(verity_core::Sun {
            direction: [0.0; 3],
            ..source
        });
        assert_eq!(zero.direction, Vec3::NEG_Z);
    }
    fn packed(name: &str) -> Option<PathBuf> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/packed")
            .join(name);
        path.exists().then_some(path)
    }

    #[test]
    fn cloud_volumes_decode() {
        let Some(noise) = packed("clouds/nubis_noise.ktx2") else {
            return;
        };
        let noise = decode_volume(&fs::read(noise).unwrap()).unwrap();
        assert_eq!(noise.size, [128; 3]);
        assert_eq!(noise.levels.len(), 8);
        assert_eq!(noise.levels[7].len(), 4);
        assert!(noise.metadata.contains("\"noise\""));
    }

    #[test]
    fn cloudscapes_decode() {
        for name in ["parkouring", "stormbird", "wdas"] {
            let Some(path) = packed(&format!("clouds/{name}.vqcloud")) else {
                continue;
            };
            let cloud = decode_vq_cloud(&fs::read(path).unwrap()).unwrap();
            assert!(cloud.metadata.contains("source"), "{name}");
            let [width, height, depth] = cloud.grid.map(|n| n as usize * 8);
            // z is up: clouds have flat bases, so they are wider a tenth of
            // the way up than nine tenths (the extreme slices are stray wisps).
            let area = |z: usize| {
                (0..height)
                    .flat_map(|y| (0..width).map(move |x| [x, y, z]))
                    .filter(|&v| cloud.profile(v.map(|n| n as u32)) > 0.0)
                    .count()
            };
            let cloudy: Vec<_> = (0..depth).filter(|&z| area(z) > 0).collect();
            assert!(!cloudy.is_empty(), "{name} has no cloud");
            let at = |fraction: f32| area(cloudy[(fraction * (cloudy.len() - 1) as f32) as usize]);
            let (bottom, top) = (at(0.1), at(0.9));
            assert!(bottom > top, "{name}: base {bottom} voxels, top {top}");
        }
    }

    #[test]
    fn vq_cloud_round_trips() {
        // One brick at (1, 0, 0) of a 2x1x1 grid; code 1 is a ramp along x.
        let mut codebook = vec![0.0f32; 16];
        codebook[8..16].copy_from_slice(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
        let mut bytes = b"VQCLOUD1".to_vec();
        for word in [2u32, 1, 1, 1, 2] {
            bytes.extend(word.to_le_bytes());
        }
        bytes.extend(8.0f32.to_le_bytes());
        bytes.extend(4u32.to_le_bytes());
        bytes.extend(b"{}  ");
        bytes.extend(bytemuck::cast_slice(&codebook));
        bytes.extend(1u32.to_le_bytes());
        bytes.extend(0.5f32.to_le_bytes());
        bytes.extend((200u32 | 100 << 8).to_le_bytes());
        bytes.extend([1u8; 64]);
        bytes.extend([1u8, 0]);
        let cloud = decode_vq_cloud(&bytes).unwrap();
        assert_eq!((cloud.grid, cloud.bricks.len(), cloud.voxel), ([2, 1, 1], 1, 8.0));
        assert_eq!(cloud.distance, [1, 0]);
        assert_eq!(cloud.profile([9, 3, 5]), 0.5);
        assert_eq!(cloud.profile([10, 3, 5]), 0.0);
        assert_eq!(cloud.profile([3, 3, 5]), 0.0);
        assert!(decode_vq_cloud(&bytes[..bytes.len() - 1]).is_err());
    }

    #[test]
    fn bc3_blocks_decode_to_their_endpoints() {
        // Red to blue with every texel on the first endpoint; alpha 200 to 100
        // with every texel on the second.
        let mut block = [0u8; 16];
        block[0] = 200;
        block[1] = 100;
        block[2..8].copy_from_slice(&[0b0100_1001, 0b1001_0010, 0b0010_0100, 0x49, 0x92, 0x24]);
        block[8..10].copy_from_slice(&0xf800u16.to_le_bytes());
        block[10..12].copy_from_slice(&0x001fu16.to_le_bytes());
        let rgba = decode_bc3(&block, 4, 4);
        assert_eq!(rgba.len(), 64);
        assert!(rgba.chunks_exact(4).all(|t| t == [255, 0, 0, 100]));

        // A size that is not a whole number of blocks stays in bounds.
        assert_eq!(decode_bc3(&block, 3, 2).len(), 24);
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
    fn previews_match_the_texture_they_came_from() {
        let Some(dir) = packed("textures") else {
            return;
        };
        let bytes = fs::read(dir.join("curtain_fabric_red_BaseColor.ktx2")).unwrap();
        let options = LoadOptions {
            format: TextureFormat::Bc7,
            max_texture_size: 256,
        };
        let preview = decode_ktx2(&bytes, options).unwrap().preview.unwrap();
        assert!(preview.width.max(preview.height) <= 128);
        assert_eq!(
            preview.rgba.len(),
            (preview.width * preview.height * 4) as usize
        );
        // It is a red curtain.
        let texel = preview.sample([0.5, 0.5]);
        assert!(texel[0] > texel[1] && texel[0] > texel[2], "{texel:?}");
        assert!(preview.sample([1.5, -0.5]) == texel, "coordinates wrap");
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
