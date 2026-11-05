//! CPU-side packing of frame data into the layouts in `shaders/lib/common.slang`.

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Quat, Vec3};
use hyperon_core::{Material, PointLight, Transform};

/// Hyperon is Z-up; glTF scenes are rotated into it on import.
pub const UP: Vec3 = Vec3::Z;
pub const NEAR: f32 = 0.1;
/// Far edge of the cluster slices; the projection itself has no far plane.
pub const CLUSTER_FAR: f32 = 100.0;
pub const CLUSTERS: usize = 16 * 9 * 24;
pub const LIGHT_WORDS: usize = 32;
/// Size of the bindless texture array.
pub const MAX_TEXTURES: u32 = 1024;
pub const NO_TEXTURE: u32 = u32::MAX;
/// Material flag bits.
pub const ALPHA_MASK: u32 = 1;
pub const ALPHA_BLEND: u32 = 2;
/// Sun shadow cascades, their square resolution and how far they reach.
pub const CASCADES: usize = 4;
pub const SHADOW_SIZE: u32 = 2048;
/// Maximum number of point lights with shadow maps, in light submission order.
pub const MAX_POINT_SHADOWS: usize = 4;
pub const POINT_SHADOW_SIZE: u32 = 512;
pub const POINT_SHADOW_FACES: usize = 6;
pub const SHADOW_DISTANCE: f32 = 60.0;
/// How far behind each cascade (toward the sun) occluders still cast.
const SHADOW_CASTER_REACH: f32 = 100.0;
/// Blend between logarithmic (1) and uniform (0) cascade splits.
const SPLIT_LAMBDA: f32 = 0.75;

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub vertical_fov: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            eye: Vec3::new(6.0, -8.0, 5.0),
            target: Vec3::ZERO,
            vertical_fov: 45f32.to_radians(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Globals {
    pub view_proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub inverse_view_proj: [[f32; 4]; 4],
    pub eye: [f32; 4],
    pub viewport: [f32; 4],
    pub projection: [f32; 4],
    pub counts: [u32; 4],
    /// xyz: unit direction the sunlight travels; w: 1 when there is a sun.
    pub sun_direction: [f32; 4],
    /// Linear color times illuminance.
    pub sun_color: [f32; 4],
    /// Last frame's view-projection, for motion vectors.
    pub previous_view_proj: [[f32; 4]; 4],
    /// World to shadow clip space per sun cascade.
    pub shadow_view_proj: [[[f32; 4]; 4]; CASCADES],
    /// View distance where each cascade ends.
    pub shadow_splits: [f32; CASCADES],
    /// World size of one shadow texel per cascade, for normal-offset bias.
    pub shadow_texels: [f32; CASCADES],
    /// Linear radiance of the sky hemisphere; w: 1 when the sun casts shadows.
    pub sky: [f32; 4],
    /// Linear radiance bounced up from the ground hemisphere.
    pub ground: [f32; 4],
    /// World-to-clip transforms for the six faces of each shadowed point light.
    pub point_shadow_view_proj: [[[f32; 4]; 4]; MAX_POINT_SHADOWS * POINT_SHADOW_FACES],
}

/// The sun's shadow cascades for one camera.
#[derive(Clone, Copy, Debug)]
pub struct Cascades {
    pub view_proj: [Mat4; CASCADES],
    pub splits: [f32; CASCADES],
    pub texels: [f32; CASCADES],
}

/// Fit sun cascades to the camera frustum: practical splits, one bounding
/// sphere per slice (stable under rotation) and texel-snapped centers
/// (stable under translation).
pub fn cascades(camera: &Camera, aspect: f32, sun_direction: Vec3) -> Cascades {
    let world_from_view = Mat4::look_at_rh(camera.eye, camera.target, UP).inverse();
    let tan_y = (camera.vertical_fov * 0.5).tan();
    let tan_x = tan_y * aspect;
    let splits: [f32; CASCADES] = std::array::from_fn(|i| {
        let t = (i + 1) as f32 / CASCADES as f32;
        let logarithmic = NEAR * (SHADOW_DISTANCE / NEAR).powf(t);
        let uniform = NEAR + (SHADOW_DISTANCE - NEAR) * t;
        SPLIT_LAMBDA * logarithmic + (1.0 - SPLIT_LAMBDA) * uniform
    });
    let direction = sun_direction.normalize();
    // Any up vector works that is not parallel to the light.
    let up = if direction.z.abs() > 0.99 {
        Vec3::Y
    } else {
        UP
    };
    let mut view_proj = [Mat4::IDENTITY; CASCADES];
    let mut texels = [0.0; CASCADES];
    for i in 0..CASCADES {
        let near = if i == 0 { NEAR } else { splits[i - 1] };
        let mut corners = Vec::with_capacity(8);
        for z in [near, splits[i]] {
            for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
                corners.push(world_from_view.transform_point3(Vec3::new(
                    x * tan_x * z,
                    y * tan_y * z,
                    -z,
                )));
            }
        }
        let center = corners.iter().copied().sum::<Vec3>() / corners.len() as f32;
        // Quantized so floating-point noise never changes the cascade's size.
        let radius = (corners
            .iter()
            .map(|c| c.distance(center))
            .fold(0.0, f32::max)
            * 16.0)
            .ceil()
            / 16.0;
        let light_view = Mat4::look_at_rh(
            center - direction * (radius + SHADOW_CASTER_REACH),
            center,
            up,
        );
        let mut projection = Mat4::orthographic_rh(
            -radius,
            radius,
            -radius,
            radius,
            0.0,
            2.0 * radius + SHADOW_CASTER_REACH,
        );
        // Snap the world origin to whole texels so edges do not crawl.
        let half = SHADOW_SIZE as f32 * 0.5;
        let origin = (projection * light_view)
            .project_point3(Vec3::ZERO)
            .truncate()
            * half;
        let offset = (origin.round() - origin) / half;
        projection.w_axis.x += offset.x;
        projection.w_axis.y += offset.y;
        view_proj[i] = projection * light_view;
        texels[i] = 2.0 * radius / SHADOW_SIZE as f32;
    }
    Cascades {
        view_proj,
        splits,
        texels,
    }
}

/// Direct and ambient lighting inputs to [`globals`].
pub struct Lighting<'a> {
    pub sun: Option<&'a crate::asset::Sun>,
    pub sky: Vec3,
    pub ground: Vec3,
}

/// One draw: a primitive's index range, its placement and its material.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Instance {
    pub model: [[f32; 4]; 4],
    pub normal_columns: [[f32; 4]; 3],
    /// x: first index, y: base vertex, z: material, w: draw index (model draws
    /// in load order, then world entities).
    pub draw: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuMaterial {
    pub base_color: [f32; 4],
    /// x: metallic, y: roughness, z: normal scale, w: alpha cutoff.
    pub params: [f32; 4],
    /// x: base color, y: metallic-roughness, z: normal texture ([`NO_TEXTURE`] if none); w: flags.
    pub textures: [u32; 4],
}

impl GpuMaterial {
    /// A plain hyperon material: factors only, no textures.
    pub fn plain(material: &Material) -> Self {
        let [r, g, b] = material.base_color;
        Self {
            base_color: [r, g, b, 1.0],
            params: [material.metallic, material.roughness, 1.0, 0.5],
            textures: [NO_TEXTURE, NO_TEXTURE, NO_TEXTURE, 0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Light {
    pub position_radius: [f32; 4],
    pub color_intensity: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    /// xyz position, w: texture u.
    pub position: [f32; 4],
    /// xyz normal, w: texture v.
    pub normal: [f32; 4],
    /// xyz tangent and w bitangent sign; all zero when the mesh has none.
    pub tangent: [f32; 4],
    /// Linear RGBA vertex color, multiplied into base color.
    pub color: [f32; 4],
}

/// Camera and frame constants for a scene viewport `[x, y, width, height]` in pixels.
pub fn globals(
    camera: &Camera,
    lighting: &Lighting,
    viewport: [u32; 4],
    counts: [u32; 3],
    previous_view_proj: Option<Mat4>,
    srgb: bool,
) -> Globals {
    let view = Mat4::look_at_rh(camera.eye, camera.target, UP);
    let aspect = viewport[2].max(1) as f32 / viewport[3].max(1) as f32;
    // Reverse-Z with an infinite far plane for even depth precision.
    let mut projection = Mat4::perspective_infinite_reverse_rh(camera.vertical_fov, aspect, NEAR);
    projection.y_axis.y *= -1.0;
    let view_proj = projection * view;
    let sun = lighting.sun;
    let shadows = cascades(camera, aspect, sun.map_or(-UP, |s| s.direction));
    let (sun_direction, sun_color) = sun.map_or(([0.0; 4], [0.0; 4]), |sun| {
        (
            sun.direction.normalize().extend(1.0).to_array(),
            sun.color.extend(0.0).to_array(),
        )
    });
    Globals {
        view_proj: view_proj.to_cols_array_2d(),
        view: view.to_cols_array_2d(),
        inverse_view_proj: view_proj.inverse().to_cols_array_2d(),
        eye: camera.eye.extend(if srgb { 1.0 } else { 0.0 }).to_array(),
        viewport: viewport.map(|v| v as f32),
        projection: [projection.x_axis.x, projection.y_axis.y, NEAR, CLUSTER_FAR],
        counts: [counts[0], counts[1], counts[2], 0],
        sun_direction,
        sun_color,
        // Without a previous frame, motion is zero.
        previous_view_proj: previous_view_proj.unwrap_or(view_proj).to_cols_array_2d(),
        shadow_view_proj: shadows.view_proj.map(|m| m.to_cols_array_2d()),
        shadow_splits: shadows.splits,
        shadow_texels: shadows.texels,
        sky: lighting
            .sky
            .extend(if sun.is_some() { 1.0 } else { 0.0 })
            .to_array(),
        ground: lighting.ground.extend(0.0).to_array(),
        point_shadow_view_proj: [[[0.0; 4]; 4]; MAX_POINT_SHADOWS * POINT_SHADOW_FACES],
    }
}

/// Six 90-degree projections, ordered +X, -X, +Y, -Y, +Z, -Z.
pub fn point_shadow_matrices(light: &PointLight) -> [Mat4; POINT_SHADOW_FACES] {
    let eye = Vec3::from_array(light.position);
    let projection = Mat4::perspective_rh(
        std::f32::consts::FRAC_PI_2,
        1.0,
        light.radius * 0.001,
        light.radius,
    );
    [
        (Vec3::X, Vec3::NEG_Y),
        (Vec3::NEG_X, Vec3::NEG_Y),
        (Vec3::Y, Vec3::Z),
        (Vec3::NEG_Y, Vec3::NEG_Z),
        (Vec3::Z, Vec3::NEG_Y),
        (Vec3::NEG_Z, Vec3::NEG_Y),
    ]
    .map(|(direction, up)| projection * Mat4::look_at_rh(eye, eye + direction, up))
}

/// Model matrix for an entity, or `None` for a degenerate transform, which
/// has no normal matrix and draws nothing visible.
pub fn entity_matrix(transform: &Transform) -> Option<Mat4> {
    if transform
        .scale
        .iter()
        .any(|s| s.abs() < 1e-8 || !s.is_finite())
    {
        return None;
    }
    let rotation = Quat::from_array(transform.rotation);
    let rotation = if rotation.length_squared() > 1e-12 {
        rotation.normalize()
    } else {
        Quat::IDENTITY
    };
    Some(Mat4::from_scale_rotation_translation(
        Vec3::from_array(transform.scale),
        rotation,
        Vec3::from_array(transform.translation),
    ))
}

/// Conservative perspective magnification for an entire transformed bound.
/// Uncertain or near-plane-crossing inputs force the original level.
pub fn lod_projection_scale(
    camera: &Camera,
    viewport: [u32; 4],
    model: Mat4,
    bounds: hyperon_strata::Bounds,
) -> f64 {
    if viewport[2] == 0
        || viewport[3] == 0
        || !model.is_finite()
        || !camera.eye.is_finite()
        || !camera.target.is_finite()
        || !camera.vertical_fov.is_finite()
        || !(0.0..std::f32::consts::PI).contains(&camera.vertical_fov)
        || (0..3).any(|i| {
            !bounds.min[i].is_finite()
                || !bounds.max[i].is_finite()
                || bounds.min[i] > bounds.max[i]
        })
    {
        return f64::INFINITY;
    }
    let forward = (camera.target - camera.eye).normalize_or_zero();
    if forward == Vec3::ZERO {
        return f64::INFINITY;
    }
    let min = Vec3::from_array(bounds.min);
    let max = Vec3::from_array(bounds.max);
    let center = model.transform_point3((min + max) * 0.5);
    let linear = Mat3::from_mat4(model);
    // Frobenius norm bounds the largest singular value, including shear.
    let stretch = (linear.x_axis.length_squared()
        + linear.y_axis.length_squared()
        + linear.z_axis.length_squared())
    .sqrt();
    let radius = (max - min).length() * 0.5 * stretch;
    let delta = center - camera.eye;
    let nearest = forward.dot(delta) - radius;
    if !nearest.is_finite() || nearest <= NEAR || !stretch.is_finite() {
        return f64::INFINITY;
    }
    let focal = viewport[3] as f64 / (2.0 * (camera.vertical_fov as f64 * 0.5).tan());
    let nearest = nearest as f64;
    // Both projection coordinates have the same pixel focal length. The
    // off-axis term bounds the perspective derivative in either direction.
    let scale =
        2.0 * focal * stretch as f64 / nearest * (1.0 + (delta.length() + radius) as f64 / nearest);
    if scale.is_finite() {
        scale
    } else {
        f64::INFINITY
    }
}

/// GPU draw record for a model matrix, index range and material.
pub fn instance(model: Mat4, first_index: u32, base_vertex: u32, material: u32) -> Instance {
    let normal = Mat3::from_mat4(model).inverse().transpose();
    let column = |c: Vec3| c.extend(0.0).to_array();
    Instance {
        model: model.to_cols_array_2d(),
        normal_columns: [
            column(normal.x_axis),
            column(normal.y_axis),
            column(normal.z_axis),
        ],
        draw: [first_index, base_vertex, material, 0],
    }
}

pub fn light(light: &PointLight) -> Light {
    let [x, y, z] = light.position;
    let [r, g, b] = light.color;
    Light {
        position_radius: [x, y, z, light.radius],
        color_intensity: [r, g, b, light.intensity],
    }
}

/// Unit cube centered on the origin: 24 vertices with face normals, 36
/// counter-clockwise indices.
pub fn cube() -> (Vec<Vertex>, Vec<u32>) {
    let (mut vertices, mut indices) = (Vec::new(), Vec::new());
    for axis in 0..3 {
        for sign in [1.0f32, -1.0] {
            let mut normal = Vec3::ZERO;
            normal[axis] = sign;
            let u = Vec3::from_array(std::array::from_fn(|i| (i == (axis + 1) % 3) as u8 as f32));
            let v = normal.cross(u);
            let base = vertices.len() as u32;
            for (a, b) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                vertices.push(Vertex {
                    position: (0.5 * (normal + u * a + v * b))
                        .extend(0.5 + 0.5 * a)
                        .to_array(),
                    normal: normal.extend(0.5 - 0.5 * b).to_array(),
                    tangent: u.extend(1.0).to_array(),
                    color: [1.0; 4],
                });
            }
            indices.extend([0, 1, 2, 0, 2, 3].map(|i| base + i));
        }
    }
    (vertices, indices)
}

/// Split-sum GGX DFG integrated with Hammersley samples over a 16x16 grid of
/// N·V (x) and perceptual roughness (y); the shader interpolates it for the
/// Fdez-Agüera multiple-scattering compensation.
pub fn brdf_lut() -> Vec<[f32; 2]> {
    const SAMPLES: u32 = 256;
    let mut table = vec![[0.0; 2]; 256];
    for y in 0..16 {
        let roughness = (y as f32 / 15.0).max(0.045);
        let alpha2 = roughness.powi(4);
        for x in 0..16 {
            let nv = (x as f32 / 15.0).max(0.0001);
            let v = Vec3::new((1.0 - nv * nv).sqrt(), 0.0, nv);
            let (mut a, mut b) = (0.0, 0.0);
            for i in 0..SAMPLES {
                let u = (i as f32 + 0.5) / SAMPLES as f32;
                let w = i.reverse_bits() as f32 * (1.0 / 4294967296.0);
                let phi = std::f32::consts::TAU * u;
                let cos_theta = ((1.0 - w) / (1.0 + (alpha2 - 1.0) * w)).sqrt();
                let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
                let h = Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);
                let vh = v.dot(h).max(0.0);
                let nl = (2.0 * vh * h - v).z.max(0.0);
                if nl > 0.0 && vh > 0.0 {
                    let gv = nl * (nv * nv * (1.0 - alpha2) + alpha2).sqrt();
                    let gl = nv * (nl * nl * (1.0 - alpha2) + alpha2).sqrt();
                    let visibility = 0.5 / (gv + gl).max(1e-8);
                    let weight = 4.0 * nl * visibility * vh / cos_theta.max(1e-8);
                    let fc = (1.0 - vh).powi(5);
                    a += (1.0 - fc) * weight;
                    b += fc * weight;
                }
            }
            table[y * 16 + x] = [a / SAMPLES as f32, b / SAMPLES as f32];
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lod_scale_falls_back_at_near_plane_and_for_invalid_bounds() {
        let camera = Camera {
            eye: Vec3::ZERO,
            target: Vec3::Y,
            vertical_fov: 1.0,
        };
        let bounds = hyperon_strata::Bounds {
            min: [-1.0; 3],
            max: [1.0; 3],
        };
        let viewport = [0, 0, 1280, 720];
        assert!(lod_projection_scale(&camera, viewport, Mat4::IDENTITY, bounds).is_infinite());
        let far = Mat4::from_translation(Vec3::Y * 100.0);
        let near = Mat4::from_translation(Vec3::Y * 10.0);
        assert!(
            lod_projection_scale(&camera, viewport, far, bounds)
                < lod_projection_scale(&camera, viewport, near, bounds)
        );
        let invalid = hyperon_strata::Bounds {
            min: [f32::NAN; 3],
            max: [1.0; 3],
        };
        assert!(lod_projection_scale(&camera, viewport, far, invalid).is_infinite());
        assert!(lod_projection_scale(&camera, [0; 4], far, bounds).is_infinite());
    }

    #[test]
    fn point_shadow_faces_cover_axes_and_light_range() {
        let light = PointLight {
            position: [2.0, -3.0, 4.0],
            color: [1.0; 3],
            intensity: 1.0,
            radius: 10.0,
        };
        let eye = Vec3::from_array(light.position);
        let axes = [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
        ];
        for (matrix, direction) in point_shadow_matrices(&light).into_iter().zip(axes) {
            let middle = matrix.project_point3(eye + direction * 5.0);
            let near = matrix.project_point3(eye + direction * 0.01);
            let far = matrix.project_point3(eye + direction * 10.0);
            assert!(middle.x.abs() < 1e-5 && middle.y.abs() < 1e-5);
            assert!(middle.z > 0.0 && middle.z < 1.0);
            assert!(near.z.abs() < 1e-4 && (far.z - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn cube_faces_wind_counter_clockwise_outward() {
        let (vertices, indices) = cube();
        assert_eq!((vertices.len(), indices.len()), (24, 36));
        for tri in indices.chunks(3) {
            let p =
                [tri[0], tri[1], tri[2]].map(|i| Vec3::from_slice(&vertices[i as usize].position));
            let n = Vec3::from_slice(&vertices[tri[0] as usize].normal);
            assert!((p[1] - p[0]).cross(p[2] - p[0]).dot(n) > 0.0);
            assert!(p.iter().all(|q| q.abs().max_element() == 0.5));
        }
    }

    #[test]
    fn normal_matrix_keeps_normals_perpendicular_under_scale() {
        let transform = Transform {
            scale: [4.0, 1.0, 0.5],
            rotation: Quat::from_rotation_y(0.4).to_array(),
            ..Transform::default()
        };
        let i = instance(entity_matrix(&transform).unwrap(), 0, 0, 0);
        let model = Mat4::from_cols_array_2d(&i.model);
        let normal = Mat3::from_cols(
            Vec3::from_slice(&i.normal_columns[0]),
            Vec3::from_slice(&i.normal_columns[1]),
            Vec3::from_slice(&i.normal_columns[2]),
        );
        // A tangent on the +X face stays perpendicular to its transformed normal.
        let tangent = model.transform_vector3(Vec3::new(0.0, 1.0, 1.0));
        assert!((normal * Vec3::X).dot(tangent).abs() < 1e-5);
        assert!(entity_matrix(&Transform {
            scale: [0.0, 1.0, 1.0],
            ..Transform::default()
        })
        .is_none());
    }

    #[test]
    fn brdf_lut_conserves_energy() {
        for [a, b] in brdf_lut() {
            assert!(a >= 0.0 && b >= 0.0 && a + b <= 1.01, "{a} + {b}");
        }
    }

    #[test]
    fn cascades_cover_their_frustum_slices() {
        let camera = Camera::default();
        let aspect = 16.0 / 9.0;
        let c = cascades(&camera, aspect, Vec3::new(-0.3, -1.0, 0.2));
        assert!(c.splits.windows(2).all(|w| w[0] < w[1]));
        assert!((c.splits[CASCADES - 1] - SHADOW_DISTANCE).abs() < 1e-3);
        let world_from_view = Mat4::look_at_rh(camera.eye, camera.target, UP).inverse();
        let tan_y = (camera.vertical_fov * 0.5).tan();
        for i in 0..CASCADES {
            let near = if i == 0 { NEAR } else { c.splits[i - 1] };
            for z in [near, c.splits[i]] {
                for (x, y) in [(-1.0, -1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0)] {
                    let corner = world_from_view.transform_point3(Vec3::new(
                        x * tan_y * aspect * z,
                        y * tan_y * z,
                        -z,
                    ));
                    let p = c.view_proj[i].project_point3(corner);
                    assert!(
                        p.x.abs() <= 1.0 && p.y.abs() <= 1.0,
                        "cascade {i} misses {corner}: {p}"
                    );
                    assert!((0.0..=1.0).contains(&p.z), "cascade {i} depth {}", p.z);
                }
            }
        }
    }

    #[test]
    fn cascades_snap_to_texels_as_the_camera_moves() {
        let half = SHADOW_SIZE as f32 * 0.5;
        for step in 0..20 {
            let shift = Vec3::new(step as f32 * 0.137, 0.0, step as f32 * 0.071);
            let camera = Camera {
                eye: Camera::default().eye + shift,
                target: shift,
                ..Camera::default()
            };
            for m in cascades(&camera, 16.0 / 9.0, Vec3::new(-0.3, -1.0, 0.2)).view_proj {
                let texel = m.project_point3(Vec3::ZERO).truncate() * half;
                assert!(
                    (texel - texel.round()).abs().max_element() < 1e-2,
                    "{texel}"
                );
            }
        }
    }

    #[test]
    fn reverse_z_maps_near_to_one() {
        let camera = Camera::default();
        let lighting = Lighting {
            sun: None,
            sky: Vec3::ZERO,
            ground: Vec3::ZERO,
        };
        let g = globals(
            &camera,
            &lighting,
            [0, 0, 1280, 720],
            [0, 0, 0],
            None,
            false,
        );
        let view_proj = Mat4::from_cols_array_2d(&g.view_proj);
        let forward = (camera.target - camera.eye).normalize();
        let near = view_proj.project_point3(camera.eye + forward * NEAR);
        let far = view_proj.project_point3(camera.eye + forward * 1000.0);
        assert!((near.z - 1.0).abs() < 1e-4 && far.z > 0.0 && far.z < 1e-3);
    }
}
