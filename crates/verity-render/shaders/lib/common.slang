// Engine-wide frame data, bindings and cluster addressing shared by every pass.
// Layouts mirror `scene.rs`; change both together.

#include "materials.slang"

static const float PI = 3.14159265359;
static const uint CLUSTER_X = 16;
static const uint CLUSTER_Y = 9;
static const uint CLUSTER_Z = 24;
static const uint CLUSTER_COUNT = CLUSTER_X * CLUSTER_Y * CLUSTER_Z;
/// 32-bit light masks per cluster: MAX_LIGHTS / 32.
static const uint LIGHT_WORDS = 32;
static const uint NO_INSTANCE = 0xffffffff;
/// Mirrors `scene::CASCADES`; a multiple of four (splits pack into float4s).
static const uint CASCADES = 8;
static const float SHADOW_SIZE = 2048.0;
static const uint MAX_POINT_SHADOWS = 4;
static const uint POINT_SHADOW_FACES = 6;
static const float POINT_SHADOW_SIZE = 512.0;

/// One draw: a primitive's index range, placement and material.
struct Instance {
    float4x4 model;
    /// Inverse-transpose of the model's upper 3x3, as columns.
    float4 normal_columns[3];
    /// x: first index, y: base vertex, z: material, w: draw index (model draws
    /// in load order, then world entities).
    uint4 draw;
};

struct Light {
    float4 position_radius;
    float4 color_intensity;
};

struct Vertex {
    /// xyz position, w: texture u.
    float4 position;
    /// xyz normal, w: texture v.
    float4 normal;
    /// xyz tangent, w bitangent sign; zero when the mesh has none.
    float4 tangent;
    float4 color;
};

/// How an instance moved since last frame; mirrors `InstanceMotion` in `lib.rs`.
struct InstanceMotion {
    /// This frame's world space to last frame's.
    float4x4 to_previous;
    /// x: 1 when vertices move on their own (soft bodies, cloth); y: then their
    /// first entry in `previous_positions`.
    uint4 info;
};

#include "matrix_types.slang"

struct Globals {
    float4x4 view_proj;
    float4x4 view;
    float4x4 inverse_view_proj;
    /// xyz: camera position; w: 1 when the swapchain encodes sRGB itself.
    float4 eye;
    /// Scene viewport in swapchain pixels: x, y, width, height.
    float4 viewport;
    /// x, y: projection scales; z: near plane; w: far edge of the cluster slices.
    float4 projection;
    /// x: light count; y: instance count; z: 1 when a GI extension supplies
    /// indirect light (the flat ambient term is then off).
    uint4 counts;
    /// xyz: unit direction the sunlight travels; w: 1 when there is a sun.
    float4 sun_direction;
    /// Linear color times illuminance.
    float4 sun_color;
    /// Last frame's view-projection, for motion vectors.
    float4x4 previous_view_proj;
    /// World to shadow clip space per sun cascade.
    float4x4 shadow_view_proj[CASCADES];
    /// View distance where each cascade's slice ends; read with `shadow_split`.
    float4 shadow_splits[CASCADES / 4];
    /// World size of one shadow texel per cascade; read with `shadow_texel`.
    float4 shadow_texels[CASCADES / 4];
    /// Sky hemisphere radiance; w: 1 when the sun casts shadows.
    float4 sky;
    /// Ground hemisphere radiance.
    float4 ground;
    float4x4 point_shadow_view_proj[MAX_POINT_SHADOWS * POINT_SHADOW_FACES];
    /// Cloud shadow map: world xy of its corner, its size; w: 1 when in use.
    float4 cloud_shadow;
    /// Sun shadow filtering: x: frame index for the noise (0 without TAA);
    /// y: tangent of the sun's angular radius; z: farthest occluder distance
    /// the blocker search is sized for.
    float4 shadow_filter;
    /// x: one bit per shadow layer (cascades, then point faces) this frame redraws.
    uint4 shadow_redraw;
    // GPU pointers to this frame's arrays; set in `upload_frame` (lib.rs).
    Instance* instances;
    Light* lights;
    InstanceMotion* instance_motion;
    /// Last frame's world position of each deforming vertex.
    float4* previous_positions;
    Vertex* vertices;
    uint* indices;
    // Matrix's buffers, in `Matrix::addresses` order.
    MatrixInstance* matrix_instances;
    MatrixAsset* matrix_assets;
    MatrixNode* matrix_nodes;
    MatrixGroup* matrix_groups;
    MatrixCluster* matrix_clusters;
    uint* matrix_indices;
    MatrixBrick* matrix_bricks;
    uint2* matrix_voxels;
    uint2* matrix_entries;
    uint2* matrix_queue;
    uint* matrix_counters;
};

[[vk::binding(0)]] ConstantBuffer<Globals> globals;
[[vk::binding(3)]] RWStructuredBuffer<uint> clusters;
[[vk::binding(7)]] Texture2D<uint2> visibility;
[[vk::binding(8)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> hdr_output;
[[vk::binding(9)]] Texture2D<float4> hdr;
// The thin G-buffer for GI and screen-space passes; see gi_surface.slang.
[[vk::binding(13)]] [[vk::image_format("rgba32ui")]] RWTexture2D<uint4> gi_gbuffer;
[[vk::binding(14)]] [[vk::image_format("rgba32ui")]] RWTexture2D<uint4> gi_receiver;
[[vk::binding(15)]] [[vk::image_format("rg32f")]] RWTexture2D<float2> gi_motion;
[[vk::binding(16)]] Texture2DArray<float> shadow_map;
[[vk::binding(17)]] SamplerComparisonState shadow_sampler;
[[vk::binding(18)]] Texture2DArray<float> point_shadow_map;

/// View distance where sun cascade `c`'s slice ends.
float shadow_split(uint c) {
    return globals.shadow_splits[c >> 2][c & 3];
}

/// World size of one texel of sun cascade `c`.
float shadow_texel(uint c) {
    return globals.shadow_texels[c >> 2][c & 3];
}

/// Vertex of an instance's triangle corner, through its index range.
Vertex instance_vertex(Instance instance, uint corner) {
    return globals.vertices[globals.indices[instance.draw.x + corner] + instance.draw.y];
}

uint cluster_index(uint3 cluster) {
    return (cluster.z * CLUSTER_Y + cluster.y) * CLUSTER_X + cluster.x;
}

/// Logarithmic depth slice of a positive view-space distance.
uint depth_slice(float depth) {
    float near = globals.projection.z;
    float far = globals.projection.w;
    float slice = floor(log(max(depth, near) / near) / log(far / near) * float(CLUSTER_Z));
    return uint(clamp(slice, 0.0, float(CLUSTER_Z - 1)));
}

/// View distance where a slice begins; the last slice runs on far past `far`.
float slice_depth(uint slice) {
    float near = globals.projection.z;
    float far = globals.projection.w;
    return slice >= CLUSTER_Z ? 1.0e6 : near * pow(far / near, float(slice) / float(CLUSTER_Z));
}
