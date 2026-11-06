// Matrix raster passes. Each tier draws from a selection list into the same
// targets as ordinary geometry; only the visibility ID says which tier it was.
// Variants differ by define, so a pass is a tier plus MATRIX_TARGET:
//   visibility: depth plus IDs; depth: the same without IDs; shadow: light depth.

#include "lib/common.slang"
#include "lib/matrix.slang"
#include "lib/matrix_lists.slang"

#define TARGET_VISIBILITY 0
#define TARGET_DEPTH 1
#define TARGET_SHADOW 2

[[vk::binding(27)]] StructuredBuffer<uint2> matrix_entries;

struct DrawConstants {
    /// Offset of this draw's list in `matrix_entries`.
    uint list;
    /// Shadow passes: the cascade or point-shadow face.
    uint cascade;
};
[[vk::push_constant]] ConstantBuffer<DrawConstants> draw;

float4 to_clip(float4 world) {
#if MATRIX_TARGET == TARGET_SHADOW
    if (draw.cascade < CASCADES)
        return mul(globals.shadow_view_proj[draw.cascade], world);
    return mul(globals.point_shadow_view_proj[draw.cascade - CASCADES], world);
#else
    return mul(globals.view_proj, world);
#endif
}

#if defined(MATRIX_CLUSTERS)

struct VertexOutput {
    float4 position: SV_Position;
    float2 uv: TEXCOORD0;
    float alpha: TEXCOORD1;
    nointerpolation uint2 id: ID;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertex: SV_VertexID, uint entry: SV_InstanceID) {
    uint2 item = matrix_entries[draw.list + entry];
    MatrixInstance instance = matrix_instances[item.x];
    MatrixCluster cluster = matrix_clusters[item.y];
    uint triangle = vertex / 3;
    VertexOutput output;
    output.uv = 0.0;
    output.alpha = 1.0;
    output.id = uint2(item.x, MATRIX_TRIANGLE << MATRIX_KIND_SHIFT | item.y << MATRIX_TRIANGLE_BITS | triangle);
    if (triangle >= cluster.triangle_count) {
        // Every corner of a padding triangle lands on one point, so it is culled.
        output.position = float4(0.0, 0.0, 0.0, 1.0);
        return output;
    }
    Vertex v = matrix_vertex(instance, cluster.first_index, vertex);
    output.position = to_clip(mul(instance.base.model, float4(v.position.xyz, 1.0)));
    output.uv = float2(v.position.w, v.normal.w);
    output.alpha = v.color.a;
    return output;
}

void alphaTest(VertexOutput input) {
    Material material = materials[matrix_instances[input.id.x].base.draw.z];
    if ((material.textures.w & ALPHA_MASK) != 0) {
        float alpha = material.base_color.a * input.alpha;
        if (material.textures.x != NO_TEXTURE)
            alpha *= textures[NonUniformResourceIndex(material.textures.x)].Sample(material_sampler, input.uv).a;
        if (alpha < material.params.w)
            discard;
    }
}

#if MATRIX_TARGET == TARGET_VISIBILITY
[shader("fragment")]
uint2 fragmentMain(VertexOutput input): SV_Target0 {
    alphaTest(input);
    return input.id;
}
#else
[shader("fragment")]
void fragmentMain(VertexOutput input) {
    alphaTest(input);
}
#endif

#elif defined(MATRIX_VOXELS)

struct VertexOutput {
    float4 position: SV_Position;
    float3 object: TEXCOORD0;
    nointerpolation uint2 item: ID;
};

/// Direction of the ray arriving at a world point: from the camera, or a light.
float3 arriving(float3 world) {
#if MATRIX_TARGET == TARGET_SHADOW
    if (draw.cascade < CASCADES)
        return globals.sun_direction.xyz;
    return world - lights[(draw.cascade - CASCADES) / POINT_SHADOW_FACES].position_radius.xyz;
#else
    return world - globals.eye.xyz;
#endif
}

[shader("vertex")]
VertexOutput vertexMain(uint vertex: SV_VertexID, uint entry: SV_InstanceID) {
    uint2 item = matrix_entries[draw.list + entry];
    MatrixInstance instance = matrix_instances[item.x];
    MatrixBrick brick = matrix_bricks[item.y];
    // Two triangles per face. Only faces toward the ray survive: marching
    // starts on the surface the ray enters, whatever the winding.
    uint face = vertex / 6;
    uint axis = face >> 1;
    float side = float(face & 1);
    uint corner = uint[6](0, 1, 2, 0, 2, 3)[vertex % 6];
    float2 square = float2(corner == 1 || corner == 2 ? 1.0 : 0.0, corner >= 2 ? 1.0 : 0.0);
    float3 unit = axis == 0 ? float3(side, square) : axis == 1 ? float3(square.x, side, square.y) : float3(square, side);
    float span = brick.voxel_size * 4.0;
    float3 object = brick.origin + unit * span;
    float3 world = mul(instance.base.model, float4(object, 1.0)).xyz;
    float3 toward = float3(axis == 0, axis == 1, axis == 2) * (side * 2.0 - 1.0);
    float3 center = brick.origin + (float3(0.5) + toward * 0.5) * span;
    float3 center_world = mul(instance.base.model, float4(center, 1.0)).xyz;
    float3 ray = mul(instance.inverse, float4(arriving(center_world), 0.0)).xyz;
    VertexOutput output;
    output.object = object;
    output.item = item;
    output.position = dot(ray, toward) < 0.0 ? to_clip(float4(world, 1.0)) : float4(0.0, 0.0, 0.0, 1.0);
    return output;
}

bool voxelHit(VertexOutput input, out float depth, out uint id) {
    MatrixInstance instance = matrix_instances[input.item.x];
    MatrixBrick brick = matrix_bricks[input.item.y];
    float3 world = mul(instance.base.model, float4(input.object, 1.0)).xyz;
    float3 ray = normalize(mul(instance.inverse, float4(arriving(world), 0.0)).xyz);
    uint bit;
    float distance;
    depth = 0.0;
    id = 0;
    if (!brick_march(brick, input.object, ray, bit, distance))
        return false;
#if MATRIX_TARGET == TARGET_SHADOW
    // Depth bias does not apply to depth written here, and voxels are coarse.
    distance += brick.voxel_size * 1.5;
#endif
    float4 clip = to_clip(mul(instance.base.model, float4(input.object + ray * distance, 1.0)));
    depth = clip.z / clip.w;
    id = MATRIX_VOXEL << MATRIX_KIND_SHIFT | input.item.y << MATRIX_VOXEL_BITS | bit;
    return true;
}

#if MATRIX_TARGET == TARGET_VISIBILITY
struct FragmentOutput {
    uint2 id: SV_Target0;
    float depth: SV_DepthLessEqual;
};

[shader("fragment")]
FragmentOutput fragmentMain(VertexOutput input) {
    FragmentOutput output;
    uint id;
    if (!voxelHit(input, output.depth, id))
        discard;
    output.id = uint2(input.item.x, id);
    return output;
}
#elif MATRIX_TARGET == TARGET_DEPTH
[shader("fragment")]
float fragmentMain(VertexOutput input): SV_DepthLessEqual {
    float depth;
    uint id;
    if (!voxelHit(input, depth, id))
        discard;
    return depth;
}
#else
[shader("fragment")]
float fragmentMain(VertexOutput input): SV_Depth {
    float depth;
    uint id;
    if (!voxelHit(input, depth, id))
        discard;
    return depth;
}
#endif

#elif defined(MATRIX_IMPOSTERS)

struct VertexOutput {
    float4 position: SV_Position;
    float3 world: TEXCOORD0;
    float pixels: TEXCOORD1;
    nointerpolation uint instance: ID;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertex: SV_VertexID, uint entry: SV_InstanceID) {
    uint index = matrix_entries[draw.list + entry].x;
    MatrixInstance instance = matrix_instances[index];
    float3 center = instance.sphere.xyz;
    float radius = instance.sphere.w;
    float3 offset = globals.eye.xyz - center;
    float distance = length(offset);
    float3 toward = offset / max(distance, 1.0e-4);
    float3 right = normalize(cross(abs(toward.z) > 0.999 ? float3(1, 0, 0) : float3(0, 0, 1), toward));
    float3 up = cross(toward, right);
    uint corner = uint[6](0, 1, 2, 0, 2, 3)[vertex];
    float2 square = float2(corner == 1 || corner == 2 ? 1.0 : -1.0, corner >= 2 ? 1.0 : -1.0);
    // The card sits at the front of the bounds, so hits only ever lie behind it.
    float3 world = center + toward * radius + (right * square.x + up * square.y) * radius;
    VertexOutput output;
    output.position = mul(globals.view_proj, float4(world, 1.0));
    output.world = world;
    output.pixels = 2.0 * radius * abs(globals.projection.y) * 0.5 * globals.viewport.w / max(distance, globals.projection.z);
    output.instance = index;
    return output;
}

bool imposterHit(VertexOutput input, out float depth) {
    MatrixInstance instance = matrix_instances[input.instance];
    MatrixAsset asset = matrix_assets[instance.info.x];
    float3 direction = normalize(input.world - globals.eye.xyz);
    ImposterHit hit = imposter_sample(instance, asset, globals.eye.xyz, direction, input.pixels);
    depth = 0.0;
    if (hit.alpha < IMPOSTER_ALPHA)
        return false;
    float3 origin = mul(instance.inverse, float4(globals.eye.xyz, 1.0)).xyz;
    float3 ray = normalize(mul(instance.inverse, float4(direction, 0.0)).xyz);
    float4 clip = mul(globals.view_proj, mul(instance.base.model, float4(origin + ray * hit.distance, 1.0)));
    depth = clip.z / clip.w;
    return true;
}

#if MATRIX_TARGET == TARGET_VISIBILITY
struct FragmentOutput {
    uint2 id: SV_Target0;
    float depth: SV_DepthLessEqual;
};

[shader("fragment")]
FragmentOutput fragmentMain(VertexOutput input) {
    FragmentOutput output;
    if (!imposterHit(input, output.depth))
        discard;
    output.id = uint2(input.instance, MATRIX_IMPOSTER << MATRIX_KIND_SHIFT);
    return output;
}
#else
[shader("fragment")]
float fragmentMain(VertexOutput input): SV_DepthLessEqual {
    float depth;
    if (!imposterHit(input, depth))
        discard;
    return depth;
}
#endif

#elif defined(MATRIX_BAKE)

// Draw one instance's source clusters into every frame of its imposter atlas.

struct VertexOutput {
    float4 position: SV_Position;
    float3 normal: NORMAL;
    float2 uv: TEXCOORD0;
    float4 color: COLOR;
    float height: TEXCOORD1;
    nointerpolation uint material: MATERIAL;
};

struct FragmentOutput {
    float4 albedo: SV_Target0;
    float4 surface: SV_Target1;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertex: SV_VertexID, uint entry: SV_InstanceID) {
    // `list` is the instance; `cascade` the frame, as y * grid + x.
    MatrixInstance instance = matrix_instances[draw.list];
    MatrixAsset asset = matrix_assets[instance.info.x];
    MatrixCluster cluster = matrix_clusters[asset.first_cluster + entry];
    VertexOutput output;
    output.normal = 0.0;
    output.uv = 0.0;
    output.color = 0.0;
    output.height = 0.0;
    output.material = instance.base.draw.z;
    output.position = float4(0.0, 0.0, 0.0, 1.0);
    if (cluster.refined != MATRIX_NONE || vertex / 3 >= cluster.triangle_count)
        return output;
    Vertex v = matrix_vertex(instance, cluster.first_index, vertex);
    ImposterFrame frame = imposter_frame(uint2(draw.cascade % IMPOSTER_GRID, draw.cascade / IMPOSTER_GRID));
    float3 local = v.position.xyz - asset.sphere.xyz;
    float radius = asset.sphere.w;
    float height = dot(local, frame.axis) / radius;
    output.position = float4(dot(local, frame.right) / radius, dot(local, frame.up) / radius, height * 0.5 + 0.5, 1.0);
    output.normal = v.normal.xyz;
    output.uv = float2(v.position.w, v.normal.w);
    output.color = v.color;
    output.height = height * 0.5 + 0.5;
    return output;
}

[shader("fragment")]
FragmentOutput fragmentMain(VertexOutput input) {
    Material material = materials[input.material];
    float4 base = material.base_color * input.color;
    if (material.textures.x != NO_TEXTURE)
        base *= textures[NonUniformResourceIndex(material.textures.x)].Sample(material_sampler, input.uv);
    if ((material.textures.w & ALPHA_MASK) != 0 && base.a < material.params.w)
        discard;
    ImposterFrame frame = imposter_frame(uint2(draw.cascade % IMPOSTER_GRID, draw.cascade / IMPOSTER_GRID));
    float3 normal = normalize(input.normal);
    // Leaves have two sides: keep the one the frame sees.
    if (dot(normal, frame.axis) < 0.0)
        normal = -normal;
    FragmentOutput output;
    output.albedo = float4(base.rgb, 1.0);
    output.surface = float4(normal * 0.5 + 0.5, input.height);
    return output;
}

#endif
