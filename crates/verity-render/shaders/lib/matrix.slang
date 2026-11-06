// Matrix virtual geometry: layouts, visibility IDs and the voxel and imposter
// math shared by the raster passes and the resolve. Layouts mirror `matrix.rs`.
//
// A visibility pixel holds (instance, id). The top two bits of `id` say what it
// is; zero is an ordinary triangle. Matrix instances have their own buffer.

static const uint MATRIX_KIND_SHIFT = 30;
static const uint MATRIX_IMPOSTER = 1;
static const uint MATRIX_TRIANGLE = 2;
static const uint MATRIX_VOXEL = 3;
static const uint MATRIX_ID_MASK = (1u << MATRIX_KIND_SHIFT) - 1;
/// Triangle IDs are `cluster << 7 | triangle`; voxel IDs `brick << 6 | voxel`.
static const uint MATRIX_TRIANGLE_BITS = 7;
static const uint MATRIX_VOXEL_BITS = 6;
static const uint MATRIX_NONE = 0xffffffff;

/// Imposter atlas: `IMPOSTER_GRID`^2 views of the upper hemisphere.
static const uint IMPOSTER_GRID = 12;
static const float IMPOSTER_FRAME = 96.0;
/// Coverage below this is a hole; blended mips dilute thin canopies.
static const float IMPOSTER_ALPHA = 0.33;
static const float IMPOSTER_MAX_LOD = 3.0;

struct MatrixInstance {
    Instance base;
    float4x4 inverse;
    /// World-space bounds: center, radius.
    float4 sphere;
    /// x: asset, y: 1 when two-sided, z: largest axis scale as float bits.
    uint4 info;
};

struct MatrixAsset {
    /// Object-space bounds: center, radius.
    float4 sphere;
    uint first_root;
    uint root_count;
    /// The imposter's bindless textures, or NO_TEXTURE.
    uint imposter_albedo;
    uint imposter_surface;
    uint first_cluster;
    uint cluster_count;
    uint2 padding;
};

struct MatrixNode {
    float4 sphere;
    float error;
    uint group;
    uint first_child;
    uint child_count;
};

struct MatrixGroup {
    float4 sphere;
    float error;
    uint first_cluster;
    uint cluster_count;
    uint padding;
};

struct MatrixCluster {
    float4 sphere;
    uint refined;
    uint first_index;
    uint triangle_count;
    uint first_brick;
    uint brick_count;
    uint depth;
    uint2 padding;
};

struct MatrixBrick {
    float3 origin;
    float voxel_size;
    uint2 mask;
    uint first_voxel;
    uint padding;
};

[[vk::binding(19)]] StructuredBuffer<MatrixInstance> matrix_instances;
[[vk::binding(20)]] StructuredBuffer<MatrixAsset> matrix_assets;
[[vk::binding(23)]] StructuredBuffer<MatrixCluster> matrix_clusters;
[[vk::binding(24)]] StructuredBuffer<uint> matrix_indices;
[[vk::binding(25)]] StructuredBuffer<MatrixBrick> matrix_bricks;
[[vk::binding(26)]] StructuredBuffer<uint2> matrix_voxels;

Vertex matrix_vertex(MatrixInstance instance, uint first_index, uint corner) {
    return vertices[matrix_indices[first_index + corner] + instance.base.draw.y];
}

bool brick_solid(MatrixBrick brick, uint bit) {
    return ((bit < 32 ? brick.mask.x : brick.mask.y) >> (bit & 31) & 1) != 0;
}

/// Walk a ray entering a brick at `start`, in object space. On a hit, `bit` is
/// the voxel and `distance` how far along `direction` its first face lies.
bool brick_march(MatrixBrick brick, float3 start, float3 direction, out uint bit, out float distance) {
    float3 q = clamp((start - brick.origin) / brick.voxel_size, 0.001, 3.999);
    int3 cell = int3(floor(q));
    int3 stride = int3(sign(direction));
    // Axes the ray does not move along never win the comparison below.
    float3 safe = select(abs(direction) < 1.0e-6, float3(1.0e-6), direction);
    float3 delta = abs(1.0 / safe);
    float3 next = (float3(cell) + float3(stride > 0) - q) / safe;
    float travelled = 0.0;
    bit = 0;
    distance = 0.0;
    for (uint step = 0; step < 10; ++step) {
        if (any(cell < 0) || any(cell > 3))
            return false;
        uint index = uint((cell.z * 4 + cell.y) * 4 + cell.x);
        if (brick_solid(brick, index)) {
            bit = index;
            distance = travelled * brick.voxel_size;
            return true;
        }
        if (next.x <= next.y && next.x <= next.z) {
            travelled = next.x;
            next.x += delta.x;
            cell.x += stride.x;
        } else if (next.y <= next.z) {
            travelled = next.y;
            next.y += delta.y;
            cell.y += stride.y;
        } else {
            travelled = next.z;
            next.z += delta.z;
            cell.z += stride.z;
        }
    }
    return false;
}

/// Index among a brick's solid voxels, which are stored in bit order.
uint voxel_rank(MatrixBrick brick, uint bit) {
    uint low = bit < 32 ? brick.mask.x & ((1u << bit) - 1) : brick.mask.x;
    uint high = bit <= 32 ? 0 : brick.mask.y & ((1u << (bit - 32)) - 1);
    return countbits(low) + countbits(high);
}

float3 decode_octahedral(uint packed) {
    float2 e = float2(packed & 0xffff, packed >> 16) / 65535.0 * 2.0 - 1.0;
    float3 n = float3(e, 1.0 - abs(e.x) - abs(e.y));
    float shift = max(-n.z, 0.0);
    n.x -= n.x >= 0.0 ? shift : -shift;
    n.y -= n.y >= 0.0 ? shift : -shift;
    return normalize(n);
}

float3 srgb_to_linear(float3 c) {
    return select(c <= 0.04045, c / 12.92, pow((c + 0.055) / 1.055, 2.4));
}

/// Upper-hemisphere direction to 0..1, and back.
float2 hemi_encode(float3 d) {
    d /= abs(d.x) + abs(d.y) + abs(d.z);
    return float2(d.x + d.y, d.x - d.y) * 0.5 + 0.5;
}

float3 hemi_decode(float2 uv) {
    float2 o = uv * 2.0 - 1.0;
    float2 xy = float2(o.x + o.y, o.x - o.y) * 0.5;
    return normalize(float3(xy, 1.0 - abs(xy.x) - abs(xy.y)));
}

/// A frame's view axis and the plane axes its image spans.
struct ImposterFrame {
    float3 axis;
    float3 right;
    float3 up;
};

ImposterFrame imposter_frame(uint2 frame) {
    ImposterFrame f;
    f.axis = hemi_decode(float2(frame) / float(IMPOSTER_GRID - 1));
    float3 reference = abs(f.axis.z) > 0.999 ? float3(1, 0, 0) : float3(0, 0, 1);
    f.right = normalize(cross(reference, f.axis));
    f.up = cross(f.axis, f.right);
    return f;
}

float2 imposter_atlas(uint2 frame, float2 uv) {
    return (float2(frame) + uv) / float(IMPOSTER_GRID);
}

struct ImposterHit {
    float alpha;
    float3 albedo;
    /// Object-space normal, and how long the mean of the blended normals was.
    float3 normal;
    float mean_length;
    /// Object-space distance along the ray.
    float distance;
};

/// A ray through an object's imposter, blending the three frames nearest the
/// view with one parallax step each. Colors are stored premultiplied, so
/// filtered edges stay clean once divided by coverage.
ImposterHit imposter_sample(MatrixInstance instance, MatrixAsset asset, float3 eye, float3 direction, float pixels) {
    float3 origin = mul(instance.inverse, float4(eye, 1.0)).xyz;
    float3 ray = normalize(mul(instance.inverse, float4(direction, 0.0)).xyz);
    float3 center = asset.sphere.xyz;
    float radius = asset.sphere.w;
    float3 view = normalize(origin - center);
    view.z = max(view.z, 0.001);
    float2 grid = hemi_encode(normalize(view)) * float(IMPOSTER_GRID - 1);
    uint2 cell = min(uint2(grid), uint2(IMPOSTER_GRID - 2));
    float2 f = grid - float2(cell);
    uint2 frames[3];
    float weights[3];
    if (f.x + f.y < 1.0) {
        frames[0] = cell;
        frames[1] = cell + uint2(1, 0);
        frames[2] = cell + uint2(0, 1);
        weights[0] = 1.0 - f.x - f.y;
        weights[1] = f.x;
        weights[2] = f.y;
    } else {
        frames[0] = cell + uint2(1, 1);
        frames[1] = cell + uint2(0, 1);
        frames[2] = cell + uint2(1, 0);
        weights[0] = f.x + f.y - 1.0;
        weights[1] = 1.0 - f.x;
        weights[2] = 1.0 - f.y;
    }
    float lod = clamp(log2(IMPOSTER_FRAME / max(pixels, 1.0)), 0.0, IMPOSTER_MAX_LOD);
    Texture2D<float4> albedo_map = textures[NonUniformResourceIndex(asset.imposter_albedo)];
    Texture2D<float4> surface_map = textures[NonUniformResourceIndex(asset.imposter_surface)];
    float4 color = 0.0;
    float4 surface = 0.0;
    float distance = 0.0;
    for (uint i = 0; i < 3; ++i) {
        ImposterFrame frame = imposter_frame(frames[i]);
        float along = dot(ray, frame.axis);
        // Rays run against the axis; edge-on frames carry little weight anyway.
        float inverse_along = abs(along) > 0.05 ? 1.0 / along : (along < 0.0 ? -20.0 : 20.0);
        float height_origin = dot(origin - center, frame.axis);
        float t = -height_origin * inverse_along;
        float3 point = origin + ray * t - center;
        float2 uv = float2(dot(point, frame.right), dot(point, frame.up)) / (2.0 * radius) + 0.5;
        float2 atlas = imposter_atlas(frames[i], saturate(uv));
        float4 first = surface_map.SampleLevel(material_sampler, atlas, lod);
        if (first.a > 0.0 && first.r + first.g + first.b > 0.0) {
            // Depth is stored in alpha, premultiplied by coverage in the albedo map.
            float coverage = albedo_map.SampleLevel(material_sampler, atlas, lod).a;
            float height = (first.a / max(coverage, 1.0e-3) * 2.0 - 1.0) * radius;
            t = (height - height_origin) * inverse_along;
            point = origin + ray * t - center;
            uv = float2(dot(point, frame.right), dot(point, frame.up)) / (2.0 * radius) + 0.5;
            atlas = imposter_atlas(frames[i], saturate(uv));
        }
        if (any(uv <= 0.0) || any(uv >= 1.0))
            continue;
        color += albedo_map.SampleLevel(material_sampler, atlas, lod) * weights[i];
        surface += surface_map.SampleLevel(material_sampler, atlas, lod) * weights[i];
        distance += t * weights[i];
    }
    ImposterHit hit;
    hit.alpha = color.a;
    float coverage = max(color.a, 1.0e-3);
    hit.albedo = color.rgb / coverage;
    float3 mean = surface.rgb / coverage * 2.0 - 1.0;
    hit.mean_length = length(mean);
    hit.normal = mean / max(hit.mean_length, 1.0e-4);
    hit.distance = distance;
    return hit;
}
