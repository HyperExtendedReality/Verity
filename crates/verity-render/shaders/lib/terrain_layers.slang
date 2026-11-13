// Textured terrain from virtual-textured layers (see `src/virtual_texture.rs`).
//
// Roles are weighted from slope, depth below the original ground, height,
// wetness and noise, as the procedural ground is; the two strongest blend by
// their height maps. Each layer is projected along the axes the normal leans
// towards, so dug walls and overhangs are textured as well as the ground.

static const uint LAYER_HEADER = 8;
static const uint LAYER_WORDS = 4;
static const uint MAX_LAYERS = 16;
static const uint LAYER_TABLE = LAYER_HEADER + LAYER_WORDS * MAX_LAYERS;
static const uint NO_PAGE = 0xffffffff;
static const float PAGE_TEXELS = 128.0;
static const float PAGE_BORDER = 4.0;
static const float SLOT_TEXELS = PAGE_TEXELS + 2.0 * PAGE_BORDER;

/// Mirrors `TerrainRole`.
static const uint ROLE_GRASS = 0;
static const uint ROLE_MEADOW = 1;
static const uint ROLE_SOIL = 2;
static const uint ROLE_LOAM = 3;
static const uint ROLE_SUBSOIL = 4;
static const uint ROLE_ROCK = 5;
static const uint ROLE_SCREE = 6;
static const uint ROLE_GRAVEL = 7;
static const uint ROLE_SAND = 8;
static const uint ROLES = 9;

/// Header: frame, layer count, albedo and surface atlas textures, pages along
/// the largest level, paged levels, pages per layer, slots along the atlas.
/// Then per layer: albedo and surface tail textures and metres per tile. Then
/// each page's atlas slot.
[[vk::binding(30, MATERIAL_SET)]] StructuredBuffer<uint> layer_table;
/// One bit per page the resolve wanted, read back by the CPU.
[[vk::binding(31, MATERIAL_SET)]] RWStructuredBuffer<uint> layer_feedback;

bool terrain_layers_ready() {
    return layer_table[1] != 0;
}

struct LayerSample {
    /// Base color, height in alpha.
    float4 albedo;
    /// Normal xy in red and green, roughness, ambient occlusion.
    float4 surface;
};

LayerSample mix(LayerSample a, LayerSample b, float t) {
    LayerSample s;
    s.albedo = lerp(a.albedo, b.albedo, t);
    s.surface = lerp(a.surface, b.surface, t);
    return s;
}

/// Pages in the levels before `mip`: a geometric series in quarters.
uint page_offset(uint mip) {
    uint across = layer_table[4];
    uint level = across >> mip;
    return (across * across - level * level) / 3 * 4;
}

uint page_id(uint layer, uint mip, float2 uv, out float2 local) {
    uint across = layer_table[4] >> mip;
    float2 scaled = frac(uv) * across;
    uint2 page = min(uint2(scaled), across - 1);
    local = scaled - page;
    return layer * layer_table[6] + page_offset(mip) + page.y * across + page.x;
}

/// A resident level at or below `PAGE_TEXELS`, with hardware mips.
LayerSample sample_tail(uint layer, float2 uv, float lod) {
    uint base = LAYER_HEADER + layer * LAYER_WORDS;
    LayerSample s;
    s.albedo = textures[NonUniformResourceIndex(layer_table[base])].SampleLevel(material_sampler, uv, lod);
    s.surface = textures[NonUniformResourceIndex(layer_table[base + 1])].SampleLevel(material_sampler, uv, lod);
    return s;
}

/// Level `mip`, or the finest resident level above it.
LayerSample sample_mip(uint layer, float2 uv, uint mip) {
    for (uint m = mip; m < layer_table[5]; ++m) {
        float2 local;
        uint slot = layer_table[LAYER_TABLE + page_id(layer, m, uv, local)];
        if (slot != NO_PAGE) {
            uint across = layer_table[7];
            float2 texel = float2(slot % across, slot / across) * SLOT_TEXELS + PAGE_BORDER + local * PAGE_TEXELS;
            float2 atlas = texel / (across * SLOT_TEXELS);
            LayerSample s;
            // The atlases are the same for every pixel.
            s.albedo = textures[layer_table[2]].SampleLevel(material_sampler, atlas, 0.0);
            s.surface = textures[layer_table[3]].SampleLevel(material_sampler, atlas, 0.0);
            return s;
        }
    }
    return sample_tail(layer, uv, 0.0);
}

/// Trilinear between the two levels around `lod`, asking for the finer one's page.
LayerSample sample_layer(uint layer, float2 uv, float lod, bool feedback) {
    uint mips = layer_table[5];
    lod = max(lod, 0.0);
    if (lod >= mips)
        return sample_tail(layer, uv, lod - mips);
    uint mip = uint(lod);
    if (feedback) {
        float2 local;
        uint id = page_id(layer, mip, uv, local);
        InterlockedOr(layer_feedback[id >> 5], 1u << (id & 31));
    }
    float t = lod - mip;
    // Close to a level, the other one adds nothing visible.
    if (t < 0.1)
        return sample_mip(layer, uv, mip);
    LayerSample coarse = mip + 1 < mips ? sample_mip(layer, uv, mip + 1) : sample_tail(layer, uv, 0.0);
    if (t > 0.9)
        return coarse;
    return mix(sample_mip(layer, uv, mip), coarse, t);
}

/// Move `amount` of every role's weight to `role`.
void cover(inout float weights[ROLES], uint role, float amount) {
    for (uint i = 0; i < ROLES; ++i)
        weights[i] *= 1.0 - amount;
    weights[role] += amount;
}

/// One layer through every projection the normal leans towards.
LayerSample sample_projected(uint layer, float3 p, float3 axes, float footprint, bool feedback,
                             out float3 bump) {
    float tile = asfloat(layer_table[LAYER_HEADER + layer * LAYER_WORDS + 2]);
    float texels = layer_table[4] * PAGE_TEXELS;
    float lod = log2(max(footprint * texels / tile, 1.0e-6));
    // Texture up is world up on walls; v runs down the image.
    float2 uvs[3] = { float2(p.y, -p.z), float2(p.x, -p.z), float2(p.x, -p.y) };
    float3 tangents[3] = { float3(0, 1, 0), float3(1, 0, 0), float3(1, 0, 0) };
    float3 bitangents[3] = { float3(0, 0, 1), float3(0, 0, 1), float3(0, 1, 0) };
    LayerSample total;
    total.albedo = 0.0;
    total.surface = 0.0;
    bump = 0.0;
    for (uint axis = 0; axis < 3; ++axis) {
        if (axes[axis] <= 0.0)
            continue;
        LayerSample s = sample_layer(layer, uvs[axis] / tile, lod, feedback);
        total.albedo += s.albedo * axes[axis];
        total.surface += s.surface * axes[axis];
        float2 slope = s.surface.xy * 2.0 - 1.0;
        bump += (tangents[axis] * slope.x + bitangents[axis] * slope.y) * axes[axis];
    }
    return total;
}

/// The textured counterpart of `terrain_surface`, with the same inputs. Asks
/// for pages when `feedback` is set.
TerrainSurface terrain_layered(float3 p, float3 normal, float surface_height, float wet, float footprint,
                               float sea_level, bool feedback) {
    float broad = terrain_noise(p * 0.055).w;
    float patches = smoothstep(0.35, 0.65, terrain_noise(p * 0.021 + 17.0).w);
    float depth = max(surface_height - p.z, 0.0);
    wet = saturate(wet);
    float cliff = 1.0 - smoothstep(0.55, 0.86, normal.z);
    float buried = smoothstep(0.5, 2.4, depth);
    float bedrock = smoothstep(2.0, 5.0, depth);
    float vegetation = smoothstep(0.3, 0.65, broad + wet * 0.18) * (1.0 - buried);
    float shore = 1.0 - smoothstep(0.6, 2.4, p.z - sea_level + broad * 1.2);
    float riverbed = smoothstep(0.55, 0.85, wet) * smoothstep(0.85, 0.97, normal.z) * (1.0 - buried);
    float high = smoothstep(48.0, 70.0, p.z + broad * 10.0) * (1.0 - buried);

    float weights[ROLES];
    for (uint i = 0; i < ROLES; ++i)
        weights[i] = 0.0;
    // Patches split both the bare ground and the grass on it into two looks.
    weights[ROLE_SOIL] = (1.0 - vegetation) * (1.0 - patches);
    weights[ROLE_LOAM] = (1.0 - vegetation) * patches;
    weights[ROLE_GRASS] = vegetation * (1.0 - patches);
    weights[ROLE_MEADOW] = vegetation * patches;
    cover(weights, ROLE_GRAVEL, riverbed);
    cover(weights, ROLE_SCREE, high);
    cover(weights, ROLE_SUBSOIL, buried);
    cover(weights, ROLE_ROCK, max(cliff, bedrock));
    cover(weights, ROLE_SAND, shore * (1.0 - cliff) * (1.0 - bedrock));

    // The two strongest roles; the rest are too faint to see under them.
    uint first = 0;
    for (uint i = 1; i < ROLES; ++i)
        if (weights[i] > weights[first])
            first = i;
    uint second = first == 0 ? 1 : 0;
    for (uint i = 0; i < ROLES; ++i)
        if (i != first && weights[i] > weights[second])
            second = i;

    // Projections the normal leans towards. Fading the faint ones to zero
    // skips them without a seam where they drop out.
    float3 axes = pow(abs(normal), 4.0);
    axes /= axes.x + axes.y + axes.z;
    axes = max(axes - 0.1, 0.0);
    axes /= axes.x + axes.y + axes.z;

    float3 bump_a, bump_b;
    LayerSample a = sample_projected(first, p, axes, footprint, feedback, bump_a);
    float share = weights[second] / max(weights[first] + weights[second], 1.0e-4);
    LayerSample b = a;
    bump_b = bump_a;
    // A faint second role fades out entirely, so most pixels sample one layer.
    float second_fade = smoothstep(0.12, 0.25, share);
    if (second_fade > 0.0)
        b = sample_projected(second, p, axes, footprint, feedback, bump_b);
    // Height blend: the taller material shows through where the two meet.
    float height_a = a.albedo.a + (1.0 - share);
    float height_b = b.albedo.a + share;
    float top = max(height_a, height_b) - 0.2;
    float wa = max(height_a - top, 0.0);
    float wb = max(height_b - top, 0.0);
    float t = wb / max(wa + wb, 1.0e-4) * second_fade;
    LayerSample s = mix(a, b, t);
    float3 bump = lerp(bump_a, bump_b, t);

    float3 albedo = s.albedo.rgb * lerp(1.0, s.surface.a, 0.6) * (0.88 + 0.24 * broad);
    float snow = smoothstep(68.0, 91.0, p.z + broad * 7.0) * smoothstep(0.55, 0.9, normal.z) * (1.0 - buried);
    albedo = lerp(albedo, float3(0.72, 0.78, 0.80), snow);

    TerrainSurface surface;
    surface.albedo = albedo * (1.0 - wet * 0.13);
    surface.roughness = lerp(s.surface.b, s.surface.b * 0.75, wet) * (1.0 - snow * 0.1);
    // The bump lies in each projection's plane; keep only its part along the surface.
    float3 tilt = bump - normal * dot(bump, normal);
    surface.normal = normalize(normal + tilt * (1.0 - snow));
    return surface;
}
