// Physically based sky and volumetric clouds, drawn behind the scene.
//
// - Shape: Schneider's Nubis model: a low-resolution "dimensional profile"
//   (tileable Perlin-Worley, a height profile per cloud genus and a coverage
//   remap, SIGGRAPH 2015/2017) up-rezzed by eroding it with wispy and billowy
//   alligator noise and sharpening the result (Nubis Evolved 2022, Nubis³ 2023).
// - Lighting: Hillaire's energy-conserving integration (SIGGRAPH 2016) with
//   Wrenninge's multiple-scattering octaves, and the Jendersie & d'Eon (2023)
//   HG + Draine fit to Mie scattering by water droplets as the phase function.
//   Sunlight takes a few detailed samples near each point and reads the summed
//   density beyond from a precomputed grid (Nubis³ voxel lighting).
// - Cost: traced at half resolution with a per-frame interleaved-gradient
//   jitter and steps that lengthen with distance, then reprojected through the
//   cloud's depth and accumulated.
// - Sky: single-scattering Rayleigh + Mie atmosphere over a spherical planet.
// - Voxel cloudscape (Nubis³): an authored dimensional profile, detail type
//   and density scale, up-rezzed with Guerrilla's curly-alligator and
//   alligator noise (mip by distance, twice-folded noise near the camera) and
//   sphere traced, with sunlight summed into its own voxel grid. Storage
//   follows VQVDB (Zephir 2025) rather than Nubis' dense grids: only cloudy
//   8^3 bricks, as one-byte codes into a codebook of 2^3 blocks, decoded once
//   into a brick atlas; the brick table holds, for empty bricks, the distance
//   to the nearest cloudy one, which replaces Nubis' dense SDF.

#include "lib/common.slang"

static const float EARTH = 6360.0e3;
static const float ATMOSPHERE = 6460.0e3;
static const float3 RAYLEIGH = float3(5.802e-6, 13.558e-6, 33.1e-6);
static const float MIE_SCATTER = 3.996e-6;
static const float MIE_EXTINCT = 4.40e-6;
static const float3 OZONE = float3(0.650e-6, 1.881e-6, 0.085e-6);
/// Farthest a camera ray marches into the cloud layer.
static const float MAX_MARCH = 60.0e3;
static const float SHAPE_SCALE = 1.0 / 7000.0;
static const float DETAIL_SCALE = 1.0 / 1000.0;
static const float WEATHER_SCALE = 1.0 / 60000.0;
static const float SUN_RADIUS = 0.0047;
static const float CIRRUS_ALTITUDE = 9000.0;
/// Farthest the light grid sums density toward the sun.
static const float LIGHT_REACH = 16.0e3;

struct Params {
    /// Layer bottom, top (metres above ground), coverage, extinction per metre.
    float4 layer;
    /// x: genus; yz: wind and seed offset; w: cirrus amount.
    float4 shape;
    /// x: frame number; y: 1 when the history is valid; zw: half-res size.
    uint4 frame;
    /// Cloud shadow map: world xy of its corner, its size; w unused.
    float4 region;
    /// Voxel cloudscape: xyz its world box's low corner; w metres per voxel,
    /// 0 with no cloudscape.
    float4 scape_min;
    /// xyz: bricks along x, y, z; w: atlas bricks along x.
    uint4 scape_grid;
    /// x: atlas bricks along y; y: cloudy bricks; z: codebook entries.
    uint4 scape_info;
};
[[vk::push_constant]] ConstantBuffer<Params> params;

[[vk::binding(0, 1)]] [[vk::image_format("rgba8")]] RWTexture3D<float4> shape_out;
[[vk::binding(1, 1)]] [[vk::image_format("r8")]] RWTexture3D<float> atlas_out;
[[vk::binding(2, 1)]] Texture3D<float4> shape_noise;
[[vk::binding(3, 1)]] Texture3D<float4> detail_noise;
[[vk::binding(4, 1)]] SamplerState wrap_sampler;
[[vk::binding(5, 1)]] SamplerState clamp_sampler;
[[vk::binding(6, 1)]] Texture2D<float4> history;
[[vk::binding(7, 1)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> current;
[[vk::binding(8, 1)]] [[vk::image_format("r32f")]] RWTexture2D<float> shadow_out;
[[vk::binding(9, 1)]] [[vk::image_format("rgba16f")]] RWTexture3D<float4> light_out;
[[vk::binding(10, 1)]] Texture3D<float4> light_grid;
/// The cloudscape's decoded dimensional profile, a brick per 10^3 cell.
[[vk::binding(11, 1)]] Texture3D<float> atlas;
/// Per brick: its atlas slot when cloudy; else EMPTY | bricks to the nearest cloudy one.
[[vk::binding(12, 1)]] Texture3D<uint> leaves;
/// The .vqcloud payload: codebook, bricks, gains, attributes, codes.
[[vk::binding(13, 1)]] StructuredBuffer<uint> vq;
[[vk::binding(14, 1)]] [[vk::image_format("rgba16f")]] RWTexture3D<float4> scape_light_out;
[[vk::binding(15, 1)]] Texture3D<float4> scape_light;

float remap(float v, float a, float b, float c, float d) {
    return c + (v - a) / (b - a) * (d - c);
}

// ---- Noise bake ---------------------------------------------------------

uint3 pcg3d(uint3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}

float3 hash3(int3 cell, int period) {
    uint3 wrapped = uint3((cell % period + period) % period);
    return float3(pcg3d(wrapped)) * (1.0 / 4294967295.0);
}

/// Inverted cellular noise, tiling every `period` cells over the unit cube.
float worley(float3 p, int period) {
    p *= period;
    int3 cell = int3(floor(p));
    float3 f = frac(p);
    float nearest = 1.0;
    for (int z = -1; z <= 1; ++z)
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x) {
                int3 o = int3(x, y, z);
                float3 d = float3(o) + hash3(cell + o, period) - f;
                nearest = min(nearest, dot(d, d));
            }
    return 1.0 - sqrt(nearest);
}

float worley_fbm(float3 p, int period) {
    return worley(p, period) * 0.625 + worley(p, period * 2) * 0.25 + worley(p, period * 4) * 0.125;
}

/// Gradient noise in [-1, 1], tiling every `period` cells.
float perlin(float3 p, int period) {
    p *= period;
    int3 cell = int3(floor(p));
    float3 f = frac(p);
    float3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float result[8];
    for (int i = 0; i < 8; ++i) {
        int3 o = int3(i & 1, (i >> 1) & 1, i >> 2);
        float3 g = normalize(hash3(cell + o, period) * 2.0 - 1.0 + 1.0e-4);
        result[i] = dot(g, f - float3(o));
    }
    float x0 = lerp(lerp(result[0], result[1], u.x), lerp(result[2], result[3], u.x), u.y);
    float x1 = lerp(lerp(result[4], result[5], u.x), lerp(result[6], result[7], u.x), u.y);
    return lerp(x0, x1, u.z) * 1.15;
}

float perlin_fbm(float3 p, int period) {
    float sum = 0.0, amplitude = 1.0, total = 0.0;
    for (int i = 0; i < 5; ++i) {
        sum += perlin(p, period << i) * amplitude;
        total += amplitude;
        amplitude *= 0.5;
    }
    return sum / total;
}

/// R: Perlin-Worley; GBA: Worley FBM at rising frequencies.
[shader("compute")]
[numthreads(4, 4, 4)]
void shapeMain(uint3 id: SV_DispatchThreadID) {
    uint size;
    uint unused;
    shape_out.GetDimensions(size, unused, unused);
    float3 p = (float3(id) + 0.5) / float(size);
    float w = worley_fbm(p, 4);
    float base = saturate(perlin_fbm(p, 4) * 0.5 + 0.5);
    float perlin_worley = saturate(remap(base, w - 1.0, 1.0, 0.0, 1.0));
    shape_out[id] = float4(perlin_worley, w, worley_fbm(p, 8), worley_fbm(p, 16));
}

// ---- Atmosphere ---------------------------------------------------------

/// Near and far distances to a sphere around the origin; far < 0 on a miss.
float2 sphere(float3 origin, float3 direction, float radius) {
    float b = dot(origin, direction);
    float c = dot(origin, origin) - radius * radius;
    float h = b * b - c;
    if (h < 0.0)
        return -1.0;
    h = sqrt(h);
    return float2(-b - h, -b + h);
}

float3 extinction(float height) {
    float rayleigh = exp(-height / 8000.0);
    float mie = exp(-height / 1200.0);
    float ozone = max(0.0, 1.0 - abs(height - 25.0e3) / 15.0e3);
    return RAYLEIGH * rayleigh + MIE_EXTINCT * mie + OZONE * ozone;
}

/// Transmittance from a point (relative to the planet's center) to space.
float3 sun_transmittance(float3 position, float3 to_sun) {
    if (sphere(position, to_sun, EARTH).x > 0.0)
        return 0.0;
    float span = sphere(position, to_sun, ATMOSPHERE).y;
    float3 depth = 0.0;
    const int STEPS = 6;
    for (int i = 0; i < STEPS; ++i) {
        float3 p = position + to_sun * (span * (i + 0.5) / STEPS);
        depth += extinction(max(0.0, length(p) - EARTH));
    }
    return exp(-depth * span / STEPS);
}

float rayleigh_phase(float mu) {
    return 3.0 / (16.0 * PI) * (1.0 + mu * mu);
}

float hg(float mu, float g) {
    float d = 1.0 + g * g - 2.0 * g * mu;
    return (1.0 - g * g) / (4.0 * PI * d * sqrt(d));
}

/// Single-scattered sky radiance along a ray, per unit sun illuminance.
float3 atmosphere(float3 origin, float3 direction, float3 to_sun, float shade = 1.0) {
    float2 top = sphere(origin, direction, ATMOSPHERE);
    float2 ground = sphere(origin, direction, EARTH);
    float end = ground.x > 0.0 ? ground.x : top.y;
    float start = max(top.x, 0.0);
    if (end <= start)
        return 0.0;
    float mu = dot(direction, to_sun);
    float rp = rayleigh_phase(mu);
    float mp = hg(mu, 0.8);
    const int STEPS = 12;
    float dt = (end - start) / STEPS;
    float3 transmittance = 1.0;
    float3 radiance = 0.0;
    for (int i = 0; i < STEPS; ++i) {
        float3 p = origin + direction * (start + dt * (i + 0.5));
        float height = max(0.0, length(p) - EARTH);
        float3 scatter = RAYLEIGH * exp(-height / 8000.0) * rp + MIE_SCATTER * exp(-height / 1200.0) * mp;
        radiance += transmittance * sun_transmittance(p, to_sun) * scatter * dt;
        transmittance *= exp(-extinction(height) * dt);
    }
    if (ground.x > 0.0) {
        // A Lambertian ground of albedo 0.3 past the scene's own geometry.
        float3 p = origin + direction * ground.x;
        radiance += transmittance * sun_transmittance(p, to_sun) * max(dot(normalize(p), to_sun), 0.0) * (0.3 / PI) * shade;
    }
    return radiance;
}

// ---- Clouds -------------------------------------------------------------

/// Jendersie & d'Eon 2023: HG + Draine blend fitted to droplets of
/// `diameter` micrometres (valid 5..50).
float draine(float mu, float g, float alpha) {
    float d = 1.0 + g * g - 2.0 * g * mu;
    return (1.0 - g * g) * (1.0 + alpha * mu * mu)
         / (4.0 * PI * (1.0 + alpha * (1.0 + 2.0 * g * g) / 3.0) * d * sqrt(d));
}

float droplet_phase(float mu, float diameter) {
    float g_hg = exp(-0.0990567 / (diameter - 1.67154));
    float g_d = exp(-2.20679 / (diameter + 3.91029) - 0.428934);
    float alpha = exp(3.62489 - 8.29288 / (diameter + 5.52825));
    float w_d = exp(-0.599085 / (diameter - 0.641583) - 0.665888);
    return lerp(hg(mu, g_hg), draine(mu, g_d, alpha), w_d);
}

/// Vertical density profile of a genus 0 (stratus) .. 1 (cumulus) at
/// `h` (0 bottom .. 1 top of the layer).
float height_profile(float h, float genus) {
    const float4 STRATUS = float4(0.0, 0.08, 0.12, 0.25);
    const float4 STRATOCUMULUS = float4(0.0, 0.15, 0.35, 0.6);
    const float4 CUMULUS = float4(0.0, 0.1, 0.65, 1.0);
    float4 g = genus < 0.5 ? lerp(STRATUS, STRATOCUMULUS, genus * 2.0)
                           : lerp(STRATOCUMULUS, CUMULUS, genus * 2.0 - 1.0);
    return smoothstep(g.x, g.y, h) * (1.0 - smoothstep(g.z, g.w, h));
}

/// 0 over the voxel cloudscape's footprint rising to 1 two kilometres out:
/// the layer's own clouds give way to it.
float scape_clearing(float2 xy) {
    if (params.scape_min.w <= 0.0)
        return 1.0;
    float2 low = params.scape_min.xy;
    float2 high = low + float2(params.scape_grid.xy) * 8.0 * params.scape_min.w;
    return saturate((length(max(max(low - xy, xy - high), 0.0)) - 300.0) / 1700.0);
}

/// Where the noise is read for world position `p`: blown along by the wind,
/// which shears tops downwind as convective clouds lean.
float3 noise_position(float3 p, float h) {
    return p + float3(params.shape.yz, 0.0) * (1.0 + h * 0.2);
}

/// The low-resolution shape of the clouds (Nubis' dimensional profile, 0..1)
/// at world position `p`, `h` up through the layer. `type` runs from wispy
/// (0) to billowy (1).
float cloud_profile(float3 p, float h, out float type) {
    type = 0.0;
    if (h <= 0.0 || h >= 1.0)
        return 0.0;
    float3 wind = float3(params.shape.yz, 0.0);
    float coverage = params.layer.z;
    // Weather: where clouds form, from the shape noise's Perlin-Worley channel.
    float2 weather_uv = (p.xy + wind.xy * 0.5) * WEATHER_SCALE;
    float weather = shape_noise.SampleLevel(wrap_sampler, float3(weather_uv, 0.3), 0).r * 0.7
                  + shape_noise.SampleLevel(wrap_sampler, float3(weather_uv * 3.1, 0.7), 0).r * 0.3;
    // The channel sits around 0.45..0.85; spread it over 0..1.
    float weather_n = saturate(remap(weather, 0.45, 0.85, 0.0, 1.0));
    // Cloudiness below this is clear sky; tuned so coverage is roughly the
    // fraction of sky covered. Weather can add at most 0.4 of it.
    float threshold = lerp(0.62, 0.1, coverage);
    if (coverage <= 0.0 || weather_n * 0.4 + 0.6 <= threshold)
        return 0.0;
    // Formations drift around the chosen genus across the sky: towers in one
    // place, flatter decks or scattered puffs in another.
    float variety = shape_noise.SampleLevel(wrap_sampler, float3(weather_uv * 0.7 + 0.37, 0.55), 0).g;
    float genus = saturate(params.shape.x + (variety - 0.55) * 1.6);
    // Stratus frays into wisps and cumulus billows, above a wispy base.
    type = lerp(0.3, 1.0, genus) * saturate(h * 6.0);
    float4 n = shape_noise.SampleLevel(wrap_sampler, noise_position(p, h) * SHAPE_SCALE, 0);
    float fbm = n.g * 0.625 + n.b * 0.25 + n.a * 0.125;
    float shape = saturate(remap(n.r, fbm - 1.0, 1.0, 0.0, 1.0));
    shape = saturate(remap(shape, 0.6, 0.9, 0.0, 1.0));
    float cloudiness = (shape * 0.6 + weather_n * 0.4) * height_profile(h, genus);
    return saturate(remap(cloudiness, threshold, 1.0, 0.0, 1.0)) * scape_clearing(p.xy);
}

/// Nubis' noise composite at a profile and type: wisps grow finer toward the
/// edge, billows rounder (Nubis³, the up-rez).
float noise_composite(float4 n, float profile, float type) {
    float wispy = lerp(n.r, n.g, profile);
    float billowy = lerp(n.b * 0.3, n.a * 0.3, pow(profile, 0.25));
    return lerp(wispy, billowy, type);
}

/// Nubis up-rez of the layer: cloud density (0..1) from a nonzero `profile`,
/// eroded by detail noise read at mip `lod`, and sharpened.
float cloud_detail(float3 p, float h, float profile, float type, float lod) {
    float4 n = detail_noise.SampleLevel(wrap_sampler, noise_position(p, h) * DETAIL_SCALE, lod);
    return pow(saturate(remap(profile, noise_composite(n, profile, type), 1.0, 0.0, 1.0)), 0.6);
}

/// Cloud density (0..1) at world position `p`, `h` up through the layer.
float cloud_density(float3 p, float h, float lod) {
    float type;
    float profile = cloud_profile(p, h, type);
    return profile > 0.0 ? cloud_detail(p, h, profile, type, lod) : 0.0;
}

/// Toward the sun, held off the horizon so paths through the layer stay finite.
float3 raised_sun() {
    float3 to_sun = -globals.sun_direction.xyz;
    to_sun.z = max(to_sun.z, 0.08);
    return normalize(to_sun);
}

/// Where the light grid holds world `xy` at layer height `h`: it spans twice
/// the shadow map around the same center.
float3 light_coord(float2 xy, float h) {
    return float3((xy - params.region.xy) / (params.region.z * 2.0) + 0.25, h);
}

float layer_height(float3 relative) {
    return (length(relative) - EARTH - params.layer.x) / (params.layer.y - params.layer.x);
}

/// World-space direction of the camera ray through viewport UV.
float3 uv_ray(float2 uv) {
    float4 near_point = mul(globals.inverse_view_proj, float4(uv * 2.0 - 1.0, 1.0, 1.0));
    return normalize(near_point.xyz / near_point.w - globals.eye.xyz);
}

/// x: distance along the ray where the layer starts; y: where it ends.
float2 layer_span(float3 origin, float3 direction) {
    float bottom = EARTH + params.layer.x;
    float top = EARTH + params.layer.y;
    float r = length(origin);
    float2 inner = sphere(origin, direction, bottom);
    float2 outer = sphere(origin, direction, top);
    float2 span;
    if (r < bottom) {
        if (sphere(origin, direction, EARTH).x > 0.0)
            return -1.0;
        span = float2(inner.y, outer.y);
    } else if (r < top) {
        span = float2(0.0, inner.x > 0.0 ? inner.x : outer.y);
    } else {
        if (outer.y < 0.0)
            return -1.0;
        span = float2(outer.x, inner.x > 0.0 ? inner.x : outer.y);
    }
    span.y = min(span.y, span.x + MAX_MARCH);
    return span;
}

// ---- Voxel cloudscape ---------------------------------------------------

/// Set in a brick table entry for an empty brick.
static const uint EMPTY = 0x80000000u;
/// Atlas cells hold a brick with a voxel of its neighbours around it, so
/// hardware filtering crosses bricks seamlessly.
static const int CELL = 10;
/// The detail noise drops a mip every this many metres from the camera (Nubis³).
static const float NOISE_MIP_DISTANCE = 500.0;

bool has_scape() {
    return params.scape_min.w > 0.0;
}

float3 scape_extent() {
    return float3(params.scape_grid.xyz) * 8.0 * params.scape_min.w;
}

/// The brick table entry for brick `b`; past the grid, empty with nothing to skip.
uint brick_entry(int3 b) {
    if (any(b < 0) || any(b >= int3(params.scape_grid.xyz)))
        return EMPTY;
    return leaves.Load(int4(b, 0));
}

/// One voxel of the cloudscape's dimensional profile, from its brick's codes.
float decode_voxel(int3 v) {
    uint slot = brick_entry(v >> 3);
    if ((slot & EMPTY) != 0)
        return 0.0;
    uint codes = params.scape_info.z, bricks = params.scape_info.y;
    uint3 local = uint3(v & 7);
    uint code_index = slot * 64 + (local.x >> 1) + 4 * (local.y >> 1) + 16 * (local.z >> 1);
    uint code = (vq[codes * 8 + 3 * bricks + code_index / 4] >> (8 * (code_index % 4))) & 255u;
    uint entry = (local.x & 1) + 2 * (local.y & 1) + 4 * (local.z & 1);
    return asfloat(vq[code * 8 + entry]) * asfloat(vq[codes * 8 + bricks + slot]);
}

uint3 atlas_cell(uint slot) {
    uint across = params.scape_grid.w, deep = params.scape_info.x;
    return uint3(slot % across, slot / across % deep, slot / (across * deep));
}

/// VQVDB's decode: every atlas texel from its brick's codes, once per cloudscape.
[shader("compute")]
[numthreads(4, 4, 4)]
void decodeMain(uint3 id: SV_DispatchThreadID) {
    uint3 size;
    atlas_out.GetDimensions(size.x, size.y, size.z);
    if (any(id >= size))
        return;
    uint3 cell = id / CELL;
    uint slot = cell.x + params.scape_grid.w * (cell.y + params.scape_info.x * cell.z);
    if (slot >= params.scape_info.y)
        return;
    uint packed = vq[params.scape_info.z * 8 + slot];
    int3 brick = int3(packed & 1023u, (packed >> 10) & 1023u, packed >> 20);
    atlas_out[id] = decode_voxel(brick * 8 + int3(id % CELL) - 1);
}

/// The cloudscape's modeling data at a point (Nubis' NVDFs).
struct Nvdf {
    float profile;
    float type;
    float scale;
    /// Metres surely clear around the point; 0 in or beside a cloudy brick.
    float clear;
};

Nvdf sample_nvdf(float3 p) {
    Nvdf n;
    n.profile = 0.0;
    n.type = 0.0;
    n.scale = 0.0;
    n.clear = 0.0;
    float3 g = (p - params.scape_min.xyz) / params.scape_min.w;
    int3 brick = int3(floor(g / 8.0));
    uint slot = brick_entry(brick);
    if ((slot & EMPTY) != 0) {
        n.clear = float(slot & 255u) * 8.0 * params.scape_min.w;
        return n;
    }
    uint3 size;
    atlas.GetDimensions(size.x, size.y, size.z);
    float3 uvw = (float3(atlas_cell(slot) * CELL) + 1.0 + g - float3(brick * 8)) / float3(size);
    n.profile = atlas.SampleLevel(clamp_sampler, uvw, 0);
    uint attributes = vq[params.scape_info.z * 8 + 2 * params.scape_info.y + slot];
    n.type = float(attributes & 255u) / 255.0;
    n.scale = float((attributes >> 8) & 255u) / 255.0;
    return n;
}

/// Nubis³ up-rez: cloud density (0..1) at `p`, `distance` from the camera,
/// from its nonzero profile.
float scape_density(float3 p, Nvdf n, float distance) {
    float lod = log2(1.0 + distance / NOISE_MIP_DISTANCE);
    float4 noise = detail_noise.SampleLevel(wrap_sampler, (p - float3(params.shape.yz, 0.0)) * 0.01, lod);
    float composite = noise_composite(noise, n.profile, n.type);
    // 0 within 50 m of the camera, 1 past 150 m.
    float near = saturate((distance - 50.0) / 100.0);
    if (near < 1.0) {
        // Twice-folded noise: finer detail up close without another fetch.
        float folded = saturate(lerp(1.0 - pow(abs(abs(noise.g * 2.0 - 1.0) * 2.0 - 1.0), 4.0),
                                     pow(abs(abs(noise.a * 2.0 - 1.0) * 2.0 - 1.0), 2.0), n.type));
        composite = lerp(folded, composite, lerp(0.9, 1.0, near));
    }
    float scale = pow(saturate(n.scale), 4.0);
    float density = saturate(remap(n.profile, composite, 1.0, 0.0, 1.0)) * scale;
    // Sharper where thin, which brings out low density detail.
    density = pow(density, lerp(0.3, 0.6, max(1.0e-4, scale)));
    // Softer up close, which hides under-sampling there.
    return near < 1.0 ? pow(density, lerp(0.5, 1.0, near)) * lerp(0.666, 1.0, near) : density;
}

/// x: where a ray from `origin` enters the cloudscape's box; y: where it leaves.
float2 scape_span(float3 origin, float3 direction) {
    float3 low = params.scape_min.xyz;
    float3 inverse = 1.0 / direction;
    float3 a = (low - origin) * inverse;
    float3 b = (low + scape_extent() - origin) * inverse;
    float3 near = min(a, b);
    float3 far = max(a, b);
    return float2(max(max(near.x, near.y), max(near.z, 0.0)), min(min(far.x, far.y), min(far.z, MAX_MARCH)));
}

/// The cloudscape's light grid: density (profile times density scale) summed
/// in metres to the box's edge, r toward the sun and g straight up, as Nubis³
/// sums it. One slice in eight is redrawn per frame.
[shader("compute")]
[numthreads(8, 8, 1)]
void scapeLightMain(uint3 id: SV_DispatchThreadID) {
    uint3 size;
    scape_light_out.GetDimensions(size.x, size.y, size.z);
    uint3 voxel = params.frame.y != 0 ? uint3(id.xy, id.z * 8 + params.frame.x % 8) : id;
    if (!has_scape() || any(voxel >= size))
        return;
    float3 p = params.scape_min.xyz + (float3(voxel) + 0.5) / float3(size) * scape_extent();
    float3 to_sun = raised_sun();
    float2 summed = 0.0;
    float reach = scape_span(p, to_sun).y;
    const int STEPS = 16;
    for (int i = 0; i < STEPS; ++i) {
        // Quadratic spacing: fine near the voxel, coarse far from it.
        float inner = float(i * i) / (STEPS * STEPS);
        float outer = float((i + 1) * (i + 1)) / (STEPS * STEPS);
        Nvdf n = sample_nvdf(p + to_sun * (reach * (inner + outer) * 0.5));
        summed.r += n.profile * n.scale * reach * (outer - inner);
    }
    float rise = scape_span(p, float3(0.0, 0.0, 1.0)).y;
    const int UP = 6;
    for (int i = 0; i < UP; ++i) {
        Nvdf n = sample_nvdf(p + float3(0.0, 0.0, rise * (i + 0.5) / UP));
        summed.g += n.profile * n.scale * rise / UP;
    }
    scape_light_out[voxel] = float4(summed, 0.0, 0.0);
}

// ---- Lighting -----------------------------------------------------------

/// Light a sample sends toward the eye per unit scattering coefficient:
/// Wrenninge's octaves of sunlight through `optical` metres of unit density
/// (extinction a = 0.25, scattering b = 0.6, eccentricity c = 0.5; a <= b
/// keeps it energy conserving), whose scattered octaves build up with depth
/// into the cloud (`profile`) for Nubis' inner glow and dark edges; skylight
/// let through by `open` (Nubis' ambient: strongest at the edges); and
/// ground bounce, which reaches the bases.
float3 sample_radiance(float optical, float open, float profile, float h, float3 light, float phases[4],
                       float3 sky_light, float3 ground_light) {
    float extinct = params.layer.w;
    float glow = lerp(0.4, 1.0, saturate(profile * 2.0));
    float3 radiance = 0.0;
    for (int o = 0; o < 4; ++o)
        radiance += (o == 0 ? 1.0 : glow * pow(0.6, o)) * light * exp(-optical * extinct * pow(0.25, o)) * phases[o];
    return radiance + sky_light * open * sqrt(1.0 - profile) + ground_light * lerp(1.0, 0.2, h);
}

/// Add a step of `dt` metres at distance `t` with extinction `sigma_t`, by
/// Hillaire's analytic integral of in-scatter over the step.
void integrate(float sigma_t, float3 radiance, float t, float dt, inout float3 scattered,
               inout float transmittance, inout float2 depth_sum) {
    float3 in_scatter = radiance * sigma_t * 0.99;
    float step_transmittance = exp(-sigma_t * dt);
    scattered += transmittance * (in_scatter - in_scatter * step_transmittance) / max(sigma_t, 1.0e-7);
    depth_sum += float2(t, 1.0) * transmittance * (1.0 - step_transmittance);
    transmittance *= step_transmittance;
}

/// Sphere trace the cloudscape over `box` (Nubis³): steps as long as the brick
/// table says is clear, else growing with the square root of distance; the
/// jitter animates near the camera and holds still beyond, so distant clouds
/// do not shimmer.
void march_scape(float3 eye, float3 direction, float2 box, float still, float jitter, float3 light,
                 float phases[4], float3 sky_light, float3 ground_light, inout float3 scattered,
                 inout float transmittance, inout float2 depth_sum) {
    float extinct = params.layer.w;
    float3 to_sun = raised_sun();
    float stride = 4.0 * params.scape_min.w;
    // shortcut: a floor on the step caps a ray at ~384 samples; Nubis renders the near 200 m at quarter resolution instead, which would let this go.
    float least = (box.y - box.x) / 384.0;
    float t = box.x;
    for (int i = 0; i < 512 && t < box.y && transmittance > 0.01; ++i) {
        float adaptive = max(max(1.0, sqrt(t) * 0.08), least);
        Nvdf n = sample_nvdf(eye + direction * t);
        float step = max(n.clear, adaptive);
        if (n.clear == 0.0) {
            float s = t + (t < 250.0 ? jitter : still) * step;
            float3 p = eye + direction * s;
            Nvdf m = sample_nvdf(p);
            float density = m.profile > 0.0 ? scape_density(p, m, s) : 0.0;
            if (density > 0.0) {
                // Two detailed samples toward the sun, then the light grid beyond them.
                float optical = 0.0;
                for (int j = 0; j < 2; ++j) {
                    float3 q = p + to_sun * (stride * (j + 0.5));
                    Nvdf l = sample_nvdf(q);
                    optical += (l.profile > 0.0 ? scape_density(q, l, s) : 0.0) * stride;
                }
                float3 uvw = (p + to_sun * 2.0 * stride - params.scape_min.xyz) / scape_extent();
                float2 summed = all(uvw >= 0.0) && all(uvw <= 1.0) ? scape_light.SampleLevel(clamp_sampler, uvw, 0).rg : 0.0;
                optical += summed.r;
                float open = 1.0 / (1.0 + 0.11 * summed.g * extinct);
                float h = saturate((p.z - params.scape_min.z) / scape_extent().z);
                float3 radiance = sample_radiance(optical, open, m.profile, h, light, phases, sky_light, ground_light);
                integrate(density * extinct, radiance, s, step, scattered, transmittance, depth_sum);
            }
        }
        t += step;
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void traceMain(uint3 id: SV_DispatchThreadID) {
    uint2 size = params.frame.zw;
    if (any(id.xy >= size))
        return;
    float2 uv = (float2(id.xy) + 0.5) / float2(size);
    float3 direction = uv_ray(uv);
    float3 eye = globals.eye.xyz;
    // The planet's center sits below the camera; altitude is z.
    float3 origin = float3(0.0, 0.0, EARTH + max(eye.z, 1.0));
    float3 to_sun = globals.sun_direction.w > 0.0 ? -globals.sun_direction.xyz : float3(0.0, 0.0, 1.0);
    float3 sun = globals.sun_direction.w > 0.0 ? globals.sun_color.rgb : 0.0;
    float shade = 1.0;
    float2 ground_hit = sphere(origin, direction, EARTH);
    if (ground_hit.x > 0.0) {
        float2 xy = eye.xy + direction.xy * ground_hit.x;
        float2 uv = (xy - params.region.xy) / params.region.z;
        uint size, unused;
        shadow_out.GetDimensions(size, unused);
        shade = all(uv >= 0.0) && all(uv < 1.0) ? shadow_out[uint2(uv * size)]
                                                 : lerp(1.0, 0.3, params.layer.z * params.layer.z);
        shade = max(shade, 0.15);
    }
    float3 sky = atmosphere(origin, direction, to_sun, shade) * sun;

    float mid = EARTH + (params.layer.x + params.layer.y) * 0.5;
    float3 mid_point = float3(0.0, 0.0, mid);
    // Ambient sky radiance, averaged over the zenith and the horizon toward and
    // away from the sun; the ground below bounces sunlight up into the bases.
    float2 flat_sun = normalize(to_sun.xy + float2(1.0e-4, 0.0));
    float3 sky_light = (atmosphere(mid_point, float3(0.0, 0.0, 1.0), to_sun) * 2.0
                      + atmosphere(mid_point, normalize(float3(flat_sun, 0.15)), to_sun)
                      + atmosphere(mid_point, normalize(float3(-flat_sun, 0.15)), to_sun)) * 0.25 * sun
                    + globals.sky.rgb;
    float3 ground_light = sun * sun_transmittance(float3(0.0, 0.0, EARTH + 1.0), to_sun)
                        * saturate(to_sun.z) * (0.3 / PI);
    float mu = dot(direction, to_sun);
    float phase0 = droplet_phase(mu, 20.0);

    // Cirrus: a thin sheet of ice crystals far above, behind the main layer.
    float2 cirrus_hit = sphere(origin, direction, EARTH + CIRRUS_ALTITUDE);
    if (params.shape.w > 0.0 && cirrus_hit.y > 0.0 && origin.z < EARTH + CIRRUS_ALTITUDE) {
        float3 p = float3(eye.xy, 0.0) + direction * cirrus_hit.y;
        float2 c = (p.xy + params.shape.yz * 2.0) / 30000.0;
        // Streaks stretched along x, as jet-stream shear draws them out.
        float streak = shape_noise.SampleLevel(wrap_sampler, float3(c.x * 0.35, c.y * 1.6, 0.85), 0).g;
        float fine = detail_noise.SampleLevel(wrap_sampler, float3(c * 6.0, 0.4), 0).r;
        float amount = saturate(remap(streak * 0.75 + fine * 0.25, 1.0 - params.shape.w * 0.6, 1.0, 0.0, 1.0));
        float alpha = (1.0 - exp(-amount * 2.0)) * exp(-cirrus_hit.y * 1.0e-5);
        float3 light = sun * sun_transmittance(origin + direction * cirrus_hit.y, to_sun);
        float3 lit = light * lerp(hg(mu, 0.6), 1.0 / (4.0 * PI), 0.4) * 2.0 + sky_light;
        sky = lerp(sky, lit, alpha);
    }

    float3 scattered = 0.0;
    float transmittance = 1.0;
    float depth = 1.0e5;
    float2 depth_sum = 0.0;
    float extinct = params.layer.w;
    float thickness = params.layer.y - params.layer.x;
    float3 light = sun * sun_transmittance(mid_point, to_sun);
    float phases[4];
    for (int o = 0; o < 4; ++o)
        phases[o] = lerp(1.0 / (4.0 * PI), phase0, pow(0.5, o));
    // Interleaved gradient noise; `jitter` rotates by the golden ratio every frame.
    float still = frac(52.9829189 * frac(dot(float2(id.xy), float2(0.06711056, 0.00583715))));
    float jitter = frac(still + float(params.frame.x % 64u) * 0.618034);
    // The cloudscape is marched where the layer's steps reach it.
    float2 box = has_scape() ? scape_span(eye, direction) : float2(1.0, 0.0);
    bool box_done = box.y <= box.x;
    float2 span = layer_span(origin, direction);
    if (span.y > span.x && span.x < MAX_MARCH) {
        float horizon = 1.0 - abs(direction.z);
        int steps = int(lerp(64.0, 128.0, horizon));
        // Steps lengthen in proportion to distance (plus the layer's
        // thickness, so they stop shrinking up close): far clouds need less.
        float growth = pow((span.y + thickness) / (span.x + thickness), 1.0 / steps);
        float light_step = thickness / 64.0;
        for (int i = 0; i < steps && transmittance > 0.01; ++i) {
            float start = (span.x + thickness) * pow(growth, float(i));
            float dt = start * (growth - 1.0);
            float t = start - thickness + dt * jitter;
            if (!box_done && t >= box.x) {
                march_scape(eye, direction, box, still, jitter, light, phases, sky_light, ground_light,
                            scattered, transmittance, depth_sum);
                box_done = true;
            }
            if (t >= box.x && t <= box.y)
                continue;
            float3 relative = origin + direction * t;
            float h = layer_height(relative);
            float3 world = float3(eye.xy + direction.xy * t, params.layer.x + h * thickness);
            float type;
            float profile = cloud_profile(world, h, type);
            if (profile <= 0.0)
                continue;
            float lod = log2(1.0 + t / 7000.0);
            float density = cloud_detail(world, h, profile, type, lod);
            if (density <= 0.0)
                continue;
            // Optical depth toward the sun: three detailed samples keep the
            // surface crisp, and the light grid holds the density beyond them.
            float optical = 0.0, travelled = 0.0;
            for (int j = 0; j < 3; ++j) {
                float stride = light_step * float(1 << j) * 0.5;
                travelled += stride;
                float reach = travelled - 0.5 * stride;
                optical += cloud_density(world + to_sun * reach, layer_height(relative + to_sun * reach), lod) * stride;
            }
            float3 coord = light_coord(world.xy + to_sun.xy * travelled, layer_height(relative + to_sun * travelled));
            float2 border = min(coord.xy, 1.0 - coord.xy);
            // Past the grid's edge, three coarse samples stand in for it.
            float inside = saturate(min(border.x, border.y) * 10.0);
            float beyond = 0.0;
            for (int j = 3; j < 6 && inside < 1.0; ++j) {
                float stride = light_step * float(1 << j) * 0.5;
                travelled += stride;
                float reach = travelled - 0.5 * stride;
                float unused;
                beyond += cloud_profile(world + to_sun * reach, layer_height(relative + to_sun * reach), unused) * stride;
            }
            float2 summed = inside > 0.0 ? light_grid.SampleLevel(clamp_sampler, coord, 0).rg : 0.0;
            optical += lerp(beyond, summed.r, inside);
            // Skylight under open sky; the cloud above passes what a
            // scattering slab of g = 0.85 would.
            float open = lerp(lerp(0.4, 1.0, h), 1.0 / (1.0 + 0.11 * summed.g * extinct), inside);
            float3 radiance = sample_radiance(optical, open, profile, h, light, phases, sky_light, ground_light);
            integrate(density * extinct, radiance, t, dt, scattered, transmittance, depth_sum);
        }
    }
    if (!box_done && transmittance > 0.01)
        march_scape(eye, direction, box, still, jitter, light, phases, sky_light, ground_light,
                    scattered, transmittance, depth_sum);
    if (depth_sum.y > 1.0e-4)
        depth = depth_sum.x / depth_sum.y;
    // Far clouds fade into the haze.
    float haze = exp(-depth * 1.5e-5);
    scattered *= haze;
    transmittance = lerp(1.0, transmittance, haze);

    float4 result = float4(sky * transmittance + scattered, transmittance);
    if (params.frame.y != 0) {
        float4 previous = mul(globals.previous_view_proj, float4(eye + direction * depth, 1.0));
        float2 previous_uv = previous.xy / previous.w * 0.5 + 0.5;
        if (previous.w > 0.0 && all(previous_uv >= 0.0) && all(previous_uv <= 1.0)) {
            float4 old = history.SampleLevel(clamp_sampler, previous_uv, 0);
            if (all(isfinite(old)))
                result = lerp(old, result, 0.1);
        }
    }
    current[id.xy] = result;
}

/// Transmittance of sunlight through the layer, indexed by where the sun ray
/// meets z = 0. Multiple scattering keeps cloud shadows from going black.
[shader("compute")]
[numthreads(8, 8, 1)]
void shadowMain(uint3 id: SV_DispatchThreadID) {
    uint size, unused;
    shadow_out.GetDimensions(size, unused);
    if (any(id.xy >= size))
        return;
    float3 to_sun = raised_sun();
    float2 xy = params.region.xy + (float2(id.xy) + 0.5) / float(size) * params.region.z;
    float start = params.layer.x / to_sun.z;
    float span = (params.layer.y - params.layer.x) / to_sun.z;
    const int STEPS = 24;
    float optical = 0.0, unused_type;
    for (int i = 0; i < STEPS; ++i) {
        float t = start + span * (i + 0.5) / STEPS;
        float3 p = float3(xy, 0.0) + to_sun * t;
        optical += cloud_profile(p, (p.z - params.layer.x) / (params.layer.y - params.layer.x), unused_type);
    }
    optical *= span / STEPS * params.layer.w;
    if (has_scape()) {
        float3 ground = float3(xy, 0.0);
        float2 box = scape_span(ground, to_sun);
        const int SCAPE_STEPS = 24;
        float summed = 0.0;
        for (int i = 0; i < SCAPE_STEPS && box.y > box.x; ++i) {
            Nvdf n = sample_nvdf(ground + to_sun * lerp(box.x, box.y, (i + 0.5) / SCAPE_STEPS));
            summed += n.profile * n.scale;
        }
        optical += summed * max(box.y - box.x, 0.0) / SCAPE_STEPS * params.layer.w;
    }
    shadow_out[id.xy] = 0.7 * exp(-optical) + 0.3 * exp(-optical * 0.1);
}

/// Nubis³ voxel lighting: the density summed from each voxel to the top of
/// the layer, r toward the sun and g straight up, so the trace needs only a
/// few light samples of its own. One slice in eight is redrawn per frame.
[shader("compute")]
[numthreads(8, 8, 1)]
void lightMain(uint3 id: SV_DispatchThreadID) {
    uint size, unused;
    light_out.GetDimensions(size, unused, unused);
    // Every slice when the history restarts.
    uint3 voxel = params.frame.y != 0 ? uint3(id.xy, id.z * 8 + params.frame.x % 8) : id;
    if (any(voxel >= size))
        return;
    float3 uvw = (float3(voxel) + 0.5) / float(size);
    float thickness = params.layer.y - params.layer.x;
    float3 p = float3(params.region.xy + (uvw.xy - 0.25) * params.region.z * 2.0,
                      params.layer.x + uvw.z * thickness);
    float3 to_sun = raised_sun();
    float rise = (1.0 - uvw.z) * thickness;
    float reach = min(rise / to_sun.z, LIGHT_REACH);
    float2 summed = 0.0;
    float unused_type;
    const int STEPS = 12;
    for (int i = 0; i < STEPS; ++i) {
        // Quadratic spacing: fine near the voxel, coarse far from it.
        float inner = float(i * i) / (STEPS * STEPS);
        float outer = float((i + 1) * (i + 1)) / (STEPS * STEPS);
        float3 s = p + to_sun * (reach * (inner + outer) * 0.5);
        summed.r += cloud_profile(s, (s.z - params.layer.x) / thickness, unused_type) * reach * (outer - inner);
    }
    const int UP = 4;
    for (int i = 0; i < UP; ++i) {
        float h = uvw.z + (1.0 - uvw.z) * (i + 0.5) / UP;
        summed.g += cloud_profile(float3(p.xy, params.layer.x + h * thickness), h, unused_type) * rise / UP;
    }
    light_out[voxel] = float4(summed, 0.0, 0.0);
}

/// Fill the sky pixels the resolve left empty.
[shader("compute")]
[numthreads(8, 8, 1)]
void applyMain(uint3 id: SV_DispatchThreadID) {
    float4 viewport = globals.viewport;
    if (any(float2(id.xy) >= viewport.zw))
        return;
    uint2 pixel = id.xy + uint2(viewport.xy);
    if (hdr_output[pixel].a != 0.0)
        return;
    float2 uv = (float2(id.xy) + 0.5) / viewport.zw;
    float4 sky = history.SampleLevel(clamp_sampler, uv, 0);
    float3 color = sky.rgb;
    if (globals.sun_direction.w > 0.0) {
        float3 direction = uv_ray(uv);
        float3 to_sun = -globals.sun_direction.xyz;
        float3 origin = float3(0.0, 0.0, EARTH + max(globals.eye.z, 1.0));
        float disk = smoothstep(cos(SUN_RADIUS * 1.2), cos(SUN_RADIUS), dot(direction, to_sun));
        if (disk > 0.0)
            color += disk * sky.a * globals.sun_color.rgb * sun_transmittance(origin, to_sun)
                   / (PI * SUN_RADIUS * SUN_RADIUS);
    }
    hdr_output[pixel] = float4(color, 1.0);
}
