// Physically based sky and volumetric clouds, drawn behind the scene.
//
// - Shape: Schneider's Nubis model (SIGGRAPH 2015/2017): tileable Perlin-Worley
//   base noise eroded by Worley detail, a height profile per cloud genus and a
//   coverage remap.
// - Lighting: Hillaire's energy-conserving integration (SIGGRAPH 2016) with
//   Wrenninge's multiple-scattering octaves, and the Jendersie & d'Eon (2023)
//   HG + Draine fit to Mie scattering by water droplets as the phase function.
// - Cost: traced at half resolution with a per-frame interleaved-gradient
//   jitter, then reprojected through the cloud's depth and accumulated.
// - Sky: single-scattering Rayleigh + Mie atmosphere over a spherical planet.

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

struct Params {
    /// Layer bottom, top (metres above ground), coverage, extinction per metre.
    float4 layer;
    /// x: genus; yz: wind and seed offset; w: cirrus amount.
    float4 shape;
    /// x: frame number; y: 1 when the history is valid; zw: half-res size.
    uint4 frame;
    /// Cloud shadow map: world xy of its corner, its size; w unused.
    float4 region;
};
[[vk::push_constant]] ConstantBuffer<Params> params;

[[vk::binding(0, 1)]] [[vk::image_format("rgba8")]] RWTexture3D<float4> shape_out;
[[vk::binding(1, 1)]] [[vk::image_format("rgba8")]] RWTexture3D<float4> detail_out;
[[vk::binding(2, 1)]] Texture3D<float4> shape_noise;
[[vk::binding(3, 1)]] Texture3D<float4> detail_noise;
[[vk::binding(4, 1)]] SamplerState wrap_sampler;
[[vk::binding(5, 1)]] SamplerState clamp_sampler;
[[vk::binding(6, 1)]] Texture2D<float4> history;
[[vk::binding(7, 1)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> current;
[[vk::binding(8, 1)]] [[vk::image_format("r32f")]] RWTexture2D<float> shadow_out;

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

[shader("compute")]
[numthreads(4, 4, 4)]
void detailMain(uint3 id: SV_DispatchThreadID) {
    uint size;
    uint unused;
    detail_out.GetDimensions(size, unused, unused);
    float3 p = (float3(id) + 0.5) / float(size);
    detail_out[id] = float4(worley_fbm(p, 2), worley_fbm(p, 4), worley_fbm(p, 8), 1.0);
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

/// Cloud density (0..1) at world position `p`, `h` up through the layer.
float cloud_density(float3 p, float h, bool detail) {
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
    // Wind shears tops downwind, as convective clouds lean.
    float3 q = p + wind + float3(params.shape.yz, 0.0) * h * 0.2;
    float4 n = shape_noise.SampleLevel(wrap_sampler, q * SHAPE_SCALE, 0);
    float fbm = n.g * 0.625 + n.b * 0.25 + n.a * 0.125;
    float shape = saturate(remap(n.r, fbm - 1.0, 1.0, 0.0, 1.0));
    shape = saturate(remap(shape, 0.6, 0.9, 0.0, 1.0));
    float cloudiness = (shape * 0.6 + weather_n * 0.4) * height_profile(h, genus);
    float base = saturate(remap(cloudiness, threshold, 1.0, 0.0, 1.0));
    if (base <= 0.0 || !detail)
        return base;
    float3 d = detail_noise.SampleLevel(wrap_sampler, q * DETAIL_SCALE + float3(0.0, 0.0, h), 0).rgb;
    float high = d.r * 0.625 + d.g * 0.25 + d.b * 0.125;
    // Wispy bases, billowy tops.
    high = lerp(high, 1.0 - high, saturate(h * 6.0));
    return saturate(remap(base, high * 0.4, 1.0, 0.0, 1.0));
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
    float2 span = layer_span(origin, direction);
    if (span.y > span.x && span.x < MAX_MARCH) {
        float extinct = params.layer.w;
        float3 light = sun * sun_transmittance(mid_point, to_sun);
        float phases[4];
        for (int o = 0; o < 4; ++o)
            phases[o] = lerp(1.0 / (4.0 * PI), phase0, pow(0.5, o));
        float horizon = 1.0 - abs(direction.z);
        int steps = int(lerp(64.0, 128.0, horizon));
        float dt = (span.y - span.x) / steps;
        // Interleaved gradient noise, rotated by the golden ratio every frame.
        float jitter = frac(52.9829189 * frac(dot(float2(id.xy), float2(0.06711056, 0.00583715)))
                            + float(params.frame.x % 64u) * 0.618034);
        float light_step = (params.layer.y - params.layer.x) / 64.0;
        float weighted_depth = 0.0, weight = 0.0;
        for (int i = 0; i < steps && transmittance > 0.01; ++i) {
            float t = span.x + dt * (i + jitter);
            float3 relative = origin + direction * t;
            float h = layer_height(relative);
            float3 world = float3(eye.xy, 0.0) + direction * t;
            float density = cloud_density(world, h, false);
            if (density <= 0.0)
                continue;
            density = cloud_density(world, h, true);
            if (density <= 0.0)
                continue;
            // Optical depth toward the sun over six growing steps.
            float optical = 0.0, travelled = 0.0;
            for (int j = 0; j < 6; ++j) {
                float stride = light_step * float(1 << j) * 0.5;
                travelled += stride;
                float3 lp = relative + to_sun * (travelled - 0.5 * stride);
                float3 lw = world + to_sun * (travelled - 0.5 * stride);
                optical += cloud_density(lw, layer_height(lp), j < 3) * stride;
            }
            float sigma_t = density * extinct;
            float sigma_s = sigma_t * 0.99;
            // Wrenninge octaves: extinction a = 0.25, scattering b = 0.6,
            // eccentricity c = 0.5 (a <= b keeps it energy conserving).
            float3 radiance = 0.0;
            for (int o = 0; o < 4; ++o)
                radiance += pow(0.6, o) * light * exp(-optical * extinct * pow(0.25, o)) * phases[o];
            // Skylight reaches the tops, ground bounce the bases.
            radiance += sky_light * lerp(0.4, 1.0, h) + ground_light * lerp(1.0, 0.2, h);
            radiance *= sigma_s;
            // Hillaire's analytic integral of in-scatter over the step.
            float step_transmittance = exp(-sigma_t * dt);
            scattered += transmittance * (radiance - radiance * step_transmittance) / max(sigma_t, 1.0e-7);
            weighted_depth += t * transmittance * (1.0 - step_transmittance);
            weight += transmittance * (1.0 - step_transmittance);
            transmittance *= step_transmittance;
        }
        if (weight > 1.0e-4)
            depth = weighted_depth / weight;
        // Far clouds fade into the haze.
        float haze = exp(-max(depth, span.x) * 1.5e-5);
        scattered *= haze;
        transmittance = lerp(1.0, transmittance, haze);
    }

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
    float3 to_sun = -globals.sun_direction.xyz;
    to_sun.z = max(to_sun.z, 0.08);
    to_sun = normalize(to_sun);
    float2 xy = params.region.xy + (float2(id.xy) + 0.5) / float(size) * params.region.z;
    float start = params.layer.x / to_sun.z;
    float span = (params.layer.y - params.layer.x) / to_sun.z;
    const int STEPS = 24;
    float optical = 0.0;
    for (int i = 0; i < STEPS; ++i) {
        float t = start + span * (i + 0.5) / STEPS;
        float3 p = float3(xy, 0.0) + to_sun * t;
        optical += cloud_density(p, (p.z - params.layer.x) / (params.layer.y - params.layer.x), false);
    }
    optical *= span / STEPS * params.layer.w;
    shadow_out[id.xy] = 0.7 * exp(-optical) + 0.3 * exp(-optical * 0.1);
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
