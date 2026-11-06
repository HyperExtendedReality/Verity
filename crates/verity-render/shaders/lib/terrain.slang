// Procedural ground, continuous in world space so it survives any excavation:
// soil, grass, layered rock, sand and snow from slope, depth and height, with
// bump detail. Each noise band fades out below the pixel footprint, which keeps
// distant ground from shimmering.

struct TerrainSurface {
    float3 albedo;
    float roughness;
    float3 normal;
};

float terrain_hash(float3 p) {
    float3 q = frac(p * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return frac((q.x + q.y) * q.z);
}

/// Value noise in [0, 1]: xyz gradient, w value.
float4 terrain_noise(float3 p) {
    float3 i = floor(p);
    float3 f = p - i;
    float3 u = f * f * (3.0 - 2.0 * f);
    float3 du = 6.0 * f * (1.0 - f);
    float a = terrain_hash(i);
    float b = terrain_hash(i + float3(1, 0, 0));
    float c = terrain_hash(i + float3(0, 1, 0));
    float d = terrain_hash(i + float3(1, 1, 0));
    float e = terrain_hash(i + float3(0, 0, 1));
    float g = terrain_hash(i + float3(1, 0, 1));
    float h = terrain_hash(i + float3(0, 1, 1));
    float k = terrain_hash(i + float3(1, 1, 1));
    float xy = a - b - c + d;
    float yz = a - c - e + h;
    float zx = a - b - e + g;
    float xyz = -a + b + c - d + e - g - h + k;
    float value = a + (b - a) * u.x + (c - a) * u.y + (e - a) * u.z
        + xy * u.x * u.y + yz * u.y * u.z + zx * u.z * u.x + xyz * u.x * u.y * u.z;
    float3 gradient = du * float3(
        b - a + xy * u.y + zx * u.z + xyz * u.y * u.z,
        c - a + yz * u.z + xy * u.x + xyz * u.z * u.x,
        e - a + zx * u.x + yz * u.y + xyz * u.x * u.y);
    return float4(gradient, value);
}

/// How much of a noise band survives a footprint: fully until it nears Nyquist.
float terrain_band(float frequency, float footprint) {
    return 1.0 - smoothstep(0.25, 1.0, frequency * footprint);
}

/// The surface at `p` with unit `normal`. `surface_height` is the ground's height
/// before any digging, `wet` its wetness, `footprint` a pixel's width in metres.
TerrainSurface terrain_surface(float3 p, float3 normal, float surface_height, float wet, float footprint,
                               float sea_level, float detail) {
    static const float FREQUENCY[3] = { 1.7, 6.0, 18.0 };
    static const float WEIGHT[3] = { 0.55, 0.28, 0.12 };
    float grain = 0.0;
    float3 slope = 0.0;
    for (uint band = 0; band < 3; ++band) {
        float keep = terrain_band(FREQUENCY[band], footprint);
        // A faded band leaves its mean, so distant ground keeps its brightness.
        grain += WEIGHT[band] * 0.5 * (1.0 - keep);
        if (keep > 0.0) {
            float4 n = terrain_noise(p * FREQUENCY[band]);
            grain += WEIGHT[band] * keep * n.w;
            slope += n.xyz * (WEIGHT[band] * keep * FREQUENCY[band]);
        }
    }
    float broad = terrain_noise(p * 0.055).w;
    float layers = lerp(0.5, terrain_noise(float3(p.x * 0.08, p.y * 0.08, p.z * 0.8 + broad * 2.0)).w,
                        terrain_band(0.8, footprint));
    float strata = smoothstep(0.2, 0.8, layers);

    float depth = max(surface_height - p.z, 0.0);
    wet = saturate(wet);
    float3 rock = lerp(float3(0.13, 0.12, 0.105), float3(0.36, 0.32, 0.26), strata) * (0.72 + grain * 0.50);
    float3 dirt = lerp(float3(0.19, 0.11, 0.055), float3(0.36, 0.27, 0.16), broad) * (0.78 + grain * 0.34);
    float3 grass = lerp(float3(0.075, 0.105, 0.025), float3(0.22, 0.27, 0.07), broad) * (0.70 + grain * 0.50);
    float cliff = 1.0 - smoothstep(0.55, 0.86, normal.z);
    float buried = smoothstep(0.5, 2.4, depth);
    float bedrock = smoothstep(2.0, 5.0, depth);
    float vegetation = smoothstep(0.3, 0.65, broad + wet * 0.18) * (1.0 - buried);
    float3 albedo = lerp(dirt, grass, vegetation);
    albedo = lerp(albedo, rock, max(cliff, bedrock));
    // Beaches and sea floor: sand within a couple of metres of the waterline and below it.
    float shore = 1.0 - smoothstep(0.6, 2.4, p.z - sea_level + broad * 1.2);
    float3 sand = lerp(float3(0.50, 0.44, 0.31), float3(0.66, 0.59, 0.43), broad) * (0.86 + grain * 0.20);
    albedo = lerp(albedo, sand, shore * (1.0 - cliff) * (1.0 - bedrock));
    float snow = smoothstep(68.0, 91.0, p.z + broad * 7.0) * smoothstep(0.55, 0.9, normal.z) * (1.0 - buried);
    albedo = lerp(albedo, float3(0.72, 0.78, 0.80) * (0.90 + grain * 0.10), snow);

    TerrainSurface surface;
    surface.albedo = albedo * (1.0 - wet * 0.13);
    surface.roughness = lerp(0.95, 0.73, wet) * (1.0 - snow * 0.1);
    // The height gradient, projected onto the surface, tilts the normal.
    float3 tilt = slope - normal * dot(slope, normal);
    surface.normal = normalize(normal - tilt * detail);
    return surface;
}

/// The surface's mean color, for GI rays that hit it: what a surface looks like
/// from far enough away for its fine detail to average out. Texture coordinates
/// hold the ground height and wetness; a tracer without them can pass `p.z` and
/// 0, losing dug walls and wet ground. Terrain materials hold `sea_level` in
/// `params.w`.
float3 terrain_albedo(float3 p, float3 normal, float surface_height, float wet, float sea_level) {
    return terrain_surface(p, normal, surface_height, wet, 1.0e4, sea_level, 0.0).albedo;
}
