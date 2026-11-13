// Surface cache (src/cascades/surface.rs): outgoing light on islands of nearly
// flat triangles, packed per instance into an atlas of ~30 cm texels. Unlike
// world probes, a texel belongs to one side of one surface, so its light never
// leaks through a wall. `surfaceUpdate` shades a rolling share of texels each
// frame; ray hits read them through `surfaceRead`.
[[vk::binding(51, 0)]] StructuredBuffer<uint4> surfaceTriangles; // per triangle: chart corners as half pairs, island + 1
[[vk::binding(52, 0)]] StructuredBuffer<uint4> surfaceIslands;   // atlas rect x, y, w, h; w = 0 when unplaced
[[vk::binding(53, 0)]] StructuredBuffer<uint4> surfaceTexels;    // triangle, instance, barycentrics b, c (unorm16), valid
// Three words per texel: direct light per source and sample mass; incoming
// bounce light per source and diffuse reflectance; dependency bounds as half
// floats and flags. Light is R11G11B10 floats.
[[vk::binding(54, 0)]] RWStructuredBuffer<uint4> surfaceRadiance;
static const uint SURFACE_ATLAS_WIDTH = 2048u; // surface::ATLAS_WIDTH
static const float SURFACE_MIN_BLEND = 0.1;

struct SurfaceTexel {
    TransportRadiance direct;   // outgoing direct light, diffuse, without self-emission
    TransportRadiance bounce;   // mean incoming light from other surfaces
    float3 reflectance;         // diffuse reflectance the bounce light leaves through
    float mass;
    DependencyContext dependency;
};

uint packSurfaceLight(float3 value) {
    // 60000 keeps rounding below the half-float infinity exponent.
    value = all(isfinite(value)) ? clamp(value, 0.0, 60000.0) : 0.0;
    uint3 h = uint3(f32tof16(value.x), f32tof16(value.y), f32tof16(value.z));
    // Round to nearest, then drop the mantissa bits R11G11B10 has no room for.
    return ((h.x + 8u) >> 4u) | (((h.y + 8u) >> 4u) << 11u) | (((h.z + 16u) >> 5u) << 22u);
}

float3 unpackSurfaceLight(uint value) {
    return float3(f16tof32((value & 0x7ffu) << 4u), f16tof32(((value >> 11u) & 0x7ffu) << 4u),
                  f16tof32((value >> 22u) << 5u));
}

SurfaceTexel loadSurfaceTexel(uint texel) {
    uint4 a = surfaceRadiance[texel * 3u];
    uint4 b = surfaceRadiance[texel * 3u + 1u];
    uint4 c = surfaceRadiance[texel * 3u + 2u];
    SurfaceTexel result;
    result.direct.background = unpackSurfaceLight(a.x);
    result.direct.emissive = unpackSurfaceLight(a.y);
    result.direct.punctual = unpackSurfaceLight(a.z);
    result.mass = asfloat(a.w);
    result.bounce.background = unpackSurfaceLight(b.x);
    result.bounce.emissive = unpackSurfaceLight(b.y);
    result.bounce.punctual = unpackSurfaceLight(b.z);
    result.reflectance = unpackSurfaceLight(b.w);
    result.dependency.minimum = float3(f16tof32(c.x), f16tof32(c.x >> 16u), f16tof32(c.y));
    result.dependency.maximum = float3(f16tof32(c.y >> 16u), f16tof32(c.z), f16tof32(c.z >> 16u));
    result.dependency.known = (c.w & 1u) != 0u;
    result.dependency.hasBounds = (c.w & 2u) != 0u;
    return result;
}

void storeSurfaceTexel(uint texel, SurfaceTexel value) {
    DependencyContext d = value.dependency;
    // Half floats round to nearest; widen first so the bounds stay conservative.
    float3 low = d.minimum - (abs(d.minimum) * 2e-3 + 1e-2);
    float3 high = d.maximum + (abs(d.maximum) * 2e-3 + 1e-2);
    bool fits = all(isfinite(low)) && all(isfinite(high))
        && all(abs(low) < 60000.0) && all(abs(high) < 60000.0);
    bool known = d.known && (!d.hasBounds || fits);
    uint flags = (known ? 1u : 0u) | (d.hasBounds && fits ? 2u : 0u);
    if (!fits) { low = 0.0; high = 0.0; }
    surfaceRadiance[texel * 3u] = uint4(packSurfaceLight(value.direct.background),
        packSurfaceLight(value.direct.emissive), packSurfaceLight(value.direct.punctual),
        asuint(value.mass));
    surfaceRadiance[texel * 3u + 1u] = uint4(packSurfaceLight(value.bounce.background),
        packSurfaceLight(value.bounce.emissive), packSurfaceLight(value.bounce.punctual),
        packSurfaceLight(value.reflectance));
    surfaceRadiance[texel * 3u + 2u] = uint4(
        f32tof16(low.x) | (f32tof16(low.y) << 16u), f32tof16(low.z) | (f32tof16(high.x) << 16u),
        f32tof16(high.y) | (f32tof16(high.z) << 16u), flags);
}

// The atlas texel under a hit, or EMPTY when its instance or island has none.
uint surfaceTexelAt(Hit hit) {
    if (params.surface.x == 0u || hit.triangle == EMPTY || hit.instance == EMPTY) return EMPTY;
    uint4 instanceWord = sceneWord(instanceBase(hit.instance) + 7u);
    if (instanceWord.y == 0u) return EMPTY;
    uint4 chart = surfaceTriangles[hit.triangle];
    if (chart.w == 0u) return EMPTY;
    uint4 rect = surfaceIslands[instanceWord.y - 1u + chart.w - 1u];
    if (rect.z == 0u) return EMPTY;
    float2 uv = float2(f16tof32(chart.x), f16tof32(chart.x >> 16u)) * hit.barycentric.x
        + float2(f16tof32(chart.y), f16tof32(chart.y >> 16u)) * hit.barycentric.y
        + float2(f16tof32(chart.z), f16tof32(chart.z >> 16u)) * hit.barycentric.z;
    int2 q = int2(floor(uv * asfloat(instanceWord.z)));
    uint2 local = uint2(clamp(q, int2(0, 0), int2(rect.zw) - 1));
    uint texel = (rect.y + local.y) * SURFACE_ATLAS_WIDTH + rect.x + local.x;
    return texel < params.surface.x ? texel : EMPTY;
}

// The cached light under a hit, once its texel has been shaded.
bool surfaceRead(Hit hit, out SurfaceTexel texel) {
    texel = (SurfaceTexel)0;
    uint at = surfaceTexelAt(hit);
    if (at == EMPTY) return false;
    texel = loadSurfaceTexel(at);
    return texel.mass > 0.0;
}

TransportRadiance lerpTransport(TransportRadiance from, TransportRadiance to, float t) {
    TransportRadiance result;
    result.background = lerp(from.background, to.background, t);
    result.emissive = lerp(from.emissive, to.emissive, t);
    result.punctual = lerp(from.punctual, to.punctual, t);
    return result;
}

// The surface point a texel record names, as a hit one metre along its normal.
void surfaceRecordPoint(uint4 record, out Hit hit, out float3 position, out float3 normal) {
    hit.triangle = record.x;
    // Records name instances in load order; the TLAS sort moves them, and the
    // words after the layer mask say where each one went.
    hit.instance = sceneWord(sceneWord(0).x + 1u + record.y / 4u)[record.y % 4u];
    hit.errors = 0u;
    hit.footprint = 0.0;
    hit.distance = 1.0;
    float2 yz = float2(record.z & 0xffffu, record.z >> 16u) / 65535.0;
    hit.barycentric = float3(saturate(1.0 - yz.x - yz.y), yz);
    uint base = instanceBase(hit.instance);
    Triangle triangle = localTriangle(hit.triangle);
    position = transformPoint(base + 3u, triangle.a.xyz * hit.barycentric.x
        + triangle.b.xyz * hit.barycentric.y + triangle.c.xyz * hit.barycentric.z);
    normal = triangleGeometricNormal(base, sceneWord(base + 6u).w, triangle);
}

bool surfaceBoxIntersects(DependencyContext dependency, float4 low, float4 high) {
    if (!dependency.known) return true;
    if (!dependency.hasBounds) return false;
    return all(dependency.minimum <= high.xyz) && all(dependency.maximum >= low.xyz);
}

// One rolling share of the atlas a frame. Each texel shades its direct light
// as a diffuse surface seen head on, traces cosine rays whose hits read other
// texels (so bounces build up across frames), and blends both into its history.
[shader("compute")]
[numthreads(64, 1, 1)]
void surfaceUpdate(uint3 thread : SV_DispatchThreadID) {
    uint texels = params.surface.x;
    uint budget = params.surface.y;
    if (texels == 0u || thread.x >= budget) return;
    // Consecutive texels share islands, so their rays stay coherent.
    uint at = (params.frame.x * budget + thread.x) % texels;
    uint4 record = surfaceTexels[at];
    if (record.w == 0u) return;
    Hit hit;
    float3 position, normal;
    surfaceRecordPoint(record, hit, position, normal);
    if (!all(isfinite(position)) || !all(isfinite(normal))) return;

    // Direct light, as the RTGI cache shades its cells: a ray one metre long
    // arriving along the normal, so the view is head on.
    transportAdmissionBypass = true;
    rtgiDirectOnly = true;
    surfaceShading = true;
    rtgiDiffuseReflectance = 0.0;
    bool supported;
    DependencyContext dependency = dependencyEmpty();
    TransportRadiance direct = outgoingTransportRadianceTracked(position + normal, -normal, hit,
        supported, dependency);
    if (!supported) return;
    float3 reflectance = rtgiDiffuseReflectance;

    // Bounce light. Escaped rays bring nothing: the sky is direct light above.
    // Hits on uncached surfaces leave the mean, rather than count as black:
    // shading them instead cost 0.3 ms on Sponza and Bistro for no net gain.
    TransportRadiance incoming = (TransportRadiance)0;
    float counted = 0.0;
    float3 origin = rayOffset(position, normal, params.tuning.z);
    // One direction sample per block of 32 texels: neighbours share an island
    // and nearly a normal, so a warp traces nearly parallel rays, which keeps
    // software traversal coherent. Each frame draws new ones.
    uint state = transportHash(at ^ transportHash(params.frame.x ^ 0x57face5u));
#ifdef GI_RAY_QUERY
    float reach = 1e20;
#else
    // A few bounce rays crossing the whole scene set the pass's time in
    // software; past the reach a ray leaves the mean like an uncached hit.
    float reach = (params.surface.w >> 8u) == 0u ? 1e20 : float(params.surface.w >> 8u) * 0.01;
#endif
    [loop] for (uint ray = 0u; ray < min(params.surface.w & 0xffu, 8u); ++ray) {
        float2 u = float2(rtgiRandom(state), rtgiRandom(state));
        float3 direction = rtgiCosineDirection(normal, u);
        DependencyContext rayDependency;
        float start, distance, padding;
        Hit bounce = traceRayTracked(origin, direction, reach, rayDependency, start, distance,
            padding);
        if ((bounce.errors & HIT_QUERY_UNSUPPORTED) != 0u) continue;
        if (bounce.triangle == EMPTY && reach < 1e20) continue;
        SurfaceTexel other;
        if (bounce.triangle != EMPTY && !surfaceRead(bounce, other)) continue;
        dependencyUnion(dependency, rayDependency);
        counted += 1.0;
        if (bounce.triangle == EMPTY) continue;
        dependencyUnion(dependency, other.dependency);
        incoming.background += other.direct.background + other.bounce.background * other.reflectance;
        incoming.emissive += other.direct.emissive + other.bounce.emissive * other.reflectance;
        incoming.punctual += other.direct.punctual + other.bounce.punctual * other.reflectance;
    }
    if (counted > 0.0) {
        incoming.background /= counted;
        incoming.emissive /= counted;
        incoming.punctual /= counted;
    }

    SurfaceTexel texel = loadSurfaceTexel(at);
    // Drop history its rays or lights no longer support, as the world cache does.
    bool reset = params.sourceHistory.y != 0u
        || dependencyIntersectsDirty(texel.dependency)
        || dependencyIntersectsEmissiveDirty(texel.dependency)
        || (params.lightDirtyMin.w > 0.5
            && surfaceBoxIntersects(texel.dependency, params.lightDirtyMin, params.lightDirtyMax));
    float mass = reset || !isfinite(texel.mass) ? 0.0 : texel.mass;
    float blend = max(1.0 / (mass + 1.0), SURFACE_MIN_BLEND);
    if (mass > 0.0) dependencyUnion(dependency, texel.dependency);
    texel.direct = lerpTransport(texel.direct, direct, blend);
    // Without a single cached hit the bounce estimate keeps its history.
    if (counted > 0.0) texel.bounce = lerpTransport(texel.bounce, incoming, blend);
    else if (mass == 0.0) texel.bounce = (TransportRadiance)0;
    texel.reflectance = reflectance;
    texel.mass = min(mass + 1.0, 1.0 / SURFACE_MIN_BLEND);
    texel.dependency = dependency;
    storeSurfaceTexel(at, texel);
}
