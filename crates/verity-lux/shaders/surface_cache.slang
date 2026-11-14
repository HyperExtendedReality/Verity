// Surface cache (src/cascades/surface.rs): outgoing light on islands of nearly
// flat triangles, packed per instance into an atlas of ~30 cm texels. Unlike
// world probes, a texel belongs to one side of one surface, so its light never
// leaks through a wall. `surfaceUpdate` shades a rolling share of texels each
// frame; ray hits read them through `surfaceRead`.
// Its buffers are pointers that cascades.slang aliases to these names:
// surfaceTriangles, per triangle: chart corners as half pairs, island + 1.
// surfaceIslands: atlas rect x, y, w, h; w = 0 when unplaced.
// surfaceTexels: triangle, instance, barycentrics b, c (unorm16), valid.
// surfaceRadiance, three words per texel: direct light per source and sample
// mass; incoming bounce light per source and diffuse reflectance; dependency
// bounds as half floats and flags. Light is R11G11B10 floats.
static const uint SURFACE_ATLAS_WIDTH = 2048u; // surface::ATLAS_WIDTH
static const float SURFACE_MIN_BLEND = 0.1;
// A settled texel reshades its direct light on one visit in this many.
static const uint SURFACE_DIRECT_EVERY = 4u;

struct SurfaceTexel {
    TransportRadiance direct;   // outgoing direct light, diffuse, without self-emission
    TransportRadiance bounce;   // mean incoming light from other surfaces
    float3 reflectance;         // diffuse reflectance the bounce light leaves through
    float mass;
    DependencyContext dependency;
};

// Half-float bits in the low 16 bits of a uint. Slang's f16tof32/f32tof16
// compile to 16-bit types, which need shaderInt16 and shaderFloat16; the
// GLSL.std.450 pack/unpack stay in 32-bit registers and need neither.
float halfToFloat(uint bits) {
    uint low = bits & 0xffffu;
    float2 pair = spirv_asm { OpExtInst $$float2 result glsl450 UnpackHalf2x16 $low };
    return pair.x;
}
uint floatToHalf(float value) {
    float2 pair = float2(value, 0.0);
    uint packed = spirv_asm { OpExtInst $$uint result glsl450 PackHalf2x16 $pair };
    return packed & 0xffffu;
}

uint packSurfaceLight(float3 value) {
    // 60000 keeps rounding below the half-float infinity exponent.
    value = all(isfinite(value)) ? clamp(value, 0.0, 60000.0) : 0.0;
    uint3 h = uint3(floatToHalf(value.x), floatToHalf(value.y), floatToHalf(value.z));
    // Round to nearest, then drop the mantissa bits R11G11B10 has no room for.
    return ((h.x + 8u) >> 4u) | (((h.y + 8u) >> 4u) << 11u) | (((h.z + 16u) >> 5u) << 22u);
}

float3 unpackSurfaceLight(uint value) {
    return float3(halfToFloat((value & 0x7ffu) << 4u), halfToFloat(((value >> 11u) & 0x7ffu) << 4u),
                  halfToFloat((value >> 22u) << 5u));
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
    result.dependency.minimum = float3(halfToFloat(c.x), halfToFloat(c.x >> 16u), halfToFloat(c.y));
    result.dependency.maximum = float3(halfToFloat(c.y >> 16u), halfToFloat(c.z), halfToFloat(c.z >> 16u));
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
        floatToHalf(low.x) | (floatToHalf(low.y) << 16u), floatToHalf(low.z) | (floatToHalf(high.x) << 16u),
        floatToHalf(high.y) | (floatToHalf(high.z) << 16u), flags);
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
    float2 uv = float2(halfToFloat(chart.x), halfToFloat(chart.x >> 16u)) * hit.barycentric.x
        + float2(halfToFloat(chart.y), halfToFloat(chart.y >> 16u)) * hit.barycentric.y
        + float2(halfToFloat(chart.z), halfToFloat(chart.z >> 16u)) * hit.barycentric.z;
    int2 q = int2(floor(uv * asfloat(instanceWord.z)));
    uint2 local = uint2(clamp(q, int2(0, 0), int2(rect.zw) - 1));
    uint texel = (rect.y + local.y) * SURFACE_ATLAS_WIDTH + rect.x + local.x;
    return texel < params.surface.x ? texel : EMPTY;
}

// Hashed cells: surfaces the atlas does not chart (double-sided ones such as
// foliage, and retained meshes such as terrain) keep their light in a hash
// grid instead. A cell's texel follows the atlas's in surfaceRadiance, at
// params.surface.x + slot, and is shaded by surfaceUpdate like any texel.
// The key is the position at a cell size that doubles with distance, and the
// dominant axis and sign of the normal facing the ray, so the two sides of a
// leaf never share light. surfaceHashKeys holds two words a slot: the key
// (0 = empty) and the frame it was last read. surfaceHashRecords holds the hit
// that claimed it: triangle, instance (in TLAS order), barycentrics b, c
// (unorm16), and 1 for the triangle's front face or 2 for its back.
static const float SURFACE_HASH_CELL = 0.3;            // metres up to SURFACE_HASH_LOD_DISTANCE
static const float SURFACE_HASH_LOD_DISTANCE = 10.0;
static const uint SURFACE_HASH_PROBES = 8u;

float surfaceHashLod(float3 position) {
    float distance = length(position - params.cameraPosition.xyz);
    return exp2(floor(log2(1.0 + distance / SURFACE_HASH_LOD_DISTANCE)));
}

// The slot holding the cell at `position` facing `normal`, claiming an empty
// one when `claim` is set (linear probing). EMPTY when the cache is off or
// the probe sequence is full.
// shortcut: dropped cells are not tombstoned, so a key placed past one becomes a miss and is claimed again; add tombstones if duplicates crowd the table.
uint surfaceHashProbe(uint hash, uint key, uint capacity, bool claim, out bool claimed) {
    claimed = false;
    uint slot = hash % capacity;
    for (uint probe = 0u; probe < SURFACE_HASH_PROBES; ++probe) {
        uint stored = surfaceHashKeys[slot * 2u].load();
        if (stored == 0u && claim) {
            stored = surfaceHashKeys[slot * 2u].compareExchange(0u, key);
            if (stored == 0u) {
                claimed = true;
                return slot;
            }
        }
        if (stored == key) return slot;
        slot = (slot + 1u) % capacity;
    }
    return EMPTY;
}

// `shared` is set when this lane took another lane's lookup (wave_batch.slang),
// which has already touched the cell.
uint surfaceHashFind(float3 position, float3 normal, bool claim, out bool claimed, out bool shared) {
    claimed = false;
    shared = false;
    uint capacity = params.surfaceHash.x;
    if (capacity == 0u || !all(isfinite(position)) || !all(isfinite(normal))) return EMPTY;
    float lod = surfaceHashLod(position);
    int3 cell = int3(floor(position / (SURFACE_HASH_CELL * lod)));
    float3 magnitude = abs(normal);
    uint axis = magnitude.x >= magnitude.y && magnitude.x >= magnitude.z ? 0u
        : (magnitude.y >= magnitude.z ? 1u : 2u);
    uint face = axis * 2u + (normal[axis] < 0.0 ? 1u : 0u);
    uint hash = transportHash(uint(lod) + transportHash(uint(cell.z) + transportHash(uint(cell.y)
        + transportHash(uint(cell.x) + transportHash(face)))));
    uint key = transportHash(hash ^ 0x9e3779b9u) | 1u;
    uint tag = claim ? 1u : 0u;
    bool follows = waveBatchFollows(key, tag);
    uint slot = follows ? EMPTY : surfaceHashProbe(hash, key, capacity, claim, claimed);
    uint first = waveBatchTake(slot, key, tag, follows, shared);
    if (shared) return first;
    return follows ? surfaceHashProbe(hash, key, capacity, claim, claimed) : slot;
}

// The cell under a hit seen along `direction`, claimed if it has none yet; the
// claiming lane records the hit so surfaceUpdate can shade the cell.
uint surfaceHashAt(Hit hit, float3 position, float3 direction, bool claim) {
    float3 front = hitGeometricNormal(hit);
    bool back = dot(front, direction) > 0.0;
    bool claimed, shared;
    uint slot = surfaceHashFind(position, back ? -front : front, claim, claimed, shared);
    if (slot == EMPTY) return EMPTY;
    if (claimed) {
        uint2 barycentric = uint2(round(saturate(hit.barycentric.yz) * 65535.0));
        surfaceHashRecords[slot] = uint4(hit.triangle, hit.instance,
            barycentric.x | (barycentric.y << 16u), back ? 2u : 1u);
    }
    if (!shared) surfaceHashKeys[slot * 2u + 1u].max(params.frame.x + 1u);
    return params.surface.x + slot;
}

// The cached light under a hit at `position` seen along `direction`, once its
// atlas texel or hashed cell has been shaded. `claim` starts a cell for an
// uncharted surface, which reads as uncached until surfaceUpdate shades it.
bool surfaceRead(Hit hit, float3 position, float3 direction, bool claim, out SurfaceTexel texel) {
    texel = (SurfaceTexel)0;
    if (hit.triangle == EMPTY || hit.instance == EMPTY) return false;
    uint at = surfaceTexelAt(hit);
    if (at == EMPTY) at = surfaceHashAt(hit, position, direction, claim);
    if (at == EMPTY) return false;
    texel = loadSurfaceTexel(at);
    return texel.mass > 0.0;
}

// The hashed cell on the far side of a thin surface hit along `direction`,
// for light passing through it. False for atlas texels, which chart one side.
bool surfaceReadFarSide(Hit hit, float3 position, float3 direction, out SurfaceTexel texel) {
    texel = (SurfaceTexel)0;
    if (hit.triangle == EMPTY || hit.instance == EMPTY || surfaceTexelAt(hit) != EMPTY) return false;
    uint at = surfaceHashAt(hit, position, -direction, true);
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

// Whether a texel's bounce rays can leave the scene: they then bring the sky,
// and neither the texel's direct light nor a hit that reads it traces the sky.
bool surfaceBounceSeesSky() {
    if ((params.surface.w & 0xffu) == 0u) return false;
#ifdef GI_RAY_QUERY
    return true;
#else
    return (params.surface.w >> 8u) == 0u;
#endif
}

// The hashed cell `slot`'s surface point, as surfaceRecordPoint gives an atlas
// texel's, facing the side that claimed it. False for an empty slot, and for
// one unread for params.surfaceHash.z frames, which this drops.
bool surfaceHashPoint(uint slot, out Hit hit, out float3 position, out float3 normal) {
    hit = (Hit)0;
    position = 0.0;
    normal = 0.0;
    if (surfaceHashKeys[slot * 2u].load() == 0u) return false;
    uint stamp = params.frame.x + 1u;
    if (stamp - surfaceHashKeys[slot * 2u + 1u].load() > params.surfaceHash.z) {
        // Cleared before the key is released, so a lane that claims the slot
        // next never shades this cell's history as its own.
        storeSurfaceTexel(params.surface.x + slot, (SurfaceTexel)0);
        surfaceHashKeys[slot * 2u].store(0u);
        return false;
    }
    uint4 record = surfaceHashRecords[slot];
    if (record.w == 0u) return false;
    hit.triangle = record.x;
    hit.instance = record.y;
    hit.errors = 0u;
    hit.footprint = 0.0;
    hit.distance = 1.0;
    float2 yz = float2(record.z & 0xffffu, record.z >> 16u) / 65535.0;
    hit.barycentric = float3(saturate(1.0 - yz.x - yz.y), yz);
    uint base = instanceBase(hit.instance);
    Triangle triangle = localTriangle(hit.triangle);
    position = transformPoint(base + 3u, triangle.a.xyz * hit.barycentric.x
        + triangle.b.xyz * hit.barycentric.y + triangle.c.xyz * hit.barycentric.z);
    normal = triangleGeometricNormal(base, sceneWord(base + 6u).w, triangle)
        * (record.w == 2u ? -1.0 : 1.0);
    return true;
}

// One rolling share of the atlas and of the hashed cells a frame: threads
// below params.surface.y take atlas texels, the next params.surfaceHash.y
// take cells. Each shades its direct light as a diffuse surface seen head on,
// traces cosine rays whose hits read other texels (so bounces build up across
// frames), and blends both into its history.
[shader("compute")]
[numthreads(64, 1, 1)]
void surfaceUpdate(uint3 thread : SV_DispatchThreadID) {
    uint texels = params.surface.x;
    uint budget = params.surface.y;
    uint at;
    Hit hit;
    float3 position, normal;
    if (thread.x < budget) {
        // Consecutive texels share islands, so their rays stay coherent.
        at = (params.frame.x * budget + thread.x) % texels;
        uint4 record = surfaceTexels[at];
        if (record.w == 0u) return;
        surfaceRecordPoint(record, hit, position, normal);
    } else {
        uint lane = thread.x - budget;
        uint cells = params.surfaceHash.x;
        if (lane >= params.surfaceHash.y || cells == 0u) return;
        uint slot = (params.frame.x * params.surfaceHash.y + lane) % cells;
        if (!surfaceHashPoint(slot, hit, position, normal)) return;
        at = texels + slot;
    }
    if (!all(isfinite(position)) || !all(isfinite(normal))) return;

    SurfaceTexel texel = loadSurfaceTexel(at);
    // Drop history its rays or lights no longer support, as the world cache does.
    bool reset = params.sourceHistory.y != 0u
        || dependencyIntersectsDirty(texel.dependency)
        || dependencyIntersectsEmissiveDirty(texel.dependency)
        || (params.lightDirtyMin.w > 0.5
            && surfaceBoxIntersects(texel.dependency, params.lightDirtyMin, params.lightDirtyMax));
    float mass = reset || !isfinite(texel.mass) ? 0.0 : texel.mass;

    // Direct light, as the RTGI cache shades its cells: a ray one metre long
    // arriving along the normal, so the view is head on.
    transportAdmissionBypass = true;
    rtgiDirectOnly = true;
    surfaceShading = true;
    rtgiDiffuseReflectance = 0.0;
    DependencyContext dependency = dependencyEmpty();
    TransportRadiance direct = texel.direct;
    float3 reflectance = texel.reflectance;
    // Lights and geometry that change reset the history above, so a settled
    // texel reshades its direct light on one visit in SURFACE_DIRECT_EVERY,
    // keeping its bounce rays every visit. Picked per thread group, so a warp
    // shades or skips together.
    bool reuseDirect = mass >= float(SURFACE_DIRECT_EVERY)
        && transportHash((thread.x / 64u) ^ transportHash(params.frame.x ^ 0xd1ec7u))
            % SURFACE_DIRECT_EVERY != 0u;
    if (!reuseDirect) {
        bool supported;
        direct = outgoingTransportRadianceTracked(position + normal, -normal, hit,
            supported, dependency);
        if (!supported) return;
        reflectance = rtgiDiffuseReflectance;
    }

    // Bounce light; escaped rays bring the sky. A hit on an uncharted surface
    // claims a hashed cell, so foliage and terrain bounce light too.
    // Hits on surfaces not shaded yet leave the mean, rather than count as
    // black: shading them instead cost 0.3 ms on Sponza and Bistro for no net gain.
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
        if (bounce.triangle != EMPTY
            && !surfaceRead(bounce, origin + direction * bounce.distance, direction, true, other))
            continue;
        dependencyUnion(dependency, rayDependency);
        counted += 1.0;
        if (bounce.triangle == EMPTY) {
            incoming.background += environmentRadiance(direction);
            continue;
        }
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
