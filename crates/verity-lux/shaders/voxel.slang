// Screen and voxel GI: a ray follows the depth buffer and takes its light from
// last frame's image, where that shows what it hit. Past what the screen shows
// it walks a voxel copy of the scene. Each voxel holds the light leaving it,
// from the sun and point lights and from one bounce ray a visit, so light bounces
// further with every frame.

#define MATERIAL_SET 2
#include "materials.slang"
#include "brdf.slang"
#include "screen_gi.slang"

static const int MAX_VOXEL_STEPS = 256;

// Per cell of the grid: 0 when empty, else its voxel's index + 1.
[[vk::binding(4, 0)]] StructuredBuffer<uint> cells;
// Per voxel: cell, material, normal as signed bytes, unused.
[[vk::binding(5, 0)]] StructuredBuffer<uint4> voxels;
// Per voxel: the light leaving it, then the mean bounce light arriving (w frames).
[[vk::binding(6, 0)]] RWStructuredBuffer<float4> radiance;

float3 gridOrigin() { return f.reflection.xyz; }
float cellSize() { return f.reflection.w; }
int gridSize() { return int(f.counts.y); }

float3 voxelNormal(uint packed) {
    int3 bytes = int3(int(packed << 24u), int(packed << 16u), int(packed << 8u)) >> 24;
    return normalize(float3(bytes));
}

// The first voxel along a ray within reach, as its index + 1, or 0.
uint walk(float3 o, float3 d, float reach) {
    float3 p = (o - gridOrigin()) / cellSize();
    float3 inv = safeInv(d);
    float3 near = -p * inv;
    float3 far = (float(gridSize()) - p) * inv;
    float3 lo = min(near, far), hi = max(near, far);
    float t = max(max(lo.x, max(lo.y, lo.z)), 0.0);
    float end = min(min(hi.x, min(hi.y, hi.z)), reach / cellSize());
    if (t >= end) return 0u;
    int size = gridSize();
    int3 c = clamp(int3(floor(p + d * (t + 1e-3))), 0, size - 1);
    int3 step = select(d >= 0.0, int3(1), int3(-1));
    float3 next = (float3(c) + select(d >= 0.0, float3(1.0), float3(0.0)) - p) * inv;
    float3 delta = abs(inv);
    [loop] for (int i = 0; i < MAX_VOXEL_STEPS; i++) {
        uint found = cells[(c.z * size + c.y) * size + c.x];
        if (found != 0u) return found;
        if (next.x <= next.y && next.x <= next.z) {
            t = next.x; next.x += delta.x; c.x += step.x;
        } else if (next.y <= next.z) {
            t = next.y; next.y += delta.y; c.y += step.y;
        } else {
            t = next.z; next.z += delta.z; c.z += step.z;
        }
        if (t > end || any(c < 0) || any(c >= size)) return 0u;
    }
    return 0u;
}

// The light a ray along d gets from a voxel. Back faces are dark, so a wall one
// voxel thick does not pass light through it.
float3 voxelSeen(uint found, float3 d) {
    uint index = found - 1u;
    if (dot(voxelNormal(voxels[index].z), d) > 0.0) return 0.0;
    return radiance[index * 2u].rgb;
}

// Irradiance / pi from the sun and point lights, shadowed by the voxels.
float3 voxelDirect(float3 x, float3 n, float3 o) {
    float3 light = 0.0;
    float lit = dot(n, f.sunDirection.xyz);
    if (lit > 0.0 && f.sunDirection.w > 0.5 && walk(o, f.sunDirection.xyz, 1e6) == 0u)
        light += f.sunColor.rgb * lit;
    [loop] for (uint i = 0u; i < f.counts.z; i++) {
        Light source = lights[i];
        float3 to = source.position.xyz - x;
        float d2 = max(dot(to, to), 1e-2);
        float3 l = to * rsqrt(d2);
        float nl = dot(n, l);
        float w = saturate(1.0 - pow(d2 * source.position.w, 2.0));
        if (nl <= 0.0 || w <= 0.0 || walk(o, l, sqrt(d2) - cellSize()) != 0u) continue;
        light += source.color.rgb * nl * w * w / d2;
    }
    return light;
}

// A quarter of the voxels a frame: their direct light and one bounce ray.
[shader("compute")]
[numthreads(64, 1, 1)]
void inject(uint3 id : SV_DispatchThreadID) {
    uint frame = f.cache.x;
    uint index = id.x * 4u + (frame & 3u);
    if (index >= f.counts.x) return;
    uint4 voxel = voxels[index];
    uint size = uint(gridSize());
    uint3 cell = uint3(voxel.x % size, (voxel.x / size) % size, voxel.x / (size * size));
    float3 x = gridOrigin() + (float3(cell) + 0.5) * cellSize();
    float3 n = voxelNormal(voxel.z);
    float3 o = x + n * (1.5 * cellSize());

    uint seed = hashU(index ^ hashU(frame));
    float u = float(seed >> 8u) / 16777216.0;
    float phi = 6.2831853 * float(hashU(seed) >> 8u) / 16777216.0;
    float3 dir = cosineDir(n, u, phi);
    uint found = walk(o, dir, f.sky.w);
    float3 seen = found != 0u ? voxelSeen(found, dir) : background(dir);

    float4 arriving = radiance[index * 2u + 1u];
    float frames = min(arriving.w + 1.0, 64.0);
    arriving = float4(lerp(arriving.rgb, seen, 1.0 / frames), frames);
    radiance[index * 2u + 1u] = bad(arriving) ? float4(0.0) : arriving;
    float3 leaving = meanAlbedo(materials[voxel.y]) * (voxelDirect(x, n, o) + arriving.rgb);
    radiance[index * 2u] = float4(leaving, 1.0);
}

// Rays from a quarter-size texel: writes the light they found / rays and the
// fraction that hit something.
[shader("compute")]
[numthreads(8, 8, 1)]
void trace(uint3 id : SV_DispatchThreadID) {
    int2 px = int2(id.xy);
    if (any(px >= traceSize())) return;
    int2 fp = framePixel(px, traceSize());
    float depth = loadDepth(fp);
    uint4 g = loadGbuffer(fp);
    if (!receives(depth, g)) {
        out0.Store(px, 0.0);
        return;
    }
    float3 p = worldFromPixel(float2(fp) + 0.5, depth);
    float3 n = gbufferNormal(g);
    float2 pf = float2(px);
    float3 noise = float3(ign(pf), ign(pf + float2(47.0, 17.0)), ign(pf + float2(13.0, 101.0)));
    int rays = clamp(int(f.cache.w), 1, 4);
    float3 light = 0.0;
    float hits = 0.0;
    float3 origin = p + n * (0.02 + 0.002 * linearDepth(depth));
    float3 voxelOrigin = p + n * (1.5 * cellSize());
    [loop] for (int r = 0; r < rays; r++) {
        float index = f.misc.x * float(rays) + float(r);
        float u = frac(noise.x + index * 0.7548776662);
        float phi = 6.2831853 * frac(noise.y + index * 0.5698402910);
        float3 dir = cosineDir(n, u, phi);
        float jitter = frac(noise.z + index * 0.6180339887);
        float4 seen = march(origin, dir, f.misc.y, STEPS, jitter);
        if (seen.w > 0.5) {
            light += clampFirefly(seen.rgb);
            hits += 1.0;
        } else {
            uint found = walk(voxelOrigin, dir, f.sky.w);
            if (found != 0u) {
                light += clampFirefly(voxelSeen(found, dir));
                hits += 1.0;
            }
        }
    }
    float4 result = float4(light, hits) / float(rays);
    out0.Store(px, bad(result) ? float4(0.0) : result);
}

// Adds the filtered light to the image, and keeps the result for next frame's rays.
[shader("compute")]
[numthreads(8, 8, 1)]
void composite(uint3 id : SV_DispatchThreadID) {
    int2 fp = int2(id.xy);
    if (any(fp >= int2(f.sizes.xy))) return;
    int2 at = target(fp).xy;
    float4 lit = hdr[at];
    float depth = loadDepth(fp);
    uint4 g = loadGbuffer(fp);
    if (!receives(depth, g)) {
        colorOut.Store(at, lit);
        return;
    }
    float3 p = worldFromPixel(float2(fp) + 0.5, depth);
    float3 n = gbufferNormal(g);
    float rough = gbufferRoughness(g);
    float4 gi = upsampleGi(fp, n, linearDepth(depth));
    float3 view = normalize(f.camera.xyz - p);
    float open = 1.0 - saturate(gi.a);
    float3 diffuse = gi.rgb * f.misc.z + open * background(n);
    float3 specular = open * background(reflect(-view, n));
    float nv = max(dot(n, view), 1e-4);
    float2 dfg = env_brdf(nv, rough);
    Brdf brdf = make_brdf(gbufferBase(g), gbufferMetallic(g), rough, nv, dfg);
    float4 result = float4(lit.rgb + diffuse * brdf.diffuse_color
        + specular * specular_albedo(brdf.f0, dfg), lit.a);
    hdr[at] = result;
    colorOut.Store(at, result);
}
