// Screen and voxel GI: a ray follows the depth buffer and takes its light from
// last frame's image, where that shows what it hit. Past what the screen shows
// it walks a voxel copy of the scene. Each voxel holds the light leaving it,
// from the sun and point lights and from one bounce ray a visit, so light bounces
// further with every frame.
//
// The voxels are a clipmap around the camera: levels of the same number of
// cells, each twice the size of the last. A level is addressed toroidally, so
// following the camera rebuilds only the slabs of cells that scrolled in. Those
// are voxelized here from the scene's triangles: `clear` frees what they held,
// `claim` gives each cell a triangle touches a voxel, `accumulate` sums the
// surface and normals in it and `finalize` keeps the cells covered enough.

#define MATERIAL_SET 2
#include "materials.slang"
#include "brdf.slang"
#include "terrain.slang"
#include "screen_gi.slang"

static const int MAX_VOXEL_STEPS = 512;
static const int MAX_START_CELLS = 3;
// Must match `MAX_LEVELS` in voxel.rs.
static const uint MAX_LEVELS = 8u;
static const uint FREE = 0xffffffffu;
static const uint CLAIMING = 0x80000000u;
// Sample rows across a cell when voxelizing; cover is in 1/COVER_ONE of a cell face.
static const int SAMPLES = 4;
static const float COVER_ONE = 4096.0;

// Per level, per cell (toroidal): 0 when empty, else its voxel's index + 1.
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> cells;
// Per voxel: cell (as `cellIndex`) or FREE, material, normal as signed bytes,
// terrain ground as `pack_ground` in pxtgi.rs packs it.
[[vk::binding(5, 0)]] RWStructuredBuffer<uint4> voxels;
// Per voxel, for the side its normal faces and then the other: the light leaving
// that side, the mean bounce light arriving (w frames), and the mean of that
// light's luminance times the way it came from (`voxelArriving` tilts by it).
[[vk::binding(6, 0)]] RWStructuredBuffer<float4> radiance;
// Each level's low cell, then each job's cells to voxelize: low corner and
// level, high corner (exclusive) and cell count.
[[vk::binding(7, 0)]] StructuredBuffer<int4> clip;
// Three corners a triangle: xyz, then w the material, the ground and nothing.
[[vk::binding(8, 0)]] StructuredBuffer<float4> triangles;
// Per voxel being built: cover and the normal sum.
[[vk::binding(9, 0)]] RWStructuredBuffer<int> build;
// The count of free voxels, then their indices.
[[vk::binding(15, 0)]] RWStructuredBuffer<int> freeVoxels;
// Big triangles' work: an indirect dispatch's group count, 1, 1, unused, then
// per entry the triangle, the job, its first column and nothing.
[[vk::binding(16, 0)]] RWStructuredBuffer<uint> large;
// Per run of 64 triangles, its bounds: low corner, then high.
[[vk::binding(17, 0)]] StructuredBuffer<float4> chunkBounds;

int gridSize() { return int(f.counts.y); }
int levels() { return int(f.reflection.z); }
uint jobs() { return f.cache.y; }
// The finest level's cell.
float cellSize() { return f.reflection.w; }
float levelCell(int level) { return f.reflection.w * float(1u << uint(level)); }
int3 levelLow(int level) { return clip[level].xyz; }

// A cell's place in `cells`. The grid size is a power of two.
uint cellIndex(int level, int3 c) {
    int g = gridSize();
    int3 t = c & (g - 1);
    return uint(level) * uint(g * g * g) + uint((t.z * g + t.y) * g + t.x);
}

// The finest level holding p.
int levelAt(float3 p) {
    for (int l = 0; l < levels() - 1; l++) {
        float3 c = p / levelCell(l) - float3(levelLow(l));
        if (all(c >= 0.0) && all(c < float(gridSize()))) return l;
    }
    return levels() - 1;
}

float3 voxelNormal(uint packed) {
    int3 bytes = int3(int(packed << 24u), int(packed << 16u), int(packed << 8u)) >> 24;
    return normalize(float3(bytes));
}

uint packNormal(float3 n) {
    int3 b = int3(round(n * 127.0)) & 255;
    return uint(b.x) | uint(b.y) << 8u | uint(b.z) << 16u;
}

// The first voxel along a ray within reach, as its index + 1, or 0. It starts in
// the finest level holding o and steps to coarser ones as it leaves each.
uint walk(float3 o, float3 d, float reach) {
    float3 inv = safeInv(d);
    float3 delta = abs(inv);
    int3 step = select(d >= 0.0, int3(1), int3(-1));
    int size = gridSize();
    float travelled = 0.0;
    // Rays start beside a surface, which at this size can be inside the next
    // voxel over. Solid cells before the first empty one don't stop them.
    bool clear = false;
    int i = 0;
    [loop] for (int l = levelAt(o); l < levels(); l++) {
        float cell = levelCell(l);
        int3 low = levelLow(l);
        float3 p = o / cell - float3(low);
        float3 near = -p * inv;
        float3 far = (float(size) - p) * inv;
        float3 lo = min(near, far), hi = max(near, far);
        float t = max(max(lo.x, max(lo.y, lo.z)), travelled / cell);
        float end = min(min(hi.x, min(hi.y, hi.z)), reach / cell);
        if (t >= end) continue;
        int3 c = clamp(int3(floor(p + d * (t + 1e-3))), 0, size - 1);
        float3 next = (float3(c) + select(d >= 0.0, float3(1.0), float3(0.0)) - p) * inv;
        [loop] for (; i < MAX_VOXEL_STEPS; i++) {
            uint found = cells[cellIndex(l, c + low)];
            if (found != 0u && (clear || i >= MAX_START_CELLS)) return found;
            clear = clear || found == 0u;
            if (next.x <= next.y && next.x <= next.z) {
                t = next.x; next.x += delta.x; c.x += step.x;
            } else if (next.y <= next.z) {
                t = next.y; next.y += delta.y; c.y += step.y;
            } else {
                t = next.z; next.z += delta.z; c.z += step.z;
            }
            if (t > end || any(c < 0) || any(c >= size)) break;
        }
        if (t > reach / cell || i >= MAX_VOXEL_STEPS) return 0u;
        travelled = end * cell;
    }
    return 0u;
}

// The light a ray along d gets from a voxel, from the side it arrives at. Sides
// light separately, so a wall one voxel thick does not pass light through it.
float3 voxelSeen(uint found, float3 d) {
    uint index = found - 1u;
    uint side = dot(voxelNormal(voxels[index].z), d) > 0.0 ? 1u : 0u;
    return radiance[index * 6u + side * 3u].rgb;
}

// The start of the renderer's `Globals` (lib/common.slang), through its sun
// shadow cascades, and its shadow map: set 2 is the renderer's frame set.
static const uint SHADOW_CASCADES = 4u;
struct RenderGlobals {
    float4x4 viewProj;
    float4x4 view;
    float4x4 inverseViewProj;
    float4 eye;
    float4 viewport;
    float4 projection;
    uint4 counts;
    float4 sunDirection;
    float4 sunColor;
    float4x4 previousViewProj;
    float4x4 shadowViewProj[SHADOW_CASCADES];
    float4 shadowSplits;
    float4 shadowTexels;
    float4 sky; // w: sun shadows on
};
[[vk::binding(0, MATERIAL_SET)]] ConstantBuffer<RenderGlobals> renderGlobals;
[[vk::binding(16, MATERIAL_SET)]] Texture2DArray<float> sunShadowMap;
[[vk::binding(17, MATERIAL_SET)]] SamplerComparisonState sunShadowSampler;

// How much sun reaches a voxel's face: the renderer's shadow map where one of
// its cascades holds the face, else a walk from o through the voxels. A walk
// through coarse cells fattens the sun's shadow edges: in Sponza's courtyard
// it blocked 8.8% of what the shadow map lit, mostly at roof edges 16 m off,
// where the shadow map blocks 0.5%.
float sunVisible(float3 x, float3 n, float3 o, float cell) {
    if (renderGlobals.sky.w != 0.0) {
        // The face, not the centre, which can sit under the surface it holds.
        float3 face = x + n * (0.5 * cell);
        [loop] for (uint c = 0u; c < SHADOW_CASCADES; c++) {
            float3 biased = face + n * renderGlobals.shadowTexels[c] * 1.5;
            float4 clip = mul(renderGlobals.shadowViewProj[c], float4(biased, 1.0));
            float3 ndc = clip.xyz / clip.w;
            if (all(abs(ndc.xy) < 0.98) && ndc.z > 0.0 && ndc.z < 1.0)
                return sunShadowMap.SampleCmpLevelZero(sunShadowSampler, float3(ndc.xy * 0.5 + 0.5, float(c)), ndc.z);
        }
    }
    return walk(o, f.sunDirection.xyz, 1e6) == 0u ? 1.0 : 0.0;
}

// Irradiance / pi from the sun and, while they are few, the point lights,
// shadowed by the voxels. Many point lights come from `voxelSampled`.
float3 voxelDirect(float3 x, float3 n, float3 o, float cell) {
    float3 light = 0.0;
    float lit = dot(n, f.sunDirection.xyz);
    if (lit > 0.0 && f.sunDirection.w > 0.5)
        light += f.sunColor.rgb * lit * sunVisible(x, n, o, cell);
    if (sampledLights()) return light;
    [loop] for (uint i = 0u; i < f.counts.z; i++) {
        float3 l;
        float reach;
        float3 c = pointLight(i, x, n, l, reach);
        if (any(c > 0.0) && walk(o, l, reach - cellSize()) == 0u) light += c;
    }
    return light;
}

// A voxel's diffuse colour. Terrain's varies across its mesh, so it is worked
// out where the voxel is.
float3 voxelAlbedo(uint4 voxel, float3 x, float3 n) {
    Material m = materials[voxel.y];
    if ((m.textures.w & TERRAIN_SURFACE) != 0) {
        float height = float(voxel.w & 0xffffu) / 16.0 - 2048.0;
        float wet = float(voxel.w >> 16u) / 65535.0;
        return terrain_albedo(x, n, height, wet, m.params.w);
    }
    return meanAlbedo(m);
}

// The point lights `voxelDirect` leaves out: one picked and shadowed, for the
// running mean of arriving light. 0 while they are few.
float3 voxelSampled(float3 x, float3 n, float3 o, inout uint seed) {
    if (!sampledLights()) return 0.0;
    float3 l;
    float reach;
    float3 c = pickPointLight(x, n, seed, l, reach);
    if (!any(c > 0.0) || walk(o, l, reach - cellSize()) != 0u) return 0.0;
    return c;
}

// A quarter of the voxels a frame, one side at a time: direct light and a bounce ray.
[shader("compute")]
[numthreads(64, 1, 1)]
void inject(uint3 id : SV_DispatchThreadID) {
    uint frame = f.cache.x;
    uint index = id.x * 4u + (frame & 3u);
    if (index >= f.counts.x) return;
    uint4 voxel = voxels[index];
    if (voxel.x == FREE) return;
    int size = gridSize();
    uint volume = uint(size * size * size);
    int level = int(voxel.x / volume);
    uint at = voxel.x % volume;
    int3 low = levelLow(level);
    int3 wrapped = int3(at % uint(size), (at / uint(size)) % uint(size), at / uint(size * size));
    int3 c = low + ((wrapped - low) & (size - 1));
    // Inside a finer level rays walk that one, and never reach this voxel.
    if (level > 0) {
        int3 fine = c * 2 - levelLow(level - 1);
        if (all(fine >= 2) && all(fine < size - 4)) return;
    }
    float cell = levelCell(level);
    float3 x = (float3(c) + 0.5) * cell;
    uint side = (frame >> 2u) & 1u;
    float3 n = voxelNormal(voxel.z) * (side == 0u ? 1.0 : -1.0);
    float3 o = x + n * (1.5 * cell);
    uint slot = index * 6u + side * 3u;

    uint seed = hashU(index ^ hashU(frame));
    float u = float(seed >> 8u) / 16777216.0;
    float phi = 6.2831853 * float(hashU(seed) >> 8u) / 16777216.0;
    float3 dir = cosineDir(n, u, phi);
    uint found = walk(o, dir, f.sky.w);
    float3 seen = found != 0u ? voxelSeen(found, dir) : background(dir);
    uint lightSeed = hashU(seed ^ 0x68bc21ebu);
    float3 sampled = voxelSampled(x, n, o, lightSeed);
    float3 came = luminance(seen) * dir;
    seen += sampled;

    // The last 64 visits; a few while the lights change.
    float4 arriving = radiance[slot + 1u];
    float frames = min(arriving.w + 1.0, f.extra.y > 0.0 ? f.extra.y : 64.0);
    arriving = float4(lerp(arriving.rgb, seen, 1.0 / frames), frames);
    radiance[slot + 1u] = bad(arriving) ? float4(0.0) : arriving;
    float3 leaving = voxelAlbedo(voxel, x, n) * (voxelDirect(x, n, o, cell) + arriving.rgb);
    // w: the sampled point lights' share of `arriving`, as a mean luminance.
    // Residual GI takes `arriving` as a surface's bounce light, and the
    // renderer already lit that surface directly.
    float direct = lerp(radiance[slot].w, luminance(sampled), 1.0 / frames);
    radiance[slot] = float4(leaving, isnan(direct) ? 0.0 : direct);
    float3 way = lerp(radiance[slot + 2u].xyz, came, 1.0 / frames);
    radiance[slot + 2u] = float4(any(isnan(way)) ? float3(0.0) : way, 0.0);
}

// ---------------------------------------------------------------- voxelizing

// The free list's first `f.counts.x` entries, and every voxel free.
[shader("compute")]
[numthreads(64, 1, 1)]
void reset(uint3 id : SV_DispatchThreadID) {
    uint index = id.x;
    if (index >= f.counts.x) return;
    if (index == 0u) freeVoxels[0] = int(f.counts.x);
    // Popped from the end, so low indices go first.
    freeVoxels[index + 1u] = int(f.counts.x - 1u - index);
    voxels[index] = uint4(FREE, 0u, 0u, 0u);
}

void release(uint index) {
    voxels[index].x = FREE;
    int count;
    InterlockedAdd(freeVoxels[0], 1, count);
    freeVoxels[count + 1] = int(index);
}

// A job's cell for this thread, or false past its end.
bool jobCell(uint3 id, out int level, out int3 c) {
    int4 low = clip[MAX_LEVELS + id.y * 2u];
    int4 high = clip[MAX_LEVELS + id.y * 2u + 1u];
    level = low.w;
    c = low.xyz;
    if (id.x >= uint(high.w)) return false;
    int3 extent = high.xyz - low.xyz;
    int i = int(id.x);
    c += int3(i % extent.x, (i / extent.x) % extent.y, i / (extent.x * extent.y));
    return true;
}

// Frees the voxels in the cells about to be rebuilt.
[shader("compute")]
[numthreads(64, 1, 1)]
void clear(uint3 id : SV_DispatchThreadID) {
    if (all(id == 0u)) {
        large[0] = 0u;
        large[1] = 1u;
        large[2] = 1u;
    }
    int level;
    int3 c;
    if (!jobCell(id, level, c)) return;
    uint at = cellIndex(level, c);
    uint found = cells[at];
    if (found == 0u) return;
    release(found - 1u);
    cells[at] = 0u;
}

// The bounds of the part of triangle abc inside the box lo..hi, or false when
// none is. Bounding the whole triangle instead, one that crosses a thin slab
// would visit every cell face it spans.
bool clippedBounds(float3 a, float3 b, float3 c, float3 lo, float3 hi, out float3 low, out float3 high) {
    float3 points[9];
    float3 kept[9];
    points[0] = a;
    points[1] = b;
    points[2] = c;
    int count = 3;
    [unroll] for (int plane = 0; plane < 6; plane++) {
        int axis = plane % 3;
        float side = plane < 3 ? 1.0 : -1.0;
        float bound = plane < 3 ? lo[axis] : hi[axis];
        int n = 0;
        for (int i = 0; i < count; i++) {
            float3 p = points[i];
            float3 q = points[(i + 1) % count];
            float dp = (p[axis] - bound) * side, dq = (q[axis] - bound) * side;
            if (dp >= 0.0) kept[n++] = p;
            if ((dp >= 0.0) != (dq >= 0.0)) kept[n++] = lerp(p, q, dp / (dp - dq));
        }
        count = n;
        for (int i = 0; i < count; i++) points[i] = kept[i];
    }
    low = 1e30;
    high = -1e30;
    for (int i = 0; i < count; i++) {
        low = min(low, points[i]);
        high = max(high, points[i]);
    }
    return count > 0;
}

// A triangle as the voxelizer rasterizes it: seen down its steepest axis k,
// across axes u and v.
struct Raster {
    float3 a, b, c, n;
    int k, u, v;
    float2 pa, pb, pc;
    float facing;
    int cover;
    int3 normal;
    uint material, ground;
};

bool rasterSetup(uint triangle, out Raster r) {
    float4 a4 = triangles[triangle * 3u];
    float4 b4 = triangles[triangle * 3u + 1u];
    float4 c4 = triangles[triangle * 3u + 2u];
    r.a = a4.xyz;
    r.b = b4.xyz;
    r.c = c4.xyz;
    r.material = asuint(a4.w);
    r.ground = asuint(b4.w);
    float3 cr = cross(r.b - r.a, r.c - r.a);
    float area = length(cr);
    r.n = cr / max(area, 1e-30);
    float3 an = abs(r.n);
    r.k = an.x >= an.y && an.x >= an.z ? 0 : (an.y >= an.z ? 1 : 2);
    r.u = (r.k + 1) % 3;
    r.v = (r.k + 2) % 3;
    r.pa = float2(r.a[r.u], r.a[r.v]);
    r.pb = float2(r.b[r.u], r.b[r.v]);
    r.pc = float2(r.c[r.u], r.c[r.v]);
    r.facing = sign((r.pb.x - r.pa.x) * (r.pc.y - r.pa.y) - (r.pb.y - r.pa.y) * (r.pc.x - r.pa.x));
    r.cover = int(round(COVER_ONE / (float(SAMPLES * SAMPLES) * max(an[r.k], 1e-6))));
    r.normal = int3(round(r.n * 127.0));
    return area > 1e-12;
}

// The cells of job j the triangle may touch, or false when none.
bool jobSpan(Raster r, uint j, out int level, out int3 from, out int3 to) {
    int4 low = clip[MAX_LEVELS + j * 2u];
    int4 high = clip[MAX_LEVELS + j * 2u + 1u];
    level = low.w;
    float cell = levelCell(level);
    from = max(int3(floor(min(r.a, min(r.b, r.c)) / cell)), low.xyz);
    to = min(int3(floor(max(r.a, max(r.b, r.c)) / cell)), high.xyz - 1);
    if (any(from > to)) return false;
    // Big triangles only: the part inside the job may be much smaller.
    int3 span = to - from;
    if (span[r.u] * span[r.v] > 16) {
        float3 clippedLow, clippedHigh;
        if (!clippedBounds(r.a, r.b, r.c, float3(low.xyz) * cell, float3(high.xyz) * cell, clippedLow, clippedHigh))
            return false;
        from = max(from, int3(floor(clippedLow / cell)));
        to = min(to, int3(floor(clippedHigh / cell)));
    }
    return all(from <= to);
}

// Visits the triangle's samples in one cell face's column: SAMPLES by SAMPLES
// across it, each standing for an equal share of the triangle's area. Claiming
// gives a cell a voxel; otherwise each sample adds its cover and the normal.
// Samples landing in the same cell as the last are merged, saving atomics on
// coarse levels, where many triangles share a voxel.
void addSamples(uint index, int samples, Raster r) {
    uint found = cells[index];
    if (samples == 0 || found == 0u) return;
    uint voxel = found - 1u;
    InterlockedAdd(build[voxel * 4u], r.cover * samples);
    InterlockedAdd(build[voxel * 4u + 1u], r.normal.x * samples);
    InterlockedAdd(build[voxel * 4u + 2u], r.normal.y * samples);
    InterlockedAdd(build[voxel * 4u + 3u], r.normal.z * samples);
}

void rasterizeColumn(Raster r, int level, int3 from, int3 to, int cu, int cv, bool claiming) {
    float cell = levelCell(level);
    uint last = FREE;
    int samples = 0;
    [loop] for (int s = 0; s < SAMPLES * SAMPLES; s++) {
        float2 q = (float2(cu, cv) + (float2(s % SAMPLES, s / SAMPLES) + 0.5) / float(SAMPLES)) * cell;
        float e0 = (r.pb.x - r.pa.x) * (q.y - r.pa.y) - (r.pb.y - r.pa.y) * (q.x - r.pa.x);
        float e1 = (r.pc.x - r.pb.x) * (q.y - r.pb.y) - (r.pc.y - r.pb.y) * (q.x - r.pb.x);
        float e2 = (r.pa.x - r.pc.x) * (q.y - r.pc.y) - (r.pa.y - r.pc.y) * (q.x - r.pc.x);
        if (min(e0 * r.facing, min(e1 * r.facing, e2 * r.facing)) < 0.0) continue;
        float depth = r.a[r.k] - (r.n[r.u] * (q.x - r.a[r.u]) + r.n[r.v] * (q.y - r.a[r.v])) / r.n[r.k];
        int ck = int(floor(depth / cell));
        if (ck < from[r.k] || ck > to[r.k]) continue;
        int3 at = 0;
        at[r.k] = ck;
        at[r.u] = cu;
        at[r.v] = cv;
        uint index = cellIndex(level, at);
        if (index != last && !claiming) {
            addSamples(last, samples, r);
            samples = 0;
        }
        if (index == last && claiming) continue;
        last = index;
        if (claiming) {
            uint before;
            InterlockedCompareExchange(cells[index], 0u, CLAIMING, before);
            if (before != 0u) continue;
            int count;
            InterlockedAdd(freeVoxels[0], -1, count);
            if (count <= 0) {
                // Out of voxels: the cell stays empty.
                InterlockedAdd(freeVoxels[0], 1);
                cells[index] = 0u;
                continue;
            }
            uint voxel = uint(freeVoxels[count]);
            voxels[voxel] = uint4(index, r.material, packNormal(r.n), r.ground);
            build[voxel * 4u] = 0;
            build[voxel * 4u + 1u] = 0;
            build[voxel * 4u + 2u] = 0;
            build[voxel * 4u + 3u] = 0;
            cells[index] = voxel + 1u;
        } else {
            samples++;
        }
    }
    if (!claiming) addSamples(last, samples, r);
}

// A triangle covering more columns of a job than SMALL goes on `large`, whose
// entries a whole group shares, CHUNK columns each: one thread walking a wall
// that lies in a slab would hold up the whole pass.
static const int SMALL = 8;
static const int CHUNK = 256;
static const uint MAX_LARGE = 65535u;

// Rasterizes a triangle's small spans of the jobs; claiming also lists its
// large ones, which `accumulate` then leaves to `accumulateLarge`.
void rasterize(uint triangle, bool claiming) {
    Raster r;
    if (!rasterSetup(triangle, r)) return;
    [loop] for (uint j = 0u; j < jobs(); j++) {
        int level;
        int3 from, to;
        if (!jobSpan(r, j, level, from, to)) continue;
        int width = to[r.u] - from[r.u] + 1;
        int columns = width * (to[r.v] - from[r.v] + 1);
        if (columns > SMALL) {
            if (!claiming) continue;
            uint chunks = uint((columns + CHUNK - 1) / CHUNK);
            uint first;
            InterlockedAdd(large[0], chunks, first);
            // shortcut: entries past MAX_LARGE in a frame are dropped and their
            // cells stay empty; raise it if huge rebuilds show holes.
            for (uint i = 0u; i < chunks && first + i < MAX_LARGE; i++) {
                uint at = 4u + (first + i) * 4u;
                large[at] = triangle;
                large[at + 1u] = j;
                large[at + 2u] = i * uint(CHUNK);
            }
            continue;
        }
        [loop] for (int i = 0; i < columns; i++)
            rasterizeColumn(r, level, from, to, from[r.u] + i % width, from[r.v] + i / width, claiming);
    }
}

// More triangles than one dispatch row holds loop round.
static const uint TRIANGLE_STRIDE = 65535u * 64u;

// Whether the run of 64 triangles holding t may reach a job. Most do not, and
// reading every triangle of a big scene for each slab is what would cost.
bool chunkNeeded(uint t) {
    float3 lo = chunkBounds[(t / 64u) * 2u].xyz;
    float3 hi = chunkBounds[(t / 64u) * 2u + 1u].xyz;
    [loop] for (uint j = 0u; j < jobs(); j++) {
        int4 low = clip[MAX_LEVELS + j * 2u];
        int4 high = clip[MAX_LEVELS + j * 2u + 1u];
        float cell = levelCell(low.w);
        if (all(hi >= float3(low.xyz) * cell) && all(lo <= float3(high.xyz) * cell)) return true;
    }
    return false;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void claim(uint3 id : SV_DispatchThreadID) {
    for (uint t = id.x; t < f.cache.z; t += TRIANGLE_STRIDE)
        if (chunkNeeded(t)) rasterize(t, true);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void accumulate(uint3 id : SV_DispatchThreadID) {
    for (uint t = id.x; t < f.cache.z; t += TRIANGLE_STRIDE)
        if (chunkNeeded(t)) rasterize(t, false);
}

// The list may have overflowed: dispatch only the entries it holds.
[shader("compute")]
[numthreads(1, 1, 1)]
void capLarge() {
    large[0] = min(large[0], MAX_LARGE);
}

// One group per entry of `large`, its threads sharing the entry's columns.
void rasterizeLarge(uint entry, uint thread, bool claiming) {
    uint at = 4u + entry * 4u;
    Raster r;
    rasterSetup(large[at], r);
    int level;
    int3 from, to;
    if (!jobSpan(r, large[at + 1u], level, from, to)) return;
    int width = to[r.u] - from[r.u] + 1;
    int columns = width * (to[r.v] - from[r.v] + 1);
    int start = int(large[at + 2u]);
    int end = min(start + CHUNK, columns);
    [loop] for (int i = start + int(thread); i < end; i += 64)
        rasterizeColumn(r, level, from, to, from[r.u] + i % width, from[r.v] + i / width, claiming);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void claimLarge(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    rasterizeLarge(group.x, thread.x, true);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void accumulateLarge(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID) {
    rasterizeLarge(group.x, thread.x, false);
}

// Keeps a voxel whose cell holds enough surface, with the mean normal of what
// fell in it, or the first where they cancel. A cell a surface barely clips
// stays solid only sometimes, in proportion to the surface in it, by a fixed
// hash so it does not shimmer: marking all of them fattens everything by up to
// a cell and plugs railings and narrow openings.
[shader("compute")]
[numthreads(64, 1, 1)]
void finalize(uint3 id : SV_DispatchThreadID) {
    int level;
    int3 c;
    if (!jobCell(id, level, c)) return;
    uint at = cellIndex(level, c);
    uint found = cells[at];
    if (found == 0u) return;
    uint voxel = found - 1u;
    float cover = float(build[voxel * 4u]) / COVER_ONE;
    uint hash = hashU(hashU(hashU(uint(c.x) ^ (uint(level) << 28u)) ^ uint(c.y)) ^ uint(c.z));
    if (float(hash >> 8u) / 16777216.0 >= cover) {
        release(voxel);
        cells[at] = 0u;
        return;
    }
    float3 sum = float3(build[voxel * 4u + 1u], build[voxel * 4u + 2u], build[voxel * 4u + 3u]);
    if (length(sum) > 0.1 * 127.0) voxels[voxel].z = packNormal(normalize(sum));
    for (uint i = 0u; i < 6u; i++) radiance[voxel * 6u + i] = 0.0;
}

// ---------------------------------------------------------------- screen

// The bounce light arriving at surface point p facing n, as the voxels around
// it have settled on it over the last 64 frames, blended between their
// centres as a texture is filtered; false where none has. Each gives the side
// facing n, and voxels facing across it (the other wall of a corner) count
// for less.
bool voxelArriving(float3 p, float3 n, out float3 arriving) {
    arriving = 0.0;
    int level = levelAt(p);
    float cell = levelCell(level);
    int3 low = levelLow(level);
    // A little into the surface, whose voxels may sit just behind it.
    float3 c = (p - n * (0.25 * cell)) / cell - 0.5;
    int3 base = int3(floor(c));
    float3 t = c - float3(base);
    float4 sum = 0.0;
    [loop] for (int i = 0; i < 8; i++) {
        int3 o = int3(i & 1, (i >> 1) & 1, i >> 2);
        int3 at = base + o;
        if (any(at < low) || any(at >= low + gridSize())) continue;
        uint found = cells[cellIndex(level, at)];
        if (found == 0u) continue;
        uint index = found - 1u;
        float3 vn = voxelNormal(voxels[index].z);
        float facing = dot(vn, n);
        uint side = facing >= 0.0 ? 0u : 1u;
        uint slot = index * 6u + side * 3u;
        float4 a = radiance[slot + 1u];
        if (a.w < 1.0) continue;
        float3 w3 = lerp(1.0 - t, t, float3(o));
        float w = w3.x * w3.y * w3.z * abs(facing);
        if (w <= 0.0) continue;
        float3 bounce = a.rgb * saturate(1.0 - radiance[slot].w / max(luminance(a.rgb), 1e-4));
        // Tilted from the voxel's normal to the pixel's. Radiance linear in
        // direction (a + b.w), sampled cosine-weighted about the side's normal
        // s, gives irradiance / pi at n of the mean plus 2/3 b.(n - s), and the
        // light's mean way across s is b/4 of it: 8/3 of that way, along n.
        float3 s = side == 0u ? vn : -vn;
        float3 way = radiance[slot + 2u].xyz;
        float3 across = way - s * dot(way, s);
        float tilt = clamp(1.0 + 8.0 / 3.0 * dot(across, n) / max(luminance(bounce), 1e-4), 0.0, 3.0);
        sum += float4(bounce * tilt * w, w);
    }
    if (sum.w < 1e-3) return false;
    arriving = sum.rgb / sum.w;
    return true;
}

// Rays from a quarter-size texel: writes the light they found / rays and the
// fraction that hit something.
//
// Residual GI (extra.x) estimates the same light with a control variate: the
// voxels' settled light at the surface, which is smooth and costs no ray, plus
// how far each ray's light differs from what the voxels show along it. A ray
// the screen can't follow would see what the voxels do, so it adds nothing and
// walks nothing; one it can carries the contact shadow and detail the voxels
// are too coarse for. Where no voxel has settled, the texel traces as before.
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
    float3 noise = blueNoise(px).xyz;
    int rays = clamp(int(f.cache.w), 1, 4);
    float3 light = 0.0;
    float hits = 0.0;
    float3 origin = p + n * (0.02 + 0.002 * linearDepth(depth));
    float3 voxelOrigin = p + n * (1.5 * levelCell(levelAt(p)));
    float3 proxy;
    bool residual = f.extra.x > 0.5 && voxelArriving(p, n, proxy);
    [loop] for (int r = 0; r < rays; r++) {
        float index = f.misc.x * float(rays) + float(r);
        float u = frac(noise.x + index * 0.7548776662);
        float phi = 6.2831853 * frac(noise.y + index * 0.5698402910);
        float3 dir = cosineDir(n, u, phi);
        float jitter = frac(noise.z + index * 0.6180339887);
        float4 seen = march(origin, dir, f.misc.y, STEPS, jitter);
        if (residual) {
            if (seen.w > 0.5) {
                uint found = walk(voxelOrigin, dir, f.sky.w);
                light += clampFirefly(seen.rgb) - (found != 0u ? voxelSeen(found, dir) : background(dir));
            }
        } else if (seen.w > 0.5) {
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
    // A residual texel's light covers the sky too: nothing is left open.
    float4 result = residual ? float4(max(proxy + light / float(rays), 0.0), 1.0)
                             : float4(light, hits) / float(rays);
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
    colorOut.Store(at, float4(diffuseLit(lit.rgb, brdf, rough, n, view) + diffuse * brdf.diffuse_color, lit.a));
}
