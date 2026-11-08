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
// that side and the mean bounce light arriving (w frames).
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
    return radiance[index * 4u + side * 2u].rgb;
}

// Irradiance / pi from the sun and, while they are few, the point lights,
// shadowed by the voxels. Many point lights come from `voxelSampled`.
float3 voxelDirect(float3 x, float3 n, float3 o) {
    float3 light = 0.0;
    float lit = dot(n, f.sunDirection.xyz);
    if (lit > 0.0 && f.sunDirection.w > 0.5 && walk(o, f.sunDirection.xyz, 1e6) == 0u)
        light += f.sunColor.rgb * lit;
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
    uint slot = index * 4u + side * 2u;

    uint seed = hashU(index ^ hashU(frame));
    float u = float(seed >> 8u) / 16777216.0;
    float phi = 6.2831853 * float(hashU(seed) >> 8u) / 16777216.0;
    float3 dir = cosineDir(n, u, phi);
    uint found = walk(o, dir, f.sky.w);
    float3 seen = found != 0u ? voxelSeen(found, dir) : background(dir);
    uint lightSeed = hashU(seed ^ 0x68bc21ebu);
    seen += voxelSampled(x, n, o, lightSeed);

    // The last 64 visits; a few while the lights change.
    float4 arriving = radiance[slot + 1u];
    float frames = min(arriving.w + 1.0, f.extra.y > 0.0 ? f.extra.y : 64.0);
    arriving = float4(lerp(arriving.rgb, seen, 1.0 / frames), frames);
    radiance[slot + 1u] = bad(arriving) ? float4(0.0) : arriving;
    float3 leaving = voxelAlbedo(voxel, x, n) * (voxelDirect(x, n, o) + arriving.rgb);
    radiance[slot] = float4(leaving, 1.0);
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

// Visits a triangle's samples in the jobs' cells: rows of SAMPLES by SAMPLES
// across each cell face it covers seen down its steepest axis, each standing for
// an equal share of its area. Claiming gives a cell a voxel; otherwise each
// sample adds its cover and the triangle's normal.
void rasterize(uint triangle, bool claiming) {
    float4 a4 = triangles[triangle * 3u];
    float4 b4 = triangles[triangle * 3u + 1u];
    float4 c4 = triangles[triangle * 3u + 2u];
    float3 a = a4.xyz, b = b4.xyz, c = c4.xyz;
    float3 cr = cross(b - a, c - a);
    float area = length(cr);
    if (!(area > 1e-12)) return;
    float3 n = cr / area;
    float3 an = abs(n);
    int k = an.x >= an.y && an.x >= an.z ? 0 : (an.y >= an.z ? 1 : 2);
    int u = (k + 1) % 3, v = (k + 2) % 3;
    float2 pa = float2(a[u], a[v]), pb = float2(b[u], b[v]), pc = float2(c[u], c[v]);
    float facing = sign((pb.x - pa.x) * (pc.y - pa.y) - (pb.y - pa.y) * (pc.x - pa.x));
    float3 lowest = min(a, min(b, c)), highest = max(a, max(b, c));
    int cover = int(round(COVER_ONE / (float(SAMPLES * SAMPLES) * an[k])));
    int3 normal = int3(round(n * 127.0));
    [loop] for (uint j = 0u; j < jobs(); j++) {
        int4 low = clip[MAX_LEVELS + j * 2u];
        int4 high = clip[MAX_LEVELS + j * 2u + 1u];
        int level = low.w;
        float cell = levelCell(level);
        int3 from = max(int3(floor(lowest / cell)), low.xyz);
        int3 to = min(int3(floor(highest / cell)), high.xyz - 1);
        if (any(from > to)) continue;
        // Big triangles only: the part inside the job may be much smaller.
        int3 span = to - from;
        if (span[u] * span[v] > 16) {
            float3 clippedLow, clippedHigh;
            float3 box = float3(low.xyz) * cell;
            if (!clippedBounds(a, b, c, box, float3(high.xyz) * cell, clippedLow, clippedHigh)) continue;
            from = max(from, int3(floor(clippedLow / cell)));
            to = min(to, int3(floor(clippedHigh / cell)));
            if (any(from > to)) continue;
        }
        [loop] for (int cu = from[u]; cu <= to[u]; cu++) {
            [loop] for (int cv = from[v]; cv <= to[v]; cv++) {
                [loop] for (int s = 0; s < SAMPLES * SAMPLES; s++) {
                    float2 q = (float2(cu, cv) + (float2(s % SAMPLES, s / SAMPLES) + 0.5) / float(SAMPLES)) * cell;
                    float e0 = (pb.x - pa.x) * (q.y - pa.y) - (pb.y - pa.y) * (q.x - pa.x);
                    float e1 = (pc.x - pb.x) * (q.y - pb.y) - (pc.y - pb.y) * (q.x - pb.x);
                    float e2 = (pa.x - pc.x) * (q.y - pc.y) - (pa.y - pc.y) * (q.x - pc.x);
                    if (min(e0 * facing, min(e1 * facing, e2 * facing)) < 0.0) continue;
                    float depth = a[k] - (n[u] * (q.x - a[u]) + n[v] * (q.y - a[v])) / n[k];
                    int ck = int(floor(depth / cell));
                    if (ck < from[k] || ck > to[k]) continue;
                    int3 at = 0;
                    at[k] = ck;
                    at[u] = cu;
                    at[v] = cv;
                    uint index = cellIndex(level, at);
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
                        voxels[voxel] = uint4(index, asuint(a4.w), packNormal(n), asuint(b4.w));
                        build[voxel * 4u] = 0;
                        build[voxel * 4u + 1u] = 0;
                        build[voxel * 4u + 2u] = 0;
                        build[voxel * 4u + 3u] = 0;
                        cells[index] = voxel + 1u;
                    } else {
                        uint found = cells[index];
                        if (found == 0u) continue;
                        uint voxel = found - 1u;
                        InterlockedAdd(build[voxel * 4u], cover);
                        InterlockedAdd(build[voxel * 4u + 1u], normal.x);
                        InterlockedAdd(build[voxel * 4u + 2u], normal.y);
                        InterlockedAdd(build[voxel * 4u + 3u], normal.z);
                    }
                }
            }
        }
    }
}

// More triangles than one dispatch row holds loop round.
static const uint TRIANGLE_STRIDE = 65535u * 64u;

[shader("compute")]
[numthreads(64, 1, 1)]
void claim(uint3 id : SV_DispatchThreadID) {
    for (uint t = id.x; t < f.cache.z; t += TRIANGLE_STRIDE) rasterize(t, true);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void accumulate(uint3 id : SV_DispatchThreadID) {
    for (uint t = id.x; t < f.cache.z; t += TRIANGLE_STRIDE) rasterize(t, false);
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
    for (uint i = 0u; i < 4u; i++) radiance[voxel * 4u + i] = 0.0;
}

// ---------------------------------------------------------------- screen

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
    float3 noise = blueNoise(px).xyz;
    int rays = clamp(int(f.cache.w), 1, 4);
    float3 light = 0.0;
    float hits = 0.0;
    float3 origin = p + n * (0.02 + 0.002 * linearDepth(depth));
    float3 voxelOrigin = p + n * (1.5 * levelCell(levelAt(p)));
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
    colorOut.Store(at, float4(diffuseLit(lit.rgb, brdf, rough, n, view) + diffuse * brdf.diffuse_color, lit.a));
}
