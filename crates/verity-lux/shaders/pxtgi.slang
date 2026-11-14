// PXTGI: bounce light traced on screen, then against the physics scene.
// Ported from the godot_physx module's PXTGI (physx_gi_effect.cpp) to world
// space. Depth is reverse-Z infinite perspective and normals are world space.
// Light is kept as Godot's GI buffer holds it: the mean radiance arriving,
// cosine weighted (irradiance / pi).
//
// Rays run against verity-physics: `RayScene` builds the trees and refits them
// as bodies move, and tells these passes where things moved.

#define MATERIAL_SET 2
#include "materials.slang"
#include "brdf.slang"
#include "terrain.slang"
// `temporal` seeds new texels with `historyPrior`, defined below.
#define HISTORY_PRIOR
// Buffers, as pointers ending `Frame` (`FrameData::buffers`): primitives, their
// tree, the radiance cache's keys and cells, static mesh tree nodes (two each)
// and triangles (three corners), the same for soft bodies (rewritten as they
// deform), and the reference path tracer's fp32 sum.
#define METHOD_POINTERS \
    Primitive* prims; Node* nodes; Atomic<uint>* cacheKeys; Cell* cells; \
    float4* geo; float4* softGeo; float4* referenceSum;
#include "screen_gi.slang"
#include "wave_batch.slang"

// Debug views, as `PxtgiDebug` numbers them.
static const int DEBUG_ROUND_TRIP = 1;
static const int DEBUG_WORLD_ONLY = 2;
static const int DEBUG_WORLD_HITS = 3;
static const int DEBUG_WORLD_POSITION = 4;
static const int DEBUG_BOUNCE_LIGHT = 5;
static const int DEBUG_REFLECTIONS = 6;

// Primitive kinds.
static const int KIND_BOX = 0;
static const int KIND_SPHERE = 1;
static const int KIND_MESH = 2;
static const int KIND_SOFT = 3;
static const uint NO_MATERIAL = 0xffffffffu;

// One thing rays hit: world-to-local rows; shape (x kind; box yzw half extents;
// sphere y radius; a mesh's first tree node and first triangle in its geometry
// buffer, as int bits); albedo (w a material index as bits, whose textures
// tint it: NO_MATERIAL for none).
struct Primitive {
    float4 r0;
    float4 r1;
    float4 r2;
    float4 shape;
    float4 albedo;
};

// Two float4 per node: lo (w: right child, or first primitive of a leaf) and
// hi (w: 0 inner, else leaf count), ints as float bits; left child follows.
struct Node {
    float4 lo;
    float4 hi;
};

// Radiance cache cell: where it was hit (w: frame last used, as bits), normal
// (w: refinements), albedo, bounce light arriving, light leaving.
struct Cell {
    float4 pos;
    float4 normal;
    float4 albedo;
    float4 incoming;
    float4 outgoing;
};

// ---------------------------------------------------------------- scene trace

// Where the ray enters a box (1e30: misses it, or only beyond best). `oi` is
// the origin times `inv`, once per ray, so each slab is one multiply-add
// (Aila and Laine, "Understanding the efficiency of ray traversal on GPUs").
float boxEntry(float3 oi, float3 inv, float3 lo, float3 hi, float best) {
    float3 t0 = lo * inv - oi;
    float3 t1 = hi * inv - oi;
    float3 nearT = min(t0, t1);
    float3 farT = max(t0, t1);
    float enter = max(max(nearT.x, nearT.y), max(nearT.z, 0.0));
    float leave = min(min(farT.x, farT.y), min(farT.z, best));
    return enter <= leave ? enter : 1e30;
}

// `soft` is a literal at every call: inlined, each walk reads one buffer
// rather than selecting between two loads.
[ForceInline]
float4 geoAt(bool soft, int at) {
    if (soft) return f.softGeo[at];
    return f.geo[at];
}

// The first corner of the static triangle the last closest-hit ray took, for
// what the corners' w holds.
static int hitCorners = 0;

[ForceInline]
float meshBoxEntry(bool soft, float3 oi, float3 inv, int at, float best) {
    return boxEntry(oi, inv, geoAt(soft, at).xyz, geoAt(soft, at + 1).xyz, best);
}

// The nearest triangle of a mesh along a local ray (either side), walking its
// own tree near child first. Shadow rays only need the first intersection.
// The trees are bounded to about this deep; a deeper one only drops the
// farthest branches.
[ForceInline]
bool hitMesh(bool soft, int nodesAt, int tris, float3 o, float3 d, inout float t, inout float3 n, bool anyHit) {
    bool found = false;
    float3 inv = safeInv(d);
    float3 oi = o * inv;
    if (meshBoxEntry(soft, oi, inv, nodesAt, t) >= 1e30) return false;
    int stack[32];
    int top = 0;
    int node = 0;
    [loop] while (true) {
        float4 lo = geoAt(soft, nodesAt + node * 2);
        float4 hi = geoAt(soft, nodesAt + node * 2 + 1);
        int count = asint(hi.w);
        if (count == 0) {
            int left = node + 1;
            int right = asint(lo.w);
            float tl = meshBoxEntry(soft, oi, inv, nodesAt + left * 2, t);
            float tr = meshBoxEntry(soft, oi, inv, nodesAt + right * 2, t);
            if (tl < 1e30 && tr < 1e30) {
                bool leftFirst = tl <= tr;
                if (top < 32) {
                    stack[top] = leftFirst ? right : left;
                    top++;
                }
                node = leftFirst ? left : right;
                continue;
            } else if (tl < 1e30) {
                node = left;
                continue;
            } else if (tr < 1e30) {
                node = right;
                continue;
            }
        } else {
            int first = asint(lo.w);
            [loop] for (int k = first; k < first + count; k++) {
                float3 a = geoAt(soft, tris + k * 3).xyz;
                float3 e1 = geoAt(soft, tris + k * 3 + 1).xyz - a;
                float3 e2 = geoAt(soft, tris + k * 3 + 2).xyz - a;
                float3 pv = cross(d, e2);
                float det = dot(e1, pv);
                if (abs(det) < 1e-12) continue;
                float invDet = 1.0 / det;
                float3 tv = o - a;
                float u = dot(tv, pv) * invDet;
                if (u < 0.0 || u > 1.0) continue;
                float3 qv = cross(tv, e1);
                float w = dot(d, qv) * invDet;
                if (w < 0.0 || u + w > 1.0) continue;
                float th = dot(e2, qv) * invDet;
                if (th > 0.0 && th < t) {
                    t = th;
                    n = cross(e1, e2);
                    found = true;
                    if (anyHit) return true;
                    hitCorners = tris + k * 3;
                }
            }
        }
        bool next = false;
        [loop] while (top > 0) {
            top--;
            int s = stack[top];
            if (meshBoxEntry(soft, oi, inv, nodesAt + s * 2, t) < 1e30) {
                node = s;
                next = true;
                break;
            }
        }
        if (!next) break;
    }
    // Facing the ray: a mesh is hit from either side.
    if (found && dot(n, d) > 0.0) n = -n;
    return found;
}

// The nearest hit of one primitive closer than best; its world normal.
bool hitPrimitive(Primitive q, float3 o, float3 d, inout float best, out float3 normal, bool anyHit) {
    normal = float3(0.0);
    float3 lo = float3(dot(q.r0.xyz, o) + q.r0.w, dot(q.r1.xyz, o) + q.r1.w, dot(q.r2.xyz, o) + q.r2.w);
    float3 ld = float3(dot(q.r0.xyz, d), dot(q.r1.xyz, d), dot(q.r2.xyz, d));
    float3 n = float3(0.0);
    float t = 0.0;
    int kind = int(q.shape.x + 0.5);
    if (kind >= KIND_MESH) {
        t = best;
        bool found = kind == KIND_SOFT
            ? hitMesh(true, asint(q.shape.y), asint(q.shape.z), lo, ld, t, n, anyHit)
            : hitMesh(false, asint(q.shape.y), asint(q.shape.z), lo, ld, t, n, anyHit);
        if (!found) return false;
    } else if (kind == KIND_BOX) {
        float3 inv = safeInv(ld);
        float3 t0 = (-q.shape.yzw - lo) * inv;
        float3 t1 = (q.shape.yzw - lo) * inv;
        float3 loT = min(t0, t1);
        float3 hiT = max(t0, t1);
        t = max(max(loT.x, loT.y), loT.z);
        float outT = min(min(hiT.x, hiT.y), hiT.z);
        if (t > outT || t <= 0.0 || t >= best) return false;
        if (t == loT.x) {
            n = float3(-sign(ld.x), 0.0, 0.0);
        } else if (t == loT.y) {
            n = float3(0.0, -sign(ld.y), 0.0);
        } else {
            n = float3(0.0, 0.0, -sign(ld.z));
        }
    } else {
        float a = dot(ld, ld);
        float b = dot(lo, ld);
        float c = dot(lo, lo) - q.shape.y * q.shape.y;
        float h = b * b - a * c;
        if (h < 0.0) return false;
        t = (-b - sqrt(h)) / a;
        if (t <= 0.0 || t >= best) return false;
        n = lo + ld * t;
    }
    best = t;
    // Local to world for a normal: the transpose of the world-to-local rows.
    normal = normalize(n.x * q.r0.xyz + n.y * q.r1.xyz + n.z * q.r2.xyz);
    return true;
}

#ifdef GI_RAY_QUERY
// The same scene as hardware acceleration structures (`Hardware` in pxtgi.rs):
// one instance per primitive, whose custom index is its index in `prims`. A
// static mesh is its triangles, in order, so a hit's primitive index is its
// triangle; anything else is a box around it whose candidates `hitPrimitive` tests.
[[vk::binding(15, 0)]] RaytracingAccelerationStructure sceneTlas;

int walkWorld(float3 o, float3 d, inout float t, out float3 normal, bool anyHit) {
    normal = float3(0.0);
    if (f.counts.x == 0u) return -1;
    RayDesc ray;
    ray.Origin = o;
    ray.Direction = d;
    ray.TMin = 0.0;
    ray.TMax = t;
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(sceneTlas, anyHit ? RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH : RAY_FLAG_NONE, 0xffu, ray);
    float3 boxNormal = 0.0;
    int boxCorners = 0;
    [loop] while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            float best = q.CommittedStatus() == COMMITTED_NOTHING ? t : q.CommittedRayT();
            float3 n;
            if (hitPrimitive(f.prims[q.CandidateInstanceID()], o, d, best, n, anyHit)) {
                boxNormal = n;
                boxCorners = hitCorners;
                q.CommitProceduralPrimitiveHit(best);
            }
        } else {
            q.CommitNonOpaqueTriangleHit();
        }
    }
    if (q.CommittedStatus() == COMMITTED_NOTHING) return -1;
    int hit = q.CommittedInstanceID();
    t = q.CommittedRayT();
    if (q.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
        normal = boxNormal;
        hitCorners = boxCorners;
        return hit;
    }
    Primitive p = f.prims[hit];
    int corners = asint(p.shape.z) + int(q.CommittedPrimitiveIndex()) * 3;
    hitCorners = corners;
    float3 a = f.geo[corners].xyz;
    float3 n = cross(f.geo[corners + 1].xyz - a, f.geo[corners + 2].xyz - a);
    normal = normalize(n.x * p.r0.xyz + n.y * p.r1.xyz + n.z * p.r2.xyz);
    // Facing the ray: a mesh is hit from either side.
    if (dot(normal, d) > 0.0) normal = -normal;
    return hit;
}
#else
// The nearest primitive along the ray, near child first; with anyHit the first
// one in the way (shadow rays). Returns its index, or -1.
int walkWorld(float3 o, float3 d, inout float t, out float3 normal, bool anyHit) {
    normal = float3(0.0);
    int hit = -1;
    if (f.counts.y == 0u) return hit;
    float3 inv = safeInv(d);
    float3 oi = o * inv;
    if (boxEntry(oi, inv, f.nodes[0].lo.xyz, f.nodes[0].hi.xyz, t) >= 1e30) return hit;
    int stack[24];
    int top = 0;
    int node = 0;
    [loop] while (true) {
        Node nd = f.nodes[node];
        int count = asint(nd.hi.w);
        if (count == 0) {
            int left = node + 1;
            int right = asint(nd.lo.w);
            float tl = boxEntry(oi, inv, f.nodes[left].lo.xyz, f.nodes[left].hi.xyz, t);
            float tr = boxEntry(oi, inv, f.nodes[right].lo.xyz, f.nodes[right].hi.xyz, t);
            if (tl < 1e30 && tr < 1e30) {
                bool leftFirst = tl <= tr;
                if (top < 24) {
                    stack[top] = leftFirst ? right : left;
                    top++;
                }
                node = leftFirst ? left : right;
                continue;
            } else if (tl < 1e30) {
                node = left;
                continue;
            } else if (tr < 1e30) {
                node = right;
                continue;
            }
        } else {
            int first = asint(nd.lo.w);
            [loop] for (int i = first; i < first + count; i++) {
                float3 n;
                if (hitPrimitive(f.prims[i], o, d, t, n, anyHit)) {
                    hit = i;
                    normal = n;
                    if (anyHit) return hit;
                }
            }
        }
        // The next waiting node that could still hold something nearer.
        bool found = false;
        [loop] while (top > 0) {
            top--;
            int s = stack[top];
            if (boxEntry(oi, inv, f.nodes[s].lo.xyz, f.nodes[s].hi.xyz, t) < 1e30) {
                node = s;
                found = true;
                break;
            }
        }
        if (!found) break;
    }
    return hit;
}
#endif

bool blocked(float3 o, float3 d, float reach) {
    float t = reach;
    float3 n;
    return walkWorld(o, d, t, n, true) >= 0;
}

// A ray that hit nothing within reach has left the scene only if it also left
// the box around everything (a long tunnel doesn't see the sky).
bool escapes(float3 o, float3 d, float reach) {
    if (f.counts.y == 0u) return true;
    float3 inv = safeInv(d);
    float3 farT = max((f.nodes[0].lo.xyz - o) * inv, (f.nodes[0].hi.xyz - o) * inv);
    return min(min(farT.x, farT.y), farT.z) <= reach;
}

// The albedo of what a ray hit. A material's texture counts by its last mip,
// the mean colour: rays carry no texture coordinates. Terrain is worked out at
// the hit `x` with normal `n`, as its mean colour varies across the mesh.
float3 hitAlbedo(int hit, float3 x, float3 n) {
    float4 a = f.prims[hit].albedo;
    uint index = asuint(a.w);
    if (index == NO_MATERIAL) return a.rgb;
    Material m = materials[index];
    if ((m.textures.w & TERRAIN_SURFACE) != 0) {
        // Corners hold the ground as `pack_ground` in pxtgi.rs laid it out.
        float height = 0.0;
        float wet = 0.0;
        for (int corner = 0; corner < 3; corner++) {
            uint ground = asuint(f.geo[hitCorners + corner].w);
            height += float(ground & 0xffffu) / 16.0 - 2048.0;
            wet += float(ground >> 16u) / 65535.0;
        }
        return terrain_albedo(x, n, height / 3.0, wet / 3.0, m.params.w);
    }
    return meanAlbedo(m);
}

// ---------------------------------------------------------------- radiance cache


// Cells a quarter metre across near the camera, doubling with each doubling
// of distance past 8 m.
uint cacheLevel(float3 x) {
    float dist = length(x - f.camera.xyz);
    return uint(clamp(floor(log2(max(dist, 1.0) / 8.0)) + 1.0, 0.0, 5.0));
}

uint facing(float3 n) {
    float3 a = abs(n);
    if (a.x > a.y && a.x > a.z) return n.x > 0.0 ? 0u : 1u;
    if (a.y > a.z) return n.y > 0.0 ? 2u : 3u;
    return n.z > 0.0 ? 4u : 5u;
}

uint cellHash(int3 cell, uint side, uint level) {
    return hashU(uint(cell.x) + hashU(uint(cell.y) + hashU(uint(cell.z) + hashU(side * 8u + level))));
}

// The slot of x's cell at a level, claiming an empty one when asked; -1 when
// the table is full there.
int cacheProbe(uint h, uint key, bool claim, out bool claimed) {
    claimed = false;
    [loop] for (uint i = 0u; i < 8u; i++) {
        uint slot = (h + i) & f.cache.y;
        uint was = f.cacheKeys[slot].load();
        if (was == key) return int(slot);
        if (was == 0u) {
            if (!claim) return -1;
            uint previous = f.cacheKeys[slot].compareExchange(0u, key);
            if (previous == 0u) {
                claimed = true;
                return int(slot);
            }
            if (previous == key) return int(slot);
        }
    }
    return -1;
}

// The slot of the cell hashing to `h`; lanes looking up the first active
// lane's cell take its lookup (wave_batch.slang).
int cacheLookup(uint h, bool claim, out bool claimed) {
    claimed = false;
    uint key = hashU(h ^ 0x9e3779b9u) | 1u;
    uint tag = claim ? 1u : 0u;
    bool follows = waveBatchFollows(key, tag);
    int slot = follows ? -1 : cacheProbe(h, key, claim, claimed);
    bool taken;
    uint first = waveBatchTake(asuint(slot), key, tag, follows, taken);
    if (taken) return asint(first);
    return follows ? cacheProbe(h, key, claim, claimed) : slot;
}

int cacheSlot(float3 x, float3 n, uint level, bool claim, out bool claimed) {
    int3 cell = int3(floor(x / (0.25 * float(1u << level))));
    return cacheLookup(cellHash(cell, facing(n), level), claim, claimed);
}

// The bounce light arriving at x (or with leaving, the light leaving it),
// blended from the eight refined cells around it as a texture is filtered: a
// cell's own value would show as a square patch in a reflection.
float4 smoothCacheAt(float3 x, float3 n, bool leaving, uint level) {
    float3 c = x / (0.25 * float(1u << level)) - 0.5;
    int3 base = int3(floor(c));
    float3 t = c - float3(base);
    uint side = facing(n);
    float4 sum = 0.0;
    [loop] for (int i = 0; i < 8; i++) {
        int3 o = int3(i & 1, (i >> 1) & 1, i >> 2);
        bool unused;
        int slot = cacheLookup(cellHash(base + o, side, level), false, unused);
        if (slot >= 0 && f.cells[slot].normal.w > 0.5) {
            float3 w3 = lerp(1.0 - t, t, float3(o));
            float w = w3.x * w3.y * w3.z;
            float3 light = leaving ? f.cells[slot].outgoing.rgb : f.cells[slot].incoming.rgb;
            sum += float4(light * w, w);
        }
    }
    return sum;
}

float3 smoothCache(float3 x, float3 n, bool leaving) {
    uint level = cacheLevel(x);
    float4 sum = smoothCacheAt(x, n, leaving, level);
    // None refined here yet (just come into view, or the camera came closer and
    // cells shrank): the coarser cells around it, softer rather than dark.
    if (sum.w <= 1e-4) sum = smoothCacheAt(x, n, leaving, level + 1u);
    return sum.w > 1e-4 ? sum.rgb / sum.w : float3(0.0);
}

void claimCell(int slot, float3 x, float3 n, float3 albedo) {
    f.cells[slot].pos = float4(x, asfloat(f.cache.x));
    f.cells[slot].normal = float4(n, 0.0);
    f.cells[slot].albedo = float4(albedo, 0.0);
    f.cells[slot].incoming = 0.0;
}

// ---------------------------------------------------------------- lighting

// Irradiance / pi at a surface patch straight from the sun and, while they are
// few, the point lights, shadowed by the scene. Point lights fall off as the
// resolve's do. Many point lights come from `sampledLight` instead.
float3 directLight(float3 x, float3 n) {
    float3 light = 0.0;
    float3 o = x + n * 0.02;
    float lit = dot(n, f.sunDirection.xyz);
    if (lit > 0.0 && f.sunDirection.w > 0.5 && !blocked(o, f.sunDirection.xyz, 1e6)) {
        light += f.sunColor.rgb * lit;
    }
    if (sampledLights()) return light;
    [loop] for (uint i = 0u; i < f.counts.z; i++) {
        float3 l;
        float reach;
        float3 c = pointLight(i, x, n, l, reach);
        if (any(c > 0.0) && !blocked(o, l, reach - 0.05)) light += c;
    }
    return light;
}

// The point lights `directLight` leaves out: one of them picked and shadowed,
// an unbiased but noisy estimate for a running mean. 0 while they are few.
float3 sampledLight(float3 x, float3 n, inout uint seed) {
    if (!sampledLights()) return 0.0;
    float3 l;
    float reach;
    float3 c = pickPointLight(x, n, seed, l, reach);
    if (!any(c > 0.0) || blocked(x + n * 0.02, l, reach - 0.05)) return 0.0;
    return c;
}

uint positionSeed(float3 x) {
    return hashU(asuint(x.x) ^ hashU(asuint(x.y) ^ hashU(asuint(x.z) ^ hashU(f.cache.x))));
}

// How far reflection rays follow the screen before the scene.
static const float REFLECT_REACH = 24.0;

// How a hit is seen: by a diffuse ray, a glossy reflection, or a mirror.
static const uint SEEN_DIFFUSE = 0u;
static const uint SEEN_GLOSSY = 1u;
static const uint SEEN_MIRROR = 2u;

// The light leaving a surface toward the ray: for a reflection, last frame's
// image where that point was on it; otherwise its cache cell once refined (or
// while young, its coarser parent), or else the direct light on its albedo. Diffuse rays skip the image: its
// shading (shadow maps, Hammon diffuse as it then was, normal-map spread) ran
// brighter than the scene's own Lambertian light, and against a path-traced
// reference of the courtyard taking it cost 0.13 error where the cache gave
// 0.08. A pixel's ray
// claims the cell and keeps it alive. A
// reflection shows the point itself, so it takes the cells around it blended;
// a mirror's lights the point directly and takes only their bounce light (lit
// directly, a glossy surface's scattered rays would flicker between light and
// shadow).
float3 worldLight(float3 x, float3 n, int hit, bool fromPixels, uint seenAs) {
    float3 s = project(f.clipFromWorld, x);
    int debug = int(f.misc.w);
    bool worldOnly = debug >= DEBUG_WORLD_ONLY && debug <= DEBUG_WORLD_POSITION;
    if (!worldOnly && seenAs != SEEN_DIFFUSE && f.extra.z > 0.5 && s.z > f.camera.w && onScreen(s.xy) && abs(surfaceDepth(s.xy) - s.z) < 0.1 + 0.02 * s.z) {
        float2 pc;
        if (previousSurface(x, pc)) return lastImage(pc);
    }
    float3 albedo = hitAlbedo(hit, x, n);
    if (f.cache.z != 0u) {
        uint level = cacheLevel(x);
        bool claimed;
        int slot = cacheSlot(x, n, level, fromPixels, claimed);
        float own = 0.0;
        if (slot >= 0 && claimed) {
            claimCell(slot, x, n, albedo);
            // Its parent too, one level coarser, for this cell and its
            // neighbours to fall back on while they are young.
            bool parentClaimed;
            int parent = cacheSlot(x, n, level + 1u, true, parentClaimed);
            if (parentClaimed) claimCell(parent, x, n, albedo);
        } else if (slot >= 0) {
            if (fromPixels) f.cells[slot].pos.w = asfloat(f.cache.x);
            own = f.cells[slot].normal.w;
        }
        float3 incoming = own > 0.5 ? f.cells[slot].incoming.rgb : 0.0;
        float known = own;
        int parent = -1;
        if (own < 4.0) {
            // A cell with fewer than four rays leans on its parent, the more the
            // younger it is: a surface come into view starts soft, not dark or
            // noisy, and sharpens as its own cell fills (Hierarchical Voxel
            // Irradiance Caching's fallback to converged parents).
            bool unused;
            int p = cacheSlot(x, n, level + 1u, false, unused);
            float coarse = p >= 0 ? f.cells[p].normal.w : 0.0;
            if (coarse > own) {
                parent = p;
                incoming = lerp(f.cells[p].incoming.rgb, incoming, own / 4.0);
                known = coarse;
            }
        }
        if (known > 0.5) {
            // A pixel's ray lights the hit itself and takes only the cell's
            // bounce light: the cell's outgoing light is the direct light where
            // it was claimed, which leaks sun across shadow edges a cell
            // straddles (0.155 to 0.146 error). The cache's own rays gained
            // nothing from it, and read only `outgoing`, which no cell writes
            // while they run. Trace runs after the cache took its fade; the
            // cache's own rays don't.
            if (seenAs == SEEN_DIFFUSE && fromPixels) return albedo * (directLight(x, n) + incoming);
            if (seenAs == SEEN_DIFFUSE) return f.cells[own > 0.5 ? slot : parent].outgoing.rgb * f.fade.rgb;
            if (seenAs == SEEN_GLOSSY) return smoothCache(x, n, true);
            return albedo * (directLight(x, n) + smoothCache(x, n, false));
        }
    }
    uint seed = positionSeed(x);
    return albedo * (directLight(x, n) + sampledLight(x, n, seed));
}

// ---------------------------------------------------------------- passes

// Cells that lit the view in the last couple of frames refresh every second
// frame, the rest every eighth: work goes where the camera looks, while what it
// turned from stays warm (and alive, for 600 frames) for when it turns back.
// Every cell near something physics moved, and all while the lights change.
bool cacheDue(uint slot, Cell cell) {
    if (f.extra.y > 0.0 || movedNear(cell.pos.xyz)) return true;
    uint period = f.cache.x - asuint(cell.pos.w) <= 2u ? 4u : 16u;
    return (slot & (period - 1u)) == (f.cache.x & (period - 1u));
}

// For a texel with no history (just come into view): the cache's bounce light
// at its surface, so it starts near the answer rather than from two rays.
bool historyPrior(float3 p, float3 n, out float4 prior) {
    prior = 0.0;
    if (f.cache.z == 0u) return false;
    float3 light = smoothCache(p, n, false);
    if (!any(light > 0.0)) return false;
    prior = float4(light, 1.0);
    return true;
}

// The arriving half of a cache update: one ray per due cell, into a running
// mean. It reads other cells' `outgoing` and writes only its own `incoming`;
// `cache_shade` then turns that into `outgoing`. Split like this no cell is
// read while it is written, so a bounce never sees a half-updated neighbour.
[shader("compute")]
[numthreads(8, 8, 1)]
void cache_gather(uint3 id : SV_DispatchThreadID) {
    uint slot = id.y * 512u + id.x;
    if (slot > f.cache.y || f.cacheKeys[slot].load() == 0u) return;
    Cell cell = f.cells[slot];
    if (f.cache.x - asuint(cell.pos.w) > 600u) {
        // Refinements first: whoever claims the slot must not read this as refined.
        f.cells[slot].normal.w = 0.0;
        f.cacheKeys[slot].store(0u);
        return;
    }
    if (!cacheDue(slot, cell)) return;
    float3 n = normalize(cell.normal.xyz);
    // A cell in use keeps its parent alive, though no pixel reads the parent.
    bool unused;
    int parent = cacheSlot(cell.pos.xyz, n, cacheLevel(cell.pos.xyz) + 1u, false, unused);
    if (parent >= 0 && asuint(f.cells[parent].pos.w) < asuint(cell.pos.w)) f.cells[parent].pos.w = cell.pos.w;
    float u = frac(float(hashU(slot) & 0xffffu) / 65536.0 + float(f.cache.x) * 0.7548776662);
    float phi = 6.2831853 * frac(float(hashU(slot ^ 0x5bd1e995u) & 0xffffu) / 65536.0 + float(f.cache.x) * 0.5698402910);
    float3 dir = cosineDir(n, u, phi);
    float3 origin = cell.pos.xyz + n * 0.03;
    float3 arriving = 0.0;
    float far = f.sky.w;
    float3 wn;
    int hit = walkWorld(origin, dir, far, wn, false);
    if (hit >= 0) {
        arriving = clampFirefly(worldLight(origin + dir * far, wn, hit, false, SEEN_DIFFUSE));
    } else if (escapes(origin, dir, f.sky.w)) {
        arriving = background(dir);
    }
    // Many point lights reach the cell through this mean, one sampled a visit.
    uint seed = hashU(slot ^ hashU(f.cache.x ^ 0x68bc21ebu));
    arriving += sampledLight(cell.pos.xyz, n, seed);
    // The last 32 samples; fewer where the light just changed.
    float limit = f.extra.y > 0.0 ? f.extra.y : (movedNear(cell.pos.xyz) ? 4.0 : 32.0);
    float samples = min(cell.normal.w + 1.0, limit);
    float3 incoming = lerp(cell.incoming.rgb * f.fade.rgb, arriving, 1.0 / samples);
    if (any(incoming != incoming)) incoming = 0.0;
    f.cells[slot].incoming = float4(incoming, 0.0);
    f.cells[slot].normal.w = samples;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cache_shade(uint3 id : SV_DispatchThreadID) {
    uint slot = id.y * 512u + id.x;
    if (slot > f.cache.y || f.cacheKeys[slot].load() == 0u) return;
    Cell cell = f.cells[slot];
    if (!cacheDue(slot, cell)) return;
    float3 n = normalize(cell.normal.xyz);
    f.cells[slot].outgoing = float4(cell.albedo.rgb * (directLight(cell.pos.xyz, n) + cell.incoming.rgb), 0.0);
}

// A few cosine-weighted rays per texel, marched against the depth buffer (a
// hit takes that point's light on last frame's image), continuing against the
// scene. Writes the hits' light / rays and the fraction that hit.
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
    int debug = int(f.misc.w);
    // The world-only debug views show the scene's light alone: no screen march.
    bool screen = f.misc.y > 0.0 && !(debug >= DEBUG_WORLD_ONLY && debug <= DEBUG_WORLD_POSITION);
    int steps = screen ? STEPS : 0;
    float3 light = 0.0;
    float hits = 0.0;
    float3 origin = p + n * (0.02 + 0.002 * linearDepth(depth));
    [loop] for (int r = 0; r < rays; r++) {
        float index = f.misc.x * float(rays) + float(r);
        float u = frac(noise.x + index * 0.7548776662);
        float phi = 6.2831853 * frac(noise.y + index * 0.5698402910);
        float3 dir = cosineDir(n, u, phi);
        float jitter = frac(noise.z + index * 0.6180339887);
        float4 seen = march(origin, dir, f.misc.y, steps, jitter);
        if (seen.w > 0.5) {
            light += clampFirefly(seen.rgb);
            hits += 1.0;
        } else if (f.counts.x > 0u) {
            // Off the screen, or past what it shows: the scene.
            float t = f.sky.w;
            float3 wn;
            int hit = walkWorld(origin, dir, t, wn, false);
            if (hit >= 0) {
                light += clampFirefly(worldLight(origin + dir * t, wn, hit, true, SEEN_DIFFUSE));
                hits += 1.0;
            } else if (escapes(origin, dir, f.sky.w)) {
                // The sky along this ray, as the cache and reference take it: the
                // composite's hemisphere at the normal is only right on open ground.
                light += background(dir);
                hits += 1.0;
            }
        }
    }
    float4 result = float4(light, hits) / float(rays);
    // One bad sample would otherwise live on in the accumulation forever.
    out0.Store(px, bad(result) ? float4(0.0) : result);
}

// One reflection ray per half-size texel, the view mirrored about a GGX
// microfacet normal (a mirror's rays stay together, a rough surface's spread),
// marched on screen, then against the scene. Surfaces rougher than the cutoff,
// and rays that find nothing, fall back on the hemisphere.
// out0: the light found, w 1 when found.
[shader("compute")]
[numthreads(8, 8, 1)]
void reflect_trace(uint3 id : SV_DispatchThreadID) {
    int2 px = int2(id.xy);
    int2 size = reflectSize();
    if (any(px >= size)) return;
    int2 fp = framePixel(px, size);
    float depth = loadDepth(fp);
    uint4 g = loadGbuffer(fp);
    float rough = gbufferRoughness(g);
    if (!receives(depth, g) || f.reflection.x < 0.5 || rough > f.reflection.y) {
        out0.Store(px, 0.0);
        return;
    }
    float3 n = gbufferNormal(g);
    float3 p = worldFromPixel(float2(fp) + 0.5, depth);
    float3 v = normalize(p - f.camera.xyz);
    float4 noise = blueNoise(px);
    float2 xi = frac(noise.xy + f.misc.x * float2(0.7548776662, 0.5698402910));
    float a2 = rough * rough * rough * rough;
    float cosT = sqrt((1.0 - xi.x) / (1.0 + (a2 - 1.0) * xi.x));
    float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
    float phi = 6.2831853 * xi.y;
    float3 h = aroundNormal(n, float3(sinT * cos(phi), sinT * sin(phi), cosT));
    float3 dir = reflect(v, h);
    if (dot(dir, n) <= 0.0) dir = reflect(v, n);
    float3 origin = p + n * (0.02 + 0.002 * linearDepth(depth));
    float jitter = frac(noise.z + f.misc.x * 0.6180339887);
    float4 result = march(origin, dir, REFLECT_REACH, 32, jitter);
    if (result.w < 0.5 && f.reflection.x > 1.5 && f.counts.x > 0u) {
        float far = 60.0;
        float3 wn;
        int hit = walkWorld(origin, dir, far, wn, false);
        if (hit >= 0) {
            uint seenAs = rough < 0.06 ? SEEN_MIRROR : SEEN_GLOSSY;
            result = float4(worldLight(origin + dir * far, wn, hit, true, seenAs), 1.0);
        }
    }
    out0.Store(px, bad(result) ? float4(0.0) : result);
}

// The reflection rays blended into what the same surface accumulated before:
// a mirror keeps a short history so its reflection follows the camera, a
// rough surface a longer one; each side weighted down by its brightness so a
// lone glint does not flare for frames.
// in0 this frame's rays, in1 previous light, in2 previous depth; out0 light, out1 depth.
[shader("compute")]
[numthreads(8, 8, 1)]
void reflect_temporal(uint3 id : SV_DispatchThreadID) {
    int2 px = int2(id.xy);
    int2 size = reflectSize();
    if (any(px >= size)) return;
    int2 fp = framePixel(px, size);
    float depth = loadDepth(fp);
    uint4 g = loadGbuffer(fp);
    if (!receives(depth, g)) {
        out0.Store(px, 0.0);
        out1.Store(px, 0.0);
        return;
    }
    float3 p = worldFromPixel(float2(fp) + 0.5, depth);
    float rough = gbufferRoughness(g);
    float4 now = in0.Load(int3(px, 0));
    float4 result = now;
    if (f.extra.w > 0.5) {
        float3 pp = project(f.prevClipFromWorld, p);
        if (onScreen(pp.xy)) {
            float2 at = pp.xy * float2(size) / frameSize();
            int2 tp = clamp(int2(at), 0, size - 1);
            float was = in2.Load(int3(tp, 0)).r;
            float4 wasLight = bilinear(in1, at, size);
            if (abs(was - pp.z) < 0.05 * pp.z && !bad(wasLight)) {
                wasLight.rgb *= f.fade.rgb;
                float take = lerp(0.45, 0.06, min(sqrt(rough * 2.5), 1.0));
                if (f.extra.y > 0.0 || movedNear(p)) take = max(take, 0.45);
                float wWas = (1.0 - take) / (1.0 + luminance(wasLight.rgb));
                float wNow = take / (1.0 + luminance(now.rgb));
                result = (wasLight * wWas + now * wNow) / (wWas + wNow);
            }
        }
    }
    out0.Store(px, result);
    out1.Store(px, float2(linearDepth(depth), 0.0));
}

// One pass of an edge-avoiding a-trous filter over the accumulated
// reflections, over a reach growing with roughness; a mirror's left alone.
// in0 light, in1 depth; out0 light.
void reflectFilter(int2 px, int step) {
    int2 size = reflectSize();
    if (any(px >= size)) return;
    float4 c = in0.Load(int3(px, 0));
    float z = in1.Load(int3(px, 0)).r;
    int2 fp = framePixel(px, size);
    uint4 g = loadGbuffer(fp);
    float rough = gbufferRoughness(g);
    float reach = 16.0 * rough;
    if (z <= 0.0 || reach < 0.3 || rough > f.reflection.y) {
        out0.Store(px, c);
        return;
    }
    float3 pos = worldFromPixel(float2(fp) + 0.5, f.camera.w / z);
    float3 n = gbufferNormal(g);
    float4 sum = 0.0;
    float weights = 0.0;
    [loop] for (int y = -2; y <= 2; y++) {
        [loop] for (int x = -2; x <= 2; x++) {
            int2 q = px + int2(x, y) * step;
            if (any(q < 0) || any(q >= size)) continue;
            float zq = in1.Load(int3(q, 0)).r;
            if (zq <= 0.0) continue;
            int2 fq = framePixel(q, size);
            uint4 gq = loadGbuffer(fq);
            float d2 = float(x * x + y * y) * float(step * step);
            float plane = abs(dot(n, worldFromPixel(float2(fq) + 0.5, f.camera.w / zq) - pos));
            float w = kernelTap(abs(x)) * kernelTap(abs(y));
            w *= exp(-d2 / (2.0 * reach * reach));
            w *= exp(-plane / (0.02 + 0.01 * z));
            // A normal map's bumps are one surface, a corner is not.
            w *= pow(max(dot(n, gbufferNormal(gq)), 0.0), 8.0);
            w *= exp(-abs(gbufferRoughness(gq) - rough) / 0.05);
            sum += in0.Load(int3(q, 0)) * w;
            weights += w;
        }
    }
    out0.Store(px, weights > 1e-6 ? sum / weights : c);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void reflect_filter1(uint3 id : SV_DispatchThreadID) { reflectFilter(int2(id.xy), 1); }

[shader("compute")]
[numthreads(8, 8, 1)]
void reflect_filter2(uint3 id : SV_DispatchThreadID) { reflectFilter(int2(id.xy), 2); }

// The half-size reflections on the same surface around this pixel, blurred
// more the rougher it is: rgb the light, a how much of it was found (faded out
// over the last quarter of the cutoff).
float4 resolveReflection(int2 fp, float3 n, float d, float rough) {
    if (f.reflection.x < 0.5 || rough > f.reflection.y) return 0.0;
    int2 size = reflectSize();
    float2 rc = (float2(fp) + 0.5) * float2(size) / frameSize() - 0.5;
    int2 rb = int2(floor(rc));
    float spread = 0.35 + 4.0 * sqrt(rough);
    float4 sum = 0.0;
    float weights = 0.0;
    [loop] for (int y = -1; y <= 2; y++) {
        [loop] for (int x = -1; x <= 2; x++) {
            int2 t = clamp(rb + int2(x, y), 0, size - 1);
            float tz = reflectDepth.Load(int3(t, 0)).r;
            if (tz <= 0.0) continue;
            float3 tn = gbufferNormal(loadGbuffer(framePixel(t, size)));
            float2 o = float2(t) - rc;
            float4 rt = reflectLight.Load(int3(t, 0));
            // A lone bright texel among dim ones is a glint, not the surface.
            float w = exp(-dot(o, o) / (2.0 * spread * spread)) * exp(-abs(tz - d) / (0.03 * d + 0.01))
                * pow(max(dot(tn, n), 0.0), 32.0) / (1.0 + luminance(rt.rgb));
            sum += rt * w;
            weights += w;
        }
    }
    if (weights <= 1e-4) return 0.0;
    float4 r = sum / weights;
    float fade = smoothstep(0.0, 1.0, (f.reflection.y - rough) / max(0.25 * f.reflection.y, 1e-3));
    return float4(r.a > 1e-3 ? r.rgb / r.a : float3(0.0), r.a * fade);
}

// What the debug views show in place of the lit image, w 1 when they do.
float4 debugView(int2 fp, float3 p, float4 gi, float4 reflection) {
    int debug = int(f.misc.w);
    // The pixel the reconstructed point projects back to: red where it lands
    // away from this one.
    float err = length(project(f.clipFromWorld, p).xy - (float2(fp) + 0.5));
    if (debug == DEBUG_ROUND_TRIP) return float4(saturate(err), saturate(1.0 - err), 0.0, 1.0);
    // One-metre bands: a floor at z = 0 shows a clean band edge at its surface.
    if (debug == DEBUG_WORLD_POSITION) return float4(frac(p), 1.0);
    if (debug == DEBUG_BOUNCE_LIGHT) return float4(gi.rgb * f.misc.z, 1.0);
    if (debug == DEBUG_WORLD_HITS) return float4(gi.aaa, 1.0);
    if (debug == DEBUG_REFLECTIONS) return float4(reflection.rgb * reflection.a, 1.0);
    return 0.0;
}

// Up to full size, each pixel taking the filtered texels around it on the same
// surface, added to the resolve's direct light. As Godot's surface shader mixes
// its GI buffer over its own ambient light by the fraction of rays that hit,
// the traced light stands where rays hit (rgb is premultiplied by that
// fraction) and the renderer's hemisphere where they found nothing. That image
// is what next frame's rays see, which is how light keeps bouncing.
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
    float d = linearDepth(depth);
    float rough = gbufferRoughness(g);
    float4 gi = upsampleGi(fp, n, d);
    float4 reflection = resolveReflection(fp, n, d, rough);
    float3 view = normalize(f.camera.xyz - p);
    float3 mirrored = reflect(-view, n);
    float open = 1.0 - saturate(gi.a);
    float3 diffuse = gi.rgb * f.misc.z + open * background(n);
    // Rays that found no reflection still only see the sky where it is open.
    float3 specular = lerp(open * background(mirrored), reflection.rgb, reflection.a);
    // The energy split the resolve gives its own ambient term.
    float nv = max(dot(n, view), 1e-4);
    float2 dfg = env_brdf(nv, rough);
    Brdf brdf = make_brdf(gbufferBase(g), gbufferMetallic(g), rough, gbufferSpecular(g), nv, dfg);
    float3 total = lit.rgb + diffuse * brdf.diffuse_color + specular * brdf.albedo;
    float4 result = float4(total, lit.a);
    // A debug view replaces what is shown, not what next frame's rays see.
    float4 shown = debugView(fp, p, gi, reflection);
    hdr[at] = shown.w > 0.5 ? float4(shown.rgb, lit.a) : result;
    colorOut.Store(at, float4(diffuseLit(lit.rgb, brdf, rough, n, view) + diffuse * brdf.diffuse_color, lit.a));
}

// Plain path tracing for `PxtgiDebug::Reference`: a mean over frames in the
// image the composite reads, through direct light at each bounce. The sum is
// kept in fp32: a running mean in that half-float image stops taking small
// steps after a few hundred frames, and drifted 13% dark over 1200.
static const int REFERENCE_PATHS = 16;
static const int REFERENCE_BOUNCES = 8;

[shader("compute")]
[numthreads(8, 8, 1)]
void reference(uint3 id : SV_DispatchThreadID) {
    int2 px = int2(id.xy);
    if (any(px >= traceSize())) return;
    int2 fp = framePixel(px, traceSize());
    float depth = loadDepth(fp);
    uint4 g = loadGbuffer(fp);
    uint at = uint(px.y) * uint(f.sizes.z) + uint(px.x);
    if (!receives(depth, g)) {
        // A pixel that receives later starts its mean from zero, not from
        // whatever the slot held (stale words read as huge floats stick).
        f.referenceSum[at] = 0.0;
        out0.Store(px, 0.0);
        return;
    }
    float3 start = worldFromPixel(float2(fp) + 0.5, depth);
    float3 startNormal = gbufferNormal(g);
    float3 light = 0.0;
    uint seed = hashU(uint(px.x) + uint(px.y) * 4099u + uint(f.reflection.z) * 16777619u);
    [loop] for (int path = 0; path < REFERENCE_PATHS; path++) {
        float3 x = start + startNormal * 0.02;
        float3 n = startNormal;
        float3 weight = 1.0;
        [loop] for (int bounce = 0; bounce < REFERENCE_BOUNCES; bounce++) {
            seed = hashU(seed);
            float u = float(seed >> 8u) / 16777216.0;
            seed = hashU(seed);
            float3 dir = cosineDir(n, u, 6.2831853 * float(seed >> 8u) / 16777216.0);
            float t = f.sky.w;
            float3 hitNormal;
            int hit = walkWorld(x, dir, t, hitNormal, false);
            if (hit < 0) {
                light += weight * background(dir);
                break;
            }
            float3 at = x + dir * t;
            n = dot(hitNormal, dir) > 0.0 ? -hitNormal : hitNormal;
            weight *= hitAlbedo(hit, at, n);
            light += weight * (directLight(at, n) + sampledLight(at, n, seed));
            x = at + n * 0.02;
        }
    }
    float4 now = float4(light / float(REFERENCE_PATHS), 1.0);
    if (bad(now)) now = float4(0.0, 0.0, 0.0, 1.0);
    float frames = f.reflection.z;
    float4 sum = (frames < 0.5 ? 0.0 : f.referenceSum[at]) + now;
    f.referenceSum[at] = sum;
    // w counts this pixel's own samples, which is fewer than `frames` for one
    // that started receiving after the restart.
    out0.Store(px, sum / max(sum.w, 1.0));
}
