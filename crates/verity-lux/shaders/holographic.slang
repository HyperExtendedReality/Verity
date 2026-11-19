// Holographic Radiance Cascades (Freeman, Sannikov and Margel 2025) across one
// clipmap level, in 3D by slicing; `holographic.rs` is the CPU reference and
// explains the method. A ray in a vertical plane meets only what is in that
// plane, so each page (a turn φ about z) cuts the level into vertical slices
// that run 2D HRC unchanged, and its cones count by |cos α|, α above the
// horizontal. Here:
//
// - `hrcMedium` makes the level's cells, two a side to each of ours, a dense medium;
// - each batch of a page's slices builds rays (`hrcBase`, `hrcExtend`), then
//   cones from the top cascade down (`hrcTop`, `hrcMerge`), from `hrcMeasures`'
//   weights;
// - `hrcPull` adds what each cell's slice gathered to `hrcSum`. Once every page
//   has, voxel.rs copies it to `hrcGather`, which `inject` reads.
//
// Included by voxel.slang.

// Must match voxel.rs.
static const int HRC_GRID = 64;
static const int HRC_REACH = 47; // ceil(HRC_GRID / sqrt 2) + 1
static const int HRC_SIDE = 2 * HRC_REACH + 1;
static const int HRC_TOP = 8; // the smallest n with 2^n >= HRC_SIDE, plus 1
static const int HRC_BATCH = 8;
// f32tof16 / f16tof32 compile to 16-bit types, which need device features the
// renderer does not enable. The GLSL pack ops do the same with 32-bit types only.
uint hrcPack2(float2 pair) {
    return spirv_asm { result:$$uint = OpExtInst glsl450 PackHalf2x16 $pair };
}
float2 hrcUnpack2(uint packed) {
    return spirv_asm { result:$$float2 = OpExtInst glsl450 UnpackHalf2x16 $packed };
}
// Cones of cascades 0 to HRC_TOP + 1, for `hrcMeasures`.
static const int HRC_CONES = (2 << (HRC_TOP + 1)) - 1;
// Extinction of a solid cell, per cell: opaque within half a cell.
static const float HRC_SOLID = 16.0;

struct HrcPush {
    int level;      // the cascade a per-cascade pass works on
    int firstSlice; // the batch's first slice
    int clipLevel;  // the clipmap level covered, or -1 when off
    uint ready;     // 1 once `hrcGather` holds every page
    float2 horizontal; // the page's way across its slices, in x and y
    float weight;      // the page's share of the half turn
    uint firstPage;    // The first page overwrites the sum; later pages add to it.
};
[[vk::push_constant]] ConstantBuffer<HrcPush> hrc;

// Per cell: the light leaving the side its normal faces (w the normal as
// `packNormal`), then the other side's (w 1 when solid, 0 when empty).
[[vk::binding(18, 0)]] RWStructuredBuffer<float4> hrcMedium;
// Per batch slice, frustum, cascade, probe and end row: a ray's light and
// transmittance as four halves.
[[vk::binding(19, 0)]] RWStructuredBuffer<uint2> hrcRays;
// Two cascades at a time (by parity), per batch slice, frustum, probe and
// cone: fluence and flux as twelve halves, in three.
[[vk::binding(20, 0)]] RWStructuredBuffer<uint2> hrcCones;
// Per frustum and cone: its weight for fluence, then for flux in x, y and z.
[[vk::binding(21, 0)]] RWStructuredBuffer<float4> hrcMeasures;
// Per cell, toroidally: fluence rgb then flux x, y and z rgb, as three float4.
[[vk::binding(22, 0)]] RWStructuredBuffer<float4> hrcSum;
[[vk::binding(23, 0)]] RWStructuredBuffer<float4> hrcGather;

int hrcColumns(int n) { return (HRC_SIDE + (1 << n) - 1) >> n; }
int hrcEnds(int n) { return (2 << n) + 1; }
// Where cascade n's rays start within a slice and frustum's.
int hrcRayStart(int n) {
    int at = 0;
    for (int k = 0; k < n; k++) at += hrcColumns(k) * HRC_SIDE * hrcEnds(k);
    return at;
}
int hrcRaysEach() { return hrcRayStart(HRC_TOP + 1); }
int hrcConesEach() {
    int most = 0;
    for (int n = 0; n <= HRC_TOP; n++) most = max(most, hrcColumns(n) * HRC_SIDE * (1 << n));
    return most;
}

float hrcCell() { return 2.0 * levelCell(hrc.clipLevel); }
// The box's low corner, in our cells.
int3 hrcLow() { return levelLow(hrc.clipLevel) / 2; }

struct HrcRay {
    float3 light;
    float through;
};

HrcRay hrcVacuum() {
    HrcRay r;
    r.light = 0.0;
    r.through = 1.0;
    return r;
}

HrcRay hrcThen(HrcRay near, HrcRay far) {
    HrcRay r;
    r.light = near.light + near.through * far.light;
    r.through = near.through * far.through;
    return r;
}

uint2 packRay(HrcRay r) {
    return uint2(hrcPack2(r.light.xy), hrcPack2(float2(r.light.z, r.through)));
}

HrcRay unpackRay(uint2 p) {
    HrcRay r;
    float2 rg = hrcUnpack2(p.x), bt = hrcUnpack2(p.y);
    r.light = float3(rg, bt.x);
    r.through = bt.y;
    return r;
}

struct HrcCone {
    float3 m[4]; // fluence, flux x, y, z
};

HrcCone hrcNone() {
    HrcCone c;
    [unroll] for (int i = 0; i < 4; i++) c.m[i] = 0.0;
    return c;
}

// The cone a ray borders: the ray's own light over `measure`, then `far`.
HrcCone hrcMerge(HrcRay r, float4 measure, HrcCone far) {
    HrcCone c;
    [unroll] for (int i = 0; i < 4; i++) c.m[i] = r.light * measure[i] + far.m[i] * r.through;
    return c;
}

// Slice `b` of the batch and frustum `f`: whose rays and cones, at index i.
uint hrcRayIndex(int b, int f, int n, int i, int y, int m) {
    return uint((b * 4 + f) * hrcRaysEach() + hrcRayStart(n) + (i * HRC_SIDE + y) * hrcEnds(n) + m + (1 << n));
}

uint hrcConeIndex(int b, int f, int n, int i, int y, int c) {
    return uint((((n & 1) * HRC_BATCH + b) * 4 + f) * hrcConesEach() + (i * HRC_SIDE + y) * (1 << n) + c) * 3u;
}

// Cascade n's ray from probe (x, y), m rows across, or vacuum where there is none.
HrcRay hrcRay(int b, int f, int n, int x, int y, int m) {
    if (x < 0 || y < 0 || y >= HRC_SIDE || (x >> n) >= hrcColumns(n)) return hrcVacuum();
    return unpackRay(hrcRays[hrcRayIndex(b, f, n, x >> n, y, m)]);
}

HrcCone hrcCone(int b, int f, int n, int x, int y, int c) {
    HrcCone cone = hrcNone();
    if (x < 0 || y < 0 || y >= HRC_SIDE || (x >> n) >= hrcColumns(n)) return cone;
    uint at = hrcConeIndex(b, f, n, x >> n, y, c);
    uint2 p0 = hrcCones[at], p1 = hrcCones[at + 1u], p2 = hrcCones[at + 2u];
    float2 h0 = hrcUnpack2(p0.x), h1 = hrcUnpack2(p0.y);
    float2 h2 = hrcUnpack2(p1.x), h3 = hrcUnpack2(p1.y);
    float2 h4 = hrcUnpack2(p2.x), h5 = hrcUnpack2(p2.y);
    cone.m[0] = float3(h0, h1.x);
    cone.m[1] = float3(h1.y, h2);
    cone.m[2] = float3(h3, h4.x);
    cone.m[3] = float3(h4.y, h5);
    return cone;
}

void hrcStoreCone(uint at, HrcCone c) {
    hrcCones[at] = uint2(hrcPack2(c.m[0].xy), hrcPack2(float2(c.m[0].z, c.m[1].x)));
    hrcCones[at + 1u] = uint2(hrcPack2(c.m[1].yz), hrcPack2(c.m[2].xy));
    hrcCones[at + 2u] = uint2(hrcPack2(float2(c.m[2].z, c.m[3].x)), hrcPack2(c.m[3].yz));
}

// Frustum f's grid in a slice: f 0 and 1 run along it, 2 and 3 up it; 1 and 3 backwards.
int2 hrcToSlice(int f, int2 uv) {
    if (f == 0) return uv;
    if (f == 1) return int2(HRC_SIDE - 1 - uv.x, uv.y);
    if (f == 2) return int2(uv.y, uv.x);
    return int2(uv.y, HRC_SIDE - 1 - uv.x);
}

int2 hrcFromSlice(int f, int2 ty) {
    if (f == 0) return ty;
    if (f == 1) return int2(HRC_SIDE - 1 - ty.x, ty.y);
    if (f == 2) return int2(ty.y, ty.x);
    return int2(HRC_SIDE - 1 - ty.y, ty.x);
}

// A step of frustum f's grid as a step along and up the slice.
float2 hrcStep(int f, float2 uv) {
    float u = (f & 1) != 0 ? -uv.x : uv.x;
    return f < 2 ? float2(u, uv.y) : float2(uv.y, u);
}

float3 hrcWorldWay(float2 along_up) {
    return float3(hrc.horizontal * along_up.x, along_up.y);
}

// Slice s's cell (t, row), as a cell of the box: the slice grid turns about its middle column.
int3 hrcBoxCell(int s, int2 ty) {
    float2 across = float2(-hrc.horizontal.y, hrc.horizontal.x);
    float2 xy = float2(HRC_GRID / 2) + hrc.horizontal * float(ty.x - HRC_REACH) + across * float(s - HRC_REACH);
    return int3(int2(round(xy)), ty.y);
}

// What a ray along d meets at box cell c: its light and extinction. Past the
// box, the background.
void hrcMediumAt(int3 c, float3 d, out float3 light, out float sigma) {
    if (any(c < 0) || any(c >= HRC_GRID)) {
        light = background(d);
        sigma = HRC_SOLID;
        return;
    }
    uint i = uint((c.z * HRC_GRID + c.y) * HRC_GRID + c.x) * 2u;
    float4 back = hrcMedium[i + 1u];
    if (back.w == 0.0) {
        light = 0.0;
        sigma = 0.0;
        return;
    }
    float4 front = hrcMedium[i];
    // As `voxelSeen`: a ray heading the way the normal points meets the back.
    light = dot(voxelNormalBytes(asuint(front.w)), d) > 0.0 ? back.rgb : front.rgb;
    sigma = HRC_SOLID;
}

HrcRay hrcThrough(float3 light, float sigma, float length) {
    HrcRay r;
    r.through = exp(-sigma * length);
    r.light = light * (1.0 - r.through);
    return r;
}

// Splits a flat thread index over (batch slice, frustum, probe column, row, slot).
bool hrcThread(uint id, int columns, int slots, out int b, out int f, out int i, out int y, out int k) {
    int per = columns * HRC_SIDE * slots;
    int at = int(id);
    k = at % slots;
    y = (at / slots) % HRC_SIDE;
    i = (at / (slots * HRC_SIDE)) % columns;
    f = (at / per) % 4;
    b = at / (per * 4);
    return b < HRC_BATCH && hrc.firstSlice + b < HRC_SIDE;
}

// ---------------------------------------------------------------- passes

// Each cell of the box from its 2x2x2 of the covered level: solid when any is,
// with their mean normal and the mean light leaving each way.
[shader("compute")]
[numthreads(64, 1, 1)]
void hrcMediumBuild(uint3 id : SV_DispatchThreadID) {
    if (id.x >= uint(HRC_GRID * HRC_GRID * HRC_GRID)) return;
    int3 c = int3(id.x % HRC_GRID, (id.x / HRC_GRID) % HRC_GRID, id.x / (HRC_GRID * HRC_GRID));
    int3 first = levelLow(hrc.clipLevel) + c * 2;
    uint found[8];
    float3 sum = 0.0;
    int count = 0;
    [unroll] for (int k = 0; k < 8; k++) {
        found[k] = cells[cellIndex(hrc.clipLevel, first + int3(k & 1, (k >> 1) & 1, k >> 2))];
        if (found[k] != 0u) {
            sum += voxelNormal(voxels[found[k] - 1u].z);
            count++;
        }
    }
    if (count == 0) {
        hrcMedium[id.x * 2u + 1u] = 0.0;
        return;
    }
    float3 n = 0.0;
    float3 front = 0.0, back = 0.0;
    [unroll] for (int k = 0; k < 8; k++) {
        if (found[k] == 0u) continue;
        uint index = found[k] - 1u;
        float3 vn = voxelNormal(voxels[index].z);
        // Normals that cancel: the first voxel's way.
        if (all(n == 0.0)) n = length(sum) > 0.1 ? normalize(sum) : vn;
        uint side = dot(vn, n) >= 0.0 ? 0u : 1u;
        front += radiance[index * 6u + side * 3u].rgb;
        back += radiance[index * 6u + (1u - side) * 3u].rgb;
    }
    hrcMedium[id.x * 2u] = float4(front / float(count), asfloat(packNormal(n)));
    hrcMedium[id.x * 2u + 1u] = float4(back / float(count), 1.0);
}

// Integrate a cone arc whose endpoints stay on the same side of the vertical.
// The endpoints are unit vectors; the frustum spans at most pi/2.
float3 hrcArcMeasure(float2 a, float2 b) {
    float det = a.x * b.y - a.y * b.x;
    float angle = atan2(det, dot(a, b));
    float orientation = (a.x + b.x >= 0.0 ? 1.0 : -1.0) * sign(angle);
    return orientation * float3(b.y - a.y,
                                0.5 * (angle + b.x * b.y - a.x * a.y),
                                0.5 * (b.y * b.y - a.y * a.y));
}

// Exact fluence and flux measure: integrate |cos alpha| times (1, cos alpha, sin alpha).
[shader("compute")]
[numthreads(64, 1, 1)]
void hrcMeasure(uint3 id : SV_DispatchThreadID) {
    if (id.x >= uint(4 * HRC_CONES)) return;
    int f = int(id.x) / HRC_CONES;
    int k = int(id.x) % HRC_CONES + 1;
    int n = firstbithigh(uint(k));
    int c = k - (1 << n);
    float s = float(1 << n);
    float low = 2.0 * float(c) - s;
    float2 a = normalize(hrcStep(f, float2(s, low)));
    float2 b = normalize(hrcStep(f, float2(s, low + 2.0)));
    float3 m;
    if (a.x * b.x < 0.0) {
        // Only cones crossing the vertical need two integrals.
        float2 pole = float2(0.0, sign(a.y));
        m = hrcArcMeasure(a, pole) + hrcArcMeasure(pole, b);
    } else {
        m = hrcArcMeasure(a, b);
    }
    hrcMeasures[id.x] = float4(m.x, hrcWorldWay(m.yz));
}

float4 hrcMeasureOf(int f, int n, int c) { return hrcMeasures[f * HRC_CONES + (1 << n) - 1 + c]; }

// Cascade 0's rays: half of each end cell.
[shader("compute")]
[numthreads(64, 1, 1)]
void hrcBase(uint3 id : SV_DispatchThreadID) {
    int b, f, x, y, k;
    if (!hrcThread(id.x, hrcColumns(0), 3, b, f, x, y, k)) return;
    int m = k - 1;
    int s = hrc.firstSlice + b;
    float3 d = normalize(hrcWorldWay(hrcStep(f, float2(1.0, float(m)))));
    float halfway = sqrt(1.0 + float(m * m)) * 0.5;
    float3 light;
    float sigma;
    hrcMediumAt(hrcBoxCell(s, hrcToSlice(f, int2(x, y))), d, light, sigma);
    HrcRay near = hrcThrough(light, sigma, halfway);
    hrcMediumAt(hrcBoxCell(s, hrcToSlice(f, int2(x + 1, y + m))), d, light, sigma);
    hrcRays[hrcRayIndex(b, f, 0, x, y, m)] = packRay(hrcThen(near, hrcThrough(light, sigma, halfway)));
}

// Cascade `hrc.level`'s rays, each two of the cascade below joined.
[shader("compute")]
[numthreads(64, 1, 1)]
void hrcExtend(uint3 id : SV_DispatchThreadID) {
    int n = hrc.level;
    int b, f, i, y, k;
    if (!hrcThread(id.x, hrcColumns(n), hrcEnds(n), b, f, i, y, k)) return;
    int step = 1 << (n - 1);
    int m = k - (1 << n);
    int x = i << n;
    HrcRay r;
    if ((m & 1) == 0) {
        r = hrcThen(hrcRay(b, f, n - 1, x, y, m / 2), hrcRay(b, f, n - 1, x + step, y + m / 2, m / 2));
    } else {
        // No probe on the midpoint: the mean of the two paths beside it.
        int lo = (m - 1) / 2, hi = lo + 1;
        HrcRay one = hrcThen(hrcRay(b, f, n - 1, x, y, lo), hrcRay(b, f, n - 1, x + step, y + lo, hi));
        HrcRay two = hrcThen(hrcRay(b, f, n - 1, x, y, hi), hrcRay(b, f, n - 1, x + step, y + hi, lo));
        r.light = (one.light + two.light) * 0.5;
        r.through = (one.through + two.through) * 0.5;
    }
    hrcRays[hrcRayIndex(b, f, n, i, y, m)] = packRay(r);
}

// The top cascade's cones, whose rays leave the grid: their edges' light.
[shader("compute")]
[numthreads(64, 1, 1)]
void hrcTop(uint3 id : SV_DispatchThreadID) {
    int n = HRC_TOP;
    int b, f, i, y, c;
    if (!hrcThread(id.x, hrcColumns(n), 1 << n, b, f, i, y, c)) return;
    int x = i << n;
    int low = 2 * c - (1 << n);
    HrcCone none = hrcNone();
    HrcCone upper = hrcMerge(hrcRay(b, f, n, x, y, low + 2), hrcMeasureOf(f, n + 1, 2 * c + 1), none);
    HrcCone lower = hrcMerge(hrcRay(b, f, n, x, y, low), hrcMeasureOf(f, n + 1, 2 * c), none);
    [unroll] for (int k = 0; k < 4; k++) upper.m[k] += lower.m[k];
    hrcStoreCone(hrcConeIndex(b, f, n, i, y, c), upper);
}

// Cascade `hrc.level`'s cones from the one above, as the paper's eqs. 14-17.
[shader("compute")]
[numthreads(64, 1, 1)]
void hrcDown(uint3 id : SV_DispatchThreadID) {
    int n = hrc.level;
    int b, f, i, y, c;
    if (!hrcThread(id.x, hrcColumns(n), 1 << n, b, f, i, y, c)) return;
    int step = 1 << n;
    int x = i << n;
    int low = 2 * c - step;
    HrcCone cone = hrcNone();
    [unroll] for (int h = 0; h < 2; h++) {
        int child = 2 * c + 1 - h;
        int edge = h == 0 ? low + 2 : low;
        float4 w = hrcMeasureOf(f, n + 1, child);
        HrcCone part;
        if ((i & 1) == 1) {
            // Along the cone's edge to the next cascade's column.
            part = hrcMerge(hrcRay(b, f, n, x, y, edge), w, hrcCone(b, f, n + 1, x + step, y + edge, child));
        } else {
            HrcCone along = hrcMerge(hrcRay(b, f, n + 1, x, y, 2 * edge), w,
                                     hrcCone(b, f, n + 1, x + 2 * step, y + 2 * edge, child));
            HrcCone here = hrcCone(b, f, n + 1, x, y, child);
            [unroll] for (int k = 0; k < 4; k++) part.m[k] = (here.m[k] + along.m[k]) * 0.5;
        }
        [unroll] for (int k = 0; k < 4; k++) cone.m[k] += part.m[k];
    }
    hrcStoreCone(hrcConeIndex(b, f, n, i, y, c), cone);
}

uint hrcSumIndex(int3 boxCell) {
    int3 t = (hrcLow() + boxCell) & (HRC_GRID - 1);
    return uint((t.z * HRC_GRID + t.y) * HRC_GRID + t.x) * 3u;
}

// Each cell whose nearest slice is in the batch adds the four frusta's base cones.
[shader("compute")]
[numthreads(64, 1, 1)]
void hrcPull(uint3 id : SV_DispatchThreadID) {
    if (id.x >= uint(HRC_GRID * HRC_GRID * HRC_GRID)) return;
    int3 c = int3(id.x % HRC_GRID, (id.x / HRC_GRID) % HRC_GRID, id.x / (HRC_GRID * HRC_GRID));
    float2 offset = float2(c.xy) - float2(HRC_GRID / 2);
    float2 across = float2(-hrc.horizontal.y, hrc.horizontal.x);
    int t = int(round(dot(offset, hrc.horizontal))) + HRC_REACH;
    int s = int(round(dot(offset, across))) + HRC_REACH;
    int b = s - hrc.firstSlice;
    if (b < 0 || b >= HRC_BATCH) return;
    HrcCone sum = hrcNone();
    [unroll] for (int f = 0; f < 4; f++) {
        int2 uv = hrcFromSlice(f, int2(t, c.z));
        HrcCone base = hrcCone(b, f, 0, uv.x, uv.y, 0);
        [unroll] for (int k = 0; k < 4; k++) sum.m[k] += base.m[k];
    }
    uint at = hrcSumIndex(c);
    float4 a = float4(sum.m[0], sum.m[1].x) * hrc.weight;
    float4 v = float4(sum.m[1].yz, sum.m[2].xy) * hrc.weight;
    float4 z = float4(sum.m[2].z, sum.m[3]) * hrc.weight;
    if (hrc.firstPage == 0u) {
        a += hrcSum[at];
        v += hrcSum[at + 1u];
        z += hrcSum[at + 2u];
    }
    hrcSum[at] = a;
    hrcSum[at + 1u] = v;
    hrcSum[at + 2u] = z;
}

// The bounce light arriving at a voxel side at x facing n, as irradiance / pi,
// and its luminance's mean way as `inject` keeps it; false outside the box, or
// where no empty cell lies just off the side, or before a full sweep.
bool hrcArriving(float3 x, float3 n, out float3 arriving, out float3 way) {
    arriving = 0.0;
    way = 0.0;
    if (hrc.clipLevel < 0 || hrc.ready == 0u) return false;
    float cell = hrcCell();
    [loop] for (int k = 1; k <= 2; k++) {
        int3 c = int3(floor((x + n * (cell * float(k))) / cell)) - hrcLow();
        if (any(c < 0) || any(c >= HRC_GRID)) return false;
        if (hrcMedium[uint((c.z * HRC_GRID + c.y) * HRC_GRID + c.x) * 2u + 1u].w != 0.0) continue;
        uint at = hrcSumIndex(c);
        float4 a = hrcGather[at], b = hrcGather[at + 1u], g = hrcGather[at + 2u];
        float3 fluence = a.rgb;
        float3 flux[3] = { float3(a.w, b.xy), float3(b.zw, g.x), g.yzw };
        float3 toward = flux[0] * n.x + flux[1] * n.y + flux[2] * n.z;
        // Irradiance of light linear in direction: fluence / 4 + flux.n / 2.
        arriving = max((fluence / 4.0 + toward / 2.0) / PI, 0.0);
        // Its luminance's cosine-weighted mean way: from the same linear light,
        // fluence n / 6pi + 3/16pi (D + n (D.n)).
        float3 d = float3(luminance(flux[0]), luminance(flux[1]), luminance(flux[2]));
        way = luminance(fluence) * n / (6.0 * PI) + 3.0 / (16.0 * PI) * (d + n * dot(d, n));
        return true;
    }
    return false;
}
