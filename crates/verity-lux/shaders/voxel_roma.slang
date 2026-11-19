// Ray-aligned occupancy map array (Zeng et al., EGSR 2023, sections 4.1–4.3).
// Include after voxel.slang's cells, gridSize, levels, levelLow, levelCell and cellIndex.
// Set 0 binding 24: one storage buffer of uints, indexed [level][axis][z word][y][x].
// With G = gridSize() and W = ceil(G/32), allocate levels() * 16 * W * G * G
// words and dispatch romaBuild(ceil(G/8), ceil(G/8), levels() * 16 * W).
// Run after all occupancy writes are visible and make the resulting words visible
// to diffuse tracing. Build and trace use the same clip data and f.cache.x seed;
// regenerate when either changes. The build overwrites every word; no clear pass.

static const uint ROMA_AXES = 16u;
[[vk::binding(24, 0)]] RWStructuredBuffer<uint> romaMasks;

struct RomaTrace {
    bool supported;  // false: run the normal voxel walk with the original direction
    uint found;      // same voxel index + 1 as cells, or zero on miss
    float3 direction; // snapped, valid on both supported hits and misses
    int level;       // valid on hit
    int3 cell;       // absolute (not toroidally wrapped) grid cell, valid on hit
    float distance;  // approximate world-space entry distance, valid on hit
};

float2 romaJitter() {
    uint seed = hashU(f.cache.x);
    return float2(uint2(seed, hashU(seed ^ 0x9e3779b9u)) >> 8u) / 16777216.0;
}

// Equal-area latitude strata on a hemisphere, successive golden-angle azimuths.
// Each axis represents both signs; the whole set is resampled across frames.
float3 romaAxis(uint i, float2 jitter) {
    float z = (float(i) + jitter.x) / float(ROMA_AXES);
    float azimuth = 6.28318530718 * frac(float(i) * 0.61803398875 + jitter.y);
    float r = sqrt(max(0.0, 1.0 - z * z));
    return normalize(float3(r * cos(azimuth), r * sin(azimuth), z));
}

struct RomaFrame {
    float3 x, y, z;
    float3 width; // size of one rotated bin, in base cells
};

RomaFrame romaFrame(float3 axis) {
    RomaFrame b;
    float3 pole = abs(axis.z) < 0.9 ? float3(0, 0, 1) : float3(0, 1, 0);
    b.x = normalize(cross(pole, axis));
    b.y = cross(axis, b.x);
    b.z = axis;
    // Project the base cube onto each axis. Its AABB in that frame still has
    // G bins per axis, with widths equal to these L1 norms.
    b.width = float3(dot(abs(b.x), float3(1.0)), dot(abs(b.y), float3(1.0)),
                     dot(abs(b.z), float3(1.0)));
    return b;
}

uint romaWordCount(uint size) { return (size + 31u) >> 5u; }

uint romaWord(uint size, uint words, uint map, uint2 column, uint word) {
    return ((map * words + word) * size + column.y) * size + column.x;
}

float3 romaColumn(RomaFrame b, uint2 column, float halfSize) {
    float2 q = (float2(column) + 0.5 - halfSize) * b.width.xy;
    return halfSize + b.x * q.x + b.y * q.y;
}

float3 romaSample(RomaFrame b, float3 column, uint z, float halfSize) {
    return column + b.z * ((float(z) + 0.5 - halfSize) * b.width.z);
}

int3 romaSource(float3 samplePoint, int size) {
    // Clamp boundary bins to the last known voxel instead of treating a bin
    // whose footprint straddles the clip boundary as empty. Trace itself only
    // processes the part of the segment inside the unexpanded clip box.
    return int3(floor(clamp(samplePoint, 0.0, float(size - 1))));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void romaBuild(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID) {
    uint size = uint(gridSize());
    uint words = romaWordCount(size);
    if (size == 0u || any(id.xy >= size)) return;
    uint map = group.z / words;
    int level = int(map / ROMA_AXES);
    if (level >= levels()) return;
    uint word = group.z % words;
    RomaFrame b = romaFrame(romaAxis(map % ROMA_AXES, romaJitter()));
    float halfSize = float(size) * 0.5;
    float3 column = romaColumn(b, id.xy, halfSize);
    float3 margin = 0.5 * (abs(b.x) * b.width.x + abs(b.y) * b.width.y + abs(b.z) * b.width.z) + 1e-3;
    int3 low = levelLow(level);
    uint mask = 0u;
    uint first = word * 32u;
    uint count = min(32u, size - first);
    // Clip the word's sample indices once instead of testing source bounds
    // at all 32 samples, especially in the padded corners of the map.
    float3 p = romaSample(b, column, first, halfSize);
    float3 delta = b.z * b.width.z;
    float begin = 0.0, end = float(count - 1u);
    [unroll] for (int k = 0; k < 3; k++) {
        if (delta[k] == 0.0) {
            if (p[k] < -margin[k] || p[k] > float(size) + margin[k]) end = -1.0;
        } else {
            float a = (-margin[k] - p[k]) / delta[k];
            float z = (float(size) + margin[k] - p[k]) / delta[k];
            begin = max(begin, min(a, z));
            end = min(end, max(a, z));
        }
    }
    uint fromBit = uint(clamp(ceil(begin - 1e-3), 0.0, float(count)));
    uint endBit = uint(clamp(floor(end + 1e-3) + 1.0, 0.0, float(count)));
    [loop] for (uint bit = fromBit; bit < endBit; bit++) {
        int3 c = low + romaSource(romaSample(b, column, first + bit, halfSize), int(size));
        if (cells[cellIndex(level, c)] != 0u) mask |= 1u << bit;
    }
    romaMasks[romaWord(size, words, map, id.xy, word)] = mask;
}

// Clip against [0, size) in base cells. The returned distances are in world units.
float2 romaClip(float3 p, float3 d, float size, float cellSize, float reach) {
    float nearT = 0.0, farT = reach;
    [unroll] for (int k = 0; k < 3; k++) {
        if (d[k] == 0.0) {
            if (p[k] < 0.0 || p[k] >= size) return float2(0.0);
        } else {
            float a = (-p[k] / d[k]) * cellSize;
            float b = ((size - p[k]) / d[k]) * cellSize;
            nearT = max(nearT, min(a, b));
            farT = min(farT, max(a, b));
        }
    }
    return float2(nearT, farT);
}

uint romaLowBits(uint count) {
    if (count == 0u) return 0u;
    return 0xffffffffu >> (32u - count);
}

// Trace one contiguous clip-level span. All queried bins overlap [start, end).
// A word fetch tests up to 32 bins at once; only set bits need source lookups.
bool romaSpan(RomaFrame b, uint axis, int level, float3 p, float cellSize,
              float signZ, float start, float end, inout RomaTrace result) {
    uint size = uint(gridSize());
    float halfSize = float(size) * 0.5;
    float3 rel = p - halfSize;
    float z0 = dot(rel, b.z);
    // Evaluate the projection at the span start, closer to the current clip.
    float3 at = rel + b.z * (signZ * start / cellSize);
    float2 xy = float2(dot(at, b.x), dot(at, b.y)) / b.width.xy + halfSize;
    if (any(xy < -1e-3) || any(xy > float(size) + 1e-3)) return false;
    uint2 column = uint2(clamp(floor(xy), 0.0, float(size - 1u)));
    float za = (z0 + signZ * start / cellSize) / b.width.z + halfSize;
    float zb = (z0 + signZ * end / cellSize) / b.width.z + halfSize;
    float minZ = min(za, zb), maxZ = max(za, zb);
    if (minZ < -1e-3 || maxZ > float(size) + 1e-3 || minZ >= maxZ) return false;
    uint first = uint(clamp(floor(minZ), 0.0, float(size - 1u)));
    uint last = uint(clamp(ceil(maxZ) - 1.0, 0.0, float(size - 1u)));
    if (last < first) return false;

    uint words = romaWordCount(size), map = uint(level) * ROMA_AXES + axis;
    int step = signZ > 0.0 ? 1 : -1;
    int minWord = int(first >> 5u), maxWord = int(last >> 5u);
    float3 columnBase = romaColumn(b, column, halfSize);
    int3 low = levelLow(level);
    float voxelHalfZ = b.width.z * 0.5;
    [loop] for (int word = step > 0 ? minWord : maxWord;
                word >= minWord && word <= maxWord; word += step) {
        uint mask = romaMasks[romaWord(size, words, map, column, uint(word))];
        if (word == minWord) mask &= ~romaLowBits(first & 31u);
        if (word == maxWord) mask &= romaLowBits((last & 31u) + 1u);
        [loop] while (mask != 0u) {
            uint bit = uint(step > 0 ? firstbitlow(mask) : firstbithigh(mask));
            uint z = uint(word) * 32u + bit;
            mask &= ~(1u << bit);
            int3 local = romaSource(romaSample(b, columnBase, z, halfSize), int(size));
            // Even at a partial first/last bin, never report a hit if the
            // representative base voxel is beyond the finite ray segment.
            float centerZ = dot(float3(local) + 0.5 - halfSize, b.z);
            float sourceT = (signZ * (centerZ - z0) - voxelHalfZ) * cellSize;
            float sourceExit = sourceT + 2.0 * voxelHalfZ * cellSize;
            float binEdge = (float(z) + (step < 0 ? 1.0 : 0.0) - halfSize) * b.width.z;
            float binEntry = signZ * (binEdge - z0) * cellSize;
            float binExit = binEntry + b.width.z * cellSize;
            float hitT = max(start, max(sourceT, binEntry));
            if (hitT >= min(end, min(sourceExit, binExit))) continue;
            int3 c = low + local;
            uint found = cells[cellIndex(level, c)];
            if (found == 0u) return false;
            result.found = found;
            result.level = level;
            result.cell = c;
            result.distance = hitT;
            return true;
        }
    }
    return true;
}

// An uncovered original segment must not turn into a definite miss merely
// because snapping rotated it back into the clipmap. Most rays fit completely
// in the outer level; take the union of level intervals if the boxes aren't nested.
bool romaOriginalCovered(float3 origin, float3 direction, float reach) {
    float covered = 0.0;
    [loop] for (int i = 0; i < levels(); i++) {
        float next = covered;
        [loop] for (int level = levels() - 1; level >= 0; level--) {
            float cellSize = levelCell(level);
            float3 p = origin / cellSize - float3(levelLow(level));
            float2 span = romaClip(p, direction, float(gridSize()), cellSize, reach);
            if (span.x <= covered) next = max(next, span.y);
            if (next >= reach) return true;
        }
        if (next <= covered) return false;
        covered = next;
    }
    return false;
}

// Diffuse only: not for visibility to a fixed light or for tracing specular
// paths. The caller biases origin off its surface. `reach` is finite and in
// world units, independent of both the original direction length and snapping.
RomaTrace romaTraceDiffuse(float3 origin, float3 direction, float3 normal, float reach) {
    RomaTrace result;
    result.supported = false;
    result.found = 0u;
    result.direction = direction;
    result.level = -1;
    result.cell = int3(0);
    result.distance = 0.0;
    float d2 = dot(direction, direction);
    if (!(reach > 0.0) || !isfinite(reach) || !all(isfinite(origin)) || !all(isfinite(normal))
        || !(d2 > 0.0) || !isfinite(d2) || levels() <= 0 || gridSize() <= 0) return result;
    direction *= rsqrt(d2);
    float2 jitter = romaJitter();
    float score = 0.0, signZ = 1.0;
    uint axis = 0u;
    float3 best = float3(0.0);
    [loop] for (uint i = 0u; i < ROMA_AXES; i++) {
        float3 a = romaAxis(i, jitter);
        float cosine = dot(a, normal);
        if (cosine == 0.0) continue;
        float side = cosine > 0.0 ? 1.0 : -1.0;
        float alignment = side * dot(a, direction);
        if (alignment > score) {
            score = alignment;
            axis = i;
            signZ = side;
            best = a;
        }
    }
    if (score == 0.0 || dot(signZ * best, normal) <= 0.0) return result;
    RomaFrame b = romaFrame(best);
    float3 snapped = signZ * best;

    // Clip boxes can move independently. At each boundary choose the finest
    // available box, splitting early if the ray enters a finer one. Never skip
    // an uncovered span to reach a later box and report a hit or a miss there.
    float start = 0.0;
    [loop] while (start < reach) {
        float end = reach;
        int chosen = -1;
        float cellSize = 0.0;
        float3 p = float3(0.0);
        [loop] for (int level = 0; level < levels(); level++) {
            float scale = levelCell(level);
            float3 localOrigin = origin / scale - float3(levelLow(level));
            float2 span = romaClip(localOrigin, snapped, float(gridSize()), scale, reach);
            if (span.y <= max(start, span.x)) continue;
            if (span.x > start) {
                end = min(end, span.x);
            } else {
                end = min(end, span.y);
                chosen = level;
                cellSize = scale;
                p = localOrigin;
                break;
            }
        }
        if (chosen < 0 || end <= start || !romaSpan(b, axis, chosen, p, cellSize, signZ, start, end, result))
            return result;
        if (result.found != 0u) {
            result.supported = true;
            result.direction = snapped;
            return result;
        }
        start = end;
    }
    if (romaOriginalCovered(origin, direction, reach)) {
        result.supported = true;
        result.direction = snapped;
    }
    return result;
}
