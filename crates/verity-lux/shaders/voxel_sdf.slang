// Include after levelAt in voxel.slang. The surface is the union of finalized
// occupied cells, not the interior of the source mesh. Each level is a bounded
// cube: the occupancy storage wraps, but distances do not cross its seam.
//
// Set 0, binding 25: one STORAGE_BUFFER of levels() * gridSize()^3 uints (4
// bytes each). Addressing is level, local z, local y, local x, with x contiguous.
// No additional scratch or push constants. With no field, consumers pass
// ready=false and use their occupancy walker; bind a dummy uint if the
// consumer entry point statically uses this descriptor. Never dispatch a
// rebuild against that dummy.
//
// After initial voxelization and whenever occupancy or levelLow changes,
// dispatch voxelSdfX, voxelSdfY, voxelSdfZ in order. Each uses
// (ceil(gridSize()^2 / 64), levels(), 1) groups; id.y selects the level.
// Synchronize finalized cells before X, and compute writes to reads/writes
// between axes and before queries. X overwrites the entire field, so it needs
// no initialization. Only report ready for the completed matching occupancy
// and clip origins; keep the field and occupancy alive and in sync across frames.
[[vk::binding(25, 0)]] RWStructuredBuffer<uint> voxelSdfDistances;

static const uint VOXEL_SDF_UNREACHABLE = 0xffffu;

// Low half: distance to occupied. High half: distance to empty. Exactly one
// half is zero. Finite distances fit in 16 bits (at most 3 * (gridSize() - 1)).
uint2 voxelSdfUnpack(uint packed) { return uint2(packed & 0xffffu, packed >> 16u); }
uint voxelSdfPack(uint2 distances) { return distances.x | (distances.y << 16u); }

uint voxelSdfIndex(int level, int3 local, int size) {
    return uint(((level * size + local.z) * size + local.y) * size + local.x);
}

// In each axis, sweep both directions: min(source[i], previous + 1) gives
// min_j(source[j] + abs(i - j)). Composing these exact 1D transforms yields
// Manhattan distance to either class of cell centre. A thread owns the entire
// line in both sweeps, so the intermediate results can be updated in place.
[ForceInline]
void voxelSdfSweep(uint3 id, int axis) {
    int size = gridSize();
    if (id.y >= uint(levels()) || id.x >= uint(size * size)) return;
    int level = int(id.y);
    int u = int(id.x) % size, v = int(id.x) / size;
    int3 local = axis == 0 ? int3(0, u, v) : (axis == 1 ? int3(u, 0, v) : int3(u, v, 0));
    uint stride = uint(axis == 0 ? 1 : (axis == 1 ? size : size * size));
    uint at = voxelSdfIndex(level, local, size);
    int3 world = axis == 0 ? levelLow(level) + local : int3(0);
    uint2 nearest = VOXEL_SDF_UNREACHABLE;
    [loop] for (int i = 0; i < size; i++, at += stride) {
        uint2 source;
        if (axis == 0) {
            bool solid = cells[cellIndex(level, world)] != 0u;
            source = solid ? uint2(0u, VOXEL_SDF_UNREACHABLE) : uint2(VOXEL_SDF_UNREACHABLE, 0u);
            world.x++;
        } else {
            source = voxelSdfUnpack(voxelSdfDistances[at]);
        }
        nearest = min(source, nearest + 1u);
        voxelSdfDistances[at] = voxelSdfPack(nearest);
    }
    nearest = VOXEL_SDF_UNREACHABLE;
    [loop] for (int i = size; i > 0; i--) {
        at -= stride;
        uint packed = voxelSdfDistances[at];
        nearest = min(voxelSdfUnpack(packed), nearest + 1u);
        uint updated = voxelSdfPack(nearest);
        if (updated != packed) voxelSdfDistances[at] = updated;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void voxelSdfX(uint3 id : SV_DispatchThreadID) { voxelSdfSweep(id, 0); }

[shader("compute")]
[numthreads(64, 1, 1)]
void voxelSdfY(uint3 id : SV_DispatchThreadID) { voxelSdfSweep(id, 1); }

[shader("compute")]
[numthreads(64, 1, 1)]
void voxelSdfZ(uint3 id : SV_DispatchThreadID) { voxelSdfSweep(id, 2); }

// Sample in clip-local cell coordinates [0, gridSize()), returning a signed
// Euclidean distance bound in cell units (negative in solid, zero permitted
// on its boundary). Empty outside this level is included as the complement;
// this helper does not sample coarser levels. False means no valid sample.
bool voxelSdfSample(int level, float3 local, bool ready, out float signedDistance) {
    signedDistance = 0.0;
    int size = gridSize();
    if (!ready || level < 0 || level >= levels() || !all(local >= 0.0) || !all(local < float(size)))
        return false;
    int3 c = int3(floor(local));
    uint2 distances = voxelSdfUnpack(voxelSdfDistances[voxelSdfIndex(level, c, size)]);
    bool solid = distances.x == 0u;
    if (solid == (distances.y == 0u)) return false;
    float3 frac = local - float3(c);
    float3 faces = min(frac, 1.0 - frac);
    float3 fromCenter = abs(frac - 0.5);
    float opposite = float(min(solid ? distances.y : distances.x, uint(3 * size)));
    // The target cell contributes at most 1.5 Manhattan units from its centre.
    // Euclidean distance >= Manhattan / sqrt(3); 0.5 is a safe multiplier.
    float bound = 0.5 * (opposite - 1.5 - (fromCenter.x + fromCenter.y + fromCenter.z));
    float magnitude = max(bound, min(faces.x, min(faces.y, faces.z)));
    if (magnitude == 0.0) {
        // At a cell seam the distance to this cell's own face is zero. If all
        // cells touching the point agree, their union gives a positive bound.
        int3 touch = int3(frac == 0.0);
        int3 low = levelLow(level);
        bool sameOccupancy = true;
        for (int z = 0; z <= touch.z; z++)
        for (int y = 0; y <= touch.y; y++)
        for (int x = 0; x <= touch.x; x++) {
            int3 neighbor = c - int3(x, y, z);
            bool neighborSolid = false;
            if (all(neighbor >= 0)) neighborSolid = cells[cellIndex(level, neighbor + low)] != 0u;
            sameOccupancy = sameOccupancy && neighborSolid == solid;
        }
        if (sameOccupancy) {
            faces = select(frac == 0.0, float3(1.0), faces);
            magnitude = min(faces.x, min(faces.y, faces.z));
        }
    }
    if (!solid) {
        signedDistance = magnitude;
    } else {
        float3 edge = min(local, float(size) - local);
        signedDistance = -min(magnitude, min(edge.x, min(edge.y, edge.z)));
    }
    return true;
}

float3 voxelSdfNextFace(int3 c, float3 p, int3 step, bool3 moving, float3 inv) {
    float3 far = (float3(c) + float3(step > 0) - p) * inv;
    return select(moving, far, float3(asfloat(0x7f800000u)));
}

// Trace o + d*t without exceeding reach, with the occupancy walk's clipmap
// order and initial solid-cell suppression. Only nonempty clip intervals are
// visited. Returns true for a resolved result (found is a voxel index + 1, or
// 0 for clear); false means the caller must use the ordinary occupancy walk.
bool voxelSdfTrace(float3 o, float3 d, float reach, bool ready, out uint found) {
    found = 0u;
    if (!ready || !all(isfinite(o)) || !all(isfinite(d)) || !isfinite(reach)) return false;
    if (reach <= 0.0) return true;
    float speedSquared = dot(d, d);
    if (!(speedSquared > 1e-20) || !isfinite(speedSquared)) return false;
    float invSpeed = rsqrt(speedSquared);
    bool3 moving = d != 0.0;
    if (any(select(moving, abs(d) < 1e-20, bool3(false)))) return false;
    float3 inv = 1.0 / select(moving, d, float3(1.0));
    int3 step = select(d < 0.0, int3(-1), int3(1));
    int size = gridSize();
    if (size <= 0 || levels() <= 0) return false;
    int firstLevel = levelAt(o);
    float infinity = asfloat(0x7f800000u);
    float travelled = 0.0;
    bool clear = false;
    int crossings = 0;
    int samples = 0;
    [loop] for (int level = firstLevel; level < levels(); level++) {
        float cell = levelCell(level);
        int3 low = levelLow(level);
        float3 p = o / cell - float3(low);
        if (!all(isfinite(p))) return false;
        float3 parallelOrigin = select(moving, float3(0.0), p);
        if (any(parallelOrigin < 0.0) || any(parallelOrigin >= float(size))) continue;
        float3 a = -p * inv, b = (float(size) - p) * inv;
        float3 near = select(moving, min(a, b), float3(-infinity));
        float3 far = select(moving, max(a, b), float3(infinity));
        float limit = reach / cell;
        float t = max(max(near.x, max(near.y, near.z)), travelled / cell);
        float end = min(min(far.x, min(far.y, far.z)), limit);
        if (t >= end) continue;
        // Leave distant cancellation cases to the original walker. Within this
        // range the clearance margin and the restart face guard dominate roundoff.
        if (any(abs(p) > 4096.0)) return false;
        precise float3 start = p + d * t;
        int3 c = clamp(int3(floor(start)), 0, size - 1);
        float3 back = (float3(c) + float3(step < 0) - p) * inv;
        float3 next = voxelSdfNextFace(c, p, step, moving, inv);
        // Use boundary times to settle rounding at the entry point (negative
        // rays at an exact integer face enter the lower cell).
        c -= select(moving, select(back > t, step, int3(0)), int3(0));
        c += select(moving, select(next <= t, step, int3(0)), int3(0));
        c = clamp(c, 0, size - 1);
        next = voxelSdfNextFace(c, p, step, moving, inv);
        [loop] for (;;) {
            if (samples++ >= MAX_VOXEL_STEPS) return false;
            uint2 distances = voxelSdfUnpack(voxelSdfDistances[voxelSdfIndex(level, c, size)]);
            bool solid = distances.x == 0u;
            if (solid == (distances.y == 0u)) return false;
            if (solid) {
                uint token = cells[cellIndex(level, low + c)];
                if (token == 0u) return false;
                if (clear || crossings >= MAX_START_CELLS) {
                    found = token;
                    return true;
                }
            } else {
                clear = true;
                if (distances.x == VOXEL_SDF_UNREACHABLE) break;
                if (distances.x > 3u) {
                    // Allow up to 1.5 Manhattan units at each end for the two
                    // cells' half-extents. Half the rest is a safe sphere
                    // radius irrespective of where the ray is in this cell.
                    float target = t + 0.5 * float(distances.x - 3u) * invSpeed;
                    if (target > end) break;
                    precise float3 probe = p + d * target;
                    float3 base = floor(probe);
                    float3 frac = probe - base;
                    float3 edge = select(moving, min(frac, 1.0 - frac), float3(1.0));
                    if (target > t && target >= min(next.x, min(next.y, next.z)) &&
                        all(edge > 1.0 / 256.0) && all(base >= 0.0) && all(base < float(size))) {
                        c = int3(base);
                        t = target;
                        next = voxelSdfNextFace(c, p, step, moving, inv);
                        continue;
                    }
                }
            }
            // Close to the occupied union, step exactly one face. An epsilon
            // here could cross an occupied corner or advance beyond reach.
            int axis = next.x <= next.y && next.x <= next.z ? 0 : (next.y <= next.z ? 1 : 2);
            if (next[axis] > end) break;
            t = next[axis];
            c[axis] += step[axis];
            if (c[axis] < 0 || c[axis] >= size) break;
            crossings++;
            next[axis] = (float(c[axis]) + float(step[axis] > 0) - p[axis]) * inv[axis];
        }
        if (end == limit) return true;
        travelled = end * cell;
    }
    return true;
}
