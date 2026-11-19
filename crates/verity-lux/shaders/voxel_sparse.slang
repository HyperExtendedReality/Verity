// Include after levelAt in voxel.slang. Build from finalized `cells`, never the
// optional dense brick counts. Payloads match voxel/sparse.rs; the clip headers
// are GPU publication metadata and are not part of the packed CPU payloads.
//
// Binding 26: 8 words per MAX_LEVELS header, then equal payload slices for the
// active levels. Hash slices have 2 + 3*S words for power-of-two S slots. DAG
// slices fit every possible distinct record with 1 + 2*N + 8*(N/8 + ... + 1)
// words, where N = max(gridSize()/4, 1)^3. A smaller slice may fail to publish.
// Binding 27: DAG-only scratch, 12 + 6*N words, reused between clip rebuilds.
// Zero the headers before first use. Dispatch the full build for each dirty clip
// level without an intervening sparse reader. Use compute storage barriers
// between passes (including from dense finalize and to the first reader).

static const uint VOXEL_SPARSE_HASH = 1u;
static const uint VOXEL_SPARSE_DAG = 2u;
static const uint VOXEL_SPARSE_BUILDING = 4u;
static const uint VOXEL_SPARSE_MAX_PROBES = 128u;
static const uint VOXEL_SPARSE_HEADER = 8u;
static const uint VOXEL_SPARSE_DAG_STAGES = 6u; // up to log2(128/4) + 1
static const uint VOXEL_SPARSE_DAG_COUNTERS = 2u * VOXEL_SPARSE_DAG_STAGES;

// Per-clip header: ready layout (or layout | BUILDING), low x/y/z, failed,
// DAG arena cursor, hash clear group count, hash build group count.
[[vk::binding(26, 0)]] RWStructuredBuffer<uint> voxelSparseWords;
[[vk::binding(27, 0)]] RWStructuredBuffer<uint> voxelSparseScratch;

struct VoxelSparsePush {
    uint layout;
    uint clipLevel;
    uint depth; // 0 = 4-cell leaf, then one stage for each parent extent
    uint reserved;
};
[[vk::push_constant]] ConstantBuffer<VoxelSparsePush> voxelSparsePush;

struct VoxelSparseRegion {
    uint header;
    uint base;
    uint capacity;
    uint brickCount;
    uint lastDepth;
};

bool voxelSparseRegion(int level, out VoxelSparseRegion region) {
    region = (VoxelSparseRegion)0;
    int size = gridSize();
    int clips = levels();
    if (level < 0 || level >= clips || clips > int(MAX_LEVELS) ||
        size < 1 || size > 128 || (size & (size - 1)) != 0) return false;
    uint words, stride;
    voxelSparseWords.GetDimensions(words, stride);
    uint headers = MAX_LEVELS * VOXEL_SPARSE_HEADER;
    if (words < headers) return false;
    uint side = max(uint(size) >> 2u, 1u);
    region.header = uint(level) * VOXEL_SPARSE_HEADER;
    region.capacity = (words - headers) / uint(clips);
    region.base = headers + uint(level) * region.capacity;
    region.brickCount = side * side * side;
    region.lastDepth = uint(firstbithigh(side));
    return true;
}

bool voxelSparseBuilding(uint layout, out VoxelSparseRegion region) {
    if (!voxelSparseRegion(int(voxelSparsePush.clipLevel), region)) return false;
    return voxelSparsePush.layout == layout &&
           voxelSparseWords[region.header] == (layout | VOXEL_SPARSE_BUILDING);
}

uint voxelSparseGroups(uint work) { return (work + 63u) >> 6u; }

uint voxelSparseHashSlots(VoxelSparseRegion region) {
    if (region.capacity < 5u || (region.capacity - 2u) % 3u != 0u) return 0u;
    uint count = (region.capacity - 2u) / 3u;
    return (count & (count - 1u)) == 0u ? count : 0u;
}

uint voxelSparseSpread(uint x) {
    x = (x | (x << 16u)) & 0x030000ffu;
    x = (x | (x << 8u)) & 0x0300f00fu;
    x = (x | (x << 4u)) & 0x030c30c3u;
    return (x | (x << 2u)) & 0x09249249u;
}

uint voxelSparseMorton(uint3 brick) {
    return voxelSparseSpread(brick.x) | (voxelSparseSpread(brick.y) << 1u) |
           (voxelSparseSpread(brick.z) << 2u);
}

uint3 voxelSparseBrick(uint morton) {
    uint3 xyz = (uint3(morton) >> uint3(0u, 1u, 2u)) & 0x09249249u;
    xyz = (xyz ^ (xyz >> 2u)) & 0x030c30c3u;
    xyz = (xyz ^ (xyz >> 4u)) & 0x0300f00fu;
    xyz = (xyz ^ (xyz >> 8u)) & 0x030000ffu;
    return (xyz ^ (xyz >> 16u)) & 0x000003ffu;
}

uint2 voxelSparseGather(uint morton) {
    int level = int(voxelSparsePush.clipLevel);
    uint size = uint(gridSize());
    uint ring = size - 1u;
    uint side = min(size, 4u);
    uint levelBase = uint(level) * size * size * size;
    uint3 low = asuint(levelLow(level)) + (voxelSparseBrick(morton) << 2u);
    uint2 mask = uint2(0u);
    for (uint z = 0u; z < side; z++) {
        for (uint y = 0u; y < side; y++) {
            uint row = levelBase + (((low.z + z) & ring) * size + ((low.y + y) & ring)) * size;
            uint bits = 0u;
            for (uint x = 0u; x < side; x++)
                bits |= uint(cells[row + ((low.x + x) & ring)] != 0u) << x;
            mask[z >> 1u] |= bits << (4u * y + 16u * (z & 1u));
        }
    }
    return mask;
}

// Hash: reset (1 group), clear (ceil(S/64)), build (ceil(N/64)), publish (1).
// DAG: reset (1), for each depth d=0..lastDepth prepare (ceil(2*(N>>3d)/64))
// and intern (ceil((N>>3d)/64)), then publish (1). All dispatches have Y=Z=1.
[shader("compute")]
[numthreads(1, 1, 1)]
void voxelSparseReset() {
    VoxelSparseRegion region;
    if (!voxelSparseRegion(int(voxelSparsePush.clipLevel), region)) return;
    uint header = region.header;
    voxelSparseWords[header] = 0u;
    uint layout = voxelSparsePush.layout;
    if (layout == VOXEL_SPARSE_HASH) {
        uint slots = voxelSparseHashSlots(region);
        if (slots == 0u) return;
        voxelSparseWords[region.base] = slots - 1u;
        voxelSparseWords[region.base + 1u] = 0u;
    } else if (layout == VOXEL_SPARSE_DAG) {
        uint words, stride;
        voxelSparseScratch.GetDimensions(words, stride);
        if (region.capacity == 0u || words < VOXEL_SPARSE_DAG_COUNTERS + 6u * region.brickCount) return;
        [unroll] for (uint i = 0u; i < VOXEL_SPARSE_DAG_COUNTERS; i++)
            voxelSparseScratch[i] = 0u;
    } else {
        return;
    }
    uint3 low = asuint(levelLow(int(voxelSparsePush.clipLevel)));
    voxelSparseWords[header + 1u] = low.x;
    voxelSparseWords[header + 2u] = low.y;
    voxelSparseWords[header + 3u] = low.z;
    voxelSparseWords[header + 4u] = 0u;
    voxelSparseWords[header + 5u] = 1u; // reserve payload word zero for the DAG root
    voxelSparseWords[header + 6u] = 0u;
    voxelSparseWords[header + 7u] = 0u;
    voxelSparseWords[header] = layout | VOXEL_SPARSE_BUILDING;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void voxelSparseHashClear(uint3 id : SV_DispatchThreadID) {
    VoxelSparseRegion region;
    if (!voxelSparseBuilding(VOXEL_SPARSE_HASH, region) || id.x >= voxelSparseHashSlots(region)) return;
    // Clearing keys alone is sufficient. Masks under zero keys are never read.
    voxelSparseWords[region.base + 2u + 3u * id.x] = 0u;
    if ((id.x & 63u) == 0u) InterlockedAdd(voxelSparseWords[region.header + 6u], 1u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void voxelSparseHashBuild(uint3 id : SV_DispatchThreadID) {
    VoxelSparseRegion region;
    if (!voxelSparseBuilding(VOXEL_SPARSE_HASH, region) || id.x >= region.brickCount) return;
    uint slots = voxelSparseHashSlots(region);
    if (slots == 0u || voxelSparseWords[region.header + 6u] != voxelSparseGroups(slots)) return;
    if ((id.x & 63u) == 0u) InterlockedAdd(voxelSparseWords[region.header + 7u], 1u);
    uint2 mask = voxelSparseGather(id.x);
    if ((mask.x | mask.y) == 0u) return;

    uint key = id.x + 1u;
    uint slot = hashU(key) & (slots - 1u);
    uint limit = min(slots, VOXEL_SPARSE_MAX_PROBES);
    for (uint probe = 1u; probe <= limit; probe++) {
        uint at = region.base + 2u + 3u * slot;
        uint before;
        InterlockedCompareExchange(voxelSparseWords[at], 0u, key, before);
        if (before == 0u) {
            // One invocation per Morton key. No builder reads another's masks;
            // they become visible to readers only after the publish barriers.
            voxelSparseWords[at + 1u] = mask.x;
            voxelSparseWords[at + 2u] = mask.y;
            InterlockedMax(voxelSparseWords[region.base + 1u], probe);
            return;
        }
        slot = (slot + 1u) & (slots - 1u);
    }
    InterlockedOr(voxelSparseWords[region.header + 4u], 1u);
}

// Scratch after the counters: 2*N intern slots, N representative IDs, N owner
// handles, and 2*N key words (enough for 2*N leaves or 8*(N/8) branch words).
// A representative ID is the representative's Morton index plus one. Owner
// handles are only read through those IDs, after the interning dispatch ends.
uint voxelSparseDagRepresentatives(uint bricks) { return VOXEL_SPARSE_DAG_COUNTERS + 2u * bricks; }
uint voxelSparseDagOwners(uint bricks) { return VOXEL_SPARSE_DAG_COUNTERS + 3u * bricks; }
uint voxelSparseDagKeys(uint bricks) { return VOXEL_SPARSE_DAG_COUNTERS + 4u * bricks; }

uint voxelSparseDagHandle(uint bricks, uint index) {
    uint representative = voxelSparseScratch[voxelSparseDagRepresentatives(bricks) + index];
    if (representative == 0u) return 0u;
    return voxelSparseScratch[voxelSparseDagOwners(bricks) + representative - 1u];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void voxelSparseDagPrepare(uint3 id : SV_DispatchThreadID) {
    VoxelSparseRegion region;
    if (!voxelSparseBuilding(VOXEL_SPARSE_DAG, region)) return;
    uint depth = voxelSparsePush.depth;
    if (depth > region.lastDepth || voxelSparseWords[region.header + 4u] != 0u) return;
    uint count = region.brickCount >> (3u * depth);
    if (id.x >= 2u * count) return;
    if (depth != 0u &&
        voxelSparseScratch[VOXEL_SPARSE_DAG_STAGES + depth - 1u] != voxelSparseGroups(8u * count)) return;
    if ((id.x & 63u) == 0u) InterlockedAdd(voxelSparseScratch[depth], 1u);
    voxelSparseScratch[VOXEL_SPARSE_DAG_COUNTERS + id.x] = 0u;
    if (id.x >= count) return;

    uint keys = voxelSparseDagKeys(region.brickCount);
    if (depth == 0u) {
        uint2 mask = voxelSparseGather(id.x);
        voxelSparseScratch[keys + 2u * id.x] = mask.x;
        voxelSparseScratch[keys + 2u * id.x + 1u] = mask.y;
    } else {
        // Dropping the low Morton triple replaces eight siblings with a parent.
        // The low triple is precisely x | y<<1 | z<<2, the DAG child order.
        uint child = 8u * id.x;
        [unroll] for (uint i = 0u; i < 8u; i++)
            voxelSparseScratch[keys + child + i] = voxelSparseDagHandle(region.brickCount, child + i);
    }
}

bool voxelSparseDagEqual(uint a, uint b, uint width) {
    for (uint i = 0u; i < width; i++)
        if (voxelSparseScratch[a + i] != voxelSparseScratch[b + i]) return false;
    return true;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void voxelSparseDagIntern(uint3 id : SV_DispatchThreadID) {
    VoxelSparseRegion region;
    if (!voxelSparseBuilding(VOXEL_SPARSE_DAG, region)) return;
    uint depth = voxelSparsePush.depth;
    if (depth > region.lastDepth) return;
    uint count = region.brickCount >> (3u * depth);
    if (id.x >= count || voxelSparseScratch[depth] != voxelSparseGroups(2u * count)) return;
    if ((id.x & 63u) == 0u) InterlockedAdd(voxelSparseScratch[VOXEL_SPARSE_DAG_STAGES + depth], 1u);

    uint width = depth == 0u ? 2u : 8u;
    uint keys = voxelSparseDagKeys(region.brickCount);
    uint mine = keys + id.x * width;
    uint mapping = voxelSparseDagRepresentatives(region.brickCount) + id.x;
    uint hash = 0u, anyWord = 0u;
    for (uint i = 0u; i < width; i++) {
        uint word = voxelSparseScratch[mine + i];
        anyWord |= word;
        hash = hashU(hash ^ word);
    }
    if (anyWord == 0u) {
        voxelSparseScratch[mapping] = 0u;
        return;
    }

    uint slotMask = 2u * count - 1u;
    uint tag = hash & 0xffff0000u;
    uint entry = tag | (id.x + 1u); // at most 32^3 representatives per stage
    uint slot = hash & slotMask;
    uint limit = min(2u * count, VOXEL_SPARSE_MAX_PROBES);
    for (uint probe = 0u; probe < limit; probe++) {
        uint before;
        InterlockedCompareExchange(voxelSparseScratch[VOXEL_SPARSE_DAG_COUNTERS + slot], 0u, entry, before);
        if (before == 0u) {
            uint handle;
            InterlockedAdd(voxelSparseWords[region.header + 5u], width, handle);
            if (handle >= region.capacity || width > region.capacity - handle) {
                InterlockedOr(voxelSparseWords[region.header + 4u], 1u);
                return;
            }
            for (uint i = 0u; i < width; i++)
                voxelSparseWords[region.base + handle + i] = voxelSparseScratch[mine + i];
            voxelSparseScratch[voxelSparseDagOwners(region.brickCount) + id.x] = handle;
            voxelSparseScratch[mapping] = id.x + 1u;
            return;
        }
        uint representative = before & 0xffffu;
        // Candidate keys were written before this dispatch. Contenders compare
        // the immutable key and never spin waiting for the winner's handle.
        if ((before & 0xffff0000u) == tag &&
            voxelSparseDagEqual(mine, keys + (representative - 1u) * width, width)) {
            voxelSparseScratch[mapping] = representative;
            return;
        }
        slot = (slot + 1u) & slotMask;
    }
    InterlockedOr(voxelSparseWords[region.header + 4u], 1u);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void voxelSparsePublish() {
    VoxelSparseRegion region;
    uint layout = voxelSparsePush.layout;
    if (layout != VOXEL_SPARSE_HASH && layout != VOXEL_SPARSE_DAG) return;
    if (!voxelSparseBuilding(layout, region)) return;
    uint header = region.header;
    if (voxelSparseWords[header + 4u] != 0u) return;
    if (layout == VOXEL_SPARSE_HASH) {
        if (voxelSparseWords[header + 7u] != voxelSparseGroups(region.brickCount)) return;
    } else {
        uint finalStage = VOXEL_SPARSE_DAG_STAGES + region.lastDepth;
        if (voxelSparseScratch[finalStage] != 1u) return;
        voxelSparseWords[region.base] = voxelSparseDagHandle(region.brickCount, 0u);
    }
    voxelSparseWords[header] = layout;
}

bool voxelSparseReadMask(uint layout, VoxelSparseRegion region, uint3 brick, out uint2 mask) {
    mask = uint2(0u);
    if (layout == VOXEL_SPARSE_HASH) {
        uint slots = voxelSparseHashSlots(region);
        if (slots == 0u || voxelSparseWords[region.base] != slots - 1u) return false;
        uint limit = voxelSparseWords[region.base + 1u];
        if (limit > min(slots, VOXEL_SPARSE_MAX_PROBES)) return false;
        uint key = voxelSparseMorton(brick) + 1u;
        uint slot = hashU(key) & (slots - 1u);
        for (uint probe = 0u; probe < limit; probe++) {
            uint at = region.base + 2u + 3u * slot;
            uint tag = voxelSparseWords[at];
            if (tag == 0u) return true;
            if (tag == key) {
                mask = uint2(voxelSparseWords[at + 1u], voxelSparseWords[at + 2u]);
                return true;
            }
            slot = (slot + 1u) & (slots - 1u);
        }
        // No occupied key was inserted after this published global probe bound.
        return true;
    }

    uint used = voxelSparseWords[region.header + 5u];
    if (used == 0u || used > region.capacity) return false;
    uint node = voxelSparseWords[region.base];
    for (uint depth = region.lastDepth; depth > 0u; depth--) {
        if (node == 0u) return true;
        if (node >= used || used - node < 8u) return false;
        uint3 xyz = (brick >> (depth - 1u)) & 1u;
        uint octant = xyz.x | (xyz.y << 1u) | (xyz.z << 2u);
        uint child = voxelSparseWords[region.base + node + octant];
        if (child >= node) return false;
        node = child;
    }
    if (node == 0u) return true;
    if (node >= used || used - node < 2u) return false;
    uint at = region.base + node;
    mask = uint2(voxelSparseWords[at], voxelSparseWords[at + 1u]);
    return true;
}

bool voxelSparseMaskOverlaps(uint2 mask, uint3 low, uint3 end) {
    if ((mask.x | mask.y) == 0u) return false;
    if (all(low == 0u) && all(end == 4u)) return true;
    uint rowMask = (1u << end.x) - (1u << low.x);
    for (uint z = low.z; z < end.z; z++) {
        uint slice = mask[z >> 1u] >> (16u * (z & 1u));
        for (uint y = low.y; y < end.y; y++)
            if ((slice & (rowMask << (4u * y))) != 0u) return true;
    }
    return false;
}

// True only if the entire in-clip portion of the world-aligned 4^3 brick
// containing `localCell` is known empty. False means occupied or unsupported;
// in either case walk the authoritative dense cells. The caller still uses
// hopBrick's existing clip/reach/step limits. A clip low need not be 4-aligned.
bool voxelSparseWorldBrickEmpty(uint layout, int level, int3 localCell) {
    if (layout != VOXEL_SPARSE_HASH && layout != VOXEL_SPARSE_DAG) return false;
    VoxelSparseRegion region;
    int size = gridSize();
    if (any(localCell < 0) || any(localCell >= size) || !voxelSparseRegion(level, region)) return false;
    uint header = region.header;
    if (voxelSparseWords[header] != layout) return false;
    int3 low = levelLow(level);
    uint3 builtLow = uint3(voxelSparseWords[header + 1u], voxelSparseWords[header + 2u],
                          voxelSparseWords[header + 3u]);
    if (any(asuint(low) != builtLow)) return false;

    // Work in clip-local space to avoid forming a signed world coordinate at
    // the high boundary. At most eight local masks intersect the world brick.
    int3 start = localCell - (((low & 3) + localCell) & 3);
    uint3 from = uint3(max(start, 0));
    uint3 end = uint3(min(start + 4, size));
    uint3 first = from >> 2u;
    uint3 last = (end - 1u) >> 2u;
    for (uint z = first.z; z <= last.z; z++) {
        for (uint y = first.y; y <= last.y; y++) {
            for (uint x = first.x; x <= last.x; x++) {
                uint3 brick = uint3(x, y, z);
                uint2 mask;
                if (!voxelSparseReadMask(layout, region, brick, mask)) return false;
                uint3 brickLow = brick << 2u;
                if (voxelSparseMaskOverlaps(mask, max(from, brickLow) - brickLow,
                                           min(end, brickLow + 4u) - brickLow)) return false;
            }
        }
    }
    return true;
}
