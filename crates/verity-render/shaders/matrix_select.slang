// GPU selection for Matrix: pick a tier per instance, then walk each instance's
// cluster tree and append what to draw to per-pass lists. Runs before shadows.
//
// `instanceMain` decides between the imposter and the clusters and seeds the
// queue with tree roots. `nodeMain` runs once per tree level, consuming one half
// of the queue and filling the other; each pass's indirect arguments are grown
// as items arrive, so no pass needs a readback or a separate setup dispatch.

#include "lib/common.slang"
#include "lib/matrix.slang"
#include "lib/matrix_lists.slang"

[[vk::binding(21)]] StructuredBuffer<MatrixNode> matrix_nodes;
[[vk::binding(22)]] StructuredBuffer<MatrixGroup> matrix_groups;
[[vk::binding(27)]] RWStructuredBuffer<uint2> matrix_entries;
[[vk::binding(28)]] RWStructuredBuffer<uint2> matrix_queue;
[[vk::binding(29)]] RWStructuredBuffer<uint> matrix_counters;

// The first eight bytes are the graphics stages' push range.
struct SelectConstants {
    [[vk::offset(8)]] uint pass;
    [[vk::offset(12)]] uint instance_count;
};
[[vk::push_constant]] ConstantBuffer<SelectConstants> select_constants;

/// Bits of a queue item's instance word.
static const uint ITEM_VISIBLE = 1u << 31;
static const uint ITEM_SHADOW = 1u << 30;
/// Take the coarsest cut whatever the distance: a distant caster needs no detail.
static const uint ITEM_COARSE = 1u << 29;
static const uint ITEM_INSTANCE = ITEM_COARSE - 1;

/// Largest allowed projected error, in pixels.
static const float ERROR_PIXELS = 1.0;
/// Below this diameter an object is its imposter.
static const float IMPOSTER_PIXELS = 64.0;
static const float COARSE_THRESHOLD = 1.0e30;

float pixels_per_unit(float distance) {
    return abs(globals.projection.y) * 0.5 * globals.viewport.w / max(distance, globals.projection.z);
}

float projected_error(MatrixInstance instance, float4 sphere, float error) {
    float3 center = mul(instance.base.model, float4(sphere.xyz, 1.0)).xyz;
    float scale = asfloat(instance.info.z);
    float distance = length(center - globals.eye.xyz) - sphere.w * scale;
    return error * scale * pixels_per_unit(distance);
}

/// Inside the side planes of the view; the projection has no far plane.
bool sphere_visible(float3 center, float radius) {
    float4x4 m = globals.view_proj;
    float4 point = float4(center, 1.0);
    float4 planes[4] = { m[3] + m[0], m[3] - m[0], m[3] + m[1], m[3] - m[1] };
    for (uint i = 0; i < 4; ++i) {
        if (dot(planes[i], point) < -radius * length(planes[i].xyz))
            return false;
    }
    return true;
}

void push_item(uint pass, uint2 item) {
    uint arguments = MATRIX_PASS_WORDS + pass * 4;
    uint slot;
    InterlockedAdd(matrix_counters[arguments + 3], 1, slot);
    if (slot < MATRIX_QUEUE) {
        matrix_queue[(pass & 1) * MATRIX_QUEUE + slot] = item;
        if (slot % 64 == 0)
            InterlockedAdd(matrix_counters[arguments], 1);
    }
}

void push_entries(uint list, uint2 entry, uint count) {
    uint slot;
    InterlockedAdd(matrix_counters[list * 4 + 1], count, slot);
    uint end = min(slot + count, matrix_list_capacity(list));
    uint base = matrix_list_base(list);
    for (uint i = slot; i < end; ++i)
        matrix_entries[base + i] = uint2(entry.x, entry.y + i - slot);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void instanceMain(uint3 id: SV_DispatchThreadID) {
    if (id.x >= select_constants.instance_count)
        return;
    MatrixInstance instance = matrix_instances[id.x];
    MatrixAsset asset = matrix_assets[instance.info.x];
    float3 center = instance.sphere.xyz;
    float radius = instance.sphere.w;
    float distance = length(center - globals.eye.xyz);
    uint flags = 0;
    if (sphere_visible(center, radius))
        flags |= ITEM_VISIBLE;
    if (distance - radius < globals.shadow_splits[CASCADES - 1])
        flags |= ITEM_SHADOW;
    if (flags == 0)
        return;
    float pixels = 2.0 * radius * pixels_per_unit(distance);
    if (asset.imposter_albedo != NO_TEXTURE && pixels < IMPOSTER_PIXELS) {
        if ((flags & ITEM_VISIBLE) != 0)
            push_entries(MATRIX_LIST_IMPOSTER, uint2(id.x, 0), 1);
        flags = (flags & ~ITEM_VISIBLE) | ITEM_COARSE;
        if ((flags & ITEM_SHADOW) == 0)
            return;
    }
    for (uint root = 0; root < asset.root_count; ++root)
        push_item(0, uint2(id.x | flags, asset.first_root + root));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void nodeMain(uint3 id: SV_DispatchThreadID) {
    uint pass = select_constants.pass;
    uint count = min(matrix_counters[MATRIX_PASS_WORDS + pass * 4 + 3], MATRIX_QUEUE);
    if (id.x >= count)
        return;
    uint2 item = matrix_queue[(pass & 1) * MATRIX_QUEUE + id.x];
    MatrixInstance instance = matrix_instances[item.x & ITEM_INSTANCE];
    MatrixNode node = matrix_nodes[item.y];
    bool coarse = (item.x & ITEM_COARSE) != 0;
    float threshold = coarse ? COARSE_THRESHOLD : ERROR_PIXELS;
    float node_error = coarse ? node.error : projected_error(instance, node.sphere, node.error);
    if (node_error <= threshold)
        return;
    if (node.group == MATRIX_NONE) {
        for (uint child = 0; child < node.child_count; ++child)
            push_item(pass + 1, uint2(item.x, node.first_child + child));
        return;
    }
    MatrixGroup group = matrix_groups[node.group];
    uint index = item.x & ITEM_INSTANCE;
    bool two_sided = instance.info.y != 0;
    for (uint i = 0; i < group.cluster_count; ++i) {
        uint cluster_index = group.first_cluster + i;
        MatrixCluster cluster = matrix_clusters[cluster_index];
        if (cluster.refined != MATRIX_NONE) {
            MatrixGroup refined = matrix_groups[cluster.refined];
            float refined_error = coarse ? refined.error
                                         : projected_error(instance, refined.sphere, refined.error);
            if (refined_error > threshold)
                continue;
        }
        bool visible = (item.x & ITEM_VISIBLE) != 0;
        if (visible) {
            float3 center = mul(instance.base.model, float4(cluster.sphere.xyz, 1.0)).xyz;
            visible = sphere_visible(center, cluster.sphere.w * asfloat(instance.info.z));
        }
        bool shadow = (item.x & ITEM_SHADOW) != 0;
        if (cluster.triangle_count > 0) {
            if (visible)
                push_entries(two_sided ? MATRIX_LIST_TWO_SIDED : MATRIX_LIST_ONE_SIDED, uint2(index, cluster_index), 1);
            if (shadow)
                push_entries(MATRIX_LIST_SHADOW, uint2(index, cluster_index), 1);
        }
        if (cluster.brick_count > 0) {
            if (visible)
                push_entries(MATRIX_LIST_VOXELS, uint2(index, cluster.first_brick), cluster.brick_count);
            if (shadow)
                push_entries(MATRIX_LIST_SHADOW_VOXELS, uint2(index, cluster.first_brick), cluster.brick_count);
        }
    }
}
