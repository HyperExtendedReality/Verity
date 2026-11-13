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

/// Inside the side planes of a projection; ours have no useful far plane.
bool sphere_inside(float4x4 m, float3 center, float radius) {
    float4 point = float4(center, 1.0);
    float4 planes[4] = { m[3] + m[0], m[3] - m[0], m[3] + m[1], m[3] - m[1] };
    for (uint i = 0; i < 4; ++i) {
        if (dot(planes[i], point) < -radius * length(planes[i].xyz))
            return false;
    }
    return true;
}

bool sphere_visible(float3 center, float radius) {
    return sphere_inside(globals.view_proj, center, radius);
}

void push_item(uint pass, uint2 item) {
    uint arguments = MATRIX_PASS_WORDS + pass * 4;
    uint slot;
    InterlockedAdd(globals.matrix_counters[arguments + 3], 1, slot);
    if (slot < MATRIX_QUEUE) {
        globals.matrix_queue[(pass & 1) * MATRIX_QUEUE + slot] = item;
        if (slot % 64 == 0)
            InterlockedAdd(globals.matrix_counters[arguments], 1);
    }
}

void push_entries(uint list, uint2 entry, uint count) {
    uint slot;
    InterlockedAdd(globals.matrix_counters[list * 4 + 1], count, slot);
    uint end = min(slot + count, matrix_list_capacity(list));
    uint base = matrix_list_base(list);
    for (uint i = slot; i < end; ++i)
        globals.matrix_entries[base + i] = uint2(entry.x, entry.y + i - slot);
}

void push_layer(uint layer, uint instance, uint cluster, uint first_brick, uint brick_count, bool triangles) {
    uint list = MATRIX_LIST_LAYERS + layer * 2;
    if (triangles)
        push_entries(list, uint2(instance, cluster), 1);
    if (brick_count > 0)
        push_entries(list + 1, uint2(instance, first_brick), brick_count);
}

/// Add a cluster to the lists of every shadow layer it can reach. `sphere` is in
/// world space. Without this every layer would draw every caster.
void push_shadow_layers(uint instance, uint cluster, uint first_brick, uint brick_count, bool triangles, float4 sphere) {
    if (globals.sun_direction.w > 0.0) {
        for (uint cascade = 0; cascade < CASCADES; ++cascade) {
            if (sphere_inside(globals.shadow_view_proj[cascade], sphere.xyz, sphere.w))
                push_layer(cascade, instance, cluster, first_brick, brick_count, triangles);
        }
    }
    uint lit = min(globals.counts.x, MAX_POINT_SHADOWS);
    for (uint light = 0; light < lit; ++light) {
        float4 reach = globals.lights[light].position_radius;
        float3 offset = sphere.xyz - reach.xyz;
        float range = reach.w + sphere.w;
        if (reach.w <= 0.0 || dot(offset, offset) > range * range)
            continue;
        for (uint face = 0; face < POINT_SHADOW_FACES; ++face) {
            uint layer = light * POINT_SHADOW_FACES + face;
            if (sphere_inside(globals.point_shadow_view_proj[layer], sphere.xyz, sphere.w))
                push_layer(CASCADES + layer, instance, cluster, first_brick, brick_count, triangles);
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void instanceMain(uint3 id: SV_DispatchThreadID) {
    if (id.x >= select_constants.instance_count)
        return;
    MatrixInstance instance = globals.matrix_instances[id.x];
    MatrixAsset asset = globals.matrix_assets[instance.info.x];
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
    float fade = imposter_fade(instance);
    if (fade > 0.0 && (flags & ITEM_VISIBLE) != 0)
        push_entries(MATRIX_LIST_IMPOSTER, uint2(id.x, 0), 1);
    if (fade >= 1.0) {
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
    uint count = min(globals.matrix_counters[MATRIX_PASS_WORDS + pass * 4 + 3], MATRIX_QUEUE);
    if (id.x >= count)
        return;
    uint2 item = globals.matrix_queue[(pass & 1) * MATRIX_QUEUE + id.x];
    MatrixInstance instance = globals.matrix_instances[item.x & ITEM_INSTANCE];
    MatrixNode node = globals.matrix_nodes[item.y];
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
    MatrixGroup group = globals.matrix_groups[node.group];
    uint index = item.x & ITEM_INSTANCE;
    bool two_sided = instance.info.y != 0;
    for (uint i = 0; i < group.cluster_count; ++i) {
        uint cluster_index = group.first_cluster + i;
        MatrixCluster cluster = globals.matrix_clusters[cluster_index];
        if (cluster.refined != MATRIX_NONE) {
            MatrixGroup refined = globals.matrix_groups[cluster.refined];
            float refined_error = coarse ? refined.error
                                         : projected_error(instance, refined.sphere, refined.error);
            if (refined_error > threshold)
                continue;
        }
        float3 center = mul(instance.base.model, float4(cluster.sphere.xyz, 1.0)).xyz;
        float radius = cluster.sphere.w * asfloat(instance.info.z);
        if ((item.x & ITEM_VISIBLE) != 0 && sphere_visible(center, radius)) {
            if (cluster.triangle_count > 0)
                push_entries(two_sided ? MATRIX_LIST_TWO_SIDED : MATRIX_LIST_ONE_SIDED, uint2(index, cluster_index), 1);
            if (cluster.brick_count > 0)
                push_entries(MATRIX_LIST_VOXELS, uint2(index, cluster.first_brick), cluster.brick_count);
        }
        if ((item.x & ITEM_SHADOW) != 0)
            push_shadow_layers(index, cluster_index, cluster.first_brick, cluster.brick_count,
                cluster.triangle_count > 0, float4(center, radius));
    }
}
