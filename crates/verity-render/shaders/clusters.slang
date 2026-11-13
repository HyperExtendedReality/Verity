// Assign point lights to view-space clusters. One thread fills one 32-light
// mask word of one cluster, so no atomics are needed.

#include "lib/common.slang"

[shader("compute")]
[numthreads(64, 1, 1)]
void clusterMain(uint3 id: SV_DispatchThreadID) {
    uint cluster = id.x / LIGHT_WORDS;
    uint word = id.x % LIGHT_WORDS;
    if (cluster >= CLUSTER_COUNT)
        return;
    uint3 c = uint3(cluster % CLUSTER_X, (cluster / CLUSTER_X) % CLUSTER_Y, cluster / (CLUSTER_X * CLUSTER_Y));

    // View-space bounds of the tile's frustum segment between two slice depths.
    float2 ndc0 = float2(c.xy) / float2(CLUSTER_X, CLUSTER_Y) * 2.0 - 1.0;
    float2 ndc1 = float2(c.xy + 1) / float2(CLUSTER_X, CLUSTER_Y) * 2.0 - 1.0;
    float2 scale = globals.projection.xy;
    float3 lo = float3(1.0e30);
    float3 hi = float3(-1.0e30);
    for (uint k = 0; k < 8; ++k) {
        float depth = slice_depth(c.z + ((k >> 2) & 1));
        float2 ndc = float2((k & 1) != 0 ? ndc1.x : ndc0.x, (k & 2) != 0 ? ndc1.y : ndc0.y);
        float3 p = float3(ndc * depth / scale, -depth);
        lo = min(lo, p);
        hi = max(hi, p);
    }

    uint bits = 0;
    uint count = globals.counts.x;
    for (uint bit = 0; bit < 32; ++bit) {
        uint i = word * 32 + bit;
        if (i >= count)
            break;
        float4 light = globals.lights[i].position_radius;
        float radius = light.w;
        if (!(radius > 0.0) || isinf(radius))
            continue;
        float3 center = mul(globals.view, float4(light.xyz, 1.0)).xyz;
        float3 offset = center - clamp(center, lo, hi);
        if (dot(offset, offset) <= radius * radius)
            bits |= 1u << bit;
    }
    clusters[cluster * LIGHT_WORDS + word] = bits;
}
