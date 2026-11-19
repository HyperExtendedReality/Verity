// CORTAO prepass; see lib/ao.slang.

#include "lib/ao.slang"
#include "lib/cortao/CORTAOPrepass.h"

[[vk::push_constant]] ConstantBuffer<CORTAOPrepassParameters> params;

/// View-space position of viewport pixel `p` (camera looks down -z), or
/// zero for sky.
float3 view_position(int2 p) {
    float2 size = globals.viewport.zw;
    p = clamp(p, int2(0, 0), int2(size) - 1);
    float depth = scene_depth.Load(int3(viewport_offset() + p, 0));
    if (depth <= 0.0)
        return 0.0;
    float linear_depth = globals.projection.z / depth;
    float2 ndc = (float2(p) + 0.5) / size * 2.0 - 1.0;
    return float3(ndc / globals.projection.xy, -1.0) * linear_depth;
}

cortao_float cortao_loadDepth(cortao_int2 pixelCoords) {
    return scene_depth.Load(int3(viewport_offset() + pixelCoords, 0));
}

/// The visibility buffer has no normals before the resolve, so take the
/// geometric normal from depth, using the neighbour on each axis that lies
/// closer in depth so silhouettes do not smear across the gap.
// shortcut: flat, no normal maps; pack resolve's shading normal if AO needs bump detail.
cortao_float3 cortao_loadNormal(cortao_int2 pixelCoords) {
    float3 c = view_position(pixelCoords);
    float3 l = view_position(pixelCoords - int2(1, 0)) - c;
    float3 r = view_position(pixelCoords + int2(1, 0)) - c;
    float3 u = view_position(pixelCoords - int2(0, 1)) - c;
    float3 d = view_position(pixelCoords + int2(0, 1)) - c;
    float3 dx = abs(l.z) < abs(r.z) ? -l : r;
    float3 dy = abs(u.z) < abs(d.z) ? -u : d;
    float3 n = normalize(cross(dx, dy));
    // Face the camera.
    return dot(n, c) > 0.0 ? -n : n;
}

void cortao_writePackedData(cortao_int2 pixelCoords, cortao_int mipLevel, cortao_uint data) {
    packed_mips[mipLevel][pixelCoords] = data;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void prepassMain(uint3 id: SV_DispatchThreadID) {
    cortao_prepass(id.xy, params);
}
