// CORTAO main pass; see lib/ao.slang.
// The trace only runs without GI (see ao.rs): active GI traces its own contact occlusion.

#include "lib/ao.slang"
#define CORTAO_SAMPLE_COUNT 12
#define CORTAO_RAY_COUNT 16
// Each occluder's rays come from a table instead of a test per ray; lutMain
// fills it once with the analytic test.
#define CORTAO_OCCLUDER_LOOKUP_TABLE 1
#include "lib/cortao/CORTAOMainPass.h"

[[vk::push_constant]] ConstantBuffer<CORTAOMainPassParameters> params;

cortao_uint cortao_samplePackedData(cortao_float2 texCoords, cortao_float mipmap) {
    return packed.SampleLevel(int_border, texCoords, mipmap);
}

/// Nearest texel, clamped: occluders past the table's largest radius cover
/// as much as its last layer.
cortao_uint cortao_sampleOccluderLookupTable(cortao_float3 texCoords) {
    int3 texel = clamp(int3(texCoords * AO_LUT_SIZE), 0, AO_LUT_SIZE - 1);
    return occluder_lut.Load(int4(texel, 0));
}

cortao_half3 cortao_loadNormal(cortao_int2 pixelCoords) {
    return cortao_decodeNormal(packed.Load(int3(pixelCoords, 0)));
}

void cortao_writeResult(cortao_int2 pixelCoords, cortao_half occlusion, cortao_half3 bentNormal,
                        cortao_half3x3 tangentSpaceTransform, cortao_uint rayBits) {
    raw_out[pixelCoords] = occlusion;
}

/// Morton order inside the group, for cache locality.
uint2 deinterleave(uint index) {
    uint2 p = uint2(index, index) >> uint2(0, 1);
    p &= 0x5555u;
    p = (p ^ (p >> 1u)) & 0x3333u;
    p = (p ^ (p >> 2u)) & 0x0f0fu;
    p = (p ^ (p >> 4u)) & 0x00ffu;
    return p;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void traceMain(uint3 group: SV_GroupID, uint index: SV_GroupIndex) {
    cortao_mainPass(group.xy * 16 + deinterleave(index), params);
}

/// Fill the occluder table at its texel centres with CORTAO's analytic test:
/// xy is the tangent-space direction as a sphere map, z the occluder radius
/// times sqrt(1/2), as cortao_computeRayOcclusion looks them up.
[shader("compute")]
[numthreads(4, 4, 4)]
void lutMain(uint3 id: SV_DispatchThreadID) {
    float3 coords = (float3(id) + 0.5) / AO_LUT_SIZE;
    // Sphere map back to a direction.
    float2 encoded = coords.xy * 4.0 - 2.0;
    float f = dot(encoded, encoded);
    float3 direction = float3(encoded * sqrt(max(1.0 - f * 0.25, 0.0)), 1.0 - f * 0.5);
    float radius = coords.z * sqrt(2.0);
    float cos_angle = max(0.0, 1.0 - radius * radius * 0.5);
    uint bits = 0u;
    // Outside the sphere map's disc there is no direction.
    if (f <= 4.0) {
        for (uint j = 0u; j < CORTAO_RAY_COUNT; j++) {
            if (dot(float3(cortao_rayDirection(j)), direction) > cos_angle)
                bits |= 1u << j;
        }
    }
    occluder_lut_out[id] = bits;
}
