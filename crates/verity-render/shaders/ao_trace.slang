// CORTAO main pass; see lib/ao.slang.

#include "lib/ao.slang"
#define CORTAO_SAMPLE_COUNT 12
#define CORTAO_RAY_COUNT 16
// shortcut: occluders are intersected analytically; CORTAO's lookup table
// (CORTAO_OCCLUDER_LOOKUP_TABLE) is faster if the trace shows in profiles.
#include "lib/cortao/CORTAOMainPass.h"

[[vk::push_constant]] ConstantBuffer<CORTAOMainPassParameters> params;

cortao_uint cortao_samplePackedData(cortao_float2 texCoords, cortao_float mipmap) {
    return packed.SampleLevel(int_border, texCoords, mipmap);
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
