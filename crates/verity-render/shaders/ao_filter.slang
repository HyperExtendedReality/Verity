// CORTAO filter pass; see lib/ao.slang.

#include "lib/ao.slang"
#define CORTAO_LARGE_FILTER 1
#define CORTAO_FILTER_SYMMETRY 1
#include "lib/cortao/CORTAOFilterPass.h"

[[vk::push_constant]] ConstantBuffer<CORTAOFilterPassParameters> params;

/// The filter's gather offsets are folded into the coordinates, so the GPU
/// needs no extended gather support.
cortao_uint4 cortao_gatherPackedData(cortao_float2 texCoords, cortao_int2 offset) {
    return packed.Gather(int_border, texCoords + float2(offset) / globals.viewport.zw);
}

cortao_half4 cortao_gatherSSAO(cortao_float2 texCoords, cortao_int2 offset) {
    return raw.Gather(float_border, texCoords + float2(offset) / globals.viewport.zw);
}

void cortao_writeFilterOutput(cortao_int2 pixelCoords, cortao_half ssao) {
    ao_out[viewport_offset() + pixelCoords] = ssao;
}

/// Each thread filters a 2x2 block.
[shader("compute")]
[numthreads(8, 8, 1)]
void filterMain(uint3 id: SV_DispatchThreadID) {
    cortao_filterPass(id.xy, params);
}
