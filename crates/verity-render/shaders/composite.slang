// Expose and tonemap the HDR scene into the swapchain inside the scene viewport.

#include "lib/common.slang"

/// The exposure `exposure.slang` adapts; read-only here, since fragment
/// shaders may not write storage buffers without fragmentStoresAndAtomics.
[[vk::binding(35)]] StructuredBuffer<uint> exposure_value;
/// Index of the exposure in the state; see `lib/exposure.slang`.
static const uint EXPOSURE_INDEX = 2;

[shader("vertex")]
float4 vertexMain(uint vertex: SV_VertexID): SV_Position {
    float2 uv = float2((vertex << 1) & 2, vertex & 2);
    return float4(uv * 2.0 - 1.0, 0.0, 1.0);
}

/// Uchimura's Gran Turismo curve (CEDEC 2017): a contrast toe, a linear
/// midtone section and a smooth shoulder to white, per channel.
float3 gt_tonemap(float3 x) {
    const float P = 1.0;  // display maximum
    const float a = 1.0;  // linear section slope
    const float m = 0.22; // linear section start
    const float l = 0.4;  // linear section length
    const float c = 1.33; // toe power
    float l0 = (P - m) * l / a;
    float S0 = m + l0;
    float S1 = m + a * l0;
    float CP = -a / (P - S1);
    float3 w0 = 1.0 - smoothstep(0.0, m, x);
    float3 w2 = step(S0, x);
    float3 w1 = 1.0 - w0 - w2;
    float3 toe = m * pow(max(x, 0.0) / m, c);
    float3 linear = m + a * (x - m);
    float3 shoulder = P - (P - S1) * exp(CP * (x - S0));
    return toe * w0 + linear * w1 + shoulder * w2;
}

[shader("fragment")]
float4 fragmentMain(float4 position: SV_Position): SV_Target0 {
    float4 color = hdr.Load(int3(int2(position.xy), 0));
    if (color.a == 0.0)
        discard;
    float3 mapped = saturate(gt_tonemap(color.rgb * asfloat(exposure_value[EXPOSURE_INDEX])));
    return float4(globals.eye.w > 0.5 ? mapped : pow(mapped, 1.0 / 2.2), 1.0);
}
