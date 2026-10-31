// Tonemap the HDR scene into the swapchain inside the scene viewport.

#include "lib/common.slang"

[shader("vertex")]
float4 vertexMain(uint vertex: SV_VertexID): SV_Position {
    float2 uv = float2((vertex << 1) & 2, vertex & 2);
    return float4(uv * 2.0 - 1.0, 0.0, 1.0);
}

[shader("fragment")]
float4 fragmentMain(float4 position: SV_Position): SV_Target0 {
    float4 color = hdr.Load(int3(int2(position.xy), 0));
    if (color.a == 0.0)
        discard;
    float3 mapped = color.rgb / (color.rgb + 1.0);
    return float4(globals.eye.w > 0.5 ? mapped : pow(mapped, 1.0 / 2.2), 1.0);
}
