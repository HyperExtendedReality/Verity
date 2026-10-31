// Debug view of the renderer's thin G-buffer, decoded with lux's own
// decoders to prove the packing matches. Four vertical bands: shading normal,
// geometric normal, base color, motion (magnified).

#include "common.slang"
#include "receiver_metadata.slang"

[[vk::binding(0, 0)]] Texture2D<uint4> gbuffer;
[[vk::binding(1, 0)]] Texture2D<uint4> rasterReceiver;
[[vk::binding(2, 0)]] Texture2D<float4> motionTexture;
[[vk::binding(3, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> hdrTarget;

struct View {
    uint4 viewport;
};
[[vk::push_constant]] ConstantBuffer<View> view;

/// Undo the composite's Reinhard tonemap so `display` appears unchanged.
float3 untonemap(float3 display) {
    float3 c = min(display, 0.98);
    return c / (1.0 - c);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void debugSurface(uint3 id: SV_DispatchThreadID) {
    uint4 viewport = view.viewport;
    if (any(id.xy >= viewport.zw))
        return;
    uint2 pixel = id.xy + viewport.xy;
    uint4 packed = gbuffer.Load(int3(pixel, 0));
    uint4 receiver = rasterReceiver.Load(int3(pixel, 0));
    if ((receiver.z & 0x80000000u) == 0u)
        return;
    float3 shading = deferredShadingNormal(packed);
    float3 color;
    switch (min(id.x * 4 / viewport.z, 3u)) {
    case 0: color = pow(shading * 0.5 + 0.5, 2.2); break;
    case 1: color = pow(receiverGeometricNormal(receiver.z, shading) * 0.5 + 0.5, 2.2); break;
    case 2: color = deferredBaseColor(packed); break;
    default: color = pow(saturate(abs(motionTexture.Load(int3(pixel, 0)).xy) * 200.0), 2.2).xyy; break;
    }
    hdrTarget[pixel] = float4(untonemap(color), 1.0);
}
