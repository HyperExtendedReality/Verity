// Screen-space ambient occlusion: CORTAO (Circular Occluder Ray Traced AO,
// https://github.com/theagentd/CORTAO, MIT; see lib/cortao/). Three passes
// over the scene viewport: ao_prepass packs linear depth and a normal into a
// mip chain, ao_trace traces rays against occluder discs, ao_filter blurs by
// depth and normal. The result multiplies the hemisphere ambient in resolve.

#include "common.slang"

#define CORTAO_HLSL 1
#define CORTAO_PERSPECTIVE_PROJECTION 1

[[vk::binding(33)]] Texture2D<float> scene_depth;
[[vk::binding(0, 1)]] [[vk::image_format("r32ui")]] RWTexture2D<uint> packed_mips[8];
[[vk::binding(1, 1)]] Texture2D<uint> packed;
[[vk::binding(2, 1)]] [[vk::image_format("r32f")]] RWTexture2D<float> raw_out;
[[vk::binding(3, 1)]] Texture2D<float> raw;
[[vk::binding(4, 1)]] [[vk::image_format("r32f")]] RWTexture2D<float> ao_out;
/// Nearest, with a zero border: off-screen samples read as sky.
[[vk::binding(5, 1)]] SamplerState int_border;
[[vk::binding(6, 1)]] SamplerState float_border;

int2 viewport_offset() {
    return int2(globals.viewport.xy);
}
