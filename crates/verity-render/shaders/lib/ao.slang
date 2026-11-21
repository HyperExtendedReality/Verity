// Screen-space ambient occlusion: CORTAO (Circular Occluder Ray Traced AO,
// https://github.com/theagentd/CORTAO, MIT; see lib/cortao/). Three passes
// over the scene viewport: ao_prepass packs linear depth and a normal into a
// mip chain, ao_trace traces rays against occluder discs, ao_filter blurs by
// depth and normal. Resolve darkens the hemisphere ambient with it: diffuse
// through a multi-bounce fit, specular through specular occlusion.

#include "common.slang"

#define CORTAO_HLSL 1
#define CORTAO_PERSPECTIVE_PROJECTION 1

[[vk::binding(33)]] Texture2D<float> scene_depth;
[[vk::binding(0, 1)]] [[vk::image_format("r32ui")]] RWTexture2D<uint> packed_mips[8];
[[vk::binding(1, 1)]] Texture2D<uint> packed;
/// Half float: occlusion comes in 1/16 steps, which it holds exactly.
[[vk::binding(2, 1)]] [[vk::image_format("r16f")]] RWTexture2D<float> raw_out;
[[vk::binding(3, 1)]] Texture2D<float> raw;
[[vk::binding(4, 1)]] [[vk::image_format("r32f")]] RWTexture2D<float> ao_out;
/// Nearest, with a zero border: off-screen samples read as sky.
[[vk::binding(5, 1)]] SamplerState int_border;
[[vk::binding(6, 1)]] SamplerState float_border;

/// CORTAO's occluder table: rays each occluder covers, by direction and size.
/// Must match ao.rs's `LUT_SIZE`.
#define AO_LUT_SIZE 32
[[vk::binding(7, 1)]] [[vk::image_format("r32ui")]] RWTexture3D<uint> occluder_lut_out;
[[vk::binding(8, 1)]] Texture3D<uint> occluder_lut;

int2 viewport_offset() {
    return int2(globals.viewport.xy);
}
