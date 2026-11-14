// Bindless materials. The renderer binds them in its frame set (0); passes
// with their own set 0, such as GI, define MATERIAL_SET to the index where
// they bind the renderer's frame set. Layout mirrors `GpuMaterial` in scene.rs.

#ifndef MATERIAL_SET
#define MATERIAL_SET 0
#endif

static const uint NO_TEXTURE = 0xffffffff;
static const uint MAX_TEXTURES = 1024;
/// Material flag bits.
static const uint ALPHA_MASK = 1;
static const uint ALPHA_BLEND = 2;
static const uint TERRAIN_SURFACE = 4;

struct Material {
    float4 base_color;
    /// x: metallic, y: perceptual roughness, z: normal scale, w: alpha cutoff.
    float4 params;
    /// x: base color, y: metallic-roughness, z: normal texture; w: flags.
    uint4 textures;
    /// Diffuse transmission: rgb its color, w the share of diffuse light that
    /// passes through instead of reflecting (KHR_materials_diffuse_transmission).
    float4 transmission;
};

[[vk::binding(10, MATERIAL_SET)]] StructuredBuffer<Material> materials;
[[vk::binding(11, MATERIAL_SET)]] SamplerState material_sampler;
[[vk::binding(12, MATERIAL_SET)]] Texture2D<float4> textures[MAX_TEXTURES];
/// The split-sum DFG table, 16x16 over (N·V, perceptual roughness).
[[vk::binding(6, MATERIAL_SET)]] StructuredBuffer<float2> brdf_lut;

/// Bilinear DFG lookup: x scales F0, y is the F90 bias.
float2 env_brdf(float nv, float roughness) {
    float2 uv = clamp(float2(nv, roughness), 0.0, 1.0) * 15.0;
    int2 a = int2(floor(uv));
    int2 b = min(a + 1, int2(15));
    float2 f = frac(uv);
    float2 lo = lerp(brdf_lut[a.y * 16 + a.x], brdf_lut[a.y * 16 + b.x], f.x);
    float2 hi = lerp(brdf_lut[b.y * 16 + a.x], brdf_lut[b.y * 16 + b.x], f.x);
    return lerp(lo, hi, f.y);
}
