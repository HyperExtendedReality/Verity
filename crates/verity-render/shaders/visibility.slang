// Visibility pass: rasterize depth plus (instance, triangle) IDs only.
// Non-indexed draws use primitive-local firstVertex for meshlet ranges, so
// VertexID / 3 remains the original triangle ID without PrimitiveID (which
// requires a geometry-shader capability many mobile GPUs lack).
// SV_VertexID would subtract firstVertex again: this needs the raw index.

#include "lib/common.slang"

struct DrawConstants {
    uint instance;
};
[[vk::push_constant]] ConstantBuffer<DrawConstants> draw;

struct VertexOutput {
    float4 position: SV_Position;
    float2 uv: TEXCOORD0;
    float alpha: TEXCOORD1;
    nointerpolation uint2 id: ID;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertex: SV_VulkanVertexID) {
    Instance instance = globals.instances[draw.instance];
    Vertex v = instance_vertex(instance, vertex);
    VertexOutput output;
    output.position = mul(globals.view_proj, mul(instance.model, float4(v.position.xyz, 1.0)));
    output.uv = float2(v.position.w, v.normal.w);
    output.alpha = v.color.a;
    output.id = uint2(draw.instance, vertex / 3);
    return output;
}

void alphaTest(VertexOutput input) {
    Material material = materials[globals.instances[draw.instance].draw.z];
    if ((material.textures.w & ALPHA_MASK) != 0) {
        float alpha = material.base_color.a * input.alpha;
        if (material.textures.x != NO_TEXTURE)
            alpha *= textures[NonUniformResourceIndex(material.textures.x)].Sample(material_sampler, input.uv).a;
        if (alpha < material.params.w)
            discard;
    }
}

// The depth prepass has no color attachment but must use the same alpha test.
#if defined(DEPTH_ONLY)
[shader("fragment")]
void fragmentMain(VertexOutput input) {
    alphaTest(input);
}
#else
[shader("fragment")]
uint2 fragmentMain(VertexOutput input): SV_Target0 {
    alphaTest(input);
    return input.id;
}
#endif
