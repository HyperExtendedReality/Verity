// Visibility pass: rasterize depth plus (instance, triangle) IDs only. One
// draw per instance; vertices are pulled through its index range. Draws are
// non-indexed so the triangle ID is VertexID / 3 without needing PrimitiveID
// (a geometry-shader capability many mobile GPUs lack).

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
VertexOutput vertexMain(uint vertex: SV_VertexID) {
    Instance instance = instances[draw.instance];
    Vertex v = instance_vertex(instance, vertex);
    VertexOutput output;
    output.position = mul(globals.view_proj, mul(instance.model, float4(v.position.xyz, 1.0)));
    output.uv = float2(v.position.w, v.normal.w);
    output.alpha = v.color.a;
    output.id = uint2(draw.instance, vertex / 3);
    return output;
}

[shader("fragment")]
uint2 fragmentMain(VertexOutput input): SV_Target0 {
    Material material = materials[instances[draw.instance].draw.z];
    if ((material.textures.w & ALPHA_MASK) != 0) {
        float alpha = material.base_color.a * input.alpha;
        if (material.textures.x != NO_TEXTURE)
            alpha *= textures[NonUniformResourceIndex(material.textures.x)].Sample(material_sampler, input.uv).a;
        if (alpha < material.params.w)
            discard;
    }
    return input.id;
}
