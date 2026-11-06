// Depth-only sun and point shadow pass; masked materials are alpha tested.

#include "lib/common.slang"

struct DrawConstants {
    uint instance;
    uint cascade;
};
[[vk::push_constant]] ConstantBuffer<DrawConstants> draw;

struct VertexOutput {
    float4 position: SV_Position;
    float2 uv: TEXCOORD0;
    float alpha: TEXCOORD1;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertex: SV_VertexID) {
    Instance instance = instances[draw.instance];
    Vertex v = instance_vertex(instance, vertex);
    VertexOutput output;
    float4 world = mul(instance.model, float4(v.position.xyz, 1.0));
    if (draw.cascade < CASCADES)
        output.position = mul(globals.shadow_view_proj[draw.cascade], world);
    else
        output.position = mul(globals.point_shadow_view_proj[draw.cascade - CASCADES], world);
    output.uv = float2(v.position.w, v.normal.w);
    output.alpha = v.color.a;
    return output;
}

[shader("fragment")]
void fragmentMain(VertexOutput input) {
    Material material = materials[instances[draw.instance].draw.z];
    if ((material.textures.w & ALPHA_MASK) != 0) {
        float alpha = material.base_color.a * input.alpha;
        if (material.textures.x != NO_TEXTURE)
            alpha *= textures[NonUniformResourceIndex(material.textures.x)].Sample(material_sampler, input.uv).a;
        if (alpha < material.params.w)
            discard;
    }
}
