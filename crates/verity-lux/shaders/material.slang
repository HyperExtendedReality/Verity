// Hit materials through Verity's bindless materials (`materials.slang`,
// bound at MATERIAL_SET). Instance word 7.w is the material index. Lux slots:
// 0 base color, 2 metallic-roughness, 3 normal; others have no texture.

#include "terrain.slang"

Material hitMaterial(Hit hit) { return materials[sceneWord(instanceBase(hit.instance) + 7).w]; }
// Kept for callers that only need an identity for the hit's material.
uint materialBase(Hit hit) { return sceneWord(instanceBase(hit.instance) + 7).w; }

uint materialTexture(Hit hit, uint slot) {
    uint4 textures = hitMaterial(hit).textures;
    return slot == 0 ? textures.x : (slot == 2 ? textures.y : (slot == 3 ? textures.z : EMPTY));
}

float2 materialUv(Hit hit, uint slot) { return hitVertexAttribute(hit, 3).xy; }

float4 sampleTexture(uint texture, float2 uv, float lod = 0) {
    if (texture == EMPTY) return 1;
    if (!isfinite(lod) || lod < 0.0) lod = 0;
    return textures[NonUniformResourceIndex(texture)].SampleLevel(material_sampler, uv, lod);
}

// Mip level from the ray footprint and the triangle's UV density.
float textureLod(Hit hit, uint slot, uint texture) {
    if (hit.footprint <= 0) return 0;
    uint4 geometryHeader = geometryWord(0);
    uint3 vertices = triangleIndices(hit.triangle, geometryHeader);
    uint instance = instanceBase(hit.instance) + 3;
    float4 row0 = asfloat(sceneWord(instance));
    float4 row1 = asfloat(sceneWord(instance + 1));
    float4 row2 = asfloat(sceneWord(instance + 2));
    float2 uv[3]; float3 positions[3];
    for (uint i = 0; i < 3; ++i) {
        uint vertex = vertexBase(vertices[i], geometryHeader);
        uv[i] = asfloat(geometryWord(vertex + 3)).xy;
        float4 position = float4(asfloat(geometryWord(vertex).xyz), 1);
        positions[i] = float3(dot(row0, position), dot(row1, position), dot(row2, position));
    }
    float3 a = positions[1] - positions[0], b = positions[2] - positions[0];
    float3 crossEdges = cross(a, b);
    float areaSquared = dot(crossEdges, crossEdges);
    if (areaSquared <= 0 || !isfinite(areaSquared)) return 0;
    float3 gradientU = cross(b, crossEdges) / areaSquared;
    float3 gradientV = cross(crossEdges, a) / areaSquared;
    float2 deltaA = uv[1] - uv[0], deltaB = uv[2] - uv[0];
    uint width, height, levels;
    textures[NonUniformResourceIndex(texture)].GetDimensions(0, width, height, levels);
    float texels = hit.footprint * max(length(gradientU * deltaA.x + gradientV * deltaB.x) * float(width),
                                      length(gradientU * deltaA.y + gradientV * deltaB.y) * float(height));
    return texels > 0 ? max(0.0, log2(texels)) : 0;
}

float4 materialSample(Hit hit, uint slot) {
#ifdef GI_WARMUP
    return 1;
#else
    uint texture = materialTexture(hit, slot);
    if (texture == EMPTY) return 1;
    return sampleTexture(texture, materialUv(hit, slot), textureLod(hit, slot, texture));
#endif
}

float4 materialBaseColor(Hit hit) {
#ifndef GI_WARMUP
    Material material = hitMaterial(hit);
    if ((material.textures.w & TERRAIN_SURFACE) != 0) {
        // Retained vertices are in world space.
        float3 position = hitVertexAttribute(hit, 0).xyz;
        float3 normal = normalize(hitVertexAttribute(hit, 1).xyz);
        float2 ground = hitVertexAttribute(hit, 3).xy;
        return float4(terrain_albedo(position, normal, ground.x, ground.y, material.params.w), 1.0);
    }
#endif
    return hitMaterial(hit).base_color * hitVertexAttribute(hit, 4) * materialSample(hit, 0);
}

float materialPerceptualRoughness(Hit hit) {
    float roughness = hitMaterial(hit).params.y * materialSample(hit, 2).g;
    return isfinite(roughness) ? saturate(roughness) : 1.0;
}

float materialMetallic(Hit hit) {
    float metallic = hitMaterial(hit).params.x * materialSample(hit, 2).b;
    return isfinite(metallic) ? saturate(metallic) : 0.0;
}

// Opaque uses -1, while -2 denotes fractional alpha coverage. Other values
// are the binary mask cutoff.
float materialAlphaCutoff(Hit hit) {
    Material material = hitMaterial(hit);
    uint flags = material.textures.w;
    if ((flags & ALPHA_BLEND) != 0) return -2.0;
    return (flags & ALPHA_MASK) != 0 ? material.params.w : -1.0;
}

float materialCoverage(Hit hit, float cutoff) {
    if (cutoff == -1.0) return 1.0;
    float alpha = hitMaterial(hit).base_color.w * hitVertexAttribute(hit, 4).w
        * materialSample(hit, 0).w;
    if (cutoff == -2.0) return isfinite(alpha) ? saturate(alpha) : 0.0;
    return alpha >= cutoff ? 1.0 : 0.0;
}

bool materialAccepts(Hit hit, float cutoff) {
    float coverage = materialCoverage(hit, cutoff);
    if (coverage <= 0.0) return false;
    if (coverage >= 1.0) return true;
    // A per-intersection, per-frame Bernoulli sample estimates fractional
    // coverage without forcing an alpha threshold on blended materials.
    int4 key = int4(asint(hit.barycentric.y), asint(hit.barycentric.z),
        int(hit.triangle), int(hit.instance ^ (params.frame.x * 0x9e3779b9u)));
    float sample = float(hashKey(key, 0x1000000u)) * (1.0 / 16777216.0);
    return sample < coverage;
}

float3 materialNormal(Hit hit, HitAttributes attributes, float3 incoming) {
    bool back = dot(attributes.geometricNormal, incoming) > 0;
    bool doubleSided = (sceneWord(instanceBase(hit.instance) + 6).w & 8) != 0;
    float3 tangent = attributes.worldTangent.xyz;
    // Meshes without tangents keep the interpolated normal.
    if (materialTexture(hit, 3) == EMPTY || !(dot(tangent, tangent) > 1e-12)) {
        float3 normal = unitVector(attributes.worldNormal, attributes.geometricNormal);
        return back && doubleSided ? -normal : normal;
    }
    float3 mapped = materialSample(hit, 3).xyz * 2 - 1;
    mapped.xy *= hitMaterial(hit).params.z;
    if (back && doubleSided) mapped = -mapped;
    // MikkTSpace retains interpolated lengths and does not orthogonalize TBN.
    float3 bitangent = attributes.worldTangent.w * cross(attributes.worldNormal, tangent);
    return unitVector(tangent * mapped.x + bitangent * mapped.y + attributes.worldNormal * mapped.z,
                      attributes.geometricNormal);
}
