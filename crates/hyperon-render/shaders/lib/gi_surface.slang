// The thin G-buffer the resolve writes for screen-space and GI passes, in the
// packing hyperon-lux's shaders decode (the Bevy deferred layout):
//
// gbuffer  x: sRGB base color (gamma 2.2) in bits 0-23, perceptual roughness 24-31
//          z: reflectance 0-7, metallic 8-15, diffuse occlusion 16-23
//          w: octahedral shading normal, 12+12 bits; bit 24 unlit, bit 26 shadow receiver
// receiver xy: surface identity (draw + 1, 0); z: bit 31 valid, 15+15-bit
//          octahedral geometric normal; w: draw index
// motion   xy: current minus previous screen UV

float2 octahedral(float3 n) {
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    float2 p = n.xy;
    if (n.z < 0.0)
        p = (1.0 - abs(p.yx)) * float2(p.x >= 0.0 ? 1.0 : -1.0, p.y >= 0.0 ? 1.0 : -1.0);
    return p * 0.5 + 0.5;
}

uint4 pack_gbuffer(float3 base_color, float roughness, float metallic, float3 normal) {
    uint3 color = uint3(round(pow(saturate(base_color), 1.0 / 2.2) * 255.0));
    uint rough = uint(round(saturate(roughness) * 255.0));
    uint metal = uint(round(saturate(metallic) * 255.0));
    uint2 oct = uint2(round(saturate(octahedral(normal)) * 4095.0));
    // Reflectance 0.5 (F0 = 0.04) and full diffuse occlusion visibility.
    uint properties = 128u | (metal << 8) | (255u << 16);
    uint packed_normal = oct.x | (oct.y << 12) | (1u << 26);
    return uint4(color.r | (color.g << 8) | (color.b << 16) | (rough << 24), 0, properties, packed_normal);
}

uint4 pack_receiver(uint draw, float3 geometric_normal) {
    uint2 oct = uint2(round(saturate(octahedral(geometric_normal)) * 32767.0));
    return uint4(draw + 1, 0, 0x80000000u | oct.x | (oct.y << 15), draw);
}
