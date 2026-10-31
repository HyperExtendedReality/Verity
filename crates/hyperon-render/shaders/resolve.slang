// Visibility-buffer resolve: rebuild each pixel's surface from its triangle,
// then shade it with the sun and clustered lights into the HDR target.

#include "lib/common.slang"
#include "lib/brdf.slang"
#include "lib/gi_surface.slang"


/// Barycentrics (b1, b2) and distance of a ray hit on a triangle's plane;
/// edge-on triangles fall back to the centroid.
float3 intersect(float3 origin, float3 direction, float3 p0, float3 p1, float3 p2) {
    float3 e1 = p1 - p0;
    float3 e2 = p2 - p0;
    float3 q = cross(direction, e2);
    float det = dot(e1, q);
    if (abs(det) < 1.0e-12)
        return float3(1.0 / 3.0, 1.0 / 3.0, dot(p0 - origin, direction));
    float3 s = origin - p0;
    float3 r = cross(s, e1);
    return float3(dot(s, q), dot(direction, r), dot(e2, r)) / det;
}

/// World-space direction of the camera ray through a viewport pixel position.
float3 pixel_ray(float2 pixel) {
    // Reverse-Z: NDC depth 1 is the near plane.
    float2 ndc = pixel / globals.viewport.zw * 2.0 - 1.0;
    float4 near_point = mul(globals.inverse_view_proj, float4(ndc, 1.0, 1.0));
    return normalize(near_point.xyz / near_point.w - globals.eye.xyz);
}

float2 interpolate(float2 a, float2 b, float2 c, float3 bary) {
    return a * (1.0 - bary.x - bary.y) + b * bary.x + c * bary.y;
}

float3 interpolate(float3 a, float3 b, float3 c, float3 bary) {
    return a * (1.0 - bary.x - bary.y) + b * bary.x + c * bary.y;
}

float4 sample_material(uint texture, float2 uv, float2 dx, float2 dy, float4 fallback) {
    if (texture == NO_TEXTURE)
        return fallback;
    return textures[NonUniformResourceIndex(texture)].SampleGrad(material_sampler, uv, dx, dy);
}

/// Sun visibility from the cascade covering view distance `depth`, with a
/// 3x3 hardware-PCF filter and a fade over the last tenth of the range.
float sun_shadow(float3 position, float3 normal, float depth) {
    float range = globals.shadow_splits[CASCADES - 1];
    if (globals.sky.w == 0.0 || depth >= range)
        return 1.0;
    uint cascade = 0;
    while (cascade < CASCADES - 1 && depth > globals.shadow_splits[cascade])
        ++cascade;
    // Offsetting along the normal by a texel or two keeps acne off grazing surfaces.
    float3 biased = position + normal * globals.shadow_texels[cascade] * 1.5;
    float4 clip = mul(globals.shadow_view_proj[cascade], float4(biased, 1.0));
    float3 ndc = clip.xyz / clip.w;
    float2 uv = ndc.xy * 0.5 + 0.5;
    float lit = 0.0;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
            lit += shadow_map.SampleCmpLevelZero(shadow_sampler, float3(uv + float2(x, y) / SHADOW_SIZE, cascade), ndc.z);
    return lerp(1.0, lit / 9.0, saturate((range - depth) / (0.1 * range)));
}
// The first four lights have six depth-array layers each. Unshadowed lights
// still contribute direct lighting; the face projection matches the raster pass.
float point_shadow(uint index, Light light, float3 position, float3 normal) {
    if (index >= MAX_POINT_SHADOWS || light.position_radius.w <= 0.0)
        return 1.0;
    float3 offset = position - light.position_radius.xyz;
    float3 a = abs(offset);
    uint face = a.x >= a.y && a.x >= a.z ? (offset.x >= 0.0 ? 0 : 1)
              : a.y >= a.z ? (offset.y >= 0.0 ? 2 : 3)
              : (offset.z >= 0.0 ? 4 : 5);
    uint layer = index * POINT_SHADOW_FACES + face;
    float3 biased = position + normal * length(offset) * (1.5 / POINT_SHADOW_SIZE);
    float4 clip = mul(globals.point_shadow_view_proj[layer], float4(biased, 1.0));
    if (clip.w <= 0.0)
        return 1.0;
    float3 ndc = clip.xyz / clip.w;
    if (ndc.z <= 0.0 || ndc.z >= 1.0)
        return 1.0;
    float2 uv = ndc.xy * 0.5 + 0.5;
    float lit = 0.0;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
            lit += point_shadow_map.SampleCmpLevelZero(
                shadow_sampler, float3(uv + float2(x, y) / POINT_SHADOW_SIZE, layer), ndc.z);
    return lit / 9.0;
}


[shader("compute")]
[numthreads(8, 8, 1)]
void resolveMain(uint3 id: SV_DispatchThreadID) {
    float4 viewport = globals.viewport;
    if (any(float2(id.xy) >= viewport.zw))
        return;
    uint2 pixel = id.xy + uint2(viewport.xy);
    uint2 hit = visibility.Load(int3(pixel, 0));
    if (hit.x == NO_INSTANCE) {
        // Alpha 0 lets the composite keep the swapchain clear color.
        hdr_output[pixel] = 0.0;
        if (globals.counts.z != 0) {
            gi_gbuffer[pixel] = 0;
            gi_receiver[pixel] = 0;
            gi_motion[pixel] = 0.0;
        }
        return;
    }

    Instance instance = instances[hit.x];
    Material material = materials[instance.draw.z];
    float3x3 model = (float3x3)instance.model;
    float3 p[3];
    float3 n[3];
    float3 t[3];
    float2 uv[3];
    float4 color[3];
    float tangent_sign = 1.0;
    for (uint k = 0; k < 3; ++k) {
        Vertex v = instance_vertex(instance, hit.y * 3 + k);
        p[k] = mul(instance.model, float4(v.position.xyz, 1.0)).xyz;
        n[k] = instance.normal_columns[0].xyz * v.normal.x
            + instance.normal_columns[1].xyz * v.normal.y
            + instance.normal_columns[2].xyz * v.normal.z;
        t[k] = mul(model, v.tangent.xyz);
        uv[k] = float2(v.position.w, v.normal.w);
        color[k] = v.color;
        tangent_sign = v.tangent.w < 0.0 ? -1.0 : 1.0;
    }

    // Barycentrics here and one pixel right and down give exact UV gradients.
    float3 eye = globals.eye.xyz;
    float2 center = float2(id.xy) + 0.5;
    float3 direction = pixel_ray(center);
    float3 b = intersect(eye, direction, p[0], p[1], p[2]);
    float3 bx = intersect(eye, pixel_ray(center + float2(1, 0)), p[0], p[1], p[2]);
    float3 by = intersect(eye, pixel_ray(center + float2(0, 1)), p[0], p[1], p[2]);
    float2 texcoord = interpolate(uv[0], uv[1], uv[2], b);
    float2 dx = interpolate(uv[0], uv[1], uv[2], bx) - texcoord;
    float2 dy = interpolate(uv[0], uv[1], uv[2], by) - texcoord;

    float4 base = material.base_color
        * (color[0] * (1.0 - b.x - b.y) + color[1] * b.x + color[2] * b.y)
        * sample_material(material.textures.x, texcoord, dx, dy, 1.0);
    // glTF packs roughness in green and metalness in blue.
    float4 metal_rough = sample_material(material.textures.y, texcoord, dx, dy, 1.0);

    Surface s;
    s.position = eye + direction * b.z;
    s.view = -direction;
    float3 normal = normalize(interpolate(n[0], n[1], n[2], b));
    float3 face = normalize(cross(p[1] - p[0], p[2] - p[0]));
    // Two-sided: shade the side facing the camera.
    if (dot(face, s.view) < 0.0) {
        normal = -normal;
        face = -face;
    }
    float normal_variance = 0.0;
    if (material.textures.z != NO_TEXTURE) {
        float3 tangent = interpolate(t[0], t[1], t[2], b);
        if (dot(tangent, tangent) < 1.0e-12) {
            // No authored tangents: derive one from the triangle's UV layout.
            float2 duv1 = uv[1] - uv[0];
            float2 duv2 = uv[2] - uv[0];
            float det = duv1.x * duv2.y - duv2.x * duv1.y;
            tangent = abs(det) > 1.0e-12 ? ((p[1] - p[0]) * duv2.y - (p[2] - p[0]) * duv1.y) / det : float3(1, 0, 0);
            tangent_sign = det < 0.0 ? -1.0 : 1.0;
        }
        tangent = normalize(tangent - normal * dot(normal, tangent));
        float3 bitangent = cross(normal, tangent) * tangent_sign;
        float3 local = sample_material(material.textures.z, texcoord, dx, dy, 0.5).xyz * 2.0 - 1.0;
        // Toksvig: filtered mips shorten the normal where it varies within the
        // footprint; widen the lobe by that variance to stop specular sparkle.
        float length_n = clamp(length(local), 0.01, 1.0);
        normal_variance = (1.0 - length_n) / length_n;
        local.xy *= material.params.z;
        normal = normalize(tangent * local.x + bitangent * local.y + normal * local.z);
    }
    s.normal = normal;
    s.base_color = base.rgb;
    s.metallic = material.params.x * metal_rough.b;
    float roughness = material.params.y * metal_rough.g;
    // Add the variance to alpha² and convert back to perceptual roughness.
    s.roughness = sqrt(sqrt(min(pow(roughness, 4.0) + 2.0 * normal_variance, 1.0)));
    float nv = max(dot(s.normal, s.view), 0.0001);
    float2 dfg = env_brdf(nv, s.roughness);
    Brdf brdf = make_brdf(s.base_color, s.metallic, s.roughness, nv, dfg);

    if (globals.counts.z != 0) {
        gi_gbuffer[pixel] = pack_gbuffer(s.base_color, s.roughness, s.metallic, s.normal);
        gi_receiver[pixel] = pack_receiver(instance.draw.w, face);
        float4 previous = mul(globals.previous_view_proj, float4(s.position, 1.0));
        float2 previous_uv = (previous.xy / previous.w) * 0.5 + 0.5;
        gi_motion[pixel] = (center / viewport.zw) - previous_uv;
    }

    // With GI, indirect light arrives from the extension instead.
    float depth = -mul(globals.view, float4(s.position, 1.0)).z;
    // Hemisphere ambient through the same diffuse and specular albedo as
    // direct light. With GI, indirect light arrives from the extension instead.
    float3 result = 0.0;
    if (globals.counts.z == 0) {
        float3 hemisphere = lerp(globals.ground.rgb, globals.sky.rgb, s.normal.y * 0.5 + 0.5);
        result = (brdf.diffuse_color + specular_albedo(brdf.f0, dfg)) * hemisphere;
    }
    if (globals.sun_direction.w > 0.0) {
        float3 l = -globals.sun_direction.xyz;
        result += evaluate_brdf(brdf, s, l) * globals.sun_color.rgb * sun_shadow(s.position, face, depth);
    }
    uint2 tile = min(uint2(float2(id.xy) / viewport.zw * float2(CLUSTER_X, CLUSTER_Y)), uint2(CLUSTER_X - 1, CLUSTER_Y - 1));
    uint first = cluster_index(uint3(tile, depth_slice(depth))) * LIGHT_WORDS;
    for (uint word = 0; word < LIGHT_WORDS; ++word) {
        uint bits = clusters[first + word];
        while (bits != 0) {
            uint i = word * 32 + firstbitlow(bits);
            bits &= bits - 1;
            Light light = lights[i];
            float3 delta = light.position_radius.xyz - s.position;
            float distance = length(delta);
            float radius = light.position_radius.w;
            if (distance >= radius)
                continue;
            float3 l = delta / max(distance, 0.0001);
            float window = pow(max(1.0 - pow(distance / radius, 4.0), 0.0), 2.0);
            float falloff = window / max(distance * distance, 0.01);
            result += evaluate_brdf(brdf, s, l) * light.color_intensity.rgb * light.color_intensity.a * falloff
                * point_shadow(i, light, s.position, face);
        }
    }
    hdr_output[pixel] = float4(result, 1.0);
}
