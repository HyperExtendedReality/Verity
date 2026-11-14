// Visibility-buffer resolve: rebuild each pixel's surface from its triangle,
// then shade it with the sun and clustered lights into the HDR target.

#include "lib/common.slang"
#include "lib/matrix.slang"
#include "lib/brdf.slang"
#include "lib/gi_surface.slang"
#include "lib/terrain.slang"
#include "lib/terrain_layers.slang"


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
/// Sunlight let through the clouds above `position`; see clouds.slang.
[[vk::binding(32, MATERIAL_SET)]] Texture2D<float> cloud_shadow_map;

float cloud_shadow(float3 position) {
    if (globals.cloud_shadow.w == 0.0)
        return 1.0;
    float3 to_sun = -globals.sun_direction.xyz;
    to_sun.z = max(to_sun.z, 0.08);
    float2 ground = position.xy - to_sun.xy * (position.z / to_sun.z);
    float2 uv = (ground - globals.cloud_shadow.xy) / globals.cloud_shadow.z;
    if (any(uv < 0.0) || any(uv > 1.0))
        return 1.0;
    return cloud_shadow_map.SampleLevel(material_sampler, uv, 0);
}

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


/// Empty pixel: alpha 0 lets the composite keep the swapchain clear color.
void clear_pixel(uint2 pixel) {
    hdr_output[pixel] = 0.0;
    gi_motion[pixel] = 0.0;
    if (globals.counts.z != 0) {
        gi_gbuffer[pixel] = 0;
        gi_receiver[pixel] = 0;
    }
}

/// Rebuild and light the surface `hit` seen through viewport pixel `id`.
/// Only the `primary` surface writes the G-buffer, receiver and motion.
float3 shade(uint2 id, uint2 pixel, uint2 hit, bool primary) {
    float4 viewport = globals.viewport;
    uint kind = hit.y >> MATRIX_KIND_SHIFT;
    bool is_matrix = kind != 0;
    MatrixInstance matrix_instance;
    Instance instance;
    if (is_matrix) {
        matrix_instance = globals.matrix_instances[hit.x];
        instance = matrix_instance.base;
    } else {
        instance = globals.instances[hit.x];
    }
    Material material = materials[instance.draw.z];
    float3x3 model = (float3x3)instance.model;
    float3 eye = globals.eye.xyz;
    float2 center = float2(id.xy) + 0.5;
    float3 direction = pixel_ray(center);
    Surface s;
    s.view = -direction;
    s.agreement = 1.0;
    float3 face;
    uint level = uint(instance.normal_columns[0].w);
    // Where this surface was last frame, for soft bodies and cloth.
    float3 previous_position = 0.0;
    float3 emission = 0.0;
    bool vertex_motion = !is_matrix && globals.instance_motion[hit.x].info.x != 0;

    if (kind == MATRIX_VOXEL) {
        uint id_bits = hit.y & MATRIX_ID_MASK;
        MatrixBrick brick = globals.matrix_bricks[id_bits >> MATRIX_VOXEL_BITS];
        uint bit = id_bits & ((1u << MATRIX_VOXEL_BITS) - 1);
        uint2 voxel = globals.matrix_voxels[brick.first_voxel + voxel_rank(brick, bit)];
        // Enter the voxel's box along the camera ray, in object space.
        float3 origin = mul(matrix_instance.inverse, float4(eye, 1.0)).xyz;
        float3 ray = normalize(mul(matrix_instance.inverse, float4(direction, 0.0)).xyz);
        float3 low = brick.origin + float3(bit & 3, (bit >> 2) & 3, bit >> 4) * brick.voxel_size;
        float3 safe = select(abs(ray) < 1.0e-6, float3(1.0e-6), ray);
        float3 t0 = (low - origin) / safe;
        float3 t1 = (low + brick.voxel_size - origin) / safe;
        float3 near = min(t0, t1);
        float enter = max(near.x, max(near.y, near.z));
        float3 side = near.x >= near.y && near.x >= near.z ? float3(1, 0, 0)
                    : near.y >= near.z ? float3(0, 1, 0) : float3(0, 0, 1);
        side *= -sign(ray);
        s.position = mul(instance.model, float4(origin + ray * enter, 1.0)).xyz;
        float3 object_normal = decode_octahedral(voxel.y);
        float3 normal = instance.normal_columns[0].xyz * object_normal.x
            + instance.normal_columns[1].xyz * object_normal.y
            + instance.normal_columns[2].xyz * object_normal.z;
        s.normal = normalize(normal);
        face = normalize(instance.normal_columns[0].xyz * side.x
            + instance.normal_columns[1].xyz * side.y
            + instance.normal_columns[2].xyz * side.z);
        if (dot(s.normal, s.view) < 0.0)
            s.normal = -s.normal;
        s.base_color = srgb_to_linear(float3(voxel.x & 255, (voxel.x >> 8) & 255, (voxel.x >> 16) & 255) / 255.0);
        s.metallic = material.params.x;
        // Normals that disagree inside a voxel act like a rougher surface.
        s.roughness = lerp(material.params.y, 1.0, float(voxel.x >> 24) / 255.0);
        level = 3;
    } else if (kind == MATRIX_IMPOSTER) {
        MatrixAsset asset = globals.matrix_assets[matrix_instance.info.x];
        ImposterHit imposter = imposter_sample(matrix_instance, asset, eye, direction, matrix_pixels(matrix_instance));
        float3 origin = mul(matrix_instance.inverse, float4(eye, 1.0)).xyz;
        float3 ray = normalize(mul(matrix_instance.inverse, float4(direction, 0.0)).xyz);
        s.position = mul(instance.model, float4(origin + ray * imposter.distance, 1.0)).xyz;
        float3 normal = instance.normal_columns[0].xyz * imposter.normal.x
            + instance.normal_columns[1].xyz * imposter.normal.y
            + instance.normal_columns[2].xyz * imposter.normal.z;
        s.normal = normalize(normal);
        face = s.normal;
        s.base_color = imposter.albedo;
        s.metallic = material.params.x;
        s.roughness = material.params.y;
        // A hemisphere of facets has a mean half a unit long.
        s.agreement = saturate(imposter.mean_length * 2.0 - 1.0);
        level = 4;
    } else {
        uint triangle = hit.y;
        MatrixCluster cluster;
        if (is_matrix) {
            uint id_bits = hit.y & MATRIX_ID_MASK;
            cluster = globals.matrix_clusters[id_bits >> MATRIX_TRIANGLE_BITS];
            triangle = id_bits & ((1u << MATRIX_TRIANGLE_BITS) - 1);
            level = min(cluster.depth, 2);
        }
        float3 p[3];
        float3 n[3];
        float3 t[3];
        float2 uv[3];
        float4 color[3];
        float tangent_sign = 1.0;
        for (uint k = 0; k < 3; ++k) {
            Vertex v = is_matrix ? matrix_vertex(matrix_instance, cluster.first_index, triangle * 3 + k)
                                 : instance_vertex(instance, triangle * 3 + k);
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
        if (any(material.emission > 0.0))
            emission = material.emission * sample_material(material.emission_texture, texcoord, dx, dy, 1.0).rgb;

        s.position = eye + direction * b.z;
        if (vertex_motion) {
            uint first = globals.instance_motion[hit.x].info.y;
            float3 q[3];
            for (uint k = 0; k < 3; ++k)
                q[k] = globals.previous_positions[first + globals.indices[instance.draw.x + triangle * 3 + k]].xyz;
            previous_position = interpolate(q[0], q[1], q[2], b);
        }
        float3 normal = normalize(interpolate(n[0], n[1], n[2], b));
        face = normalize(cross(p[1] - p[0], p[2] - p[0]));
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
        if ((material.textures.w & TERRAIN_SURFACE) != 0) {
            float3 across = interpolate(p[0], p[1], p[2], bx) - s.position;
            float3 along = interpolate(p[0], p[1], p[2], by) - s.position;
            float footprint = sqrt(max(dot(across, across), dot(along, along)));
            TerrainSurface ground;
            if (terrain_layers_ready()) {
                // A sixteenth of the pixels ask for pages each frame, a different sixteenth each time.
                uint frame = layer_table[0];
                bool feedback = (pixel.x & 3) == (frame & 3) && (pixel.y & 3) == ((frame >> 2) & 3);
                ground = terrain_layered(
                    s.position, s.normal, texcoord.x, texcoord.y, footprint, material.params.w, feedback);
            } else {
                ground = terrain_surface(
                    s.position, s.normal, texcoord.x, texcoord.y, footprint, material.params.w, material.params.z);
            }
            s.base_color = ground.albedo;
            s.roughness = ground.roughness;
            // Bumps can tilt a grazing normal past the viewer, where the specular term blows up.
            s.normal = normalize(ground.normal + s.view * max(0.05 - dot(ground.normal, s.view), 0.0));
        }
    }
    float nv = max(dot(s.normal, s.view), 0.0001);
    float2 dfg = env_brdf(nv, s.roughness);
    Brdf brdf = make_brdf(s.base_color, s.metallic, s.roughness, nv, dfg);
    // Diffuse transmission (KHR_materials_diffuse_transmission): that share of
    // the diffuse light passes through instead of reflecting, and leaves the
    // far side as a Lambertian lobe in the transmission color. `transmitted`
    // is radiance per unit irradiance arriving from behind.
    float transmits = saturate(material.transmission.w) * (1.0 - saturate(s.metallic));
    brdf.diffuse_color *= 1.0 - transmits;
    float3 transmitted = max(material.transmission.rgb, 0.0) * transmits / PI;

    if (primary && globals.counts.z != 0) {
        // GI reflects off the surface's diffuse albedo, which transmission lowers.
        // shortcut: GI arriving at the far side does not pass through to the viewer; add a back-facing GI lookup when foliage looks too dark under GI.
        gi_gbuffer[pixel] = pack_gbuffer(s.base_color * (1.0 - transmits), s.roughness, s.metallic, s.normal);
        gi_receiver[pixel] = pack_receiver(instance.draw.w, face);
    }
    // Camera and object motion, for GI and TAA. Matrix instances are placed at
    // load and never move; deforming vertices carry their own last positions.
    float4 moved = float4(s.position, 1.0);
    if (vertex_motion)
        moved.xyz = previous_position;
    else if (!is_matrix)
        moved = mul(globals.instance_motion[hit.x].to_previous, moved);
    float4 previous = mul(globals.previous_view_proj, moved);
    float2 previous_uv = (previous.xy / previous.w) * 0.5 + 0.5;
    if (primary)
        gi_motion[pixel] = (center / viewport.zw) - previous_uv;

    // With GI, indirect light arrives from the extension instead.
    float depth = -mul(globals.view, float4(s.position, 1.0)).z;
    // Hemisphere ambient through the same diffuse and specular albedo as
    // direct light. With GI, indirect light arrives from the extension instead.
    float3 result = 0.0;
    if (globals.counts.z == 0) {
        float3 hemisphere = lerp(globals.ground.rgb, globals.sky.rgb, s.normal.z * 0.5 + 0.5);
        result = (brdf.diffuse_color + specular_albedo(brdf.f0, dfg)) * hemisphere;
        // The hemisphere behind the surface, through it.
        result += transmitted * PI * lerp(globals.ground.rgb, globals.sky.rgb, -s.normal.z * 0.5 + 0.5);
    } else if ((globals.counts.z & 2) != 0) {
        float3 mirrored = reflect(-s.view, s.normal);
        result = specular_albedo(brdf.f0, dfg) * lerp(globals.ground.rgb, globals.sky.rgb, mirrored.z * 0.5 + 0.5);
    }
    if (globals.sun_direction.w > 0.0) {
        float3 l = -globals.sun_direction.xyz;
        result += evaluate_brdf(brdf, s, l) * globals.sun_color.rgb * sun_shadow(s.position, face, depth) * cloud_shadow(s.position);
        // Sun behind a leaf, shadowed on the side it lights.
        float behind = -dot(s.normal, l);
        if (transmits > 0.0 && behind > 0.0)
            result += transmitted * behind * globals.sun_color.rgb * sun_shadow(s.position, -face, depth) * cloud_shadow(s.position);
    }
    uint2 tile = min(uint2(float2(id.xy) / viewport.zw * float2(CLUSTER_X, CLUSTER_Y)), uint2(CLUSTER_X - 1, CLUSTER_Y - 1));
    uint first = cluster_index(uint3(tile, depth_slice(depth))) * LIGHT_WORDS;
    for (uint word = 0; word < LIGHT_WORDS; ++word) {
        uint bits = clusters[first + word];
        while (bits != 0) {
            uint i = word * 32 + firstbitlow(bits);
            bits &= bits - 1;
            Light light = globals.lights[i];
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
            float behind = -dot(s.normal, l);
            if (transmits > 0.0 && behind > 0.0)
                result += transmitted * behind * light.color_intensity.rgb * light.color_intensity.a * falloff
                    * point_shadow(i, light, s.position, -face);
        }
    }
    result += emission;
    if (globals.counts.w != 0) {
        float3 colors[5] = {
            float3(0.1, 0.8, 0.2), float3(0.1, 0.5, 1.0),
            float3(1.0, 0.8, 0.1), float3(1.0, 0.3, 0.1),
            float3(0.8, 0.1, 0.9)
        };
        result = colors[min(level, 4)];
    }
    return result;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void resolveMain(uint3 id: SV_DispatchThreadID) {
    float4 viewport = globals.viewport;
    if (any(float2(id.xy) >= viewport.zw))
        return;
    uint2 pixel = id.xy + uint2(viewport.xy);
    uint2 hit = visibility.Load(int3(pixel, 0));
    if (hit.x == NO_INSTANCE)
        clear_pixel(pixel);
    else
        hdr_output[pixel] = float4(shade(id.xy, pixel, hit, true), 1.0);
}

/// Every sample of the multisampled visibility pass; `visibility` holds sample 0.
[[vk::binding(34)]] Texture2DMS<uint2> visibility_samples;
static const uint MSAA_SAMPLES = 4;

/// Weight that keeps a bright sample from dominating an edge (Karis 2014).
float edge_weight(float3 color, uint samples) {
    return float(samples) / (1.0 + max(color.r, max(color.g, color.b)));
}

/// The resolve with 4x MSAA visibility: shade sample 0's triangle and the
/// first other triangle in the pixel, blended by how many samples each covers.
/// Sample 0's surface writes the single-sample outputs, matching the depth
/// resolved from sample 0.
[shader("compute")]
[numthreads(8, 8, 1)]
void resolveMsaaMain(uint3 id: SV_DispatchThreadID) {
    float4 viewport = globals.viewport;
    if (any(float2(id.xy) >= viewport.zw))
        return;
    uint2 pixel = id.xy + uint2(viewport.xy);
    uint2 hit = visibility.Load(int3(pixel, 0));
    // shortcut: an edge against the sky is not blended, since the sky is drawn
    // later into empty pixels; TAA still smooths those edges.
    if (hit.x == NO_INSTANCE) {
        clear_pixel(pixel);
        return;
    }
    uint own = 0;
    uint2 other = hit;
    uint others = 0;
    for (uint s = 0; s < MSAA_SAMPLES; ++s) {
        uint2 h = visibility_samples.Load(int2(pixel), s);
        if (all(h == hit)) {
            ++own;
        } else if (h.x != NO_INSTANCE) {
            if (others == 0)
                other = h;
            if (all(h == other))
                ++others;
        }
    }
    float3 color = shade(id.xy, pixel, hit, true);
    if (others != 0) {
        float3 second = shade(id.xy, pixel, other, false);
        float a = edge_weight(color, own);
        float b = edge_weight(second, others);
        color = (color * a + second * b) / (a + b);
    }
    hdr_output[pixel] = float4(color, 1.0);
}
