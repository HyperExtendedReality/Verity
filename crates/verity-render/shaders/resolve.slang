// Visibility-buffer resolve: rebuild each pixel's surface from its triangle,
// then shade it with the sun and clustered lights into the HDR target.

#include "lib/common.slang"
#include "lib/matrix.slang"
#include "lib/brdf.slang"
#include "lib/gi_surface.slang"
#include "lib/terrain.slang"
#include "lib/terrain_layers.slang"
#include "lib/exposure.slang"

/// Light a white diffuse surface would reflect at the primary surface, for
/// metering: set by `shade`.
static float metered_light;


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

/// Ambient light reaching each pixel, 0 to 1; see ao.rs. 1 while AO is off.
[[vk::binding(36, MATERIAL_SET)]] Texture2D<float> ambient_occlusion;

/// Occlusion with the light that bounces between the occluding surfaces put
/// back, so bright albedo darkens less (Jimenez et al. 2016, GTAO). Lux's
/// composite applies the same fit to its SSAO.
float3 multi_bounce_occlusion(float occlusion, float3 albedo) {
    float3 a = 2.0404 * albedo - 0.3324;
    float3 b = -4.7951 * albedo + 0.6417;
    float3 c = 2.7552 * albedo + 0.6903;
    return max(float3(occlusion), ((occlusion * a + b) * occlusion + c) * occlusion);
}

/// Share of the reflected ambient that escapes, from AO: glossy lobes seen
/// head-on lose less than the cosine-weighted AO says (Lagarde and de
/// Rousiers 2014). `roughness` is perceptual.
float specular_occlusion(float nv, float occlusion, float roughness) {
    float alpha = roughness * roughness;
    return saturate(pow(nv + occlusion, exp2(-16.0 * alpha - 1.0)) - 1.0 + occlusion);
}

/// Taps of the blocker search and of the filter, on a Vogel disk.
static const uint SHADOW_SEARCH_TAPS = 12;
static const uint SHADOW_FILTER_TAPS = 16;
/// Share of a cascade's half-width at its edge where it hands pixels over to
/// the next one, and the widest filter in texels, which stays inside that band.
static const float SHADOW_HANDOVER = 0.08;
static const float SHADOW_MAX_RADIUS = 48.0;

/// Interleaved gradient noise (Jimenez 2014): a per-pixel value in [0, 1)
/// whose neighbors differ enough for TAA to average it away.
float gradient_noise(float2 pixel) {
    return frac(52.9829189 * frac(dot(pixel, float2(0.06711056, 0.00583715))));
}

/// Tap `i` of `count` on a unit Vogel disk turned by `angle`: even coverage
/// for any count, and a different pattern per pixel once turned.
float2 vogel_disk(uint i, uint count, float angle) {
    float r = sqrt((float(i) + 0.5) / float(count));
    float theta = float(i) * 2.39996323 + angle;
    return r * float2(cos(theta), sin(theta));
}

/// Sun visibility at `position` (view distance `depth`, geometric normal
/// `normal`) with soft shadows that widen with distance from their caster,
/// as the sun's disk makes them (PCSS, Fernando 2005).
///
/// The finest cascade whose map holds the point is used, not the one its
/// distance falls in: cascades are spheres around their slices and often
/// reach further. Near a map's edge pixels pass to the next cascade with
/// rising odds, so the seam dissolves under TAA. Each tap compares against
/// the receiver's plane, not its center (Isidoro 2006), which keeps acne off
/// slopes however wide the filter. Shadows fade out over the last tenth of
/// the range.
float sun_shadow(float3 position, float3 normal, float depth, uint2 pixel) {
    float range = shadow_split(CASCADES - 1);
    if (globals.sky.w == 0.0 || depth >= range)
        return 1.0;
    float2 seed = float2(pixel) + 5.588238 * globals.shadow_filter.x;
    float handover_noise = gradient_noise(seed);
    float angle = 6.28318531 * gradient_noise(seed + float2(47.0, 17.0));

    // A slice's own cascade always holds it; the one before often does too.
    uint first = 0;
    while (first < CASCADES - 1 && depth > shadow_split(first))
        ++first;
    first = first > 0 ? first - 1 : 0;
    uint cascade = CASCADES;
    for (uint c = first; c < CASCADES; ++c) {
        // Orthographic: w is 1.
        float3 ndc = mul(globals.shadow_view_proj[c], float4(position, 1.0)).xyz;
        float edge = max(abs(ndc.x), abs(ndc.y));
        if (edge >= 1.0 || ndc.z <= 0.0 || ndc.z >= 1.0)
            continue;
        cascade = c;
        float handover = saturate((edge - (1.0 - 2.0 * SHADOW_HANDOVER)) / SHADOW_HANDOVER);
        if (handover_noise >= handover)
            break;
    }
    if (cascade == CASCADES)
        return 1.0;

    float4x4 m = globals.shadow_view_proj[cascade];
    float texel = shadow_texel(cascade);
    // A texel along the normal keeps the lookup off the surface it shades.
    float3 ndc = mul(m, float4(position + normal * texel, 1.0)).xyz;
    float2 uv = ndc.xy * 0.5 + 0.5;
    float meters_per_uv = texel * SHADOW_SIZE;
    // Rows of an orthographic light transform are orthogonal axes scaled by
    // the projection, so the normal in clip space is each row's dot / its
    // squared length. Its plane gives map depth as a function of uv.
    float3 r0 = m[0].xyz, r1 = m[1].xyz, r2 = m[2].xyz;
    float3 n = float3(dot(r0, normal) / dot(r0, r0), dot(r1, normal) / dot(r1, r1), dot(r2, normal) / dot(r2, r2));
    float depth_per_meter = length(r2);
    float nz = n.z >= 0.0 ? max(n.z, 1.0e-6) : min(n.z, -1.0e-6);
    // Steeper than 4:1 against the light only adds bias where acne cannot form.
    float steepest = 4.0 * depth_per_meter * meters_per_uv;
    float2 slope = clamp(-2.0 * n.xy / nz, -steepest, steepest);

    // Blocker search: how far the occluders above this point sit from it.
    float tan_sun = globals.shadow_filter.y;
    float search = clamp(globals.shadow_filter.z * tan_sun / texel, 2.0, SHADOW_MAX_RADIUS) / SHADOW_SIZE;
    float blocker_sum = 0.0;
    uint blockers = 0;
    [unroll] for (uint i = 0; i < SHADOW_SEARCH_TAPS; ++i) {
        float2 offset = vogel_disk(i, SHADOW_SEARCH_TAPS, angle) * search;
        int2 tap = clamp(int2((uv + offset) * SHADOW_SIZE), 0, int(SHADOW_SIZE) - 1);
        float stored = shadow_map.Load(int4(tap, cascade, 0));
        if (stored < ndc.z + dot(slope, offset)) {
            blocker_sum += stored;
            ++blockers;
        }
    }
    float fade = saturate((range - depth) / (0.1 * range));
    if (blockers == 0)
        return 1.0;

    // Penumbra half-width: the sun's angular radius over the blocker distance,
    // and at least a texel so cascade texels never show as stairs.
    float distance = max(ndc.z - blocker_sum / float(blockers), 0.0) / depth_per_meter;
    float radius = clamp(distance * tan_sun / texel, 1.0, SHADOW_MAX_RADIUS) / SHADOW_SIZE;
    if (blockers == SHADOW_SEARCH_TAPS && radius <= search)
        return 1.0 - fade;
    float lit = 0.0;
    [unroll] for (uint i = 0; i < SHADOW_FILTER_TAPS; ++i) {
        float2 offset = vogel_disk(i, SHADOW_FILTER_TAPS, angle) * radius;
        lit += shadow_map.SampleCmpLevelZero(
            shadow_sampler, float3(uv + offset, cascade), ndc.z + dot(slope, offset));
    }
    return lerp(1.0, lit / float(SHADOW_FILTER_TAPS), fade);
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
    Specular specular = material_specular(material);
    Brdf brdf = make_brdf(s.base_color, s.metallic, s.roughness, specular, nv, dfg);
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
        // The Matrix debug view gives GI no albedo, so its tiers' colors stay flat.
        float3 gi_albedo = globals.counts.w != 0 ? 0.0 : s.base_color * (1.0 - transmits);
        gi_gbuffer[pixel] = pack_gbuffer(gi_albedo, s.roughness, s.metallic, specular, s.normal);
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
    float3 hemisphere = lerp(globals.ground.rgb, globals.sky.rgb, s.normal.z * 0.5 + 0.5);
    // shortcut: with GI the hemisphere stands in for the extension's indirect
    // light, so dark GI-lit interiors meter like the sky; meter GI's own
    // irradiance if interiors come out too dark.
    float3 diffuse_light = hemisphere;
    if (globals.counts.z == 0) {
        // AO runs only without GI; GI traces its own occlusion (see ao.rs).
        float occlusion = saturate(ambient_occlusion.Load(int3(pixel, 0)));
        result = (brdf.diffuse_color * multi_bounce_occlusion(occlusion, brdf.diffuse_color)
            + brdf.albedo * specular_occlusion(nv, occlusion, s.roughness)) * hemisphere;
        // The hemisphere behind the surface, through it.
        result += transmitted * PI * lerp(globals.ground.rgb, globals.sky.rgb, -s.normal.z * 0.5 + 0.5);
    } else if ((globals.counts.z & 2) != 0) {
        float3 mirrored = reflect(-s.view, s.normal);
        result = brdf.albedo * lerp(globals.ground.rgb, globals.sky.rgb, mirrored.z * 0.5 + 0.5);
    }
    if (globals.sun_direction.w > 0.0) {
        float3 l = -globals.sun_direction.xyz;
        float3 sun = globals.sun_color.rgb * sun_shadow(s.position, face, depth, pixel) * cloud_shadow(s.position);
        result += evaluate_brdf(brdf, s, l) * sun;
        diffuse_light += sun * max(dot(s.normal, l), 0.0) / PI;
        // Sun behind a leaf, shadowed on the side it lights.
        float behind = -dot(s.normal, l);
        if (transmits > 0.0 && behind > 0.0)
            result += transmitted * behind * globals.sun_color.rgb * sun_shadow(s.position, -face, depth, pixel) * cloud_shadow(s.position);
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
            float3 arriving = light.color_intensity.rgb * light.color_intensity.a * falloff
                * point_shadow(i, light, s.position, face);
            result += evaluate_brdf(brdf, s, l) * arriving;
            diffuse_light += arriving * max(dot(s.normal, l), 0.0) / PI;
            float behind = -dot(s.normal, l);
            if (transmits > 0.0 && behind > 0.0)
                result += transmitted * behind * light.color_intensity.rgb * light.color_intensity.a * falloff
                    * point_shadow(i, light, s.position, -face);
        }
    }
    // Emission stays out of the meter, so a bright sign does not darken the scene.
    result += emission;
    if (primary)
        metered_light = dot(diffuse_light, float3(0.2126, 0.7152, 0.0722));
    if (globals.counts.w != 0) {
        float3 colors[5] = {
            float3(0.1, 0.8, 0.2), float3(0.1, 0.5, 1.0),
            float3(1.0, 0.8, 0.1), float3(1.0, 0.3, 0.1),
            float3(0.8, 0.1, 0.9)
        };
        // Lit from the eye so shapes read, and metered on that light alone:
        // metering the scene's own light blew the flat colors out to white.
        float light = 0.4 + 0.6 * nv;
        result = colors[min(level, 4)] * light;
        if (primary)
            metered_light = light;
    }
    return result;
}

/// One pixel in 16 meters the exposure: plenty for an average.
bool metered(uint2 id) {
    return (id.x & 3) == 0 && (id.y & 3) == 0;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void resolveMain(uint3 id: SV_DispatchThreadID) {
    float4 viewport = globals.viewport;
    if (any(float2(id.xy) >= viewport.zw))
        return;
    uint2 pixel = id.xy + uint2(viewport.xy);
    uint2 hit = visibility.Load(int3(pixel, 0));
    bool surface = hit.x != NO_INSTANCE;
    if (surface)
        hdr_output[pixel] = float4(shade(id.xy, pixel, hit, true), 1.0);
    else
        clear_pixel(pixel);
    meter_exposure(surface && metered(id.xy), metered_light);
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
        meter_exposure(false, 0.0);
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
    meter_exposure(metered(id.xy), metered_light);
}
