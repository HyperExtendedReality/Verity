// Multiple-scattering GGX specular (Fdez-Agüera energy compensation over a
// split-sum DFG table) with Hammon's diffuse. Binding-free: callers supply the
// DFG lookup, so the same BSDF serves direct lighting and GI.

static const float BRDF_PI = 3.14159265359;

struct Surface {
    float3 position;
    float3 normal;
    /// Unit vector toward the viewer.
    float3 view;
    float3 base_color;
    float metallic;
    float roughness;
};

/// View-dependent terms shared by every light at one shading point.
struct Brdf {
    float3 f0;
    /// Energy the single-scatter lobe loses, re-added as multiple scattering.
    float3 multiple_scatter;
    float3 diffuse_color;
    float alpha;
    float nv;
};

float pow5(float x) {
    float x2 = x * x;
    return x2 * x2 * x;
}

float3 brdf_f0(float3 base_color, float metallic) {
    return lerp(float3(0.04), max(base_color, 0.0), saturate(metallic));
}

/// Directional specular albedo, single plus multiple scattering, from the
/// DFG terms (x scales F0, y is the F90 bias).
float3 specular_albedo(float3 f0, float2 dfg) {
    float ess = saturate(dfg.x + dfg.y);
    float3 fss = f0 * dfg.x + dfg.y;
    float3 favg = f0 + (1.0 - f0) / 21.0;
    float3 fms = fss * favg / max(1.0 - (1.0 - ess) * favg, 0.0001);
    return fss + fms * (1.0 - ess);
}

/// `nv` is N·V; `dfg` the table lookup at (N·V, perceptual roughness).
Brdf make_brdf(float3 base_color, float metallic, float roughness, float nv, float2 dfg) {
    Brdf b;
    b.nv = max(nv, 0.0001);
    float perceptual = clamp(roughness, 0.045, 1.0);
    b.alpha = perceptual * perceptual;
    b.f0 = brdf_f0(base_color, metallic);
    float ess = saturate(dfg.x + dfg.y);
    float3 single = b.f0 * dfg.x + dfg.y;
    float3 albedo = specular_albedo(b.f0, dfg);
    b.multiple_scatter = albedo - single;
    b.diffuse_color = max(base_color, 0.0) * (1.0 - saturate(metallic)) * max(0.0, 1.0 - albedo);
    return b;
}

/// Hammon's diffuse approximation, including its multiple-scattering term,
/// for unit albedo.
float hammon(float nl, float nv, float nh, float vh, float alpha) {
    float smooth_term = 1.05 * (1.0 - pow5(1.0 - nl)) * (1.0 - pow5(1.0 - nv));
    float facing = vh * vh;
    float rough = facing * (0.9 - 0.4 * facing) * (0.5 + nh) / max(nh, 0.0001);
    return lerp(smooth_term, rough, alpha) / BRDF_PI + 0.1159 * alpha;
}

/// Hammon diffuse BRDF per steradian for unit albedo, from directions.
float hammon_diffuse(float3 n, float3 v, float3 l, float roughness) {
    float nl = dot(n, l);
    float nv = dot(n, v);
    if (nl <= 0.0 || nv <= 0.0)
        return 0.0;
    float3 h = normalize(l + v);
    float alpha = clamp(roughness, 0.045, 1.0);
    alpha *= alpha;
    return hammon(nl, nv, saturate(dot(n, h)), saturate(dot(v, h)), alpha);
}

/// BRDF per steradian; the caller applies N·L.
float3 brdf_value(Brdf b, float3 n, float3 v, float3 l) {
    float nl = dot(n, l);
    if (nl <= 0.0)
        return 0.0;
    float3 h = normalize(l + v);
    float nh = max(dot(n, h), 0.0);
    float vh = max(dot(v, h), 0.0);
    float a2 = b.alpha * b.alpha;
    float d = a2 / (BRDF_PI * pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
    float gv = nl * sqrt(b.nv * b.nv * (1.0 - a2) + a2);
    float gl = b.nv * sqrt(nl * nl * (1.0 - a2) + a2);
    float visibility = 0.5 / max(gv + gl, 0.0001);
    float3 fresnel = b.f0 + (1.0 - b.f0) * pow5(1.0 - vh);
    float3 specular = d * visibility * fresnel + b.multiple_scatter / BRDF_PI;
    return b.diffuse_color * hammon(nl, b.nv, nh, vh, b.alpha) + specular;
}

/// Reflected radiance per unit incident irradiance from direction `l`, times N·L.
float3 evaluate_brdf(Brdf b, Surface s, float3 l) {
    return brdf_value(b, s.normal, s.view, l) * max(dot(s.normal, l), 0.0);
}
