// Multiple-scattering GGX specular (Fdez-Agüera energy compensation over a
// split-sum DFG table) with energy-preserving Oren–Nayar diffuse. Binding-free:
// callers supply the DFG lookup, so the same BSDF serves direct lighting and GI.

static const float BRDF_PI = 3.14159265359;

struct Surface {
    float3 position;
    float3 normal;
    /// Unit vector toward the viewer.
    float3 view;
    float3 base_color;
    float metallic;
    float roughness;
    /// 1 for one surface. Towards 0 the point stands for many facets turned
    /// every way, and `normal` is only their mean.
    float agreement;
};

/// View-dependent terms shared by every light at one shading point.
struct Brdf {
    float3 f0;
    /// Directional specular albedo over its single-scattering part. Scaling the
    /// GGX lobe by it restores the energy multiple scattering adds, in the
    /// lobe's own shape: rough highlights brighten rather than gaining a haze.
    float3 specular_scale;
    float3 diffuse_color;
    float roughness;
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
    b.roughness = clamp(roughness, 0.045, 1.0);
    b.alpha = b.roughness * b.roughness;
    b.f0 = brdf_f0(base_color, metallic);
    float3 single = b.f0 * dfg.x + dfg.y;
    float3 albedo = specular_albedo(b.f0, dfg);
    b.specular_scale = albedo / max(single, 0.0001);
    b.diffuse_color = max(base_color, 0.0) * (1.0 - saturate(metallic)) * max(0.0, 1.0 - albedo);
    return b;
}

// Energy-preserving Oren–Nayar, EON (Portsmouth, Kutz and Hill, JCGT 2025): the
// rough diffuse model OpenPBR adopted. Fujii's Oren–Nayar adds back what rough
// surfaces scatter towards the light, and a reciprocal lobe adds what it loses
// to multiple scattering, so a white surface reflects all of its light at every
// angle. Constants and the albedo fit are the paper's listing.
static const float EON_C1 = 0.5 - 2.0 / (3.0 * BRDF_PI);
static const float EON_C2 = 2.0 / 3.0 - 28.0 / (15.0 * BRDF_PI);

/// Fujii Oren–Nayar directional albedo; the paper's fit is within 0.1%.
float eon_fon_albedo(float mu, float r) {
    float m = 1.0 - mu;
    float g = m * (0.0571085289 + m * (0.491881867 + m * (-0.332181442 + m * 0.0714429953)));
    return (1.0 + r * g) / (1.0 + EON_C1 * r);
}

/// EON per steradian; the caller applies N·L. `rho` is the single-scattering
/// albedo, `r` the roughness in [0, 1] (0 is Lambert), `lv` is L·V.
float3 eon(float3 rho, float r, float nl, float nv, float lv) {
    if (nl <= 0.0 || nv <= 0.0)
        return 0.0;
    float s = lv - nl * nv;
    float s_over_t = s > 0.0 ? s / max(nl, nv) : s;
    float a = 1.0 / (1.0 + EON_C1 * r);
    float3 single = rho / BRDF_PI * a * (1.0 + r * s_over_t);
    float average = a * (1.0 + EON_C2 * r);
    float3 rho_ms = rho * rho * average / (1.0 - rho * (1.0 - average));
    const float eps = 1.0e-7;
    float3 multiple = rho_ms / BRDF_PI * max(eps, 1.0 - eon_fon_albedo(nl, r))
        * max(eps, 1.0 - eon_fon_albedo(nv, r)) / max(eps, 1.0 - average);
    return single + multiple;
}

/// The diffuse lobe alone, per steradian.
float3 brdf_diffuse(Brdf b, float3 n, float3 v, float3 l) {
    return eon(b.diffuse_color, b.roughness, dot(n, l), b.nv, dot(l, v));
}

/// EON per steradian for unit albedo, from directions and perceptual roughness.
/// shortcut: scaling it by a colour afterwards gives dark surfaces a little too
/// much multiple scattering; pass the colour to `eon` where it is known.
float diffuse_unit(float3 n, float3 v, float3 l, float roughness) {
    return eon(1.0, clamp(roughness, 0.045, 1.0), dot(n, l), dot(n, v), dot(l, v)).x;
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
    float3 specular = d * visibility * fresnel * b.specular_scale;
    return brdf_diffuse(b, n, v, l) + specular;
}

/// Reflected radiance per unit incident irradiance from direction `l`, times N·L.
float3 evaluate_brdf(Brdf b, Surface s, float3 l) {
    float n_dot_l = dot(s.normal, l);
    // Facets spread over the hemisphere around their mean average (1 + n.l) / 4,
    // which a clamped cosine of the mean does not.
    float cosine = lerp((1.0 + n_dot_l) * 0.25, max(n_dot_l, 0.0), s.agreement);
    return brdf_value(b, s.normal, s.view, l) * cosine;
}
