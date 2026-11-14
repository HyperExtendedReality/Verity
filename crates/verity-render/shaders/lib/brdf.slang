// Multiple-scattering GGX specular (Fdez-Agüera energy compensation over a
// split-sum DFG table) with energy-preserving Oren–Nayar diffuse, and OpenPBR's
// Fresnel: exact for dielectrics from `specular_ior`, F82-tint for metals. Binding-free:
// callers supply the DFG lookup, so the same BSDF serves direct lighting and GI.

#pragma once

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

/// OpenPBR's specular inputs beyond base colour, metallic and roughness.
struct Specular {
    /// `specular_color`: tints the dielectric Fresnel, and is the F82 tint of
    /// metals (their reflectance near grazing). White leaves both untinted.
    float3 color;
    /// `specular_ior`: the dielectric's index of refraction.
    float ior;
};

Specular default_specular() {
    Specular s;
    s.color = 1.0;
    s.ior = 1.5;
    return s;
}

/// View-dependent terms shared by every light at one shading point.
struct Brdf {
    float3 tint;
    float ior;
    float metallic;
    /// Metal reflectance at normal incidence, and its F82-tint coefficient.
    float3 metal_f0;
    float3 metal_f82;
    /// Directional specular albedo, single plus multiple scattering.
    float3 albedo;
    /// `albedo` over its single-scattering part. Scaling the GGX lobe by it
    /// restores the energy multiple scattering adds, in the lobe's own shape:
    /// rough highlights brighten rather than gaining a haze.
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

/// Unpolarised Fresnel reflectance of a dielectric, exact. `c` is the cosine
/// of incidence and `ior` the inside's index over the outside's.
float fresnel_dielectric(float c, float ior) {
    float g2 = ior * ior - 1.0 + c * c;
    if (g2 < 0.0)
        return 1.0; // Total internal reflection.
    float g = sqrt(g2);
    float a = (g - c) / (g + c);
    float b = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
    return 0.5 * a * a * (1.0 + b * b);
}

float ior_f0(float ior) {
    float r = (ior - 1.0) / (ior + 1.0);
    return r * r;
}

/// F82-tint (OpenPBR, after Kutz et al. 2021): Schlick minus a term that
/// dips near grazing, sized so reflectance at cos = 1/7 (82°) is `tint` times
/// Schlick's there. A white tint gives plain Schlick.
static const float F82_COSINE = 1.0 / 7.0;

float3 f82_coefficient(float3 f0, float3 tint) {
    float m = 1.0 - F82_COSINE;
    float3 schlick = f0 + (1.0 - f0) * pow5(m);
    return schlick * (1.0 - tint) / (F82_COSINE * pow5(m) * m);
}

float3 fresnel_f82(float3 f0, float3 coefficient, float c) {
    float m = 1.0 - c;
    return saturate(f0 + (1.0 - f0) * pow5(m) - coefficient * c * pow5(m) * m);
}

/// Single-scattering specular albedo from the DFG terms (x scales F0, y is the
/// F90 bias). The DFG table integrates Schlick's (1 - cos)^5; F82_SHARE is the
/// F82 term's cos (1 - cos)^6 over it, on the same 16x16 grid of (N·V,
/// perceptual roughness). verity-render's `f82_share_table_matches_shader`
/// test recomputes it and prints a fresh one when it drifts.
static const float F82_SHARE[256] = {
    0.0074, 0.0623, 0.1156, 0.1600, 0.1955, 0.2222, 0.2400, 0.2489, 0.2489, 0.2400, 0.2223, 0.1956, 0.1601, 0.1157, 0.0624, 0.0000,
    0.0133, 0.0627, 0.1156, 0.1599, 0.1954, 0.2221, 0.2399, 0.2488, 0.2489, 0.2401, 0.2224, 0.1958, 0.1604, 0.1162, 0.0633, 0.0000,
    0.0339, 0.0680, 0.1168, 0.1595, 0.1944, 0.2209, 0.2387, 0.2478, 0.2485, 0.2405, 0.2239, 0.1988, 0.1655, 0.1413, 0.1652, 0.1664,
    0.0565, 0.0819, 0.1216, 0.1601, 0.1928, 0.2182, 0.2360, 0.2456, 0.2474, 0.2411, 0.2275, 0.2069, 0.1828, 0.1661, 0.1939, 0.1704,
    0.0774, 0.1002, 0.1313, 0.1634, 0.1921, 0.2155, 0.2324, 0.2425, 0.2456, 0.2419, 0.2318, 0.2166, 0.2004, 0.1892, 0.1913, 0.1725,
    0.0961, 0.1178, 0.1432, 0.1695, 0.1936, 0.2140, 0.2294, 0.2393, 0.2436, 0.2422, 0.2357, 0.2252, 0.2131, 0.2037, 0.1960, 0.1707,
    0.1121, 0.1329, 0.1548, 0.1766, 0.1967, 0.2141, 0.2277, 0.2370, 0.2418, 0.2421, 0.2382, 0.2309, 0.2211, 0.2114, 0.2008, 0.1695,
    0.1254, 0.1452, 0.1648, 0.1834, 0.2005, 0.2152, 0.2271, 0.2356, 0.2407, 0.2419, 0.2397, 0.2343, 0.2263, 0.2158, 0.2022, 0.1686,
    0.1363, 0.1552, 0.1729, 0.1893, 0.2042, 0.2170, 0.2274, 0.2351, 0.2399, 0.2417, 0.2404, 0.2363, 0.2292, 0.2190, 0.2039, 0.1680,
    0.1450, 0.1630, 0.1795, 0.1942, 0.2074, 0.2188, 0.2280, 0.2349, 0.2394, 0.2414, 0.2407, 0.2373, 0.2308, 0.2209, 0.2052, 0.1678,
    0.1518, 0.1693, 0.1845, 0.1982, 0.2102, 0.2204, 0.2287, 0.2350, 0.2392, 0.2411, 0.2407, 0.2377, 0.2316, 0.2219, 0.2059, 0.1679,
    0.1569, 0.1743, 0.1887, 0.2015, 0.2126, 0.2218, 0.2294, 0.2351, 0.2390, 0.2408, 0.2405, 0.2376, 0.2320, 0.2224, 0.2065, 0.1683,
    0.1608, 0.1782, 0.1921, 0.2041, 0.2144, 0.2231, 0.2301, 0.2354, 0.2389, 0.2405, 0.2402, 0.2374, 0.2319, 0.2225, 0.2068, 0.1688,
    0.1637, 0.1814, 0.1950, 0.2064, 0.2161, 0.2242, 0.2307, 0.2355, 0.2388, 0.2403, 0.2398, 0.2371, 0.2316, 0.2225, 0.2068, 0.1693,
    0.1658, 0.1842, 0.1975, 0.2083, 0.2175, 0.2250, 0.2312, 0.2358, 0.2386, 0.2400, 0.2394, 0.2367, 0.2313, 0.2223, 0.2067, 0.1698,
    0.1674, 0.1867, 0.1996, 0.2102, 0.2189, 0.2262, 0.2316, 0.2359, 0.2386, 0.2396, 0.2389, 0.2362, 0.2308, 0.2219, 0.2065, 0.1702
};

float f82_share(float nv, float roughness) {
    float2 uv = clamp(float2(nv, roughness), 0.0, 1.0) * 15.0;
    int2 a = int2(floor(uv));
    int2 b = min(a + 1, int2(15));
    float2 f = frac(uv);
    float lo = lerp(F82_SHARE[a.y * 16 + a.x], F82_SHARE[a.y * 16 + b.x], f.x);
    float hi = lerp(F82_SHARE[b.y * 16 + a.x], F82_SHARE[b.y * 16 + b.x], f.x);
    return lerp(lo, hi, f.y);
}

float3 specular_single(float3 f0, float3 f82, float share, float2 dfg) {
    return max(f0 * dfg.x + dfg.y * (1.0 - f82 * share), 0.0);
}

/// Directional specular albedo, single plus multiple scattering.
float3 specular_albedo(float3 f0, float3 f82, float share, float2 dfg) {
    float ess = saturate(dfg.x + dfg.y);
    float3 fss = specular_single(f0, f82, share, dfg);
    float3 favg = max(f0 + (1.0 - f0) / 21.0 - f82 / 126.0, 0.0);
    float3 fms = fss * favg / max(1.0 - (1.0 - ess) * favg, 0.0001);
    return fss + fms * (1.0 - ess);
}

/// The dielectric layer's specular albedo, which the diffuse beneath it loses.
/// shortcut: the DFG table is Schlick's, so this takes Schlick at the IOR's F0
/// for the exact Fresnel the lobe uses; `f82_albedo_fit` bounds the gap (0.06
/// at grazing on smooth water, as plain Schlick was). Tabulate per IOR if it shows.
float3 dielectric_albedo(Specular s, float2 dfg) {
    float ior = clamp(s.ior, 1.0, 4.0);
    return max(s.color, 0.0) * specular_albedo(ior_f0(ior), 0.0, 0.0, dfg);
}

/// `nv` is N·V; `dfg` the table lookup at (N·V, perceptual roughness).
Brdf make_brdf(float3 base_color, float metallic, float roughness, Specular specular, float nv, float2 dfg) {
    Brdf b;
    b.nv = max(nv, 0.0001);
    b.roughness = clamp(roughness, 0.045, 1.0);
    b.alpha = b.roughness * b.roughness;
    b.metallic = saturate(metallic);
    b.tint = max(specular.color, 0.0);
    b.ior = clamp(specular.ior, 1.0, 4.0);
    b.metal_f0 = saturate(base_color);
    b.metal_f82 = f82_coefficient(b.metal_f0, saturate(b.tint));
    float3 dielectric = dielectric_albedo(specular, dfg);
    float3 dielectric_single = b.tint * specular_single(ior_f0(b.ior), 0.0, 0.0, dfg);
    float share = f82_share(b.nv, b.roughness);
    b.albedo = lerp(dielectric, specular_albedo(b.metal_f0, b.metal_f82, share, dfg), b.metallic);
    float3 single = lerp(dielectric_single, specular_single(b.metal_f0, b.metal_f82, share, dfg), b.metallic);
    b.specular_scale = b.albedo / max(single, 0.0001);
    // Metal and dielectric mix linearly; only the dielectric covers diffuse.
    b.diffuse_color = max(base_color, 0.0) * (1.0 - b.metallic) * max(0.0, 1.0 - dielectric);
    return b;
}

/// Fresnel of the mixed dielectric and metal lobes at cosine `vh`.
float3 brdf_fresnel(Brdf b, float vh) {
    return lerp(b.tint * fresnel_dielectric(vh, b.ior), fresnel_f82(b.metal_f0, b.metal_f82, vh), b.metallic);
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
    float3 specular = d * visibility * brdf_fresnel(b, vh) * b.specular_scale;
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
