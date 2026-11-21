int2 wrapIrradianceTexel(int2 pixel) {
    if (pixel.x < 0) pixel = int2(-pixel.x - 1, 5 - pixel.y);
    else if (pixel.x >= 6) pixel = int2(11 - pixel.x, 5 - pixel.y);
    if (pixel.y < 0) pixel = int2(5 - pixel.x, -pixel.y - 1);
    else if (pixel.y >= 6) pixel = int2(5 - pixel.x, 11 - pixel.y);
    return pixel;
}
uint irradianceTexelIndex(int2 pixel) {
    pixel = wrapIrradianceTexel(pixel);
    return uint(pixel.y * 6 + pixel.x);
}


struct DiffuseSources {
    float4 background;
    float4 punctual;
};

// Every spatial corner (and both LODs) uses the same normal. Compute the
// octahedral footprint once per receiver instead of once per probe.
struct IrradianceLookup {
    uint4 texels;
    float4 weights;
};

IrradianceLookup irradianceLookup(float3 normal) {
    IrradianceLookup lookup;
    float2 grid = encodeNormal(normal) * 6.0 - 0.5;
    int2 base = int2(floor(grid));
    float2 fraction = frac(grid);
    float2 complement = 1.0 - fraction;
    lookup.weights = float4(complement.x * complement.y, fraction.x * complement.y,
                            complement.x * fraction.y, fraction.x * fraction.y);
    lookup.texels = uint4(irradianceTexelIndex(base),
                          irradianceTexelIndex(base + int2(1, 0)),
                          irradianceTexelIndex(base + int2(0, 1)),
                          irradianceTexelIndex(base + int2(1, 1)));
    return lookup;
}


DiffuseSources diffuseProbePage(uint page, IrradianceLookup lookup, bool punctualActive) {
    DiffuseSources result = (DiffuseSources)0;
    [unroll] for (uint corner = 0u; corner < 4u; ++corner) {
        float bilinearWeight = lookup.weights[corner];
        if (!(bilinearWeight > 0.0)) continue;
        uint texel = lookup.texels[corner];
        float4 background = directionalProbes[page].irradiance[texel];
        if (all(isfinite(background))) result.background += background * bilinearWeight;
        if (punctualActive) {
            float4 punctual = directionalProbes[page].punctualIrradiance[texel];
            if (all(isfinite(punctual))) result.punctual += punctual * bilinearWeight;
        }
    }
    return result;
}

// `previous` reads the probes last frame packed, as `findIrradianceProbe` does.
DiffuseSources diffuseAtLod(float3 position, IrradianceLookup lookup, uint lod, bool previous,
                            bool punctualActive, inout DependencyContext dependency) {
    DiffuseSources result = (DiffuseSources)0;
    float spacingValue = spacing(params, 0u, lod);
    float3 grid = position / spacingValue - 0.5;
    int3 base = int3(floor(grid));
    float3 fraction = frac(grid);
    uint pageBase = ((params.frame.x + (previous ? 1u : 0u)) & 1u)
        * probeDirectoryCapacity(params);
    [loop] for (uint corner = 0u; corner < 8u; ++corner) {
        uint3 bit = uint3(corner & 1u, (corner >> 1u) & 1u, corner >> 2u);
        float3 axis = float3(
            bit.x == 0u ? 1.0 - fraction.x : fraction.x,
            bit.y == 0u ? 1.0 - fraction.y : fraction.y,
            bit.z == 0u ? 1.0 - fraction.z : fraction.z);
        float spatialWeight = axis.x * axis.y * axis.z;
        if (!(spatialWeight > 0.0)) continue;
        int4 key = int4(base + int3(bit), int(lod << 18u));
        uint slot = findIrradianceProbe(key, previous);
        if (slot == EMPTY) continue;
        uint page = pageBase + slot;
        DiffuseSources probe = diffuseProbePage(page, lookup, punctualActive);
        if (previous) dependencyUnion(dependency, probeDependency(
            directionalProbes[page].dependencyMin, directionalProbes[page].dependencyMax));
        result.background += probe.background * spatialWeight;
        result.punctual += probe.punctual * spatialWeight;
    }
    return result;
}

DiffuseSources sampleSplitProbes(float3 position, float3 normal, bool previous,
                                 bool punctualActive, inout DependencyContext dependency) {
    uint lowLod, highLod;
    float lodBlend;
    splitLodBlendForParameters(position, params, lowLod, highLod, lodBlend);
    IrradianceLookup lookup = irradianceLookup(normal);
    DiffuseSources result = diffuseAtLod(position, lookup, lowLod, previous,
        punctualActive, dependency);
    if (lodBlend > 0.0) {
        DiffuseSources high = diffuseAtLod(position, lookup, highLod, previous,
            punctualActive, dependency);
        result.background = lerp(result.background, high.background, lodBlend);
        result.punctual = lerp(result.punctual, high.punctual, lodBlend);
    }
    return result;
}

float3 normalizeIrradiance(float4 sample) {
    return sample.w > 0.0 ? sample.xyz / sample.w : 0.0;
}

float3 resolveSplitDiffuse(float3 position, float3 normal, out float coverage,
                           out float3 background, out float backgroundCoverage,
                           out float anyCoverage) {
    DependencyContext unused = dependencyEmpty();
    bool punctualActive = transportSourceActive(params, PUNCTUAL_SOURCE);
    DiffuseSources sample = sampleSplitProbes(position, normal, false, punctualActive, unused);
    background = normalizeIrradiance(sample.background);
    backgroundCoverage = saturate(sample.background.w);
    coverage = backgroundCoverage;
    anyCoverage = backgroundCoverage;
    if (punctualActive) {
        float punctualCoverage = saturate(sample.punctual.w);
        coverage = min(coverage, punctualCoverage);
        anyCoverage = max(anyCoverage, punctualCoverage);
    }
    return background + normalizeIrradiance(sample.punctual);
}

#ifndef GI_COMPOSITE_RESOLVE
// Further bounces: the irradiance last frame's probes hold at a ray's hit, which
// the hit reflects. Each frame adds a bounce, and reflectance below one makes the
// series converge. Hits outside every probe, off screen, keep one bounce.
TransportRadiance bounceFeedback(float3 position, float3 normal,
                                 inout DependencyContext dependency,
                                 out float backgroundCoverage) {
    TransportRadiance result = (TransportRadiance)0;
    backgroundCoverage = 0.0;
    if (params.cacheTime.z == 0u) return result;
    bool punctualActive = transportSourceActive(params, PUNCTUAL_SOURCE);
    DiffuseSources sample = sampleSplitProbes(position, normal, true, punctualActive, dependency);
    result.background = normalizeIrradiance(sample.background);
    result.punctual = normalizeIrradiance(sample.punctual);
    backgroundCoverage = saturate(sample.background.w);
    return result;
}

float3 emitterLight(float3 position, float3 normal, float3 geometric, float3 view,
                    bool receiver, bool weighted, bool resample,
                    SurfaceBsdf material, float2 ab, float ggxProbability, inout uint state);

TransportRadiance outgoingTransportRadianceTracked(float3 origin, float3 direction, Hit hit,
    out bool querySupported, inout DependencyContext dependency, uint admissionLimit = 0u);

uint2 previousCascadeGuide(float3 position, uint lod) {
    int4 key = int4(int3(floor(position / spacing(params, 0u, lod))), int(lod << 18u));
    uint slot = findIrradianceProbe(key, true);
    if (slot == EMPTY) return 0u;
    uint page = ((params.frame.x + 1u) & 1u) * probeDirectoryCapacity(params) + slot;
    uint4 record = directionalProbes[page].cacheGeneration;
    return record.x + 1u == params.frame.x && record.y != 0u ? record.zw : uint2(0u);
}

uint guideMass(uint2 packed, uint bin) {
    return (packed[bin >> 2u] >> ((bin & 3u) * 8u)) & 255u;
}

float3 sampleCascadeGuide(uint2 packed, float4 random, float3 normal, out float weight) {
    uint total = 0u;
    [unroll] for (uint i = 0u; i < 8u; ++i) total += guideMass(packed, i);
    float3 direction;
    uint bin;
    float2 uv = min(random.zw, 0.99999994);
    if (total == 0u) {
        weight = 1.0;
        return cosineDirection(normal, uv);
    }
    if (random.x < 0.5) {
        float target = min(random.y, 0.99999994) * float(total);
        bin = 0u;
        [loop] for (; bin < 7u; ++bin) {
            float mass = float(guideMass(packed, bin));
            if (target < mass) break;
            target -= mass;
        }
        direction = decodeDirection((float2(bin & 3u, bin >> 2u) + uv) / float2(4.0, 2.0));
    } else {
        direction = cosineDirection(normal, uv);
        uint2 cell = min(uint2(encodeDirection(direction) * float2(4.0, 2.0)), uint2(3u, 1u));
        bin = cell.x + 4u * cell.y;
    }
    // Eight equal-solid-angle bins cover 4π. Scale both PDFs by π so the
    // estimator outputs irradiance / π like Split's projected diffuse probes.
    float cosine = saturate(dot(normal, direction));
    float pdfPi = 0.5 * cosine + float(guideMass(packed, bin)) / float(total);
    weight = pdfPi > 0.0 ? cosine / pdfPi : 0.0;
    return direction;
}

bool traceCascadeGather(float3 position, float3 normal, float3 geometric, uint seed,
                        out float3 diffuse, out float3 background) {
    diffuse = background = 0.0;
    uint low, high;
    float blend;
    splitLodBlendForParameters(position, params, low, high, blend);
    uint lod = blend < 0.5 ? low : high;
    uint2 packed = 0u;
    if ((params.ownerSequence.y & 3u) == 2u) packed = previousCascadeGuide(position, lod);
    float4 random = float4(transportRandom(seed, 0u), transportRandom(seed, 1u),
                           transportRandom(seed, 2u), transportRandom(seed, 3u));
    float weight;
    float3 direction = sampleCascadeGuide(packed, random, normal, weight);
    if (weight <= 0.0 || dot(geometric, direction) <= 0.0) return true;
    transportAdmissionBypass = true;
    surfaceFarDistance = surfaceCacheDistance(lod);
    float3 origin = rayOffset(position, geometric, params.tuning.z);
    DependencyContext dependency;
    float start, distance, padding;
    Hit hit = traceRayTracked(origin, direction, 1e20, dependency, start, distance, padding);
    bool supported;
    TransportRadiance radiance = outgoingTransportRadianceTracked(origin, direction, hit,
        supported, dependency, 0u);
    background = (radiance.background + radiance.emissive) * weight;
    diffuse = background + radiance.punctual * weight;
    return supported && all(isfinite(diffuse)) && all(isfinite(background));
}

[ForceInline]
float emitterPdf(Hit hit, float3 direction);
bool materialGatherIncoming(Hit hit, float3 position, float3 direction, out TransportRadiance light);
float2 rtgiGatherNoise(uint2 pixel);

// The material's directional reflectance under uniform light, split into
// its diffuse (under the dielectric layer) and specular lobes.
void materialReflectance(SurfaceBsdf material, float2 ab, out float3 diffuse,
                         out float3 specular) {
    float3 dielectric = bsdfSpecularEnergy(material, ab);
    diffuse = bsdfDiffuseColor(material) * (1.0 - dielectric);
    specular = lerp(dielectric, material.baseColor, material.metallic);
}

float materialGgxProbability(SurfaceBsdf material, float2 ab) {
    float3 diffuse, specular;
    materialReflectance(material, ab, diffuse, specular);
    const float3 luma = float3(0.2126, 0.7152, 0.0722);
    float total = dot(diffuse + specular, luma);
    return total > 0.0 ? clamp(dot(specular, luma) / total, 0.1, 0.95) : 0.5;
}

bool sampleMaterialDirection(float3 normal, float3 geometric, float3 view,
                             SurfaceBsdf material, float ggxProbability, float2 xi,
                             float choose, out float3 direction, out float pdf) {
    pdf = 0.0;
    float ggxPdf;
    if (choose < ggxProbability) {
        direction = bsdfSampleVisibleGgx(xi, material, normal, view, ggxPdf);
        if (!(ggxPdf > 0.0)) return false;
    } else {
        direction = cosineDirection(normal, xi);
        ggxPdf = bsdfVisibleGgxPdf(material, normal, view, direction);
    }
    float cosine = dot(normal, direction);
    if (!(cosine > 0.0) || !(dot(geometric, direction) > 0.0)) return false;
    pdf = lerp(cosine / PI, ggxPdf, ggxProbability);
    return pdf > 0.0 && isfinite(pdf);
}

// One BSDF-sampled ray in the receiver's local shading frame. The existing
// transport cache approximates the radiance at the next surface; the receiver
// itself evaluates the opaque diffuse + specular BSDF, not a Lambert albedo.
bool traceMaterialGather(float3 position, float3 normal, float3 geometric, float3 view,
                         SurfaceBsdf material, float2 ab, float ggxProbability,
                         float2 xi, float choose, out float3 shaded, out float3 background,
                         out float3 listedEmission) {
    shaded = background = listedEmission = 0.0;
    float3 direction;
    float pdf;
    if (!sampleMaterialDirection(normal, geometric, view, material, ggxProbability,
        xi, choose, direction, pdf)) return true; // clipped draw has zero contribution
    float3 weight = bsdfEvaluate(material, normal, view, direction, ab)
        * (dot(normal, direction) / pdf);
    transportAdmissionBypass = true;
    float3 origin = rayOffset(position, geometric, params.tuning.z);
    DependencyContext dependency;
    float start, distance, padding;
    Hit hit = traceRayTracked(origin, direction, 1e20, dependency, start, distance, padding);
    TransportRadiance light;
    if ((hit.errors & HIT_QUERY_UNSUPPORTED) != 0u) return false;
    bool cached = materialGatherIncoming(hit, origin + direction * hit.distance, direction, light);
    if (hit.triangle != EMPTY) {
        // Balance the BSDF-sampled emission against the explicit emitter ray.
        float lightPdf = emitterPdf(hit, direction);
        float3 emission = hitEmission(hit, 0.0);
        if (lightPdf > 0.0) listedEmission = emission * weight * (pdf / (pdf + lightPdf));
        else light.emissive += emission;
    }
    // Keep the paired estimate of listed emission even when the hit cache is
    // cold. Split's diffuse fallback does not contain first hits on that list.
    if (!cached) return false;
    background = (light.background + light.emissive) * weight;
    shaded = background + light.punctual * weight;
    return all(isfinite(shaded)) && all(isfinite(background));
}

SurfaceBsdf materialSpaceBsdf(uint4 deferred) {
    SurfaceBsdf material;
    material.baseColor = deferredBaseColor(deferred);
    material.metallic = float((deferred.z >> 8u) & 255u) / 255.0;
    material.perceptualRoughness = float(deferred.x >> 24u) / 255.0;
    uint3 tint = (deferred.yyy >> uint3(0u, 8u, 16u)) & 255u;
    material.specularColor = pow(float3(tint) / 255.0, 2.2);
    // The thin G-buffer stores r, with F0 = 0.16 r² (IOR >= 1).
    float sqrtF0 = 0.4 * float(deferred.z & 255u) / 255.0;
    material.ior = (1.0 + sqrtF0) / (1.0 - sqrtF0);
    return material;
}

// Specialize each entry point so the ordinary probe path cannot retain tracing code.
[ForceInline]
void resolveWorld(uint3 thread, bool traceGather, bool materialGather) {
    if (any(thread.xy >= giResolveExtent(params))) return;
    uint scale = giResolveScale(params);
    uint2 full = params.viewport.xy + min(thread.xy * scale + scale / 2u,
                                          params.viewport.zw - 1u);
    uint4 deferred = gbuffer.Load(int3(full, 0));
    uint4 receiver = rasterReceiver.Load(int3(full, 0));
    float depth = depthTexture.Load(int3(full, 0));
    bool eligible = depth > 0.0 && isfinite(depth)
        && (receiver.z & 0x80000000u) != 0u
        && (deferred.w & (1u << 24u)) == 0u;
    float3 diffuse = 0.0;
    float3 background = 0.0;
    float backgroundCoverage = 0.0, anyCoverage = 0.0;
    float coverage = 0.0;
    bool supported = false;
    if (eligible) {
        float3 normal = deferredShadingNormal(deferred);
        if (all(isfinite(normal)) && dot(normal, normal) > 0.01) {
            normal = normalize(normal);
            float3 position = worldPosition(float2(full) + 0.5, depth, params);
            float3 geometric = receiverGeometricNormal(receiver.z, normal);
            uint at = thread.y * giResolveExtent(params).x + thread.x;
            bool traced = false;
            bool materialSample = false;
            float3 listedEmission = 0.0;
            // What a cold-cache fallback reflects of the probes' light.
            float3 fallbackReflectance = materialGather ? deferredDiffuseAlbedo(deferred) : 1.0;
            if (traceGather) {
                uint seed = transportHash(at ^ transportHash(params.frame.x ^ 0xc017ae7u));
                if (materialGather) {
                    SurfaceBsdf material = materialSpaceBsdf(deferred);
                    float3 view = bsdfUnitOrZero(params.cameraPosition.xyz - position);
                    float nDotV = dot(normal, view);
                    materialSample = nDotV > 0.0 && dot(geometric, view) > 0.0;
                    if (materialSample) {
                        float2 ab = bsdfDfg(material.perceptualRoughness, nDotV);
                        float ggxProbability = materialGgxProbability(material, ab);
                        // Treat the probes' light as uniform, so both lobes
                        // reflect it; dropping the specular one darkened glossy
                        // and metal receivers wherever a ray hit a cold cache.
                        float3 diffuseReflectance, specularReflectance;
                        materialReflectance(material, ab, diffuseReflectance, specularReflectance);
                        fallbackReflectance = diffuseReflectance + specularReflectance;
                        traced = traceMaterialGather(position, normal, geometric, view, material, ab,
                            ggxProbability, rtgiGatherNoise(thread.xy), transportRandom(seed, 0u),
                            diffuse, background, listedEmission);
                        if (params.lightCounts.y != 0u) {
                            uint state = transportHash(seed ^ 0x51ed270bu);
                            transportAdmissionBypass = true;
                            listedEmission += emitterLight(position, normal, geometric, view,
                                false, true, true, material, ab, ggxProbability, state);
                        }
                    }
                } else {
                    traced = traceCascadeGather(position, normal, geometric, seed, diffuse, background);
                }
            }
            if (traced) {
                coverage = backgroundCoverage = 1.0;
            } else {
                diffuse = resolveSplitDiffuse(position, normal, coverage, background, backgroundCoverage, anyCoverage);
            }
            // Direct light from listed emitters, which the probes leave out:
            // one shadow ray a pixel a frame, smoothed by temporal and denoise.
            if (!materialSample && params.lightCounts.y != 0u) {
                uint state = transportHash(at ^ transportHash(params.frame.x * 0x9e3779b9u + 0x51ed270bu));
                transportAdmissionBypass = true;
                diffuse += emitterLight(position, normal, geometric,
                    normal, true, false, true, (SurfaceBsdf)0, 0.0, 0.5, state);
            }
            // Unsupported queries fall back to diffuse probes in the same
            // (material-weighted) units.
            if (materialGather && !traced) {
                diffuse *= fallbackReflectance;
                background *= fallbackReflectance;
            }
            diffuse += listedEmission;
            background += listedEmission;
            // Store BSDF-weighted light per unit reflectance; the composite
            // multiplies each full-resolution pixel by its own reflectance.
            if (materialGather) {
                float3 reflectance = materialSpaceAlbedo(deferred);
                diffuse /= reflectance;
                background /= reflectance;
            }
            supported = coverage > 0.0 && all(isfinite(diffuse));
        }
    }
    irradiance.Store(int2(thread.xy), float4(diffuse,
        rawGiSupportAlpha(eligible, supported, coverage)));
    bool backgroundSupported = eligible && backgroundCoverage > 0.0 && all(isfinite(background));
    backgroundOutput.Store(int2(thread.xy), float4(background,
        rawGiSupportAlpha(eligible, backgroundSupported, backgroundCoverage)));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void worldResolve(uint3 thread : SV_DispatchThreadID) { resolveWorld(thread, false, false); }

[shader("compute")]
[numthreads(8, 8, 1)]
void guidedResolve(uint3 thread : SV_DispatchThreadID) { resolveWorld(thread, true, false); }

[shader("compute")]
[numthreads(8, 8, 1)]
void materialResolve(uint3 thread : SV_DispatchThreadID) { resolveWorld(thread, true, true); }
#endif
