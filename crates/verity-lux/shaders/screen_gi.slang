// What screen-traced GI methods share: the frame, its images, a ray marched
// against the depth buffer, and light accumulated over time and filtered.
// Include after the material headers. Depth is reverse-Z infinite perspective
// and normals are world space. Light is the mean radiance arriving, cosine
// weighted (irradiance / pi).

static const float PI = 3.14159265;
static const int STEPS = 24;
// Frames of history a `HISTORY_PRIOR` estimate counts for.
static const float PRIOR_FRAMES = 2.0;
static const int MAX_MOVED = 16;


struct Frame {
    float4x4 clipFromWorld;
    float4x4 worldFromClip;
    float4x4 prevClipFromWorld;
    float4 camera;       // xyz position, w near plane
    uint4 sizes;         // xy frame, zw trace
    uint4 origin;        // xy where the viewport starts in the targets, zw the half-size reflections
    float4 misc;         // frame index, screen ray distance, intensity, debug
    float4 sunDirection; // toward the sun; w 1 when there is one
    float4 sunColor;     // irradiance / pi
    float4 sky;          // radiance from straight up; w world ray distance
    float4 ground;       // radiance from straight down
    uint4 counts;        // primitives, nodes, point lights, moved regions
    uint4 cache;         // frame, slot mask, on, rays per texel
    float4 extra;        // x unused, y history cap while lights change (0 steady), z history valid, w reflection history valid
    float4 reflection;   // mode (0 off, 1 screen, 2 and the scene), roughness cutoff
    // Light is linear in its sources: when their power changes, everything
    // remembered scales with it, this frame only. Waiting instead would take
    // many frames, as last frame's image feeds this frame's rays.
    float4 fade;
    // Where physics moved something since last frame, grown by how far its
    // shadow and bounce light reach. Light there is stale.
    float4 movedMin[MAX_MOVED];
    float4 movedMax[MAX_MOVED];
};


// A point light: position (w 1 / range^2) and irradiance / pi at a metre.
struct Light {
    float4 position;
    float4 color;
};


[[vk::binding(0, 0)]] ConstantBuffer<Frame> f;
[[vk::binding(1, 0)]] Texture2D<float> depthTexture;
[[vk::binding(2, 0)]] Texture2D<uint4> gbuffer;
[[vk::binding(3, 0)]] Texture2D<float4> colorTexture; // last frame's lit image

[[vk::binding(10, 0)]] StructuredBuffer<Light> lights;
[[vk::binding(11, 0)]] [format("rgba16f")] RWTexture2D<float4> hdr;
[[vk::binding(12, 0)]] [format("rgba16f")] WTexture2D<float4> colorOut;
// build.rs's 64x64 blue-noise tile: four u16 ranks per texel.
[[vk::binding(14, 0)]] StructuredBuffer<uint2> blueNoiseRanks;

// Per-pass inputs and outputs.
[[vk::binding(0, 1)]] Texture2D<float4> in0;
[[vk::binding(1, 1)]] Texture2D<float4> in1;
[[vk::binding(2, 1)]] Texture2D<float4> in2;
[[vk::binding(3, 1)]] Texture2D<float4> in3;
[[vk::binding(4, 1)]] [format("rgba16f")] WTexture2D<float4> out0;
// rg32f: depth passes write (linear depth, packed normal), see `guide`.
[[vk::binding(5, 1)]] [format("rg32f")] WTexture2D<float2> out1;
[[vk::binding(6, 1)]] [format("rgba16f")] WTexture2D<float4> out2;
[[vk::binding(7, 1)]] [format("r32f")] WTexture2D<float> out3;
// The composite's inputs: filtered light and its linear depth, and the
// filtered reflections and their depth.
[[vk::binding(8, 1)]] Texture2D<float4> giLight;
[[vk::binding(9, 1)]] Texture2D<float4> giDepth;
[[vk::binding(10, 1)]] Texture2D<float4> reflectLight;
[[vk::binding(11, 1)]] Texture2D<float4> reflectDepth;

// ---------------------------------------------------------------- view helpers

float2 frameSize() { return float2(f.sizes.xy); }
int2 traceSize() { return int2(f.sizes.zw); }
int2 reflectSize() { return int2(f.origin.zw); }

float linearDepth(float d) { return f.camera.w / d; }

// World point at continuous pixel position pc (texel + 0.5) and device depth.
float3 worldFromPixel(float2 pc, float depth) {
    float2 uv = pc / frameSize();
    float4 w = mul(f.worldFromClip, float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0));
    return w.xyz / w.w;
}

// Continuous pixel position and linear depth of world point x under m.
float3 project(float4x4 m, float3 x) {
    float4 c = mul(m, float4(x, 1.0));
    float2 ndc = c.xy / c.w;
    return float3((ndc.x * 0.5 + 0.5) * frameSize().x, (0.5 - ndc.y * 0.5) * frameSize().y, c.w);
}

bool onScreen(float2 pc) { return all(pc >= 0.0) && all(pc < frameSize()); }

int3 target(int2 p) { return int3(p + int2(f.origin.xy), 0); }
float loadDepth(int2 p) { return depthTexture.Load(target(p)); }
uint4 loadGbuffer(int2 p) { return gbuffer.Load(target(p)); }
float3 lastImage(float2 pc) { return colorTexture.Load(target(int2(pc))).rgb * f.fade.rgb; }

float surfaceDepth(float2 pc) {
    float d = loadDepth(int2(pc));
    return d > 0.0 ? linearDepth(d) : 1e9;
}

int2 framePixel(int2 px, int2 size) {
    return int2((float2(px) + 0.5) * frameSize() / float2(size));
}

float3 decodeNormal(float2 packed) {
    float2 p = packed * 2.0 - 1.0;
    float3 n = float3(p, 1.0 - abs(p.x) - abs(p.y));
    float t = saturate(-n.z);
    n.x += n.x >= 0.0 ? -t : t;
    n.y += n.y >= 0.0 ? -t : t;
    return normalize(n);
}

float3 gbufferNormal(uint4 g) {
    return decodeNormal(float2(float(g.w & 4095u), float((g.w >> 12u) & 4095u)) / 4095.0);
}

// The G-buffer's 24-bit normal as a float, exact in fp32, so a quarter-size
// depth image carries it and filters need not reload the full-size G-buffer.
float packedNormal(uint4 g) { return float(g.w & 0xffffffu); }
float3 guideNormal(float packed) { return gbufferNormal(uint4(0u, 0u, 0u, uint(packed))); }

float3 gbufferBase(uint4 g) {
    float3 base = float3(float(g.x & 255u), float((g.x >> 8u) & 255u), float((g.x >> 16u) & 255u)) / 255.0;
    return pow(base, 2.2);
}

float gbufferMetallic(uint4 g) { return float((g.z >> 8u) & 255u) / 255.0; }
float3 gbufferAlbedo(uint4 g) { return gbufferBase(g) * (1.0 - gbufferMetallic(g)); }
float gbufferRoughness(uint4 g) { return float(g.x >> 24u) / 255.0; }

// Specular reflectance at normal incidence.
float3 gbufferF0(uint4 g) {
    float reflectance = float(g.z & 255u) / 255.0;
    float metallic = gbufferMetallic(g);
    return 0.16 * reflectance * reflectance * (1.0 - metallic) + gbufferBase(g) * metallic;
}

// What a ray leaving the scene along d sees: the renderer's hemisphere.
float3 background(float3 d) { return lerp(f.ground.rgb, f.sky.rgb, d.z * 0.5 + 0.5); }

// A texture read at continuous texel position t (texel centres at +0.5),
// blended from the four nearest texels as a linear sampler clamped to the
// edge would: unfilterable formats share the bindings.
float4 bilinear(Texture2D<float4> tex, float2 t, int2 size) {
    float2 c = t - 0.5;
    int2 b = int2(floor(c));
    float2 fr = c - float2(b);
    float4 a = tex.Load(int3(clamp(b, 0, size - 1), 0));
    float4 bx = tex.Load(int3(clamp(b + int2(1, 0), 0, size - 1), 0));
    float4 by = tex.Load(int3(clamp(b + int2(0, 1), 0, size - 1), 0));
    float4 bxy = tex.Load(int3(clamp(b + int2(1, 1), 0, size - 1), 0));
    return lerp(lerp(a, bx, fr.x), lerp(by, bxy, fr.x), fr.y);
}

// Lit surfaces only: the unlit flag is gbuffer bit 24.
bool receives(float depth, uint4 g) { return depth > 0.0 && (g.w & (1u << 24u)) == 0u; }

float luminance(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

// One very bright hit (a glint on last frame's image) would survive the filter
// as a flickering blob; cap it far above any lit surface.
float3 clampFirefly(float3 c) { return c * min(1.0, 4.0 / max(luminance(c), 1e-4)); }

bool bad(float4 v) { return any(v != v) || any(abs(v) > 1e30); }

float3 safeInv(float3 d) { return 1.0 / select(abs(d) < 1e-20, float3(1e-20), d); }

// Whether physics moved something around x since last frame.
bool movedNear(float3 x) {
    [loop] for (uint i = 0u; i < f.counts.w; i++) {
        if (all(x >= f.movedMin[i].xyz) && all(x <= f.movedMax[i].xyz)) return true;
    }
    return false;
}

// A march step found the ray behind the depth buffer: bisect the crossing so
// the hit doesn't glitter from texel to texel and frame to frame.
float refineHit(float3 origin, float3 dir, float inFront, float behind) {
    [loop] for (int i = 0; i < 6; i++) {
        float mid = 0.5 * (inFront + behind);
        float3 q = project(f.clipFromWorld, origin + dir * mid);
        float2 qp = clamp(q.xy, 0.0, frameSize() - 1.0);
        if (q.z > surfaceDepth(qp)) {
            behind = mid;
        } else {
            inFront = mid;
        }
    }
    return behind;
}

void tangentFrame(float3 n, out float3 t, out float3 b) {
    t = normalize(cross(n, abs(n.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0)));
    b = cross(n, t);
}

float3 aroundNormal(float3 n, float3 v) {
    float3 t, b;
    tangentFrame(n, t, b);
    return normalize(t * v.x + b * v.y + n * v.z);
}

float3 cosineDir(float3 n, float u, float phi) {
    return aroundNormal(n, float3(sqrt(u) * cos(phi), sqrt(u) * sin(phi), sqrt(max(0.0, 1.0 - u))));
}

// Interleaved gradient noise, stepped along an R2 sequence per frame and ray.
float ign(float2 p) { return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715)))); }

// Four independent blue-noise values in [0, 1) for a pixel, which callers step
// along golden-ratio sequences per frame and ray.
float4 blueNoise(int2 px) {
    uint2 w = blueNoiseRanks[uint(px.x & 63) + uint(px.y & 63) * 64u];
    return (float4(w.x & 0xffffu, w.x >> 16, w.y & 0xffffu, w.y >> 16) + 0.5) / 4096.0;
}

// A material's diffuse colour averaged over its surface, for hits that carry no
// texture coordinates. Metallic comes from the texture too: glTF's factor is
// usually 1, scaling a texture that says where the metal is.
float3 meanAlbedo(Material m) {
    float3 base = m.base_color.rgb;
    float metallic = m.params.x;
    if (m.textures.x != NO_TEXTURE)
        base *= textures[NonUniformResourceIndex(m.textures.x)].SampleLevel(material_sampler, float2(0.5), 16.0).rgb;
    if (m.textures.y != NO_TEXTURE)
        metallic *= textures[NonUniformResourceIndex(m.textures.y)].SampleLevel(material_sampler, float2(0.5), 16.0).b;
    return base * (1.0 - metallic);
}

// What of a lit pixel a diffuse ray should see. A sun glint is for this view
// alone, and left in, it brightens what bounces off every shiny surface.
float3 diffuseLit(float3 lit, Brdf b, float roughness, float3 n, float3 view) {
    float3 sun = f.sunDirection.xyz;
    if (f.sunDirection.w < 0.5 || dot(n, sun) <= 0.0) return lit;
    float3 full = brdf_value(b, n, view, sun);
    return lit * saturate(brdf_diffuse(b, n, view, sun) / max(full, 1e-6));
}

bool cameraStill() {
    return all(f.clipFromWorld[0] == f.prevClipFromWorld[0])
        && all(f.clipFromWorld[1] == f.prevClipFromWorld[1])
        && all(f.clipFromWorld[2] == f.prevClipFromWorld[2])
        && all(f.clipFromWorld[3] == f.prevClipFromWorld[3]);
}

// The a-trous kernel's taps, by distance from the centre.
float kernelTap(int i) { return i == 0 ? 0.375 : (i == 1 ? 0.25 : 0.0625); }


uint hashU(uint x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

float random01(inout uint seed) {
    seed = hashU(seed);
    return float(seed >> 8u) / 16777216.0;
}

// Up to this many point lights are all shaded, each with its own shadow ray.
// Past it, a point is lit by one light picked by resampled importance sampling
// (Talbot et al. 2005), and callers keep a running mean of what it gives.
static const uint EXACT_LIGHTS = 8u;
static const uint LIGHT_CANDIDATES = 16u;

bool sampledLights() { return f.counts.z > EXACT_LIGHTS; }

// Unshadowed irradiance / pi from point light i at x facing n, and the ray
// that tests its shadow.
float3 pointLight(uint i, float3 x, float3 n, out float3 l, out float reach) {
    Light source = lights[i];
    float3 to = source.position.xyz - x;
    float d2 = max(dot(to, to), 1e-2);
    l = to * rsqrt(d2);
    reach = sqrt(d2);
    float w = saturate(1.0 - pow(d2 * source.position.w, 2.0));
    return source.color.rgb * (max(dot(n, l), 0.0) * w * w / d2);
}

// One point light at x picked from uniform candidates in proportion to its
// unshadowed luminance, weighted so that, shadowed, it is an unbiased estimate
// of all of them. 0 when no candidate reaches x.
float3 pickPointLight(float3 x, float3 n, inout uint seed, out float3 l, out float reach) {
    uint count = f.counts.z;
    float total = 0.0, chosenTarget = 0.0;
    float3 chosen = 0.0;
    l = n;
    reach = 0.0;
    [loop] for (uint k = 0u; k < LIGHT_CANDIDATES; k++) {
        uint i = min(uint(random01(seed) * float(count)), count - 1u);
        float3 cl;
        float cr;
        float3 c = pointLight(i, x, n, cl, cr);
        float target = dot(c, float3(0.2126, 0.7152, 0.0722));
        if (target <= 0.0) continue;
        total += target;
        if (random01(seed) * total < target) {
            chosen = c;
            chosenTarget = target;
            l = cl;
            reach = cr;
        }
    }
    if (chosenTarget <= 0.0) return 0.0;
    // Candidate weights are target / (1 / count); W = sum / (M * target).
    return chosen * (total * float(count) / (float(LIGHT_CANDIDATES) * chosenTarget));
}

// All tracing passes bind the previous diffuse linear depth as in1. Reject
// previous colour when another surface occupied the projected location.
bool previousSurface(float3 p, out float2 pc) {
    pc = 0.0;
    if (f.extra.z < 0.5) return false;
    float3 pp = project(f.prevClipFromWorld, p);
    if (pp.z <= f.camera.w || !onScreen(pp.xy)) return false;
    int2 size = traceSize();
    int2 tp = clamp(int2(pp.xy * float2(size) / frameSize()), 0, size - 1);
    float depth = in1.Load(int3(tp, 0)).r;
    if (depth <= 0.0 || abs(depth - pp.z) >= 0.05 * pp.z) return false;
    pc = pp.xy;
    return true;
}


// A ray marched against the depth buffer in `steps` steps growing to reach;
// on a hit, the light that point had in last frame's image (w 1).
float4 march(float3 origin, float3 dir, float reach, int steps, float jitter) {
    float last = 0.0;
    [loop] for (int s = 0; s < steps; s++) {
        float along = reach * pow((float(s) + jitter) / float(steps), 2.0);
        float inFront = last;
        last = along;
        // How far this step went: a surface thinner than that could be stepped over.
        float stride = reach * 2.0 * (float(s) + jitter + 1.0) / float(steps * steps);
        float3 q = project(f.clipFromWorld, origin + dir * along);
        if (q.z < f.camera.w || !onScreen(q.xy)) break;
        float sd = loadDepth(int2(q.xy));
        if (sd <= 0.0) continue;
        float surface = linearDepth(sd);
        float behind = q.z - surface;
        if (behind > 0.0 && behind < max(0.1 + 0.02 * surface, 1.5 * stride)) {
            float3 hq = project(f.clipFromWorld, origin + dir * refineHit(origin, dir, inFront, along));
            float2 hp = clamp(hq.xy, 0.0, frameSize() - 1.0);
            float hd = loadDepth(int2(hp));
            float3 hit = worldFromPixel(floor(hp) + 0.5, hd > 0.0 ? hd : sd);
            float2 pc;
            if (previousSurface(hit, pc)) return float4(lastImage(pc), 1.0);
            break;
        }
    }
    return 0.0;
}


// This frame's rays blended into what the same surface accumulated before
// (reprojected onto last frame; a different depth there starts over). Light
// that just changed keeps less history: the whole view when a light did, and
// around whatever physics moved.
// in0 trace, in1 previous light, in2 previous depth, in3 previous moments;
// out0 light, out1 linear depth, out2 moments, out3 variance.
[shader("compute")]
[numthreads(8, 8, 1)]
void temporal(uint3 id : SV_DispatchThreadID) {
    int2 px = int2(id.xy);
    int2 size = traceSize();
    if (any(px >= size)) return;
    int2 fp = framePixel(px, size);
    float depth = loadDepth(fp);
    uint4 g = loadGbuffer(fp);
    if (!receives(depth, g)) {
        out0.Store(px, 0.0);
        out1.Store(px, 0.0);
        out2.Store(px, 0.0);
        out3.Store(px, 0.0);
        return;
    }
    float3 p = worldFromPixel(float2(fp) + 0.5, depth);
    float4 now = in0.Load(int3(px, 0));
    float l = luminance(now.rgb);
    float4 result = now;
    float3 m = float3(l, l * l, 1.0); // mean, mean of the square, frames of history
#ifdef HISTORY_PRIOR
    // No history yet: a method's world-space estimate stands in for a couple
    // of frames, so a texel coming into view starts near the answer.
    float4 prior;
    if (historyPrior(p, gbufferNormal(g), prior)) {
        result = lerp(prior, now, 1.0 / (PRIOR_FRAMES + 1.0));
        float lp = luminance(prior.rgb);
        m = float3(lerp(float2(lp, lp * lp), float2(l, l * l), 1.0 / (PRIOR_FRAMES + 1.0)), PRIOR_FRAMES + 1.0);
    }
#endif
    if (f.extra.z > 0.5) {
        float3 pp = project(f.prevClipFromWorld, p);
        if (onScreen(pp.xy)) {
            // Depth nearest, light and moments bilinear, as Godot samples them.
            // A texel's light belongs to the pixel at its centre, so a still view
            // lands exactly on it. Scaling the pixel's position instead lands an
            // eighth of a texel off, which drags history across the image.
            float2 at = (pp.xy - 0.5) * float2(size) / frameSize();
            int2 tp = clamp(int2(at), 0, size - 1);
            float was = in2.Load(int3(tp, 0)).r;
            if (abs(was - pp.z) < 0.05 * pp.z) {
                float4 wasLight = bilinear(in1, at, size);
                float4 wasM = bilinear(in3, at, size);
                if (!bad(wasLight) && !bad(wasM)) {
                    float fade = luminance(f.fade.rgb);
                    wasLight.rgb *= f.fade.rgb;
                    wasM.xy *= float2(fade, fade * fade);
                    // A surface just uncovered takes its first frames at full weight.
                    // A still camera has nothing to smear, so it can average longer.
                    float steady = cameraStill() ? 48.0 : 32.0;
                    float limit = f.extra.y > 0.0 ? f.extra.y : (movedNear(p) ? 4.0 : steady);
                    float frames = min(wasM.z + 1.0, limit);
                    float alpha = 1.0 / frames;
                    result = lerp(wasLight, now, alpha);
                    m = float3(lerp(wasM.xy, float2(l, l * l), alpha), frames);
                }
            }
        }
    }
    // The filter stops at edges by the noise left in the accumulated light: one
    // frame's variance over the frames averaged. A young history hasn't seen
    // enough frames to tell: taken as noisy.
    float variance = max(m.y - m.x * m.x, 0.0) / m.z;
    if (m.z < 4.0) variance = max(variance, 0.25 * m.x * m.x + 0.01);
    out0.Store(px, result);
    out1.Store(px, float2(linearDepth(depth), packedNormal(g)));
    out2.Store(px, float4(m, 0.0));
    out3.Store(px, variance);
}

// One pass of an edge-aware a-trous filter: neighbours on the same plane and
// facing the same way, averaged more where still noisy.
// in0 light, in1 variance, in2 depth and normal; out0 light, out1 variance.
void atrous(int2 px, int step) {
    int2 size = traceSize();
    if (any(px >= size)) return;
    float4 c = in0.Load(int3(px, 0));
    float v = in1.Load(int3(px, 0)).r;
    float2 guide = in2.Load(int3(px, 0)).rg;
    float z = guide.x;
    if (z <= 0.0) {
        out0.Store(px, c);
        out1.Store(px, v);
        return;
    }
    int2 fp = framePixel(px, size);
    float3 pos = worldFromPixel(float2(fp) + 0.5, f.camera.w / z);
    float3 n = guideNormal(guide.y);
    float lc = luminance(c.rgb);
    float sigma = 4.0 * sqrt(v) + 1e-3;
    float4 sum = 0.0;
    float weights = 0.0;
    float varSum = 0.0;
    [loop] for (int y = -2; y <= 2; y++) {
        [loop] for (int x = -2; x <= 2; x++) {
            int2 q = px + int2(x, y) * step;
            if (any(q < 0) || any(q >= size)) continue;
            float2 gq = in2.Load(int3(q, 0)).rg;
            float zq = gq.x;
            if (zq <= 0.0) continue;
            int2 fq = framePixel(q, size);
            float3 nq = guideNormal(gq.y);
            float4 cq = in0.Load(int3(q, 0));
            float plane = abs(dot(n, worldFromPixel(float2(fq) + 0.5, f.camera.w / zq) - pos));
            float w = kernelTap(abs(x)) * kernelTap(abs(y));
            w *= exp(-plane / (0.02 + 0.01 * z));
            w *= pow(max(dot(n, nq), 0.0), 32.0);
            w *= exp(-abs(luminance(cq.rgb) - lc) / sigma);
            sum += cq * w;
            weights += w;
            varSum += in1.Load(int3(q, 0)).r * w * w;
        }
    }
    out0.Store(px, weights > 1e-6 ? sum / weights : c);
    out1.Store(px, weights > 1e-6 ? varSum / (weights * weights) : v);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void atrous1(uint3 id : SV_DispatchThreadID) { atrous(int2(id.xy), 1); }

[shader("compute")]
[numthreads(8, 8, 1)]
void atrous2(uint3 id : SV_DispatchThreadID) { atrous(int2(id.xy), 2); }

[shader("compute")]
[numthreads(8, 8, 1)]
void atrous4(uint3 id : SV_DispatchThreadID) { atrous(int2(id.xy), 4); }


// The filtered quarter-size light around full-size pixel fp, on the same
// surface (facing n at linear depth d): rgb premultiplied by the fraction of
// rays that hit, a that fraction.
float4 upsampleGi(int2 fp, float3 n, float d) {
    int2 size = traceSize();
    float2 tc = float2(fp) * float2(size) / frameSize() - 0.5;
    int2 base = int2(floor(tc));
    float4 sum = 0.0;
    float weights = 0.0;
    [loop] for (int y = -1; y <= 2; y++) {
        [loop] for (int x = -1; x <= 2; x++) {
            int2 t = clamp(base + int2(x, y), 0, size - 1);
            float2 tg = giDepth.Load(int3(t, 0)).rg;
            float tz = tg.x;
            if (tz <= 0.0) continue;
            float3 tn = guideNormal(tg.y);
            float2 o = float2(t) - tc;
            float w = exp(-dot(o, o) * 0.5) * exp(-abs(tz - d) / (0.03 * d + 0.01)) * pow(max(dot(tn, n), 0.0), 8.0);
            sum += giLight.Load(int3(t, 0)) * w;
            weights += w;
        }
    }
    if (weights <= 1e-4) {
        // A sliver no texel shares (a face seen edge on): the texels at a
        // similar depth, whichever way they face.
        [loop] for (int y = -1; y <= 2; y++) {
            [loop] for (int x = -1; x <= 2; x++) {
                int2 t = clamp(base + int2(x, y), 0, size - 1);
                float tz = giDepth.Load(int3(t, 0)).r;
                if (tz <= 0.0) continue;
                float2 o = float2(t) - tc;
                float w = exp(-dot(o, o) * 0.5) * exp(-abs(tz - d) / (0.1 * d + 0.05));
                sum += giLight.Load(int3(t, 0)) * w;
                weights += w;
            }
        }
    }
    return weights > 1e-4 ? sum / weights : float4(0.0);
}


// The final image, for reading back on the CPU.
[[vk::binding(13, 0)]] RWStructuredBuffer<float4> captured;

[shader("compute")]
[numthreads(8, 8, 1)]
void capture(uint3 id : SV_DispatchThreadID) {
    int2 fp = int2(id.xy);
    if (any(fp >= int2(f.sizes.xy))) return;
    captured[fp.y * int(f.sizes.x) + fp.x] = hdr[target(fp).xy];
}
