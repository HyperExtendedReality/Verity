// Temporal anti-aliasing. The projection moves by a sub-pixel Halton(2, 3)
// offset each frame (`taa.rs`) and the samples accumulate in a history
// reprojected through the depth buffer, so a still image converges to an
// 8x supersampled one. Two tiers share the pass:
//
//  Desktop: 3x3 neighbourhood variance clipping in YCoCg (Salvi 2016) inside
//    the min/max box, clipping toward its centre (Karis 2014), motion from the
//    nearest depth in 3x3 so edges keep their history, and a 5-tap
//    Catmull-Rom history fetch (Jimenez 2016) that stays sharp under motion.
//  Mobile: the 5-tap cross, a YCoCg min/max clamp and one bilinear history
//    tap: the single-pass shape of mobile temporal resolves such as Arm ASR
//    and Snapdragon GSR 2, at about half the texture traffic.
//
// Both tiers blend in a luma-compressed space so HDR highlights do not
// flicker, and against ghosting raise this frame's weight with velocity, with
// how far the clip moved the history, and to all of it where the history's
// depth shows a disocclusion, and by the reactive mask in HDR alpha above 1
// for passes without motion. The history's alpha holds that depth. The copy
// back to HDR applies contrast-adaptive sharpening (AMD CAS) to undo the
// resampling blur. Motion comes from the resolve, per object; the sky, with
// no surface there, is reprojected through the camera.

#include "lib/common.slang"

struct Params {
    /// x: bit 0 for the desktop tier, bit 1 when MSAA already smoothed the
    /// edges; y: 1 when the history is valid;
    /// z: sharpening from 0 to 1, as float bits.
    uint4 frame;
};
[[vk::push_constant]] ConstantBuffer<Params> params;

[[vk::binding(33)]] Texture2D<float> scene_depth;
[[vk::binding(0, 1)]] Texture2D<float4> history;
[[vk::binding(1, 1)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> accumulated;
[[vk::binding(2, 1)]] SamplerState clamp_sampler;

/// Weight of this frame against a still, matching history.
static const float CURRENT_WEIGHT = 0.1;
/// ...with MSAA, where history only has to smooth shading: less ghosting.
static const float MSAA_WEIGHT = 0.2;
/// ...rising to this at `FAST_PIXELS` of motion per frame.
static const float MOVING_WEIGHT = 0.25;
static const float FAST_PIXELS = 16.0;
/// Relative depth change that marks history as another surface.
static const float DISOCCLUSION = 0.1;
/// CAS's negative lobe at full sharpening.
static const float SHARPEST = 0.2;
/// Largest finite rgba16f value: the sky's linear depth.
static const float FAR_DEPTH = 65000.0;

static const int2 OFFSETS[9] = {
    int2(0, 0), int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1),
    int2(1, 1), int2(-1, 1), int2(1, -1), int2(-1, -1),
};

/// Reversible tonemap: blending here weights each sample by 1 / (1 + luma).
float3 compress(float3 c) {
    return c / (1.0 + max(c.r, max(c.g, c.b)));
}

float3 expand(float3 c) {
    return c / max(1.0 - max(c.r, max(c.g, c.b)), 1.0e-4);
}

float3 to_ycocg(float3 c) {
    return float3(dot(c, float3(0.25, 0.5, 0.25)), dot(c, float3(0.5, 0.0, -0.5)), dot(c, float3(-0.25, 0.5, -0.25)));
}

float3 from_ycocg(float3 c) {
    return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

/// Pull `value` along the line to the box centre until it is inside the box.
float3 clip_box(float3 lo, float3 hi, float3 value) {
    float3 centre = 0.5 * (hi + lo);
    float3 extent = 0.5 * (hi - lo) + 1.0e-5;
    float3 offset = value - centre;
    float3 units = abs(offset / extent);
    float largest = max(units.x, max(units.y, units.z));
    return largest > 1.0 ? centre + offset / largest : value;
}

/// Bicubic Catmull-Rom from five bilinear taps, the corners dropped.
float3 catmull_rom(float2 uv, float2 size) {
    float2 position = uv * size;
    float2 centre = floor(position - 0.5) + 0.5;
    float2 f = position - centre;
    float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    float2 w3 = f * f * (-0.5 + 0.5 * f);
    float2 w12 = w1 + w2;
    float2 t0 = (centre - 1.0) / size;
    float2 t3 = (centre + 2.0) / size;
    float2 t12 = (centre + w2 / w12) / size;
    float3 sum = history.SampleLevel(clamp_sampler, float2(t12.x, t0.y), 0).rgb * (w12.x * w0.y)
        + history.SampleLevel(clamp_sampler, float2(t0.x, t12.y), 0).rgb * (w0.x * w12.y)
        + history.SampleLevel(clamp_sampler, t12, 0).rgb * (w12.x * w12.y)
        + history.SampleLevel(clamp_sampler, float2(t3.x, t12.y), 0).rgb * (w3.x * w12.y)
        + history.SampleLevel(clamp_sampler, float2(t12.x, t3.y), 0).rgb * (w12.x * w3.y);
    float weight = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    // Negative lobes can ring below zero next to bright edges.
    return max(sum / weight, 0.0);
}

/// Blend this frame's HDR into the history; writes the next history.
[shader("compute")]
[numthreads(8, 8, 1)]
void resolveMain(uint3 id: SV_DispatchThreadID) {
    float4 viewport = globals.viewport;
    if (any(float2(id.xy) >= viewport.zw))
        return;
    bool desktop = (params.frame.x & 1u) != 0u;
    float still_weight = (params.frame.x & 2u) != 0u ? MSAA_WEIGHT : CURRENT_WEIGHT;
    int2 origin = int2(viewport.xy);
    int2 last = int2(viewport.zw) - 1;
    int2 local = int2(id.xy);
    float4 current = hdr.Load(int3(origin + local, 0));
    float3 color = compress(current.rgb);

    // Neighbourhood moments and box, and the nearest surface (reverse-Z: largest depth).
    float3 m1 = 0.0;
    float3 m2 = 0.0;
    float3 lo = 1.0e9;
    float3 hi = -1.0e9;
    float depth = -1.0;
    int2 nearest = local;
    uint count = desktop ? 9u : 5u;
    for (uint i = 0u; i < count; ++i) {
        int2 p = clamp(local + OFFSETS[i], int2(0, 0), last);
        float3 c = compress(hdr.Load(int3(origin + p, 0)).rgb);
        c = to_ycocg(c);
        m1 += c;
        m2 += c * c;
        lo = min(lo, c);
        hi = max(hi, c);
        float d = scene_depth.Load(int3(origin + p, 0));
        if (d > depth) {
            depth = d;
            nearest = p;
        }
    }

    // Reproject in homogeneous space: sky at depth 0 is a point at infinity
    // and still lands where the camera's rotation puts it.
    float2 nearest_uv = (float2(nearest) + 0.5) / viewport.zw;
    float4 world = mul(globals.inverse_view_proj, float4(nearest_uv * 2.0 - 1.0, depth, 1.0));
    float4 previous = mul(globals.previous_view_proj, world);
    // Surfaces carry their own motion from the resolve, the sky only the camera's.
    float2 motion = depth > 0.0 ? gi_motion[origin + nearest]
                                : nearest_uv - (previous.xy / previous.w * 0.5 + 0.5);
    float2 history_uv = (float2(local) + 0.5) / viewport.zw - motion;

    // Reverse-Z with an infinite far plane: view depth is near / depth.
    float linear_depth = depth > 0.0 ? min(globals.projection.z / depth, FAR_DEPTH) : FAR_DEPTH;
    float3 result = color;
    bool valid = params.frame.y != 0u && previous.w > 0.0
        && all(history_uv >= 0.0) && all(history_uv <= 1.0);
    if (valid) {
        float3 sampled = desktop
            ? catmull_rom(history_uv, viewport.zw)
            : history.SampleLevel(clamp_sampler, history_uv, 0).rgb;
        float3 old = to_ycocg(compress(sampled));
        float3 kept;
        if (desktop) {
            float3 mean = m1 / 9.0;
            float3 sigma = sqrt(max(m2 / 9.0 - mean * mean, 0.0));
            kept = clip_box(max(lo, mean - sigma), min(hi, mean + sigma), old);
        } else {
            kept = clamp(old, lo, hi);
        }
        float speed = length(motion * viewport.zw);
        float weight = lerp(still_weight, MOVING_WEIGHT, saturate(speed / FAST_PIXELS));
        // A large luma correction means the history showed something else.
        weight = max(weight, saturate(abs(old.x - kept.x) * 2.0));
        // Reactive mask: passes without motion vectors write alpha 1 + r to
        // keep at most 1 - r of the history.
        weight = max(weight, saturate(current.a - 1.0));
        if (depth > 0.0) {
            // The surface's depth last frame against what the history saw there,
            // from the camera alone: an object rushing at it only rejects early.
            float expected = previous.w / world.w;
            int2 texel = clamp(int2(history_uv * viewport.zw), int2(0, 0), last);
            float seen = history.Load(int3(texel, 0)).a;
            if (abs(seen - expected) > DISOCCLUSION * expected)
                weight = 1.0;
        }
        result = lerp(from_ycocg(kept), color, weight);
    }
    accumulated[id.xy] = float4(expand(result), linear_depth);
}

/// Copy the new history into the HDR target for the composite, sharpened
/// with contrast-adaptive sharpening; HDR keeps its alpha.
[shader("compute")]
[numthreads(8, 8, 1)]
void applyMain(uint3 id: SV_DispatchThreadID) {
    float4 viewport = globals.viewport;
    if (any(float2(id.xy) >= viewport.zw))
        return;
    int2 local = int2(id.xy);
    int2 last = int2(viewport.zw) - 1;
    float3 centre = compress(history.Load(int3(local, 0)).rgb);
    float3 cross = 0.0;
    float3 lo = centre;
    float3 hi = centre;
    for (uint i = 1u; i < 5u; ++i) {
        float3 c = compress(history.Load(int3(clamp(local + OFFSETS[i], int2(0, 0), last), 0)).rgb);
        cross += c;
        lo = min(lo, c);
        hi = max(hi, c);
    }
    // Sharpen less where the neighbourhood already spans the full range.
    float3 amount = sqrt(saturate(min(lo, 1.0 - hi) / max(hi, 1.0e-4)));
    float3 w = amount * (-SHARPEST * saturate(asfloat(params.frame.z)));
    float3 sharpened = saturate((centre + cross * w) / (1.0 + 4.0 * w));
    uint2 pixel = id.xy + uint2(viewport.xy);
    hdr_output[pixel] = float4(expand(sharpened), hdr_output[pixel].a);
}
