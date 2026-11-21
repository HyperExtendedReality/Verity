// Temporal anti-aliasing. The projection moves by a sub-pixel Halton(2, 3)
// offset each frame (`taa.rs`) and the samples accumulate in a history
// reprojected through the depth buffer, so a still image converges to an
// 16x supersampled one. Two tiers share the pass:
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
// flicker, and against ghosting raise this frame's weight with velocity, to
// all of it where no history texel in 3x3 saw this pixel's depth (a
// disocclusion), and by the reactive mask in HDR alpha above 1
// for passes without motion. The history's alpha holds that depth. The copy
// back to HDR applies contrast-adaptive sharpening (AMD CAS) to undo the
// resampling blur. Motion comes from the resolve, per object; the sky, with
// no surface there, is reprojected through the camera.
//
// This frame's colour is reconstructed at the pixel centre from the
// neighbourhood with a Blackman-Harris window around the jittered sample
// positions (Karis 2014), so it does not wobble with the jitter where history
// is rejected. The variance clip tightens with speed, where stale history
// shows as ghosting, and stays wide when still, where clipping shows as flicker.
//
// Every pass reads its 3x3 neighbourhoods from a group-shared tile, each
// texel loaded and converted once per group instead of once per neighbour.

#include "lib/common.slang"

struct Params {
    /// x: bit 0 for the desktop tier, bit 1 when MSAA already smoothed the
    /// edges, bit 2 for SMAA T2x (see `smaa.slang`); y: 1 when the history is valid;
    /// z: sharpening from 0 to 1, as float bits; w: frames in the history.
    uint4 frame;
    /// xy: this frame's sub-pixel jitter in pixels (`taa.rs`).
    float4 jitter;
};
[[vk::push_constant]] ConstantBuffer<Params> params;

[[vk::binding(33)]] Texture2D<float> scene_depth;
[[vk::binding(0, 1)]] Texture2D<float4> history;
[[vk::binding(1, 1)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> accumulated;
[[vk::binding(2, 1)]] SamplerState clamp_sampler;
// SMAA T2x only: edges (r: left, g: top) and blend weights (x: pull from
// above, y: above pulls from here, z: pull from the left, w: left pulls from here).
[[vk::binding(3, 1)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> smaa_edges;
[[vk::binding(4, 1)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> smaa_weights;
/// `accumulated`, sampled: in SMAA T2x's apply pass, last frame's result.
[[vk::binding(5, 1)]] Texture2D<float4> previous_smaa;
// The TAA tiers reuse SMAA's two images, which swap each frame (`taa.rs`),
// for each history texel's length in frames / 255 (r): written through the
// edges binding, read back next frame through the weights binding.
#define length_out smaa_edges
#define length_in smaa_weights

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
/// Variance clip width in standard deviations when still and at `FAST_PIXELS`.
static const float STILL_GAMMA = 1.25;
static const float MOVING_GAMMA = 0.75;
/// Blackman-Harris window, approximated as exp(-x² * this) (Karis 2014).
static const float FILTER_FALLOFF = 2.29;
static const uint GROUP = 8u;
/// Longest history length counted, in frames; it fits an 8-bit channel.
static const float MAX_LENGTH = 255.0;

static const int2 OFFSETS[9] = {
    int2(0, 0), int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1),
    int2(1, 1), int2(-1, 1), int2(1, -1), int2(-1, -1),
};

/// Reversible tonemap: blending here weights each sample by 1 / (1 + luma).
float3 compress(float3 c) {
    c = max(c, 0.0);
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

/// This group's pixels and a one-pixel border: compressed YCoCg colour and
/// reverse-Z depth.
static const uint TILE = GROUP + 2u;
groupshared float4 tile[TILE * TILE];

/// Fill `tile` from HDR or, with `from_history`, from the history (both
/// clamped to the viewport); depth only when `with_depth`. Every thread in the
/// group must call it, before any returns.
void load_tile(uint2 group, uint index, bool from_history, bool with_depth) {
    float4 viewport = globals.viewport;
    int2 origin = int2(viewport.xy);
    int2 last = int2(viewport.zw) - 1;
    int2 corner = int2(group * GROUP) - 1;
    for (uint i = index; i < TILE * TILE; i += GROUP * GROUP) {
        int2 p = clamp(corner + int2(i % TILE, i / TILE), int2(0, 0), last);
        float3 c = from_history ? history.Load(int3(p, 0)).rgb : hdr.Load(int3(origin + p, 0)).rgb;
        float d = with_depth ? scene_depth.Load(int3(origin + p, 0)) : 0.0;
        tile[i] = float4(to_ycocg(compress(c)), d);
    }
    GroupMemoryBarrierWithGroupSync();
}

/// The tile entry `offset` from this thread's pixel.
float4 tile_at(uint2 thread, int2 offset) {
    int2 t = int2(thread) + 1 + offset;
    return tile[t.y * TILE + t.x];
}

/// Whether no history texel around `history_uv` saw this pixel's surface
/// (reverse-Z `own` depth) at the depth the camera expects there. Any of 3x3
/// counts, and the histories hold each pixel's own depth, so a sub-pixel pole
/// this frame's jitter missed does not read as a disocclusion and drop the
/// history that held it: that dropping is what made thin, distant geometry shimmer.
bool disoccluded(float own, int2 local, float2 history_uv, bool smaa) {
    float4 viewport = globals.viewport;
    float2 uv = (float2(local) + 0.5) / viewport.zw;
    float4 world = mul(globals.inverse_view_proj, float4(uv * 2.0 - 1.0, own, 1.0));
    float expected = mul(globals.previous_view_proj, world).w / world.w;
    int2 last = int2(viewport.zw) - 1;
    int2 texel = int2(floor(history_uv * viewport.zw));
    for (uint i = 0u; i < 9u; ++i) {
        int2 p = clamp(texel + OFFSETS[i], int2(0, 0), last);
        float seen = smaa ? previous_smaa.Load(int3(p, 0)).a : history.Load(int3(p, 0)).a;
        if (abs(seen - expected) <= DISOCCLUSION * expected)
            return false;
    }
    return true;
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

/// Frames the history holds at `uv`, at most the frames since its reset
/// (the stored lengths are stale or undefined beyond that).
float history_frames(float2 uv) {
    int2 last = int2(globals.viewport.zw) - 1;
    int2 p = clamp(int2(uv * globals.viewport.zw), int2(0, 0), last);
    return min(round(length_in[p].r * 255.0), float(params.frame.w));
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
void resolveMain(
    uint3 id: SV_DispatchThreadID,
    uint3 group: SV_GroupID,
    uint3 thread: SV_GroupThreadID,
    uint index: SV_GroupIndex
) {
    float4 viewport = globals.viewport;
    if ((params.frame.x & 4u) != 0u) {
        if (all(float2(id.xy) < viewport.zw))
            smaa_blend(int2(id.xy));
        return;
    }
    load_tile(group.xy, index, false, true);
    if (any(float2(id.xy) >= viewport.zw))
        return;
    bool desktop = (params.frame.x & 1u) != 0u;
    float still_weight = (params.frame.x & 2u) != 0u ? MSAA_WEIGHT : CURRENT_WEIGHT;
    int2 origin = int2(viewport.xy);
    int2 last = int2(viewport.zw) - 1;
    int2 local = int2(id.xy);
    float4 current = hdr.Load(int3(origin + local, 0));
    float2 jitter = params.jitter.xy;

    // Neighbourhood moments and box, this frame's colour reconstructed at the
    // pixel centre, and the nearest surface (reverse-Z: largest depth).
    float3 m1 = 0.0;
    float3 m2 = 0.0;
    float3 lo = 1.0e9;
    float3 hi = -1.0e9;
    float3 filtered = 0.0;
    float filter_sum = 0.0;
    float depth = -1.0;
    int2 nearest = local;
    uint count = desktop ? 9u : 5u;
    for (uint i = 0u; i < count; ++i) {
        float4 s = tile_at(thread.xy, OFFSETS[i]);
        float3 c = s.rgb;
        m1 += c;
        m2 += c * c;
        lo = min(lo, c);
        hi = max(hi, c);
        // The neighbour's sample lies at its offset less the jitter.
        float2 d = float2(OFFSETS[i]) - jitter;
        float w = exp(-FILTER_FALLOFF * dot(d, d));
        filtered += c * w;
        filter_sum += w;
        if (s.a > depth) {
            depth = s.a;
            nearest = clamp(local + OFFSETS[i], int2(0, 0), last);
        }
    }
    float3 color = from_ycocg(filtered / filter_sum);

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
    float own = tile_at(thread.xy, int2(0, 0)).a;
    float linear_depth = own > 0.0 ? min(globals.projection.z / own, FAR_DEPTH) : FAR_DEPTH;
    float3 result = color;
    // Frames the new history holds, in effect: one where it restarts.
    float frames = 1.0;
    bool valid = params.frame.y != 0u && (depth > 0.0 || previous.w > 0.0)
        && all(history_uv >= 0.0) && all(history_uv <= 1.0);
    if (valid) {
        float3 sampled = desktop
            ? catmull_rom(history_uv, viewport.zw)
            : history.SampleLevel(clamp_sampler, history_uv, 0).rgb;
        float3 old = to_ycocg(compress(sampled));
        float speed = saturate(length(motion * viewport.zw) / FAST_PIXELS);
        float3 kept;
        if (desktop) {
            float3 mean = m1 / 9.0;
            float3 sigma = sqrt(max(m2 / 9.0 - mean * mean, 0.0)) * lerp(STILL_GAMMA, MOVING_GAMMA, speed);
            kept = clip_box(max(lo, mean - sigma), min(hi, mean + sigma), old);
        } else {
            kept = clamp(old, lo, hi);
        }
        // The clip alone answers history that showed something else; raising
        // this frame's weight by how far it moved also threw away sub-pixel
        // features each frame the jitter missed them, so they shimmered.
        float weight = lerp(still_weight, MOVING_WEIGHT, speed);
        // A young history averages evenly: with only the fixed weight, its
        // first, noisy frame would linger as 0.9^n of the image for many
        // frames, after a reset and wherever a disocclusion restarted it.
        weight = max(weight, 1.0 / (history_frames(history_uv) + 1.0));
        // Reactive mask: passes without motion vectors write alpha 1 + r to
        // keep at most 1 - r of the history.
        weight = max(weight, saturate(current.a - 1.0));
        // From the camera alone: an object rushing at a surface only rejects early.
        if (own > 0.0 && disoccluded(own, local, history_uv, false))
            weight = 1.0;
        result = lerp(from_ycocg(kept), color, weight);
        frames = 1.0 / weight;
    }
    result = clamp(result, 0.0, 1.0 - 1.0e-4);
    accumulated[id.xy] = float4(expand(result), linear_depth);
    length_out[id.xy] = float4(min(round(frames), MAX_LENGTH) / 255.0, 0.0, 0.0, 0.0);
}

/// Copy the new history into the HDR target for the composite, sharpened
/// with contrast-adaptive sharpening; HDR keeps its alpha.
[shader("compute")]
[numthreads(8, 8, 1)]
void applyMain(
    uint3 id: SV_DispatchThreadID,
    uint3 group: SV_GroupID,
    uint3 thread: SV_GroupThreadID,
    uint index: SV_GroupIndex
) {
    float4 viewport = globals.viewport;
    bool smaa = (params.frame.x & 4u) != 0u;
    load_tile(group.xy, index, true, smaa);
    if (any(float2(id.xy) >= viewport.zw))
        return;
    int2 local = int2(id.xy);
    if (smaa) {
        smaa_temporal(local, thread.xy);
        return;
    }
    float3 centre = from_ycocg(tile_at(thread.xy, int2(0, 0)).rgb);
    float3 cross = 0.0;
    float3 lo = centre;
    float3 hi = centre;
    for (uint i = 1u; i < 5u; ++i) {
        float3 c = from_ycocg(tile_at(thread.xy, OFFSETS[i]).rgb);
        cross += c;
        lo = min(lo, c);
        hi = max(hi, c);
    }
    // Sharpen less where the neighbourhood already spans the full range.
    float3 amount = sqrt(saturate(min(lo, 1.0 - hi) / max(hi, 1.0e-4)));
    float3 w = amount * (-SHARPEST * saturate(asfloat(params.frame.z)));
    //float3 sharpened = saturate((centre + cross * w) / (1.0 + 4.0 * w));
    float3 sharpened = clamp((centre + cross * w) / (1.0 + 4.0 * w), lo, hi);
    uint2 pixel = id.xy + uint2(viewport.xy);
    // Nothing reads HDR alpha after this, and next frame rewrites it: no read back.
    hdr_output[pixel] = float4(expand(sharpened), 1.0);
}

// SMAA T2x (Jimenez et al. 2012): morphological anti-aliasing of each frame,
// then a blend of exactly two frames whose projections are jittered to two
// diagonal sub-pixel positions (`taa.rs`). With no longer history there is
// no ghosting trail and surface textures stay sharp.
//
// Each frame: luma edges with local contrast adaptation, then for every edge
// the distance to both ends of its line and how the silhouette turns there,
// which give the blend weights; the resolve blends each pixel with the
// neighbours across its edges. The area under the reconstructed silhouette
// is computed analytically for orthogonal patterns (MLAA's L, Z and U
// shapes) in place of SMAA's precomputed area texture; diagonal and corner
// patterns are not detected.
// The apply pass then reprojects last frame's result through the motion
// vectors, clamps it to this frame's neighbourhood and mixes it in by at
// most a half, less with speed and none on disocclusion.

static const float SMAA_THRESHOLD = 0.1;
/// An edge is dropped where a neighbouring edge is this much stronger.
static const float SMAA_LOCAL_CONTRAST = 2.0;
/// Pixels searched along an edge line in each direction.
static const int SMAA_MAX_SEARCH = 16;
/// Motion, in pixels per frame, at which last frame stops being blended.
static const float SMAA_FAST_PIXELS = 24.0;

float3 smaa_color(int2 local) {
    int2 last = int2(globals.viewport.zw) - 1;
    return compress(hdr.Load(int3(int2(globals.viewport.xy) + clamp(local, int2(0, 0), last), 0)).rgb);
}

float smaa_luma(int2 local) {
    // Gamma-like, as SMAA's threshold expects.
    return sqrt(dot(smaa_color(local), float3(0.2126, 0.7152, 0.0722)));
}

float2 smaa_edge(int2 p) {
    if (any(p < 0) || any(p >= int2(globals.viewport.zw)))
        return 0.0;
    return smaa_edges[p].rg;
}

float4 smaa_weight(int2 p) {
    if (any(p < 0) || any(p >= int2(globals.viewport.zw)))
        return 0.0;
    return smaa_weights[p];
}

/// Luma of this group's pixels with two pixels of border before and one after.
static const uint LUMA_TILE = GROUP + 3u;
groupshared float luma_tile[LUMA_TILE * LUMA_TILE];

float tile_luma(uint2 thread, int2 offset) {
    int2 t = int2(thread) + 2 + offset;
    return luma_tile[t.y * LUMA_TILE + t.x];
}

[shader("compute")]
[numthreads(8, 8, 1)]
void smaaEdgesMain(
    uint3 id: SV_DispatchThreadID,
    uint3 group: SV_GroupID,
    uint3 thread: SV_GroupThreadID,
    uint index: SV_GroupIndex
) {
    int2 corner = int2(group.xy * GROUP) - 2;
    for (uint i = index; i < LUMA_TILE * LUMA_TILE; i += GROUP * GROUP)
        luma_tile[i] = smaa_luma(corner + int2(i % LUMA_TILE, i / LUMA_TILE));
    GroupMemoryBarrierWithGroupSync();
    if (any(float2(id.xy) >= globals.viewport.zw))
        return;
    uint2 t = thread.xy;
    float l = tile_luma(t, int2(0, 0));
    float2 near = float2(tile_luma(t, int2(-1, 0)), tile_luma(t, int2(0, -1)));
    float2 delta = abs(l - near);
    float2 edges = step(SMAA_THRESHOLD, delta);
    if (all(edges == 0.0)) {
        smaa_edges[id.xy] = 0.0;
        return;
    }
    float2 far = float2(tile_luma(t, int2(1, 0)), tile_luma(t, int2(0, 1)));
    float2 farther = float2(tile_luma(t, int2(-2, 0)), tile_luma(t, int2(0, -2)));
    float2 strongest = max(abs(l - far), abs(near - farther));
    float largest = max(max(delta.x, delta.y), max(strongest.x, strongest.y));
    edges *= step(largest, SMAA_LOCAL_CONTRAST * delta);
    smaa_edges[id.xy] = float4(edges, 0.0, 0.0);
}

/// How the silhouette turns at an end of an edge line: toward the neighbour
/// across the edge (+0.5), toward this side (-0.5), or not at all.
float smaa_turn(float toward_neighbour, float toward_self) {
    return toward_neighbour > 0.0 && toward_self == 0.0 ? 0.5
         : toward_self > 0.0 && toward_neighbour == 0.0 ? -0.5
         : 0.0;
}

/// Height of the reconstructed silhouette above the edge line at `t` along a
/// line `span` pixels long, from the turns at its ends: an L rises to one
/// end, a Z crosses the line in the middle, a U dips to it in the middle.
float smaa_height(float t, float span, float start, float end) {
    if (start != 0.0 && end != 0.0) {
        if (start != end)
            return lerp(start, end, t / span);
        float middle = 0.5 * span;
        return t < middle ? start * (1.0 - t / middle) : end * (t - middle) / middle;
    }
    if (start != 0.0)
        return start * (1.0 - t / span);
    return end * (t / span);
}

/// Area on the neighbour's side (x) and this side (y) of the line under a
/// straight piece of silhouette from height `a` to `b` over `width` pixels.
float2 smaa_piece(float a, float b, float width) {
    if (a >= 0.0 && b >= 0.0)
        return float2(0.5 * (a + b) * width, 0.0);
    if (a <= 0.0 && b <= 0.0)
        return float2(0.0, -0.5 * (a + b) * width);
    float f = a / (a - b);
    float2 parts = 0.5 * width * float2(abs(a) * f, abs(b) * (1.0 - f));
    return a > 0.0 ? parts : parts.yx;
}

/// Areas this pixel shares across one edge: (on the neighbour's side, on
/// this side). `before` and `after` are how far the line runs on each side;
/// pieces split at half pixels, where a U's kink can fall.
float2 smaa_area(int before, int after, float start, float end) {
    float span = float(before + after + 1);
    float t = float(before);
    float h0 = smaa_height(t, span, start, end);
    float h1 = smaa_height(t + 0.5, span, start, end);
    float h2 = smaa_height(t + 1.0, span, start, end);
    return smaa_piece(h0, h1, 0.5) + smaa_piece(h1, h2, 0.5);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void smaaWeightsMain(uint3 id: SV_DispatchThreadID) {
    if (any(float2(id.xy) >= globals.viewport.zw))
        return;
    int2 p = int2(id.xy);
    float2 edges = smaa_edge(p);
    float4 weights = 0.0;
    if (edges.g > 0.0) {
        // An edge along the top: search left and right along the line.
        int left = 0;
        while (left < SMAA_MAX_SEARCH && smaa_edge(p - int2(left + 1, 0)).g > 0.0)
            ++left;
        int right = 0;
        while (right < SMAA_MAX_SEARCH && smaa_edge(p + int2(right + 1, 0)).g > 0.0)
            ++right;
        int2 start = int2(p.x - left, p.y);
        int2 end = int2(p.x + right + 1, p.y);
        float a = left < SMAA_MAX_SEARCH
            ? smaa_turn(smaa_edge(start - int2(0, 1)).r, smaa_edge(start).r) : 0.0;
        float b = right < SMAA_MAX_SEARCH
            ? smaa_turn(smaa_edge(end - int2(0, 1)).r, smaa_edge(end).r) : 0.0;
        float2 area = smaa_area(left, right, a, b);
        weights.xy = float2(area.y, area.x);
    }
    if (edges.r > 0.0) {
        // An edge along the left: search up and down along the line.
        int up = 0;
        while (up < SMAA_MAX_SEARCH && smaa_edge(p - int2(0, up + 1)).r > 0.0)
            ++up;
        int down = 0;
        while (down < SMAA_MAX_SEARCH && smaa_edge(p + int2(0, down + 1)).r > 0.0)
            ++down;
        int2 start = int2(p.x, p.y - up);
        int2 end = int2(p.x, p.y + down + 1);
        float a = up < SMAA_MAX_SEARCH
            ? smaa_turn(smaa_edge(start - int2(1, 0)).g, smaa_edge(start).g) : 0.0;
        float b = down < SMAA_MAX_SEARCH
            ? smaa_turn(smaa_edge(end - int2(1, 0)).g, smaa_edge(end).g) : 0.0;
        float2 area = smaa_area(up, down, a, b);
        weights.zw = float2(area.y, area.x);
    }
    smaa_weights[id.xy] = weights;
}

/// Blend this frame's pixel with the neighbours across its edges into the
/// next history, its alpha the linear depth for the disocclusion test.
void smaa_blend(int2 local) {
    float4 here = smaa_weight(local);
    float top = here.x;
    float left = here.z;
    float bottom = smaa_weight(local + int2(0, 1)).y;
    float right = smaa_weight(local + int2(1, 0)).w;
    float3 color = smaa_color(local);
    float across = max(left, right);
    float along = max(top, bottom);
    if (across > 0.0 && across >= along) {
        color = (left * lerp(color, smaa_color(local - int2(1, 0)), left)
               + right * lerp(color, smaa_color(local + int2(1, 0)), right)) / (left + right);
    } else if (along > 0.0) {
        color = (top * lerp(color, smaa_color(local - int2(0, 1)), top)
               + bottom * lerp(color, smaa_color(local + int2(0, 1)), bottom)) / (top + bottom);
    }
    float depth = scene_depth.Load(int3(int2(globals.viewport.xy) + local, 0));
    float linear_depth = depth > 0.0 ? min(globals.projection.z / depth, FAR_DEPTH) : FAR_DEPTH;
    accumulated[local] = float4(expand(color), linear_depth);
}

/// Last frame's SMAA result, bilinear.
float4 smaa_previous(float2 uv) {
    return previous_smaa.SampleLevel(clamp_sampler, uv, 0);
}

/// Mix this frame's SMAA result (in `history`) with last frame's (in
/// `accumulated`) into HDR.
void smaa_temporal(int2 local, uint2 thread) {
    float4 viewport = globals.viewport;
    int2 origin = int2(viewport.xy);
    int2 last = int2(viewport.zw) - 1;
    float4 centre = tile_at(thread, int2(0, 0));
    float3 current = from_ycocg(centre.rgb);
    float3 lo = 1.0e9;
    float3 hi = -1.0e9;
    float depth = -1.0;
    int2 nearest = local;
    for (uint i = 0u; i < 9u; ++i) {
        float4 s = tile_at(thread, OFFSETS[i]);
        lo = min(lo, s.rgb);
        hi = max(hi, s.rgb);
        if (s.a > depth) {
            depth = s.a;
            nearest = clamp(local + OFFSETS[i], int2(0, 0), last);
        }
    }
    float2 nearest_uv = (float2(nearest) + 0.5) / viewport.zw;
    float4 world = mul(globals.inverse_view_proj, float4(nearest_uv * 2.0 - 1.0, depth, 1.0));
    float4 previous = mul(globals.previous_view_proj, world);
    float2 motion = depth > 0.0 ? gi_motion[origin + nearest]
                                : nearest_uv - (previous.xy / previous.w * 0.5 + 0.5);
    float2 history_uv = (float2(local) + 0.5) / viewport.zw - motion;
    uint2 pixel = uint2(local + origin);
    float alpha = hdr_output[pixel].a;
    float3 result = current;
    if (params.frame.y != 0u && previous.w > 0.0 && all(history_uv >= 0.0) && all(history_uv <= 1.0)) {
        float4 old = smaa_previous(history_uv);
        float3 kept = from_ycocg(clamp(to_ycocg(compress(old.rgb)), lo, hi));
        float speed = length(motion * viewport.zw);
        float weight = 0.5 * (1.0 - saturate(speed / SMAA_FAST_PIXELS));
        // Reactive mask, as in the TAA resolve.
        weight = min(weight, 1.0 - saturate(alpha - 1.0));
        float own = centre.a;
        if (own > 0.0 && disoccluded(own, local, history_uv, true))
            weight = 0.0;
        result = lerp(current, kept, weight);
    }
    result = clamp(result, 0.0, 1.0 - 1.0e-4);
    hdr_output[pixel] = float4(expand(result), alpha);
}
