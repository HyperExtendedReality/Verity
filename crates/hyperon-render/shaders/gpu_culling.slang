// Conservative current-camera frustum culling. There is intentionally no
// previous-frame depth test: temporal occlusion alone can produce holes.
struct Job {
    float4x4 view_proj;
    float4x4 model;
    float4 bounds_min;
    float4 bounds_max;
    uint first_vertex;
    uint vertex_count;
    uint2 padding;
};

// VkDrawIndirectCommand: non-indexed vertex-pulling draw.
struct DrawCommand {
    uint vertex_count;
    uint instance_count;
    uint first_vertex;
    uint first_instance;
};

[[vk::binding(0)]] StructuredBuffer<Job> jobs;
[[vk::binding(1)]] RWStructuredBuffer<DrawCommand> commands;
[[vk::binding(2)]] Texture2D<float> current_depth;
[[vk::binding(3)]] RWStructuredBuffer<float> pyramid;
// All entries share this layout; the original frustum pass reads only count.
struct Constants {
    uint count;
    uint level;
    uint src_offset;
    uint dst_offset;
    uint src_width;
    uint src_height;
    uint dst_width;
    uint dst_height;
    uint4 viewport;
};
[[vk::push_constant]] ConstantBuffer<Constants> constants;

bool visible(Job job) {
    float3 lo = job.bounds_min.xyz;
    float3 hi = job.bounds_max.xyz;
    if (!all(isfinite(lo)) || !all(isfinite(hi)) || any(lo > hi)) return true;

    bool left = true, right = true, bottom = true, top = true, near_plane = true, far_plane = true;
    for (uint corner = 0; corner < 8; ++corner) {
        float4 object = float4(
            (corner & 1) != 0 ? hi.x : lo.x,
            (corner & 2) != 0 ? hi.y : lo.y,
            (corner & 4) != 0 ? hi.z : lo.z, 1.0);
        // Match the vertex shader's two transforms rather than combining them.
        float4 position = mul(job.view_proj, mul(job.model, object));
        float4 magnitude = mul(abs(job.view_proj), mul(abs(job.model), abs(object)));
        if (!all(isfinite(position)) || !all(isfinite(magnitude))) return true;
        float epsilon = 1e-5 * max(1.0, max(max(magnitude.x, magnitude.y), max(magnitude.z, magnitude.w)));
        float4 sides = float4(position.x + position.w, position.w - position.x,
                              position.y + position.w, position.w - position.y);
        float2 depths = float2(position.z, position.w - position.z);
        if (!all(isfinite(sides)) || !all(isfinite(depths)) || !isfinite(epsilon)) return true;
        left       = left       && sides.x < -epsilon;
        right      = right      && sides.y < -epsilon;
        bottom     = bottom     && sides.z < -epsilon;
        top        = top        && sides.w < -epsilon;
        near_plane = near_plane && depths.x < -epsilon;
        far_plane  = far_plane  && depths.y < -epsilon;
    }
    return !(left || right || bottom || top || near_plane || far_plane);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cullMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= constants.count) return;
    Job job = jobs[id.x];
    DrawCommand command;
    command.vertex_count = visible(job) ? job.vertex_count : 0;
    command.instance_count = 1;
    command.first_vertex = job.first_vertex;
    command.first_instance = 0;
    commands[id.x] = command;
}

// Mip zero copies the current depth. Higher levels take the minimum because
// reverse-Z's smallest depth is the farthest (and clear depth is zero).
[shader("compute")]
[numthreads(8, 8, 1)]
void pyramidMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= constants.dst_width || id.y >= constants.dst_height) return;
    float depth = 0.0;
    if (constants.level == 0) {
        depth = current_depth.Load(int3(id.xy, 0));
        if (!isfinite(depth) || depth < 0.0 || depth > 1.0) depth = 0.0;
    } else {
        depth = 1.0;
        for (uint y = 0; y < 2; ++y) {
            for (uint x = 0; x < 2; ++x) {
                uint2 p = id.xy * 2 + uint2(x, y);
                // Missing texels in odd-sized levels are treated as clear.
                float d = 0.0;
                if (p.x < constants.src_width && p.y < constants.src_height)
                    d = pyramid[constants.src_offset + p.y * constants.src_width + p.x];
                depth = min(depth, d);
            }
        }
    }
    pyramid[constants.dst_offset + id.y * constants.dst_width + id.x] = depth;
}

// Establish a screen-space rectangle and nearest reverse depth. Interval
// arithmetic expands each corner by a bound on both matrix multiplications.
// Near-plane intersections, screen-edge intersections and nonfinite values
// are deliberately not eligible for occlusion rejection.
bool occluded(Job job) {
    uint4 vp = constants.viewport;
    uint image_width = constants.src_width;
    uint image_height = constants.src_height;
    if (vp.z == 0 || vp.w == 0 || vp.x >= image_width || vp.y >= image_height
        || vp.z > image_width - vp.x || vp.w > image_height - vp.y)
        return false;
    float3 lo = job.bounds_min.xyz;
    float3 hi = job.bounds_max.xyz;
    if (!all(isfinite(lo)) || !all(isfinite(hi)) || any(lo > hi)) return false;
    float2 ndc_lo = float2(1e30, 1e30);
    float2 ndc_hi = float2(-1e30, -1e30);
    float closest = 0.0;
    for (uint corner = 0; corner < 8; ++corner) {
        float4 object = float4((corner & 1) != 0 ? hi.x : lo.x,
            (corner & 2) != 0 ? hi.y : lo.y,
            (corner & 4) != 0 ? hi.z : lo.z, 1.0);
        float4 p = mul(job.view_proj, mul(job.model, object));
        float4 magnitude = mul(abs(job.view_proj), mul(abs(job.model), abs(object)));
        if (!all(isfinite(p)) || !all(isfinite(magnitude))) return false;
        float error = 1e-5 * max(1.0, max(max(magnitude.x, magnitude.y), max(magnitude.z, magnitude.w)));
        if (!isfinite(error) || p.w <= error || p.z <= error || p.w - p.z <= 2.0 * error)
            return false;
        float w_lo = p.w - error;
        float w_hi = p.w + error;
        if (!isfinite(w_hi)) return false;
        float2 a = (p.xy - error) / w_lo;
        float2 b = (p.xy - error) / w_hi;
        float2 c = (p.xy + error) / w_lo;
        float2 d = (p.xy + error) / w_hi;
        float2 lower = min(min(a, b), min(c, d));
        float2 upper = max(max(a, b), max(c, d));
        float nearest = (p.z + error) / w_lo;
        if (!all(isfinite(lower)) || !all(isfinite(upper)) || !isfinite(nearest))
            return false;
        ndc_lo = min(ndc_lo, lower);
        ndc_hi = max(ndc_hi, upper);
        closest = max(closest, nearest);
    }
    if (any(ndc_lo <= -1.0) || any(ndc_hi >= 1.0) || closest >= 1.0) return false;
    float2 origin = float2(vp.xy);
    float2 size = float2(vp.zw);
    float2 pixel_lo = floor(origin + (ndc_lo * 0.5 + 0.5) * size) - 2.0;
    float2 pixel_hi = ceil(origin + (ndc_hi * 0.5 + 0.5) * size) + 2.0;
    // Touching a viewport edge is uncertain, and must be drawn.
    if (any(pixel_lo <= origin) || any(pixel_hi >= origin + size)) return false;
    uint2 first = uint2(pixel_lo);
    uint2 last = uint2(pixel_hi);
    uint width = image_width;
    uint height = image_height;
    uint offset = 0;
    uint level = 0;
    while ((last.x - first.x > 1 || last.y - first.y > 1) && level + 1 < constants.level) {
        offset += width * height;
        width = (width + 1) / 2;
        height = (height + 1) / 2;
        first /= 2;
        last /= 2;
        ++level;
    }
    if (last.x >= width || last.y >= height || last.x - first.x > 1 || last.y - first.y > 1)
        return false;
    float farthest = 1.0;
    for (uint y = first.y; y <= last.y; ++y)
        for (uint x = first.x; x <= last.x; ++x)
            farthest = min(farthest, pyramid[offset + y * width + x]);
    // A strict, biased comparison handles depth precision and equal surfaces.
    return isfinite(farthest) && farthest > closest + 1e-4;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void occlusionMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= constants.count) return;
    Job job = jobs[id.x];
    DrawCommand command;
    command.vertex_count = visible(job) && !occluded(job) ? job.vertex_count : 0;
    command.instance_count = 1;
    command.first_vertex = job.first_vertex;
    command.first_instance = 0;
    commands[id.x] = command;
}
