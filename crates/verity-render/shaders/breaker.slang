// Parametric shore surface. Cell-centered wave data and profile coefficients
// are the same ones used by the CPU's sparse gameplay queries.
struct Cell {
    float4 wave;      // phase, direction x/y, unbroken height
    float4 medium;    // wavenumber, depth, height limit, group arrival
};
[[vk::binding(0, 0)]] StructuredBuffer<Cell> cells;

struct Constants {
    float4x4 view_proj;
    uint4 dimensions; // width, height, x/y subdivision count
    float4 domain;    // origin x/y, cell size, sea level
    float4 animation; // omega, time, lean, horizontal harmonic
    float4 profile;   // vertical harmonic, groups, group phase per metre x/y
};
[[vk::push_constant]] ConstantBuffer<Constants> wave;

Cell sampleCell(float2 p) {
    float2 q = clamp((p - wave.domain.xy) / wave.domain.z - 0.5,
                     0.0, float2(wave.dimensions.xy - 1));
    uint2 a = uint2(floor(q));
    uint2 b = min(a + 1, wave.dimensions.xy - 1);
    float2 f = frac(q);
    uint w = wave.dimensions.x;
    Cell c00 = cells[a.x + a.y * w];
    Cell c10 = cells[b.x + a.y * w];
    Cell c01 = cells[a.x + b.y * w];
    Cell c11 = cells[b.x + b.y * w];
    Cell result;
    result.wave = lerp(lerp(c00.wave, c10.wave, f.x), lerp(c01.wave, c11.wave, f.x), f.y);
    result.medium = lerp(lerp(c00.medium, c10.medium, f.x), lerp(c01.medium, c11.medium, f.x), f.y);
    return result;
}

struct VertexOutput {
    float4 position: SV_Position;
    float3 world: TEXCOORD0;
    float breaking: TEXCOORD1;
    float2 rest: TEXCOORD2;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertex: SV_VertexID) {
    uint quad = vertex / 6;
    uint corner = vertex % 6;
    uint2 offsets[6] = { uint2(0,0), uint2(1,0), uint2(0,1),
                         uint2(0,1), uint2(1,0), uint2(1,1) };
    uint2 segments = wave.dimensions.zw;
    uint2 point = uint2(quad % segments.x, quad / segments.x) + offsets[corner];
    float2 p = wave.domain.xy + float2(point) / float2(segments)
               * float2(wave.dimensions.xy) * wave.domain.z;
    Cell c = sampleCell(p);
    float2 direction = c.wave.yz;
    float len = length(direction);
    direction = len > 1e-6 ? direction / len : float2(1, 0);
    float k = max(c.medium.x, 1e-6);
    float depth = max(c.medium.y, 0.05);
    // verity_ocean::group_envelope and break_wave.
    float t = wave.animation.x / 7.0 * (wave.animation.y - c.medium.w);
    float across = dot(wave.profile.zw, p);
    float unbroken = c.wave.w * (1.0 + wave.profile.y
        * (0.6 * sin(t + across) + 0.4 * sin(1.618 * t - 0.7 * across)));
    float limit = c.medium.z;
    float height = min(unbroken, limit);
    float b = limit > 0.0 ? saturate((unbroken / limit - 0.9) / 0.3) : 0.0;
    float theta = c.wave.x - wave.animation.x * wave.animation.y;
    float horizontal = min(0.5 * height / tanh(k * depth), 0.9 / k);
    float wet = c.medium.y > 0.0 ? 1.0 : 0.0;
    float x = wet * (-sin(theta) - wave.animation.w * b * sin(2.0 * theta)) * horizontal;
    float z = wet * (cos(theta) + wave.profile.x * b * cos(2.0 * theta)) * 0.5 * height;
    float angle = wave.animation.z * b;
    float dx = x * cos(angle) + z * sin(angle);
    float dz = z * cos(angle) - x * sin(angle);
    VertexOutput output;
    output.world = float3(p + direction * dx, wave.domain.w + dz);
    output.position = mul(wave.view_proj, float4(output.world, 1));
    output.breaking = b;
    output.rest = p;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input): SV_Target0 {
    float2 domainMax = wave.domain.xy + float2(wave.dimensions.xy) * wave.domain.z;
    if (any(input.world.xy < wave.domain.xy) || any(input.world.xy > domainMax)) discard;
    float3 n = normalize(cross(ddx(input.world), ddy(input.world)));
    n = n.z < 0 ? -n : n;
    float light = 0.3 + 0.7 * saturate(dot(n, normalize(float3(-0.4, 0.3, 0.85))));
    if (sampleCell(input.rest).medium.y <= 0.0) discard;
    float foam = saturate(input.breaking * (1.0 - n.z) * 2.5);
    return float4(lerp(float3(0.025, 0.19, 0.23) * light,
                       float3(0.7, 0.85, 0.82) * light, foam), 1.0);
}
