// Deform a shared rest mesh by a soft body's lattice, into one car's own vertices.
// Runs only when that car's shape changed; a car at rest is a plain transformed draw.

struct Vertex {
    float4 position;
    float4 normal;
    float4 tangent;
    float4 color;
};

// The corners of a vertex's lattice cell, bit 0 along x, 1 along y, 2 along z.
struct Skin {
    uint nodes[8];
    float weights[8];
};

struct Constants {
    // Rows taking rest space to cell coordinates, and the inverse transpose of that.
    float4 basis[3];
    float4 normal_basis[3];
    uint first_rest;
    uint count;
    uint first_out;
    uint shape;
    uint nodes;
    uint dents;
    uint2 padding;
};

[[vk::binding(0)]] StructuredBuffer<Vertex> rest;
[[vk::binding(1)]] StructuredBuffer<Skin> skin;
// Lattice nodes, then two entries per dent: point and radius, normal and depth.
[[vk::binding(2)]] StructuredBuffer<float4> shape;
[[vk::binding(3)]] RWStructuredBuffer<Vertex> output;
[[vk::push_constant]] ConstantBuffer<Constants> constants;

static const uint MISSING = 0xffffffffu;
// Folds across a dent's radius: a crushed panel creases, it does not bowl.
static const float CREASES = 3.0;
static const float CREASE_DEPTH = 0.5;

float3 node(uint index) {
    return shape[constants.shape + index].xyz;
}

float3 apply(float4 rows[3], float3 v) {
    return float3(dot(rows[0].xyz, v), dot(rows[1].xyz, v), dot(rows[2].xyz, v));
}

// Add each dent's displacement of a rest point, and tilt its normal to match.
void press(float3 point, inout float3 offset, inout float3 normal) {
    for (uint d = 0; d < constants.dents; ++d) {
        float4 a = shape[constants.shape + constants.nodes + d * 2];
        float4 b = shape[constants.shape + constants.nodes + d * 2 + 1];
        float3 to = point - a.xyz;
        float radius2 = a.w * a.w;
        float s2 = dot(to, to) / radius2;
        if (s2 >= 1.0)
            continue;
        float3 inward = b.xyz;
        float3 across = normalize(cross(inward, abs(inward.z) < 0.9 ? float3(0, 0, 1) : float3(1, 0, 0)));
        float3 along = cross(inward, across);
        float k = CREASES * 6.2831853 / a.w;
        float p = dot(to, across) * k;
        float q = dot(to, along) * k;
        float fold = 1.0 + CREASE_DEPTH * sin(p) * sin(q);
        float fall = (1.0 - s2) * (1.0 - s2);
        offset += inward * (b.w * fall * fold);
        float3 slope = b.w * (to * (-4.0 * (1.0 - s2) / radius2 * fold)
            + (across * (cos(p) * sin(q)) + along * (sin(p) * cos(q))) * (fall * CREASE_DEPTH * k));
        normal -= (slope - normal * dot(normal, slope)) * dot(normal, inward);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void embedMain(uint3 id: SV_DispatchThreadID) {
    if (id.x >= constants.count)
        return;
    Vertex v = rest[constants.first_rest + id.x];
    Skin s = skin[constants.first_rest + id.x];

    float3 position = float3(0.0);
    for (uint k = 0; k < 8; ++k) {
        if (s.nodes[k] != MISSING)
            position += node(s.nodes[k]) * s.weights[k];
    }
    // How the cell's coordinates map to space: the columns of its deformation.
    float3 axes[3];
    for (uint axis = 0; axis < 3; ++axis) {
        uint bit = 1u << axis;
        float3 sum = float3(0.0);
        float total = 0.0;
        for (uint c = 0; c < 8; ++c) {
            uint far = c | bit;
            if ((c & bit) == 0 && s.nodes[c] != MISSING && s.nodes[far] != MISSING) {
                float weight = s.weights[c] + s.weights[far];
                sum += (node(s.nodes[far]) - node(s.nodes[c])) * weight;
                total += weight;
            }
        }
        // Beside a hole in the lattice this axis keeps its rest size and direction.
        axes[axis] = abs(total) > 1.0e-5 ? sum / total : constants.normal_basis[axis].xyz;
    }

    float3 offset = float3(0.0);
    float3 normal = v.normal.xyz;
    press(v.position.xyz, offset, normal);
    float3 cell = apply(constants.basis, offset);
    position += axes[0] * cell.x + axes[1] * cell.y + axes[2] * cell.z;

    // Normals transform by the cofactors of the deformation.
    float3 n = apply(constants.normal_basis, normal);
    normal = cross(axes[1], axes[2]) * n.x + cross(axes[2], axes[0]) * n.y + cross(axes[0], axes[1]) * n.z;
    float3 t = apply(constants.basis, v.tangent.xyz);
    float3 tangent = axes[0] * t.x + axes[1] * t.y + axes[2] * t.z;

    Vertex result = v;
    result.position.xyz = position;
    result.normal.xyz = normalize(normal);
    result.tangent.xyz = dot(tangent, tangent) > 0.0 ? normalize(tangent) : float3(0.0);
    output[constants.first_out + id.x] = result;
}
