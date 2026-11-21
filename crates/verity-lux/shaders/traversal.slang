struct Hit { float distance; uint triangle; uint instance; uint errors; float3 barycentric; float footprint; };

// A malformed or partially resident hierarchy can contain a cycle that never
// grows the traversal stack (a node whose only intersecting child links back to
// an ancestor), and its leaf ranges can cover the whole scene. Both would let a
// single ray run for seconds and trip the GPU watchdog, so every traversal also
// carries a step budget and a bounded leaf scan.
static const uint MAX_TRAVERSAL_STEPS = 4096u;
static const uint MAX_LEAF_TRIANGLES = 256u;
static const uint MAX_LEAF_INSTANCES = 4096u;

// A denied traversal is not a miss: treating it as sky or an unoccluded
// shadow sample would inject energy (or claim false visibility). The ticket
// counter is reset by the cache-clear pass once per frame. A zero budget is
// unbounded while still recording attempts when the buffer is present.
static const uint HIT_QUERY_UNSUPPORTED = 4u;
// Ray-traced GI has no cascade query budget, and nothing resets its counters.
static bool transportAdmissionBypass = false;
bool admitTransportQuery(uint admissionLimit) {
    if (transportAdmissionBypass) return true;
    uint attemptIndex = 2u * params.size.w + 5u;
    uint admittedIndex = 2u * params.size.w + 6u;
    heads[attemptIndex].add(1u);
    if (params.sampling.y == 0u) {
        heads[admittedIndex].add(1u);
        return true;
    }

    uint limit = params.sampling.y;
    if (admissionLimit != 0u) limit = min(limit, admissionLimit);
    // Every receiver starts its first query together; a compare-exchange retry
    // loop on this one counter serializes them. Take a ticket and return it
    // when over the limit: never over-admits, and the final count is exact.
    if (heads[admittedIndex].add(1u) < limit) return true;
    heads[admittedIndex].add(0xffffffffu);
    return false;
}
uint ordinaryTransportAdmissionLimit() {
    return uint(max(params.intervalHistory.w, 0.0));
}
bool admitTransportQuery() {
    return admitTransportQuery(ordinaryTransportAdmissionLimit());
}

// Scene and geometry words are addressed by values that ultimately come from
// shader-computed triangles and material records. A single stale or malformed
// index must not become an out-of-range buffer address: that faults the GPU
// (device lost) instead of producing a wrong colour. Every read below is
// therefore clamped to the bound buffer.
uint4 geometryWord(uint index) {
    uint words = params.wordLengths.x;
    return words == 0u ? (uint4)0 : geometryWords[min(index, words - 1u)];
}
uint4 sceneWord(uint index) {
    uint words = params.wordLengths.y;
    return words == 0u ? (uint4)0 : sceneWords[min(index, words - 1u)];
}

uint3 triangleIndices(uint triangle, uint4 header) { return geometryWord(header.x + triangle).xyz; }
uint3 triangleIndices(uint triangle) { return triangleIndices(triangle, geometryWord(0)); }
uint vertexBase(uint vertex, uint4 header) { return header.y + vertex * header.z; }
uint vertexBase(uint vertex) { return vertexBase(vertex, geometryWord(0)); }
Triangle localTriangle(uint3 indices, uint4 header) {
    Triangle triangle;
    triangle.a = float4(asfloat(geometryWord(vertexBase(indices.x, header)).xyz), 0);
    triangle.b = float4(asfloat(geometryWord(vertexBase(indices.y, header)).xyz), 0);
    triangle.c = float4(asfloat(geometryWord(vertexBase(indices.z, header)).xyz), 0);
    triangle.albedo = 0; triangle.emission = 0;
    return triangle;
}
Triangle localTriangle(uint index, uint4 header) {
    return localTriangle(triangleIndices(index, header), header);
}
Triangle localTriangle(uint index) { return localTriangle(index, geometryWord(0)); }
float4 hitVertexAttribute(Hit hit, uint attribute, uint3 indices, uint4 header) {
    return asfloat(geometryWord(vertexBase(indices.x, header) + attribute)) * hit.barycentric.x
         + asfloat(geometryWord(vertexBase(indices.y, header) + attribute)) * hit.barycentric.y
         + asfloat(geometryWord(vertexBase(indices.z, header) + attribute)) * hit.barycentric.z;
}
float4 hitVertexAttribute(Hit hit, uint attribute) {
    uint4 header = geometryWord(0);
    return hitVertexAttribute(hit, attribute, triangleIndices(hit.triangle, header), header);
}
float3 unitVector(float3 value, float3 fallback) {
    float scale = max(abs(value.x), max(abs(value.y), abs(value.z)));
    if (scale == 0 || !isfinite(scale)) return fallback;
    // DX12 division can lower to a reciprocal. Keep that reciprocal normal
    // even at f32::MAX, instead of letting flush-to-zero erase the direction.
    value /= max(scale * 0.25, asfloat(0x00800000u));
    return value / length(value);
}

BvhNode sceneNode(uint index) {
    uint4 a = sceneWord(1 + index * 2), b = sceneWord(2 + index * 2);
    BvhNode node; node.minimum = asfloat(a.xyz); node.first = a.w;
    node.maximum = asfloat(b.xyz); node.count = b.w; return node;
}
[ForceInline]
uint instanceBase(uint instance) {
    uint4 header = sceneWord(0);
    return header.y + instance * header.w;
}
uint2 hitInstanceIdentity(Hit hit) { return sceneWord(instanceBase(hit.instance) + 7).xy; }
bool instanceVisibleRecord(uint4 mask) {
    uint count = min(mask.y, params.layers.y);
    uint wholeWords = count / 4u;
    // One scene/page lookup and one light lookup cover four layer bits.
    [loop] for (uint word = 0; word < wholeWords; ++word) {
        uint4 instanceBits = sceneWord(mask.x + word);
        uint4 cameraBits = lights[params.layers.x + word].source;
        if (any((instanceBits & cameraBits) != uint4(0))) return true;
    }
    if (wholeWords * 4u < count) {
        uint4 instanceBits = sceneWord(mask.x + wholeWords);
        uint4 cameraBits = lights[params.layers.x + wholeWords].source;
        [unroll] for (uint i = 0; i < 3u; ++i) {
            if (wholeWords * 4u + i < count && (instanceBits[i] & cameraBits[i]) != 0) return true;
        }
    }
    return false;
}
bool instanceVisible(uint instance) {
    return instanceVisibleRecord(sceneWord(instanceBase(instance) + 6));
}
[ForceInline]
float3 transformPoint(uint base, float3 value, bool direction = false) {
    float4 h = float4(value, direction ? 0 : 1);
    return float3(dot(asfloat(sceneWord(base)), h), dot(asfloat(sceneWord(base+1)), h), dot(asfloat(sceneWord(base+2)), h));
}
[ForceInline]
float3 transformDirection(uint base, float3 value, bool transpose = false) {
    float3 a = asfloat(sceneWord(base)).xyz;
    float3 b = asfloat(sceneWord(base + 1)).xyz;
    float3 c = asfloat(sceneWord(base + 2)).xyz;
    // Only direction matters. Scale operands before multiplying so finite,
    // large/small imported attributes do not overflow under instance scale.
    float3 magnitude = max(abs(a), max(abs(b), abs(c)));
    float scale = max(magnitude.x, max(magnitude.y, magnitude.z));
    if (scale == 0) return 0;
    scale = max(scale * 0.25, asfloat(0x00800000u));
    a /= scale; b /= scale; c /= scale;
    value = unitVector(value, float3(0));
    return transpose ? a * value.x + b * value.y + c * value.z
                     : float3(dot(a, value), dot(b, value), dot(c, value));
}
struct HitAttributes {
    float3 geometricNormal;
    float3 worldNormal; // normalized per vertex, then interpolated (MikkTSpace)
    float4 worldTangent;
};
float3 triangleGeometricNormal(uint base, uint flags, Triangle triangle) {
    return ((flags & 4) != 0 ? -1 : 1) * unitVector(transformDirection(base,
        cross(triangle.b.xyz - triangle.a.xyz, triangle.c.xyz - triangle.a.xyz), true), float3(0, 1, 0));
}
float3 hitGeometricNormal(Hit hit) {
    uint base = instanceBase(hit.instance);
    return triangleGeometricNormal(base, sceneWord(base + 6).w, localTriangle(hit.triangle));
}
HitAttributes hitAttributes(Hit hit) {
    uint base = instanceBase(hit.instance);
    uint4 header = geometryWord(0);
    uint3 indices = triangleIndices(hit.triangle, header);
    Triangle triangle = localTriangle(indices, header);
    uint instanceFlags = sceneWord(base + 6).w;
    float sign = (instanceFlags & 4) != 0 ? -1 : 1;
    HitAttributes attributes;
    attributes.geometricNormal = triangleGeometricNormal(base, instanceFlags, triangle);
    attributes.worldNormal = 0; attributes.worldTangent = 0;
    [loop] for (uint i = 0; i < 3; ++i) {
        uint vertex = vertexBase(indices[i], header);
        uint flags = geometryWord(vertex).w;
        float3 normal = attributes.geometricNormal;
        if ((flags & 1) != 0)
            normal = unitVector(transformDirection(base, asfloat(geometryWord(vertex + 1).xyz), true), normal);
        // Match Bevy: normalize transformed normals/tangents before interpolation;
        // retain the interpolated lengths for the later normal-map TBN transform.
        attributes.worldNormal += normal * hit.barycentric[i];
        float4 tangent = asfloat(geometryWord(vertex + 2));
        attributes.worldTangent += float4(unitVector(transformDirection(base + 3, tangent.xyz), float3(0)),
                                          tangent.w * sign) * hit.barycentric[i];
    }
    return attributes;
}
#include "material.slang"
float3 hitShadingNormal(Hit hit, HitAttributes attributes, float3 incoming) {
    return materialNormal(hit, attributes, incoming);
}
// Emitted radiance at a hit, scene-linear: Verity's HDR target has no exposure.
// An unlit instance shows its base colour.
float3 hitEmission(Hit hit, float3 baseColor) {
    if ((sceneWord(instanceBase(hit.instance) + 6).w & 16) != 0) return baseColor;
    float3 emission = hitMaterial(hit).emission;
    if (!any(emission > 0.0)) return 0.0;
    return emission * materialSample(hit, 4).rgb;
}
Triangle worldTriangle(Hit hit) {
    uint base = instanceBase(hit.instance);
    Triangle triangle = localTriangle(hit.triangle);
    triangle.a.xyz = transformPoint(base + 3, triangle.a.xyz);
    triangle.b.xyz = transformPoint(base + 3, triangle.b.xyz);
    triangle.c.xyz = transformPoint(base + 3, triangle.c.xyz);
    float4 color = materialBaseColor(hit);
    float metallic = materialMetallic(hit);
    // Preserve the low two cull-mode bits for transport shading.
    triangle.albedo = float4(color.xyz * (1 - metallic), float(sceneWord(base + 6).w & 3u));
    if ((sceneWord(base + 6).w & 16) != 0) triangle.albedo.xyz = 0;
    float3 emission = hitEmission(hit,color.xyz);
    triangle.emission = float4(emission, 0);
    return triangle;
}
float3 emitterRadiance(Light light, float3 barycentric) {
    if (!instanceVisible(light.source.x)) return float3(0);
    Hit hit; hit.distance = 0; hit.instance = light.source.x; hit.triangle = light.source.y;
    hit.barycentric = barycentric; hit.errors = 0; hit.footprint = 0;
    float coverage = materialCoverage(hit, materialAlphaCutoff(hit));
    if (!(coverage > 0.0)) return 0.0;
    float3 baseColor = materialBaseColor(hit).rgb;
    return max(hitEmission(hit, baseColor), 0.0) * coverage;
}
float sourceCosine(Triangle triangle, float3 direction) {
    float c = dot(normalize(cross(triangle.b.xyz - triangle.a.xyz, triangle.c.xyz - triangle.a.xyz)), -direction);
    uint cullMode = uint(triangle.albedo.w) & 3u;
    return cullMode == 0 ? abs(c) : (cullMode == 2 ? -c : c);
}
float sourceCosine(Light light, float3 direction) {
    float c = dot(light.direction.xyz, -direction);
    return light.position.w == -2 ? abs(c) : c;
}

[ForceInline]
bool conventionalLightIsPunctual(Light light) {
    // Conventional records use position.w > 0 for finite point/spot sources;
    // directional records encode the direction to the source in position.xyz.
    return light.position.w > 0.0;
}
[ForceInline]
void addConventionalLightRadiance(inout float3 background, inout float3 punctual,
                                  bool finiteLight, float3 contribution) {
    if (finiteLight) punctual += contribution;
    else background += contribution;
}

[ForceInline]
bool boxRayInterval(float3 origin, float3 direction, BvhNode node, float limit,
                    out float entry, out float exit) {
    float near = 0.0, far = limit;
    // One reciprocal per ray once inlined and hoisted out of the walk, not six
    // divides per box; the far margin below covers its extra roundoff.
    float3 inverse = 1.0 / direction;
    [unroll] for (uint axis = 0; axis < 3; ++axis) {
        if (direction[axis] == 0) {
            if (origin[axis] < node.minimum[axis] || origin[axis] > node.maximum[axis]) {
                entry = 0.0; exit = 0.0;
                return false;
            }
        } else {
            float a = (node.minimum[axis] - origin[axis]) * inverse[axis];
            float b = (node.maximum[axis] - origin[axis]) * inverse[axis];
            float axisExit = max(a, b);
            // Conservative far endpoint accounts for slab division roundoff.
            near = max(near, min(a, b)); far = min(far, axisExit + abs(axisExit) * 3.5762791e-7);
        }
    }
    entry = near; exit = far;
    return near <= far;
}
[ForceInline]
float boxEntryDistance(float3 origin, float3 direction, BvhNode node, float limit) {
    float entry, exit;
    return boxRayInterval(origin, direction, node, limit, entry, exit) ? entry : -1.0;
}
[ForceInline]
bool intersectBox(float3 origin, float3 direction, BvhNode node, float limit) {
    return boxEntryDistance(origin, direction, node, limit) >= 0.0;
}
[ForceInline]
float orientedEdge(float2 a, float2 b) {
    // Evaluate a shared edge in a canonical order and apply its sign afterwards.
    // GPU contraction of a*b-c*d can otherwise give opposite directed edges
    // different rounding (one product is fused, the other is rounded first).
    bool reverse = a.x > b.x || (a.x == b.x && a.y > b.y);
    float2 first = reverse ? b : a, second = reverse ? a : b;
    precise float productA = first.x * second.y;
    precise float productB = first.y * second.x;
    precise float edge = productA - productB;
    float error = (abs(productA) + abs(productB)) * 2.3841858e-7;
    if (abs(edge) <= error) edge = 0;
    return reverse ? -edge : edge;
}
struct TriangleRay {
    uint kx, ky, kz;
    float sx, sy;
    bool valid;
};

TriangleRay triangleRay(float3 direction) {
    // The shear and dominant axis depend only on the local ray, not on the
    // triangle. Reuse them for every primitive visited in this instance.
    TriangleRay ray = (TriangleRay)0;
    float3 magnitude = abs(direction);
    ray.kz = magnitude.x > magnitude.y ? (magnitude.x > magnitude.z ? 0 : 2) : (magnitude.y > magnitude.z ? 1 : 2);
    ray.valid = direction[ray.kz] != 0;
    if (!ray.valid) return ray;
    ray.kx = (ray.kz + 1) % 3;
    ray.ky = (ray.kx + 1) % 3;
    if (direction[ray.kz] < 0) { uint swap = ray.kx; ray.kx = ray.ky; ray.ky = swap; }
    ray.sx = direction[ray.kx] / direction[ray.kz];
    ray.sy = direction[ray.ky] / direction[ray.kz];
    return ray;
}

float3 intersectTriangle(float3 origin, float3 direction, TriangleRay ray, Triangle triangle) {
    // Sheared edge functions give adjacent triangles matching edge decisions.
    // No absolute determinant epsilon: local directions retain world t and can
    // be arbitrarily short/long under nonuniform instance scale.
    if (!ray.valid) return float3(-1, 0, 0);
    uint kx = ray.kx, ky = ray.ky, kz = ray.kz;
    float sx = ray.sx, sy = ray.sy;
    float3 a = triangle.a.xyz - origin, b = triangle.b.xyz - origin, c = triangle.c.xyz - origin;
    float2 pa = float2(a[kx] - sx * a[kz], a[ky] - sy * a[kz]);
    float2 pb = float2(b[kx] - sx * b[kz], b[ky] - sy * b[kz]);
    float2 pc = float2(c[kx] - sx * c[kz], c[ky] - sy * c[kz]);
    float largest = max(max(max(abs(pa.x), abs(pa.y)), max(abs(pb.x), abs(pb.y))), max(abs(pc.x), abs(pc.y)));
    if (largest == 0 || !isfinite(largest)) return float3(-1, 0, 0);
    // Exact power-of-two rescaling preserves shared-edge signs while keeping
    // edge products normal for small/large local mesh coordinates. Do not use
    // reciprocal(largest): desktop GPU flush-to-zero can erase that reciprocal.
    uint exponent = (asuint(largest) >> 23) & 255u;
    float scale = asfloat((254u - clamp(exponent, 1u, 253u)) << 23);
    pa *= scale; pb *= scale; pc *= scale;
    float3 edges = float3(orientedEdge(pb, pc), orientedEdge(pc, pa), orientedEdge(pa, pb));
    if (any(edges < 0) && any(edges > 0)) return float3(-1, 0, 0);
    float determinant = edges.x + edges.y + edges.z;
    if (determinant == 0 || !isfinite(determinant)) return float3(-1, 0, 0);
    // ponytail: f32-only edge predicates on portable WGSL; extremely skinny
    // triangles whose edge differences underflow need an exact-predicate backend.
    float3 weights = edges / determinant;
    float distance = dot(weights, float3(a[kz], b[kz], c[kz])) / direction[kz];
    return isfinite(distance) ? float3(distance, weights.yz) : float3(-1, 0, 0);
}

Hit traceTriangle(float3 origin, float3 direction, TriangleRay ray, float3 worldDirection,
                  float coneWidth, float coneSpread, uint triangleIndex, uint instance,
                  uint base, uint flags, uint4 geometryHeader, Hit hit) {
    Triangle triangle = localTriangle(triangleIndex, geometryHeader);
    if ((flags & 3u) != 0) {
        float facing = dot(cross(triangle.b.xyz-triangle.a.xyz, triangle.c.xyz-triangle.a.xyz), direction)
                       * ((flags & 4) != 0 ? -1 : 1);
        if (((flags & 1) != 0 && facing >= 0) || ((flags & 2) != 0 && facing <= 0)) return hit;
    }
    float3 intersection = intersectTriangle(origin, direction, ray, triangle);
    float distance = intersection.x;
    // Reject sub-precision self hits at translated triangle boundaries.
    if (distance > 0.000025 && distance < hit.distance) {
        Hit candidate = hit; candidate.triangle = triangleIndex; candidate.instance = instance;
        candidate.barycentric = float3(1 - intersection.y - intersection.z, intersection.yz);
        candidate.footprint = 0;
        float cutoff = materialAlphaCutoff(candidate);
        if (cutoff != -1.0 && (coneWidth > 0 || coneSpread > 0)) {
            float cosine = abs(dot(triangleGeometricNormal(base, flags, triangle), worldDirection));
            candidate.footprint = (max(coneWidth, 0.0) + max(coneSpread, 0.0) * distance)
                                  / max(cosine, 0.001);
        }
        if (!materialAccepts(candidate, cutoff)) return hit;
        hit.distance = distance; hit.triangle = triangleIndex; hit.instance = instance;
        hit.barycentric = float3(1 - intersection.y - intersection.z, intersection.yz);
        hit.footprint = candidate.footprint;
    }
    return hit;
}

[ForceInline]
uint2 boundedTriangleSpan(uint first, uint count, uint4 geometry, out bool truncated) {
    uint begin = max(first, geometry.y);
    uint instanceEnd = geometry.y + geometry.z;
    uint requestedEnd = first + count;
    uint end = min(requestedEnd, instanceEnd);
    uint boundedEnd = min(end, begin + min(MAX_LEAF_TRIANGLES, 0xffffffffu - begin));
    truncated = first < geometry.y || instanceEnd < geometry.y || requestedEnd < first
        || requestedEnd > instanceEnd || count > MAX_LEAF_TRIANGLES || boundedEnd < end;
    if (requestedEnd < first || instanceEnd < geometry.y) boundedEnd = begin;
    return uint2(begin, boundedEnd);
}

// Immutable software BLAS arena, as packed by scene/wide.rs. A reference's
// high nibble tags the bounds/leaf format; the remaining bits are local to
// this mesh. Zero is unused, 0 is an axis-aligned branch, 1–8 are inline span
// counts, 9 is a full span, 10–12 are oriented branches, 13 is an indirect leaf.
static const uint WIDE_KIND_SHIFT = 28u;
static const uint WIDE_PAYLOAD = 0x0fffffffu;
static const uint WIDE_INLINE_COUNT = 8u;
static const uint WIDE_FULL_SPAN = 9u;
static const uint WIDE_ORIENTED_XY = 10u;
static const uint WIDE_INDIRECT_SPAN = 13u;
static const float WIDE_RELATIVE_PAD = 9.5367431640625e-7; // 8 * f32 epsilon
static const float WIDE_ABSOLUTE_PAD = 9.4039548065783e-38; // 8 * f32 min normal
static const uint WIDE_RECORD_WORDS = 4u;
static const uint WIDE_HEADER_WORDS = 2u;
static const uint WIDE_ENHANCED = 0x80000000u;
// The packer skips two levels of the depth-48 binary tree per wide level.
// With four children, at most three deferred entries per level are live.
static const uint WIDE_STACK_SIZE = 72u;

uint wideSceneCapacity() { return params.wordLengths.y; }

[ForceInline]
float3 wideDecodeBound(float3 frameMin, float3 frameMax, float3 gridStep, uint3 bin) {
    // The step has a 16-bit significand, so the 8-bit bin multiply is exact.
    // The exact frame endpoints also cover flat axes and overflowing spans.
    float3 interpolated = frameMin + gridStep * float3(bin);
    return select(bin == 0u, frameMin, select(bin == 255u, frameMax, interpolated));
}

[ForceInline]
uint2 wideChildEntry(uint reference, uint shift, uint3 lower, uint3 upper,
                     float3 frameMin, float3 frameMax, float3 gridStep,
                     float3 origin, float3 direction, float limit) {
    if (reference != 0u) {
        BvhNode child = (BvhNode)0;
        child.minimum = wideDecodeBound(frameMin, frameMax, gridStep, (lower >> shift) & 255u);
        child.maximum = wideDecodeBound(frameMin, frameMax, gridStep, (upper >> shift) & 255u);
        float entry = boxEntryDistance(origin, direction, child, limit);
        if (entry >= 0.0) return uint2(reference, asuint(entry) & 0x7fffffffu);
    }
    // The unsigned sort below puts absent or missed children last.
    return uint2(0u, 0xffffffffu);
}

float3 wideProject(float3 p, uint u, uint v) {
    precise float sum = p[u] + p[v], difference = p[v] - p[u];
    precise float a = sum * asfloat(0x3f3504f3u);
    precise float b = difference * asfloat(0x3f3504f3u);
    p[u] = a;
    p[v] = b;
    return p;
}

// An upper bound on the time spent inside the mesh-space root. Unreliable
// axes impose no constraint. The current hit limit is already an upper bound.
float wideOrientedExit(float3 origin, float3 direction, BvhNode root, float limit) {
    float exit = limit;
    [unroll] for (uint axis = 0u; axis < 3u; ++axis) {
        float d = direction[axis];
        float magnitude = abs(d);
        if (magnitude < 1.0e-30 || magnitude > 1.0e30 || !isfinite(d)) continue;
        float endpoint = d > 0.0 ? root.maximum[axis] : root.minimum[axis];
        float delta = endpoint - origin[axis];
        // Inflate before division so a subnormal difference flushed to zero
        // cannot become a significant underestimate after scaling.
        float time = max(0.0, delta * (d > 0.0 ? 1.0 : -1.0) + WIDE_ABSOLUTE_PAD) / magnitude;
        if (isfinite(delta) && isfinite(time))
            exit = min(exit, time + abs(time) * WIDE_RELATIVE_PAD + WIDE_ABSOLUTE_PAD);
    }
    return exit;
}

uint2 wideOrientedChildEntry(uint reference, uint shift, uint3 lower, uint3 upper,
                             float3 frameMin, float3 frameMax, float3 gridStep,
                             float3 origin, float3 direction, float3 pad, float limit) {
    if (reference == 0u) return uint2(0u, 0xffffffffu);
    const uint2 keep = uint2(reference, 0u);
    if (!all(isfinite(origin)) || !all(isfinite(direction)) || !all(isfinite(pad))) return keep;
    float3 lo = wideDecodeBound(frameMin, frameMax, gridStep, (lower >> shift) & 255u);
    float3 hi = wideDecodeBound(frameMin, frameMax, gridStep, (upper >> shift) & 255u);
    // A second margin rounds expanded endpoints outwards, including large,
    // nearly equal endpoints where adding the ray error alone could round in.
    pad += max(abs(lo), abs(hi)) * WIDE_RELATIVE_PAD;
    lo -= pad;
    hi += pad;
    if (!all(isfinite(lo)) || !all(isfinite(hi))) return keep;
    float near = 0.0, far = limit;
    [unroll] for (uint axis = 0u; axis < 3u; ++axis) {
        float d = direction[axis];
        if (d == 0.0) {
            if (origin[axis] < lo[axis] || origin[axis] > hi[axis])
                return uint2(0u, 0xffffffffu);
        } else {
            if (abs(d) < 1.0e-30 || abs(d) > 1.0e30) return keep;
            // Expand both sides of the subtraction as well, before division.
            float2 delta = float2(lo[axis], hi[axis]) - origin[axis];
            if (!all(isfinite(delta))) return keep;
            delta += float2(-WIDE_ABSOLUTE_PAD, WIDE_ABSOLUTE_PAD);
            float2 time = delta / d;
            if (!all(isfinite(time))) return keep;
            float a = min(time.x, time.y), b = max(time.x, time.y);
            near = max(near, a - abs(a) * WIDE_RELATIVE_PAD - WIDE_ABSOLUTE_PAD);
            far = min(far, b + abs(b) * WIDE_RELATIVE_PAD + WIDE_ABSOLUTE_PAD);
        }
    }
    return near <= far ? uint2(reference, asuint(near)) : uint2(0u, 0xffffffffu);
}

[ForceInline]
void orderWideEntries(inout uint2 a, inout uint2 b) {
    if (a.y > b.y) {
        uint2 swap = a;
        a = b;
        b = swap;
    }
}

// Ordinary leaves use contiguous spans; indirect leaves keep their source IDs.
bool wideLeafSpan(uint kind, uint payload, uint records, uint recordCount, out uint2 span) {
    span = uint2(payload, kind);
    if (kind <= WIDE_INLINE_COUNT) return true;
    if (kind != WIDE_FULL_SPAN || payload == 0u || payload > recordCount) return false;
    span = sceneWord(records + (payload - 1u) * WIDE_RECORD_WORDS).xy;
    return span.y != 0u;
}

// Keep one wide walker per shader. Inlining separate enhanced/default walks
// at each ray site makes the driver compile the full traversal repeatedly.
[noinline]
Hit traceWideRecords(
    float3 origin, float3 direction, float3 worldDirection, BvhNode root,
    uint records, uint recordCount, uint current,
    uint instance, uint base, uint flags, uint4 geometry, uint4 geometryHeader,
    Hit hit, bool anyHit, bool shadow, float coneWidth, float coneSpread) {
    TriangleRay ray = triangleRay(direction);
    uint2 stack[WIDE_STACK_SIZE];
    uint count = 0u, steps = 0u;
    [loop] while (current != 0u || count != 0u) {
        if (++steps > MAX_TRAVERSAL_STEPS) {
            hit.errors |= HIT_QUERY_UNSUPPORTED;
            break;
        }
        if (current == 0u) {
            uint2 deferred = stack[--count];
            if (!shadow && asfloat(deferred.y) > hit.distance) continue;
            current = deferred.x;
        }
        uint kind = current >> WIDE_KIND_SHIFT;
        uint payload = current & WIDE_PAYLOAD;
        current = 0u;
        uint first = 0u, triangles = 0u, indirectAt = 0u;
        uint4 ids = 0u;
        bool oriented = kind - WIDE_ORIENTED_XY < 3u;
        if (kind == 0u || oriented) {
            if (payload == 0u || payload > recordCount) {
                hit.errors |= HIT_QUERY_UNSUPPORTED;
                continue;
            }
            uint at = records + (payload - 1u) * WIDE_RECORD_WORDS;
            uint4 m = sceneWord(at), n = sceneWord(at + 1u);
            uint4 q = sceneWord(at + 2u), r = sceneWord(at + 3u);
            float3 frameMin = asfloat(m.xyz), frameMax = asfloat(n.xyz);
            precise float3 extent = frameMax - frameMin;
            float3 grid = asfloat(asuint(extent) & 0xffffff00u);
            float3 gridStep = select(isfinite(grid), grid * (1.0 / 256.0), float3(0.0));
            uint3 lo = uint3(q.x, q.z, r.x), hi = uint3(q.y, q.w, r.y);
            uint2 child0, child1, child2, child3;
            if (!oriented) {
                child0 = wideChildEntry(m.w,  0u, lo, hi, frameMin, frameMax, gridStep, origin, direction, hit.distance);
                child1 = wideChildEntry(n.w,  8u, lo, hi, frameMin, frameMax, gridStep, origin, direction, hit.distance);
                child2 = wideChildEntry(r.z, 16u, lo, hi, frameMin, frameMax, gridStep, origin, direction, hit.distance);
                child3 = wideChildEntry(r.w, 24u, lo, hi, frameMin, frameMax, gridStep, origin, direction, hit.distance);
            } else {
                uint frame = kind - WIDE_ORIENTED_XY;
                uint u = frame == 2u ? 1u : 0u, v = frame == 0u ? 1u : 2u;
                float exit = wideOrientedExit(origin, direction, root, hit.distance);
                float originError = WIDE_RELATIVE_PAD * max(abs(origin[u]), abs(origin[v])) + WIDE_ABSOLUTE_PAD;
                float directionError = WIDE_RELATIVE_PAD * max(abs(direction[u]), abs(direction[v])) + WIDE_ABSOLUTE_PAD;
                // Even the untouched axis may flush a subnormal direction.
                float3 pad = WIDE_ABSOLUTE_PAD + exit * WIDE_ABSOLUTE_PAD;
                pad[u] = pad[v] = originError + exit * directionError;
                float3 o = wideProject(origin, u, v), d = wideProject(direction, u, v);
                child0 = wideOrientedChildEntry(m.w,  0u, lo, hi, frameMin, frameMax, gridStep, o, d, pad, exit);
                child1 = wideOrientedChildEntry(n.w,  8u, lo, hi, frameMin, frameMax, gridStep, o, d, pad, exit);
                child2 = wideOrientedChildEntry(r.z, 16u, lo, hi, frameMin, frameMax, gridStep, o, d, pad, exit);
                child3 = wideOrientedChildEntry(r.w, 24u, lo, hi, frameMin, frameMax, gridStep, o, d, pad, exit);
            }
            orderWideEntries(child0, child1);
            orderWideEntries(child2, child3);
            orderWideEntries(child0, child2);
            orderWideEntries(child1, child3);
            orderWideEntries(child1, child2);
            uint deferredCount = uint(child1.x != 0u) + uint(child2.x != 0u) + uint(child3.x != 0u);
            if (count + deferredCount <= WIDE_STACK_SIZE) {
                if (child3.x != 0u) stack[count++] = child3;
                if (child2.x != 0u) stack[count++] = child2;
                if (child1.x != 0u) stack[count++] = child1;
                current = child0.x;
                continue;
            }
            // As in binary traversal, try the full primitive span if the
            // topology violates the packing depth limit.
            hit.errors |= 1u;
            first = geometry.y; triangles = geometry.z;
            count = 0u;
        } else if (kind == WIDE_INDIRECT_SPAN) {
            if (payload == 0u || payload > recordCount || geometry.z > 0xffffffffu - geometry.y) {
                hit.errors |= HIT_QUERY_UNSUPPORTED;
                continue;
            }
            indirectAt = records + (payload - 1u) * WIDE_RECORD_WORDS;
            ids = sceneWord(indirectAt);
            triangles = ids.x;
            if (triangles == 0u || triangles > WIDE_INLINE_COUNT) {
                hit.errors |= HIT_QUERY_UNSUPPORTED;
                continue;
            }
        } else {
            uint2 leaf;
            if (!wideLeafSpan(kind, payload, records, recordCount, leaf) || leaf.x > 0xffffffffu - geometry.y) {
                hit.errors |= HIT_QUERY_UNSUPPORTED;
                continue;
            }
            first = geometry.y + leaf.x; triangles = leaf.y;
        }
        uint2 range = uint2(1u, triangles + 1u);
        if (indirectAt == 0u) {
            bool truncated;
            range = boundedTriangleSpan(first, triangles, geometry, truncated);
            if (truncated) hit.errors |= HIT_QUERY_UNSUPPORTED;
        }
        [loop] for (uint j = range.x; j < range.y; ++j) {
            uint i = j;
            if (indirectAt != 0u) {
                if ((j & 3u) == 0u) ids = sceneWord(indirectAt + (j >> 2u));
                uint source = ids[j & 3u];
                if (source >= geometry.z) {
                    hit.errors |= HIT_QUERY_UNSUPPORTED;
                    continue;
                }
                i = geometry.y + source;
            }
            if (shadow) {
                if (occludedTriangle(origin, direction, ray, i, instance, flags, hit.distance, geometryHeader)) {
                    hit.triangle = i;
                    return hit;
                }
            } else {
                hit = traceTriangle(origin, direction, ray, worldDirection, coneWidth, coneSpread,
                                    i, instance, base, flags, geometryHeader, hit);
                if (anyHit && hit.triangle != EMPTY) return hit;
            }
        }
    }
    return hit;
}

[ForceInline]
Hit traceWideInstance(float3 origin, float3 direction, float3 worldDirection,
                      uint instance, uint base, uint flags, uint4 geometry, uint4 geometryHeader,
                      Hit hit, bool anyHit, bool shadow, float coneWidth, float coneSpread) {
    uint capacity = wideSceneCapacity();
    uint rootWord = geometry.w;
    if (rootWord <= base || rootWord > capacity || capacity - rootWord < WIDE_HEADER_WORDS) {
        hit.errors |= HIT_QUERY_UNSUPPORTED;
        return hit;
    }
    uint4 a = sceneWord(rootWord), b = sceneWord(rootWord + 1u);
    uint records = rootWord + WIDE_HEADER_WORDS;
    uint recordCount = a.w & ~WIDE_ENHANCED;
    if (b.w == 0u || recordCount > (capacity - records) / WIDE_RECORD_WORDS) {
        hit.errors |= HIT_QUERY_UNSUPPORTED;
        return hit;
    }
    BvhNode root = (BvhNode)0;
    root.minimum = asfloat(a.xyz); root.maximum = asfloat(b.xyz);
    if (!intersectBox(origin, direction, root, hit.distance)) return hit;
    return traceWideRecords(origin, direction, worldDirection, root, records, recordCount, b.w,
                            instance, base, flags, geometry, geometryHeader,
                            hit, anyHit, shadow, coneWidth, coneSpread);
}

Hit traceInstance(float3 origin, float3 direction, uint instance, Hit hit, bool anyHit,
                  float coneWidth, float coneSpread, uint4 geometryHeader) {
    uint base = instanceBase(instance);
    uint4 instanceRecord = sceneWord(base + 6);
    if (!instanceVisibleRecord(instanceRecord)) return hit;
    // Both transforms use the same three inverse-matrix rows. Loading them
    // once avoids repeating paged-scene/overlay resolution for every instance.
    float4 row0 = asfloat(sceneWord(base));
    float4 row1 = asfloat(sceneWord(base + 1));
    float4 row2 = asfloat(sceneWord(base + 2));
    float4 point = float4(origin, 1);
    float4 vector = float4(direction, 0);
    float3 localOrigin = float3(dot(row0, point), dot(row1, point), dot(row2, point));
    // Keep this unnormalized so triangle/box distances remain world distances.
    float3 localDirection = float3(dot(row0, vector), dot(row1, vector), dot(row2, vector));
    uint flags = instanceRecord.w;
    uint4 geometry = sceneWord(base + 8);
    if (geometry.w != 0u)
        return traceWideInstance(localOrigin, localDirection, direction, instance, base, flags,
                                 geometry, geometryHeader, hit, anyHit, false, coneWidth, coneSpread);
    uint2 stack[49]; uint count = 0;
    uint currentNode = geometry.x;
    BvhNode node = sceneNode(currentNode);
    bool currentNodeReady = intersectBox(localOrigin, localDirection, node, hit.distance);
    TriangleRay ray = (TriangleRay)0;
    if (currentNodeReady) ray = triangleRay(localDirection);
    uint traversalSteps = 0u;
    [loop] while (currentNodeReady || count != 0) {
        if (++traversalSteps > MAX_TRAVERSAL_STEPS) {
            hit.errors |= HIT_QUERY_UNSUPPORTED;
            break;
        }
        if (!currentNodeReady) {
            // The box was already tested; only a closer hit can prune it.
            uint2 deferred = stack[--count];
            if (asfloat(deferred.y) > hit.distance) continue;
            currentNode = deferred.x;
            node = sceneNode(currentNode);
        }
        currentNodeReady = false;
        if (node.count == 0) {
            if (count + 2 > 49) {
                // Unexpected topology cannot silently drop an occluder. Fall
                // back to this instance's primitive range and report the flag.
                hit.errors |= 1;
                node.first = geometry.y; node.count = geometry.z;
                count = 0;
            } else {
                uint leftIndex = node.first;
                uint rightIndex = leftIndex + 1;
                BvhNode leftNode = sceneNode(leftIndex);
                BvhNode rightNode = sceneNode(rightIndex);
                float leftEntry = boxEntryDistance(localOrigin, localDirection, leftNode, hit.distance);
                float rightEntry = boxEntryDistance(localOrigin, localDirection, rightNode, hit.distance);
                if (leftEntry >= 0.0 && rightEntry >= 0.0) {
                    if (leftEntry <= rightEntry) {
                        stack[count++] = uint2(rightIndex, asuint(rightEntry));
                        currentNode = leftIndex;
                        node = leftNode;
                    } else {
                        stack[count++] = uint2(leftIndex, asuint(leftEntry));
                        currentNode = rightIndex;
                        node = rightNode;
                    }
                    currentNodeReady = true;
                } else if (leftEntry >= 0.0) {
                    currentNode = leftIndex;
                    node = leftNode;
                    currentNodeReady = true;
                } else if (rightEntry >= 0.0) {
                    currentNode = rightIndex;
                    node = rightNode;
                    currentNodeReady = true;
                }
                continue;
            }
        }
        // A clipped closest-hit scan cannot prove that a reported hit is
        // nearest, or that a miss is actually empty.
        bool truncated;
        uint2 range = boundedTriangleSpan(node.first, node.count, geometry, truncated);
        if (truncated) hit.errors |= HIT_QUERY_UNSUPPORTED;
        [loop] for (uint i = range.x; i < range.y; ++i) {
            hit = traceTriangle(localOrigin, localDirection, ray, direction, coneWidth, coneSpread,
                                i, instance, base, flags, geometryHeader, hit);
            if (anyHit && hit.triangle != EMPTY) return hit;
        }
    }
    return hit;
}
#ifdef GI_RAY_QUERY
// Geometry index is zero for an ordinary BLAS and the first local triangle
// for a triangle cluster. Non-opaque candidates retain software hit tests.
// Only visible, unconditionally opaque two-sided geometry can commit itself.
[[vk::binding(50, 0)]] RaytracingAccelerationStructure sceneTlas;
static const uint HARDWARE_OPAQUE = 4u; // params.sourceHistory.z

[ForceInline]
uint hardwareTriangle(uint instance, uint primitive, float3 origin, float3 direction,
                      out uint base, out uint4 record,
                      out float3 localOrigin, out float3 localDirection) {
    base = instanceBase(instance);
    record = sceneWord(base + 6);
    float4 row0 = asfloat(sceneWord(base));
    float4 row1 = asfloat(sceneWord(base + 1));
    float4 row2 = asfloat(sceneWord(base + 2));
    float4 point = float4(origin, 1);
    float4 vector = float4(direction, 0);
    localOrigin = float3(dot(row0, point), dot(row1, point), dot(row2, point));
    // Unnormalized, as in traceInstance, so local distances stay world distances.
    localDirection = float3(dot(row0, vector), dot(row1, vector), dot(row2, vector));
    return sceneWord(base + 8).y + primitive;
}

Hit hardwareClosestHit(float3 origin, float3 direction, Hit hit, bool anyHit,
                       float coneWidth, float coneSpread, uint4 geometryHeader) {
    bool hasOpaque = (params.sourceHistory.z & HARDWARE_OPAQUE) != 0u;
    if (hasOpaque && !(hit.distance > 0.000025)) return hit;
    RayDesc ray;
    ray.Origin = origin; ray.Direction = direction;
    ray.TMin = hasOpaque ? asfloat(asuint(0.000025f) + 1u) : 0.0;
    ray.TMax = hit.distance;
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(sceneTlas, RAY_FLAG_NONE, 0xffu, ray);
    [loop] while (query.Proceed()) {
        if (query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) continue;
        uint instance = query.CandidateInstanceID();
        uint base; uint4 record; float3 localOrigin, localDirection;
        uint primitive = query.CandidateGeometryIndex() + query.CandidatePrimitiveIndex();
        uint triangle = hardwareTriangle(instance, primitive, origin, direction,
            base, record, localOrigin, localDirection);
        if (!instanceVisibleRecord(record)) continue;
        Hit candidate = traceTriangle(localOrigin, localDirection, triangleRay(localDirection),
            direction, coneWidth, coneSpread, triangle, instance, base, record.w,
            geometryHeader, hit);
        // traceTriangle returns its input unchanged unless it accepts a closer hit.
        if (!(candidate.distance < hit.distance)) continue;
        hit = candidate;
        query.CommitNonOpaqueTriangleHit();
        if (anyHit) query.Abort();
    }
    if (hasOpaque && query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        float t = query.CommittedRayT();
        if (t > 0.000025 && t < hit.distance) {
            uint instance = query.CommittedInstanceID();
            uint primitive = query.CommittedGeometryIndex() + query.CommittedPrimitiveIndex();
            uint triangle = sceneWord(instanceBase(instance) + 8).y + primitive;
            // A committed non-opaque candidate already passed the software
            // intersection, culling and alpha tests above. Keep its payload.
            if (instance == hit.instance && triangle == hit.triangle) return hit;
            float2 uv = query.CommittedTriangleBarycentrics();
            hit.distance = t;
            hit.instance = instance;
            hit.triangle = triangle;
            hit.barycentric = float3(1.0 - uv.x - uv.y, uv);
            hit.footprint = 0;
        }
    }
    return hit;
}

bool hardwareOccluded(float3 origin, float3 direction, float limit, uint4 geometryHeader) {
    bool hasOpaque = (params.sourceHistory.z & HARDWARE_OPAQUE) != 0u;
    if (hasOpaque && !(limit > 0.000025)) return false;
    RayDesc ray;
    ray.Origin = origin; ray.Direction = direction;
    ray.TMin = hasOpaque ? asfloat(asuint(0.000025f) + 1u) : 0.0;
    ray.TMax = limit;
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(sceneTlas, RAY_FLAG_NONE, 0xffu, ray);
    [loop] while (query.Proceed()) {
        if (query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) continue;
        uint instance = query.CandidateInstanceID();
        uint base; uint4 record; float3 localOrigin, localDirection;
        uint primitive = query.CandidateGeometryIndex() + query.CandidatePrimitiveIndex();
        uint triangle = hardwareTriangle(instance, primitive, origin, direction,
            base, record, localOrigin, localDirection);
        if (!instanceVisibleRecord(record)) continue;
        if (occludedTriangle(localOrigin, localDirection, triangleRay(localDirection),
                             triangle, instance, record.w, limit, geometryHeader))
            return true;
    }
    return hasOpaque && query.CommittedStatus() == COMMITTED_TRIANGLE_HIT
        && query.CommittedRayT() > 0.000025 && query.CommittedRayT() < limit;
}
#endif
Hit traceRayTracked(float3 origin, float3 direction, float limit,
                    out DependencyContext dependency, out float dependencyStart,
                    out float dependencyDistance, out float dependencyPadding,
                    bool anyHit = false,
                    float coneWidth = 0, float coneSpread = 0, uint admissionLimit = 0u) {
    Hit hit; hit.distance = limit; hit.triangle = EMPTY; hit.instance = EMPTY; hit.errors = 0; hit.barycentric = 0; hit.footprint = 0;
    dependency = dependencyUnknown();
    dependencyStart = 0.0;
    dependencyDistance = 0.0;
    dependencyPadding = 0.0;
    if (params.frame.z == 0) { dependency = dependencyEmpty(); return hit; }
    uint effectiveAdmissionLimit = admissionLimit == 0u
        ? ordinaryTransportAdmissionLimit() : admissionLimit;
    if (!admitTransportQuery(effectiveAdmissionLimit)) { hit.errors |= HIT_QUERY_UNSUPPORTED; return hit; }
    // The near-child walk retains each deferred box's entry distance so a
    // closer hit can prune it without retesting the box.
    uint2 stack[49]; uint count = 0;
    uint currentNode = 0;
    BvhNode node = sceneNode(currentNode);
    float rootEntry, rootExit;
    bool intersectsRoot = boxRayInterval(origin, direction, node, hit.distance,
                                          rootEntry, rootExit);
    dependency = dependencyEmpty();
    if (!intersectsRoot) return hit;
    uint4 geometryHeader = geometryWord(0);
    dependencyStart = max(rootEntry, 0.0);
    dependencyDistance = min(limit, rootExit);
#ifdef GI_RAY_QUERY
    hit = hardwareClosestHit(origin, direction, hit, anyHit, coneWidth, coneSpread,
                             geometryHeader);
#else
    bool currentNodeReady = rootEntry >= 0.0;
    uint traversalSteps = 0u;
    [loop] while (currentNodeReady || count != 0) {
        if (++traversalSteps > MAX_TRAVERSAL_STEPS) {
            hit.errors |= HIT_QUERY_UNSUPPORTED;
            break;
        }
        if (!currentNodeReady) {
            uint2 deferred = stack[--count];
            if (asfloat(deferred.y) > hit.distance) continue;
            currentNode = deferred.x;
            node = sceneNode(currentNode);
        }
        currentNodeReady = false;
        if (node.count == 0) {
            if (count + 2 > 49) {
                hit.errors |= 2;
                node.first = 0; node.count = params.frame.z;
                count = 0;
            } else {
                uint leftIndex = node.first;
                uint rightIndex = leftIndex + 1;
                BvhNode leftNode = sceneNode(leftIndex);
                BvhNode rightNode = sceneNode(rightIndex);
                float leftEntry = boxEntryDistance(origin, direction, leftNode, hit.distance);
                float rightEntry = boxEntryDistance(origin, direction, rightNode, hit.distance);
                if (leftEntry >= 0.0 && rightEntry >= 0.0) {
                    if (leftEntry <= rightEntry) {
                        stack[count++] = uint2(rightIndex, asuint(rightEntry));
                        currentNode = leftIndex;
                        node = leftNode;
                    } else {
                        stack[count++] = uint2(leftIndex, asuint(leftEntry));
                        currentNode = rightIndex;
                        node = rightNode;
                    }
                    currentNodeReady = true;
                } else if (leftEntry >= 0.0) {
                    currentNode = leftIndex;
                    node = leftNode;
                    currentNodeReady = true;
                } else if (rightEntry >= 0.0) {
                    currentNode = rightIndex;
                    node = rightNode;
                    currentNodeReady = true;
                }
                continue;
            }
        }
        uint requestedEnd = node.first + node.count;
        bool truncated = requestedEnd < node.first || node.first > params.frame.z
            || requestedEnd > params.frame.z || node.count > MAX_LEAF_INSTANCES;
        uint instanceBegin = min(node.first, params.frame.z);
        uint instanceEnd = min(requestedEnd, params.frame.z);
        uint boundedEnd = min(instanceEnd, instanceBegin + MAX_LEAF_INSTANCES);
        truncated = truncated || boundedEnd < instanceEnd;
        if (truncated) hit.errors |= HIT_QUERY_UNSUPPORTED;
        [loop] for (uint i = instanceBegin; i < boundedEnd; ++i) {
            hit = traceInstance(origin, direction, i, hit, anyHit, coneWidth, coneSpread, geometryHeader);
            if (anyHit && hit.triangle != EMPTY) {
                // Any accepted hit is a complete proof for an any-hit query,
                // regardless of earlier branches that could not be inspected.
                hit.errors &= ~HIT_QUERY_UNSUPPORTED;
                count = 0;
                break;
            }
        }
    }
#endif
    if (hit.triangle != EMPTY && (coneWidth > 0 || coneSpread > 0)) {
        uint base = instanceBase(hit.instance);
        uint flags = sceneWord(base + 6).w;
        Triangle triangle = localTriangle(hit.triangle);
        float cosine = abs(dot(triangleGeometricNormal(base, flags, triangle), direction));
        hit.footprint = (max(coneWidth, 0.0) + max(coneSpread, 0.0) * hit.distance)
                        / max(cosine, 0.001);
    }
    float dependencyEnd = hit.triangle != EMPTY ? min(hit.distance, rootExit) : rootExit;
    float padding = max(coneWidth, 0.0) + max(coneSpread, 0.0) * max(dependencyEnd, 0.0);
    dependencyDistance = max(dependencyEnd, 0.0);
    dependencyPadding = padding;
    dependencyIncludeSegment(dependency, origin + direction * dependencyStart,
                             origin + direction * dependencyEnd, padding);
    return hit;
}
Hit traceRay(float3 origin, float3 direction, float limit, bool anyHit = false,
             float coneWidth = 0, float coneSpread = 0, uint admissionLimit = 0u) {
    DependencyContext unusedDependency;
    float unusedStart, unusedDistance, unusedPadding;
    return traceRayTracked(origin, direction, limit, unusedDependency,
                           unusedStart, unusedDistance, unusedPadding, anyHit,
                           coneWidth, coneSpread, admissionLimit);
}

bool occludedTriangle(float3 origin, float3 direction, TriangleRay ray, uint triangleIndex,
                      uint instance, uint flags, float limit, uint4 geometryHeader) {
    Triangle triangle = localTriangle(triangleIndex, geometryHeader);
    if ((flags & 3u) != 0) {
        float facing = dot(cross(triangle.b.xyz-triangle.a.xyz,triangle.c.xyz-triangle.a.xyz),direction)
                       * ((flags & 4) != 0 ? -1 : 1);
        if (((flags & 1) != 0 && facing >= 0) || ((flags & 2) != 0 && facing <= 0)) return false;
    }
    float3 intersection = intersectTriangle(origin,direction,ray,triangle);
    if (!(intersection.x > 0.000025 && intersection.x < limit)) return false;
    Hit candidate; candidate.distance = intersection.x; candidate.triangle = triangleIndex;
    candidate.instance = instance; candidate.errors = 0;
    candidate.barycentric = float3(1-intersection.y-intersection.z,intersection.yz);
    candidate.footprint = 0;
    float cutoff = materialAlphaCutoff(candidate);
    if (cutoff == -1.0) return true;
    return materialAccepts(candidate,cutoff);
}

bool occludedInstance(float3 origin, float3 direction, uint instance, float limit,
                      uint4 geometryHeader, out bool querySupported) {
    querySupported = true;
    uint base = instanceBase(instance);
    uint4 instanceRecord = sceneWord(base+6);
    if (!instanceVisibleRecord(instanceRecord)) return false;
    float4 row0 = asfloat(sceneWord(base));
    float4 row1 = asfloat(sceneWord(base + 1));
    float4 row2 = asfloat(sceneWord(base + 2));
    float4 point = float4(origin, 1);
    float4 vector = float4(direction, 0);
    float3 localOrigin = float3(dot(row0, point), dot(row1, point), dot(row2, point));
    float3 localDirection = float3(dot(row0, vector), dot(row1, vector), dot(row2, vector));
    uint flags = instanceRecord.w;
    uint4 geometry = sceneWord(base+8);
    if (geometry.w != 0u) {
        Hit hit = (Hit)0;
        hit.distance = limit; hit.triangle = EMPTY; hit.instance = EMPTY;
        hit = traceWideInstance(localOrigin, localDirection, direction, instance, base, flags,
                                geometry, geometryHeader, hit, true, true, 0.0, 0.0);
        bool blocked = hit.triangle != EMPTY;
        querySupported = blocked || (hit.errors & HIT_QUERY_UNSUPPORTED) == 0u;
        return blocked;
    }
    uint stack[49]; uint count = 0;
    uint currentNode = geometry.x;
    BvhNode node = sceneNode(currentNode);
    bool currentNodeReady = intersectBox(localOrigin,localDirection,node,limit);
    TriangleRay ray = (TriangleRay)0;
    if (currentNodeReady) ray = triangleRay(localDirection);
    uint traversalSteps = 0u;
    bool incomplete = false;
    [loop] while (currentNodeReady || count != 0) {
        if (++traversalSteps > MAX_TRAVERSAL_STEPS) {
            querySupported = false;
            return false;
        }
        if (!currentNodeReady) {
            currentNode = stack[--count];
            node = sceneNode(currentNode);
            // The deferred node passed the unchanged ray/limit test at its parent.
        }
        currentNodeReady = false;
        if (node.count == 0) {
            if (count + 2 > 49) {
                node.first = geometry.y; node.count = geometry.z;
                count = 0;
            } else {
                uint leftIndex = node.first;
                uint rightIndex = leftIndex + 1;
                BvhNode leftNode = sceneNode(leftIndex);
                BvhNode rightNode = sceneNode(rightIndex);
                float leftEntry = boxEntryDistance(localOrigin,localDirection,leftNode,limit);
                float rightEntry = boxEntryDistance(localOrigin,localDirection,rightNode,limit);
                if (leftEntry >= 0.0 && rightEntry >= 0.0) {
                    if (leftEntry <= rightEntry) {
                        stack[count++] = rightIndex;
                        currentNode = leftIndex;
                        node = leftNode;
                    } else {
                        stack[count++] = leftIndex;
                        currentNode = rightIndex;
                        node = rightNode;
                    }
                    currentNodeReady = true;
                } else if (leftEntry >= 0.0) {
                    currentNode = leftIndex;
                    node = leftNode;
                    currentNodeReady = true;
                } else if (rightEntry >= 0.0) {
                    currentNode = rightIndex;
                    node = rightNode;
                    currentNodeReady = true;
                }
                continue;
            }
        }
        // Scan only the proven primitive range and preserve incompleteness if
        // an oversized or malformed range was clipped. A real accepted hit is
        // still a complete proof that this shadow ray is blocked.
        bool truncated;
        uint2 range = boundedTriangleSpan(node.first, node.count, geometry, truncated);
        incomplete = incomplete || truncated;
        [loop] for (uint i = range.x; i < range.y; ++i) {
            if (occludedTriangle(localOrigin,localDirection,ray,i,instance,flags,limit,geometryHeader))
                return true;
        }
    }
    querySupported = !incomplete;
    return false;
}

bool traceOccluded(float3 origin, float3 direction, float limit) {
    bool querySupported;
    bool result = traceOccluded(origin, direction, limit, querySupported);
    // Preserve conservative behavior for callers that cannot consume a
    // support bit. Production estimators use the out-parameter overload below.
    return querySupported ? result : true;
}

bool traceOccluded(float3 origin, float3 direction, float limit, out bool querySupported) {
    return traceOccluded(origin, direction, limit, querySupported,
                         ordinaryTransportAdmissionLimit());
}

bool traceOccluded(float3 origin, float3 direction, float limit,
                    out bool querySupported, uint admissionLimit) {
    bool intersectsRoot;
    float rootEntry, rootExit;
    return traceOccluded(origin, direction, limit, querySupported, admissionLimit,
                         intersectsRoot, rootEntry, rootExit);
}

bool traceOccluded(float3 origin, float3 direction, float limit,
                    out bool querySupported, uint admissionLimit,
                    out bool intersectsRoot, out float rootEntry, out float rootExit) {
    intersectsRoot = false;
    rootEntry = 0.0;
    rootExit = 0.0;
    querySupported = true;
    if (params.frame.z == 0) return false;
    if (admissionLimit == 0u) admissionLimit = ordinaryTransportAdmissionLimit();
    if (!admitTransportQuery(admissionLimit)) { querySupported = false; return false; }
    uint stack[49]; uint count = 0;
    uint currentNode = 0;
    BvhNode node = sceneNode(currentNode);
    intersectsRoot = boxRayInterval(origin, direction, node, limit, rootEntry, rootExit);
    bool currentNodeReady = intersectsRoot && rootEntry >= 0.0;
    if (!currentNodeReady) return false;
    uint4 geometryHeader = geometryWord(0);
#ifdef GI_RAY_QUERY
    // Hardware traversal is never truncated, so every admitted query is complete.
    return hardwareOccluded(origin, direction, limit, geometryHeader);
#else
    uint traversalSteps = 0u;
    bool incomplete = false;
    [loop] while (currentNodeReady || count != 0) {
        if (++traversalSteps > MAX_TRAVERSAL_STEPS) {
            querySupported = false;
            return false;
        }
        if (!currentNodeReady) {
            currentNode = stack[--count];
            node = sceneNode(currentNode);
            // The deferred node passed the unchanged ray/limit test at its parent.
        }
        currentNodeReady = false;
        if (node.count == 0) {
            if (count + 2 > 49) {
                node.first = 0; node.count = params.frame.z;
                count = 0;
            } else {
                uint leftIndex = node.first;
                uint rightIndex = leftIndex + 1;
                BvhNode leftNode = sceneNode(leftIndex);
                BvhNode rightNode = sceneNode(rightIndex);
                float leftEntry = boxEntryDistance(origin,direction,leftNode,limit);
                float rightEntry = boxEntryDistance(origin,direction,rightNode,limit);
                if (leftEntry >= 0.0 && rightEntry >= 0.0) {
                    if (leftEntry <= rightEntry) {
                        stack[count++] = rightIndex;
                        currentNode = leftIndex;
                        node = leftNode;
                    } else {
                        stack[count++] = leftIndex;
                        currentNode = rightIndex;
                        node = rightNode;
                    }
                    currentNodeReady = true;
                } else if (leftEntry >= 0.0) {
                    currentNode = leftIndex;
                    node = leftNode;
                    currentNodeReady = true;
                } else if (rightEntry >= 0.0) {
                    currentNode = rightIndex;
                    node = rightNode;
                    currentNodeReady = true;
                }
                continue;
            }
        }
        uint requestedEnd = node.first + node.count;
        bool truncated = requestedEnd < node.first || node.first > params.frame.z
            || requestedEnd > params.frame.z || node.count > MAX_LEAF_INSTANCES;
        uint instanceBegin = min(node.first, params.frame.z);
        uint instanceEnd = min(requestedEnd, params.frame.z);
        uint boundedEnd = min(instanceEnd, instanceBegin + MAX_LEAF_INSTANCES);
        incomplete = incomplete || truncated || boundedEnd < instanceEnd;
        [loop] for (uint instance = instanceBegin; instance < boundedEnd; ++instance) {
            bool instanceSupported;
            bool blocked = occludedInstance(origin,direction,instance,limit,
                                             geometryHeader,instanceSupported);
            incomplete = incomplete || !instanceSupported;
            if (blocked) {
                // Any accepted blocker proves the result even if another branch
                // could not be fully inspected.
                querySupported = true;
                return true;
            }
        }
    }
    querySupported = !incomplete;
    return false;
#endif
}

// Deferred primary traces run hit shading twice. Recording stores each
// visibility query instead of tracing it and answers "visible"; that answer
// is discarded with the rest of the recording run. Queries depend only on
// the hit and earlier query positions, never on earlier answers, so replay
// issues the same sequence (or a prefix, when a query is unsupported) and
// reads the answers the visibility pass stored. Queries beyond the per-slot
// capacity are traced during replay.
static const uint VISIBILITY_TRACE = 0u;
static const uint VISIBILITY_RECORD = 1u;
static const uint VISIBILITY_REPLAY = 2u;
static uint visibilityMode = VISIBILITY_TRACE;
static uint visibilitySlot = 0u;
static uint visibilityQueries = 0u;
static const uint VISIBILITY_BLOCKED = 1u;
static const uint VISIBILITY_SUPPORTED = 2u;
static const uint VISIBILITY_INTERSECTS_ROOT = 4u;

void storeVisibilityQuery(uint query, float3 origin, float3 direction, float limit,
                          uint admissionLimit) {
    uint word = deferredQueryHead(query, params);
    [unroll] for (uint i = 0u; i < 3u; ++i) {
        heads[word + i].store(asuint(origin[i]));
        heads[word + 3u + i].store(asuint(direction[i]));
    }
    heads[word + 6u].store(asuint(limit));
    heads[word + 7u].store(admissionLimit);
}

// Trace one stored query and keep the root interval its dependency needs.
void traceVisibilityQuery(uint query) {
    uint word = deferredQueryHead(query, params);
    float3 origin, direction;
    [unroll] for (uint i = 0u; i < 3u; ++i) {
        origin[i] = asfloat(heads[word + i].load());
        direction[i] = asfloat(heads[word + 3u + i].load());
    }
    bool supported, intersectsRoot;
    float entry, exit;
    bool blocked = traceOccluded(origin, direction, asfloat(heads[word + 6u].load()),
        supported, heads[word + 7u].load(), intersectsRoot, entry, exit);
    heads[word + 8u].store((blocked ? VISIBILITY_BLOCKED : 0u)
        | (supported ? VISIBILITY_SUPPORTED : 0u)
        | (intersectsRoot ? VISIBILITY_INTERSECTS_ROOT : 0u));
    heads[word + 9u].store(asuint(entry));
    heads[word + 10u].store(asuint(exit));
}

bool traceOccludedTracked(float3 origin, float3 direction, float limit,
                          out bool querySupported, out DependencyContext dependency,
                          uint admissionLimit = 0u) {
    bool intersectsRoot;
    float entry, exit;
    bool blocked;
    uint deferredQuery = visibilityQueries;
    if (visibilityMode != VISIBILITY_TRACE) ++visibilityQueries;
    if (visibilityMode == VISIBILITY_RECORD) {
        if (deferredQuery < DEFERRED_VISIBILITY_QUERIES)
            storeVisibilityQuery(visibilitySlot * DEFERRED_VISIBILITY_QUERIES + deferredQuery,
                                 origin, direction, limit, admissionLimit);
        querySupported = true;
        dependency = dependencyEmpty();
        return false;
    }
    bool replayed = false;
    if (visibilityMode == VISIBILITY_REPLAY && deferredQuery < DEFERRED_VISIBILITY_QUERIES) {
        uint word = deferredQueryHead(
            visibilitySlot * DEFERRED_VISIBILITY_QUERIES + deferredQuery, params);
        float3 recordedOrigin, recordedDirection;
        [unroll] for (uint i = 0u; i < 3u; ++i) {
            recordedOrigin[i] = asfloat(heads[word + i].load());
            recordedDirection[i] = asfloat(heads[word + 3u + i].load());
        }
        // The recording ran in another kernel. Use its answer only for the
        // same ray; any divergence is traced here instead.
        if (distance(recordedDirection, direction) <= 1e-4
            && distance(recordedOrigin, origin) <= 1e-4 * max(1.0, length(origin))) {
            // The dependency segment follows the ray that was actually traced.
            origin = recordedOrigin;
            direction = recordedDirection;
            uint flags = heads[word + 8u].load();
            blocked = (flags & VISIBILITY_BLOCKED) != 0u;
            querySupported = (flags & VISIBILITY_SUPPORTED) != 0u;
            intersectsRoot = (flags & VISIBILITY_INTERSECTS_ROOT) != 0u;
            entry = asfloat(heads[word + 9u].load());
            exit = asfloat(heads[word + 10u].load());
            replayed = true;
        }
    }
    if (!replayed)
    blocked = traceOccluded(origin, direction, limit, querySupported, admissionLimit,
                            intersectsRoot, entry, exit);
    dependency = dependencyUnknown();
    if (!querySupported) return blocked;
    dependency = dependencyEmpty();
    if (!intersectsRoot) return blocked;
    dependencyIncludeSegment(dependency, origin + direction * max(entry, 0.0),
                             origin + direction * exit, 0.0);
    return blocked;
}

[ForceInline]
void dependencyIncludeVisibilityTarget(inout DependencyContext dependency,
                                       float3 target) {
    dependencyIncludeSegment(dependency, target, target, 0.0);
}
