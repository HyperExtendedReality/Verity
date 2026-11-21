// NILE: Nested Interleaving of Low-Dimensional Elements (Ahmed et al. 2026), §§3–4.
// https://research.nvidia.com/labs/rtr/publication/ahmed2026nile/nile.pdf
// Each semantic element takes one Sobol dimension as a fixed-point interval
// index. It subdivides its own domain into equal-measure children. Directions
// and emitter points use square elements in the existing importance warps' UV
// spaces, so the estimators keep their original measures and PDFs.

// One raster receiver and 32 traced vertices. Unused draws still own their
// slots; branching and early termination never move another element's index.
static const uint NILE_VERTICES = 33u;
static const uint NILE_ELEMENTS_PER_VERTEX = 5u;
static const uint NILE_DIRECTION = 0u;
static const uint NILE_LOBE = 1u;
static const uint NILE_EMITTER_CHOICE = 2u;
static const uint NILE_EMITTER_POINT = 3u;
static const uint NILE_ROULETTE = 4u;

// Sobol direction columns expanded from Joe and Kuo's new-joe-kuo-6.21201
// (dimensions 1–165, columns 1–32), https://web.maths.unsw.edu.au/~fkuo/sobol/.
// For row (s, a, m), v[i] = m[i] << (31-i) for i < s (zero-based), then
// v[i] = v[i-s] ^ (v[i-s] >> s) ^ XOR(v[i-k] where bit s-1-k of a is set)
// for 1 <= k < s. Dimension 1 is the binary radical inverse.
//
// Copyright (c) 2008, Frances Y. Kuo and Stephen Joe
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of the
//   University of New South Wales and the University of Waikato and its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

struct NileSample {
    uint index;
    uint tileSeed;
    uint pixelSeed;
};

NileSample nileStart(uint2 pixel, uint sampleIndex) {
    // Z-index aligned 4x4 tiles around a fixed 16-spp design block (Fig. 1).
    // At that rate each tile consumes an aligned 256-index Sobol block. Use
    // XOR to allow higher sample bits without changing already issued samples;
    // a fixed index translation also preserves every aligned dyadic block
    // within each pixel. Beyond the tile prefix the pixels scramble separately.
    uint2 inTile = pixel & 3u;
    uint morton = (inTile.x & 1u) | ((inTile.y & 1u) << 1u)
        | ((inTile.x & 2u) << 1u) | ((inTile.y & 2u) << 2u);
    uint2 tile = pixel >> 2u;
    NileSample sample;
    sample.index = sampleIndex ^ (morton << 4u);
    sample.tileSeed = transportHash(tile.x ^ transportHash(tile.y + 0x2545f491u));
    sample.pixelSeed = transportHash(sample.tileSeed ^ (morton + 0x6c8e9cf5u));
    return sample;
}

uint nileSobol(uint index, uint dimension) {
    // Joe–Kuo's original binary order. Each set bit selects a column of
    // the same high-dimensional digital sequence, not a hashed dimension.
    uint bits = 0u;
    uint row = dimension * 32u;
    [loop] while (index != 0u) {
        uint column = row + uint(firstbitlow(index));
        // The columns follow the 64x64 two-word blue-noise ranks.
        uint2 word = blueNoiseRanks[4096u + (column >> 1u)];
        bits ^= word[column & 1u];
        index &= index - 1u;
    }
    return bits;
}

uint nilePermuteTree(uint digits, uint seed) {
    // Fast nested binary scrambling (Laine–Karras/Burley style). Digits run
    // from least to most significant here: an even multiply-XOR, an add,
    // and an odd multiply are all bijections on *every* low-bit prefix.
    digits ^= digits * 0x9e3779b8u;
    digits += seed;
    digits *= (seed >> 16u) | 1u;
    return digits ^ (digits * 0x846ca68au);
}

uint nileInterval(NileSample sample, uint dimension) {
    uint digits = reversebits(nileSobol(sample.index, dimension));
    uint key = (dimension + 1u) * 0x9e3779b9u;
    // Keep the first eight binary decisions shared across the tile. For the
    // 16-spp design block, they cover all 256 interval addresses. The rest
    // use pixel-specific permutations below that same parent interval, rather
    // than replaying an identical shifted stream at neighboring pixels.
    uint parent = nilePermuteTree(digits, transportHash(sample.tileSeed ^ key)) & 0xffu;
    uint child = nilePermuteTree(digits >> 8u, transportHash(sample.pixelSeed ^ key ^ parent));
    return reversebits(parent | (child << 8u));
}

float nile1D(NileSample sample, uint dimension) {
    // Center a 23-bit cell, keeping both endpoints out even after fp32 rounding.
    return (float(nileInterval(sample, dimension) >> 9u) + 0.5) * (1.0 / 8388608.0);
}

uint2 nileCompactEven(uint2 bits) {
    bits &= 0x55555555u;
    bits = (bits | (bits >> 1u)) & 0x33333333u;
    bits = (bits | (bits >> 2u)) & 0x0f0f0f0fu;
    bits = (bits | (bits >> 4u)) & 0x00ff00ffu;
    return (bits | (bits >> 8u)) & 0x0000ffffu;
}

float2 nile2D(NileSample sample, uint dimension) {
    // Each base-4 digit selects one of the four equal-area subquadrants.
    // The compacted Morton bits are precisely those recursive decisions.
    uint bits = nileInterval(sample, dimension);
    uint2 cell = nileCompactEven(uint2(bits >> 1u, bits));
    uint jitter = transportHash(bits ^ sample.pixelSeed ^ 0xe37a1c35u);
    uint2 fine = uint2(jitter, jitter >> 7u) & 127u;
    return (float2((cell << 7u) | fine) + 0.5) * (1.0 / 8388608.0);
}
