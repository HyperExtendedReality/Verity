// Auto exposure metered from diffuse light only, as in Fox Engine: the
// resolve sums log2 luminance of the light a white diffuse surface would
// reflect, so albedo, emissives and specular do not move the exposure.
// `composite.slang` adapts toward it and exposes the scene before tonemapping.

/// x: summed log2 luminance in fixed point; y: samples; z: exposure (float
/// bits); w: 1 once an exposure was metered. Mirrors `EXPOSURE_INITIAL` in lib.rs.
[[vk::binding(35)]] RWStructuredBuffer<uint> exposure_state;
static const uint EXPOSURE_SUM = 0;
static const uint EXPOSURE_COUNT = 1;
static const uint EXPOSURE_VALUE = 2;
static const uint EXPOSURE_SEEDED = 3;
/// Fixed-point steps per stop; with log2 clamped to +-12 a 4K frame metered
/// at one pixel in 16 stays inside an int.
static const float EXPOSURE_SCALE = 64.0;
static const float MIN_LOG = -12.0;
static const float MAX_LOG = 12.0;
static const float MIN_EXPOSURE = 1.0 / 4096.0;
static const float MAX_EXPOSURE = 4096.0;

/// Add one wave's metered samples. Every lane of the wave must call it.
void meter_exposure(bool sampled, float luminance) {
    float value = sampled ? clamp(log2(max(luminance, 1.0e-6)), MIN_LOG, MAX_LOG) : 0.0;
    int sum = WaveActiveSum(int(round(value * EXPOSURE_SCALE)));
    uint count = WaveActiveCountBits(sampled);
    if (WaveIsFirstLane() && count != 0) {
        InterlockedAdd(exposure_state[EXPOSURE_SUM], uint(sum));
        InterlockedAdd(exposure_state[EXPOSURE_COUNT], count);
    }
}
