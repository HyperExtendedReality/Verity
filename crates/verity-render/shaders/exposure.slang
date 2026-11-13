// Adapt the auto exposure to the resolve's metering; see `lib/exposure.slang`.

#include "lib/exposure.slang"

/// Seconds for the exposure to cover ~63% of the way to its target.
static const float ADAPT_SECONDS = 0.6;
/// Exposure that maps the metered diffuse light of a white surface to this.
static const float KEY = 0.75;

struct AdaptParams {
    /// Seconds since the last frame; compute push constants start at byte 8.
    [[vk::offset(8)]] float seconds;
};
[[vk::push_constant]] ConstantBuffer<AdaptParams> adapt;

/// One thread: turn the resolve's metering into this frame's exposure and
/// reset the sums for the next frame.
[shader("compute")]
[numthreads(1, 1, 1)]
void adaptMain() {
    uint count = exposure_state[EXPOSURE_COUNT];
    float current = asfloat(exposure_state[EXPOSURE_VALUE]);
    if (count != 0) {
        float mean_log = float(int(exposure_state[EXPOSURE_SUM])) / (float(count) * EXPOSURE_SCALE);
        float target = clamp(KEY / exp2(mean_log), MIN_EXPOSURE, MAX_EXPOSURE);
        // The first metered frame jumps straight to the target.
        bool seeded = exposure_state[EXPOSURE_SEEDED] != 0;
        float rate = 1.0 - exp(-clamp(adapt.seconds, 0.0, 1.0) / ADAPT_SECONDS);
        // Adapt in log space so brightening and darkening feel alike.
        current = seeded ? exp2(lerp(log2(current), log2(target), rate)) : target;
        exposure_state[EXPOSURE_VALUE] = asuint(current);
        exposure_state[EXPOSURE_SEEDED] = 1;
    }
    exposure_state[EXPOSURE_SUM] = 0;
    exposure_state[EXPOSURE_COUNT] = 0;
}
