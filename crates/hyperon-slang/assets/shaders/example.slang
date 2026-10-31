// A storage-buffer compute shader suitable for a Vulkan compute pipeline.
[[vk::binding(0, 0)]]
RWStructuredBuffer<float> outputValues;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint count;
    uint stride;
    outputValues.GetDimensions(count, stride);
    if (threadId.x < count) {
        outputValues[threadId.x] = float(threadId.x) * 2.0;
    }
}
