// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_fp8.h>

// Same 256-thread row reduction, arithmetic and BF16 output rounding as
// pi05_adarms, followed by the precise division used by pi05_quantize_fp8.
extern "C" __global__ void pi05_adarms_quantize_fp8(
    const bf16* x, const float* modulation, uint8_t* packed,
    uint32_t width, float eps, float scale) {
    uint32_t row = blockIdx.x;
    float sum = 0;
    for (uint32_t j = threadIdx.x; j < width; j += 256) {
        float v = f(x[row * width + j]); sum += v * v;
    }
    float inv = rsqrtf(sum_block(sum) / width + eps);
    for (uint32_t j = threadIdx.x; j < width; j += 256) {
        float value = rounded(f(x[row * width + j]) * inv * (1.0f + modulation[j])
                              + modulation[width + j]);
        packed[row * width + j] = __nv_fp8_e4m3(value / scale).__x;
    }
}

extern "C" __global__ void pi05_gelu_mul_quantize_fp8(
    const bf16* gate, const bf16* up, uint8_t* packed,
    uint32_t count, float scale) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        float value = rounded(rounded(gelu(f(gate[i]))) * f(up[i]));
        packed[i] = __nv_fp8_e4m3(value / scale).__x;
    }
}
