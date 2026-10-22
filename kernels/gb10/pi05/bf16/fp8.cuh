// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_fp8.h>

// Calibrated tensor scale is a dequantization multiplier. Conversion saturates
// finite magnitudes to 448 and uses round-to-nearest-even, including subnormals.
extern "C" __global__ void pi05_quantize_fp8(
    const bf16* input, uint8_t* packed, uint32_t count, float scale) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) packed[i] = __nv_fp8_e4m3(f(input[i]) / scale).__x;
}
