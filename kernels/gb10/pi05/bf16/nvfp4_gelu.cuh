// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_fp4.h>
#include <cuda_fp8.h>

// Fuse the gated GELU producer with its sole NVFP4 consumer. Each lane computes
// two distinct values exactly once. Both BF16 roundings from pi05_gelu_mul are
// retained before block amax, so packing sees identical values to the two-kernel
// path. Padding scales and tiny-block behavior are also unchanged.
extern "C" __global__ void pi05_gelu_mul_quantize_nvfp4(
    const bf16* gate, const bf16* up, uint8_t* packed, uint8_t* scales,
    uint32_t rows, uint32_t k, float global_scale) {
    uint32_t groups = k / 16;
    uint32_t group_index = (blockIdx.x * blockDim.x + threadIdx.x) / 8;
    uint32_t row = group_index / groups;
    uint32_t group = group_index % groups;
    uint32_t lane = threadIdx.x % 8;
    if (row >= ((rows + 127) / 128) * 128) return;
    float2 pair = make_float2(0.0f, 0.0f);
    if (row < rows) {
        uint64_t offset = uint64_t(row) * k + group * 16 + lane * 2;
        pair.x = rounded(rounded(gelu(f(gate[offset]))) * f(up[offset]));
        pair.y = rounded(rounded(gelu(f(gate[offset + 1]))) * f(up[offset + 1]));
    }
    float amax = fmaxf(fabsf(pair.x), fabsf(pair.y));
    for (int delta = 4; delta; delta >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta, 8));
    __nv_fp8_e4m3 block_scale(amax / (6.0f * global_scale));
    if (lane == 0) {
        uint64_t offset = (uint64_t(row / 128) * (groups / 4) + group / 4) * 512
            + (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4;
        scales[offset] = block_scale.__x;
    }
    if (row < rows) {
        float divisor = float(block_scale) * global_scale;
        divisor = divisor >= 1.0e-5f ? divisor : 1.0f;
        pair.x /= divisor;
        pair.y /= divisor;
        packed[uint64_t(row) * (k / 2) + group * 8 + lane] =
            __nv_cvt_float2_to_fp4x2(pair, __NV_E2M1, cudaRoundNearest);
    }
}
