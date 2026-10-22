// SPDX-License-Identifier: AGPL-3.0-only
// MQA has one shared KV head: flatten token/head queries into one GEMM.
extern "C" __global__ void pi05_mqa_pad_kv(
    const bf16* k, const bf16* v, bf16* kpad, bf16* vpad,
    uint32_t keys, uint32_t padded, uint32_t dim) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < padded * dim) {
        kpad[i] = i < keys * dim ? k[i] : b(0);
        vpad[i] = i < keys * dim ? v[i] : b(0);
    }
}

// Preserve eager BF16 QK rounding, scaling, probabilities, and output.
extern "C" __global__ void pi05_mqa_softmax(bf16* scores, uint32_t keys, uint32_t padded) {
    extern __shared__ float values[];
    __shared__ float maximum, warpmax[8];
    bf16* row = scores + blockIdx.x * padded;
    float local_max = -INFINITY;
    for (uint32_t j = threadIdx.x; j < padded; j += 256) {
        values[j] = j < keys ? rounded(f(row[j]) * 0.0625f) : -INFINITY;
        local_max = fmaxf(local_max, values[j]);
    }
    for (int d = 16; d; d >>= 1) local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, d));
    if ((threadIdx.x & 31) == 0) warpmax[threadIdx.x >> 5] = local_max;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -INFINITY;
        for (int i = 0; i < 8; ++i) m = fmaxf(m, warpmax[i]);
        maximum = m;
    }
    __syncthreads();
    float sum = 0;
    for (uint32_t j = threadIdx.x; j < padded; j += 256) {
        values[j] = expf(values[j] - maximum);
        sum += values[j];
    }
    float denom = sum_block(sum);
    for (uint32_t j = threadIdx.x; j < padded; j += 256) row[j] = b(values[j] / denom);
}
