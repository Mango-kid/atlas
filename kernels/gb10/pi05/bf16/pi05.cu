// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
#include <mma.h>

using bf16 = __nv_bfloat16;
__device__ float f(bf16 x) { return __bfloat162float(x); }
__device__ bf16 b(float x) { return __float2bfloat16_rn(x); }
__device__ float rounded(float x) { return f(b(x)); }

// One block per row. All callers use exactly 256 threads.
__device__ float sum_block(float x) {
    __shared__ float scratch[8];
    for (int d = 16; d; d >>= 1) x += __shfl_down_sync(0xffffffff, x, d);
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = x;
    __syncthreads();
    x = threadIdx.x < 8 ? scratch[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32) {
        for (int d = 16; d; d >>= 1) x += __shfl_down_sync(0xffffffff, x, d);
        if (threadIdx.x == 0) scratch[0] = x;
    }
    __syncthreads();
    float result = scratch[0];
    __syncthreads();
    return result;
}

extern "C" __global__ void pi05_cast_bf16(const float* in, bf16* out, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = b(in[i]);
}
extern "C" __global__ void pi05_cast_f32(const bf16* in, float* out, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f(in[i]);
}
// Full FP32 products, avoiding implicit TF32 precision changes. Small time/
// action projections and the FP32 SigLIP patch convolution use this path.
extern "C" __global__ void pi05_linear_f32(
    const float* a, const float* w, const float* bias, float* out,
    uint32_t m, uint32_t n, uint32_t k) {
    __shared__ float aa[16][16], ww[16][16];
    uint32_t row = blockIdx.y * 16 + threadIdx.y;
    uint32_t col = blockIdx.x * 16 + threadIdx.x;
    float acc = 0;
    for (uint32_t base = 0; base < k; base += 16) {
        aa[threadIdx.y][threadIdx.x] = row < m && base + threadIdx.x < k
            ? a[row * k + base + threadIdx.x] : 0;
        uint32_t wr = blockIdx.x * 16 + threadIdx.y;
        ww[threadIdx.y][threadIdx.x] = wr < n && base + threadIdx.x < k
            ? w[wr * k + base + threadIdx.x] : 0;
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < 16; ++j) acc = fmaf(aa[threadIdx.y][j], ww[threadIdx.x][j], acc);
        __syncthreads();
    }
    if (row < m && col < n) out[row * n + col] = acc + (bias ? bias[col] : 0.0f);
}

extern "C" __global__ void pi05_layer_norm(
    const bf16* x, const bf16* w, const bf16* bias, bf16* out, uint32_t width, float eps) {
    uint32_t row = blockIdx.x;
    float sum = 0;
    for (uint32_t j = threadIdx.x; j < width; j += 256) sum += f(x[row * width + j]);
    float mean = sum_block(sum) / width;
    float variance = 0;
    for (uint32_t j = threadIdx.x; j < width; j += 256) {
        float d = f(x[row * width + j]) - mean;
        variance += d * d;
    }
    float inv = rsqrtf(sum_block(variance) / width + eps);
    for (uint32_t j = threadIdx.x; j < width; j += 256)
        out[row * width + j] = b((f(x[row * width + j]) - mean) * inv * f(w[j]) + f(bias[j]));
}
extern "C" __global__ void pi05_rms_norm(
    const bf16* x, const float* weight, bf16* out, uint32_t width, float eps) {
    uint32_t row = blockIdx.x;
    float sum = 0;
    for (uint32_t j = threadIdx.x; j < width; j += 256) {
        float v = f(x[row * width + j]); sum += v * v;
    }
    float inv = rsqrtf(sum_block(sum) / width + eps);
    for (uint32_t j = threadIdx.x; j < width; j += 256)
        out[row * width + j] = b(f(x[row * width + j]) * inv * (1.0f + weight[j]));
}
extern "C" __global__ void pi05_adarms(
    const bf16* x, const float* modulation, bf16* out, uint32_t width, float eps) {
    uint32_t row = blockIdx.x;
    float sum = 0;
    for (uint32_t j = threadIdx.x; j < width; j += 256) {
        float v = f(x[row * width + j]); sum += v * v;
    }
    float inv = rsqrtf(sum_block(sum) / width + eps);
    for (uint32_t j = threadIdx.x; j < width; j += 256)
        out[row * width + j] = b(f(x[row * width + j]) * inv * (1.0f + modulation[j]) + modulation[width + j]);
}
extern "C" __global__ void pi05_residual(
    bf16* x, const bf16* delta, const float* gate, uint32_t total, uint32_t width) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        float d = f(delta[i]);
        if (gate) d = rounded(d * rounded(gate[i % width]));
        x[i] = b(f(x[i]) + d);
    }
}
__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}
extern "C" __global__ void pi05_gelu(const bf16* x, bf16* out, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = b(gelu(f(x[i])));
}
extern "C" __global__ void pi05_gelu_mul(const bf16* gate, const bf16* up, bf16* out, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = b(rounded(gelu(f(gate[i]))) * f(up[i]));
}
extern "C" __global__ void pi05_silu_f32(float* x, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] / (1.0f + expf(-x[i]));
}
extern "C" __global__ void pi05_scale(const bf16* x, bf16* out, uint32_t n, float scale) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = b(f(x[i]) * scale);
}
extern "C" __global__ void pi05_add_f32(const float* a, const float* z, float* out, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + z[i];
}
extern "C" __global__ void pi05_axpy_f32(float* x, const float* v, uint32_t n, float dt) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] + dt * v[i];
}
extern "C" __global__ void pi05_embedding(
    const bf16* table, const uint32_t* ids, bf16* out, uint32_t total, uint32_t width, float scale) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) out[i] = b(f(table[uint64_t(ids[i / width]) * width + i % width]) * scale);
}

// Gemma uses rotate_half (not adjacent pairs). Cos/sin and each multiply
// round through BF16 in eager PyTorch before the final addition.
extern "C" __global__ void pi05_rope(
    bf16* x, const int32_t* positions, uint32_t rows, uint32_t heads, uint32_t dim) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t halfdim = dim / 2;
    if (i >= rows * heads * halfdim) return;
    uint32_t d = i % halfdim;
    uint32_t headrow = i / halfdim;
    // OpenPi casts registered inv_freq buffers to BF16 with the backbone,
    // then promotes them back to FP32 when forming angles. Preserve that
    // rounding before multiplying by position (material beyond image tokens).
    float frequency = rounded(powf(10000.0f, -2.0f * d / dim));
    float angle = positions[headrow / heads] * frequency;
    float c = rounded(cosf(angle)), s = rounded(sinf(angle));
    uint32_t base = headrow * dim + d;
    float lo = f(x[base]), hi = f(x[base + halfdim]);
    x[base] = b(rounded(lo * c) - rounded(hi * s));
    x[base + halfdim] = b(rounded(hi * c) + rounded(lo * s));
}

// Token-major Q/K -> head-major BF16 scores. Tensor-core products retain
// FP32 accumulation, then separately round QK and scaling like eager Gemma.
extern "C" __global__ void pi05_attention_qk(
    const bf16* q, const bf16* k, bf16* scores, const uint8_t* mask,
    uint32_t qrows, uint32_t kvrows, uint32_t qheads, uint32_t kvheads, uint32_t dim) {
    using namespace nvcuda;
    __shared__ bf16 aa[256], ww[1024];
    __shared__ float cc[1024];
    uint32_t warp = threadIdx.x / 32, qh = blockIdx.z;
    uint32_t kh = qh / (qheads / kvheads);
    uint32_t rowbase = blockIdx.y * 16, colbase = blockIdx.x * 64;
    wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> af;
    wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major> wf;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (uint32_t base = 0; base < dim; base += 16) {
        for (uint32_t j = threadIdx.x; j < 256; j += 128) {
            uint32_t r = rowbase + j / 16, d = base + j % 16;
            aa[j] = r < qrows && d < dim ? q[(r * qheads + qh) * dim + d] : b(0);
        }
        for (uint32_t j = threadIdx.x; j < 1024; j += 128) {
            uint32_t c = colbase + j / 16, d = base + j % 16;
            ww[j] = c < kvrows && d < dim ? k[(c * kvheads + kh) * dim + d] : b(0);
        }
        __syncthreads();
        wmma::load_matrix_sync(af, aa, 16);
        wmma::load_matrix_sync(wf, ww + warp * 256, 16);
        wmma::mma_sync(acc, af, wf, acc);
        __syncthreads();
    }
    wmma::store_matrix_sync(cc + warp * 256, acc, 16, wmma::mem_row_major);
    __syncthreads();
    float scale = rsqrtf(float(dim));
    for (uint32_t j = threadIdx.x; j < 1024; j += 128) {
        uint32_t r = rowbase + (j % 256) / 16, c = colbase + (j / 256) * 16 + j % 16;
        if (r < qrows && c < kvrows)
            scores[(qh * qrows + r) * kvrows + c] = b((!mask || mask[c]) ? rounded(cc[j]) * scale : -INFINITY);
    }
}

// One block per head/query row; scores become BF16 probabilities in place.
extern "C" __global__ void pi05_attention_softmax(bf16* scores, uint32_t keys) {
    extern __shared__ float values[];
    __shared__ float maximum;
    bf16* row = scores + blockIdx.x * keys;
    float local_max = -INFINITY;
    for (uint32_t j = threadIdx.x; j < keys; j += 256) {
        values[j] = f(row[j]);
        local_max = fmaxf(local_max, values[j]);
    }
    __shared__ float warpmax[8];
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
    for (uint32_t j = threadIdx.x; j < keys; j += 256) {
        values[j] = expf(values[j] - maximum);
        sum += values[j];
    }
    float denom = sum_block(sum);
    for (uint32_t j = threadIdx.x; j < keys; j += 256) row[j] = b(values[j] / denom);
}

// Head-major probabilities times token-major V -> token-major output.
extern "C" __global__ void pi05_attention_pv(
    const bf16* probs, const bf16* v, bf16* out,
    uint32_t qrows, uint32_t kvrows, uint32_t qheads, uint32_t kvheads, uint32_t dim) {
    using namespace nvcuda;
    __shared__ bf16 aa[256], ww[1024];
    __shared__ float cc[1024];
    uint32_t warp = threadIdx.x / 32, qh = blockIdx.z;
    uint32_t kh = qh / (qheads / kvheads);
    uint32_t rowbase = blockIdx.y * 16, colbase = blockIdx.x * 64;
    wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> af;
    wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::col_major> wf;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (uint32_t base = 0; base < kvrows; base += 16) {
        for (uint32_t j = threadIdx.x; j < 256; j += 128) {
            uint32_t r = rowbase + j / 16, key = base + j % 16;
            aa[j] = r < qrows && key < kvrows ? probs[(qh * qrows + r) * kvrows + key] : b(0);
        }
        for (uint32_t j = threadIdx.x; j < 1024; j += 128) {
            uint32_t d = colbase + j / 16, key = base + j % 16;
            ww[j] = d < dim && key < kvrows ? v[(key * kvheads + kh) * dim + d] : b(0);
        }
        __syncthreads();
        wmma::load_matrix_sync(af, aa, 16);
        wmma::load_matrix_sync(wf, ww + warp * 256, 16);
        wmma::mma_sync(acc, af, wf, acc);
        __syncthreads();
    }
    wmma::store_matrix_sync(cc + warp * 256, acc, 16, wmma::mem_row_major);
    __syncthreads();
    for (uint32_t j = threadIdx.x; j < 1024; j += 128) {
        uint32_t r = rowbase + (j % 256) / 16, d = colbase + (j / 256) * 16 + j % 16;
        if (r < qrows && d < dim) out[(r * qheads + qh) * dim + d] = b(cc[j]);
    }
}


#include "nvfp4.cuh"

#include "fp8.cuh"
#include "attention_mqa.cuh"
