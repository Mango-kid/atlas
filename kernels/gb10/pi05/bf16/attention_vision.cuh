// SPDX-License-Identifier: AGPL-3.0-only
// The 72-channel vision heads use five 16-column WMMA tiles, avoiding the
// unused columns of two 64-column tiles. V loads follow contiguous channels;
// each output retains the original ordered sixteen K=16 accumulations.
extern "C" __global__ void pi05_vision_pv(
    const bf16* probs, const bf16* v, bf16* out,
    uint32_t qrows, uint32_t kvrows, uint32_t qheads, uint32_t kvheads, uint32_t dim) {
    using namespace nvcuda;
    __shared__ bf16 aa[256], ww[1280];
    __shared__ float cc[1280];
    uint32_t warp = threadIdx.x / 32, qh = blockIdx.z;
    uint32_t kh = qh / (qheads / kvheads);
    uint32_t rowbase = blockIdx.y * 16, colbase = blockIdx.x * 80;
    wmma::fragment<wmma::matrix_a,16,16,16,bf16,wmma::row_major> af;
    wmma::fragment<wmma::matrix_b,16,16,16,bf16,wmma::row_major> wf;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (uint32_t base = 0; base < kvrows; base += 16) {
        for (uint32_t j = threadIdx.x; j < 256; j += 160) {
            uint32_t r = rowbase + j / 16, key = base + j % 16;
            aa[j] = r < qrows && key < kvrows ? probs[(qh * qrows + r) * kvrows + key] : b(0);
        }
        for (uint32_t j = threadIdx.x; j < 1280; j += 160) {
            uint32_t d = colbase + j % 80, key = base + j / 80;
            ww[j] = d < dim && key < kvrows ? v[(key * kvheads + kh) * dim + d] : b(0);
        }
        __syncthreads();
        wmma::load_matrix_sync(af, aa, 16);
        wmma::load_matrix_sync(wf, ww + warp * 16, 80);
        wmma::mma_sync(acc, af, wf, acc);
        __syncthreads();
    }
    wmma::store_matrix_sync(cc + warp * 256, acc, 16, wmma::mem_row_major);
    __syncthreads();
    for (uint32_t j = threadIdx.x; j < 1280; j += 160) {
        uint32_t r = rowbase + (j % 256) / 16;
        uint32_t d = colbase + (j / 256) * 16 + j % 16;
        if (r < qrows && d < dim) out[(r * qheads + qh) * dim + d] = b(cc[j]);
    }
}

// One warp per 256-key row. Each of eight 32-value partial sums and the final
// eight-part reduction use the same shuffle tree as the 256-thread baseline.
// This removes block barriers without changing the floating-point sum order.
extern "C" __global__ void pi05_vision_softmax(bf16* scores) {
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t rowid = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    bf16* row = scores + rowid * 256;
    float vals[8], maximum = -INFINITY;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        vals[i] = f(row[lane + i * 32]);
        maximum = fmaxf(maximum, vals[i]);
    }
    for (int d = 16; d; d >>= 1)
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, d));
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    float sum = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        vals[i] = expf(vals[i] - maximum);
        float partial = vals[i];
        for (int d = 16; d; d >>= 1)
            partial += __shfl_down_sync(0xffffffff, partial, d);
        float first = __shfl_sync(0xffffffff, partial, 0);
        if (lane == i) sum = first;
    }
    for (int d = 16; d; d >>= 1) sum += __shfl_down_sync(0xffffffff, sum, d);
    sum = __shfl_sync(0xffffffff, sum, 0);
    #pragma unroll
    for (int i = 0; i < 8; i++) row[lane + i * 32] = b(vals[i] / sum);
}
