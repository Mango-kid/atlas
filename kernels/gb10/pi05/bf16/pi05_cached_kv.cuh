// SPDX-License-Identifier: AGPL-3.0-only
// Clear only writable cache padding; live prefix and suffix rows are untouched.
extern "C" __global__ void pi05_mqa_clear_tail(
    bf16* k, bf16* v, uint32_t keys, uint32_t padded) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (padded - keys) * 256) {
        k[keys * 256 + i] = b(0);
        v[keys * 256 + i] = b(0);
    }
}
