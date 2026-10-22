// SPDX-License-Identifier: AGPL-3.0-only
// Build once with the same arithmetic as pi05_rope, including BF16 inv_freq.
extern "C" __global__ void pi05_rope_table(bf16* table, uint32_t positions, uint32_t dim) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x, halfdim = dim / 2;
    if (i >= positions * halfdim) return;
    uint32_t d = i % halfdim, position = i / halfdim;
    float frequency = rounded(powf(10000.0f, -2.0f * d / dim));
    float angle = position * frequency;
    table[2 * i] = b(cosf(angle));
    table[2 * i + 1] = b(sinf(angle));
}

// Key is optional: the combined launch shares the lookup table and removes
// one launch per layer without changing any rounded multiply or add.
__device__ __forceinline__ void pi05_rope_cached_element(
    bf16* query, bf16* key, const int32_t* positions, const bf16* table,
    uint32_t i, uint32_t rows, uint32_t heads, uint32_t dim) {
    uint32_t halfdim = dim / 2;
    uint32_t qcount = rows * heads * halfdim;
    if (i >= qcount + (key ? rows * halfdim : 0)) return;
    bf16* target = query;
    uint32_t target_heads = heads;
    if (i >= qcount) {
        i -= qcount;
        target = key;
        target_heads = 1;
    }
    uint32_t d = i % halfdim, headrow = i / halfdim;
    uint32_t index = positions[headrow / target_heads] * halfdim + d;
    float c = f(table[2 * index]), s = f(table[2 * index + 1]);
    uint32_t base = headrow * dim + d;
    float lo = f(target[base]), hi = f(target[base + halfdim]);
    target[base] = b(rounded(lo * c) - rounded(hi * s));
    target[base + halfdim] = b(rounded(hi * c) + rounded(lo * s));
}

extern "C" __global__ void pi05_rope_cached(
    bf16* query, bf16* key, const int32_t* positions, const bf16* table,
    uint32_t rows, uint32_t heads, uint32_t dim) {
    pi05_rope_cached_element(query, key, positions, table,
        blockIdx.x * blockDim.x + threadIdx.x, rows, heads, dim);
}

// All writers finish before the subsequent attention launch on this stream.
// Only the current K slice is rotated; every K/V padding row is cleared anew.
extern "C" __global__ void pi05_rope_cached_clear_tail(
    bf16* query, bf16* key, bf16* value, const int32_t* positions, const bf16* table,
    uint32_t rows, uint32_t prefix_rows, uint32_t padded) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t rotations = rows * 9 * 128, keys = prefix_rows + rows;
    if (i < rotations) {
        pi05_rope_cached_element(query, key + prefix_rows * 256, positions,
            table, i, rows, 8, 256);
    }
    if (i < (padded - keys) * 256) {
        uint32_t offset = keys * 256 + i;
        key[offset] = b(0);
        value[offset] = b(0);
    }
}
