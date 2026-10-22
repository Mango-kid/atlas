// SPDX-License-Identifier: AGPL-3.0-only
//! Native NVFP4 E2M1 x E2M1 -> BF16 with FP32 accumulation.
use super::*;
use anyhow::ensure;

#[derive(Default)]
struct Descriptors {
    desc: cublasLtMatmulDesc_t,
    layouts: [cublasLtMatrixLayout_t; 3],
    pref: cublasLtMatmulPreference_t,
}
impl Drop for Descriptors {
    fn drop(&mut self) {
        unsafe {
            if !self.pref.is_null() {
                cublasLtMatmulPreferenceDestroy(self.pref);
            }
            for layout in self.layouts {
                if !layout.is_null() {
                    cublasLtMatrixLayoutDestroy(layout);
                }
            }
            if !self.desc.is_null() {
                cublasLtMatmulDescDestroy(self.desc);
            }
        }
    }
}
// CUDA 13 cublasLt.h: algorithm64, size_t workspace, status32, float waves,
// four reserved int32 values. Exact result layout, rather than packed bytes.
#[repr(C)]
#[derive(Default)]
struct Heuristic {
    algo: [u64; 8],
    workspace: usize,
    state: i32,
    waves: f32,
    reserved: [i32; 4],
}

// CUDA 13 cublasLt.h / library_types.h. FP4 inputs use hardware block scaling.
const CUDA_R_4F_E2M1: i32 = 33;
const SCALE_MODE_VEC16_UE4M3: i32 = 1;

/// Native E2M1 Tensor Core GEMM with FP32 accumulation and BF16 output.
/// Packed inputs are row-major [M,K/2] and [N,K/2], low nibble first.
/// E4M3 dequantization scales are in NVIDIA's 128x4 swizzle, padded to
/// ceil(rows/128)*128 * ceil(K/64)*4 bytes. `alpha` is the product of the
/// two FP32 tensor dequantization scales. Optional bias is BF16 `[N]`.
/// Uses the shared process workspace; callers serialize on the model stream.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_gemm_act_weight_t(
    act: u64,
    act_scales: u64,
    weight: u64,
    weight_scales: u64,
    alpha: f32,
    bias: Option<u64>,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        m > 0 && n > 0 && n.is_multiple_of(8) && k > 0 && k.is_multiple_of(64),
        "NVFP4 geometry"
    );
    ensure!(
        alpha.is_finite() && alpha > 0.0,
        "NVFP4 global scale product"
    );
    let output_bytes = u64::from(m)
        .checked_mul(u64::from(n))
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| anyhow::anyhow!("NVFP4 output size overflow"))?;
    let scale_bytes = |rows: u32| u64::from(rows).div_ceil(128) * 128 * u64::from(k) / 16;
    let spans = [
        (act, u64::from(m) * u64::from(k) / 2),
        (act_scales, scale_bytes(m)),
        (weight, u64::from(n) * u64::from(k) / 2),
        (weight_scales, scale_bytes(n)),
        (out, output_bytes),
        (bias.unwrap_or(0), u64::from(n) * 2),
    ];
    let spans = &spans[..if bias.is_some() { 6 } else { 5 }];
    for (i, &(ptr, bytes)) in spans.iter().enumerate() {
        ensure!(
            ptr > 0 && ptr.is_multiple_of(16) && ptr.checked_add(bytes).is_some(),
            "NVFP4 pointer"
        );
        for &(other, len) in &spans[..i] {
            ensure!(ptr >= other + len || other >= ptr + bytes, "NVFP4 alias");
        }
    }
    let context = ctx()?;
    let mut h = Descriptors::default();
    unsafe {
        chk(
            cublasLtMatmulDescCreate(&mut h.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "NVFP4 desc",
        )?;
        for (attr, value) in [
            (DESC_TRANSA, CUBLAS_OP_T),
            (DESC_TRANSB, CUBLAS_OP_N),
            (DESC_A_SCALE_MODE, SCALE_MODE_VEC16_UE4M3),
            (DESC_B_SCALE_MODE, SCALE_MODE_VEC16_UE4M3),
        ] {
            chk(
                cublasLtMatmulDescSetAttribute(
                    h.desc,
                    attr,
                    (&value as *const i32).cast(),
                    size_of::<i32>(),
                ),
                "NVFP4 mode",
            )?;
        }
        // cuBLAS A is weight^T and B is activation under the column-major view.
        for (attr, address) in [
            (DESC_A_SCALE_POINTER, weight_scales),
            (DESC_B_SCALE_POINTER, act_scales),
        ] {
            let ptr = address as *const c_void;
            chk(
                cublasLtMatmulDescSetAttribute(
                    h.desc,
                    attr,
                    (&ptr as *const *const c_void).cast(),
                    size_of::<*const c_void>(),
                ),
                "NVFP4 scale pointer",
            )?;
        }
        if let Some(bias) = bias {
            let epilogue = 4i32;
            let ptr = bias as *const c_void;
            chk(
                cublasLtMatmulDescSetAttribute(
                    h.desc,
                    7,
                    (&epilogue as *const i32).cast(),
                    size_of::<i32>(),
                ),
                "NVFP4 bias epilogue",
            )?;
            chk(
                cublasLtMatmulDescSetAttribute(
                    h.desc,
                    8,
                    (&ptr as *const *const c_void).cast(),
                    size_of::<*const c_void>(),
                ),
                "NVFP4 bias pointer",
            )?;
        }
        for (i, (dtype, rows, cols)) in [
            (CUDA_R_4F_E2M1, k, n),
            (CUDA_R_4F_E2M1, k, m),
            (CUDA_R_16BF, n, m),
        ]
        .into_iter()
        .enumerate()
        {
            chk(
                cublasLtMatrixLayoutCreate(
                    &mut h.layouts[i],
                    dtype,
                    u64::from(rows),
                    u64::from(cols),
                    i64::from(rows),
                ),
                "NVFP4 layout",
            )?;
        }
        chk(
            cublasLtMatmulPreferenceCreate(&mut h.pref),
            "NVFP4 preference",
        )?;
        chk(
            cublasLtMatmulPreferenceSetAttribute(
                h.pref,
                PREF_MAX_WORKSPACE_BYTES,
                (&context.ws_size as *const usize).cast(),
                size_of::<usize>(),
            ),
            "NVFP4 workspace",
        )?;
        let [la, lb, lc] = h.layouts;
        let mut result = Heuristic::default();
        let mut returned = 0;
        chk(
            cublasLtMatmulAlgoGetHeuristic(
                context.handle,
                h.desc,
                la,
                lb,
                lc,
                lc,
                h.pref,
                1,
                (&mut result as *mut Heuristic).cast(),
                &mut returned,
            ),
            "NVFP4 heuristic",
        )?;
        ensure!(
            returned == 1 && result.state == 0 && result.workspace <= context.ws_size,
            "cuBLASLt no NVFP4 algorithm M={m} N={n} K={k}"
        );
        let beta = 0f32;
        chk(
            cublasLtMatmul(
                context.handle,
                h.desc,
                (&alpha as *const f32).cast(),
                weight as *const c_void,
                la,
                act as *const c_void,
                lb,
                (&beta as *const f32).cast(),
                out as *const c_void,
                lc,
                out as *mut c_void,
                lc,
                result.algo.as_ptr().cast(),
                context.workspace as *mut c_void,
                context.ws_size,
                stream as *mut c_void,
            ),
            "NVFP4 matmul",
        )?;
    }
    Ok(())
}
