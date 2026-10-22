// SPDX-License-Identifier: AGPL-3.0-only
//! Shared-KV prefix and expert attention using flattened token/head GEMMs.
use crate::ops::Ops;
use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kernel_args::KernelLaunch;

/// BF16 probabilities followed by padded K and V, all resident during replay.
pub fn mqa_scratch_bytes(query_rows: usize, heads: usize, keys: usize, dim: usize) -> usize {
    let padded = keys.div_ceil(16) * 16;
    (query_rows * heads + 2 * dim) * padded * 2
}

impl Ops<'_> {
    /// MQA uses flattened GEMMs; vision and explicitly masked attention use WMMA.
    /// Scratch must hold both the score matrix and `mqa_scratch_bytes` for MQA.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        q: DevicePtr,
        k: DevicePtr,
        v: DevicePtr,
        out: DevicePtr,
        qrows: usize,
        kvrows: usize,
        qheads: usize,
        kvheads: usize,
        dim: usize,
        mask: DevicePtr,
        scores: DevicePtr,
    ) -> Result<()> {
        if kvheads == 1 && qheads == 8 && dim == 256 && mask == DevicePtr::NULL {
            self.attention_mqa(q, k, v, out, qrows, kvrows, qheads, dim, scores)
        } else {
            self.attention_wmma(
                q, k, v, out, qrows, kvrows, qheads, kvheads, dim, mask, scores,
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn attention_mqa(
        &self,
        q: DevicePtr,
        k: DevicePtr,
        v: DevicePtr,
        out: DevicePtr,
        qrows: usize,
        kvrows: usize,
        heads: usize,
        dim: usize,
        scratch: DevicePtr,
    ) -> Result<()> {
        ensure!(
            qrows > 0 && qrows <= 712 && heads == 8 && kvrows > 0 && kvrows <= 1024 && dim == 256,
            "unsupported MQA shape"
        );
        let padded = kvrows.div_ceil(16) * 16;
        let rows = qrows * heads;
        let kpad = DevicePtr(scratch.0 + (rows * padded * 2) as u64);
        let vpad = DevicePtr(kpad.0 + (padded * dim * 2) as u64);
        KernelLaunch::new(self.gpu, self.gpu.kernel("pi05", "pi05_mqa_pad_kv")?)
            .grid([(padded * dim).div_ceil(256) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(k)
            .arg_ptr(v)
            .arg_ptr(kpad)
            .arg_ptr(vpad)
            .arg_u32(kvrows as u32)
            .arg_u32(padded as u32)
            .arg_u32(dim as u32)
            .launch(self.stream)?;
        spark_runtime::cublaslt::bf16_gemm_act_weight_t(
            q.0,
            kpad.0,
            scratch.0,
            rows as u32,
            padded as u32,
            dim as u32,
            self.stream,
        )?;
        KernelLaunch::new(self.gpu, self.gpu.kernel("pi05", "pi05_mqa_softmax")?)
            .grid([rows as u32, 1, 1])
            .block([256, 1, 1])
            .shared_mem((padded * 4) as u32)
            .arg_ptr(scratch)
            .arg_u32(kvrows as u32)
            .arg_u32(padded as u32)
            .launch(self.stream)?;
        spark_runtime::cublaslt::bf16_gemm_act_weight_n(
            scratch.0,
            vpad.0,
            out.0,
            rows as u32,
            dim as u32,
            padded as u32,
            self.stream,
        )
    }
}
