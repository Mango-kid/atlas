// SPDX-License-Identifier: AGPL-3.0-only
//! Resident tensorwise E4M3 activation quantization and native FP8 GEMM.
use crate::ops::Ops;
use anyhow::{Result, ensure};
use spark_runtime::{gpu::DevicePtr, kernel_args::KernelLaunch};

pub struct Fp8Scratch {
    pub packed: DevicePtr,
    pub max_rows: usize,
    pub max_k: usize,
}

impl Fp8Scratch {
    pub fn packed_bytes(rows: usize, k: usize) -> usize {
        rows * k
    }
}

impl Ops<'_> {
    pub fn quantize_fp8(
        &self,
        x: DevicePtr,
        m: usize,
        k: usize,
        scale: f32,
        scratch: &Fp8Scratch,
    ) -> Result<()> {
        ensure!(
            m > 0 && m <= scratch.max_rows && k > 0 && k <= scratch.max_k && k.is_multiple_of(16),
            "FP8 scratch geometry"
        );
        ensure!(scale.is_finite() && scale > 0.0, "FP8 activation scale");
        let count = u32::try_from(
            m.checked_mul(k)
                .ok_or_else(|| anyhow::anyhow!("FP8 element count overflow"))?,
        )?;
        KernelLaunch::new(self.gpu, self.gpu.kernel("pi05", "pi05_quantize_fp8")?)
            .grid([count.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(scratch.packed)
            .arg_u32(count)
            .arg_f32(scale)
            .launch(self.stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn linear_fp8(
        &self,
        x: DevicePtr,
        weight: DevicePtr,
        bias: Option<DevicePtr>,
        out: DevicePtr,
        m: usize,
        n: usize,
        k: usize,
        activation_scale: f32,
        weight_scale: f32,
        scratch: &Fp8Scratch,
    ) -> Result<()> {
        let alpha = activation_scale * weight_scale;
        ensure!(alpha.is_finite() && alpha > 0.0, "FP8 scale product");
        self.quantize_fp8(x, m, k, activation_scale, scratch)?;
        spark_runtime::cublaslt::fp8_gemm_act_weight_t_scaled(
            scratch.packed.0,
            weight.0,
            bias.map(|p| p.0),
            out.0,
            u32::try_from(m)?,
            u32::try_from(n)?,
            u32::try_from(k)?,
            alpha,
            false,
            self.stream,
        )
    }
}
