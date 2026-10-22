// SPDX-License-Identifier: AGPL-3.0-only
//! Calibrated NVFP4 activation quantization and native hardware GEMM.
use crate::ops::Ops;
use anyhow::{Result, ensure};
use spark_runtime::{gpu::DevicePtr, kernel_args::KernelLaunch};

/// Caller-owned storage shared by sequential linears, allocated before capture.
pub struct Nvfp4Scratch {
    pub packed: DevicePtr,
    pub scales: DevicePtr,
    pub max_rows: usize,
    pub max_k: usize,
}

impl Nvfp4Scratch {
    pub fn packed_bytes(rows: usize, k: usize) -> usize {
        rows * k.div_ceil(2)
    }

    /// NVIDIA VEC16_UE4M3 128x4 scale layout including zero padding.
    pub fn scale_bytes(rows: usize, k: usize) -> usize {
        rows.div_ceil(128) * 128 * k.div_ceil(64) * 4
    }
}

impl Ops<'_> {
    /// E2M1 pairs and E4M3 block scales. Global scale is a calibrated FP32
    /// dequantization multiplier, never a host-computed per-inference amax.
    pub fn quantize_nvfp4(
        &self,
        x: DevicePtr,
        m: usize,
        k: usize,
        global_scale: f32,
        scratch: &Nvfp4Scratch,
    ) -> Result<()> {
        ensure!(
            m > 0 && m <= scratch.max_rows && k > 0 && k <= scratch.max_k && k.is_multiple_of(64),
            "NVFP4 scratch geometry"
        );
        ensure!(
            (6.0 * global_scale).is_finite() && global_scale > 0.0,
            "NVFP4 activation global scale"
        );
        let groups = m.div_ceil(128) * 128 * (k / 16);
        KernelLaunch::new(self.gpu, self.gpu.kernel("pi05", "pi05_quantize_nvfp4")?)
            .grid([u32::try_from(groups.div_ceil(32))?, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(scratch.packed)
            .arg_ptr(scratch.scales)
            .arg_u32(u32::try_from(m)?)
            .arg_u32(u32::try_from(k)?)
            .arg_f32(global_scale)
            .launch(self.stream)
    }

    /// Exact fusion of BF16 gated GELU and the existing NVFP4 codec.
    #[allow(clippy::too_many_arguments)]
    pub fn gelu_mul_quantize_nvfp4(
        &self,
        gate: DevicePtr,
        up: DevicePtr,
        m: usize,
        k: usize,
        global_scale: f32,
        scratch: &Nvfp4Scratch,
    ) -> Result<()> {
        ensure!(
            m > 0 && m <= scratch.max_rows && k > 0 && k <= scratch.max_k && k.is_multiple_of(64),
            "NVFP4 scratch geometry"
        );
        ensure!(
            (6.0 * global_scale).is_finite() && global_scale > 0.0,
            "NVFP4 activation global scale"
        );
        let groups = m.div_ceil(128) * 128 * (k / 16);
        KernelLaunch::new(
            self.gpu,
            self.gpu.kernel("pi05", "pi05_gelu_mul_quantize_nvfp4")?,
        )
        .grid([u32::try_from(groups.div_ceil(32))?, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(scratch.packed)
        .arg_ptr(scratch.scales)
        .arg_u32(u32::try_from(m)?)
        .arg_u32(u32::try_from(k)?)
        .arg_f32(global_scale)
        .launch(self.stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn linear_nvfp4(
        &self,
        x: DevicePtr,
        weight: DevicePtr,
        weight_scales: DevicePtr,
        bias: Option<DevicePtr>,
        out: DevicePtr,
        m: usize,
        n: usize,
        k: usize,
        activation_global_scale: f32,
        weight_global_scale: f32,
        scratch: &Nvfp4Scratch,
    ) -> Result<()> {
        self.quantize_nvfp4(x, m, k, activation_global_scale, scratch)?;
        spark_runtime::cublaslt::nvfp4_gemm_act_weight_t(
            scratch.packed.0,
            scratch.scales.0,
            weight.0,
            weight_scales.0,
            activation_global_scale * weight_global_scale,
            bias.map(|p| p.0),
            out.0,
            u32::try_from(m)?,
            u32::try_from(n)?,
            u32::try_from(k)?,
            self.stream,
        )
    }
}
