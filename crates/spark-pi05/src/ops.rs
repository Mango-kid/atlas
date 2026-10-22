// SPDX-License-Identifier: AGPL-3.0-only
//! Native pi0.5 CUDA primitives. Buffers use token-major row-major layouts.
pub use crate::ops_fp8::Fp8Scratch;
pub use crate::ops_nvfp4::Nvfp4Scratch;
use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// All GPU I/O and launches use Atlas's backend boundary.
pub struct Ops<'a> {
    pub rope_table: Option<DevicePtr>,
    pub gpu: &'a dyn GpuBackend,
    pub stream: u64,
    pub nvfp4_scratch: Option<&'a Nvfp4Scratch>,
    pub fp8_scratch: Option<&'a Fp8Scratch>,
}

impl<'a> Ops<'a> {
    pub fn new(gpu: &'a dyn GpuBackend, stream: u64) -> Result<Self> {
        // Fail before inference if this binary lacks the model's kernels.
        gpu.kernel("pi05", "pi05_attention_qk")?;
        Ok(Self {
            gpu,
            stream,
            nvfp4_scratch: None,
            fp8_scratch: None,
            rope_table: None,
        })
    }

    fn kernel(&self, name: &str) -> Result<KernelHandle> {
        self.gpu.kernel("pi05", name)
    }

    fn u32(n: usize) -> Result<u32> {
        Ok(u32::try_from(n)?)
    }

    fn unary(&self, name: &str, x: DevicePtr, out: DevicePtr, n: usize) -> Result<()> {
        let n = Self::u32(n)?;
        KernelLaunch::new(self.gpu, self.kernel(name)?)
            .grid([n.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(out)
            .arg_u32(n)
            .launch(self.stream)
    }

    pub fn cast_to_bf16(&self, x: DevicePtr, out: DevicePtr, n: usize) -> Result<()> {
        self.unary("pi05_cast_bf16", x, out, n)
    }
    pub fn cast_to_f32(&self, x: DevicePtr, out: DevicePtr, n: usize) -> Result<()> {
        self.unary("pi05_cast_f32", x, out, n)
    }
    pub fn gelu_bf16(&self, x: DevicePtr, out: DevicePtr, n: usize) -> Result<()> {
        self.unary("pi05_gelu", x, out, n)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn linear_bf16(
        &self,
        x: DevicePtr,
        weight: DevicePtr,
        bias: Option<DevicePtr>,
        out: DevicePtr,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let (m, n, k) = (Self::u32(m)?, Self::u32(n)?, Self::u32(k)?);
        if let Some(bias) = bias {
            return spark_runtime::cublaslt::bf16_gemm_act_weight_t_bias(
                x.0,
                weight.0,
                bias.0,
                out.0,
                m,
                n,
                k,
                self.stream,
            );
        }
        spark_runtime::cublaslt::bf16_gemm_act_weight_t(x.0, weight.0, out.0, m, n, k, self.stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn linear_f32(
        &self,
        x: DevicePtr,
        weight: DevicePtr,
        bias: Option<DevicePtr>,
        out: DevicePtr,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let (m, n, k) = (Self::u32(m)?, Self::u32(n)?, Self::u32(k)?);
        KernelLaunch::new(self.gpu, self.kernel("pi05_linear_f32")?)
            .grid([n.div_ceil(16), m.div_ceil(16), 1])
            .block([16, 16, 1])
            .arg_ptr(x)
            .arg_ptr(weight)
            .arg_ptr(bias.unwrap_or(DevicePtr::NULL))
            .arg_ptr(out)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k)
            .launch(self.stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm(
        &self,
        x: DevicePtr,
        weight: DevicePtr,
        bias: DevicePtr,
        out: DevicePtr,
        m: usize,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu, self.kernel("pi05_layer_norm")?)
            .grid([Self::u32(m)?, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(weight)
            .arg_ptr(bias)
            .arg_ptr(out)
            .arg_u32(Self::u32(n)?)
            .arg_f32(eps)
            .launch(self.stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn norm(
        &self,
        name: &str,
        x: DevicePtr,
        weight: DevicePtr,
        out: DevicePtr,
        m: usize,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu, self.kernel(name)?)
            .grid([Self::u32(m)?, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(weight)
            .arg_ptr(out)
            .arg_u32(Self::u32(n)?)
            .arg_f32(eps)
            .launch(self.stream)
    }

    pub fn rms_norm(
        &self,
        x: DevicePtr,
        weight: DevicePtr,
        out: DevicePtr,
        m: usize,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        self.norm("pi05_rms_norm", x, weight, out, m, n, eps)
    }

    /// Modulation is FP32 `[scale, shift, gate]`; gate is used by `residual`.
    pub fn adarms(
        &self,
        x: DevicePtr,
        modulation: DevicePtr,
        out: DevicePtr,
        m: usize,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        self.norm("pi05_adarms", x, modulation, out, m, n, eps)
    }

    /// Gate points to the FP32 gate row; multiplication rounds through BF16.
    pub fn residual(
        &self,
        x: DevicePtr,
        delta: DevicePtr,
        gate: Option<DevicePtr>,
        m: usize,
        n: usize,
    ) -> Result<()> {
        let total = Self::u32(
            m.checked_mul(n)
                .ok_or_else(|| anyhow::anyhow!("residual shape overflow"))?,
        )?;
        KernelLaunch::new(self.gpu, self.kernel("pi05_residual")?)
            .grid([total.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(delta)
            .arg_ptr(gate.unwrap_or(DevicePtr::NULL))
            .arg_u32(total)
            .arg_u32(Self::u32(n)?)
            .launch(self.stream)
    }

    fn binary(
        &self,
        name: &str,
        x: DevicePtr,
        y: DevicePtr,
        out: DevicePtr,
        n: usize,
    ) -> Result<()> {
        let n = Self::u32(n)?;
        KernelLaunch::new(self.gpu, self.kernel(name)?)
            .grid([n.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(y)
            .arg_ptr(out)
            .arg_u32(n)
            .launch(self.stream)
    }
    pub fn gelu_mul(&self, gate: DevicePtr, up: DevicePtr, out: DevicePtr, n: usize) -> Result<()> {
        self.binary("pi05_gelu_mul", gate, up, out, n)
    }
    pub fn add_f32(&self, x: DevicePtr, y: DevicePtr, out: DevicePtr, n: usize) -> Result<()> {
        self.binary("pi05_add_f32", x, y, out, n)
    }
    pub fn silu_f32(&self, x: DevicePtr, n: usize) -> Result<()> {
        let n = Self::u32(n)?;
        KernelLaunch::new(self.gpu, self.kernel("pi05_silu_f32")?)
            .grid([n.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_u32(n)
            .launch(self.stream)
    }
    pub fn scale_bf16(&self, x: DevicePtr, out: DevicePtr, n: usize, scale: f32) -> Result<()> {
        let n = Self::u32(n)?;
        KernelLaunch::new(self.gpu, self.kernel("pi05_scale")?)
            .grid([n.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(out)
            .arg_u32(n)
            .arg_f32(scale)
            .launch(self.stream)
    }
    pub fn axpy_f32(&self, x: DevicePtr, v: DevicePtr, n: usize, dt: f32) -> Result<()> {
        let n = Self::u32(n)?;
        KernelLaunch::new(self.gpu, self.kernel("pi05_axpy_f32")?)
            .grid([n.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(v)
            .arg_u32(n)
            .arg_f32(dt)
            .launch(self.stream)
    }

    pub fn rope(
        &self,
        x: DevicePtr,
        positions: DevicePtr,
        rows: usize,
        heads: usize,
        dim: usize,
    ) -> Result<()> {
        ensure!(
            dim > 0 && dim.is_multiple_of(2),
            "RoPE dimension must be positive and even"
        );
        let total = Self::u32(rows * heads * (dim / 2))?;
        KernelLaunch::new(self.gpu, self.kernel("pi05_rope")?)
            .grid([total.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(positions)
            .arg_u32(Self::u32(rows)?)
            .arg_u32(Self::u32(heads)?)
            .arg_u32(Self::u32(dim)?)
            .launch(self.stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_wmma(
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
        ensure!(
            kvheads > 0 && qheads.is_multiple_of(kvheads),
            "invalid attention head grouping"
        );
        ensure!(
            kvrows > 0 && kvrows <= 1024,
            "attention requires 1..=1024 keys"
        );
        KernelLaunch::new(self.gpu, self.kernel("pi05_attention_qk")?)
            .grid([
                Self::u32(kvrows)?.div_ceil(64),
                Self::u32(qrows)?.div_ceil(16),
                Self::u32(qheads)?,
            ])
            .block([128, 1, 1])
            .arg_ptr(q)
            .arg_ptr(k)
            .arg_ptr(scores)
            .arg_ptr(mask)
            .arg_u32(Self::u32(qrows)?)
            .arg_u32(Self::u32(kvrows)?)
            .arg_u32(Self::u32(qheads)?)
            .arg_u32(Self::u32(kvheads)?)
            .arg_u32(Self::u32(dim)?)
            .launch(self.stream)?;
        KernelLaunch::new(self.gpu, self.kernel("pi05_attention_softmax")?)
            .grid([Self::u32(qheads * qrows)?, 1, 1])
            .block([256, 1, 1])
            .shared_mem(Self::u32(kvrows * 4)?)
            .arg_ptr(scores)
            .arg_u32(Self::u32(kvrows)?)
            .launch(self.stream)?;
        KernelLaunch::new(self.gpu, self.kernel("pi05_attention_pv")?)
            .grid([
                Self::u32(dim)?.div_ceil(64),
                Self::u32(qrows)?.div_ceil(16),
                Self::u32(qheads)?,
            ])
            .block([128, 1, 1])
            .arg_ptr(scores)
            .arg_ptr(v)
            .arg_ptr(out)
            .arg_u32(Self::u32(qrows)?)
            .arg_u32(Self::u32(kvrows)?)
            .arg_u32(Self::u32(qheads)?)
            .arg_u32(Self::u32(kvheads)?)
            .arg_u32(Self::u32(dim)?)
            .launch(self.stream)
    }

    pub fn embedding(
        &self,
        table: DevicePtr,
        ids: DevicePtr,
        out: DevicePtr,
        rows: usize,
        width: usize,
        scale: f32,
    ) -> Result<()> {
        let total = Self::u32(rows * width)?;
        KernelLaunch::new(self.gpu, self.kernel("pi05_embedding")?)
            .grid([total.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(table)
            .arg_ptr(ids)
            .arg_ptr(out)
            .arg_u32(total)
            .arg_u32(Self::u32(width)?)
            .arg_f32(scale)
            .launch(self.stream)
    }
}
