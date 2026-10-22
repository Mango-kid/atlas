// SPDX-License-Identifier: AGPL-3.0-only
//! Affine projections and exact reuse of compatible quantized activations.

use crate::{
    ops::Ops,
    weights::{Fp8Weight, Nvfp4Weight, Weights},
};
use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

pub(in crate::model) struct Linear {
    pub weight: DevicePtr,
    pub bias: Option<DevicePtr>,
    pub input: usize,
    pub output: usize,
    pub nvfp4: Option<Nvfp4Weight>,
    pub fp8: Option<Fp8Weight>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SharedInput {
    Fp8 { input: usize, scale_bits: u32 },
    Nvfp4 { input: usize, scale_bits: u32 },
}

impl Linear {
    fn shared_input(projections: &[(&Self, DevicePtr)]) -> Option<SharedInput> {
        if projections.len() < 2 {
            return None;
        }
        let signature = |linear: &Self| match (linear.fp8, linear.nvfp4) {
            (Some(q), None) => Some(SharedInput::Fp8 {
                input: linear.input,
                scale_bits: q.activation_scale.to_bits(),
            }),
            (None, Some(q)) => Some(SharedInput::Nvfp4 {
                input: linear.input,
                scale_bits: q.activation_global_scale.to_bits(),
            }),
            _ => None,
        };
        let first = signature(projections[0].0)?;
        projections
            .iter()
            .all(|(linear, _)| signature(linear) == Some(first))
            .then_some(first)
    }

    /// All projections consume the same input. Only exact scale/type/dimension
    /// matches share activation bytes; output shape, weight scale and bias remain
    /// specific to each GEMM. The stream orders all reads before workspace reuse.
    pub fn bf16_group(
        ops: &Ops<'_>,
        x: DevicePtr,
        rows: usize,
        projections: &[(&Self, DevicePtr)],
    ) -> Result<()> {
        match Self::shared_input(projections) {
            Some(SharedInput::Fp8 { input, scale_bits }) => {
                let scratch = ops
                    .fp8_scratch
                    .context("FP8 model requires resident quantization workspace")?;
                let scale = f32::from_bits(scale_bits);
                ops.quantize_fp8(x, rows, input, scale, scratch)?;
                for &(linear, out) in projections {
                    linear.fp8_prequantized(ops, out, rows)?;
                }
            }
            Some(SharedInput::Nvfp4 { input, scale_bits }) => {
                let scratch = ops
                    .nvfp4_scratch
                    .context("NVFP4 model requires resident quantization workspace")?;
                let scale = f32::from_bits(scale_bits);
                ops.quantize_nvfp4(x, rows, input, scale, scratch)?;
                for &(linear, out) in projections {
                    let quant = linear.nvfp4.context("shared NVFP4 projection metadata")?;
                    spark_runtime::cublaslt::nvfp4_gemm_act_weight_t(
                        scratch.packed.0,
                        scratch.scales.0,
                        linear.weight.0,
                        quant.scales.0,
                        scale * quant.weight_global_scale,
                        linear.bias.map(|p| p.0),
                        out.0,
                        u32::try_from(rows)?,
                        u32::try_from(linear.output)?,
                        u32::try_from(input)?,
                        ops.stream,
                    )?;
                }
            }
            None => {
                for &(linear, out) in projections {
                    linear.bf16(ops, x, out, rows)?;
                }
            }
        }
        Ok(())
    }

    /// Consume already packed compatible activations without another quantizer.
    fn fp8_prequantized(&self, ops: &Ops<'_>, out: DevicePtr, rows: usize) -> Result<()> {
        let quant = self.fp8.context("shared FP8 projection metadata")?;
        let scratch = ops
            .fp8_scratch
            .context("FP8 model requires resident quantization workspace")?;
        let alpha = quant.activation_scale * quant.weight_scale;
        ensure!(alpha.is_finite() && alpha > 0.0, "FP8 scale product");
        spark_runtime::cublaslt::fp8_gemm_act_weight_t_scaled(
            scratch.packed.0,
            self.weight.0,
            self.bias.map(|p| p.0),
            out.0,
            u32::try_from(rows)?,
            u32::try_from(self.output)?,
            u32::try_from(self.input)?,
            alpha,
            false,
            ops.stream,
        )
    }

    /// Only compatible FP8 expert groups fuse normalization and packing.
    /// BF16, NVFP4 and incompatible scales retain the original two-stage path.
    #[allow(clippy::too_many_arguments)]
    pub fn normalized_group(
        ops: &Ops<'_>,
        x: DevicePtr,
        norm: DevicePtr,
        rows: usize,
        width: usize,
        weight: DevicePtr,
        modulation: Option<DevicePtr>,
        eps: f32,
        projections: &[(&Self, DevicePtr)],
    ) -> Result<()> {
        if let (Some(modulation), Some(SharedInput::Fp8 { input, scale_bits })) =
            (modulation, Self::shared_input(projections))
        {
            ensure!(input == width, "normalization projection width");
            let scratch = ops
                .fp8_scratch
                .context("FP8 model requires resident quantization workspace")?;
            ops.adarms_quantize_fp8(
                x,
                modulation,
                rows,
                width,
                eps,
                f32::from_bits(scale_bits),
                scratch,
            )?;
            for &(linear, out) in projections {
                linear.fp8_prequantized(ops, out, rows)?;
            }
            Ok(())
        } else {
            match modulation {
                Some(modulation) => ops.adarms(x, modulation, norm, rows, width, eps)?,
                None => ops.rms_norm(x, weight, norm, rows, width, eps)?,
            }
            Self::bf16_group(ops, norm, rows, projections)
        }
    }

    pub fn load(w: &Weights, name: &str, input: usize, output: usize, bias: bool) -> Result<Self> {
        Ok(Self {
            weight: w.get(&format!("{name}.weight"), &[output, input])?,
            nvfp4: w.nvfp4(&format!("{name}.weight"))?,
            fp8: w.fp8(&format!("{name}.weight"))?,
            bias: if bias {
                Some(w.get(&format!("{name}.bias"), &[output])?)
            } else {
                None
            },
            input,
            output,
        })
    }

    pub fn bf16(&self, ops: &Ops<'_>, x: DevicePtr, out: DevicePtr, rows: usize) -> Result<()> {
        if let Some(quant) = self.fp8 {
            return ops.linear_fp8(
                x,
                self.weight,
                self.bias,
                out,
                rows,
                self.output,
                self.input,
                quant.activation_scale,
                quant.weight_scale,
                ops.fp8_scratch
                    .context("FP8 model requires resident quantization workspace")?,
            );
        }
        if let Some(quant) = self.nvfp4 {
            return ops.linear_nvfp4(
                x,
                self.weight,
                quant.scales,
                self.bias,
                out,
                rows,
                self.output,
                self.input,
                quant.activation_global_scale,
                quant.weight_global_scale,
                ops.nvfp4_scratch
                    .context("NVFP4 model requires resident quantization workspace")?,
            );
        }
        ops.linear_bf16(
            x,
            self.weight,
            self.bias,
            out,
            rows,
            self.output,
            self.input,
        )
    }

    /// Preserve both BF16 GELU/product roundings while eliminating the
    /// materialized activation when the consumer uses FP8 or NVFP4.
    pub fn gelu_mul_bf16(
        &self,
        ops: &Ops<'_>,
        gate: DevicePtr,
        up: DevicePtr,
        out: DevicePtr,
        rows: usize,
    ) -> Result<()> {
        if self.fp8.is_some() {
            let scratch = ops
                .fp8_scratch
                .context("FP8 model requires resident quantization workspace")?;
            let quant = self.fp8.context("FP8 projection metadata")?;
            ops.gelu_mul_quantize_fp8(gate, up, rows, self.input, quant.activation_scale, scratch)?;
            self.fp8_prequantized(ops, out, rows)
        } else if let Some(quant) = self.nvfp4 {
            let scratch = ops
                .nvfp4_scratch
                .context("NVFP4 model requires resident quantization workspace")?;
            ops.gelu_mul_quantize_nvfp4(
                gate,
                up,
                rows,
                self.input,
                quant.activation_global_scale,
                scratch,
            )?;
            spark_runtime::cublaslt::nvfp4_gemm_act_weight_t(
                scratch.packed.0,
                scratch.scales.0,
                self.weight.0,
                quant.scales.0,
                quant.activation_global_scale * quant.weight_global_scale,
                self.bias.map(|p| p.0),
                out.0,
                u32::try_from(rows)?,
                u32::try_from(self.output)?,
                u32::try_from(self.input)?,
                ops.stream,
            )
        } else {
            ops.gelu_mul(gate, up, gate, rows * self.input)?;
            self.bf16(ops, gate, out, rows)
        }
    }

    pub fn f32(&self, ops: &Ops<'_>, x: DevicePtr, out: DevicePtr, rows: usize) -> Result<()> {
        ensure!(
            self.nvfp4.is_none() && self.fp8.is_none(),
            "quantization cannot replace a sensitive FP32 projection"
        );
        ops.linear_f32(
            x,
            self.weight,
            self.bias,
            out,
            rows,
            self.output,
            self.input,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp8(input: usize, scale: f32) -> Linear {
        Linear {
            weight: DevicePtr::NULL,
            bias: None,
            input,
            output: 256,
            nvfp4: None,
            fp8: Some(Fp8Weight {
                activation_scale: scale,
                weight_scale: 0.5,
            }),
        }
    }

    fn nvfp4(input: usize, scale: f32) -> Linear {
        let mut linear = fp8(input, scale);
        linear.fp8 = None;
        linear.nvfp4 = Some(Nvfp4Weight {
            scales: DevicePtr::NULL,
            activation_global_scale: scale,
            weight_global_scale: 0.25,
        });
        linear
    }

    fn shared(linears: &[&Linear]) -> bool {
        let outputs: Vec<_> = linears.iter().map(|&l| (l, DevicePtr::NULL)).collect();
        Linear::shared_input(&outputs).is_some()
    }

    #[test]
    fn compatible_groups_preserve_individual_outputs_biases_and_weight_scales() {
        for mut pair in [
            [fp8(1024, 0.1), fp8(1024, 0.1)],
            [nvfp4(2048, 0.1), nvfp4(2048, 0.1)],
        ] {
            pair[1].output = 2048;
            pair[1].bias = Some(DevicePtr::NULL);
            if let Some(q) = &mut pair[1].fp8 {
                q.weight_scale = 0.75;
            }
            if let Some(q) = &mut pair[1].nvfp4 {
                q.weight_global_scale = 0.75;
            }
            assert!(shared(&[&pair[0], &pair[1]]));
            assert!(shared(&[&pair[0], &pair[1], &pair[0]]));
        }
    }

    #[test]
    fn mixed_dimensions_precision_or_scale_use_individual_dispatch() {
        let reference = fp8(1024, 0.1);
        let next_scale = f32::from_bits(0.1_f32.to_bits() + 1);
        let mut bf16 = fp8(1024, 0.1);
        bf16.fp8 = None;
        let mut ambiguous = fp8(1024, 0.1);
        ambiguous.nvfp4 = nvfp4(1024, 0.1).nvfp4;
        for other in [
            fp8(2048, 0.1),
            fp8(1024, next_scale),
            nvfp4(1024, 0.1),
            bf16,
            ambiguous,
        ] {
            assert!(!shared(&[&reference, &other]));
        }
        assert!(!shared(&[]));
        assert!(!shared(&[&reference]));
        let nv = nvfp4(2048, 0.1);
        assert!(!shared(&[&nv, &nvfp4(2048, next_scale)]));
        assert!(!shared(&[&nv, &nvfp4(1024, 0.1)]));
    }
}
