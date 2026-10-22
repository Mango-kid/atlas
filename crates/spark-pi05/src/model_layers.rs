// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::buffers::Scratch;
use crate::{ops::Ops, weights::Weights};
#[path = "model_linear.rs"]
mod linear;
pub(super) use linear::Linear;

pub(super) const PREFIX: &str = "paligemma_with_expert.paligemma.model";
pub(super) const EXPERT: &str = "paligemma_with_expert.gemma_expert.model";

pub(super) struct VisionLayer {
    norm1: (DevicePtr, DevicePtr),
    norm2: (DevicePtr, DevicePtr),
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    up: Linear,
    down: Linear,
}

impl VisionLayer {
    pub fn load(w: &Weights, index: usize) -> Result<Self> {
        let p = format!("{PREFIX}.vision_tower.vision_model.encoder.layers.{index}");
        let norm = |name| -> Result<(DevicePtr, DevicePtr)> {
            Ok((
                w.get(&format!("{p}.{name}.weight"), &[1152])?,
                w.get(&format!("{p}.{name}.bias"), &[1152])?,
            ))
        };
        Ok(Self {
            norm1: norm("layer_norm1")?,
            norm2: norm("layer_norm2")?,
            q: Linear::load(w, &format!("{p}.self_attn.q_proj"), 1152, 1152, true)?,
            k: Linear::load(w, &format!("{p}.self_attn.k_proj"), 1152, 1152, true)?,
            v: Linear::load(w, &format!("{p}.self_attn.v_proj"), 1152, 1152, true)?,
            o: Linear::load(w, &format!("{p}.self_attn.out_proj"), 1152, 1152, true)?,
            up: Linear::load(w, &format!("{p}.mlp.fc1"), 1152, 4304, true)?,
            down: Linear::load(w, &format!("{p}.mlp.fc2"), 4304, 1152, true)?,
        })
    }

    pub fn forward(&self, ops: &Ops<'_>, s: &Scratch, eps: f32) -> Result<()> {
        ops.layer_norm(s.hidden, self.norm1.0, self.norm1.1, s.norm, 512, 1152, eps)?;
        Linear::bf16_group(
            ops,
            s.norm,
            512,
            &[(&self.q, s.q), (&self.k, s.k), (&self.v, s.v)],
        )?;
        // Matrix projections batch both views; attention remains independent per camera.
        for image in 0..2 {
            let offset = image * 256 * 1152 * 2;
            ops.attention(
                s.q.offset(offset),
                s.k.offset(offset),
                s.v.offset(offset),
                s.attention.offset(offset),
                256,
                256,
                16,
                16,
                72,
                DevicePtr::NULL,
                s.scores,
            )?;
        }
        self.o.bf16(ops, s.attention, s.delta, 512)?;
        ops.residual(s.hidden, s.delta, None, 512, 1152)?;
        ops.layer_norm(s.hidden, self.norm2.0, self.norm2.1, s.norm, 512, 1152, eps)?;
        self.up.bf16(ops, s.norm, s.up, 512)?;
        ops.gelu_bf16(s.up, s.gate, 512 * 4304)?;
        self.down.bf16(ops, s.gate, s.delta, 512)?;
        ops.residual(s.hidden, s.delta, None, 512, 1152)
    }
}

pub(super) struct TransformerLayer {
    pub norm1: DevicePtr,
    pub norm2: DevicePtr,
    pub modulation1: Option<Linear>,
    pub modulation2: Option<Linear>,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
    pub width: usize,
    pub mlp: usize,
}

impl TransformerLayer {
    pub fn load(w: &Weights, index: usize, expert: bool) -> Result<Self> {
        let (root, width, mlp) = if expert {
            (EXPERT.to_string(), 1024, 4096)
        } else {
            (format!("{PREFIX}.language_model"), 2048, 16384)
        };
        let p = format!("{root}.layers.{index}");
        let norm = |name| {
            if expert {
                Ok(DevicePtr::NULL)
            } else {
                w.get(&format!("{p}.{name}.weight"), &[width])
            }
        };
        let modulation = |name| -> Result<Option<Linear>> {
            if expert {
                Ok(Some(Linear::load(
                    w,
                    &format!("{p}.{name}.dense"),
                    1024,
                    3072,
                    true,
                )?))
            } else {
                Ok(None)
            }
        };
        Ok(Self {
            norm1: norm("input_layernorm")?,
            norm2: norm("post_attention_layernorm")?,
            modulation1: modulation("input_layernorm")?,
            modulation2: modulation("post_attention_layernorm")?,
            q: Linear::load(w, &format!("{p}.self_attn.q_proj"), width, 2048, false)?,
            k: Linear::load(w, &format!("{p}.self_attn.k_proj"), width, 256, false)?,
            v: Linear::load(w, &format!("{p}.self_attn.v_proj"), width, 256, false)?,
            o: Linear::load(w, &format!("{p}.self_attn.o_proj"), 2048, width, false)?,
            gate: Linear::load(w, &format!("{p}.mlp.gate_proj"), width, mlp, false)?,
            up: Linear::load(w, &format!("{p}.mlp.up_proj"), width, mlp, false)?,
            down: Linear::load(w, &format!("{p}.mlp.down_proj"), mlp, width, false)?,
            width,
            mlp,
        })
    }

    /// The last prefix layer only needs K/V: its output has no consumer.
    #[allow(clippy::too_many_arguments)]
    pub fn cache_only(
        &self,
        ops: &Ops<'_>,
        s: &Scratch,
        rows: usize,
        positions: DevicePtr,
        key: DevicePtr,
        value: DevicePtr,
        eps: f32,
    ) -> Result<()> {
        ops.rms_norm(s.hidden, self.norm1, s.norm, rows, self.width, eps)?;
        Linear::bf16_group(ops, s.norm, rows, &[(&self.k, key), (&self.v, value)])?;
        ops.rope(key, positions, rows, 1, 256)
    }

    /// Expert modulation pointers each contain scale, shift, gate in FP32.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        ops: &Ops<'_>,
        s: &Scratch,
        rows: usize,
        positions: DevicePtr,
        key: DevicePtr,
        value: DevicePtr,
        prefix_rows: usize,
        modulation: Option<(DevicePtr, DevicePtr)>,
        eps: f32,
    ) -> Result<()> {
        self.normalize(
            ops,
            s.hidden,
            s.norm,
            self.norm1,
            modulation.map(|m| m.0),
            rows,
            eps,
        )?;
        Linear::bf16_group(
            ops,
            s.norm,
            rows,
            &[
                (&self.q, s.q),
                (&self.k, key.offset(prefix_rows * 256 * 2)),
                (&self.v, value.offset(prefix_rows * 256 * 2)),
            ],
        )?;
        ops.rope(s.q, positions, rows, 8, 256)?;
        ops.rope(key.offset(prefix_rows * 256 * 2), positions, rows, 1, 256)?;
        ops.attention(
            s.q,
            key,
            value,
            s.attention,
            rows,
            prefix_rows + rows,
            8,
            1,
            256,
            DevicePtr::NULL,
            s.scores,
        )?;
        self.o.bf16(ops, s.attention, s.delta, rows)?;
        ops.residual(
            s.hidden,
            s.delta,
            modulation.map(|m| m.0.offset(2 * self.width * 4)),
            rows,
            self.width,
        )?;
        self.normalize(
            ops,
            s.hidden,
            s.norm,
            self.norm2,
            modulation.map(|m| m.1),
            rows,
            eps,
        )?;
        Linear::bf16_group(ops, s.norm, rows, &[(&self.gate, s.gate), (&self.up, s.up)])?;
        ops.gelu_mul(s.gate, s.up, s.gate, rows * self.mlp)?;
        self.down.bf16(ops, s.gate, s.delta, rows)?;
        ops.residual(
            s.hidden,
            s.delta,
            modulation.map(|m| m.1.offset(2 * self.width * 4)),
            rows,
            self.width,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn normalize(
        &self,
        ops: &Ops<'_>,
        x: DevicePtr,
        out: DevicePtr,
        weight: DevicePtr,
        modulation: Option<DevicePtr>,
        rows: usize,
        eps: f32,
    ) -> Result<()> {
        match modulation {
            Some(m) => ops.adarms(x, m, out, rows, self.width, eps),
            None => ops.rms_norm(x, weight, out, rows, self.width, eps),
        }
    }
}
