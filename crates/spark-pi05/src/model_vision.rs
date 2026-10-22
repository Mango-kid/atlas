// SPDX-License-Identifier: AGPL-3.0-only
//! Batch camera projections while keeping each camera's attention independent.
use super::{NativeModel, TraceSink, capture};
use crate::ops::Ops;
use anyhow::Result;

impl NativeModel {
    pub(super) fn embed_images(
        &self,
        ops: &Ops<'_>,
        sink: &mut Option<&mut dyn TraceSink>,
    ) -> Result<()> {
        let s = &self.image_scratch;
        ops.linear_f32(
            self.patches,
            self.patch_weight,
            Some(self.patch_bias),
            s.f32,
            512,
            1152,
            588,
        )?;
        for image in 0..2 {
            let patch = s.f32.offset(image * 256 * 1152 * 4);
            ops.add_f32(patch, self.patch_position, patch, 256 * 1152)?;
        }
        ops.cast_to_bf16(s.f32, s.hidden, 512 * 1152)?;
        for image in 0..2 {
            capture(
                self.gpu.as_ref(),
                sink,
                &format!("image.{image}.embedding"),
                &[256, 1152],
                s.hidden.offset(image * 256 * 1152 * 2),
                true,
            )?;
        }
        for (index, layer) in self.vision.iter().enumerate() {
            layer.forward(ops, s, self.config.layer_norm_eps)?;
            if index == 0 || index == 26 {
                for image in 0..2 {
                    capture(
                        self.gpu.as_ref(),
                        sink,
                        &format!("image.{image}.layer.{index}"),
                        &[256, 1152],
                        s.hidden.offset(image * 256 * 1152 * 2),
                        true,
                    )?;
                }
            }
        }
        ops.layer_norm(
            s.hidden,
            self.vision_norm.0,
            self.vision_norm.1,
            s.norm,
            512,
            1152,
            self.config.layer_norm_eps,
        )?;
        self.projector
            .bf16(ops, s.norm, self.prefix_scratch.hidden, 512)?;
        for image in 0..2 {
            capture(
                self.gpu.as_ref(),
                sink,
                &format!("image.{image}.projected"),
                &[256, 2048],
                self.prefix_scratch.hidden.offset(image * 256 * 2048 * 2),
                true,
            )?;
        }
        Ok(())
    }
}
