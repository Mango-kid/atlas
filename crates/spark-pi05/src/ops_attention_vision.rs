// SPDX-License-Identifier: AGPL-3.0-only
//! Exact vision attention for 256 tokens, 16 heads, and 72 channels per head.
use crate::ops::Ops;
use anyhow::Result;
use spark_runtime::{gpu::DevicePtr, kernel_args::KernelLaunch};

impl Ops<'_> {
    /// Preserves the baseline QK, softmax reduction tree, and PV accumulation.
    pub fn attention_vision(
        &self,
        q: DevicePtr,
        k: DevicePtr,
        v: DevicePtr,
        out: DevicePtr,
        scores: DevicePtr,
    ) -> Result<()> {
        KernelLaunch::new(self.gpu, self.gpu.kernel("pi05", "pi05_attention_qk")?)
            .grid([4, 16, 16])
            .block([128, 1, 1])
            .arg_ptr(q)
            .arg_ptr(k)
            .arg_ptr(scores)
            .arg_ptr(DevicePtr::NULL)
            .arg_u32(256)
            .arg_u32(256)
            .arg_u32(16)
            .arg_u32(16)
            .arg_u32(72)
            .launch(self.stream)?;
        KernelLaunch::new(self.gpu, self.gpu.kernel("pi05", "pi05_vision_softmax")?)
            .grid([1024, 1, 1])
            .block([128, 1, 1])
            .arg_ptr(scores)
            .launch(self.stream)?;
        KernelLaunch::new(self.gpu, self.gpu.kernel("pi05", "pi05_vision_pv")?)
            .grid([1, 16, 16])
            .block([160, 1, 1])
            .arg_ptr(scores)
            .arg_ptr(v)
            .arg_ptr(out)
            .arg_u32(256)
            .arg_u32(256)
            .arg_u32(16)
            .arg_u32(16)
            .arg_u32(72)
            .launch(self.stream)
    }
}
