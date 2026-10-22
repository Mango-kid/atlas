// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit diagnostic boundary; normal inference never downloads stage data.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Receives synchronized intermediate tensors, always represented as FP32.
/// The consumer owns serialization, storage and comparison policy.
pub trait TraceSink {
    fn tensor(&mut self, name: &str, shape: &[usize], values: &[f32]) -> Result<()>;
}

pub(super) fn capture(
    gpu: &dyn GpuBackend,
    sink: &mut Option<&mut dyn TraceSink>,
    name: &str,
    shape: &[usize],
    ptr: DevicePtr,
    bf16: bool,
) -> Result<()> {
    let Some(sink) = sink.as_deref_mut() else {
        return Ok(());
    };
    let count = shape.iter().try_fold(1usize, |a, b| a.checked_mul(*b));
    let count = count.ok_or_else(|| anyhow::anyhow!("trace shape overflow"))?;
    ensure!(
        count <= 4 * 1024 * 1024,
        "trace tensor exceeds diagnostic limit"
    );
    let mut bytes = vec![0; count * if bf16 { 2 } else { 4 }];
    gpu.copy_d2h_on_stream(ptr, &mut bytes, gpu.default_stream())?;
    let values = if bf16 {
        bytes
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect::<Vec<_>>()
    } else {
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect::<Vec<_>>()
    };
    sink.tensor(name, shape, &values)
}
