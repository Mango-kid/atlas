// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::sync::Arc;

/// Owned allocations; partially constructed models release memory on errors too.
pub(super) struct Buffers {
    gpu: Arc<dyn GpuBackend>,
    allocations: Vec<DevicePtr>,
}

impl Buffers {
    pub fn new(gpu: Arc<dyn GpuBackend>) -> Self {
        Self {
            gpu,
            allocations: Vec::new(),
        }
    }

    pub fn alloc(&mut self, bytes: usize) -> Result<DevicePtr> {
        let ptr = self.gpu.alloc(bytes)?;
        self.allocations.push(ptr);
        Ok(ptr)
    }

    pub fn f32(&mut self, values: &[f32]) -> Result<DevicePtr> {
        let ptr = self.alloc(values.len() * 4)?;
        upload_f32(self.gpu.as_ref(), ptr, values)?;
        Ok(ptr)
    }
}

impl Drop for Buffers {
    fn drop(&mut self) {
        let _ = self.gpu.synchronize(self.gpu.default_stream());
        for ptr in self.allocations.drain(..).rev() {
            let _ = self.gpu.free(ptr);
        }
    }
}

pub(super) fn upload_f32(gpu: &dyn GpuBackend, ptr: DevicePtr, values: &[f32]) -> Result<()> {
    gpu.copy_h2d(bytemuck::cast_slice(values), ptr)
}

pub(super) struct Scratch {
    pub hidden: DevicePtr,
    pub norm: DevicePtr,
    pub q: DevicePtr,
    pub k: DevicePtr,
    pub v: DevicePtr,
    pub attention: DevicePtr,
    pub scores: DevicePtr,
    pub delta: DevicePtr,
    pub gate: DevicePtr,
    pub up: DevicePtr,
    pub f32: DevicePtr,
}

impl Scratch {
    pub fn new(buffers: &mut Buffers, rows: usize, hidden: usize, mlp: usize) -> Result<Self> {
        Ok(Self {
            hidden: buffers.alloc(rows * hidden * 2)?,
            norm: buffers.alloc(rows * hidden * 2)?,
            q: buffers.alloc(rows * 2048 * 2)?,
            k: buffers.alloc(rows * 1152 * 2)?,
            v: buffers.alloc(rows * 1152 * 2)?,
            attention: buffers.alloc(rows * 2048 * 2)?,
            scores: buffers.alloc(
                (rows * 722 * 16 * 2)
                    .max(crate::ops_attention::mqa_scratch_bytes(rows, 8, 722, 256)),
            )?,
            delta: buffers.alloc(rows * hidden * 2)?,
            gate: buffers.alloc(rows * mlp * 2)?,
            up: buffers.alloc(rows * mlp * 2)?,
            f32: buffers.alloc(rows * hidden * 4)?,
        })
    }
}
