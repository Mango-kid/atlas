// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    buffers::Buffers,
    layers::{EXPERT, Linear, TransformerLayer},
};
use crate::{ops::Ops, weights::Weights};
use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

fn time_embeddings(steps: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(steps * 1024);
    let dt = -1.0_f32 / steps as f32;
    let mut time = 1.0_f32;
    for _ in 0..steps {
        for cosine in [false, true] {
            for i in 0..512 {
                let period = 0.004_f64 * 1000_f64.powf(i as f64 / 511.0);
                let angle = (2.0 * std::f64::consts::PI / period) * time as f64;
                values.push(if cosine {
                    angle.cos() as f32
                } else {
                    angle.sin() as f32
                });
            }
        }
        time += dt;
    }
    values
}

type Modulations = (Vec<(DevicePtr, DevicePtr)>, DevicePtr);

pub(super) fn prepare_time(
    ops: &Ops<'_>,
    weights: &Weights,
    expert: &[TransformerLayer],
    buffers: &mut Buffers,
    steps: usize,
) -> Result<Modulations> {
    let time = buffers.f32(&time_embeddings(steps))?;
    let hidden = buffers.alloc(steps * 1024 * 4)?;
    let condition = buffers.alloc(steps * 1024 * 4)?;
    Linear::load(weights, "time_mlp_in", 1024, 1024, true)?.f32(ops, time, hidden, steps)?;
    ops.silu_f32(hidden, steps * 1024)?;
    Linear::load(weights, "time_mlp_out", 1024, 1024, true)?.f32(ops, hidden, condition, steps)?;
    ops.silu_f32(condition, steps * 1024)?;
    let mut modulation = Vec::with_capacity(expert.len());
    for layer in expert {
        let first = buffers.alloc(steps * 3072 * 4)?;
        let second = buffers.alloc(steps * 3072 * 4)?;
        layer
            .modulation1
            .as_ref()
            .expect("expert always has adaptive RMSNorm")
            .f32(ops, condition, first, steps)?;
        layer
            .modulation2
            .as_ref()
            .expect("expert always has adaptive RMSNorm")
            .f32(ops, condition, second, steps)?;
        modulation.push((first, second));
    }
    let final_modulation = buffers.alloc(steps * 3072 * 4)?;
    Linear::load(weights, &format!("{EXPERT}.norm.dense"), 1024, 3072, true)?.f32(
        ops,
        condition,
        final_modulation,
        steps,
    )?;
    Ok((modulation, final_modulation))
}

#[cfg(test)]
mod tests {
    use super::time_embeddings;

    #[test]
    fn time_embedding_layout_and_float32_euler_schedule() {
        let values = time_embeddings(10);
        assert_eq!(values.len(), 10 * 1024);
        // Last period is four seconds: t=1 gives sin(pi/2), cos(pi/2).
        assert!((values[511] - 1.0).abs() < 1e-7);
        assert!(values[1023].abs() < 1e-7);
        let mut time = 1.0_f32;
        for _ in 0..9 {
            time -= 0.1_f32;
        }
        let expected = (std::f64::consts::TAU / 0.004 * time as f64).sin() as f32;
        assert!((values[9 * 1024] - expected).abs() < 1e-6);
    }
}
