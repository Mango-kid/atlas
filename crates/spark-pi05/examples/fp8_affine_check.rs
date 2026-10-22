// SPDX-License-Identifier: AGPL-3.0-only
//! Validate real calibrated FP8 linear fixtures, including two-camera batches.
use anyhow::{Result, ensure};
use serde::Deserialize;
use spark_runtime::{
    cuda_backend::AtlasCudaBackend,
    gpu::{DevicePtr, GpuBackend},
};
use std::path::Path;
#[derive(Deserialize)]
struct Case {
    name: String,
    path: String,
    m: usize,
    n: usize,
    k: usize,
    activation_scale: f32,
    weight_scale: f32,
    bias: bool,
}
fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 3,
        "usage: fp8_affine_check FIXTURE_ROOT OUTPUT_JSON"
    );
    let root = Path::new(&args[1]);
    let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(root.join("affine-cases.json"))?)?;
    let gpu = AtlasCudaBackend::new(
        0,
        &[(
            "pi05",
            include_bytes!(concat!(env!("OUT_DIR"), "/pi05.ptx")),
        )],
    )?;
    let stream = gpu.default_stream();
    let ops = spark_pi05::ops::Ops::new(&gpu, stream)?;
    let mut records = Vec::new();
    for c in cases {
        for repeat in 1..=if c.m == 256 { 2 } else { 1 } {
            let path = root.join(&c.path);
            let input = std::fs::read(path.join("input.bf16"))?.repeat(repeat);
            let activation = std::fs::read(path.join("activation.fp8"))?.repeat(repeat);
            let expected = std::fs::read(path.join("expected.bf16"))?.repeat(repeat);
            let m = c.m * repeat;
            let a = upload(&gpu, &input)?;
            let w = upload(&gpu, &std::fs::read(path.join("weight.fp8"))?)?;
            let bias = if c.bias {
                Some(upload(&gpu, &std::fs::read(path.join("bias.bf16"))?)?)
            } else {
                None
            };
            let out = gpu.alloc(m * c.n * 2)?;
            let scratch = spark_pi05::ops::Fp8Scratch {
                packed: gpu.alloc(m * c.k)?,
                max_rows: m,
                max_k: c.k,
            };
            ops.linear_fp8(
                a,
                w,
                bias,
                out,
                m,
                c.n,
                c.k,
                c.activation_scale,
                c.weight_scale,
                &scratch,
            )?;
            let mut actual_activation = vec![0; m * c.k];
            gpu.copy_d2h_on_stream(scratch.packed, &mut actual_activation, stream)?;
            let activation_different = actual_activation
                .iter()
                .zip(&activation)
                .filter(|(a, b)| a != b)
                .count();
            ensure!(
                activation_different == 0,
                "{} activation mismatch: {activation_different}",
                c.name
            );
            let mut actual = vec![0; m * c.n * 2];
            gpu.copy_d2h_on_stream(out, &mut actual, stream)?;
            let mut max_abs = 0_f32;
            let mut sum = 0_f64;
            let mut different = 0;
            for (aa, bb) in actual.chunks_exact(2).zip(expected.chunks_exact(2)) {
                let a = half::bf16::from_le_bytes([aa[0], aa[1]]).to_f32();
                let b = half::bf16::from_le_bytes([bb[0], bb[1]]).to_f32();
                let err = (a - b).abs();
                ensure!(
                    a.is_finite() && err <= 0.002 + 0.008 * b.abs(),
                    "{} m={m} output {a} != {b}",
                    c.name
                );
                max_abs = max_abs.max(err);
                sum += f64::from(err);
                different += usize::from(a != b);
            }
            records.push(serde_json::json!({"name":c.name,"m":m,"n":c.n,"k":c.k,"activation_different":activation_different,"max_abs":max_abs,"mean_abs":sum/(m*c.n) as f64,"different":different}));
            for ptr in [a, w, out, scratch.packed] {
                gpu.free(ptr)?;
            }
            if let Some(ptr) = bias {
                gpu.free(ptr)?;
            }
        }
    }
    std::fs::write(&args[2], serde_json::to_vec_pretty(&records)?)?;
    Ok(())
}
