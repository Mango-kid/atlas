// SPDX-License-Identifier: AGPL-3.0-only
//! Capture actual Atlas NVFP4 affine outputs for independent ModelOpt analysis.
use anyhow::{Result, ensure};
use serde::Deserialize;
use spark_pi05::ops::{Nvfp4Scratch, Ops};
use spark_runtime::{
    cuda_backend::AtlasCudaBackend,
    gpu::{DevicePtr, GpuBackend},
};
use std::path::Path;

#[derive(Deserialize)]
struct Case {
    name: String,
    m: usize,
    n: usize,
    k: usize,
    activation_global_scale: f32,
    weight_global_scale: f32,
}

fn upload(gpu: &dyn GpuBackend, path: &Path) -> Result<DevicePtr> {
    let bytes = std::fs::read(path)?;
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, ptr)?;
    Ok(ptr)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 2, "usage: nvfp4_capture FIXTURE_DIRECTORY");
    let root = Path::new(&args[1]);
    let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(root.join("cases.json"))?)?;
    let ptx = include_bytes!(concat!(env!("OUT_DIR"), "/pi05.ptx"));
    let gpu = AtlasCudaBackend::new(0, &[("pi05", ptx)])?;
    let ops = Ops::new(&gpu, gpu.default_stream())?;
    for case in cases {
        let path = root.join(case.name);
        let input = upload(&gpu, &path.join("input.bf16"))?;
        let weight = upload(&gpu, &path.join("weight.fp4"))?;
        let scales = upload(&gpu, &path.join("weight.scale"))?;
        let output = gpu.alloc(case.m * case.n * 2)?;
        let scratch = Nvfp4Scratch {
            packed: gpu.alloc(Nvfp4Scratch::packed_bytes(case.m, case.k))?,
            scales: gpu.alloc(Nvfp4Scratch::scale_bytes(case.m, case.k))?,
            max_rows: case.m,
            max_k: case.k,
        };
        ops.linear_nvfp4(
            input,
            weight,
            scales,
            None,
            output,
            case.m,
            case.n,
            case.k,
            case.activation_global_scale,
            case.weight_global_scale,
            &scratch,
        )?;
        for (ptr, bytes, name) in [
            (output, case.m * case.n * 2, "native.bf16"),
            (
                scratch.packed,
                Nvfp4Scratch::packed_bytes(case.m, case.k),
                "native.fp4",
            ),
            (
                scratch.scales,
                Nvfp4Scratch::scale_bytes(case.m, case.k),
                "native.scale",
            ),
        ] {
            let mut data = vec![0; bytes];
            gpu.copy_d2h_on_stream(ptr, &mut data, ops.stream)?;
            std::fs::write(path.join(name), data)?;
        }
        for ptr in [
            input,
            weight,
            scales,
            output,
            scratch.packed,
            scratch.scales,
        ] {
            gpu.free(ptr)?;
        }
    }
    Ok(())
}
