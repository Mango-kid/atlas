// SPDX-License-Identifier: AGPL-3.0-only
//! Actual native NVFP4 codec/GEMM oracle and CUDA-event microbenchmark.
use anyhow::{Result, ensure};
use serde::Deserialize;
use spark_runtime::{
    cuda_backend::AtlasCudaBackend,
    gpu::{DevicePtr, GpuBackend},
};
use std::{ffi::c_void, path::Path};

#[derive(Deserialize)]
struct Case {
    name: String,
    m: u32,
    n: u32,
    k: u32,
    activation_global_scale: f32,
    weight_global_scale: f32,
}

unsafe extern "C" {
    fn cuEventCreate(event: *mut *mut c_void, flags: u32) -> i32;
    fn cuEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cuEventSynchronize(event: *mut c_void) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
    fn cuEventDestroy_v2(event: *mut c_void) -> i32;
}

fn upload(gpu: &dyn GpuBackend, path: &Path) -> Result<DevicePtr> {
    let data = std::fs::read(path)?;
    let ptr = gpu.alloc(data.len())?;
    gpu.copy_h2d(&data, ptr)?;
    Ok(ptr)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 3,
        "usage: nvfp4_check FIXTURE_DIRECTORY OUTPUT_JSON"
    );
    let root = Path::new(&args[1]);
    let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(root.join("cases.json"))?)?;
    let ptx = include_bytes!(concat!(env!("OUT_DIR"), "/pi05.ptx"));
    let gpu = AtlasCudaBackend::new(0, &[("pi05", ptx)])?;
    let stream = gpu.default_stream();
    let ops = spark_pi05::ops::Ops::new(&gpu, stream)?;
    let mut records = Vec::new();
    for c in cases {
        let path = root.join(&c.name);
        let a = upload(&gpu, &path.join("input.bf16"))?;
        let w = upload(&gpu, &path.join("weight.fp4"))?;
        let ws = upload(&gpu, &path.join("weight.scale"))?;
        let scratch = spark_pi05::ops::Nvfp4Scratch {
            packed: gpu.alloc(c.m as usize * c.k as usize / 2)?,
            scales: gpu.alloc(spark_pi05::ops::Nvfp4Scratch::scale_bytes(
                c.m as usize,
                c.k as usize,
            ))?,
            max_rows: c.m as usize,
            max_k: c.k as usize,
        };
        ops.quantize_nvfp4(
            a,
            c.m as usize,
            c.k as usize,
            c.activation_global_scale,
            &scratch,
        )?;
        for (ptr, file) in [
            (scratch.packed, "activation.fp4"),
            (scratch.scales, "activation.scale"),
        ] {
            let expected = std::fs::read(path.join(file))?;
            let mut actual = vec![0; expected.len()];
            gpu.copy_d2h_on_stream(ptr, &mut actual, stream)?;
            let different = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
            ensure!(
                different == 0,
                "{} {} has {} different bytes",
                c.name,
                file,
                different
            );
        }
        let bias = upload(&gpu, &path.join("bias.bf16"))?;
        let out = gpu.alloc(c.m as usize * c.n as usize * 2)?;
        for use_bias in [false, true] {
            let run = || {
                ops.linear_nvfp4(
                    a,
                    w,
                    ws,
                    use_bias.then_some(bias),
                    out,
                    c.m as usize,
                    c.n as usize,
                    c.k as usize,
                    c.activation_global_scale,
                    c.weight_global_scale,
                    &scratch,
                )
            };
            for _ in 0..10 {
                run()?;
            }
            let mut bytes = vec![0u8; c.m as usize * c.n as usize * 2];
            gpu.copy_d2h_on_stream(out, &mut bytes, stream)?;
            let expected = std::fs::read(path.join(if use_bias {
                "expected_bias.bf16"
            } else {
                "expected.bf16"
            }))?;
            ensure!(bytes.len() == expected.len(), "oracle length mismatch");
            let mut max_abs = 0_f32;
            let mut mean_abs = 0_f64;
            let mut different = 0usize;
            for (a, b) in bytes.chunks_exact(2).zip(expected.chunks_exact(2)) {
                let actual = half::bf16::from_le_bytes([a[0], a[1]]).to_f32();
                let wanted = half::bf16::from_le_bytes([b[0], b[1]]).to_f32();
                let error = (actual - wanted).abs();
                ensure!(
                    error <= 0.002 + 0.008 * wanted.abs(),
                    "{} bias={use_bias}: mismatch {actual} vs {wanted}",
                    c.name
                );
                max_abs = max_abs.max(error);
                mean_abs += f64::from(error);
                different += usize::from(a != b);
            }
            let times = benchmark(&gpu, stream, &run)?;
            records.push(serde_json::json!({"name":c.name,"shape":[c.m,c.n,c.k],"bias":use_bias,"mode":"nvfp4","different":different,"max_abs":max_abs,"mean_abs":mean_abs/(c.m*c.n) as f64,"mean_ms":times.iter().sum::<f32>()/times.len() as f32,"samples_ms":times}));
        }
        if c.name == "ragged_small" {
            gpu.begin_capture(stream)?;
            ops.linear_nvfp4(
                a,
                w,
                ws,
                None,
                out,
                c.m as usize,
                c.n as usize,
                c.k as usize,
                c.activation_global_scale,
                c.weight_global_scale,
                &scratch,
            )?;
            let graph = gpu.end_capture(stream)?;
            gpu.copy_h2d(&vec![0; c.m as usize * c.k as usize * 2], a)?;
            gpu.launch_graph(graph, stream)?;
            let mut zero_output = vec![0; c.m as usize * c.n as usize * 2];
            gpu.copy_d2h_on_stream(out, &mut zero_output, stream)?;
            ensure!(
                zero_output.iter().all(|&b| b == 0),
                "graph reused stale activations"
            );
            gpu.destroy_graph(graph)?;
            gpu.copy_h2d(&std::fs::read(path.join("input.bf16"))?, a)?;
        }
        let bf16_weight = upload(&gpu, &path.join("weight.bf16"))?;
        let times = benchmark(&gpu, stream, &|| {
            ops.linear_bf16(
                a,
                bf16_weight,
                None,
                out,
                c.m as usize,
                c.n as usize,
                c.k as usize,
            )
        })?;
        records.push(serde_json::json!({"name":c.name,"shape":[c.m,c.n,c.k],"mode":"bf16","bias":false,"mean_ms":times.iter().sum::<f32>()/times.len() as f32,"samples_ms":times}));
        gpu.free(bf16_weight)?;
        for ptr in [a, w, ws, scratch.packed, scratch.scales, bias, out] {
            gpu.free(ptr)?;
        }
    }
    std::fs::write(&args[2], serde_json::to_vec_pretty(&records)?)?;
    Ok(())
}

fn benchmark(gpu: &dyn GpuBackend, stream: u64, run: &impl Fn() -> Result<()>) -> Result<Vec<f32>> {
    for _ in 0..10 {
        run()?;
    }
    gpu.begin_capture(stream)?;
    run()?;
    let graph = gpu.end_capture(stream)?;
    let mut times = Vec::new();
    // Direct event FFI is confined to this diagnostic executable.
    unsafe {
        let (mut start, mut end) = (std::ptr::null_mut(), std::ptr::null_mut());
        ensure!(
            cuEventCreate(&mut start, 0) == 0 && cuEventCreate(&mut end, 0) == 0,
            "create events"
        );
        for _ in 0..5 {
            ensure!(
                cuEventRecord(start, stream as *mut c_void) == 0,
                "record start"
            );
            for _ in 0..100 {
                gpu.launch_graph(graph, stream)?;
            }
            ensure!(
                cuEventRecord(end, stream as *mut c_void) == 0 && cuEventSynchronize(end) == 0,
                "record/sync end"
            );
            let mut ms = 0_f32;
            ensure!(cuEventElapsedTime(&mut ms, start, end) == 0, "event time");
            times.push(ms / 100.0);
        }
        cuEventDestroy_v2(start);
        cuEventDestroy_v2(end);
    }
    gpu.destroy_graph(graph)?;

    Ok(times)
}
