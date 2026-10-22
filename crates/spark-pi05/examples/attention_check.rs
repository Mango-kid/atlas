// SPDX-License-Identifier: AGPL-3.0-only
//! Compare flattened MQA against the validated eager WMMA attention path.
use anyhow::{Result, ensure};
use spark_runtime::{
    cuda_backend::AtlasCudaBackend,
    gpu::{DevicePtr, GpuBackend},
};
use std::ffi::c_void;
unsafe extern "C" {
    fn cuEventCreate(event: *mut *mut c_void, flags: u32) -> i32;
    fn cuEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cuEventSynchronize(event: *mut c_void) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
    fn cuEventDestroy_v2(event: *mut c_void) -> i32;
}
fn upload(gpu: &dyn GpuBackend, n: usize, seed: u64, scale: f32) -> Result<DevicePtr> {
    let mut state = seed;
    let values: Vec<half::bf16> = (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            half::bf16::from_f32(((state >> 40) as f32 / 8388608.0 - 1.0) * scale)
        })
        .collect();
    let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, ptr)?;
    Ok(ptr)
}
fn read(gpu: &dyn GpuBackend, ptr: DevicePtr, count: usize, stream: u64) -> Result<Vec<f32>> {
    let mut bytes = vec![0; count * 2];
    gpu.copy_d2h_on_stream(ptr, &mut bytes, stream)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|x| half::bf16::from_le_bytes([x[0], x[1]]).to_f32())
        .collect())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 2, "usage: attention_check OUTPUT_JSON");
    let ptx = include_bytes!(concat!(env!("OUT_DIR"), "/pi05.ptx"));
    let gpu = AtlasCudaBackend::new(0, &[("pi05", ptx)])?;
    let stream = gpu.default_stream();
    let ops = spark_pi05::ops::Ops::new(&gpu, stream)?;
    let mut records = Vec::new();
    for (qrows, keys) in [
        (10, 16),
        (10, 513),
        (10, 526),
        (10, 528),
        (10, 722),
        (10, 1024),
        (518, 518),
        (712, 712),
    ] {
        let rows = qrows * 8;
        for scale in [0.0, 1.0, 4.0] {
            let q = upload(&gpu, rows * 256, 42, scale)?;
            let k = upload(&gpu, keys * 256, 43, scale)?;
            let v = upload(&gpu, keys * 256, 44, 1.0)?;
            let old = gpu.alloc(rows * 256 * 2)?;
            let out = gpu.alloc(rows * 256 * 2)?;
            let scratch = gpu.alloc(spark_pi05::ops_attention::mqa_scratch_bytes(
                qrows, 8, keys, 256,
            ))?;
            let old_run = || {
                ops.attention_wmma(
                    q,
                    k,
                    v,
                    old,
                    qrows,
                    keys,
                    8,
                    1,
                    256,
                    DevicePtr::NULL,
                    scratch,
                )
            };
            let new_run = || {
                ops.attention(
                    q,
                    k,
                    v,
                    out,
                    qrows,
                    keys,
                    8,
                    1,
                    256,
                    DevicePtr::NULL,
                    scratch,
                )
            };
            old_run()?;
            let expected = read(&gpu, old, rows * 256, stream)?;
            new_run()?;
            let actual = read(&gpu, out, rows * 256, stream)?;
            let mut max_abs = 0_f32;
            let mut sum_sq = 0_f64;
            let mut dot = 0_f64;
            let mut norm_a = 0_f64;
            let mut norm_b = 0_f64;
            for (&a, &b) in actual.iter().zip(&expected) {
                ensure!(a.is_finite(), "nonfinite output");
                let err = (a - b).abs();
                max_abs = max_abs.max(err);
                sum_sq += f64::from(err).powi(2);
                dot += f64::from(a) * f64::from(b);
                norm_a += f64::from(a).powi(2);
                norm_b += f64::from(b).powi(2);
            }
            let rmse = (sum_sq / actual.len() as f64).sqrt();
            let cosine = dot / (norm_a * norm_b).sqrt();
            ensure!(
                cosine > 0.99999 && rmse < 0.001,
                "MQA mismatch keys={keys} scale={scale}: cosine={cosine} rmse={rmse} max={max_abs}"
            );
            let old_ms = benchmark(&gpu, stream, &old_run)?;
            let new_ms = benchmark(&gpu, stream, &new_run)?;
            records.push(serde_json::json!({"qrows":qrows,"keys":keys,"scale":scale,"max_abs":max_abs,"rmse":rmse,"cosine":cosine,"old_ms":old_ms,"new_ms":new_ms}));
            for ptr in [q, k, v, old, out, scratch] {
                gpu.free(ptr)?;
            }
        }
    }
    std::fs::write(&args[1], serde_json::to_vec_pretty(&records)?)?;
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
