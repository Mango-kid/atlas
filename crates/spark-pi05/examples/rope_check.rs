// SPDX-License-Identifier: AGPL-3.0-only
//! Bit-exact cached/fused RoPE oracle over every supported position.
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
fn input(count: usize, seed: u64, scale: f32) -> Vec<u8> {
    let mut state = seed;
    (0..count)
        .flat_map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let value = match i % 257 {
                0 => 0.0,
                1 => -0.0,
                _ => ((state >> 40) as f32 / 8388608.0 - 1.0) * scale,
            };
            half::bf16::from_f32(value).to_le_bytes()
        })
        .collect()
}
fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}
fn read(gpu: &dyn GpuBackend, ptr: DevicePtr, bytes: usize, stream: u64) -> Result<Vec<u8>> {
    let mut out = vec![0; bytes];
    gpu.copy_d2h_on_stream(ptr, &mut out, stream)?;
    Ok(out)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 2, "usage: rope_check OUTPUT_JSON");
    let gpu = AtlasCudaBackend::new(
        0,
        &[(
            "pi05",
            include_bytes!(concat!(env!("OUT_DIR"), "/pi05.ptx")),
        )],
    )?;
    let stream = gpu.default_stream();
    let mut ops = spark_pi05::ops::Ops::new(&gpu, stream)?;
    let table = gpu.alloc(722 * 256 * 2)?;
    ops.prepare_rope_table(table, 722, 256)?;
    ops.rope_table = Some(table);
    let mut results = Vec::new();
    for (rows, start, reverse) in [
        (722, 0, false),
        (722, 0, true),
        (533, 0, false),
        (712, 0, false),
        (10, 512, false),
        (10, 712, false),
    ] {
        let positions: Vec<i32> = (start..start + rows)
            .map(|p| if reverse { 721 - p as i32 } else { p as i32 })
            .collect();
        let pos = upload(&gpu, bytemuck::cast_slice(&positions))?;
        for seed in [1, 42, 123] {
            for scale in [0.0, 1e-39, 1.0, 1e30] {
                let qb = input(rows * 8 * 256, seed, scale);
                let kb = input(rows * 256, seed + 1, scale);
                let q = upload(&gpu, &qb)?;
                let k = upload(&gpu, &kb)?;
                let nq = upload(&gpu, &qb)?;
                let nk = upload(&gpu, &kb)?;
                let old_run = || {
                    ops.rope(q, pos, rows, 8, 256)?;
                    ops.rope(k, pos, rows, 1, 256)
                };
                let new_run = || ops.rope_pair(nq, nk, pos, rows, 8, 256);
                old_run()?;
                let old_q = read(&gpu, q, qb.len(), stream)?;
                let old_k = read(&gpu, k, kb.len(), stream)?;
                new_run()?;
                ensure!(
                    read(&gpu, nq, qb.len(), stream)? == old_q
                        && read(&gpu, nk, kb.len(), stream)? == old_k,
                    "fused RoPE differs rows={rows} start={start} seed={seed} scale={scale}"
                );
                gpu.copy_h2d(&qb, nq)?;
                gpu.copy_h2d(&kb, nk)?;
                ops.rope_cached(nq, None, pos, table, rows, 8, 256)?;
                ops.rope_cached(nk, None, pos, table, rows, 1, 256)?;
                ensure!(
                    read(&gpu, nq, qb.len(), stream)? == old_q
                        && read(&gpu, nk, kb.len(), stream)? == old_k,
                    "cached single RoPE differs"
                );
                let (old_ms, new_ms) = if scale == 1.0 {
                    (
                        benchmark(&gpu, stream, &old_run)?,
                        benchmark(&gpu, stream, &new_run)?,
                    )
                } else {
                    (vec![], vec![])
                };
                gpu.begin_capture(stream)?;
                new_run()?;
                let graph = gpu.end_capture(stream)?;
                let changed_q = input(rows * 8 * 256, seed + 100, scale);
                let changed_k = input(rows * 256, seed + 101, scale);
                gpu.copy_h2d(&changed_q, q)?;
                gpu.copy_h2d(&changed_k, k)?;
                gpu.copy_h2d(&changed_q, nq)?;
                gpu.copy_h2d(&changed_k, nk)?;
                old_run()?;
                gpu.launch_graph(graph, stream)?;
                ensure!(
                    read(&gpu, nq, qb.len(), stream)? == read(&gpu, q, qb.len(), stream)?
                        && read(&gpu, nk, kb.len(), stream)? == read(&gpu, k, kb.len(), stream)?,
                    "changed-input graph differs"
                );
                gpu.destroy_graph(graph)?;
                results.push(serde_json::json!({"rows":rows,"start":start,"reverse":reverse,"seed":seed,"scale":scale,"bitexact":true,"changed_input_graph_exact":true,"old_ms":old_ms,"new_ms":new_ms}));
                for ptr in [q, k, nq, nk] {
                    gpu.free(ptr)?;
                }
            }
        }
        gpu.free(pos)?;
    }
    gpu.free(table)?;
    std::fs::write(&args[1], serde_json::to_vec_pretty(&results)?)?;
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
