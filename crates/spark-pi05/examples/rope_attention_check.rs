// SPDX-License-Identifier: AGPL-3.0-only
//! Exact fused RoPE/tail-clear oracle with poisoned padding and graph replay.
use anyhow::{Result, ensure};
use spark_runtime::{
    cuda_backend::AtlasCudaBackend,
    gpu::{DevicePtr, GpuBackend},
};
use std::ffi::c_void;
const DIM: usize = 256;
const CAPACITY: usize = 736;
const GUARD: usize = 16;
unsafe extern "C" {
    fn cuEventCreate(event: *mut *mut c_void, flags: u32) -> i32;
    fn cuEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cuEventSynchronize(event: *mut c_void) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
    fn cuEventDestroy_v2(event: *mut c_void) -> i32;
}
fn values(count: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..count)
        .flat_map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            half::bf16::from_f32((state >> 40) as f32 / 8388608.0 - 1.0).to_le_bytes()
        })
        .collect()
}
fn cache_input(keys: usize, seed: u64) -> Vec<u8> {
    let mut data = values(keys * DIM, seed);
    data.extend((keys * DIM..(CAPACITY + GUARD) * DIM).flat_map(|_| [0xc1, 0x7f]));
    data
}
fn read(gpu: &dyn GpuBackend, ptr: DevicePtr, bytes: usize, stream: u64) -> Result<Vec<u8>> {
    let mut out = vec![0; bytes];
    gpu.copy_d2h_on_stream(ptr, &mut out, stream)?;
    Ok(out)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 2, "usage: rope_attention_check OUTPUT_JSON");
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
    let q = gpu.alloc(712 * 8 * DIM * 2)?;
    let nq = gpu.alloc(712 * 8 * DIM * 2)?;
    let k = gpu.alloc((CAPACITY + GUARD) * DIM * 2)?;
    let nk = gpu.alloc((CAPACITY + GUARD) * DIM * 2)?;
    let v = gpu.alloc((CAPACITY + GUARD) * DIM * 2)?;
    let nv = gpu.alloc((CAPACITY + GUARD) * DIM * 2)?;
    let old_out = gpu.alloc(712 * 8 * DIM * 2)?;
    let new_out = gpu.alloc(712 * 8 * DIM * 2)?;
    let scratch = gpu.alloc(spark_pi05::ops_attention::mqa_scratch_bytes(
        712, 8, CAPACITY, DIM,
    ))?;
    let pos = gpu.alloc(722 * 4)?;
    let mut records = Vec::new();
    let mut cases = vec![
        (10, 722),
        (10, 523),
        (712, 712),
        (533, 533),
        (10, 528),
        (1, 722),
    ];
    cases.extend((523..=538).map(|keys| (10, keys)));
    for (rows, keys) in cases {
        let prefix = keys - rows;
        let count = rows * 8 * DIM;
        let positions: Vec<i32> = (prefix as i32..keys as i32).collect();
        gpu.copy_h2d(bytemuck::cast_slice(&positions), pos)?;
        let old = || {
            ops.rope_pair(q, k.offset(prefix * DIM * 2), pos, rows, 8, DIM)?;
            ops.attention_mqa_cached(q, k, v, old_out, rows, keys, CAPACITY, scratch)
        };
        let new = || {
            ops.rope_attention_mqa_cached(nq, nk, nv, pos, new_out, rows, prefix, CAPACITY, scratch)
        };
        let inputs = |seed| -> Result<()> {
            for (a, b, bytes) in [
                (q, nq, values(count, seed)),
                (k, nk, cache_input(keys, seed + 1)),
                (v, nv, cache_input(keys, seed + 2)),
            ] {
                gpu.copy_h2d(&bytes, a)?;
                gpu.copy_h2d(&bytes, b)?;
            }
            Ok(())
        };
        let compare = || -> Result<()> {
            for (a, b, bytes, label) in [
                (old_out, new_out, count * 2, "attention"),
                (q, nq, count * 2, "query"),
                (k, nk, (CAPACITY + GUARD) * DIM * 2, "key+guard"),
                (v, nv, (CAPACITY + GUARD) * DIM * 2, "value+guard"),
            ] {
                ensure!(
                    read(&gpu, a, bytes, stream)? == read(&gpu, b, bytes, stream)?,
                    "fused {label} mismatch rows={rows} keys={keys}"
                );
            }
            Ok(())
        };
        inputs(42)?;
        old()?;
        new()?;
        compare()?;
        gpu.begin_capture(stream)?;
        new()?;
        let graph = gpu.end_capture(stream)?;
        for replay in 0..3 {
            inputs(100 + replay)?;
            old()?;
            gpu.launch_graph(graph, stream)?;
            compare()?;
        }
        gpu.destroy_graph(graph)?;
        let old_ms = benchmark(&gpu, stream, &old)?;
        let new_ms = benchmark(&gpu, stream, &new)?;
        records.push(serde_json::json!({"rows":rows,"keys":keys,"bitexact":true,"cache_and_guard_exact":true,"changed_input_graph_replays":3,"old_ms":old_ms,"new_ms":new_ms}));
    }
    for p in [q, nq, k, nk, v, nv, old_out, new_out, scratch, pos, table] {
        gpu.free(p)?;
    }
    std::fs::write(&args[1], serde_json::to_vec_pretty(&records)?)?;
    Ok(())
}
fn benchmark(gpu: &dyn GpuBackend, stream: u64, run: &impl Fn() -> Result<()>) -> Result<Vec<f32>> {
    for _ in 0..10 {
        run()?;
    }
    gpu.begin_capture(stream)?;
    for _ in 0..32 {
        run()?;
    }
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
            times.push(ms / 3200.0);
        }
        cuEventDestroy_v2(start);
        cuEventDestroy_v2(end);
    }
    gpu.destroy_graph(graph)?;

    Ok(times)
}
