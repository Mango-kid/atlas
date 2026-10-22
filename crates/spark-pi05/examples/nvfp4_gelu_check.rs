// SPDX-License-Identifier: AGPL-3.0-only
//! Byte-exact gated-GELU/NVFP4 fusion regression and graph-replay timing.
use anyhow::{Result, ensure};
use spark_pi05::ops::{Nvfp4Scratch, Ops};
use spark_runtime::{cuda_backend::AtlasCudaBackend, gpu::GpuBackend};
use std::ffi::c_void;

unsafe extern "C" {
    fn cuEventCreate(event: *mut *mut c_void, flags: u32) -> i32;
    fn cuEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cuEventSynchronize(event: *mut c_void) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> i32;
    fn cuEventDestroy_v2(event: *mut c_void) -> i32;
}

fn benchmark(gpu: &dyn GpuBackend, run: &impl Fn() -> Result<()>) -> Result<f32> {
    let stream = gpu.default_stream();
    for _ in 0..3 {
        run()?;
    }
    gpu.begin_capture(stream)?;
    run()?;
    let graph = gpu.end_capture(stream)?;
    let mut elapsed = 0.0;
    unsafe {
        let (mut start, mut end) = (std::ptr::null_mut(), std::ptr::null_mut());
        ensure!(
            cuEventCreate(&mut start, 0) == 0 && cuEventCreate(&mut end, 0) == 0,
            "events"
        );
        ensure!(cuEventRecord(start, stream as *mut c_void) == 0, "record");
        for _ in 0..100 {
            gpu.launch_graph(graph, stream)?;
        }
        ensure!(
            cuEventRecord(end, stream as *mut c_void) == 0 && cuEventSynchronize(end) == 0,
            "sync"
        );
        ensure!(cuEventElapsedTime(&mut elapsed, start, end) == 0, "time");
        cuEventDestroy_v2(start);
        cuEventDestroy_v2(end);
    }
    gpu.destroy_graph(graph)?;
    Ok(elapsed / 100.0)
}

fn main() -> Result<()> {
    let output = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: nvfp4_gelu_check OUTPUT_JSON"))?;
    let gpu = AtlasCudaBackend::new(
        0,
        &[(
            "pi05",
            include_bytes!(concat!(env!("OUT_DIR"), "/pi05.ptx")),
        )],
    )?;
    let stream = gpu.default_stream();
    let ops = Ops::new(&gpu, stream)?;
    let mut records = Vec::new();
    for (m, k) in [
        (1, 64),
        (10, 4096),
        (128, 2048),
        (512, 16384),
        (518, 16384),
        (712, 16384),
    ] {
        let count = m * k;
        let mut rng = 42_u64;
        let mut gates = Vec::with_capacity(count * 2);
        let mut ups = Vec::with_capacity(count * 2);
        for i in 0..count {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let gate = match i % 31 {
                0 => -0.0,
                1 => 0.0,
                2 => 1.0e-30,
                3 => -1.0e-30,
                4 => -16.0,
                5 => 16.0,
                _ => ((rng & 0xffff) as f32 / 65536.0 - 0.5) * 16.0,
            };
            let up = if i % 13 == 0 {
                0.0
            } else {
                ((rng >> 32) as u32 as f32 / u32::MAX as f32 - 0.5) * 8.0
            };
            gates.extend_from_slice(&half::bf16::from_f32(gate).to_le_bytes());
            ups.extend_from_slice(&half::bf16::from_f32(up).to_le_bytes());
        }
        let gate = gpu.alloc(gates.len())?;
        let up = gpu.alloc(ups.len())?;
        let intermediate = gpu.alloc(count * 2)?;
        gpu.copy_h2d(&gates, gate)?;
        gpu.copy_h2d(&ups, up)?;
        let make_scratch = || -> Result<Nvfp4Scratch> {
            Ok(Nvfp4Scratch {
                packed: gpu.alloc(Nvfp4Scratch::packed_bytes(m, k))?,
                scales: gpu.alloc(Nvfp4Scratch::scale_bytes(m, k))?,
                max_rows: m,
                max_k: k,
            })
        };
        let original = make_scratch()?;
        let fused = make_scratch()?;
        for scale in [0.00001, 0.003125, 0.1] {
            let separate = || -> Result<()> {
                ops.gelu_mul(gate, up, intermediate, count)?;
                ops.quantize_nvfp4(intermediate, m, k, scale, &original)
            };
            let combined = || ops.gelu_mul_quantize_nvfp4(gate, up, m, k, scale, &fused);
            // Poison all scale padding before each path to test complete writes.
            let size = Nvfp4Scratch::scale_bytes(m, k);
            gpu.copy_h2d(&vec![0x77; size], original.scales)?;
            gpu.copy_h2d(&vec![0x55; size], fused.scales)?;
            separate()?;
            combined()?;
            for (a, b, size) in [
                (
                    original.packed,
                    fused.packed,
                    Nvfp4Scratch::packed_bytes(m, k),
                ),
                (original.scales, fused.scales, size),
            ] {
                let mut baseline = vec![0; size];
                let mut candidate = vec![0; size];
                gpu.copy_d2h_on_stream(a, &mut baseline, stream)?;
                gpu.copy_d2h_on_stream(b, &mut candidate, stream)?;
                let mismatch = baseline
                    .iter()
                    .zip(&candidate)
                    .filter(|(x, y)| x != y)
                    .count();
                ensure!(
                    mismatch == 0,
                    "{m}x{k} scale{scale}: {mismatch} mismatched bytes"
                );
            }
            let separate_ms = benchmark(&gpu, &separate)?;
            let fused_ms = benchmark(&gpu, &combined)?;
            records.push(serde_json::json!({"m":m,"k":k,"scale":scale,"packed_and_scales_bitexact":true,"separate_ms":separate_ms,"fused_ms":fused_ms}));
        }
        for ptr in [
            gate,
            up,
            intermediate,
            original.packed,
            original.scales,
            fused.packed,
            fused.scales,
        ] {
            gpu.free(ptr)?;
        }
    }
    std::fs::write(output, serde_json::to_vec_pretty(&records)?)?;
    Ok(())
}
