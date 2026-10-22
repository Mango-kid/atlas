// SPDX-License-Identifier: AGPL-3.0-only
//! Bitwise cached-KV oracle with poisoned padding and changing graph inputs.
use anyhow::{Result, ensure};
use spark_runtime::{
    cuda_backend::AtlasCudaBackend,
    gpu::{DevicePtr, GpuBackend},
};

const DIM: usize = 256;
const CAPACITY: usize = 736;
const GUARD: usize = 16;

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
    let mut bytes = values(keys * DIM, seed);
    // Both padding and guard contain NaNs: a missed clear or excess clear fails.
    bytes.extend((keys * DIM..(CAPACITY + GUARD) * DIM).flat_map(|_| [0xc1, 0x7f]));
    bytes
}

fn read(gpu: &dyn GpuBackend, ptr: DevicePtr, count: usize) -> Result<Vec<u8>> {
    let mut bytes = vec![0; count];
    gpu.copy_d2h_on_stream(ptr, &mut bytes, gpu.default_stream())?;
    Ok(bytes)
}

fn check_cache(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    mut original: Vec<u8>,
    keys: usize,
) -> Result<()> {
    let padded = keys.div_ceil(16) * 16;
    original[keys * DIM * 2..padded * DIM * 2].fill(0);
    ensure!(
        read(gpu, ptr, original.len())? == original,
        "cached attention modified live rows/guard or left nonzero padding: keys={keys}"
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 2, "usage: cached_attention_check OUTPUT_JSON");
    let ptx = include_bytes!(concat!(env!("OUT_DIR"), "/pi05.ptx"));
    let gpu = AtlasCudaBackend::new(0, &[("pi05", ptx)])?;
    let stream = gpu.default_stream();
    let ops = spark_pi05::ops::Ops::new(&gpu, stream)?;
    let q = gpu.alloc(712 * 8 * DIM * 2)?;
    let k = gpu.alloc((CAPACITY + GUARD) * DIM * 2)?;
    let v = gpu.alloc((CAPACITY + GUARD) * DIM * 2)?;
    let expected = gpu.alloc(712 * 8 * DIM * 2)?;
    let actual = gpu.alloc(712 * 8 * DIM * 2)?;
    let scratch = gpu.alloc(spark_pi05::ops_attention::mqa_scratch_bytes(
        712, 8, CAPACITY, DIM,
    ))?;
    let mut records = Vec::new();
    // Deliberately reuse one cache in decreasing/increasing order, including
    // aligned lengths, maximum length and every possible tail width.
    let mut cases = vec![(10, 722), (10, 513), (712, 712), (513, 513), (10, 528)];
    cases.extend((513..=528).map(|keys| (10, keys)));
    for (qrows, keys) in cases {
        let count = qrows * 8 * DIM;
        gpu.copy_h2d(&values(count, 41), q)?;
        gpu.copy_h2d(&cache_input(keys, 42), k)?;
        gpu.copy_h2d(&cache_input(keys, 43), v)?;
        let old = || ops.attention_mqa(q, k, v, expected, qrows, keys, 8, DIM, scratch);
        let new = || ops.attention_mqa_cached(q, k, v, actual, qrows, keys, CAPACITY, scratch);
        old()?;
        new()?;
        gpu.synchronize(stream)?;
        ensure!(
            read(&gpu, expected, count * 2)? == read(&gpu, actual, count * 2)?,
            "eager cached output differs: qrows={qrows} keys={keys}"
        );
        gpu.begin_capture(stream)?;
        new()?;
        let graph = gpu.end_capture(stream)?;
        for replay in 0..3 {
            let key_bytes = cache_input(keys, 100 + replay);
            let value_bytes = cache_input(keys, 200 + replay);
            gpu.copy_h2d(&values(count, 300 + replay), q)?;
            gpu.copy_h2d(&key_bytes, k)?;
            gpu.copy_h2d(&value_bytes, v)?;
            old()?;
            gpu.launch_graph(graph, stream)?;
            ensure!(
                read(&gpu, expected, count * 2)? == read(&gpu, actual, count * 2)?,
                "graph output differs: qrows={qrows} keys={keys} replay={replay}"
            );
            check_cache(&gpu, k, key_bytes, keys)?;
            check_cache(&gpu, v, value_bytes, keys)?;
        }
        gpu.destroy_graph(graph)?;
        records.push(
            serde_json::json!({"qrows":qrows,"keys":keys,"eager_bitexact":true,
            "graph_replays_bitexact":3,"live_cache_and_guard_unchanged":true}),
        );
    }
    for ptr in [q, k, v, expected, actual, scratch] {
        gpu.free(ptr)?;
    }
    std::fs::write(&args[1], serde_json::to_vec_pretty(&records)?)?;
    Ok(())
}
