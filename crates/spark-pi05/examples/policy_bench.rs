// SPDX-License-Identifier: AGPL-3.0-only

//! Synchronized native policy benchmark. Inputs and initial noise are explicit.
use anyhow::{Context, Result, ensure};
use spark_pi05::{NativePolicy, preprocess::PolicyRequest};
use std::{path::Path, time::Instant};
mod support;

fn main() -> Result<()> {
    let mut args: Vec<_> = std::env::args().collect();
    let manifests = support::take_quantization_manifests(&mut args)?;
    ensure!(
        args.len() >= 9,
        "usage: policy_bench WEIGHTS PROFILE TOKENIZER NORMS WARMUP RUNS OUTPUT REQUEST... [--nvfp4-manifest PATH [--fp8-manifest PATH]]"
    );
    let warmup: usize = args[5].parse()?;
    let runs: usize = args[6].parse()?;
    ensure!(warmup > 0 && runs > 0, "warmup and runs must be positive");
    let requests: Vec<PolicyRequest> = args[8..]
        .iter()
        .map(|p| {
            serde_json::from_slice(&std::fs::read(p).with_context(|| format!("read {p}"))?)
                .context("parse observation")
        })
        .collect::<Result<_>>()?;
    ensure!(runs >= requests.len(), "runs must cover every request");
    let load_start = Instant::now();
    let mut policy = NativePolicy::load_with_quantization(
        Path::new(&args[1]),
        Path::new(&args[2]),
        Path::new(&args[3]),
        Path::new(&args[4]),
        0,
        manifests.nvfp4.as_deref(),
        manifests.fp8.as_deref(),
    )?;
    let load_s = load_start.elapsed().as_secs_f64();
    eprintln!("native model loaded in {load_s:.2}s");
    for i in 0..warmup {
        policy.infer(&requests[i % requests.len()])?;
    }
    let mut elapsed = Vec::with_capacity(runs);
    for i in 0..runs {
        let start = Instant::now();
        policy.infer(&requests[i % requests.len()])?;
        elapsed.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let mut sorted = elapsed.clone();
    sorted.sort_by(f64::total_cmp);
    let percentile = |p: f64| sorted[((sorted.len() - 1) as f64 * p).round() as usize];
    let by_case: Vec<_> = args[8..]
        .iter()
        .enumerate()
        .map(|(index, path)| {
            let mut samples: Vec<_> = elapsed
                .iter()
                .enumerate()
                .filter_map(|(i, &ms)| (i % requests.len() == index).then_some(ms))
                .collect();
            samples.sort_by(f64::total_cmp);
            serde_json::json!({"request":path,"runs":samples.len(),
            "mean_ms": samples.iter().sum::<f64>() / samples.len() as f64,
            "p50_ms": samples[(samples.len()-1)/2],
            "p95_ms": samples[((samples.len()-1) as f64 * 0.95).round() as usize]})
        })
        .collect();
    let outputs = requests
        .iter()
        .map(|r| policy.infer(r))
        .collect::<Result<Vec<_>>>()?;
    let result = serde_json::json!({
        "backend":policy.backend(), "nvfp4_manifest":manifests.nvfp4, "fp8_manifest":manifests.fp8,
        "load_s":load_s,"warmup":warmup,"runs":runs,
        "mean_ms":elapsed.iter().sum::<f64>()/runs as f64,"p50_ms":percentile(0.5),"p95_ms":percentile(0.95),
        "min_ms":sorted[0],"max_ms":sorted[sorted.len()-1],"samples_ms":elapsed,
        "requests":args[8..],"outputs":outputs,"by_case":by_case,
        "measurement":"synchronized preprocessing + native GPU inference + output transform; loading/warmup excluded"
    });
    std::fs::write(&args[7], serde_json::to_vec_pretty(&result)?)?;
    eprintln!(
        "mean {:.3} ms, p95 {:.3} ms",
        result["mean_ms"], result["p95_ms"]
    );
    Ok(())
}
