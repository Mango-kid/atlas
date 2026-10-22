// SPDX-License-Identifier: AGPL-3.0-only
//! Warm graph replay enclosed by the CUDA profiler API for Nsight Systems.
use anyhow::{Result, ensure};
use spark_pi05::{NativePolicy, preprocess::PolicyRequest};
use std::path::Path;
mod support;

unsafe extern "C" {
    fn cuProfilerStart() -> i32;
    fn cuProfilerStop() -> i32;
}

fn main() -> Result<()> {
    let mut args: Vec<_> = std::env::args().collect();
    let manifests = support::take_quantization_manifests(&mut args)?;
    ensure!(
        args.len() == 6,
        "usage: policy_profile WEIGHTS PROFILE TOKENIZER NORMS REQUEST [--nvfp4-manifest PATH [--fp8-manifest PATH]]"
    );
    let request: PolicyRequest = serde_json::from_slice(&std::fs::read(&args[5])?)?;
    let mut policy = NativePolicy::load_with_quantization(
        Path::new(&args[1]),
        Path::new(&args[2]),
        Path::new(&args[3]),
        Path::new(&args[4]),
        0,
        manifests.nvfp4.as_deref(),
        manifests.fp8.as_deref(),
    )?;
    for _ in 0..5 {
        policy.infer(&request)?;
    }
    unsafe {
        ensure!(cuProfilerStart() == 0, "start CUDA profiler");
    }
    for _ in 0..3 {
        let result = policy.infer(&request)?;
        eprintln!("{} {:.3} ms", result.backend, result.timing_ms);
    }
    unsafe {
        ensure!(cuProfilerStop() == 0, "stop CUDA profiler");
    }
    Ok(())
}
