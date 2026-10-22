// SPDX-License-Identifier: AGPL-3.0-only

use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let source = "../../kernels/gb10/pi05/bf16/pi05.cu";
    println!("cargo:rerun-if-changed={source}");
    println!("cargo:rerun-if-changed=../../kernels/gb10/pi05/bf16/nvfp4.cuh");
    println!("cargo:rerun-if-changed=../../kernels/gb10/pi05/bf16/nvfp4_gelu.cuh");
    println!("cargo:rerun-if-changed=../../kernels/gb10/pi05/bf16/fp8.cuh");
    println!("cargo:rerun-if-changed=../../kernels/gb10/pi05/bf16/fp8_fused.cuh");
    println!("cargo:rerun-if-changed=../../kernels/gb10/pi05/bf16/attention_mqa.cuh");
    println!("cargo:rerun-if-changed=../../kernels/gb10/pi05/bf16/attention_vision.cuh");
    println!("cargo:rerun-if-changed=../../kernels/gb10/pi05/bf16/pi05_cached_kv.cuh");
    println!("cargo:rerun-if-changed=../../kernels/gb10/pi05/bf16/rope.cuh");
    for name in ["ATLAS_SKIP_BUILD", "ATLAS_PI05_BUILD", "CUDA_HOME"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo OUT_DIR")).join("pi05.ptx");
    let skip = matches!(
        env::var("ATLAS_SKIP_BUILD").as_deref(),
        Ok("1") | Ok("true")
    );
    // Enables the isolated policy build without compiling unrelated chat targets.
    let policy_only = env::var("ATLAS_PI05_BUILD").as_deref() == Ok("1");
    if skip && !policy_only {
        fs::write(output, []).expect("write no-GPU stub");
        return;
    }
    let cuda = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let status = Command::new(format!("{cuda}/bin/nvcc"))
        .args([
            "--ptx",
            "-O3",
            "-arch=sm_121",
            "--fmad=false",
            "-std=c++17",
            source,
            "-o",
        ])
        .arg(output)
        .status()
        .expect("run nvcc for native pi0.5");
    assert!(status.success(), "pi0.5 CUDA compilation failed");
}
