// SPDX-License-Identifier: AGPL-3.0-only
//! Capture native intermediates for reference comparison, never benchmarking.

use anyhow::{Context, Result, ensure};
use serde::Serialize;
use spark_pi05::{NativePolicy, model::TraceSink, preprocess::PolicyRequest};
use std::{
    collections::BTreeMap,
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
};
mod support;

#[derive(Serialize)]
struct TensorMetadata {
    file: String,
    shape: Vec<usize>,
    dtype: &'static str,
}

struct FileTrace {
    directory: PathBuf,
    tensors: BTreeMap<String, TensorMetadata>,
}

impl TraceSink for FileTrace {
    fn tensor(&mut self, name: &str, shape: &[usize], values: &[f32]) -> Result<()> {
        ensure!(
            name.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_'),
            "invalid trace name"
        );
        ensure!(
            !self.tensors.contains_key(name),
            "duplicate trace tensor {name}"
        );
        let file = format!("{name}.f32");
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.directory.join(&file))
            .with_context(|| format!("create trace {file}"))?;
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        output.write_all(&bytes)?;
        self.tensors.insert(
            name.into(),
            TensorMetadata {
                file,
                shape: shape.to_vec(),
                dtype: "float32-le",
            },
        );
        Ok(())
    }
}

fn main() -> Result<()> {
    let mut args: Vec<_> = std::env::args().collect();
    let manifests = support::take_quantization_manifests(&mut args)?;
    ensure!(
        args.len() == 7,
        "usage: policy_trace WEIGHTS PROFILE TOKENIZER NORMS REQUEST OUTPUT_DIRECTORY [--nvfp4-manifest PATH [--fp8-manifest PATH]]"
    );
    let request: PolicyRequest = serde_json::from_slice(&std::fs::read(&args[5])?)?;
    request.validate()?;
    // A fresh directory prevents accidentally comparing stale or mixed runs.
    std::fs::create_dir(&args[6]).context("create new trace output directory")?;
    let mut trace = FileTrace {
        directory: PathBuf::from(&args[6]),
        tensors: BTreeMap::new(),
    };
    let mut policy = NativePolicy::load_with_quantization(
        Path::new(&args[1]),
        Path::new(&args[2]),
        Path::new(&args[3]),
        Path::new(&args[4]),
        0,
        manifests.nvfp4.as_deref(),
        manifests.fp8.as_deref(),
    )?;
    let response = policy.infer_traced(&request, &mut trace)?;
    std::fs::write(
        trace.directory.join("response.json"),
        serde_json::to_vec_pretty(&response)?,
    )?;
    let metadata = serde_json::json!({
        "format": "atlas-pi05-trace-v1",
        "backend": policy.backend(),
        "nvfp4_manifest": manifests.nvfp4,
        "fp8_manifest": manifests.fp8,
        "request": args[5],
        "timing_caveat": "Tracing synchronizes and downloads intermediate tensors; not a latency measurement",
        "tensors": trace.tensors,
    });
    std::fs::write(
        trace.directory.join("metadata.json"),
        serde_json::to_vec_pretty(&metadata)?,
    )?;
    eprintln!(
        "wrote {} tensors to {}",
        trace.tensors.len(),
        trace.directory.display()
    );
    Ok(())
}
