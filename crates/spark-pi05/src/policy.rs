// SPDX-License-Identifier: AGPL-3.0-only

//! Filesystem/GPU assembly boundary for the native policy.

use crate::{
    config::Config,
    model::NativeModel,
    preprocess::{PolicyRequest, Preprocessor},
    weights::{SafeTensorFile, Weights},
};
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use spark_runtime::{cuda_backend::AtlasCudaBackend, gpu::GpuBackend};
use std::{path::Path, sync::Arc, time::Instant};

static PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pi05.ptx"));

#[derive(Debug, Serialize)]
pub struct PolicyResponse {
    pub actions: Vec<[f64; 7]>,
    /// Synchronized preprocessing, inference and output transformation time.
    pub timing_ms: f64,
    pub backend: &'static str,
}

pub struct NativePolicy {
    preprocess: Preprocessor,
    model: NativeModel,
    backend: &'static str,
}

impl NativePolicy {
    pub fn load(
        checkpoint: &Path,
        config: &Path,
        tokenizer: &Path,
        norm_stats: &Path,
        gpu_ordinal: usize,
    ) -> Result<Self> {
        Self::load_with_nvfp4(checkpoint, config, tokenizer, norm_stats, gpu_ordinal, None)
    }

    /// Use the explicit manifest to replace PaliGemma prefix linear weights
    /// with native NVFP4. A failed or incompatible manifest never falls back.
    pub fn load_with_nvfp4(
        checkpoint: &Path,
        config: &Path,
        tokenizer: &Path,
        norm_stats: &Path,
        gpu_ordinal: usize,
        nvfp4_manifest: Option<&Path>,
    ) -> Result<Self> {
        Self::load_with_quantization(
            checkpoint,
            config,
            tokenizer,
            norm_stats,
            gpu_ordinal,
            nvfp4_manifest,
            None,
        )
    }

    /// Load explicit precision artifacts, with FP8 layered over the NVFP4 prefix.
    /// Invalid or incompatible artifacts fail loading without a precision fallback.
    pub fn load_with_quantization(
        checkpoint: &Path,
        config: &Path,
        tokenizer: &Path,
        norm_stats: &Path,
        gpu_ordinal: usize,
        nvfp4_manifest: Option<&Path>,
        fp8_manifest: Option<&Path>,
    ) -> Result<Self> {
        ensure!(
            fp8_manifest.is_none() || nvfp4_manifest.is_some(),
            "FP8 manifest requires an NVFP4 prefix manifest"
        );
        let config: Config =
            serde_json::from_slice(&std::fs::read(config).context("read pi0.5 profile")?)?;
        config.validate()?;
        let preprocess =
            Preprocessor::from_bytes(&std::fs::read(tokenizer)?, &std::fs::read(norm_stats)?)?;
        ensure!(
            !PTX.is_empty(),
            "native pi0.5 kernels were not built; rebuild with CUDA (ATLAS_PI05_BUILD=1 for an isolated policy build)"
        );
        let mut source = SafeTensorFile::open(checkpoint)?;
        let gpu: Arc<dyn GpuBackend> =
            Arc::new(AtlasCudaBackend::new(gpu_ordinal, &[("pi05", PTX)])?);
        let (weights, backend) = if let (Some(nvfp4), Some(fp8)) = (nvfp4_manifest, fp8_manifest) {
            (
                Weights::load_nvfp4_fp8(Arc::clone(&gpu), &mut source, nvfp4, fp8)?,
                "atlas-native-fp8-nvfp4",
            )
        } else if let Some(manifest) = nvfp4_manifest {
            (
                Weights::load_nvfp4(Arc::clone(&gpu), &mut source, manifest)?,
                "atlas-native-nvfp4",
            )
        } else {
            (
                Weights::load(Arc::clone(&gpu), &mut source)?,
                "atlas-native-bf16",
            )
        };
        let model = NativeModel::new(gpu, weights, config)?;
        Ok(Self {
            preprocess,
            model,
            backend,
        })
    }

    /// Precision identity of the successfully loaded policy.
    pub fn backend(&self) -> &'static str {
        self.backend
    }

    pub fn infer(&mut self, request: &PolicyRequest) -> Result<PolicyResponse> {
        let start = Instant::now();
        let input = self.preprocess.prepare(request)?;
        let normalized = self
            .model
            .infer(&input.patches, &input.token_ids, &input.noise)?;
        let actions = self.preprocess.decode_actions(&normalized)?;
        Ok(PolicyResponse {
            actions,
            timing_ms: start.elapsed().as_secs_f64() * 1000.0,
            backend: self.backend,
        })
    }

    /// Diagnostic harness entry point; tracing intentionally synchronizes stages.
    pub fn infer_traced(
        &mut self,
        request: &PolicyRequest,
        sink: &mut dyn crate::model::TraceSink,
    ) -> Result<PolicyResponse> {
        let start = Instant::now();
        let input = self.preprocess.prepare(request)?;
        let normalized =
            self.model
                .infer_traced(&input.patches, &input.token_ids, &input.noise, sink)?;
        let actions = self.preprocess.decode_actions(&normalized)?;
        Ok(PolicyResponse {
            actions,
            timing_ms: start.elapsed().as_secs_f64() * 1000.0,
            backend: self.backend,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_without_nvfp4_fails_before_asset_or_gpu_access() {
        let missing = Path::new("/no-policy-assets");
        let result = NativePolicy::load_with_quantization(
            missing,
            missing,
            missing,
            missing,
            0,
            None,
            Some(missing),
        );
        let error = result.err().expect("FP8 requires NVFP4");
        assert_eq!(
            error.to_string(),
            "FP8 manifest requires an NVFP4 prefix manifest"
        );
    }
}
