// SPDX-License-Identifier: AGPL-3.0-only
//! Calibrated FP8 overlay bound to a verified NVFP4 checkpoint manifest.
use super::{DType, Fp8Weight, Weight, WeightSource, Weights};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use spark_runtime::gpu::GpuBackend;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Component, Path},
    sync::Arc,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    format_version: u32,
    model_type: String,
    profile: String,
    precision: String,
    weight_file: String,
    base_checkpoint_sha256: String,
    nvfp4_manifest_sha256: String,
    base_tensor_sha256: BTreeMap<String, String>,
    calibration: serde_json::Value,
    tensors: Vec<Entry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    name: String,
    shape: [usize; 2],
    data_offset: u64,
    weight_scale: f32,
    activation_scale: f32,
    packed_sha256: String,
}

fn expected_shapes(precision: &str) -> Result<BTreeMap<String, [usize; 2]>> {
    ensure!(
        matches!(
            precision,
            "fp8-vision-bf16" | "fp8-vision-expert-bf16" | "fp8-expert-bf16"
        ),
        "unsupported FP8 precision recipe"
    );
    let c: crate::config::Config = serde_json::from_str(include_str!("../pi05_libero.json"))?;
    let mut shapes = BTreeMap::new();
    let root = "paligemma_with_expert.paligemma.model";
    if precision != "fp8-expert-bf16" {
        for i in 0..c.vision_layers {
            for (name, shape) in [
                ("self_attn.q_proj", [c.vision_hidden, c.vision_hidden]),
                ("self_attn.k_proj", [c.vision_hidden, c.vision_hidden]),
                ("self_attn.v_proj", [c.vision_hidden, c.vision_hidden]),
                ("self_attn.out_proj", [c.vision_hidden, c.vision_hidden]),
                ("mlp.fc1", [c.vision_intermediate, c.vision_hidden]),
                ("mlp.fc2", [c.vision_hidden, c.vision_intermediate]),
            ] {
                shapes.insert(
                    format!("{root}.vision_tower.vision_model.encoder.layers.{i}.{name}.weight"),
                    shape,
                );
            }
        }
        shapes.insert(
            format!("{root}.multi_modal_projector.linear.weight"),
            [c.hidden_size, c.vision_hidden],
        );
    }
    if precision != "fp8-vision-bf16" {
        for i in 0..c.num_layers {
            for (name, shape) in [
                (
                    "self_attn.q_proj",
                    [c.num_heads * c.head_dim, c.expert_hidden],
                ),
                (
                    "self_attn.k_proj",
                    [c.num_kv_heads * c.head_dim, c.expert_hidden],
                ),
                (
                    "self_attn.v_proj",
                    [c.num_kv_heads * c.head_dim, c.expert_hidden],
                ),
                (
                    "self_attn.o_proj",
                    [c.expert_hidden, c.num_heads * c.head_dim],
                ),
                ("mlp.gate_proj", [c.expert_intermediate, c.expert_hidden]),
                ("mlp.up_proj", [c.expert_intermediate, c.expert_hidden]),
                ("mlp.down_proj", [c.expert_hidden, c.expert_intermediate]),
            ] {
                shapes.insert(
                    format!("paligemma_with_expert.gemma_expert.model.layers.{i}.{name}.weight"),
                    shape,
                );
            }
        }
    }
    Ok(shapes)
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl Manifest {
    fn validate(&self, length: u64) -> Result<()> {
        ensure!(
            self.format_version == 1 && self.model_type == "pi05" && self.profile == "pi05_libero",
            "unsupported FP8 manifest version or model"
        );
        let expected = expected_shapes(&self.precision)?;
        ensure!(
            self.tensors.len() == expected.len(),
            "FP8 tensor inventory mismatch"
        );
        ensure!(
            valid_digest(&self.base_checkpoint_sha256) && valid_digest(&self.nvfp4_manifest_sha256),
            "invalid FP8 checkpoint binding"
        );
        ensure!(
            self.calibration.as_object().is_some_and(|v| !v.is_empty()),
            "missing FP8 calibration provenance"
        );
        ensure!(
            !self.weight_file.is_empty()
                && Path::new(&self.weight_file)
                    .components()
                    .all(|p| matches!(p, Component::Normal(_))),
            "FP8 payload path must be relative without traversal"
        );
        let mut seen = BTreeSet::new();
        let mut spans = Vec::new();
        for e in &self.tensors {
            ensure!(
                seen.insert(&e.name) && expected.get(&e.name) == Some(&e.shape),
                "unexpected FP8 tensor or shape: {}",
                e.name
            );
            ensure!(valid_digest(&e.packed_sha256), "invalid FP8 payload digest");
            let alpha = e.weight_scale * e.activation_scale;
            ensure!(
                e.weight_scale.is_finite()
                    && e.weight_scale > 0.0
                    && e.activation_scale.is_finite()
                    && e.activation_scale > 0.0
                    && (1.0 / e.activation_scale).is_finite()
                    && alpha.is_finite()
                    && alpha > 0.0,
                "invalid FP8 scales for {}",
                e.name
            );
            let bytes = e.shape[0] * e.shape[1];
            let end = e
                .data_offset
                .checked_add(bytes as u64)
                .context("FP8 offset overflow")?;
            ensure!(end <= length, "truncated FP8 payload for {}", e.name);
            spans.push((e.data_offset, end));
        }
        spans.sort_unstable();
        ensure!(
            spans.windows(2).all(|w| w[0].1 <= w[1].0),
            "overlapping FP8 payload spans"
        );
        ensure!(
            self.base_tensor_sha256.keys().eq(expected.keys())
                && self.base_tensor_sha256.values().all(|v| valid_digest(v)),
            "FP8 source tensor hash inventory mismatch"
        );
        Ok(())
    }

    fn validate_binding(
        &self,
        manifest_hash: Option<&str>,
        hashes: &BTreeMap<String, String>,
    ) -> Result<()> {
        ensure!(
            manifest_hash == Some(self.nvfp4_manifest_sha256.as_str()),
            "FP8 overlay requires its exact NVFP4 manifest"
        );
        ensure!(
            self.base_tensor_sha256
                .iter()
                .all(|(name, hash)| hashes.get(name) == Some(hash)),
            "FP8 source weights differ from verified checkpoint"
        );
        Ok(())
    }
}

fn validate_payload(bytes: &[u8], digest: &str) -> Result<()> {
    ensure!(
        format!("{:x}", Sha256::digest(bytes)) == digest,
        "FP8 SHA-256 mismatch"
    );
    ensure!(
        bytes.iter().all(|b| b & 0x7f != 0x7f),
        "FP8 payload contains non-finite E4M3 values"
    );
    Ok(())
}

impl Weights {
    /// Overlay calibrated FP8 vision/expert matrices on the verified prefix checkpoint.
    pub fn load_nvfp4_fp8(
        gpu: Arc<dyn GpuBackend>,
        source: &mut dyn WeightSource,
        nvfp4_path: &Path,
        fp8_path: &Path,
    ) -> Result<Self> {
        let mut bytes = Vec::new();
        File::open(fp8_path)
            .context("open FP8 manifest")?
            .take(4 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        ensure!(bytes.len() <= 4 * 1024 * 1024, "FP8 manifest too large");
        let manifest: Manifest = serde_json::from_slice(&bytes).context("parse FP8 manifest")?;
        manifest.validate(u64::MAX)?;
        let mut payload = File::open(
            fp8_path
                .parent()
                .context("FP8 manifest parent")?
                .join(&manifest.weight_file),
        )?;
        manifest.validate(payload.metadata()?.len())?;
        let mut weights = Self::load_nvfp4(gpu, source, nvfp4_path)?;
        ensure!(
            weights.base_checkpoint_sha256.as_deref()
                == Some(manifest.base_checkpoint_sha256.as_str()),
            "FP8 source checkpoint digest contradicts the NVFP4 manifest"
        );
        manifest.validate_binding(
            weights.nvfp4_manifest_sha256.as_deref(),
            &weights.base_tensor_sha256,
        )?;
        for e in manifest.tensors {
            let old = weights.resolve(&e.name)?;
            ensure!(
                old.dtype == DType::Bf16 && old.shape == e.shape,
                "FP8 may replace only the declared BF16 affine weights"
            );
            payload.seek(SeekFrom::Start(e.data_offset))?;
            let mut bytes = vec![0; e.shape[0] * e.shape[1]];
            payload.read_exact(&mut bytes)?;
            validate_payload(&bytes, &e.packed_sha256).with_context(|| e.name.clone())?;
            let ptr = weights.gpu.alloc(bytes.len())?;
            // Record new ownership before either free or upload can fail.
            let old = weights
                .map
                .insert(
                    e.name,
                    Weight {
                        ptr,
                        shape: e.shape.to_vec(),
                        dtype: DType::Fp8,
                        nvfp4: None,
                        fp8: Some(Fp8Weight {
                            weight_scale: e.weight_scale,
                            activation_scale: e.activation_scale,
                        }),
                    },
                )
                .context("FP8 base weight missing")?;
            weights.gpu.free(old.ptr)?;
            weights.gpu.copy_h2d(&bytes, ptr)?;
        }
        Ok(weights)
    }
}

#[cfg(test)]
#[path = "weights_fp8_tests.rs"]
mod tests;
