// SPDX-License-Identifier: AGPL-3.0-only

//! Versioned, hashed checkpoint boundary for the calibrated NVFP4 prefix.
use super::{DType, Nvfp4Weight, TensorData, Weight, WeightSource, Weights};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use spark_runtime::gpu::GpuBackend;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path};
use std::sync::Arc;

const UNUSED: &str = "paligemma_with_expert.gemma_expert.lm_head.weight";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    format_version: u32,
    model_type: String,
    profile: String,
    precision: String,
    weight_file: String,
    base_checkpoint_sha256: String,
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
    scale_offset: u64,
    weight_global_scale: f32,
    activation_global_scale: f32,
    packed_sha256: String,
    scales_sha256: String,
}

impl Entry {
    fn lengths(&self) -> (usize, usize) {
        let [n, k] = self.shape;
        (n * k / 2, n.div_ceil(128) * 128 * k.div_ceil(64) * 4)
    }
}

fn expected_shapes() -> Result<BTreeMap<String, [usize; 2]>> {
    let c: crate::config::Config = serde_json::from_str(include_str!("../pi05_libero.json"))?;
    let mut shapes = BTreeMap::new();
    for i in 0..c.num_layers {
        for (suffix, shape) in [
            (
                "self_attn.q_proj",
                [c.num_heads * c.head_dim, c.hidden_size],
            ),
            (
                "self_attn.k_proj",
                [c.num_kv_heads * c.head_dim, c.hidden_size],
            ),
            (
                "self_attn.v_proj",
                [c.num_kv_heads * c.head_dim, c.hidden_size],
            ),
            (
                "self_attn.o_proj",
                [c.hidden_size, c.num_heads * c.head_dim],
            ),
            ("mlp.gate_proj", [c.intermediate_size, c.hidden_size]),
            ("mlp.up_proj", [c.intermediate_size, c.hidden_size]),
            ("mlp.down_proj", [c.hidden_size, c.intermediate_size]),
        ] {
            shapes.insert(format!("paligemma_with_expert.paligemma.model.language_model.layers.{i}.{suffix}.weight"), shape);
        }
    }
    Ok(shapes)
}

fn valid_digest(digest: &str) -> bool {
    digest.len() == 64
        && digest
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn verify_digest(bytes: &[u8], expected: &str, name: &str) -> Result<()> {
    ensure!(
        format!("{:x}", Sha256::digest(bytes)) == expected,
        "{name}: SHA-256 mismatch"
    );
    Ok(())
}

impl Manifest {
    fn validate(&self, file_len: u64) -> Result<()> {
        ensure!(
            self.format_version == 1
                && self.model_type == "pi05"
                && self.profile == "pi05_libero"
                && self.precision == "nvfp4-prefix-bf16",
            "unsupported NVFP4 manifest version, architecture, or precision recipe"
        );
        ensure!(
            valid_digest(&self.base_checkpoint_sha256),
            "invalid source checkpoint digest"
        );
        ensure!(
            self.calibration.as_object().is_some_and(|v| !v.is_empty()),
            "missing calibration provenance"
        );
        ensure!(
            !self.weight_file.is_empty()
                && Path::new(&self.weight_file)
                    .components()
                    .all(|part| matches!(part, Component::Normal(_))),
            "weight_file must be a relative path without traversal"
        );
        let expected = expected_shapes()?;
        ensure!(
            self.tensors.len() == expected.len(),
            "NVFP4 inventory must cover every prefix linear"
        );
        let mut seen = BTreeSet::new();
        let mut spans = Vec::new();
        for entry in &self.tensors {
            ensure!(
                seen.insert(&entry.name),
                "duplicate NVFP4 tensor {}",
                entry.name
            );
            ensure!(
                expected.get(&entry.name) == Some(&entry.shape),
                "unexpected NVFP4 tensor or shape: {}",
                entry.name
            );
            ensure!(
                entry.weight_global_scale.is_finite()
                    && entry.weight_global_scale > 0.0
                    && entry.activation_global_scale.is_finite()
                    && entry.activation_global_scale > 0.0,
                "{}: global scales must be finite and positive",
                entry.name
            );
            let alpha = entry.weight_global_scale * entry.activation_global_scale;
            ensure!(
                alpha.is_finite() && alpha > 0.0,
                "{}: global scale product overflow/underflow",
                entry.name
            );
            ensure!(
                (6.0 * entry.activation_global_scale).is_finite(),
                "{}: activation scale exceeds FP32 quantizer range",
                entry.name
            );
            ensure!(
                valid_digest(&entry.packed_sha256) && valid_digest(&entry.scales_sha256),
                "invalid quantized tensor digest"
            );
            let (packed, scales) = entry.lengths();
            for (start, bytes) in [(entry.data_offset, packed), (entry.scale_offset, scales)] {
                let end = start
                    .checked_add(bytes as u64)
                    .context("NVFP4 span overflow")?;
                ensure!(
                    end <= file_len,
                    "{}: truncated quantized tensor",
                    entry.name
                );
                spans.push((start, end));
            }
        }
        spans.sort_unstable();
        ensure!(
            spans.windows(2).all(|p| p[0].1 <= p[1].0),
            "overlapping NVFP4 payload spans"
        );
        ensure!(
            self.base_tensor_sha256.values().all(|v| valid_digest(v)),
            "invalid retained tensor digest"
        );
        Ok(())
    }

    fn validate_source(&self, source: &dyn WeightSource) -> Result<()> {
        let quantized: BTreeSet<_> = self.tensors.iter().map(|e| e.name.as_str()).collect();
        let names = source.names();
        let actual: BTreeSet<_> = names.iter().map(String::as_str).collect();
        ensure!(
            quantized.is_subset(&actual),
            "base checkpoint is missing quantized prefix tensors"
        );
        let retained: BTreeSet<_> = actual
            .into_iter()
            .filter(|n| *n != UNUSED && !quantized.contains(n))
            .collect();
        let hashes: BTreeSet<_> = self.base_tensor_sha256.keys().map(String::as_str).collect();
        ensure!(
            retained == hashes,
            "retained BF16 tensor inventory does not match NVFP4 manifest"
        );
        Ok(())
    }
}

struct VerifiedBase<'a> {
    source: &'a mut dyn WeightSource,
    hashes: &'a BTreeMap<String, String>,
}

impl WeightSource for VerifiedBase<'_> {
    fn names(&self) -> Vec<String> {
        self.hashes.keys().cloned().collect()
    }
    fn read(&mut self, name: &str) -> Result<TensorData> {
        let tensor = self.source.read(name)?;
        verify_digest(
            &tensor.bytes,
            self.hashes.get(name).context("unhashed base tensor")?,
            name,
        )?;
        Ok(tensor)
    }
}

fn read_payload(
    file: &mut File,
    offset: u64,
    length: usize,
    digest: &str,
    name: &str,
) -> Result<Vec<u8>> {
    file.seek(SeekFrom::Start(offset))?;
    let mut bytes = vec![0; length];
    file.read_exact(&mut bytes)?;
    verify_digest(&bytes, digest, name)?;
    Ok(bytes)
}

impl Weights {
    /// Loads calibrated packed weights, verifying all consumed base/payload bytes.
    pub fn load_nvfp4(
        gpu: Arc<dyn GpuBackend>,
        source: &mut dyn WeightSource,
        path: &Path,
    ) -> Result<Self> {
        let manifest_file = File::open(path).context("open NVFP4 manifest")?;
        let mut bytes = Vec::new();
        manifest_file
            .take(4 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        ensure!(bytes.len() <= 4 * 1024 * 1024, "NVFP4 manifest too large");
        let manifest: Manifest = serde_json::from_slice(&bytes).context("parse NVFP4 manifest")?;
        let manifest_sha256 = format!("{:x}", Sha256::digest(&bytes));
        // Validate path before opening a payload; validate spans against its actual length below.
        manifest.validate(u64::MAX)?;
        let mut file = File::open(
            path.parent()
                .context("manifest parent")?
                .join(&manifest.weight_file),
        )?;
        manifest.validate(file.metadata()?.len())?;
        manifest.validate_source(source)?;
        let mut weights = Self::load(
            gpu,
            &mut VerifiedBase {
                source,
                hashes: &manifest.base_tensor_sha256,
            },
        )?;
        for entry in manifest.tensors {
            let (packed_len, scales_len) = entry.lengths();
            let packed = read_payload(
                &mut file,
                entry.data_offset,
                packed_len,
                &entry.packed_sha256,
                &entry.name,
            )?;
            let scales = read_payload(
                &mut file,
                entry.scale_offset,
                scales_len,
                &entry.scales_sha256,
                &entry.name,
            )?;
            ensure!(
                scales.iter().all(|&v| v <= 0x7e),
                "{}: non-finite/negative E4M3 block scale",
                entry.name
            );
            let ptr = weights.gpu.alloc(packed.len())?;
            weights.map.insert(
                entry.name.clone(),
                Weight {
                    ptr,
                    shape: entry.shape.to_vec(),
                    dtype: DType::Nvfp4,
                    nvfp4: None,
                    fp8: None,
                },
            );
            weights.gpu.copy_h2d(&packed, ptr)?;
            let scale_ptr = weights.gpu.alloc(scales.len())?;
            weights.scale_allocations.push(scale_ptr);
            weights.gpu.copy_h2d(&scales, scale_ptr)?;
            weights
                .map
                .get_mut(&entry.name)
                .context("loaded quantized weight")?
                .nvfp4 = Some(Nvfp4Weight {
                scales: scale_ptr,
                weight_global_scale: entry.weight_global_scale,
                activation_global_scale: entry.activation_global_scale,
            });
        }
        weights.nvfp4_manifest_sha256 = Some(manifest_sha256);
        weights.base_checkpoint_sha256 = Some(manifest.base_checkpoint_sha256);
        weights.base_tensor_sha256 = manifest.base_tensor_sha256;
        Ok(weights)
    }
}

#[cfg(test)]
#[path = "weights_nvfp4_tests.rs"]
mod tests;
