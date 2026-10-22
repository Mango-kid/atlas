// SPDX-License-Identifier: AGPL-3.0-only

//! Checkpoint I/O boundary and owned GPU weights.

use anyhow::{Context, Result, ensure};
use serde_json::Value;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

#[path = "weights_fp8.rs"]
mod fp8;
#[path = "weights_nvfp4.rs"]
mod nvfp4;

#[derive(Debug, Clone, Copy)]
pub struct Fp8Weight {
    pub weight_scale: f32,
    pub activation_scale: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Nvfp4Weight {
    pub scales: DevicePtr,
    pub weight_global_scale: f32,
    pub activation_global_scale: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Bf16,
    F32,
    Nvfp4,
    Fp8,
}

pub struct Weight {
    pub ptr: DevicePtr,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub nvfp4: Option<Nvfp4Weight>,
    pub fp8: Option<Fp8Weight>,
}

pub struct TensorData {
    pub name: String,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

/// Storage boundary: model assembly never opens checkpoint files.
pub trait WeightSource {
    fn names(&self) -> Vec<String>;
    fn read(&mut self, name: &str) -> Result<TensorData>;
}

pub struct SafeTensorFile {
    file: File,
    tensors: BTreeMap<String, (Vec<usize>, atlas_core::safetensors::TensorSpan)>,
}

impl SafeTensorFile {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let file_len = file.metadata()?.len();
        let mut len = [0u8; 8];
        file.read_exact(&mut len)?;
        let header_len = u64::from_le_bytes(len);
        ensure!(
            header_len <= 64 * 1024 * 1024,
            "safetensors header too large"
        );
        ensure!(header_len + 8 <= file_len, "truncated safetensors header");
        let mut bytes = vec![0; header_len as usize];
        file.read_exact(&mut bytes)?;
        let header: Value = serde_json::from_slice(&bytes)?;
        let tensors = parse_header(&header, header_len + 8, file_len)?;
        Ok(Self { file, tensors })
    }
}

fn parse_header(
    header: &Value,
    data_start: u64,
    file_len: u64,
) -> Result<BTreeMap<String, (Vec<usize>, atlas_core::safetensors::TensorSpan)>> {
    let mut tensors = BTreeMap::new();
    for (name, entry) in header
        .as_object()
        .context("tensor header must be an object")?
    {
        if name == "__metadata__" {
            continue;
        }
        ensure!(
            entry["dtype"] == "BF16",
            "{name}: expected converted BF16 weights"
        );
        let shape: Vec<usize> = serde_json::from_value(entry["shape"].clone())?;
        ensure!(
            !shape.is_empty() && !shape.contains(&0),
            "{name}: empty shape"
        );
        let bytes = shape
            .iter()
            .try_fold(2usize, |a, b| a.checked_mul(*b))
            .context("tensor size overflow")?;
        let span = atlas_core::safetensors::tensor_span(
            name,
            &entry["data_offsets"],
            data_start,
            file_len,
        )?;
        ensure!(
            span.len == bytes as u64,
            "{name}: shape/byte length mismatch"
        );
        tensors.insert(name.clone(), (shape, span));
    }
    ensure!(!tensors.is_empty(), "empty checkpoint");
    Ok(tensors)
}

impl WeightSource for SafeTensorFile {
    fn names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    fn read(&mut self, name: &str) -> Result<TensorData> {
        let (shape, span) = self
            .tensors
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        self.file.seek(SeekFrom::Start(span.abs_offset))?;
        let mut bytes = vec![0; span.len as usize];
        self.file.read_exact(&mut bytes)?;
        Ok(TensorData {
            name: name.into(),
            shape: shape.clone(),
            bytes,
        })
    }
}

pub struct Weights {
    gpu: Arc<dyn GpuBackend>,
    map: BTreeMap<String, Weight>,
    scale_allocations: Vec<DevicePtr>,
    nvfp4_manifest_sha256: Option<String>,
    base_checkpoint_sha256: Option<String>,
    base_tensor_sha256: BTreeMap<String, String>,
}

impl Weights {
    pub fn load(gpu: Arc<dyn GpuBackend>, source: &mut dyn WeightSource) -> Result<Self> {
        let mut weights = Self {
            gpu,
            map: BTreeMap::new(),
            scale_allocations: Vec::new(),
            nvfp4_manifest_sha256: None,
            base_checkpoint_sha256: None,
            base_tensor_sha256: BTreeMap::new(),
        };
        for name in source.names() {
            // The expert does not generate vocabulary logits. This tensor alone is 0.5 GB.
            if name == "paligemma_with_expert.gemma_expert.lm_head.weight" {
                continue;
            }
            let data = source.read(&name)?;
            let dtype = if promote_to_f32(&name) {
                DType::F32
            } else {
                DType::Bf16
            };
            let bytes = if dtype == DType::F32 {
                data.bytes
                    .chunks_exact(2)
                    .flat_map(|b| {
                        half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]]))
                            .to_f32()
                            .to_le_bytes()
                    })
                    .collect()
            } else {
                data.bytes
            };
            let ptr = weights.gpu.alloc(bytes.len())?;
            // Record ownership before upload so all error paths release the allocation.
            weights.map.insert(
                name.clone(),
                Weight {
                    ptr,
                    shape: data.shape,
                    dtype,
                    nvfp4: None,
                    fp8: None,
                },
            );
            weights
                .gpu
                .copy_h2d(&bytes, ptr)
                .with_context(|| format!("upload {name}"))?;
        }
        Ok(weights)
    }

    fn resolve(&self, name: &str) -> Result<&Weight> {
        // OpenPi's safetensors export deduplicates the tied embedding into lm_head.
        let alias = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight";
        let name = if name == alias && !self.map.contains_key(name) {
            "paligemma_with_expert.paligemma.lm_head.weight"
        } else {
            name
        };
        self.map
            .get(name)
            .with_context(|| format!("missing pi0.5 weight {name}"))
    }

    pub fn get(&self, name: &str, shape: &[usize]) -> Result<DevicePtr> {
        let weight = self.resolve(name)?;
        ensure!(
            weight.shape == shape,
            "{name}: shape {:?}, expected {shape:?}",
            weight.shape
        );
        Ok(weight.ptr)
    }

    pub fn dtype(&self, name: &str) -> Result<DType> {
        Ok(self.resolve(name)?.dtype)
    }

    pub fn nvfp4(&self, name: &str) -> Result<Option<Nvfp4Weight>> {
        Ok(self.resolve(name)?.nvfp4)
    }

    pub fn has_nvfp4(&self) -> bool {
        !self.scale_allocations.is_empty()
    }

    pub fn fp8(&self, name: &str) -> Result<Option<Fp8Weight>> {
        Ok(self.resolve(name)?.fp8)
    }

    pub fn has_fp8(&self) -> bool {
        self.map.values().any(|weight| weight.fp8.is_some())
    }
}

fn promote_to_f32(name: &str) -> bool {
    [
        "action_in_proj.",
        "action_out_proj.",
        "time_mlp_in.",
        "time_mlp_out.",
    ]
    .iter()
    .any(|p| name.starts_with(p))
        || [
            "patch_embedding",
            "position_embedding",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]
        .iter()
        .any(|p| name.contains(p))
}

impl Drop for Weights {
    fn drop(&mut self) {
        for weight in self.map.values() {
            if let Err(e) = self.gpu.free(weight.ptr) {
                tracing::warn!("free pi0.5 weight: {e}");
            }
        }
        for ptr in self.scale_allocations.drain(..) {
            let _ = self.gpu.free(ptr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn rejects_corrupt_tensor_sizes_and_spans() {
        let good = json!({"w": {"dtype":"BF16", "shape":[2,3], "data_offsets":[0,12]}});
        assert!(parse_header(&good, 100, 112).is_ok());
        assert!(parse_header(&good, 100, 111).is_err());
        let mut bad = good.clone();
        bad["w"]["shape"] = json!([3, 3]);
        assert!(parse_header(&bad, 100, 112).is_err());
        bad["w"]["data_offsets"] = json!([12, 0]);
        assert!(parse_header(&bad, 100, 112).is_err());
    }

    #[test]
    fn precision_matches_openpi_mixed_precision_loader() {
        assert!(promote_to_f32("time_mlp_in.weight"));
        assert!(promote_to_f32(
            "paligemma_with_expert.gemma_expert.model.layers.0.input_layernorm.dense.weight"
        ));
        assert!(!promote_to_f32(
            "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight"
        ));
        assert!(!promote_to_f32(
            "paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.0.layer_norm1.weight"
        ));
    }
}
