// SPDX-License-Identifier: AGPL-3.0-only

//! Native SigLIP / PaliGemma prefix and flow-matching action expert.

#[path = "model_buffers.rs"]
mod buffers;
#[path = "model_graph.rs"]
mod graph;
#[path = "model_layers.rs"]
mod layers;
#[path = "model_trace.rs"]
mod trace;
pub use trace::TraceSink;
use trace::capture;

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::sync::Arc;

use crate::{
    config::Config,
    ops::{Nvfp4Scratch, Ops},
    weights::Weights,
};
use buffers::{Buffers, Scratch, upload_f32};
#[path = "model_time.rs"]
mod time;
#[path = "model_vision.rs"]
mod vision;
use layers::{Linear, PREFIX, TransformerLayer, VisionLayer};
use time::prepare_time;

/// All weights and scratch buffers remain resident across observations.
pub struct NativeModel {
    gpu: Arc<dyn GpuBackend>,
    _weights: Weights,
    _buffers: Buffers,
    config: Config,
    graphs: graph::GraphCache,
    patch_weight: DevicePtr,
    patch_bias: DevicePtr,
    patch_position: DevicePtr,
    vision: Vec<VisionLayer>,
    vision_norm: (DevicePtr, DevicePtr),
    projector: Linear,
    embedding: DevicePtr,
    prefix: Vec<TransformerLayer>,
    expert: Vec<TransformerLayer>,
    action_in: Linear,
    action_out: Linear,
    patches: DevicePtr,
    tokens: DevicePtr,
    positions: DevicePtr,
    image_scratch: Scratch,
    prefix_scratch: Scratch,
    expert_scratch: Scratch,
    cache: Vec<(DevicePtr, DevicePtr)>,
    modulation: Vec<(DevicePtr, DevicePtr)>,
    final_modulation: DevicePtr,
    noise: DevicePtr,
    velocity: DevicePtr,
    nvfp4_scratch: Option<Nvfp4Scratch>,
    fp8_scratch: Option<crate::ops_fp8::Fp8Scratch>,
}

impl Drop for NativeModel {
    fn drop(&mut self) {
        // A failed infer may still have queued work referencing resident weights.
        let _ = self.gpu.synchronize(self.gpu.default_stream());
    }
}

impl NativeModel {
    pub fn new(gpu: Arc<dyn GpuBackend>, weights: Weights, config: Config) -> Result<Self> {
        config.validate()?;
        let ops = Ops::new(gpu.as_ref(), gpu.default_stream())?;
        let mut buffers = Buffers::new(Arc::clone(&gpu));
        let vroot = format!("{PREFIX}.vision_tower.vision_model");
        let patch_weight = weights.get(
            &format!("{vroot}.embeddings.patch_embedding.weight"),
            &[1152, 3, 14, 14],
        )?;
        let patch_bias =
            weights.get(&format!("{vroot}.embeddings.patch_embedding.bias"), &[1152])?;
        let patch_position = weights.get(
            &format!("{vroot}.embeddings.position_embedding.weight"),
            &[256, 1152],
        )?;
        let vision = (0..27)
            .map(|i| VisionLayer::load(&weights, i))
            .collect::<Result<Vec<_>>>()?;
        let vision_norm = (
            weights.get(&format!("{vroot}.post_layernorm.weight"), &[1152])?,
            weights.get(&format!("{vroot}.post_layernorm.bias"), &[1152])?,
        );
        let projector = Linear::load(
            &weights,
            &format!("{PREFIX}.multi_modal_projector.linear"),
            1152,
            2048,
            true,
        )?;
        // safetensors metadata ties embed_tokens to this stored LM-head tensor.
        let embedding = weights.get(
            "paligemma_with_expert.paligemma.lm_head.weight",
            &[257152, 2048],
        )?;
        let prefix = (0..18)
            .map(|i| TransformerLayer::load(&weights, i, false))
            .collect::<Result<Vec<_>>>()?;
        let expert = (0..18)
            .map(|i| TransformerLayer::load(&weights, i, true))
            .collect::<Result<Vec<_>>>()?;
        let action_in = Linear::load(&weights, "action_in_proj", 32, 1024, true)?;
        let action_out = Linear::load(&weights, "action_out_proj", 1024, 32, true)?;
        let patches = buffers.alloc(2 * 256 * 588 * 4)?;
        let tokens = buffers.alloc(200 * 4)?;
        let positions = buffers.alloc(722 * 4)?;
        let image_scratch = Scratch::new(&mut buffers, 512, 1152, 4304)?;
        let prefix_scratch = Scratch::new(&mut buffers, 712, 2048, 16384)?;
        let expert_scratch = Scratch::new(&mut buffers, 10, 1024, 4096)?;
        let nvfp4_scratch = if weights.has_nvfp4() {
            Some(Nvfp4Scratch {
                packed: buffers.alloc(Nvfp4Scratch::packed_bytes(712, 16384))?,
                scales: buffers.alloc(Nvfp4Scratch::scale_bytes(712, 16384))?,
                max_rows: 712,
                max_k: 16384,
            })
        } else {
            None
        };
        let fp8_scratch = if weights.has_fp8() {
            Some(crate::ops_fp8::Fp8Scratch {
                packed: buffers.alloc(crate::ops_fp8::Fp8Scratch::packed_bytes(512, 4304))?,
                max_rows: 512,
                max_k: 4304,
            })
        } else {
            None
        };
        let mut cache = Vec::with_capacity(18);
        for _ in 0..18 {
            cache.push((buffers.alloc(722 * 256 * 2)?, buffers.alloc(722 * 256 * 2)?));
        }
        let (modulation, final_modulation) =
            prepare_time(&ops, &weights, &expert, &mut buffers, config.num_steps)?;
        let noise = buffers.alloc(10 * 32 * 4)?;
        let velocity = buffers.alloc(10 * 32 * 4)?;
        gpu.synchronize(gpu.default_stream())?;
        let graphs = graph::GraphCache::new(Arc::clone(&gpu));
        Ok(Self {
            graphs,
            gpu,
            _weights: weights,
            _buffers: buffers,
            config,
            patch_weight,
            patch_bias,
            patch_position,
            vision,
            vision_norm,
            projector,
            embedding,
            prefix,
            expert,
            action_in,
            action_out,
            patches,
            tokens,
            positions,
            image_scratch,
            prefix_scratch,
            expert_scratch,
            cache,
            modulation,
            final_modulation,
            noise,
            velocity,
            nvfp4_scratch,
            fp8_scratch,
        })
    }

    /// Returns normalized `[10,32]` actions. Noise is explicit for reproducibility.
    pub fn infer(&mut self, patches: &[f32], token_ids: &[u32], noise: &[f32]) -> Result<Vec<f32>> {
        self.infer_impl(patches, token_ids, noise, None)
    }

    /// Diagnostic inference with synchronized stage snapshots. Exclude this
    /// path from latency measurements because snapshots download GPU memory.
    pub fn infer_traced(
        &mut self,
        patches: &[f32],
        token_ids: &[u32],
        noise: &[f32],
        sink: &mut dyn TraceSink,
    ) -> Result<Vec<f32>> {
        self.infer_impl(patches, token_ids, noise, Some(sink))
    }

    fn infer_impl(
        &mut self,
        patches: &[f32],
        token_ids: &[u32],
        noise: &[f32],
        sink: Option<&mut dyn TraceSink>,
    ) -> Result<Vec<f32>> {
        ensure!(
            patches.len() == 2 * 256 * 588,
            "expected two 224x224 RGB images in patch order"
        );
        ensure!(
            !token_ids.is_empty() && token_ids.len() <= self.config.max_token_len,
            "prompt must contain 1..=200 tokens"
        );
        ensure!(
            token_ids
                .iter()
                .all(|&id| (id as usize) < self.config.vocab_size),
            "token exceeds vocabulary"
        );
        ensure!(
            noise.len() == 320 && noise.iter().all(|x| x.is_finite()),
            "noise must contain 320 finite values"
        );
        ensure!(
            patches.iter().all(|x| x.is_finite()),
            "image patches must be finite"
        );
        upload_f32(self.gpu.as_ref(), self.patches, patches)?;
        upload_f32(self.gpu.as_ref(), self.noise, noise)?;
        self.gpu
            .copy_h2d(bytemuck::cast_slice(token_ids), self.tokens)?;
        let rows = 512 + token_ids.len();
        let positions: Vec<i32> = (0..(rows + 10) as i32).collect();
        self.gpu
            .copy_h2d(bytemuck::cast_slice(&positions), self.positions)?;
        let stream = self.gpu.default_stream();
        if sink.is_some() {
            self.compute(token_ids.len(), sink)?;
        } else if let Some(graph) = self.graphs.get(rows) {
            self.gpu.launch_graph(graph, stream)?;
        } else {
            // Warm vendor algorithms before capture: first-use allocations may
            // not occur inside a CUDA graph. Reset the mutated Euler state.
            self.compute(token_ids.len(), None)?;
            self.gpu.synchronize(stream)?;
            upload_f32(self.gpu.as_ref(), self.noise, noise)?;
            self.gpu.begin_capture(stream)?;
            if let Err(error) = self.compute(token_ids.len(), None) {
                self.gpu.abort_capture_if_active(stream);
                return Err(error);
            }
            let graph = match self.gpu.end_capture(stream) {
                Ok(graph) => graph,
                Err(error) => {
                    self.gpu.abort_capture_if_active(stream);
                    return Err(error);
                }
            };
            self.graphs.insert(rows, graph)?;
            self.gpu.launch_graph(graph, stream)?;
        }
        let mut result = vec![0_f32; 320];
        self.gpu
            .copy_d2h_on_stream(self.noise, bytemuck::cast_slice_mut(&mut result), stream)?;
        ensure!(
            result.iter().all(|x| x.is_finite()),
            "native pi0.5 produced non-finite actions"
        );
        Ok(result)
    }

    fn compute(&self, token_count: usize, mut sink: Option<&mut dyn TraceSink>) -> Result<()> {
        let rows = 512 + token_count;
        let mut ops = Ops::new(self.gpu.as_ref(), self.gpu.default_stream())?;
        ops.nvfp4_scratch = self.nvfp4_scratch.as_ref();
        ops.fp8_scratch = self.fp8_scratch.as_ref();
        self.embed_images(&ops, &mut sink)?;
        ops.embedding(
            self.embedding,
            self.tokens,
            self.prefix_scratch.hidden.offset(512 * 2048 * 2),
            token_count,
            2048,
            (2048_f32).sqrt(),
        )?;
        capture(
            self.gpu.as_ref(),
            &mut sink,
            "prefix.embedding",
            &[rows, 2048],
            self.prefix_scratch.hidden,
            true,
        )?;
        for (index, (layer, &(key, value))) in self.prefix.iter().zip(&self.cache).enumerate() {
            if index + 1 == self.prefix.len() {
                layer.cache_only(
                    &ops,
                    &self.prefix_scratch,
                    rows,
                    self.positions,
                    key,
                    value,
                    self.config.rms_norm_eps,
                )?;
                capture(
                    self.gpu.as_ref(),
                    &mut sink,
                    &format!("prefix.{index}.key"),
                    &[rows, 1, 256],
                    key,
                    true,
                )?;
                capture(
                    self.gpu.as_ref(),
                    &mut sink,
                    &format!("prefix.{index}.value"),
                    &[rows, 1, 256],
                    value,
                    true,
                )?;
                break;
            }
            layer.forward(
                &ops,
                &self.prefix_scratch,
                rows,
                self.positions,
                key,
                value,
                0,
                None,
                self.config.rms_norm_eps,
            )?;
            if index == 0 {
                capture(
                    self.gpu.as_ref(),
                    &mut sink,
                    "prefix.0.hidden",
                    &[rows, 2048],
                    self.prefix_scratch.hidden,
                    true,
                )?;
                capture(
                    self.gpu.as_ref(),
                    &mut sink,
                    "prefix.0.key",
                    &[rows, 1, 256],
                    key,
                    true,
                )?;
                capture(
                    self.gpu.as_ref(),
                    &mut sink,
                    "prefix.0.value",
                    &[rows, 1, 256],
                    value,
                    true,
                )?;
            }
        }
        // Prefix final RMSNorm is unnecessary: only per-layer K/V are consumed.
        let dt = -1.0 / self.config.num_steps as f32;
        for step in 0..self.config.num_steps {
            self.action_in
                .f32(&ops, self.noise, self.expert_scratch.f32, 10)?;
            ops.cast_to_bf16(
                self.expert_scratch.f32,
                self.expert_scratch.hidden,
                10 * 1024,
            )?;
            capture(
                self.gpu.as_ref(),
                &mut sink,
                &format!("step.{step}.embedding"),
                &[10, 1024],
                self.expert_scratch.hidden,
                true,
            )?;
            for ((layer, &(key, value)), &(m1, m2)) in
                self.expert.iter().zip(&self.cache).zip(&self.modulation)
            {
                let offset = step * 3072 * 4;
                layer.forward(
                    &ops,
                    &self.expert_scratch,
                    10,
                    self.positions.offset(rows * 4),
                    key,
                    value,
                    rows,
                    Some((m1.offset(offset), m2.offset(offset))),
                    self.config.rms_norm_eps,
                )?;
            }
            ops.adarms(
                self.expert_scratch.hidden,
                self.final_modulation.offset(step * 3072 * 4),
                self.expert_scratch.norm,
                10,
                1024,
                self.config.rms_norm_eps,
            )?;
            capture(
                self.gpu.as_ref(),
                &mut sink,
                &format!("step.{step}.normalized"),
                &[10, 1024],
                self.expert_scratch.norm,
                true,
            )?;
            ops.cast_to_f32(self.expert_scratch.norm, self.expert_scratch.f32, 10 * 1024)?;
            self.action_out
                .f32(&ops, self.expert_scratch.f32, self.velocity, 10)?;
            capture(
                self.gpu.as_ref(),
                &mut sink,
                &format!("step.{step}.velocity"),
                &[10, 32],
                self.velocity,
                false,
            )?;
            ops.axpy_f32(self.noise, self.velocity, 320, dt)?;
            capture(
                self.gpu.as_ref(),
                &mut sink,
                &format!("step.{step}.actions"),
                &[10, 32],
                self.noise,
                false,
            )?;
        }
        Ok(())
    }
}
