// SPDX-License-Identifier: AGPL-3.0-only

//! Host transforms for the fixed `pi05_libero` checkpoint profile.
//!
//! Callers obtain assets through the I/O boundary and supply their contents here.
//! Images are explicit 224×224 RGB bytes in HWC order. The caller supplies the
//! initial Gaussian sample so repeated inference can use identical noise.

use anyhow::{Context, Result, ensure};
use sentencepiece_rs::SentencePieceProcessor;
use serde::{Deserialize, Serialize};

use crate::config::{
    ACTION_DIM, ACTION_HORIZON, ACTION_OUTPUT_DIM, IMAGE_COUNT, IMAGE_SIZE, MAX_PROMPT_TOKENS,
    PATCH_SIZE, STATE_DIM,
};

const RGB_CHANNELS: usize = 3;
const MAX_PROMPT_BYTES: usize = 65_536;

/// Explicit policy input, independent of HTTP or WebSocket transport.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyRequest {
    /// Flattened 224×224×3 RGB bytes, row major.
    #[serde(rename = "observation/image")]
    pub image: Vec<u8>,
    /// Flattened wrist camera image with the same layout.
    #[serde(rename = "observation/wrist_image")]
    pub wrist_image: Vec<u8>,
    /// Required for the LIBERO observation contract. This checkpoint's pi0.5
    /// expert does not consume state (`discrete_state_input=false`).
    #[serde(rename = "observation/state")]
    pub state: [f64; STATE_DIM],
    pub prompt: String,
    /// Row-major `[10,32]` initial noise, supplied explicitly for reproducibility.
    pub noise: Vec<f32>,
}

impl PolicyRequest {
    pub fn validate(&self) -> Result<()> {
        let image_len = IMAGE_SIZE * IMAGE_SIZE * RGB_CHANNELS;
        ensure!(
            self.image.len() == image_len,
            "observation/image must contain {image_len} RGB bytes"
        );
        ensure!(
            self.wrist_image.len() == image_len,
            "observation/wrist_image must contain {image_len} RGB bytes"
        );
        ensure!(
            self.state.iter().all(|x| x.is_finite()),
            "observation/state must be finite"
        );
        ensure!(
            self.prompt.len() <= MAX_PROMPT_BYTES,
            "prompt exceeds {MAX_PROMPT_BYTES} bytes"
        );
        validate_actions(&self.noise, "noise")
    }
}

/// Dense valid prefix only: the masked third camera and prompt padding never
/// contribute keys or values and are omitted from native inference.
#[derive(Debug)]
pub struct PreparedInput {
    /// `[2,256,588]`, with each patch flattened in channel, y, x order.
    pub patches: Vec<f32>,
    /// BOS, encoded prompt, separately encoded newline, truncated to 200 tokens.
    pub token_ids: Vec<u32>,
    pub noise: Vec<f32>,
}

pub struct Preprocessor {
    tokenizer: SentencePieceProcessor,
    normalizer: ActionNormalizer,
}

impl Preprocessor {
    pub fn from_bytes(tokenizer_model: &[u8], norm_stats_json: &[u8]) -> Result<Self> {
        let tokenizer = SentencePieceProcessor::from_serialized_model(tokenizer_model)
            .context("loading PaliGemma SentencePiece model")?;
        ensure!(tokenizer.bos_id().is_some(), "tokenizer has no BOS token");
        Ok(Self {
            tokenizer,
            normalizer: ActionNormalizer::from_json(norm_stats_json)?,
        })
    }

    pub fn prepare(&self, request: &PolicyRequest) -> Result<PreparedInput> {
        request.validate()?;
        let mut patches = Vec::with_capacity(IMAGE_COUNT * request.image.len());
        patches.extend(image_patches(&request.image)?);
        patches.extend(image_patches(&request.wrist_image)?);
        Ok(PreparedInput {
            patches,
            token_ids: self.tokenize(&request.prompt)?,
            noise: request.noise.clone(),
        })
    }

    /// Matches OpenPi's `PaligemmaTokenizer` with continuous state input.
    pub fn tokenize(&self, prompt: &str) -> Result<Vec<u32>> {
        ensure!(
            prompt.len() <= MAX_PROMPT_BYTES,
            "prompt exceeds {MAX_PROMPT_BYTES} bytes"
        );
        let mut ids = vec![
            self.tokenizer
                .bos_id()
                .context("tokenizer has no BOS token")?,
        ];
        ids.extend(self.tokenizer.encode_to_ids(&clean_prompt(prompt))?);
        ids.extend(self.tokenizer.encode_to_ids("\n")?);
        ids.truncate(MAX_PROMPT_TOKENS);
        ids.into_iter()
            .map(|id| u32::try_from(id).context("token id exceeds u32"))
            .collect()
    }

    /// Inverse quantile normalization and removal of padded action dimensions.
    /// `pi05_libero` sets `extra_delta_transform=false`, so no state offset is
    /// added and no clipping is performed.
    pub fn decode_actions(&self, actions: &[f32]) -> Result<Vec<[f64; ACTION_OUTPUT_DIM]>> {
        self.normalizer.decode(actions)
    }
}

fn clean_prompt(prompt: &str) -> String {
    prompt.trim().replace(['_', '\n'], " ")
}

fn image_patches(image: &[u8]) -> Result<Vec<f32>> {
    ensure!(
        image.len() == IMAGE_SIZE * IMAGE_SIZE * RGB_CHANNELS,
        "image must be 224×224 RGB"
    );
    let mut result = Vec::with_capacity(image.len());
    for patch_y in 0..IMAGE_SIZE / PATCH_SIZE {
        for patch_x in 0..IMAGE_SIZE / PATCH_SIZE {
            for channel in 0..RGB_CHANNELS {
                for dy in 0..PATCH_SIZE {
                    for dx in 0..PATCH_SIZE {
                        let y = patch_y * PATCH_SIZE + dy;
                        let x = patch_x * PATCH_SIZE + dx;
                        let pixel = image[(y * IMAGE_SIZE + x) * RGB_CHANNELS + channel];
                        // Preserve the reference's float32 operation order.
                        result.push(f32::from(pixel) / 255.0 * 2.0 - 1.0);
                    }
                }
            }
        }
    }
    Ok(result)
}

fn validate_actions(values: &[f32], name: &str) -> Result<()> {
    let count = ACTION_HORIZON * ACTION_DIM;
    ensure!(
        values.len() == count,
        "{name} must contain {count} values ([10,32])"
    );
    ensure!(
        values.iter().all(|x| x.is_finite()),
        "{name} must contain only finite values"
    );
    Ok(())
}

#[derive(Deserialize)]
struct NormFile {
    norm_stats: NormStats,
}

#[derive(Deserialize)]
struct NormStats {
    actions: ActionNormalizer,
}

#[derive(Deserialize)]
struct ActionNormalizer {
    q01: [f64; ACTION_OUTPUT_DIM],
    q99: [f64; ACTION_OUTPUT_DIM],
}

impl ActionNormalizer {
    fn from_json(json: &[u8]) -> Result<Self> {
        let file: NormFile =
            serde_json::from_slice(json).context("loading LIBERO action quantiles")?;
        let actions = file.norm_stats.actions;
        for (lo, hi) in actions.q01.iter().zip(actions.q99.iter()) {
            ensure!(
                lo.is_finite() && hi.is_finite() && hi >= lo,
                "invalid action quantile interval"
            );
        }
        Ok(actions)
    }

    fn decode(&self, actions: &[f32]) -> Result<Vec<[f64; ACTION_OUTPUT_DIM]>> {
        validate_actions(actions, "model actions")?;
        let result = actions
            .chunks_exact(ACTION_DIM)
            .map(|row| {
                std::array::from_fn(|i| {
                    // NumPy performs the first two operations in the input's f32
                    // dtype, then promotes when multiplying the f64 statistics.
                    f64::from((row[i] + 1.0) / 2.0) * (self.q99[i] - self.q01[i] + 1e-6)
                        + self.q01[i]
                })
            })
            .collect::<Vec<_>>();
        ensure!(
            result.iter().flatten().all(|x| x.is_finite()),
            "denormalized actions are not finite"
        );
        Ok(result)
    }
}

#[cfg(test)]
mod tests;
