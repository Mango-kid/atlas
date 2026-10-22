// SPDX-License-Identifier: AGPL-3.0-only

//! Explicit checkpoint profile. Other pi0/pi0.5 variants fail before GPU allocation.

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

pub const ACTION_DIM: usize = 32;
pub const ACTION_HORIZON: usize = 10;
pub const MAX_PROMPT_TOKENS: usize = 200;
pub const IMAGE_SIZE: usize = 224;
pub const PATCH_SIZE: usize = 14;
pub const IMAGE_COUNT: usize = 2;
pub const STATE_DIM: usize = 8;
pub const ACTION_OUTPUT_DIM: usize = 7;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub model_type: String,
    pub profile: String,
    pub action_dim: usize,
    pub action_horizon: usize,
    pub num_steps: usize,
    pub max_token_len: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub vision_hidden: usize,
    pub vision_intermediate: usize,
    pub vision_layers: usize,
    pub vision_heads: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub expert_hidden: usize,
    pub expert_intermediate: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub layer_norm_eps: f32,
    pub rope_theta: f32,
    pub discrete_state_input: bool,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        let supported: Self = serde_json::from_str(include_str!("../pi05_libero.json"))?;
        ensure!(
            self == &supported,
            "unsupported pi0.5 profile; this build requires the exact pi05_libero architecture and ten denoising steps"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_wrong_profile_or_incomplete_configuration() {
        let mut c: Config = serde_json::from_str(include_str!("../pi05_libero.json")).unwrap();
        c.validate().unwrap();
        c.discrete_state_input = true;
        assert!(c.validate().is_err());
        assert!(serde_json::from_str::<Config>("{}").is_err());
    }
}
