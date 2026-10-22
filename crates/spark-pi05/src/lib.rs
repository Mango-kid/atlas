// SPDX-License-Identifier: AGPL-3.0-only

//! Native pi0.5 policy inference on Atlas's CUDA backend.
//! The policy returns continuous actions, independently of the chat scheduler.

pub mod config;
#[cfg(feature = "cuda")]
pub mod model;
#[cfg(feature = "cuda")]
pub mod ops;
#[cfg(feature = "cuda")]
pub mod ops_attention;
#[cfg(feature = "cuda")]
pub mod ops_fp8;
#[cfg(feature = "cuda")]
pub mod ops_nvfp4;
#[cfg(feature = "cuda")]
mod policy;
pub mod preprocess;
#[cfg(feature = "cuda")]
pub mod weights;
#[cfg(feature = "cuda")]
pub use policy::{NativePolicy, PolicyResponse};
