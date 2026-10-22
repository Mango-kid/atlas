// SPDX-License-Identifier: AGPL-3.0-only

use std::{net::IpAddr, path::PathBuf};

#[derive(clap::Args, Debug)]
pub struct PolicyArgs {
    #[command(subcommand)]
    pub command: PolicyCommand,
}

#[derive(clap::Subcommand, Debug)]
pub enum PolicyCommand {
    /// Infer one pi0.5 action chunk from a JSON observation; print JSON to stdout.
    Infer(InferArgs),
    /// Serve native pi0.5 actions over HTTP with one serialized GPU worker.
    Serve(ServeArgs),
}

#[derive(clap::Args, Debug, Clone)]
pub struct ModelArgs {
    /// Converted OpenPi BF16 model.safetensors file.
    #[arg(long)]
    pub checkpoint: PathBuf,
    /// Explicit NVFP4 prefix manifest; omitted preserves the BF16 checkpoint path.
    #[arg(long)]
    pub nvfp4_manifest: Option<PathBuf>,
    /// Explicit FP8 vision/expert overlay; requires an NVFP4 prefix manifest.
    #[arg(long, requires = "nvfp4_manifest")]
    pub fp8_manifest: Option<PathBuf>,
    /// Explicit Atlas pi05_libero architecture JSON.
    #[arg(long)]
    pub config: PathBuf,
    /// Original PaliGemma SentencePiece .model file.
    #[arg(long)]
    pub tokenizer: PathBuf,
    /// LIBERO norm_stats.json file shipped with the checkpoint.
    #[arg(long)]
    pub norm_stats: PathBuf,
    #[arg(long)]
    pub gpu_ordinal: usize,
}

#[derive(clap::Args, Debug)]
pub struct InferArgs {
    #[command(flatten)]
    pub model: ModelArgs,
    /// JSON request file, or '-' to read standard input.
    #[arg(long)]
    pub input: PathBuf,
    /// Maximum request size in bytes, including image arrays.
    #[arg(long, value_parser = positive_usize)]
    pub max_body_bytes: usize,
}

#[derive(clap::Args, Debug)]
pub struct ServeArgs {
    #[command(flatten)]
    pub model: ModelArgs,
    /// Explicit listening IP address.
    #[arg(long)]
    pub host: IpAddr,
    #[arg(long, value_parser = clap::value_parser!(u16).range(1..))]
    pub port: u16,
    /// Maximum waiting inference requests; excess requests receive HTTP 429.
    #[arg(long, value_parser = queue_capacity)]
    pub queue_capacity: usize,
    /// Maximum JSON request size in bytes, including image arrays.
    #[arg(long, value_parser = positive_usize)]
    pub max_body_bytes: usize,
}

fn positive_usize(value: &str) -> Result<usize, String> {
    match value.parse::<usize>() {
        Ok(n) if n > 0 && n < usize::MAX => Ok(n),
        _ => Err("must be a positive integer smaller than usize::MAX".into()),
    }
}

fn queue_capacity(value: &str) -> Result<usize, String> {
    let n = positive_usize(value)?;
    if n > tokio::sync::Semaphore::MAX_PERMITS {
        return Err("queue capacity exceeds the channel limit".into());
    }
    Ok(n)
}
