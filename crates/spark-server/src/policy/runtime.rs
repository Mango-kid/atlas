// SPDX-License-Identifier: AGPL-3.0-only

use std::{io::Read, net::SocketAddr};

use anyhow::{Context, Result, ensure};
use spark_pi05::{NativePolicy, PolicyResponse, preprocess::PolicyRequest};
use tokio::sync::{mpsc, oneshot};

use super::{args, http};

pub(super) struct Job {
    pub request: PolicyRequest,
    pub reply: oneshot::Sender<Result<PolicyResponse>>,
}

fn load(model: &args::ModelArgs) -> Result<NativePolicy> {
    NativePolicy::load_with_quantization(
        &model.checkpoint,
        &model.config,
        &model.tokenizer,
        &model.norm_stats,
        model.gpu_ordinal,
        model.nvfp4_manifest.as_deref(),
        model.fp8_manifest.as_deref(),
    )
}

/// File/stdin I/O boundary. The same byte bound applies to either input source.
fn read_request(args: &args::InferArgs) -> Result<PolicyRequest> {
    let reader: Box<dyn Read> = if args.input.as_os_str() == "-" {
        Box::new(std::io::stdin())
    } else {
        Box::new(std::fs::File::open(&args.input).context("opening policy request")?)
    };
    let mut bytes = Vec::new();
    reader
        .take(args.max_body_bytes as u64 + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= args.max_body_bytes,
        "policy request exceeds --max-body-bytes"
    );
    let request: PolicyRequest =
        serde_json::from_slice(&bytes).context("parsing policy request JSON")?;
    request.validate()?;
    Ok(request)
}

pub(super) async fn infer(args: args::InferArgs) -> Result<()> {
    tokio::task::spawn_blocking(move || {
        let request = read_request(&args)?;
        let mut policy = load(&args.model)?;
        let response = policy.infer(&request)?;
        // Logging is directed to stderr by the policy command's entry point.
        serde_json::to_writer(std::io::stdout().lock(), &response)?;
        println!();
        Ok(())
    })
    .await
    .context("policy inference worker failed")?
}

pub(super) async fn serve(args: args::ServeArgs) -> Result<()> {
    let address = SocketAddr::new(args.host, args.port);
    // Bind before expensive GPU loading, so a port conflict fails immediately.
    let listener = tokio::net::TcpListener::bind(address).await?;
    let (sender, mut receiver) = mpsc::channel::<Job>(args.queue_capacity);
    let (ready_tx, ready_rx) = oneshot::channel::<Result<&'static str>>();
    let worker = std::thread::Builder::new()
        .name("atlas-pi05-policy".into())
        .spawn(move || {
            let mut policy = match load(&args.model) {
                Ok(policy) => policy,
                Err(error) => {
                    let _ = ready_tx.send(Err(error));
                    return;
                }
            };
            if ready_tx.send(Ok(policy.backend())).is_err() {
                return;
            }
            // This thread owns the model for its complete lifetime, including
            // destruction. GPU workspaces can never be used concurrently.
            while let Some(job) = receiver.blocking_recv() {
                if !job.reply.is_closed() {
                    let _ = job.reply.send(policy.infer(&job.request));
                }
            }
        })?;
    let backend = ready_rx
        .await
        .context("policy worker exited while loading")??;
    tracing::info!(%address, backend, "Atlas native pi0.5 policy ready at /v1/actions");
    let app = http::router(sender, args.max_body_bytes, backend);
    let serving = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown())
        .await;
    // The router has released its sender. Drain work and destroy GPU resources
    // on the owning thread without blocking the async runtime.
    tokio::task::spawn_blocking(move || {
        worker
            .join()
            .map_err(|_| anyhow::anyhow!("policy worker panicked"))
    })
    .await??;
    serving.context("policy HTTP server failed")
}

async fn shutdown() {
    #[cfg(unix)]
    {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {},
                    _ = term.recv() => {},
                }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
