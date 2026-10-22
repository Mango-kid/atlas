// SPDX-License-Identifier: AGPL-3.0-only

//! Native vision-language-action entry point, separate from text generation.

pub mod args;

#[cfg(feature = "cuda")]
mod http;
#[cfg(feature = "cuda")]
mod runtime;

pub async fn dispatch(args: args::PolicyArgs) -> anyhow::Result<()> {
    #[cfg(feature = "cuda")]
    {
        match args.command {
            args::PolicyCommand::Infer(args) => runtime::infer(args).await,
            args::PolicyCommand::Serve(args) => runtime::serve(args).await,
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = args;
        anyhow::bail!("native pi0.5 policy requires Atlas built with the cuda feature")
    }
}

#[cfg(test)]
mod tests;
