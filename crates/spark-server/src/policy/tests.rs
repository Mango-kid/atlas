// SPDX-License-Identifier: AGPL-3.0-only

use super::args::*;
use clap::Parser;

#[derive(Parser)]
struct TestCli {
    #[command(flatten)]
    policy: PolicyArgs,
}

const ASSETS: [&str; 10] = [
    "--checkpoint",
    "model.safetensors",
    "--config",
    "pi05_libero.json",
    "--tokenizer",
    "paligemma_tokenizer.model",
    "--norm-stats",
    "norm_stats.json",
    "--gpu-ordinal",
    "0",
];

#[test]
fn infer_parses_explicit_assets_and_input() {
    let mut argv = vec!["policy", "infer"];
    argv.extend(ASSETS);
    argv.extend(["--input", "observation.json", "--max-body-bytes", "2097152"]);
    let parsed = TestCli::try_parse_from(argv).expect("valid infer command");
    let PolicyCommand::Infer(args) = parsed.policy.command else {
        panic!("expected infer");
    };
    assert_eq!(args.model.gpu_ordinal, 0);
    assert_eq!(args.input.to_string_lossy(), "observation.json");
    assert_eq!(args.max_body_bytes, 2_097_152);
    assert!(args.model.nvfp4_manifest.is_none());
    assert!(args.model.fp8_manifest.is_none());
}

#[test]
fn fp8_manifest_requires_explicit_nvfp4_and_both_paths() {
    let mut argv = vec!["policy", "infer"];
    argv.extend(ASSETS);
    argv.extend(["--input", "observation.json", "--max-body-bytes", "2097152"]);
    argv.extend(["--fp8-manifest", "fp8/manifest.json"]);
    assert!(TestCli::try_parse_from(&argv).is_err());
    argv.push("--nvfp4-manifest");
    assert!(TestCli::try_parse_from(&argv).is_err());
    argv.push("nvfp4/manifest.json");
    let parsed = TestCli::try_parse_from(argv).expect("explicit mixed precision manifests");
    let PolicyCommand::Infer(args) = parsed.policy.command else {
        panic!("expected infer");
    };
    assert_eq!(args.model.fp8_manifest, Some("fp8/manifest.json".into()));
    assert_eq!(
        args.model.nvfp4_manifest,
        Some("nvfp4/manifest.json".into())
    );
}

#[test]
fn serve_requires_bind_address_and_bounded_queue() {
    let mut argv = vec!["policy", "serve"];
    argv.extend(ASSETS);
    assert!(TestCli::try_parse_from(&argv).is_err());
    argv.extend([
        "--host",
        "127.0.0.1",
        "--port",
        "8001",
        "--queue-capacity",
        "2",
        "--max-body-bytes",
        "2097152",
    ]);
    let parsed = TestCli::try_parse_from(&argv).expect("valid serve command");
    let PolicyCommand::Serve(args) = parsed.policy.command else {
        panic!("expected serve");
    };
    assert_eq!(args.host.to_string(), "127.0.0.1");
    assert_eq!(args.port, 8001);
    *argv.last_mut().expect("body limit argument") = "0";
    assert!(TestCli::try_parse_from(&argv).is_err());
}

#[test]
fn missing_model_assets_fail_at_cli_boundary() {
    assert!(TestCli::try_parse_from(["policy", "infer", "--input", "-"]).is_err());
}

#[test]
fn nvfp4_manifest_is_explicit_and_requires_a_path() {
    let mut argv = vec!["policy", "infer"];
    argv.extend(ASSETS);
    argv.extend(["--input", "observation.json", "--max-body-bytes", "2097152"]);
    argv.push("--nvfp4-manifest");
    assert!(TestCli::try_parse_from(&argv).is_err());
    argv.push("quantized/manifest.json");
    let parsed = TestCli::try_parse_from(argv).expect("explicit NVFP4 manifest");
    let PolicyCommand::Infer(args) = parsed.policy.command else {
        panic!("expected infer");
    };
    assert_eq!(
        args.model.nvfp4_manifest,
        Some("quantized/manifest.json".into())
    );
}
