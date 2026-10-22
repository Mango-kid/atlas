// SPDX-License-Identifier: AGPL-3.0-only

//! Shared argument handling for the policy's diagnostic harnesses.

use anyhow::{Result, ensure};
use std::path::PathBuf;

pub struct QuantizationManifests {
    pub nvfp4: Option<PathBuf>,
    pub fp8: Option<PathBuf>,
}

/// Preserve positional requests; explicit precision flags follow in either order.
pub fn take_quantization_manifests(args: &mut Vec<String>) -> Result<QuantizationManifests> {
    let mut manifests = QuantizationManifests {
        nvfp4: None,
        fp8: None,
    };
    let Some(index) = args.iter().position(|arg| arg.starts_with("--")) else {
        return Ok(manifests);
    };
    let tail = &args[index..];
    ensure!(
        tail.len().is_multiple_of(2),
        "precision flags require a file path"
    );
    for pair in tail.chunks_exact(2) {
        let slot = match pair[0].as_str() {
            "--nvfp4-manifest" => &mut manifests.nvfp4,
            "--fp8-manifest" => &mut manifests.fp8,
            _ => anyhow::bail!("unknown trailing precision flag {}", pair[0]),
        };
        ensure!(slot.is_none(), "duplicate precision flag {}", pair[0]);
        ensure!(
            !pair[1].is_empty() && !pair[1].starts_with("--"),
            "{} requires a file path",
            pair[0]
        );
        *slot = Some(PathBuf::from(&pair[1]));
    }
    ensure!(
        manifests.fp8.is_none() || manifests.nvfp4.is_some(),
        "--fp8-manifest requires --nvfp4-manifest"
    );
    args.truncate(index);
    Ok(manifests)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| (*s).into()).collect()
    }

    #[test]
    fn precision_flags_preserve_all_positional_requests() {
        for tail in [
            vec![],
            vec!["--nvfp4-manifest", "nv.json"],
            vec!["--nvfp4-manifest", "nv.json", "--fp8-manifest", "fp.json"],
            vec!["--fp8-manifest", "fp.json", "--nvfp4-manifest", "nv.json"],
        ] {
            let mut args = strings(&["policy_bench", "first.json", "second.json"]);
            let original = args.clone();
            args.extend(strings(&tail));
            let manifests = take_quantization_manifests(&mut args).unwrap();
            assert_eq!(args, original);
            assert_eq!(
                manifests.nvfp4,
                (!tail.is_empty()).then(|| "nv.json".into())
            );
            assert_eq!(manifests.fp8, (tail.len() == 4).then(|| "fp.json".into()));
        }
    }

    #[test]
    fn malformed_precision_flags_leave_arguments_intact() {
        for tail in [
            vec!["--nvfp4-manifest"],
            vec!["--nvfp4-manifest", ""],
            vec!["--nvfp4-manifest", "--unknown"],
            vec!["--nvfp4-manifest", "q.json", "request.json"],
            vec!["--nvfp4-manifest", "q.json", "--nvfp4-manifest", "q2.json"],
            vec!["--fp8-manifest", "fp.json"],
            vec!["--nvfp4-manifest", "nv.json", "--fp8-manifest"],
            vec![
                "--nvfp4-manifest",
                "nv.json",
                "--fp8-manifest",
                "fp.json",
                "--fp8-manifest",
                "fp2.json",
            ],
            vec!["--unknown", "q.json"],
        ] {
            let mut args = strings(&tail);
            let original = args.clone();
            assert!(take_quantization_manifests(&mut args).is_err());
            assert_eq!(args, original);
        }
    }
}
