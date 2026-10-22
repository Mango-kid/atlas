# SPDX-License-Identifier: AGPL-3.0-only
"""Export expert-only and vision/expert FP8 policy oracles with one model load."""

import argparse
import dataclasses
from pathlib import Path

import torch
from export_nvfp4_precise_reference import (
    configs,
    export_loaded,
    install_attention_mask_dtype_fix,
    install_fp8_linears,
    install_precise_linears,
    policy_config,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--overlays", type=Path, required=True)
    parser.add_argument("--references", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage-case", action="append", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    config = configs.get_config("pi05_libero")
    config = dataclasses.replace(
        config, model=dataclasses.replace(config.model, pytorch_compile_mode=None)
    )
    policy = policy_config.create_trained_policy(config, str(args.checkpoint))
    install_attention_mask_dtype_fix(policy._model)
    with torch.no_grad():
        quant_manifest = install_precise_linears(policy._model, args.artifact)
        # Expert -> vision+expert adds vision overrides; no old override needs removal.
        for scope in ["expert", "vision-expert"]:
            run = argparse.Namespace(**vars(args))
            run.output = args.output / scope
            run.output.mkdir()
            run.fp8_overlay = args.overlays / scope
            fp8_manifest = install_fp8_linears(
                policy._model, run.fp8_overlay, args.artifact
            )
            export_loaded(policy, run, quant_manifest, fp8_manifest)
            print(f"Completed precise FP8 scope {scope}", flush=True)


if __name__ == "__main__":
    main()
