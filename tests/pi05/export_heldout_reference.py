# SPDX-License-Identifier: AGPL-3.0-only
"""Paired BF16 and exact packed FP4/FP8 arithmetic on unseen LIBERO episodes."""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path

# Pin before any reference framework initializes CUDA.
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"

import jax
import numpy as np
import torch
from analyze_nvfp4_quality import metrics
from deployment_scripts.trt_model_forward import install_attention_mask_dtype_fix
from export_nvfp4_precise_reference import install_precise_linears, write_json
from fp8_reference import install_fp8_linears
from openpi.models.model import Observation
from openpi.policies import policy_config
from openpi.training import config as configs


@torch.no_grad()
def evaluate(policy, paths, output):
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for path in paths:
        request = json.loads(path.read_text())
        name = path.name.removesuffix(".request.json")
        obs = {
            key: np.asarray(request[key], dtype=np.uint8).reshape(224, 224, 3)
            for key in ["observation/image", "observation/wrist_image"]
        }
        obs["observation/state"] = np.asarray(
            request["observation/state"], dtype=np.float64
        )
        obs["prompt"] = request["prompt"]
        host = policy._input_transform(obs)
        data = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).cuda()[None], host)
        noise = torch.tensor(
            request["noise"], dtype=torch.float32, device="cuda"
        ).reshape(1, 10, 32)
        normalized = (
            policy._model.sample_actions(
                "cuda", Observation.from_dict(data), noise=noise, num_steps=10
            )[0]
            .float()
            .cpu()
            .numpy()
        )
        actions = policy._output_transform(
            {"state": host["state"], "actions": normalized}
        )["actions"]
        assert np.isfinite(actions).all() and np.isfinite(normalized).all()
        tokens = np.asarray(host["tokenized_prompt"])[
            np.asarray(host["tokenized_prompt_mask"], dtype=bool)
        ].tolist()
        write_json(
            output / f"{name}.reference.json",
            {
                "actions": actions.tolist(),
                "normalized_actions": normalized.tolist(),
                "token_ids": tokens,
            },
        )
        results[name] = actions
        print(f"{output.name}: {name}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    for flag in ["checkpoint", "artifact", "fp8-overlay", "requests"]:
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads((args.requests / "selection.json").read_text())
    assert not {row["episode"] for row in selection["rows"]}.intersection(
        selection["excluded_calibration_episodes"]
    )
    paths = [args.requests / f"{row['case']}.request.json" for row in selection["rows"]]
    for path, row in zip(paths, selection["rows"], strict=True):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["request_sha256"]
    torch.manual_seed(42)
    config = configs.get_config("pi05_libero")
    config = dataclasses.replace(
        config, model=dataclasses.replace(config.model, pytorch_compile_mode=None)
    )
    policy = policy_config.create_trained_policy(config, str(args.checkpoint))
    install_attention_mask_dtype_fix(policy._model)
    with torch.no_grad():
        bf16 = evaluate(policy, paths, args.requests / "bf16")
        manifest = install_precise_linears(policy._model, args.artifact)
        overlay = install_fp8_linears(policy._model, args.fp8_overlay, args.artifact)
        assert overlay["precision"] == "fp8-expert-bf16"
        precise = evaluate(policy, paths, args.requests / "precise-expert")
    quality = [{"case": name, **metrics(precise[name], bf16[name])} for name in bf16]
    write_json(args.requests / "reference-quality.json", quality)
    write_json(
        args.requests / "reference-manifest.json",
        {
            "purpose": "Implementation and output-drift diagnostics on episodes excluded from calibration; not a robot success evaluation",
            "base_checkpoint_sha256": manifest["base_checkpoint_sha256"],
            "quant_manifest_sha256": hashlib.sha256(
                (args.artifact / "manifest.json").read_bytes()
            ).hexdigest(),
            "fp8_overlay_sha256": hashlib.sha256(
                (args.fp8_overlay / "manifest.json").read_bytes()
            ).hexdigest(),
            "selection_sha256": hashlib.sha256(
                (args.requests / "selection.json").read_bytes()
            ).hexdigest(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "denoising_steps": 10,
            "cases": len(paths),
            "arithmetic": "BF16 reference unchanged; prefix actual packed E2M1*E4M3, expert actual packed E4M3; activation precise tensor division; F32 GEMM TF32off and scale product before BF16 output",
        },
    )


if __name__ == "__main__":
    main()
