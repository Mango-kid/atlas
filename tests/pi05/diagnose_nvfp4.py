# SPDX-License-Identifier: AGPL-3.0-only
"""Diagnose calibrated activation clipping and reversible headroom variants."""

import argparse
import dataclasses
import json
from pathlib import Path

import torch
from export_nvfp4 import (
    configs,
    export_references,
    install_attention_mask_dtype_fix,
    policy_config,
    quantize,
    write_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--references", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    config = configs.get_config("pi05_libero")
    config = dataclasses.replace(
        config, model=dataclasses.replace(config.model, pytorch_compile_mode=None)
    )
    policy = policy_config.create_trained_policy(config, str(args.checkpoint))
    install_attention_mask_dtype_fix(policy._model)
    saved = json.loads((args.artifact / "calibrated-scales.json").read_text())
    model, layers = quantize(policy, args.calibration, saved)
    base_amax = {name: module.input_quantizer._amax.clone() for name, module in layers}
    for _, module in layers:
        module.weight_quantizer.disable()
        module.input_quantizer.disable()
    baseline = args.output / "disabled"
    baseline.mkdir()
    export_references(policy, model, layers, args.references, baseline)
    print("Completed quantizers-disabled BF16 control", flush=True)
    for _, module in layers:
        module.weight_quantizer.enable()
        module.input_quantizer.enable()
    for factor in [1, 2, 4, 8]:
        folder = args.output / f"headroom-{factor}"
        folder.mkdir()
        stats, handles = {}, []
        for name, module in layers:
            module.input_quantizer._amax = base_amax[name] * factor
            threshold = float(module.input_quantizer._amax.item())

            def capture(quantizer, inputs, name=name, threshold=threshold, stats=stats):
                x = inputs[0].detach().float().reshape(-1, 16)
                maxima = x.abs().amax(-1)
                row = stats.setdefault(
                    name,
                    {
                        "calls": 0,
                        "blocks": 0,
                        "clipped_blocks": 0,
                        "max_ratio": 0.0,
                        "per_call": [],
                    },
                )
                clipped = int((maxima > threshold).count_nonzero().item())
                ratio = float(maxima.max().item()) / threshold
                row["calls"] += 1
                row["blocks"] += maxima.numel()
                row["clipped_blocks"] += clipped
                row["max_ratio"] = max(row["max_ratio"], ratio)
                row["per_call"].append(
                    {
                        "blocks": maxima.numel(),
                        "clipped_blocks": clipped,
                        "max_ratio": ratio,
                    }
                )

            handles.append(module.input_quantizer.register_forward_pre_hook(capture))
        try:
            export_references(policy, model, layers, args.references, folder)
        finally:
            for handle in handles:
                handle.remove()
        write_json(folder / "activation-clipping.json", stats)
        print(f"Completed activation headroom {factor}", flush=True)


if __name__ == "__main__":
    main()
