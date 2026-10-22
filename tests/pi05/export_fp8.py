# SPDX-License-Identifier: AGPL-3.0-only
"""Calibrate FP8 vision/expert overlays bound to an existing NVFP4 checkpoint.

This is offline tooling only. Calibration uses the original BF16 model and all
10 denoising steps. Reference affine math decodes E4M3 to F32, multiplies with
TF32 disabled, applies the F32 scale product and bias, then rounds once to BF16.
"""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path

os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"

import numpy as np
import torch
from deployment_scripts.trt_model_forward import install_attention_mask_dtype_fix
from export_nvfp4 import calibration_rows, prepare, write_json
from fp8_format import (
    EXPERT,
    PALIGEMMA,
    bound_source_hashes,
    tensor_names,
    verify_source,
)
from nvfp4_format import append_aligned
from openpi.policies import policy_config
from openpi.training import config as configs


def raw_bytes(tensor):
    return tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()


def fp8_scale(amax):
    scale = torch.tensor(amax, device="cuda", dtype=torch.float32) / 448
    if not torch.isfinite(scale).item() or scale.item() <= 0:
        raise ValueError("FP8 requires a finite positive calibrated amax")
    return float(scale.item())


def encode(value, scale):
    return (value.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)


def fixture_names():
    vision = PALIGEMMA + "vision_tower.vision_model.encoder.layers.0."
    return [
        vision + name + ".weight" for name in ["self_attn.q_proj", "mlp.fc1", "mlp.fc2"]
    ] + [
        PALIGEMMA + "multi_modal_projector.linear.weight",
        EXPERT + "0.self_attn.q_proj.weight",
        EXPERT + "0.mlp.gate_proj.weight",
        EXPERT + "0.mlp.down_proj.weight",
        EXPERT + "17.self_attn.v_proj.weight",
    ]


def calibrate(policy, layers, calibration, output):
    stats = {name: {"activation_amax": 0.0, "calls": 0} for name, _ in layers}
    captures, handles = {}, []
    for name, module in layers:

        def capture(layer, inputs, name=name):
            x = inputs[0].detach()
            if x.dtype != torch.bfloat16:
                raise ValueError(f"expected BF16 projection input: {name}: {x.dtype}")
            maximum = float(x.float().abs().max().item())
            if not np.isfinite(maximum):
                raise ValueError(f"nonfinite calibration input: {name}")
            stats[name]["activation_amax"] = max(
                stats[name]["activation_amax"], maximum
            )
            stats[name]["calls"] += 1
            if name in fixture_names() and name not in captures:
                captures[name] = x.reshape(-1, x.shape[-1]).clone()

        handles.append(module.register_forward_pre_hook(capture))
    try:
        with torch.no_grad():
            for index, (obs, noise) in enumerate(calibration_rows(calibration)):
                _, observation, sample_noise = prepare(policy, obs, noise)
                actions = policy._model.sample_actions(
                    "cuda", observation, noise=sample_noise, num_steps=10
                )
                if not torch.isfinite(actions).all().item():
                    raise ValueError(f"nonfinite BF16 calibration action row {index}")
                print(f"FP8 calibration {index + 1}/32", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    for name, module in layers:
        stats[name]["weight_amax"] = float(
            module.weight.detach().float().abs().max().item()
        )
        stats[name]["weight_scale"] = fp8_scale(stats[name]["weight_amax"])
        stats[name]["activation_scale"] = fp8_scale(stats[name]["activation_amax"])
        if stats[name]["calls"] == 0:
            raise ValueError(f"uncalibrated projection: {name}")
    write_json(output / "calibrated-scales.json", stats)
    return stats, captures


def export_fixture(root, index, name, module, weight, entry, x):
    folder = root / f"affine-{index}"
    folder.mkdir()
    divisor = torch.tensor(
        entry["activation_scale"], device=x.device, dtype=torch.float32
    )
    activation = encode(x, divisor)
    alpha = float(
        np.float32(
            np.float32(entry["weight_scale"]) * np.float32(entry["activation_scale"])
        )
    )
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        expected = (
            torch.nn.functional.linear(activation.float(), weight.float()) * alpha
        )
        if module.bias is not None:
            expected = expected + module.bias.float()
        expected = expected.bfloat16()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    for filename, value in [
        ("input.bf16", x),
        ("activation.fp8", activation),
        ("weight.fp8", weight),
        ("expected.bf16", expected),
    ]:
        (folder / filename).write_bytes(raw_bytes(value))
    if module.bias is not None:
        (folder / "bias.bf16").write_bytes(raw_bytes(module.bias))
    m, k = x.shape
    n = weight.shape[0]
    return {
        "name": name,
        "path": folder.name,
        "m": m,
        "n": n,
        "k": k,
        "weight_scale": entry["weight_scale"],
        "activation_scale": entry["activation_scale"],
        "bias": module.bias is not None,
        "source": "First call on pinned calibration row0, noise seed42",
        "arithmetic": "F32 decoded E4M3 GEMM TF32off, F32 alpha product, optional F32 bias, one BF16 round",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--nvfp4", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source_bytes = (args.nvfp4 / "manifest.json").read_bytes()
    source = json.loads(source_bytes)
    names = tensor_names("vision-expert")
    hashes = bound_source_hashes(source, names)
    torch.manual_seed(42)
    config = configs.get_config("pi05_libero")
    config = dataclasses.replace(
        config, model=dataclasses.replace(config.model, pytorch_compile_mode=None)
    )
    policy = policy_config.create_trained_policy(config, str(args.checkpoint))
    install_attention_mask_dtype_fix(policy._model)
    modules = dict(policy._model.named_modules())
    layers = [(name, modules[name.removesuffix(".weight")]) for name in names]
    if any(
        not isinstance(module, torch.nn.Linear) or module.weight.dtype != torch.bfloat16
        for _, module in layers
    ):
        raise ValueError("overlay inventory must resolve to BF16 Linear weights")
    stats, captures = calibrate(policy, layers, args.calibration, args.output)
    streams, entries = {}, {"vision": [], "expert": [], "vision-expert": []}
    for scope in entries:
        folder = args.output / scope
        folder.mkdir()
        streams[scope] = (folder / "weights.bin.partial").open("wb")
    fixtures = []
    try:
        with torch.no_grad():
            for index, (name, module) in enumerate(layers):
                verify_source(raw_bytes(module.weight), hashes[name], name)
                weight = encode(module.weight.detach(), stats[name]["weight_scale"])
                data = raw_bytes(weight)
                entry = {
                    "name": name,
                    "shape": list(module.weight.shape),
                    "weight_scale": stats[name]["weight_scale"],
                    "activation_scale": stats[name]["activation_scale"],
                }
                for scope, scope_entries in entries.items():
                    if (scope == "vision" and index >= 163) or (
                        scope == "expert" and index < 163
                    ):
                        continue
                    offset, digest = append_aligned(streams[scope], data)
                    scope_entries.append(
                        {**entry, "data_offset": offset, "packed_sha256": digest}
                    )
                if name in captures:
                    fixtures.append(
                        export_fixture(
                            args.output,
                            len(fixtures),
                            name,
                            module,
                            weight,
                            entry,
                            captures[name],
                        )
                    )
                if (index + 1) % 20 == 0:
                    print(f"Exported FP8 {index + 1}/289", flush=True)
    finally:
        for stream in streams.values():
            stream.close()
    write_json(args.output / "affine-cases.json", fixtures)
    selection = json.loads((args.calibration / "selection.json").read_text())
    for scope, tensors in entries.items():
        manifest = {
            "format_version": 1,
            "model_type": "pi05",
            "profile": "pi05_libero",
            "precision": "fp8-" + scope + "-bf16",
            "weight_file": "weights.bin",
            "base_checkpoint_sha256": source["base_checkpoint_sha256"],
            "nvfp4_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "base_tensor_sha256": bound_source_hashes(
                source, [entry["name"] for entry in tensors]
            ),
            "calibration": {
                "dataset": selection["dataset"],
                "revision": selection["revision"],
                "rows": [row["global_index"] for row in selection["rows"]],
                "noise_seeds": list(range(42, 74)),
                "selection_sha256": hashlib.sha256(
                    (args.calibration / "selection.json").read_bytes()
                ).hexdigest(),
                "algorithm": "max on unquantized BF16 model; full10-step denoising",
                "encoding": "Row-major E4M3FN RN, saturated[-448,448]",
                "scale_convention": "F32 dequant multiplier amax/448; static weight and activation scales",
                "bias": "Retain source BF16; add in F32 after GEMM scaling before BF16 output",
                "torch_version": torch.__version__,
                "scope": "Native experimental overlay; FP8 attention and FP16 residual changes excluded",
            },
            "tensors": tensors,
        }
        folder = args.output / scope
        write_json(folder / "manifest.json.partial", manifest)
        (folder / "weights.bin.partial").replace(folder / "weights.bin")
        (folder / "manifest.json.partial").replace(folder / "manifest.json")
        print(f"Published {scope}: {len(tensors)} FP8 weights", flush=True)


if __name__ == "__main__":
    main()
