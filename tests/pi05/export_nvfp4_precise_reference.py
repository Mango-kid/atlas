# SPDX-License-Identifier: AGPL-3.0-only
"""Independent OpenPi reference for native FP4/FP8 math, without BF16 QDQ.

The actual packed checkpoint is decoded with PyTorch. Quantized affine operands
stay FP32 until GEMM output conversion; TF32 is disabled only for those GEMMs.
This is an implementation oracle, not a robot-quality or latency benchmark.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import types
from pathlib import Path

os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"

import jax
import numpy as np
import torch
from analyze_nvfp4_quality import metrics as behavior_metrics
from deployment_scripts.trt_model_forward import install_attention_mask_dtype_fix
from export_reference import StageCapture
from fp8_reference import install_fp8_linears
from openpi.models.model import Observation
from openpi.policies import policy_config
from openpi.training import config as configs

PREFIX = "paligemma_with_expert.paligemma.model.language_model.layers."
PROJECTIONS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def decode_weight(payload, entry):
    n, k = entry["shape"]
    payload.seek(entry["data_offset"])
    packed = payload.read(n * k // 2)
    payload.seek(entry["scale_offset"])
    padded_n = ((n + 127) // 128) * 128
    scales = payload.read(padded_n * k // 16)
    assert hashlib.sha256(packed).hexdigest() == entry["packed_sha256"]
    assert hashlib.sha256(scales).hexdigest() == entry["scales_sha256"]
    packed = (
        torch.from_numpy(np.frombuffer(packed, dtype=np.uint8).copy())
        .cuda()
        .reshape(n, k // 2)
    )
    swizzled = np.frombuffer(scales, dtype=np.uint8).reshape(
        padded_n // 128, k // 64, 32, 4, 4
    )
    linear = (
        swizzled.transpose(0, 3, 2, 1, 4).copy().reshape(padded_n, k // 16)[:n].copy()
    )
    scale = torch.from_numpy(linear).cuda().view(torch.float8_e4m3fn).float()
    codes = torch.empty((n, k), device="cuda", dtype=torch.uint8)
    codes[:, ::2], codes[:, 1::2] = packed & 15, packed >> 4
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    decoded = levels[(codes & 7).long()] * torch.where(codes & 8 != 0, -1.0, 1.0)
    decoded *= scale.repeat_interleave(16, dim=1)
    assert torch.isfinite(decoded).all().item()
    return decoded


def decode_activation(x, global_scale):
    original_shape = x.shape
    blocks = x.float().reshape(-1, x.shape[-1] // 16, 16)
    scales = (
        (blocks.abs().amax(-1) / (6 * global_scale))
        .clamp(max=448)
        .to(torch.float8_e4m3fn)
        .float()
    )
    effective = scales * global_scale
    safe = torch.where(effective >= 1e-5, effective, 1.0)
    values = blocks / safe[..., None]
    bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    indices = torch.bucketize(values.abs().contiguous(), bounds)
    for i in [1, 3, 5]:
        indices += (values.abs() == bounds[i]).to(indices.dtype)
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=x.device)
    decoded = levels[indices] * torch.where(torch.signbit(values), -1.0, 1.0)
    decoded *= scales[..., None]
    return decoded.reshape(original_shape)


def install_precise_linears(model, artifact):
    manifest = json.loads((artifact / "manifest.json").read_text())
    expected = {
        f"{PREFIX}{layer}.{projection}.weight"
        for layer in range(18)
        for projection in PROJECTIONS
    }
    assert {entry["name"] for entry in manifest["tensors"]} == expected
    assert len(manifest["tensors"]) == 126
    modules = dict(model.named_modules())
    with (artifact / manifest["weight_file"]).open("rb") as payload:
        for i, entry in enumerate(manifest["tensors"]):
            module = modules[entry["name"].removesuffix(".weight")]
            assert list(module.weight.shape) == entry["shape"] and module.bias is None
            decoded = decode_weight(payload, entry)
            activation_global = float(np.float32(entry["activation_global_scale"]))
            weight_global = float(np.float32(entry["weight_global_scale"]))
            alpha = float(
                np.float32(np.float32(activation_global) * np.float32(weight_global))
            )

            def forward(
                self,
                x,
                decoded=decoded,
                activation_global=activation_global,
                alpha=alpha,
            ):
                activation = decode_activation(x, activation_global)
                previous = torch.backends.cuda.matmul.allow_tf32
                try:
                    torch.backends.cuda.matmul.allow_tf32 = False
                    result = torch.nn.functional.linear(activation, decoded)
                    assert result.dtype == torch.float32
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = previous
                return (result * alpha).to(x.dtype)

            module.forward = types.MethodType(forward, module)
            if (i + 1) % 14 == 0:
                print(f"Installed precise FP4 affine {i + 1}/126", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--fp8-overlay", type=Path)
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
    model = policy._model
    install_attention_mask_dtype_fix(model)
    with torch.no_grad():
        quant_manifest = install_precise_linears(model, args.artifact)
        fp8_manifest = (
            install_fp8_linears(model, args.fp8_overlay, args.artifact)
            if args.fp8_overlay
            else None
        )
        export_loaded(policy, args, quant_manifest, fp8_manifest)


@torch.no_grad()
def export_loaded(policy, args, quant_manifest, fp8_manifest):
    """Export seven fixed cases from an already loaded reference model."""
    model = policy._model
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    paths = sorted(
        path for root in args.references for path in root.glob("*.request.json")
    )
    assert len(paths) == 7
    reports = []
    for path in paths:
        name = path.name.removesuffix(".request.json")
        request = json.loads(path.read_text())
        obs = {
            key: np.asarray(request[key], dtype=np.uint8).reshape(224, 224, 3)
            for key in ["observation/image", "observation/wrist_image"]
        }
        obs["observation/state"] = np.asarray(
            request["observation/state"], dtype=np.float64
        )
        obs["prompt"] = request["prompt"]
        noise = np.asarray(request["noise"], dtype=np.float32).reshape(10, 32)
        host = policy._input_transform(obs)
        data = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).cuda()[None], host)
        tokens = np.asarray(host["tokenized_prompt"])[
            np.asarray(host["tokenized_prompt_mask"], dtype=bool)
        ].tolist()
        capture = StageCapture(model, len(tokens)) if name in args.stage_case else None
        try:
            normalized = (
                model.sample_actions(
                    "cuda",
                    Observation.from_dict(data),
                    noise=torch.from_numpy(noise).cuda()[None],
                    num_steps=10,
                )[0]
                .float()
                .cpu()
                .numpy()
            )
        finally:
            if capture:
                capture.close()
        actions = policy._output_transform(
            {"state": host["state"], "actions": normalized}
        )["actions"]
        assert np.isfinite(actions).all() and np.isfinite(normalized).all()
        assert torch.backends.cuda.matmul.allow_tf32 == original_tf32
        write_json(
            args.output / f"{name}.reference.json",
            {
                "actions": actions.tolist(),
                "normalized_actions": normalized.tolist(),
                "token_ids": tokens,
            },
        )
        (args.output / path.name).write_bytes(path.read_bytes())
        if capture:
            np.savez(args.output / f"{name}.stages.npz", **capture.values)
        original = np.asarray(
            json.loads(path.with_name(f"{name}.reference.json").read_text())["actions"],
            dtype=np.float64,
        )
        x, y = actions.astype(np.float64).ravel(), original.ravel()
        behavior = behavior_metrics(actions, original)
        reports.append(
            {
                "case": name,
                "cosine_vs_bf16": float(x @ y / np.linalg.norm(x) / np.linalg.norm(y)),
                "rmse_vs_bf16": float(np.sqrt(np.mean((x - y) ** 2))),
                "max_abs_vs_bf16": float(np.max(np.abs(x - y))),
                "arm_cosine_vs_bf16": behavior["arm_cosine"],
                "arm_rmse_vs_bf16": behavior["arm_rmse"],
                "gripper_sign_changes": behavior["gripper_sign_changes"],
                "gripper": behavior["gripper"],
                "bf16_gripper": behavior["bf16_gripper"],
            }
        )
        print(
            f"Precise reference {name}: cosine vsBF16 {reports[-1]['cosine_vs_bf16']:.8f}",
            flush=True,
        )
    write_json(args.output / "reference-quality.json", reports)
    write_json(
        args.output / "manifest.json",
        {
            "profile": "pi05_libero",
            "purpose": "Independent native FP4 arithmetic implementation oracle; not robot quality",
            "quant_manifest_sha256": hashlib.sha256(
                (args.artifact / "manifest.json").read_bytes()
            ).hexdigest(),
            "fp8_overlay_sha256": hashlib.sha256(
                (args.fp8_overlay / "manifest.json").read_bytes()
            ).hexdigest()
            if args.fp8_overlay
            else None,
            "fp8_precision": fp8_manifest["precision"] if fp8_manifest else None,
            "camera_batching": "Unchanged OpenPi reference",
            "fp8_activation_normalization": "IEEE F32 tensor/tensor division; no scalar reciprocal multiply",
            "base_checkpoint_sha256": quant_manifest["base_checkpoint_sha256"],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "arithmetic": "Prefix E2M1*E4M3 and optional overlay E4M3 operands stay FP32; GEMM TF32off; F32 global-scale product then optional F32 bias before BF16 output",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": os.environ[
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"
            ],
            "prefix_matmul_allow_tf32": False,
            "other_matmul_allow_tf32": original_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "vision_attention": model.paligemma_with_expert.paligemma.model.vision_tower.config._attn_implementation,
            "cases": [row["case"] for row in reports],
            "stage_cases": args.stage_case,
        },
    )


if __name__ == "__main__":
    main()
