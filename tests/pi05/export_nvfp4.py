# SPDX-License-Identifier: AGPL-3.0-only
"""Calibrate prefix-only W4A4 NVFP4 and export a native Atlas checkpoint.

Requires the pinned OpenPi/ModelOpt reference image. The vision encoder, action
expert, attention, and sensitive FP32 paths retain the BF16 reference behavior.
No TensorRT engine or Python runtime is needed to consume the resulting payload.
"""

import argparse
import dataclasses
import hashlib
import io
import json
from pathlib import Path

import jax
import modelopt
import modelopt.torch.quantization as mtq
import numpy as np
import pyarrow.parquet as pq
import torch
from deployment_scripts.trt_model_forward import install_attention_mask_dtype_fix
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
from nvfp4_format import append_aligned, base_tensor_hashes, swizzle_scales
from openpi.models.model import Observation
from openpi.policies import policy_config
from openpi.training import config as configs
from PIL import Image

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


def prepare(policy, obs, noise):
    host = policy._input_transform(dict(obs))
    data = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to("cuda")[None], host)
    return host, Observation.from_dict(data), torch.from_numpy(noise).to("cuda")[None]


def calibration_rows(root):
    selection = json.loads((root / "selection.json").read_text())
    if len(selection["rows"]) != 32:
        raise ValueError("expected the pinned 32-observation calibration selection")
    for index, item in enumerate(selection["rows"]):
        row = pq.read_table(root / item["file"]).slice(item["row"], 1).to_pylist()[0]
        obs = {
            "observation/image": np.array(
                Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            ),
            "observation/wrist_image": np.array(
                Image.open(io.BytesIO(row["wrist_image"]["bytes"])).convert("RGB")
            ),
            "observation/state": np.asarray(row["state"], dtype=np.float32),
            "prompt": item["task"],
        }
        noise = (
            np.random.default_rng(42 + index)
            .standard_normal((10, 32))
            .astype(np.float32)
        )
        yield obs, noise


def quantize(policy, calibration, saved=None):
    attribute = {
        "num_bits": (2, 1),
        "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        "axis": None,
        "enable": True,
    }
    quant_cfg = {
        "quant_cfg": {
            "default": {"enable": False},
            PREFIX + "*.weight_quantizer": attribute,
            PREFIX + "*.input_quantizer": attribute,
        },
        "algorithm": "max",
    }

    def forward_loop(model):
        if saved is not None:
            modules = dict(model.named_modules())
            expected = {
                f"{PREFIX}{layer}.{projection}.weight"
                for layer in range(18)
                for projection in PROJECTIONS
            }
            if {entry["name"] for entry in saved["tensors"]} != expected or len(
                saved["tensors"]
            ) != 126:
                raise ValueError("saved calibration tensor inventory differs")
            for entry in saved["tensors"]:
                module = modules[entry["name"].removesuffix(".weight")]
                for kind in ["weight", "activation"]:
                    quantizer = (
                        module.weight_quantizer
                        if kind == "weight"
                        else module.input_quantizer
                    )
                    amax = torch.tensor(
                        entry[kind + "_amax"], dtype=torch.float32, device="cuda"
                    )
                    if (
                        amax.numel() != 1
                        or not torch.isfinite(amax).all().item()
                        or not (amax > 0).all().item()
                    ):
                        raise ValueError("invalid saved calibration amax")
                    quantizer._calibrator.collect(amax)
            print("Restored all252 saved calibration amax values", flush=True)
            return
        model.eval()
        with torch.no_grad():
            for index, (obs, noise) in enumerate(calibration_rows(calibration)):
                _, observation, sample_noise = prepare(policy, obs, noise)
                result = model.sample_actions(
                    "cuda", observation, noise=sample_noise, num_steps=10
                )
                if not torch.isfinite(result).all().item():
                    raise ValueError(f"nonfinite calibration actions at row {index}")
                print(f"Calibrated {index + 1}/32", flush=True)

    model = mtq.quantize(policy._model, quant_cfg, forward_loop=forward_loop)
    modules = dict(model.named_modules())
    names = [
        f"{PREFIX}{layer}.{projection}"
        for layer in range(18)
        for projection in PROJECTIONS
    ]
    for name in names:
        module = modules[name]
        if (
            not module.weight_quantizer.is_enabled
            or not module.input_quantizer.is_enabled
        ):
            raise ValueError(f"missing W4A4 quantizer: {name}")
        if module.output_quantizer.is_enabled:
            raise ValueError(f"unexpected output quantizer: {name}")
        if tuple(module.weight_quantizer.num_bits) != (2, 1) or tuple(
            module.input_quantizer.num_bits
        ) != (2, 1):
            raise ValueError(f"incorrect quantization format: {name}")
    enabled = [
        name
        for name, module in modules.items()
        if isinstance(module, TensorQuantizer) and module.is_enabled
    ]
    if len(enabled) != 252 or any(not name.startswith(PREFIX) for name in enabled):
        raise ValueError(
            f"expected exactly 252 enabled prefix quantizers, got {len(enabled)}"
        )
    return model, [(name, modules[name]) for name in names]


def match_fakequant_midpoints(weight, packed, scales, scale_2, oracle):
    """Preserve ModelOpt fakequant midpoint choices without tolerating QDQ error.

    Triton approximate division can move a normalized FP4 midpoint by one F32
    ulp compared with Torch's RN division. Recover the oracle's E2M1 choice,
    but permit changed codes only at those adjacent midpoint boundaries.
    """
    restored = packed.dequantize(
        dtype=torch.bfloat16, scale=scales, double_scale=scale_2, block_sizes={-1: 16}
    )
    difference = (restored.float() - oracle.float()).abs()
    quality = {
        "original_max_abs": float(difference.max().item()),
        "original_different_values": int(torch.count_nonzero(difference).item()),
    }
    if quality["original_different_values"]:
        blocks = weight.float().reshape(*weight.shape[:-1], -1, 16)
        oracle_max = oracle.float().abs().reshape(*blocks.shape).amax(-1)
        recovered_scales = torch.where(
            oracle_max > 0, oracle_max / (6 * scale_2), scales.float()
        ).to(torch.float8_e4m3fn)
        scale_changed = recovered_scales.view(torch.uint8) != scales.view(torch.uint8)
        if scale_changed.any().item():
            raw_scales = blocks.abs().amax(-1) / (6 * scale_2)
            midpoints = (recovered_scales.float() + scales.float()) / 2
            ulps = (
                torch.nextafter(midpoints, torch.full_like(midpoints, float("inf")))
                - midpoints
            )
            adjacent = (
                recovered_scales.view(torch.uint8).to(torch.int16)
                - scales.view(torch.uint8).to(torch.int16)
            ).abs() == 1
            if torch.any(
                scale_changed & ~(adjacent & ((raw_scales - midpoints).abs() <= ulps))
            ).item():
                raise ValueError(
                    "ModelOpt block scale disagreement away from a one-ULP adjacent FP8 midpoint"
                )
            scales.copy_(recovered_scales)
            regenerated, _, _ = NVFP4QTensor.quantize(
                weight,
                16,
                weights_scaling_factor=scales,
                weights_scaling_factor_2=scale_2,
            )
            packed._quantized_data.copy_(regenerated._quantized_data)
        quality["recovered_scale_midpoints"] = int(scale_changed.count_nonzero().item())
        effective = (scales.float() * scale_2).repeat_interleave(16, dim=-1)
        normalized = weight.float().abs() / effective
        thresholds = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=weight.device
        )
        recovered = torch.bucketize(
            (oracle.float().abs() / effective).contiguous(), thresholds
        ).to(torch.uint8)
        original = torch.empty_like(weight, dtype=torch.uint8)
        original[:, ::2] = packed._quantized_data & 7
        original[:, 1::2] = (packed._quantized_data >> 4) & 7
        changed = recovered != original
        midpoint_index = torch.minimum(recovered, original).long().clamp(max=6)
        midpoint = thresholds[midpoint_index]
        ulp = (
            torch.nextafter(midpoint, torch.full_like(midpoint, float("inf")))
            - midpoint
        )
        allowed = (
            (recovered.to(torch.int16) - original.to(torch.int16)).abs() == 1
        ) & ((normalized - midpoint).abs() <= ulp)
        if torch.any(changed & ~allowed).item():
            raise ValueError(
                "ModelOpt packed/fakequant disagreement away from a one-ULP adjacent FP4 midpoint"
            )
        recovered |= (weight < 0).to(torch.uint8) << 3
        packed._quantized_data.copy_(recovered[:, ::2] | (recovered[:, 1::2] << 4))
        quality["recovered_midpoint_codes"] = int(changed.count_nonzero().item())
        restored = packed.dequantize(
            dtype=torch.bfloat16,
            scale=scales,
            double_scale=scale_2,
            block_sizes={-1: 16},
        )
    difference = (restored.float() - oracle.float()).abs()
    quality.update(
        max_abs=float(difference.max().item()),
        different_values=int(difference.count_nonzero().item()),
        values=weight.numel(),
    )
    if quality["different_values"]:
        raise ValueError(
            f"packed weight QDQ differs from independent ModelOpt oracle: {quality}"
        )
    return quality


def export_weights(layers, output):
    entries, quality = [], []
    with (output / "weights.bin.partial").open("wb") as stream, torch.no_grad():
        for index, (name, module) in enumerate(layers):
            weight = module.weight.detach()
            weight_global = NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(
                module.weight_quantizer
            )
            activation_global = NVFP4QTensor.get_activation_scaling_factor(
                module.input_quantizer
            )
            if weight_global.numel() != 1 or activation_global.numel() != 1:
                raise ValueError(f"global scales must be scalars: {name}")
            globals_ = [float(weight_global.item()), float(activation_global.item())]
            if not all(np.isfinite(x) and x > 0 for x in globals_):
                raise ValueError(f"invalid calibrated global scale: {name}")
            packed, scales, scale_2 = NVFP4QTensor.quantize(
                weight, 16, weights_scaling_factor_2=weight_global
            )
            oracle = module.weight_quantizer(weight)
            error = {
                "name": name,
                **match_fakequant_midpoints(weight, packed, scales, scale_2, oracle),
            }
            quality.append(error)
            data = packed._quantized_data.cpu().numpy().tobytes()
            scale_bytes = swizzle_scales(
                scales.view(torch.uint8).cpu().numpy()
            ).tobytes()
            rows, width = weight.shape
            if (
                len(data) != rows * width // 2
                or len(scale_bytes)
                != ((rows + 127) // 128) * 128 * ((width + 63) // 64) * 4
            ):
                raise ValueError(f"incorrect packed payload size: {name}")
            data_offset, packed_sha = append_aligned(stream, data)
            scale_offset, scales_sha = append_aligned(stream, scale_bytes)
            entries.append(
                {
                    "name": name + ".weight",
                    "shape": list(weight.shape),
                    "data_offset": data_offset,
                    "scale_offset": scale_offset,
                    "weight_global_scale": globals_[0],
                    "activation_global_scale": globals_[1],
                    "packed_sha256": packed_sha,
                    "scales_sha256": scales_sha,
                }
            )
            print(
                f"Packed {index + 1}/126 {name}: QDQ max error {error['max_abs']:.6g}",
                flush=True,
            )
    write_json(output / "weight-qdq-check.json", quality)
    return entries


class AffineCapture:
    """Independent ModelOpt activation QDQ and affine outputs for kernel testing."""

    def __init__(self, layers):
        self.arrays, self.handles, self.names = {}, [], []
        wanted = {
            PREFIX + "0.self_attn.q_proj",
            PREFIX + "0.mlp.down_proj",
            PREFIX + "17.self_attn.v_proj",
        }
        for name, module in layers:
            if name not in wanted:
                continue
            index = len(self.names)
            self.names.append(name + ".weight")

            def capture(layer, inputs, output, index=index):
                if f"input_{index}" in self.arrays:
                    return
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])[:8].contiguous()
                qx = layer.input_quantizer(x)
                self.arrays[f"input_{index}"] = x.float().cpu().numpy()
                self.arrays[f"quantized_input_{index}"] = qx.float().cpu().numpy()
                self.arrays[f"output_{index}"] = (
                    output.detach()
                    .reshape(-1, output.shape[-1])[:8]
                    .float()
                    .cpu()
                    .numpy()
                )

            self.handles.append(module.register_forward_hook(capture))

    def close(self):
        for handle in self.handles:
            handle.remove()


def export_references(policy, model, layers, roots, output):
    references = output / "references"
    references.mkdir(exist_ok=True)
    paths = sorted(path for root in roots for path in root.glob("*.request.json"))
    if len(paths) != 7:
        raise ValueError(f"expected seven paired requests, found {len(paths)}")
    reports = []
    for index, path in enumerate(paths):
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
        host, observation, sample_noise = prepare(policy, obs, noise)
        capture = AffineCapture(layers) if index == 0 else None
        try:
            with torch.no_grad():
                normalized = (
                    model.sample_actions(
                        "cuda", observation, noise=sample_noise, num_steps=10
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
        if not np.isfinite(actions).all() or not np.isfinite(normalized).all():
            raise ValueError(f"nonfinite quantized reference: {path.name}")
        tokens = np.asarray(host["tokenized_prompt"])[
            np.asarray(host["tokenized_prompt_mask"], dtype=bool)
        ].tolist()
        name = path.name.removesuffix(".request.json")
        write_json(
            references / f"{name}.reference.json",
            {
                "actions": actions.tolist(),
                "normalized_actions": normalized.tolist(),
                "token_ids": tokens,
            },
        )
        (references / path.name).write_bytes(path.read_bytes())
        original = np.asarray(
            json.loads(path.with_name(f"{name}.reference.json").read_text())["actions"],
            dtype=np.float64,
        )
        x, y = actions.astype(np.float64).ravel(), original.ravel()
        reports.append(
            {
                "case": name,
                "cosine_vs_bf16": float(x @ y / np.linalg.norm(x) / np.linalg.norm(y)),
                "rmse_vs_bf16": float(np.sqrt(np.mean((x - y) ** 2))),
                "max_abs_vs_bf16": float(np.max(np.abs(x - y))),
            }
        )
        if capture:
            np.savez(output / "affine-oracles.npz", **capture.arrays)
            write_json(
                output / "affine-oracles.json",
                {"weight_names": capture.names, "request": path.name},
            )
        print(
            f"Fakequant reference {name}: cosine {reports[-1]['cosine_vs_bf16']:.8f}",
            flush=True,
        )
    write_json(output / "reference-quality.json", reports)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--references", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-calibration", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise FileExistsError(
            "refusing to overwrite a published native quantization artifact"
        )
    torch.manual_seed(42)
    config = configs.get_config("pi05_libero")
    config = dataclasses.replace(
        config, model=dataclasses.replace(config.model, pytorch_compile_mode=None)
    )
    policy = policy_config.create_trained_policy(config, str(args.checkpoint))
    install_attention_mask_dtype_fix(policy._model)
    saved = (
        json.loads(args.reuse_calibration.read_text())
        if args.reuse_calibration
        else None
    )
    if saved and (
        saved["base_checkpoint_sha256"] != args.checkpoint_sha256
        or saved["selection_sha256"]
        != hashlib.sha256(
            (args.calibration / "selection.json").read_bytes()
        ).hexdigest()
    ):
        raise ValueError(
            "saved calibration does not match checkpoint and pinned selection"
        )
    model, layers = quantize(policy, args.calibration, saved)
    write_json(
        args.output / "calibrated-scales.json",
        {
            "base_checkpoint_sha256": args.checkpoint_sha256,
            "selection_sha256": hashlib.sha256(
                (args.calibration / "selection.json").read_bytes()
            ).hexdigest(),
            "tensors": [
                {
                    "name": name + ".weight",
                    "weight_amax": module.weight_quantizer.export_amax()
                    .float()
                    .cpu()
                    .tolist(),
                    "activation_amax": module.input_quantizer.export_amax()
                    .float()
                    .cpu()
                    .tolist(),
                    "weight_global_scale": float(
                        NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(
                            module.weight_quantizer
                        ).item()
                    ),
                    "activation_global_scale": float(
                        NVFP4QTensor.get_activation_scaling_factor(
                            module.input_quantizer
                        ).item()
                    ),
                }
                for name, module in layers
            ],
        },
    )
    entries = export_weights(layers, args.output)
    export_references(policy, model, layers, args.references, args.output)
    print("Hashing retained BF16 base tensors", flush=True)
    base_hashes = base_tensor_hashes(
        args.checkpoint / "model.safetensors", {entry["name"] for entry in entries}
    )
    selection = json.loads((args.calibration / "selection.json").read_text())
    manifest = {
        "format_version": 1,
        "model_type": "pi05",
        "profile": "pi05_libero",
        "precision": "nvfp4-prefix-bf16",
        "weight_file": "weights.bin",
        "base_checkpoint_sha256": args.checkpoint_sha256,
        "base_tensor_sha256": base_hashes,
        "calibration": {
            "dataset": selection["dataset"],
            "revision": selection["revision"],
            "rows": [item["global_index"] for item in selection["rows"]],
            "noise_seeds": list(range(42, 74)),
            "selection_sha256": hashlib.sha256(
                (args.calibration / "selection.json").read_bytes()
            ).hexdigest(),
            "algorithm": "max",
            "activation_global_scale": "calibrated_amax/(6*448)",
            "block_size": 16,
            "block_scales": "E4M3FN dynamic, cuBLAS 128x4 swizzle",
            "packed_values": "E2M1, row-major, low nibble first; ModelOpt fakequant midpoint choices recovered with exact BF16 QDQ gate",
            "weight_midpoint_note": "ModelOpt Triton approximate division can choose adjacent FP8 block scales or FP4 values at midpoints. Recovery requires distance <=1 F32 ULP at each boundary and exact reconstruction of every BF16 fakequant weight.",
            "modelopt_version": modelopt.__version__,
            "torch_version": torch.__version__,
            "vision_attention": model.paligemma_with_expert.paligemma.model.vision_tower.config._attn_implementation,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "note": "Prefix linear W4A4 only; vision/expert/attention remain BF16. Numerical agreement is not robot success.",
        },
        "tensors": entries,
    }
    write_json(args.output / "manifest.json.partial", manifest)
    (args.output / "weights.bin.partial").replace(args.output / "weights.bin")
    (args.output / "manifest.json.partial").replace(args.output / "manifest.json")
    print(
        f"Published {len(entries)} calibrated W4A4 tensors and {len(base_hashes)} base hashes",
        flush=True,
    )


if __name__ == "__main__":
    main()
