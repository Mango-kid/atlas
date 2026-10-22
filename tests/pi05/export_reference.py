# SPDX-License-Identifier: AGPL-3.0-only
"""Export deterministic OpenPi fixtures for native Atlas pi05_libero validation.

Run inside the pinned OpenPi reference image with its source on PYTHONPATH.
JSON requests contain flattened RGB bytes and explicit [10,32] Gaussian noise;
no model runtime is required to consume those requests in Atlas.
"""

import argparse
import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch

from deployment_scripts.trt_model_forward import install_attention_mask_dtype_fix
from openpi.policies import policy_config
from openpi.shared.image_tools import resize_with_pad
from openpi.training import config as configs


def observations(calibration):
    prompts = ["do something", "pick_up the red cup\nand place it on the plate", "open the drawer"]
    for seed, prompt in zip([42, 7, 123], prompts, strict=True):
        rng = np.random.default_rng(seed)
        yield f"synthetic_{seed}", {
            "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/state": rng.standard_normal(8).astype(np.float32),
            "prompt": prompt,
        }, seed
    selected = json.loads((calibration / "selection.json").read_text())["rows"]
    for index in [0, 15, 31]:
        item = selected[index]
        row = pq.read_table(calibration / item["file"]).slice(item["row"], 1).to_pylist()[0]
        yield f"real_{item['global_index']}", {
            "observation/image": np.array(Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")),
            "observation/wrist_image": np.array(Image.open(io.BytesIO(row["wrist_image"]["bytes"])).convert("RGB")),
            "observation/state": np.asarray(row["state"], dtype=np.float32),
            "prompt": item["task"],
        }, 42


def request_json(obs, noise):
    return {
        "observation/image": obs["observation/image"].ravel().tolist(),
        "observation/wrist_image": obs["observation/wrist_image"].ravel().tolist(),
        "observation/state": obs["observation/state"].tolist(),
        "prompt": obs["prompt"],
        "noise": noise.ravel().tolist(),
    }


class StageCapture:
    """Optional eager-only module snapshots; keeps native/reference rows aligned."""

    def __init__(self, model, token_count):
        self.values = {}
        self.handles = []
        self.counts = {}
        self.valid_rows = np.r_[0:512, 768:768 + token_count]
        root = model.paligemma_with_expert
        vision = root.paligemma.model.vision_tower.vision_model
        self.hook(vision.embeddings, "image.{}.embedding", image=True, cast_bf16=True)
        for layer in [0, 26]:
            self.hook(vision.encoder.layers[layer], "image.{}.layer." + str(layer), image=True)
        self.hook(root.paligemma.model.multi_modal_projector, "image.{}.projected", image=True)
        self.hook(root.paligemma.language_model.layers[0], "prefix.0.hidden", prefix=True)
        self.hook(root.gemma_expert.model.norm, "step.{}.normalized")
        self.hook(model.action_in_proj, "step.{}.embedding", cast_bf16=True)
        self.hook(model.action_out_proj, "step.{}.velocity")

        self.original_embed_prefix = model.embed_prefix

        def embed_prefix(*args, **kwargs):
            result = self.original_embed_prefix(*args, **kwargs)
            value = result[0].detach().float().cpu().numpy()[0]
            self.values["prefix.embedding"] = value[self.valid_rows].copy()
            return result

        model.embed_prefix = embed_prefix
        self.original_denoise = model.denoise_step
        self.model = model
        self.step = 0

        def denoise(*args, **kwargs):
            if "prefix.0.key" not in self.values:
                cache = args[2]
                for layer in [0, 17]:
                    for kind, tensor in zip(["key", "value"], cache[layer], strict=True):
                        value = tensor.detach().float().cpu().numpy()[0].transpose(1, 0, 2)
                        self.values[f"prefix.{layer}.{kind}"] = value[self.valid_rows].copy()
            velocity = self.original_denoise(*args, **kwargs)
            actions = args[3] + torch.tensor(-0.1, dtype=torch.float32, device=velocity.device) * velocity
            self.values[f"step.{self.step}.actions"] = actions.detach().float().cpu().numpy()[0].copy()
            self.step += 1
            return velocity

        model.denoise_step = denoise

    def hook(self, module, name, image=False, prefix=False, cast_bf16=False):
        def capture(_module, _args, output):
            index = self.counts.get(name, 0)
            self.counts[name] = index + 1
            if image and index >= 2:
                return
            tensor = output[0] if isinstance(output, tuple) else output
            if cast_bf16:
                tensor = tensor.to(torch.bfloat16)
            value = tensor.detach().float().cpu().numpy()[0]
            if prefix:
                value = value[self.valid_rows]
            self.values[name.format(index)] = value.copy()
        self.handles.append(module.register_forward_hook(capture))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.model.denoise_step = self.original_denoise
        self.model.embed_prefix = self.original_embed_prefix


def write_json(path, value):
    path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True, help="SHA256 of the exact converted model.safetensors.")
    parser.add_argument("--compiled", action="store_true", help="Use upstream max-autotune inference.")
    parser.add_argument("--stages", action="store_true", help="Save eager first-case internal tensors as NPZ.")
    parser.add_argument("--vision-attention", choices=["eager", "sdpa"], help="Diagnostic override; omitted preserves the upstream setting.")
    parser.add_argument("--disable-tf32", action="store_true", help="Diagnostic full-FP32 matrix multiplication and convolution.")
    parser.add_argument("--case", action="append", dest="cases", help="Export only this named case; repeat to select multiple cases.")
    args = parser.parse_args()
    if args.compiled and args.stages:
        parser.error("stage hooks require eager execution")
    cases = list(observations(args.calibration))
    if args.cases:
        unknown = set(args.cases) - {name for name, _, _ in cases}
        if unknown:
            parser.error(f"unknown cases: {sorted(unknown)}")
        cases = [case for case in cases if case[0] in args.cases]
    if args.disable_tf32:
        os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    config = configs.get_config("pi05_libero")
    if not args.compiled:
        config = dataclasses.replace(config, model=dataclasses.replace(config.model, pytorch_compile_mode=None))
    policy = policy_config.create_trained_policy(config, str(args.checkpoint))
    install_attention_mask_dtype_fix(policy._model)
    vision = policy._model.paligemma_with_expert.paligemma.model.vision_tower
    if args.vision_attention is not None:
        vision.config._attn_implementation = args.vision_attention
    if args.disable_tf32:
        # PI0Pytorch.__init__ sets precision="high", so apply this afterwards.
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    sample_actions = policy._sample_actions
    normalized = []

    def capture_actions(*a, **kw):
        result = sample_actions(*a, **kw)
        normalized.append(result.detach().float().cpu().numpy()[0].copy())
        return result

    policy._sample_actions = capture_actions
    manifest = {
        "profile": "pi05_libero", "mode": "compiled" if args.compiled else "eager",
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "checkpoint_sha256": args.checkpoint_sha256,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "vision_attention": policy._model.paligemma_with_expert.paligemma.model.vision_tower.config._attn_implementation,
        "diagnostic_overrides": {"vision_attention": args.vision_attention, "disable_tf32": args.disable_tf32},
        "num_steps": 10, "action_horizon": 10, "action_dim": 32,
        "note": "Numerical fixtures; real inputs are from the calibration selection, not robot success evaluation.",
        "cases": [],
    }
    for index, (name, obs, seed) in enumerate(cases):
        # LIBERO stores 256x256 images. Canonicalize with OpenPi's own transform
        # before creating the explicit 224x224 native API request.
        for key in ["observation/image", "observation/wrist_image"]:
            obs[key] = np.asarray(resize_with_pad(obs[key], 224, 224))
        noise = np.random.default_rng(seed).standard_normal((10, 32)).astype(np.float32)
        transformed = policy._input_transform(dict(obs))
        mask = np.asarray(transformed["tokenized_prompt_mask"], dtype=bool)
        tokens = np.asarray(transformed["tokenized_prompt"])[mask].tolist()
        capture = StageCapture(policy._model, len(tokens)) if args.stages and index == 0 else None
        try:
            actions = policy.infer(obs, noise=noise)["actions"]
            torch.cuda.synchronize()
        finally:
            if capture:
                capture.close()
        assert actions.shape == (10, 7) and normalized[-1].shape == (10, 32)
        assert np.isfinite(actions).all() and np.isfinite(normalized[-1]).all()
        request_name = f"{name}.request.json"
        expected_name = f"{name}.reference.json"
        request_hash = write_json(args.output / request_name, request_json(obs, noise))
        expected_hash = write_json(args.output / expected_name, {
            "actions": actions.tolist(), "normalized_actions": normalized[-1].tolist(), "token_ids": tokens,
        })
        row = {"name": name, "request": request_name, "reference": expected_name,
               "request_sha256": request_hash, "reference_sha256": expected_hash}
        if capture:
            stage_name = f"{name}.stages.npz"
            np.savez(args.output / stage_name, **capture.values)
            row["stages"] = stage_name
        manifest["cases"].append(row)
        print(f"Exported {name}: {len(tokens)} tokens, finite actions", flush=True)
    write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
