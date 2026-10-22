# SPDX-License-Identifier: AGPL-3.0-only
"""Independent F32 arithmetic oracle for calibrated E4M3 affine overlays."""

import hashlib
import json
import types

import numpy as np
import torch
from export_fp8 import encode, raw_bytes
from fp8_format import bound_source_hashes, tensor_names, verify_source


def install_fp8_linears(model, overlay, nvfp4):
    manifest = json.loads((overlay / "manifest.json").read_text())
    source_bytes = (nvfp4 / "manifest.json").read_bytes()
    if manifest["nvfp4_manifest_sha256"] != hashlib.sha256(source_bytes).hexdigest():
        raise ValueError("FP8 overlay is bound to a different NVFP4 manifest")
    scopes = {
        "fp8-vision-bf16": "vision",
        "fp8-expert-bf16": "expert",
        "fp8-vision-expert-bf16": "vision-expert",
    }
    if manifest["precision"] not in scopes:
        raise ValueError("unsupported explicit FP8 overlay precision")
    names = tensor_names(scopes[manifest["precision"]])
    if {entry["name"] for entry in manifest["tensors"]} != set(names) or len(
        manifest["tensors"]
    ) != len(names):
        raise ValueError("FP8 overlay tensor inventory differs")
    if manifest["base_tensor_sha256"] != bound_source_hashes(
        json.loads(source_bytes), names
    ):
        raise ValueError("FP8 source hashes differ from the bound NVFP4 manifest")
    modules = dict(model.named_modules())
    with (overlay / manifest["weight_file"]).open("rb") as payload:
        for index, entry in enumerate(manifest["tensors"]):
            module = modules[entry["name"].removesuffix(".weight")]
            if list(module.weight.shape) != entry["shape"]:
                raise ValueError("FP8 reference projection shape differs")
            verify_source(
                raw_bytes(module.weight),
                manifest["base_tensor_sha256"][entry["name"]],
                entry["name"],
            )
            n, k = entry["shape"]
            payload.seek(entry["data_offset"])
            raw = payload.read(n * k)
            if (
                len(raw) != n * k
                or hashlib.sha256(raw).hexdigest() != entry["packed_sha256"]
            ):
                raise ValueError("FP8 reference payload hash differs")
            decoded = (
                torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy())
                .cuda()
                .view(torch.float8_e4m3fn)
                .reshape(n, k)
                .float()
            )
            activation_scale = float(np.float32(entry["activation_scale"]))
            divisor = torch.tensor(activation_scale, device="cuda", dtype=torch.float32)
            alpha = float(
                np.float32(
                    np.float32(entry["activation_scale"])
                    * np.float32(entry["weight_scale"])
                )
            )

            def forward(self, x, decoded=decoded, divisor=divisor, alpha=alpha):
                # Tensor/tensor division is IEEE F32; a Python scalar RHS can
                # lower to reciprocal multiplication and flip FP8 midpoints.
                activation = encode(x, divisor).float()
                previous = torch.backends.cuda.matmul.allow_tf32
                try:
                    torch.backends.cuda.matmul.allow_tf32 = False
                    result = torch.nn.functional.linear(activation, decoded) * alpha
                    if self.bias is not None:
                        result = result + self.bias.float()
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = previous
                return result.to(x.dtype)

            module.forward = types.MethodType(forward, module)
            if (index + 1) % 40 == 0:
                print(
                    f"Installed precise FP8 affine {index + 1}/{len(names)}", flush=True
                )
    return manifest
