# SPDX-License-Identifier: AGPL-3.0-only
"""Export strict tensor-division FP8 affine oracles, retaining old fixtures."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def read_tensor(path, dtype, shape):
    return (
        torch.from_numpy(np.frombuffer(path.read_bytes(), dtype=np.uint8).copy())
        .cuda()
        .view(dtype)
        .reshape(shape)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = json.loads((args.source / "affine-cases.json").read_text())
    result = []
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad():
        for case in cases:
            old = args.source / case["path"]
            m, n, k = case["m"], case["n"], case["k"]
            x = read_tensor(old / "input.bf16", torch.bfloat16, (m, k))
            w = read_tensor(old / "weight.fp8", torch.float8_e4m3fn, (n, k))
            bias = (
                read_tensor(old / "bias.bf16", torch.bfloat16, (n,))
                if case["bias"]
                else None
            )
            divisor = torch.tensor(
                case["activation_scale"], dtype=torch.float32, device="cuda"
            )
            alpha = float(
                np.float32(
                    np.float32(case["activation_scale"])
                    * np.float32(case["weight_scale"])
                )
            )
            for repeat in [1, 2] if m == 256 else [1]:
                folder = args.output / f"{case['path']}-m{m * repeat}"
                folder.mkdir()
                xx = x.repeat(repeat, 1)
                activation = (
                    (xx.float() / divisor).clamp(-448, 448).to(torch.float8_e4m3fn)
                )
                expected = (
                    torch.nn.functional.linear(activation.float(), w.float()) * alpha
                )
                if bias is not None:
                    expected = expected + bias.float()
                for filename, tensor in [
                    ("input.bf16", xx),
                    ("activation.fp8", activation),
                    ("expected.bf16", expected.bfloat16()),
                ]:
                    (folder / filename).write_bytes(
                        tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                    )
                (folder / "weight.fp8").hardlink_to(old / "weight.fp8")
                if bias is not None:
                    (folder / "bias.bf16").hardlink_to(old / "bias.bf16")
                result.append(
                    {
                        **case,
                        "path": folder.name,
                        "m": m * repeat,
                        "arithmetic": "IEEE F32 tensor/tensor activation division; F32 decoded FP8 GEMM TF32off; F32 alpha+bias; BF16 output",
                    }
                )
    (args.output / "affine-cases.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Published {len(result)} precise FP8 affine cases", flush=True)


if __name__ == "__main__":
    main()
