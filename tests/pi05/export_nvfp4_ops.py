# SPDX-License-Identifier: AGPL-3.0-only
"""Independent PyTorch FP4 codec/GEMM oracle for native Atlas CUDA probes."""
import json
import os
from pathlib import Path
import sys
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
import torch

torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = False
root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=False)


def encode(x, global_scale):
    rows, k = x.shape
    blocks = x.float().reshape(rows, k // 16, 16)
    scale = (blocks.abs().amax(-1) / (6 * global_scale)).clamp(max=448).to(torch.float8_e4m3fn)
    effective = scale.float() * global_scale
    effective = torch.where(effective >= 1e-5, effective, 1.0)
    values = blocks / effective[..., None]
    thresholds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    codes = torch.bucketize(values.abs().contiguous(), thresholds)
    # Round midpoint ties to the even E2M1 significand, including 0.25 -> 0.
    for i in [1, 3, 5]:
        codes += (values.abs() == thresholds[i]).to(codes.dtype)
    code = codes.to(torch.uint8) | (torch.signbit(values).to(torch.uint8) << 3)
    code = code.reshape(rows, k)
    packed = code[:, ::2] | (code[:, 1::2] << 4)
    padded = torch.zeros(((rows + 127) // 128) * 128, k // 16, device=x.device, dtype=torch.uint8)
    padded[:rows] = scale.view(torch.uint8)
    r = torch.arange(padded.shape[0], device=x.device)[:, None]
    g = torch.arange(k // 16, device=x.device)[None, :]
    offsets = ((r // 128) * (k // 64) + g // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + g % 4
    swizzled = torch.empty(padded.numel(), device=x.device, dtype=torch.uint8)
    swizzled[offsets.flatten()] = padded.flatten()
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=x.device)
    decoded = levels[(code & 7).long()] * torch.where(code & 8 != 0, -1.0, 1.0)
    decoded *= scale.float().repeat_interleave(16, dim=1)
    return packed, swizzled, decoded


def write(path, tensor):
    path.write_bytes(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())


cases = []
for name, m, n, k in [
    ("ties_clipping", 10, 256, 1024),
    ("ragged_small", 10, 256, 1024),
    ("prefix_kv", 518, 256, 2048),
    ("prefix_q", 518, 2048, 2048),
    ("prefix_up", 518, 16384, 2048),
    ("prefix_down", 518, 2048, 16384),
    ("max_prefix", 712, 256, 2048),
]:
    path = root / name
    path.mkdir()
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * .02
    # Nonuniform scales expose swizzle errors; zeros and subnormal values cover guards.
    x[0, :16] = 0
    x[1, :16] *= 1e-7
    x[2, :16] *= 7
    x[-1, -16:] *= .125
    ag = float(x.float().abs().max() / 2688 * 1.125)
    if name == "ties_clipping":
        values = [.25, .75, 1.25, 1.75, 2.5, 3.5, 5, 6, -.25, -.75, -1.25, -1.75, -2.5, -3.5, -5, -6]
        x[0] = torch.tensor(values, device=x.device, dtype=x.dtype).repeat(k // 16)
        x[1] *= 100  # Explicit clipping beyond calibrated range.
        ag = 1.0 / 448
    wg = float(w.float().abs().max() / 2688 * 1.125)
    ap, ass, ad = encode(x, ag)
    wp, ws, wd = encode(w, wg)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    alpha = (torch.tensor(ag, dtype=torch.float32) * torch.tensor(wg, dtype=torch.float32)).item()
    expected = (ad @ wd.T) * alpha
    for filename, tensor in [("input.bf16", x), ("weight.bf16", w), ("bias.bf16", bias),
                             ("activation.fp4", ap), ("activation.scale", ass),
                             ("weight.fp4", wp), ("weight.scale", ws),
                             ("expected.bf16", expected.bfloat16()),
                             ("expected_bias.bf16", (expected + bias.float()).bfloat16())]:
        write(path / filename, tensor)
    cases.append(dict(name=name, m=m, n=n, k=k, activation_global_scale=ag, weight_global_scale=wg))
(root / "cases.json").write_text(json.dumps(cases, indent=2))

(root / "reference_metadata.json").write_text(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"], "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "arithmetic": "FP32 E2M1*E4M3 operands, FP32 GEMM then FP32 product of global scales, BF16 output"}, indent=2))
