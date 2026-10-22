# SPDX-License-Identifier: AGPL-3.0-only
"""Independent native FP8 codec/affine oracles, TF32 explicitly disabled."""
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
cases = []
for name, m, n, k in [
    ("unit_scale", 10, 256, 1024),
    ("division_midpoint", 10, 256, 1024),
    ("vision_qkv", 256, 1152, 1152),
    ("vision_qkv_batch", 512, 1152, 1152),
    ("vision_up_batch", 512, 4304, 1152),
    ("vision_down_batch", 512, 1152, 4304),
    ("vision_projector_batch", 512, 2048, 1152),
    ("expert_q", 10, 2048, 1024),
    ("expert_kv", 10, 256, 1024),
    ("expert_o", 10, 1024, 2048),
    ("expert_up", 10, 4096, 1024),
    ("expert_down", 10, 1024, 4096),
]:
    path = root / name
    path.mkdir()
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * .02
    x[0, :16] = 0
    x[1, :16] *= 1e-7
    activation_scale = float(x.float().abs().max() / 448 * 1.125)
    weight_scale = float(w.float().abs().max() / 448 * 1.125)
    if name == "unit_scale":
        activation_scale = weight_scale = 1.
        x[0, :12] = torch.tensor([.0009765625, .0029296875, 1.0625, 1.1875, 1.3125, 1.4375, 1000, -1000, 448, -448, 0., -0.], device="cuda")
    if name == "division_midpoint":
        activation_scale = 0.0267857164144516
        x[0, 0] = -0.00018310546875
    activation_divisor = torch.tensor(activation_scale, device="cuda", dtype=torch.float32)
    weight_divisor = torch.tensor(weight_scale, device="cuda", dtype=torch.float32)
    packed = (x.float() / activation_divisor).clamp(-448,448).to(torch.float8_e4m3fn)
    weight = (w.float() / weight_divisor).clamp(-448,448).to(torch.float8_e4m3fn)
    alpha = (torch.tensor(activation_scale) * torch.tensor(weight_scale)).item()
    expected = (packed.float() @ weight.float().T) * alpha
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    for filename, tensor in [("input.bf16",x), ("weight.bf16",w), ("weight.fp8",weight),
                             ("activation.fp8",packed), ("bias.bf16",bias),
                             ("expected.bf16",expected.bfloat16()),
                             ("expected_bias.bf16",(expected+bias.float()).bfloat16())]:
        (path / filename).write_bytes(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    cases.append(dict(name=name,m=m,n=n,k=k,activation_scale=activation_scale,weight_scale=weight_scale))
(root / "cases.json").write_text(json.dumps(cases,indent=2))
(root / "reference_metadata.json").write_text(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"], matmul_allow_tf32=False,
    arithmetic="FP32 decodedFP8 GEMM, alpha=FP32 scale product, optionalbiasFP32 beforeBF16output"),indent=2))
