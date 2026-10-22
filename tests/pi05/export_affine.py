# SPDX-License-Identifier: AGPL-3.0-only
"""Export PyTorch BF16 affine oracles for Atlas's actual Rust cuBLASLt API."""
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

torch.manual_seed(42)
root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=False)
cases = []
for name, m, n, k in (
    ("vision_qkv_o", 256, 1152, 1152),
    ("vision_up", 256, 4304, 1152),
    ("vision_down", 256, 1152, 4304),
    ("vision_projector", 256, 2048, 1152),
):
    directory = root / name
    directory.mkdir()
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    for filename, tensor in (
        ("input", x), ("weight", weight), ("bias", bias),
        ("expected", F.linear(x, weight)),
        ("expected_bias", F.linear(x, weight, bias)),
    ):
        (directory / f"{filename}.bf16").write_bytes(tensor.contiguous().view(torch.uint16).cpu().numpy().tobytes())
    cases.append(dict(name=name, m=m, n=n, k=k))
(root / "cases.json").write_text(json.dumps(cases, indent=2))
