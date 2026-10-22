# SPDX-License-Identifier: AGPL-3.0-only
"""GPU oracle for native pi0.5 primitives; Python is only the test harness.

Run inside the pinned OpenPi image: python ops_parity.py /path/pi05.ptx
"""
import ctypes as ct
import json
import sys

import torch
import torch.nn.functional as F

torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = False
torch.cuda.init()
torch.empty(1, device="cuda")
cuda = ct.CDLL("libcuda.so.1")
cuda.cuModuleLoadData.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p]
cuda.cuModuleGetFunction.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p]
cuda.cuLaunchKernel.argtypes = [ct.c_void_p] + [ct.c_uint] * 7 + [ct.c_void_p] * 3
module = ct.c_void_p()
ptx = open(sys.argv[1], "rb").read()
assert cuda.cuModuleLoadData(ct.byref(module), ptx) == 0


def launch(name, args, grid, block=(256, 1, 1), shared=0):
    function = ct.c_void_p()
    assert cuda.cuModuleGetFunction(ct.byref(function), module, name.encode()) == 0
    values = []
    for a in args:
        if isinstance(a, torch.Tensor):
            values.append(ct.c_uint64(a.data_ptr()))
        elif a is None:
            values.append(ct.c_uint64(0))
        elif isinstance(a, float):
            values.append(ct.c_float(a))
        else:
            values.append(ct.c_uint(a))
    pointers = (ct.c_void_p * len(values))(*[ct.addressof(a) for a in values])
    torch.cuda.synchronize()
    result = cuda.cuLaunchKernel(function, *grid, *block, shared, None, pointers, None)
    assert result == 0, (name, result)
    torch.cuda.synchronize()


records = []


def check(name, actual, expected, atol=0.02, rtol=0.02):
    error = (actual.float() - expected.float()).abs()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    records.append({"name": name, "max_abs": error.max().item(), "mean_abs": error.mean().item()})


def rand(*shape, dtype=torch.bfloat16):
    return torch.randn(*shape, device="cuda", dtype=dtype)


for width in (72, 1024, 1152, 2048):
    rows = 10
    x = rand(rows, width)
    w = rand(width)
    bias = rand(width)
    out = torch.empty_like(x)
    launch("pi05_layer_norm", [x, w, bias, out, width, 1e-6], (rows, 1, 1))
    check(f"layer_norm_{width}", out, F.layer_norm(x, (width,), w, bias, 1e-6))
    wf = w.float()
    launch("pi05_rms_norm", [x, wf, out, width, 1e-6], (rows, 1, 1))
    norm = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    check(f"rms_norm_{width}", out, (norm * (1 + wf)).bfloat16())
    mod = rand(3 * width, dtype=torch.float32)
    launch("pi05_adarms", [x, mod, out, width, 1e-6], (rows, 1, 1))
    check(f"adarms_{width}", out, (norm * (1 + mod[:width]) + mod[width:2*width]).bfloat16())
    actual = x.clone()
    gate = mod[2*width:]
    delta = rand(rows, width)
    launch("pi05_residual", [actual, delta, gate, rows * width, width], ((rows * width + 255)//256, 1, 1))
    check(f"gated_residual_{width}", actual, x + gate.bfloat16() * delta, 0, 0)

for m, n, k in ((10, 1024, 32), (1, 3072, 1024), (256, 1152, 588)):
    a = rand(m, k, dtype=torch.float32)
    w = rand(n, k, dtype=torch.float32)
    bias = rand(n, dtype=torch.float32)
    out = torch.empty(m, n, device="cuda")
    launch("pi05_linear_f32", [a, w, bias, out, m, n, k], ((n+15)//16, (m+15)//16, 1), (16,16,1))
    check(f"linear_f32_{m}_{n}_{k}", out, F.linear(a, w, bias), 0.0003, 0.0003)

for rows, kvrows, qheads, kvheads, dim in ((256,256,16,16,72), (712,712,8,1,256), (10,722,8,1,256)):
    q, k, v = rand(rows,qheads,dim), rand(kvrows,kvheads,dim), rand(kvrows,kvheads,dim)
    out = torch.empty_like(q)
    scores = torch.empty(qheads,rows,kvrows,device="cuda",dtype=torch.bfloat16)
    launch("pi05_attention_qk", [q,k,scores,None,rows,kvrows,qheads,kvheads,dim], ((kvrows+63)//64,(rows+15)//16,qheads),(128,1,1))
    launch("pi05_attention_softmax", [scores,kvrows], (qheads*rows,1,1), shared=kvrows*4)
    launch("pi05_attention_pv", [scores,v,out,rows,kvrows,qheads,kvheads,dim], ((dim+63)//64,(rows+15)//16,qheads),(128,1,1))
    qq = q.transpose(0,1)
    kk = k.repeat_interleave(qheads//kvheads,1).permute(1,2,0)
    vv = v.repeat_interleave(qheads//kvheads,1).transpose(0,1)
    probabilities = (qq @ kk * dim**-0.5).float().softmax(-1).bfloat16()
    expected = (probabilities @ vv).transpose(0,1)
    check(f"attention_{rows}_{dim}", out, expected)

# BF16 affine and no-bias projections are checked through the actual Rust
# cuBLASLt API by examples/affine_check.rs with export_affine.py fixtures.

rows, heads, dim = 20,8,256
x = rand(rows,heads,dim)
# Use the actual patched model module as the oracle: its inv_freq buffer is
# rounded by backbone.to(bfloat16), not retained at its construction dtype.
from transformers import GemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding
positions = torch.arange(rows, device="cuda", dtype=torch.int32) * 37
rotary = GemmaRotaryEmbedding(GemmaConfig(hidden_size=2048, num_attention_heads=8, head_dim=dim), device="cuda").bfloat16()
cs, sn = rotary(x, positions[None, :])
cs, sn = cs[0, :, None, :], sn[0, :, None, :]
expected = x * cs + torch.cat((-x[...,dim//2:],x[...,:dim//2]),-1) * sn
actual = x.clone()
launch("pi05_rope", [actual,positions,rows,heads,dim], ((rows*heads*dim//2+255)//256,1,1))
check("rope", actual, expected, 0, 0)
print(json.dumps(records, indent=2))
