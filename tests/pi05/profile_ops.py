# SPDX-License-Identifier: AGPL-3.0-only
"""CUDA-event microprofile for native pi0.5 primitives and vendor baselines."""
import ctypes as ct
import hashlib
import json
import platform
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


def kernel(name, args, grid, block=(256, 1, 1), shared=0):
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

    def launch():
        # Capture values as well as pointers to keep scalar backing alive.
        assert len(values) == len(pointers)
        stream = torch.cuda.current_stream().cuda_stream
        result = cuda.cuLaunchKernel(function, *grid, *block, shared, stream, pointers, None)
        assert result == 0, (name, result)

    return launch


def measure(fn, iterations=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    # Use a CUDA graph to exclude Python dispatch gaps from kernel timings.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(5):
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) / iterations)
    return {"mean_ms": sum(values)/len(values), "samples_ms": values, "iterations": iterations}


def rand(*shape, dtype=torch.bfloat16):
    return torch.randn(*shape, device="cuda", dtype=dtype)


records = []
for label, rows, kvrows, qheads, kvheads, dim in (
    ("vision",256,256,16,16,72),
    ("prefix",516,516,8,1,256),
    ("suffix",10,526,8,1,256),
):
    q,k,v = rand(rows,qheads,dim),rand(kvrows,kvheads,dim),rand(kvrows,kvheads,dim)
    out = torch.empty_like(q)
    scores = torch.empty(qheads,rows,kvrows,device="cuda",dtype=torch.bfloat16)
    qk = kernel("pi05_attention_qk",[q,k,scores,None,rows,kvrows,qheads,kvheads,dim],((kvrows+63)//64,(rows+15)//16,qheads),(128,1,1))
    softmax = kernel("pi05_attention_softmax",[scores,kvrows],(qheads*rows,1,1),shared=kvrows*4)
    pv = kernel("pi05_attention_pv",[scores,v,out,rows,kvrows,qheads,kvheads,dim],((dim+63)//64,(rows+15)//16,qheads),(128,1,1))
    def native():
        qk()
        softmax()
        pv()
    qq = q.transpose(0,1).contiguous()
    kk = k.repeat_interleave(qheads//kvheads,1).transpose(0,1).contiguous()
    vv = v.repeat_interleave(qheads//kvheads,1).transpose(0,1).contiguous()
    def eager():
        probs = (qq @ kk.transpose(-1,-2) * dim**-0.5).float().softmax(-1).bfloat16()
        return probs @ vv
    records.append({"name":label+"_attention","shape":[rows,kvrows,qheads,kvheads,dim],
                    "native":measure(native),"torch_eager":measure(eager),
                    "torch_sdpa":measure(lambda:F.scaled_dot_product_attention(qq[None],kk[None],vv[None]))})

# BF16 affine timings now come from the Rust affine_check executable,
# which exercises the production cuBLASLt fused-bias API.

for label,m,n,k,count in (
    ("patch",256,1152,588,2),
    ("action_in",10,1024,32,10),
    ("action_out",10,32,1024,10),
):
    a,w,bias = rand(m,k,dtype=torch.float32),rand(n,k,dtype=torch.float32),rand(n,dtype=torch.float32)
    out = torch.empty(m,n,device="cuda")
    native = kernel("pi05_linear_f32",[a,w,bias,out,m,n,k],((n+15)//16,(m+15)//16,1),(16,16,1))
    records.append({"name":label,"shape":[m,n,k],"calls_per_policy":count,
                    "native":measure(native),"torch_linear":measure(lambda:F.linear(a,w,bias))})

print(json.dumps({"host":platform.node(),"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,
                  "ptx_sha256":hashlib.sha256(ptx).hexdigest(),
                  "method":"CUDAevents around 100 CUDAgraph replays; 5 repetitions; 10 warmups; singlegraphkernel body; TF32 disabled",
                  "records":records},indent=2))
