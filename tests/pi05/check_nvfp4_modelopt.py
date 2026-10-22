# SPDX-License-Identifier: AGPL-3.0-only
"""Characterize native activation NVFP4 versus independently captured ModelOpt.

Diagnostic only: production kernels and reference tensors are left unchanged.
Usage: check_nvfp4_modelopt.py PI05_PTX EXPORTED_MANIFEST_DIR OUTPUT_JSON
"""
import ctypes as ct
import json
from pathlib import Path
import sys

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def approximate_normalization(x, scale, ratio, global_scale_ptr):
    # Mirrors the arithmetic in ModelOpt's fp4_fake_quant_kernel solely to
    # localize differences. The saved ModelOpt outputs remain the oracle.
    global_scale = tl.load(global_scale_ptr)
    group = tl.program_id(0)
    offsets = group * 16 + tl.arange(0, 16)
    values = tl.load(x + offsets).to(tl.float32)
    maximum = tl.max(tl.abs(values), 0)
    scaled = tl.minimum(maximum / (6.0 * global_scale), 448.0)
    fp8 = scaled.to(tl.float8e4nv).to(tl.float32)
    effective = fp8 * global_scale
    effective = tl.where(effective >= 1e-5, effective, 1.0)
    tl.store(scale + group, fp8)
    tl.store(ratio + offsets, tl.abs(values) / effective)


def unpack(packed):
    result = torch.empty(packed.shape[0], packed.shape[1] * 2, device="cuda", dtype=torch.uint8)
    result[:, ::2], result[:, 1::2] = packed & 15, packed >> 4
    return result


def quantized_levels(ratio):
    bounds = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0], device="cuda")
    indices = torch.bucketize(ratio.contiguous(), bounds)
    for i in [1, 3, 5]:
        indices += (ratio == bounds[i]).to(indices.dtype)
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    return indices, levels[indices]


def main():
    ptx_path, root, output = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    arrays = np.load(root / "affine-oracles.npz")
    names = json.loads((root / "affine-oracles.json").read_text())["weight_names"]
    manifest = json.loads((root / "manifest.json").read_text())
    entries = {entry["name"]: entry for entry in manifest["tensors"]}
    fixtures = output.parent / "native-nvfp4-affine-fixtures"
    fixtures.mkdir(exist_ok=True)
    cases = []
    torch.cuda.init()
    torch.empty(1, device="cuda")
    cuda = ct.CDLL("libcuda.so.1")
    cuda.cuModuleLoadData.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p]
    cuda.cuModuleGetFunction.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p]
    cuda.cuLaunchKernel.argtypes = [ct.c_void_p] + [ct.c_uint] * 7 + [ct.c_void_p] * 3
    module, function = ct.c_void_p(), ct.c_void_p()
    assert cuda.cuModuleLoadData(ct.byref(module), ptx_path.read_bytes()) == 0
    assert cuda.cuModuleGetFunction(ct.byref(function), module, b"pi05_quantize_nvfp4") == 0
    records = []
    for i, name in enumerate(names):
        x = torch.from_numpy(arrays[f"input_{i}"]).cuda().bfloat16().contiguous()
        expected = torch.from_numpy(arrays[f"quantized_input_{i}"]).cuda().bfloat16()
        rows, k = x.shape
        entry = entries[name]
        global_scale = entry["activation_global_scale"]
        case_path = fixtures / f"case_{i}"
        case_path.mkdir(exist_ok=True)
        (case_path / "input.bf16").write_bytes(x.view(torch.uint8).cpu().numpy().tobytes())
        n, weight_k = entry["shape"]
        assert weight_k == k
        with (root / manifest["weight_file"]).open("rb") as payload:
            for offset, size, filename in [(entry["data_offset"], n*k//2, "weight.fp4"),
                                           (entry["scale_offset"], ((n+127)//128)*128*(k//16), "weight.scale")]:
                destination = case_path / filename
                payload.seek(offset)
                data = payload.read(size)
                assert len(data) == size
                destination.write_bytes(data)
        cases.append(dict(name=f"case_{i}", m=rows, n=n, k=k, activation_global_scale=global_scale,
                          weight_global_scale=entry["weight_global_scale"]))
        packed = torch.empty(rows, k // 2, device="cuda", dtype=torch.uint8)
        swizzled = torch.empty(((rows + 127) // 128) * 128 * k // 16, device="cuda", dtype=torch.uint8)
        values = [ct.c_uint64(t.data_ptr()) for t in [x, packed, swizzled]] + [ct.c_uint(rows), ct.c_uint(k), ct.c_float(global_scale)]
        pointers = (ct.c_void_p * len(values))(*[ct.addressof(a) for a in values])
        blocks = (((rows + 127) // 128) * 128 * (k // 16) + 31) // 32
        torch.cuda.synchronize()
        assert cuda.cuLaunchKernel(function, blocks, 1, 1, 256, 1, 1, 0, None, pointers, None) == 0
        torch.cuda.synchronize()
        r = torch.arange(rows, device="cuda")[:, None]
        g = torch.arange(k // 16, device="cuda")[None, :]
        offsets = ((r // 128) * (k // 64) + g // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + g % 4
        scale = swizzled[offsets].contiguous().view(torch.float8_e4m3fn).float()
        effective = scale.repeat_interleave(16, dim=1) * global_scale
        codes = unpack(packed)
        levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device="cuda")
        native = (levels[(codes & 7).long()] * torch.where(codes & 8 != 0, -1., 1.) * effective).bfloat16()
        ratios = torch.empty_like(x, dtype=torch.float32)
        approximate_scales = torch.empty_like(scale)
        global_tensor = torch.tensor(global_scale, device="cuda", dtype=torch.float32)
        approximate_normalization[(rows * (k // 16),)](x, approximate_scales, ratios, global_tensor)
        approx_codes, approx_levels = quantized_levels(ratios)
        approximate_effective = approximate_scales.repeat_interleave(16, dim=1) * global_scale
        approximate_effective = torch.where(approximate_effective >= 1e-5, approximate_effective, 1.)
        approximate_qdq = (approx_levels * torch.where(x >= 0, 1., -1.) * approximate_effective).bfloat16()
        different = native != expected
        precise_ratio = x.float().abs() / torch.where(effective >= 1e-5, effective, 1.)
        bounds = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0], device="cuda")
        distance = (precise_ratio[..., None] - bounds).abs().min(-1).values
        near_midpoint = distance <= 2 * torch.finfo(torch.float32).eps * precise_ratio.clamp(min=1)
        scale_changed = (approximate_scales != scale).repeat_interleave(16, dim=1)
        err = (native.float() - expected.float()).abs()
        samples = []
        for row, col in different.nonzero()[:12].tolist():
            samples.append({"row": row, "column": col, "input": x[row, col].item(),
                            "native": native[row, col].item(), "modelopt": expected[row, col].item(),
                            "precise_ratio": precise_ratio[row, col].item(), "approximate_ratio": ratios[row, col].item(),
                            "native_scale": scale[row, col // 16].item(), "approximate_scale": approximate_scales[row, col // 16].item(),
                            "native_code": int(codes[row, col].item() & 7), "approximate_code": approx_codes[row, col].item()})
        records.append({"weight_name": name, "shape": list(x.shape), "global_scale": global_scale,
                        "different": int(different.sum()), "total": x.numel(), "max_abs": err.max().item(),
                        "mean_abs": err.mean().item(), "rmse": err.square().mean().sqrt().item(),
                        "cosine": min(1.0, torch.nn.functional.cosine_similarity(native.double().flatten(), expected.double().flatten(), dim=0).item()),
                        "scale_blocks_different": int((approximate_scales != scale).sum()),
                        "differences_near_midpoint": int((different & near_midpoint).sum()),
                        "differences_in_changed_scale_block": int((different & scale_changed).sum()),
                        "approximate_diagnostic_differs_from_saved_oracle": int((approximate_qdq != expected).sum()), "examples": samples})
        if (case_path / "native.bf16").exists():
            bits = np.fromfile(case_path / "native.bf16", dtype="<u2").astype(np.uint32) << 16
            actual = bits.view(np.float32).astype(np.float64).reshape(rows, n)
            wanted = arrays[f"output_{i}"].astype(np.float64)
            delta = actual - wanted
            records[-1]["affine"] = {"shape": [rows, n], "max_abs": float(np.abs(delta).max()),
                                    "rmse": float(np.sqrt(np.mean(delta**2))),
                                    "relative_l2": float(np.linalg.norm(delta)/np.linalg.norm(wanted)),
                                    "cosine": float(np.vdot(actual, wanted)/np.linalg.norm(actual)/np.linalg.norm(wanted))}
    (fixtures / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    output.write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
