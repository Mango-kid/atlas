# SPDX-License-Identifier: AGPL-3.0-only
"""Atlas NVFP4 interchange layout; independent of GPU execution."""

import hashlib
import json
import struct

import numpy as np


def swizzle_scales(scales):
    """Row-major E4M3 bytes -> cuBLAS 128-row x 4-block scale tiles."""
    if scales.dtype != np.uint8 or scales.ndim != 2:
        raise ValueError("scales must be a two-dimensional uint8 array")
    rows, groups = scales.shape
    padded_rows = (rows + 127) // 128 * 128
    padded_groups = (groups + 3) // 4 * 4
    padded = np.zeros((padded_rows, padded_groups), dtype=np.uint8)
    padded[:rows, :groups] = scales
    return (
        padded.reshape(padded_rows // 128, 4, 32, padded_groups // 4, 4)
        .transpose(0, 3, 2, 1, 4)
        .copy()
        .ravel()
    )


def base_tensor_hashes(path, quantized_names):
    """Hash raw BF16 checkpoint bytes; no decoding or dtype conversion."""
    result = {}
    with path.open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_size))
        for name, tensor in sorted(header.items()):
            if (
                name == "__metadata__"
                or name in quantized_names
                or name == "paligemma_with_expert.gemma_expert.lm_head.weight"
            ):
                continue
            if tensor["dtype"] != "BF16":
                raise ValueError(f"expected BF16 checkpoint tensor: {name}")
            start, end = tensor["data_offsets"]
            source.seek(8 + header_size + start)
            remaining = end - start
            digest = hashlib.sha256()
            while remaining:
                block = source.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError(f"truncated checkpoint tensor: {name}")
                digest.update(block)
                remaining -= len(block)
            result[name] = digest.hexdigest()
    return result


def append_aligned(stream, data):
    """Tensor starts are aligned independently; lengths derive from matrix shapes."""
    padding = (-stream.tell()) % 256
    stream.write(bytes(padding))
    offset = stream.tell()
    stream.write(data)
    return offset, hashlib.sha256(data).hexdigest()
