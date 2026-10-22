# SPDX-License-Identifier: AGPL-3.0-only
"""Explicit pi05 FP8 overlay inventory and source-checkpoint binding."""

import hashlib

PALIGEMMA = "paligemma_with_expert.paligemma.model."
EXPERT = "paligemma_with_expert.gemma_expert.model.layers."


def tensor_names(scope):
    vision = [
        f"{PALIGEMMA}vision_tower.vision_model.encoder.layers.{layer}.{projection}.weight"
        for layer in range(27)
        for projection in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
            "mlp.fc1",
            "mlp.fc2",
        ]
    ] + [f"{PALIGEMMA}multi_modal_projector.linear.weight"]
    if scope == "vision":
        return vision
    if scope not in ["expert", "vision-expert"]:
        raise ValueError(f"unsupported explicit FP8 scope: {scope}")
    expert = [
        f"{EXPERT}{layer}.{projection}.weight"
        for layer in range(18)
        for projection in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]
    ]

    return expert if scope == "expert" else vision + expert


def bound_source_hashes(manifest, names):
    retained = manifest["base_tensor_sha256"]
    if any(name not in retained for name in names):
        raise ValueError(
            "FP8 overlay selects a tensor absent from the bound NVFP4 retained base"
        )
    return {name: retained[name] for name in names}


def verify_source(raw_bf16, expected, name):
    if hashlib.sha256(raw_bf16).hexdigest() != expected:
        raise ValueError(
            f"source BF16 tensor differs from bound NVFP4 checkpoint: {name}"
        )
