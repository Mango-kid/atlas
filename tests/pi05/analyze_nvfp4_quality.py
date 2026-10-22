# SPDX-License-Identifier: AGPL-3.0-only
"""Summarize policy behavior independently of native execution parity."""

import argparse
import json
from pathlib import Path

import numpy as np


def metrics(actions, baseline):
    x, y = np.asarray(actions, dtype=np.float64), np.asarray(baseline, dtype=np.float64)

    def cosine(a, b):
        return float(a.ravel() @ b.ravel() / np.linalg.norm(a) / np.linalg.norm(b))

    return {
        "cosine": cosine(x, y),
        "rmse": float(np.mean((x - y) ** 2) ** 0.5),
        "max_abs": float(np.max(np.abs(x - y))),
        "arm_cosine": cosine(x[:, :6], y[:, :6]),
        "arm_rmse": float(np.mean((x[:, :6] - y[:, :6]) ** 2) ** 0.5),
        "gripper_sign_changes": int(
            np.count_nonzero(np.signbit(x[:, 6]) != np.signbit(y[:, 6]))
        ),
        "gripper": x[:, 6].tolist(),
        "bf16_gripper": y[:, 6].tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, action="append", required=True)
    parser.add_argument("--native", type=Path, required=True)
    args = parser.parse_args()
    originals = {
        p.name.removesuffix(".reference.json"): json.loads(p.read_text())["actions"]
        for root in args.baseline
        for p in root.glob("*.reference.json")
    }
    report = {
        "published_profile": "nvfp4-prefix-bf16",
        "published_activation_headroom": 1,
        "headroom_variants_promoted": False,
        "precision_scope": "Only 126 prefix language linear weights and activations use NVFP4; vision, expert and attention preserve the BF16 reference, sensitive paths remain FP32. This differs from the tutorial combined FP8/NVFP4/FP16 recipe.",
        "calibration": "32 pinned real LIBERO observations; test synthetic images are out of that calibration distribution. The three real fixtures overlap the calibration population, so their agreement is not held-out generalization evidence.",
        "interpretation": "Quantizers-disabled control reproduces six eager BF16 cases exactly; the legacy compiled reference differs slightly. All tested headroom factors retain synthetic gripper-mode sensitivity after clipping is eliminated. This behavioral drift is separate from native arithmetic parity and is not a robot success evaluation.",
        "variants": {},
    }
    for name in ["disabled", "headroom-1", "headroom-2", "headroom-4", "headroom-8"]:
        folder = args.diagnostics / name
        cases = {
            p.name.removesuffix(".reference.json"): json.loads(p.read_text())["actions"]
            for p in (folder / "references").glob("*.reference.json")
        }
        variant = {
            "cases": {
                case: metrics(actions, originals[case])
                for case, actions in sorted(cases.items())
            }
        }
        if name != "disabled":
            layers = json.loads((folder / "activation-clipping.json").read_text())
            blocks = sum(row["blocks"] for row in layers.values())
            clipped = sum(row["clipped_blocks"] for row in layers.values())
            variant["clipping"] = {
                "blocks": blocks,
                "clipped_blocks": clipped,
                "fraction": clipped / blocks,
                "max_ratio_to_calibrated_range": max(
                    row["max_ratio"] for row in layers.values()
                ),
                "layers_with_clipping": sum(
                    row["clipped_blocks"] > 0 for row in layers.values()
                ),
                "scope": "All prefix rows including masked image/prompt padding and first-case affine probe rows; not a count of task-relevant tokens alone.",
            }
        report["variants"][name] = variant
    native = json.loads(args.native.read_text())
    report["variants"]["native-headroom-1"] = {
        "cases": {
            Path(path).name.removesuffix(".request.json"): metrics(
                output["actions"],
                originals[Path(path).name.removesuffix(".request.json")],
            )
            for path, output in zip(native["requests"], native["outputs"], strict=True)
        }
    }
    destination = args.diagnostics / "quality-report.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
