# SPDX-License-Identifier: AGPL-3.0-only
"""Compare native action outputs with paired fixed-input/noise OpenPi fixtures."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--references", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-cosine", type=float, required=True)
    parser.add_argument("--maximum-absolute-error", type=float, required=True)
    args = parser.parse_args()
    native = json.loads(args.native.read_text())
    cases = []
    for request, result in zip(native["requests"], native["outputs"], strict=True):
        name = Path(request).name.removesuffix(".request.json")
        paths = [root / f"{name}.reference.json" for root in args.references]
        path = next((p for p in paths if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"missing reference for {name}")
        reference = json.loads(path.read_text())
        actual = np.asarray(result["actions"], dtype=np.float64)
        expected = np.asarray(reference["actions"], dtype=np.float64)
        if actual.shape != (10, 7) or expected.shape != actual.shape:
            raise ValueError(f"{name}: invalid action shape")
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError(f"{name}: non-finite actions")
        error = actual - expected
        cosine = float(np.dot(actual.ravel(), expected.ravel()) / np.linalg.norm(actual) / np.linalg.norm(expected))
        maximum = float(np.abs(error).max())
        cases.append({"case": name, "cosine": cosine,
                      "rmse": float(np.sqrt(np.mean(error**2))), "max_abs": maximum,
                      "pass": cosine >= args.minimum_cosine and maximum <= args.maximum_absolute_error})
    report = {"native": str(args.native), "references": [str(p) for p in args.references],
              "thresholds": {"minimum_cosine": args.minimum_cosine,
                             "maximum_absolute_error": args.maximum_absolute_error},
              "cases": cases, "pass": bool(cases) and all(case["pass"] for case in cases),
              "note": "Numerical agreement on paired observations/noise; not robot task success."}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
