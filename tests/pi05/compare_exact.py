# SPDX-License-Identifier: AGPL-3.0-only
"""Require identical decoded action bits for paired native policy runs."""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


def action_bits(actions):
    if len(actions) != 10 or any(len(row) != 7 for row in actions):
        raise ValueError("expected 10 by 7 decoded actions")
    values = [float(value) for row in actions for value in row]
    if not all(map(math.isfinite, values)):
        raise ValueError("non-finite decoded actions")
    return [struct.pack("<d", value) for value in values]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_bytes())
    candidate = json.loads(args.candidate.read_bytes())
    if baseline["backend"] != candidate["backend"]:
        raise ValueError("execution precision/backend changed")
    if baseline["requests"] != candidate["requests"] or not baseline["requests"]:
        raise ValueError("paired ordered request paths required")
    cases = []
    for request, old, new in zip(
        baseline["requests"], baseline["outputs"], candidate["outputs"], strict=True
    ):
        before, after = action_bits(old["actions"]), action_bits(new["actions"])
        changed = sum(a != b for a, b in zip(before, after, strict=True))
        cases.append({"request": request, "changed_values": changed, "bit_exact": not changed})
    report = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        "backend": baseline["backend"],
        "cases": cases,
        "compared_values": len(cases) * 70,
        "pass": all(case["bit_exact"] for case in cases),
        "comparison": "Decoded JSON float64 bits, including signed zero; timings excluded.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(cases)} paired cases, bit-exact={report['pass']}")
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
