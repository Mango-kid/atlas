# SPDX-License-Identifier: AGPL-3.0-only
"""Save native HTTP outputs and paired action drift on held-out observations."""

import argparse
import hashlib
import json
import math
import urllib.request
from pathlib import Path

from analyze_nvfp4_quality import metrics


def validate(actions):
    if len(actions) != 10 or any(
        len(row) != 7 or not all(map(math.isfinite, row)) for row in actions
    ):
        raise ValueError("expected finite 10 by 7 actions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--references", type=Path, action="append", required=True)
    parser.add_argument("--expected-backend", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    requests = sorted(args.requests.glob("*.request.json"))
    if not requests:
        raise ValueError("no request fixtures")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    cases = []
    for path in requests:
        name = path.name.removesuffix(".request.json")
        body = path.read_bytes()
        request = urllib.request.Request(
            args.url.rstrip("/") + "/v1/actions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with opener.open(request, timeout=180) as response:
            result = json.load(response)
        if result["backend"] != args.expected_backend:
            raise ValueError("unexpected execution backend")
        validate(result["actions"])
        comparisons = {}
        for directory in args.references:
            reference_path = directory / f"{name}.reference.json"
            reference_bytes = reference_path.read_bytes()
            reference = json.loads(reference_bytes)
            validate(reference["actions"])
            comparisons[str(directory)] = {
                "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
                **metrics(result["actions"], reference["actions"]),
            }
        cases.append(
            {
                "case": name,
                "request_sha256": hashlib.sha256(body).hexdigest(),
                "response": result,
                "comparisons": comparisons,
            }
        )
    report = {
        "url": args.url,
        "backend": args.expected_backend,
        "cases": cases,
        "note": "Paired numerical drift, not robot task success. Timings may include graph capture and are not latency benchmarks.",
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(cases)} paired cases to {args.output}")


if __name__ == "__main__":
    main()
