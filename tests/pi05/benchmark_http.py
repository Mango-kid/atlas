# SPDX-License-Identifier: AGPL-3.0-only
"""Measure Atlas JSON policy round trips separately from synchronized inference."""
import argparse
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path


def stats(samples):
    ordered = sorted(samples)
    return {"mean_ms": statistics.mean(samples), "p50_ms": statistics.median(samples),
            "p95_ms": ordered[round((len(ordered) - 1) * 0.95)], "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--expected-native", type=Path, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup <= 0 or args.runs <= 0:
        parser.error("warmup and runs must be positive")
    body = args.request.read_bytes()
    baseline = json.loads(args.expected_native.read_text())
    expected_backend = baseline["backend"]
    if expected_backend not in ("atlas-native-bf16", "atlas-native-nvfp4", "atlas-native-fp8-nvfp4"):
        raise ValueError("reference must identify a native Atlas policy backend")
    index = next(i for i, path in enumerate(baseline["requests"])
                 if Path(path).name == args.request.name)
    expected = baseline["outputs"][index]["actions"]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(args.url.rstrip("/") + "/healthz", timeout=10) as response:
        health = json.load(response)
    elapsed, server = [], []
    max_error = 0.0
    for iteration in range(args.warmup + args.runs):
        request = urllib.request.Request(args.url.rstrip("/") + "/v1/actions",
                                         data=body, headers={"Content-Type": "application/json"})
        start = time.perf_counter_ns()
        with opener.open(request, timeout=180) as response:
            result = json.load(response)
        ms = (time.perf_counter_ns() - start) / 1e6
        actions = result["actions"]
        if result["backend"] != expected_backend or len(actions) != 10:
            raise ValueError("invalid policy response")
        if any(len(row) != 7 or not all(map(math.isfinite, row)) for row in actions):
            raise ValueError("invalid action shape or non-finite values")
        error = max(abs(a - b) for row, ref in zip(actions, expected, strict=True)
                    for a, b in zip(row, ref, strict=True))
        max_error = max(max_error, error)
        if error > 1e-6:
            raise ValueError(f"HTTP action mismatch: {error}")
        if iteration >= args.warmup:
            elapsed.append(ms)
            server.append(result["timing_ms"])
    report = {"url": args.url, "health": health, "backend": expected_backend, "request": str(args.request),
              "request_bytes": len(body), "warmup": args.warmup, "runs": args.runs,
              "round_trip": stats(elapsed), "server_inference": stats(server),
              "max_abs_vs_native": max_error, "pass": True,
              "measurement": "Sequential JSON HTTP requests; includes network, JSON parsing, and inference. Pre-serialized body; loading/warmup excluded. No proxy. urllib opens a connection per request."}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("round_trip", "server_inference")}))
    print("round trip", statistics.mean(elapsed), "server", statistics.mean(server))


if __name__ == "__main__":
    main()
