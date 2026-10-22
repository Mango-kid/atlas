# SPDX-License-Identifier: AGPL-3.0-only
"""Extract a strict FP8 overlay scope without changing calibrated tensor bytes."""

import argparse
import hashlib
import json
from pathlib import Path

from fp8_format import tensor_names
from nvfp4_format import append_aligned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--scope", choices=["vision", "expert", "vision-expert"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_bytes = (args.source / "manifest.json").read_bytes()
    manifest = json.loads(source_bytes)
    wanted = set(tensor_names(args.scope))
    entries = [entry for entry in manifest["tensors"] if entry["name"] in wanted]
    if {entry["name"] for entry in entries} != wanted or len(entries) != len(wanted):
        raise ValueError("source does not contain the exact requested scope")
    args.output.mkdir(parents=True, exist_ok=False)
    output_entries = []
    with (
        (args.source / manifest["weight_file"]).open("rb") as source,
        (args.output / "weights.bin.partial").open("wb") as target,
    ):
        for entry in entries:
            n, k = entry["shape"]
            source.seek(entry["data_offset"])
            data = source.read(n * k)
            if (
                len(data) != n * k
                or hashlib.sha256(data).hexdigest() != entry["packed_sha256"]
            ):
                raise ValueError("source FP8 payload differs from manifest")
            offset, _ = append_aligned(target, data)
            output_entries.append({**entry, "data_offset": offset})
    manifest.update(
        precision=f"fp8-{args.scope}-bf16",
        weight_file="weights.bin",
        tensors=output_entries,
        base_tensor_sha256={
            name: manifest["base_tensor_sha256"][name] for name in sorted(wanted)
        },
    )
    manifest["calibration"]["extracted_from_manifest_sha256"] = hashlib.sha256(
        source_bytes
    ).hexdigest()
    manifest["calibration"]["scope_extraction"] = (
        "Same calibrated E4M3 payload/scales; no recalibration or re-encoding"
    )
    (args.output / "manifest.json.partial").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    (args.output / "weights.bin.partial").replace(args.output / "weights.bin")
    (args.output / "manifest.json.partial").replace(args.output / "manifest.json")
    print(f"Published {args.scope}: {len(output_entries)} unchanged calibrated weights")


if __name__ == "__main__":
    main()
