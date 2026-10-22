# SPDX-License-Identifier: AGPL-3.0-only
"""Select pinned LIBERO episodes excluded from calibration; prepare raw requests."""

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(calibration, output, count):
    selection = json.loads((calibration / "selection.json").read_text())
    excluded = {row["episode"] for row in selection["rows"]}
    info = json.loads((calibration / "meta/info.json").read_text())
    episodes = [
        json.loads(line)
        for line in (calibration / "meta/episodes.jsonl").read_text().splitlines()
    ]
    eligible = [
        episode for episode in episodes if episode["episode_index"] not in excluded
    ]
    assert 0 < count <= len(eligible)
    selected = {
        eligible[(2 * i + 1) * len(eligible) // (2 * count)]["episode_index"]
        for i in range(count)
    }
    rows, offset = [], 0
    for episode in episodes:
        index = episode["episode_index"]
        if index in selected:
            local = episode["length"] // 2
            rows.append(
                {
                    "episode": index,
                    "row": local,
                    "global_index": offset + local,
                    "task": episode["tasks"][0],
                    "file": info["data_path"].format(
                        episode_chunk=index // 1000, episode_index=index
                    ),
                    "noise_seed": 20261022 + len(rows),
                }
            )
        offset += episode["length"]
    assert len(rows) == count and not selected.intersection(excluded)
    output.mkdir(parents=True, exist_ok=True)
    for name in ["info.json", "episodes.jsonl", "tasks.jsonl"]:
        (output / "meta").mkdir(exist_ok=True)
        shutil.copyfile(calibration / "meta" / name, output / "meta" / name)
    base = f"https://huggingface.co/datasets/{selection['dataset']}/resolve/{selection['revision']}/"

    def fetch(row):
        destination = output / row["file"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            partial = destination.with_suffix(".partial")
            with (
                urllib.request.urlopen(base + row["file"], timeout=180) as response,
                partial.open("wb") as target,
            ):
                shutil.copyfileobj(response, target)
            partial.rename(destination)
        row["parquet_sha256"] = sha(destination)
        row["bytes"] = destination.stat().st_size
        print(f"Downloaded heldout episode {row['episode']}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, rows))
    result = {
        "dataset": selection["dataset"],
        "revision": selection["revision"],
        "selection": "16 midpoint quantiles of sorted eligible episodes; middle frame per episode"
        if count == 16
        else f"{count} midpoint quantiles of sorted eligible episodes; middle frame per episode",
        "excluded_calibration_episodes": sorted(excluded),
        "calibration_selection_sha256": sha(calibration / "selection.json"),
        "metadata_sha256": {
            name: sha(output / "meta" / name)
            for name in ["info.json", "episodes.jsonl", "tasks.jsonl"]
        },
        "rows": rows,
    }
    (output / "selection.json").write_text(json.dumps(result, indent=2) + "\n")


def requests(source, output):
    import io

    import numpy as np
    import pyarrow.parquet as pq
    from openpi.shared.image_tools import resize_with_pad
    from PIL import Image

    selection = json.loads((source / "selection.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    for row in selection["rows"]:
        path = source / row["file"]
        assert sha(path) == row["parquet_sha256"]
        record = pq.read_table(path).slice(row["row"], 1).to_pylist()[0]
        request = {}
        for key in ["image", "wrist_image"]:
            image = np.asarray(
                Image.open(io.BytesIO(record[key]["bytes"])).convert("RGB")
            )
            request[f"observation/{key}"] = (
                np.asarray(resize_with_pad(image, 224, 224)).ravel().tolist()
            )
        state = np.asarray(record["state"], dtype=np.float32)
        assert state.shape == (8,) and np.isfinite(state).all()
        request["observation/state"] = state.tolist()
        request["prompt"] = row["task"]
        request["noise"] = (
            np.random.default_rng(row["noise_seed"])
            .standard_normal((10, 32))
            .astype(np.float32)
            .ravel()
            .tolist()
        )
        name = f"heldout_episode_{row['episode']:06}_frame_{row['row']:04}"
        destination = output / f"{name}.request.json"
        destination.write_text(
            json.dumps(request, separators=(",", ":"), allow_nan=False) + "\n"
        )
        row["case"] = name
        row["request_sha256"] = sha(destination)
    selection["resize"] = "OpenPi resize_with_pad LINEAR, round-to-even uint8, 224x224"
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(f"Prepared {len(selection['rows'])} heldout requests", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["download", "requests"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()
    if args.mode == "download":
        download(args.source, args.output, args.count)
    else:
        requests(args.source, args.output)
