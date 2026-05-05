#!/usr/bin/env python3
"""Plot mean runtime by Llama block version for 4-GPU runs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt


GPU_COUNT = 4
FILE_PATTERN = re.compile(r"^llama_(?P<gpus>\d+)gpu_(?P<version>v\d+)_stats\.json$")
INCLUDED_VERSIONS = {"v1", "v2"}
ANNOTATIONS = {
    "v1": "baseline",
    "v2": "minimal constraints",
}


def _extract_mean_ms(payload: dict[str, Any]) -> float:
    run = payload.get("latest") or (payload.get("runs") or [None])[0]
    if not run:
        raise ValueError("Missing run data")
    expressions = run.get("expressions") or []
    if not expressions:
        raise ValueError("Missing expressions data")
    expr = expressions[0]
    avg = expr.get("average_runtime_ms")
    if avg is not None:
        return float(avg)
    times = expr.get("times_ms") or []
    if not times:
        raise ValueError("Missing average_runtime_ms and times_ms")
    return float(mean(times))


def _version_sort_key(version: str) -> int:
    return int(version[1:])


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    per_version: dict[str, float] = {}
    for file_path in sorted(base_dir.glob("*_stats.json")):
        match = FILE_PATTERN.match(file_path.name)
        if not match or int(match.group("gpus")) != GPU_COUNT:
            continue
        version = match.group("version")
        if version not in INCLUDED_VERSIONS:
            continue
        with file_path.open("r", encoding="utf-8") as f:
            per_version[version] = _extract_mean_ms(json.load(f))

    if not per_version:
        print(f"No matching {GPU_COUNT} GPU llama *_stats.json files found.")
        return

    versions = sorted(per_version.keys(), key=_version_sort_key)
    means = [per_version[v] for v in versions]
    labels = [f"{v}\n{ANNOTATIONS.get(v, '')}" for v in versions]

    plt.figure(figsize=(max(8, len(versions) * 1.5), 5))
    bars = plt.bar(labels, means, color="steelblue")
    plt.ylabel("Mean Runtime (ms)")
    plt.title(f"Llama Block Mean Runtime ({GPU_COUNT} GPU)")
    plt.tight_layout()

    for bar, val in zip(bars, means):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    output_path = base_dir / "mean_runtime_4gpu.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved 4-GPU mean runtime plot to: {output_path}")


if __name__ == "__main__":
    main()
