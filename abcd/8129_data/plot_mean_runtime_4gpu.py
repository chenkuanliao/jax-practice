#!/usr/bin/env python3
"""Plot mean runtime by version for 4-GPU runs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt


GPU_COUNT = 4
FILE_PATTERN = re.compile(
    r"^(?P<label>[a-zA-Z0-9]+)_(?P<gpus>\d+)gpu_(?P<version>v\d+(?:-\d+)?)_stats\.json$"
)


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else float("nan")


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
    return float(_safe_mean(times))


def _version_sort_key(version: str) -> tuple[int, int]:
    # v1-3 -> (1, 3)
    major, minor = version[1:].split("-")
    return int(major), int(minor)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    stats_files = sorted(base_dir.glob("*_stats.json"))

    # keep one runtime per version for the selected GPU count
    per_version: dict[str, float] = {}
    label_name = None

    for file_path in stats_files:
        match = FILE_PATTERN.match(file_path.name)
        if not match:
            continue

        if int(match.group("gpus")) != GPU_COUNT:
            continue

        label_name = match.group("label")
        version = match.group("version")

        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        per_version[version] = _extract_mean_ms(payload)

    if not per_version:
        print(f"No matching {GPU_COUNT} GPU *_stats.json files found.")
        return

    versions = sorted(per_version.keys(), key=_version_sort_key)
    means = [per_version[v] for v in versions]

    plt.figure(figsize=(max(8, len(versions) * 1.4), 5))
    bars = plt.bar(versions, means, color="mediumpurple")
    plt.ylabel("Mean Runtime (ms)")
    plt.title(f"Mean Runtime by Version ({GPU_COUNT} GPU)")
    plt.suptitle(
        "v1: implicit output sharding | v2: explicit output sharding spec",
        fontsize=10,
        y=0.98,
    )
    plt.figtext(
        0.5,
        0.01,
        "-1: row+column sharding | -2: row sharding | -3: column sharding",
        ha="center",
        fontsize=9,
    )
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout(rect=(0, 0.04, 1, 0.92))

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
    run_label = f" for label '{label_name}'" if label_name else ""
    print(f"Saved 4-GPU mean runtime plot{run_label} to: {output_path}")


if __name__ == "__main__":
    main()
