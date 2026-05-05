#!/usr/bin/env python3
"""Compare runtime speedup when increasing GPU count.

This script scans stats JSON files in the same directory (e.g. abcd_4gpu_v2-1_stats.json),
pairs runs by version and expression label, and prints speedup from lower GPU count to
higher GPU count based on min/max/average runtime. The plot uses mean (average) runtime.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


FILE_PATTERN = re.compile(
    r"^(?P<label>[a-zA-Z0-9]+)_(?P<gpus>\d+)gpu_(?P<version>v\d+(?:-\d+)?)_stats\.json$"
)


def _extract_metrics(payload: dict[str, Any]) -> dict[str, float]:
    run = payload.get("latest") or (payload.get("runs") or [None])[0]
    if not run:
        raise ValueError("Missing run data")

    expressions = run.get("expressions") or []
    if not expressions:
        raise ValueError("Missing expressions data")

    expr = expressions[0]
    times = expr.get("times_ms") or []

    avg_ms = expr.get("average_runtime_ms")
    min_ms = expr.get("min_ms")
    max_ms = expr.get("max_ms")

    if avg_ms is None and times:
        avg_ms = sum(times) / len(times)
    if min_ms is None and times:
        min_ms = min(times)
    if max_ms is None and times:
        max_ms = max(times)

    return {
        "min_ms": float(min_ms),
        "max_ms": float(max_ms),
        "avg_ms": float(avg_ms),
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    stats_files = sorted(base_dir.glob("*_stats.json"))

    grouped: dict[tuple[str, str], dict[int, dict[str, float]]] = {}
    # key: (label, version) -> {gpu_count: metrics}
    for file_path in stats_files:
        match = FILE_PATTERN.match(file_path.name)
        if not match:
            continue

        label = match.group("label")
        version = match.group("version")
        gpu_count = int(match.group("gpus"))

        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        metrics = _extract_metrics(payload)

        grouped.setdefault((label, version), {})[gpu_count] = metrics

    if not grouped:
        print("No matching *_stats.json files found.")
        return

    print("GPU Scaling Speedup Report (runtime-based)")
    print("Speedup = lower_gpu_runtime / higher_gpu_runtime")
    print("-" * 72)

    plot_labels: list[str] = []
    mean_speedups: list[float] = []

    for (label, version), per_gpu in sorted(grouped.items()):
        gpu_counts = sorted(per_gpu.keys())
        if len(gpu_counts) < 2:
            continue

        low_gpu = gpu_counts[0]
        high_gpu = gpu_counts[-1]
        low = per_gpu[low_gpu]
        high = per_gpu[high_gpu]

        print(f"{label} {version}: {low_gpu} GPU -> {high_gpu} GPU")
        print(f"  min speedup:    {low['min_ms'] / high['min_ms']:.4f}x")
        print(f"  max speedup:    {low['max_ms'] / high['max_ms']:.4f}x")
        mean_speedup = low["avg_ms"] / high["avg_ms"]
        print(f"  mean speedup:   {mean_speedup:.4f}x")
        print()

        plot_labels.append(f"{label} {version}\n{low_gpu}-> {high_gpu} GPU")
        mean_speedups.append(mean_speedup)

    if not mean_speedups:
        print("No pairs with at least two GPU counts were found for plotting.")
        return

    plt.figure(figsize=(max(8, len(plot_labels) * 1.8), 5))
    bars = plt.bar(plot_labels, mean_speedups, color="steelblue")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.ylabel("Mean Speedup (x)")
    plt.title("Mean Runtime Speedup by GPU Scaling")
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

    for bar, val in zip(bars, mean_speedups):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}x",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    output_path = base_dir / "mean_gpu_speedup.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved mean speedup plot to: {output_path}")


if __name__ == "__main__":
    main()
