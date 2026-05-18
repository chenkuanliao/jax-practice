#!/usr/bin/env python3
"""Plot 4-to-8 GPU speedup by Llama block version."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

from plot_mean_runtime_4gpu import ANNOTATIONS
from plot_mean_runtime_4gpu import INCLUDED_VERSIONS
from plot_mean_runtime_4gpu import _extract_mean_ms
from plot_mean_runtime_4gpu import _version_sort_key


FILE_PATTERN = re.compile(r"^llama_(?P<gpus>\d+)gpu_(?P<version>v\d+)_stats\.json$")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    means: dict[int, dict[str, float]] = {4: {}, 8: {}}

    for file_path in sorted(base_dir.glob("*_stats.json")):
        match = FILE_PATTERN.match(file_path.name)
        if not match:
            continue
        gpu_count = int(match.group("gpus"))
        if gpu_count not in means:
            continue
        version = match.group("version")
        if version not in INCLUDED_VERSIONS:
            continue
        with file_path.open("r", encoding="utf-8") as f:
            means[gpu_count][version] = _extract_mean_ms(json.load(f))

    versions = sorted(set(means[4]) & set(means[8]), key=_version_sort_key)
    if not versions:
        print("No paired 4-GPU and 8-GPU llama stats files found.")
        return

    speedups = [means[4][v] / means[8][v] for v in versions]
    labels = [f"{v}\n{ANNOTATIONS.get(v, '')}" for v in versions]

    plt.figure(figsize=(max(8, len(versions) * 1.5), 5))
    bars = plt.bar(labels, speedups, color="darkorange")
    plt.axhline(2.0, color="gray", linestyle="--", linewidth=1, label="ideal 4->8 scaling")
    plt.ylabel("Speedup (4-GPU mean / 8-GPU mean)")
    plt.title("Llama Block 4-to-8 GPU Speedup")
    plt.legend()
    plt.tight_layout()

    for bar, val in zip(bars, speedups):
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
    print(f"Saved GPU speedup plot to: {output_path}")


if __name__ == "__main__":
    main()
