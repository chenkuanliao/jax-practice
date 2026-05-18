#!/usr/bin/env python3
"""Plot mean runtime by Llama block version for 8-GPU runs."""

from __future__ import annotations

from plot_mean_runtime_4gpu import GPU_COUNT as _GPU_COUNT
from plot_mean_runtime_4gpu import ANNOTATIONS
from plot_mean_runtime_4gpu import FILE_PATTERN
from plot_mean_runtime_4gpu import INCLUDED_VERSIONS
from plot_mean_runtime_4gpu import _extract_mean_ms
from plot_mean_runtime_4gpu import _version_sort_key

import json
from pathlib import Path

import matplotlib.pyplot as plt


GPU_COUNT = 8


def main() -> None:
    _ = _GPU_COUNT
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
    bars = plt.bar(labels, means, color="seagreen")
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

    output_path = base_dir / "mean_runtime_8gpu.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved 8-GPU mean runtime plot to: {output_path}")


if __name__ == "__main__":
    main()
