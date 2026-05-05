#!/usr/bin/env python3

import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import numpy as np


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    n: int
    dtype_name: str
    warmup_rounds: int
    timed_rounds: int
    topology: str = "2_islands_8gpu_v100"
    mode: str = "jax_jit_named_sharding"


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_8_gpus() -> list[Any]:
    devices = jax.devices()
    if len(devices) != 8:
        raise RuntimeError(
            f"Expected 8 visible GPUs, got {len(devices)}. "
            "Run with CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7."
        )
    return devices


def make_sharded_array(shape: tuple[int, ...], sharding: Any, seed: int, dtype: Any) -> Any:
    key = jax.random.PRNGKey(seed)
    x = jax.random.normal(key, shape, dtype=dtype)
    return jax.device_put(x, sharding)


def summarize_times(times_ms: list[float]) -> dict[str, float]:
    return {
        "average_runtime_ms": sum(times_ms) / len(times_ms),
        "median_ms": statistics.median(times_ms),
        "stdev_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }


def gemm_tflops(n: int, gemm_count: int, seconds: float) -> float:
    flops = gemm_count * 2 * (n**3)
    return flops / seconds / 1e12


def make_data_output_path(script_file: str, output_json: str) -> Path:
    script_dir = Path(script_file).resolve().parent
    data_dir = script_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / Path(output_json).name


def append_run_to_json(output_path: Path, result: dict[str, Any]) -> dict[str, Any]:
    merged = {
        "runs": [result],
        "latest": result,
        "last_updated_at_utc": iso_now(),
    }
    output_path.write_text(json.dumps(merged, indent=2))
    return merged


def run_expression_benchmark(
    fn: Any,
    args: tuple[Any, ...],
    expression: str,
    label_slug: str,
    gemm_count: int,
    warmup_rounds: int,
    timed_rounds: int,
    n: int,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    y = fn(*args)
    y.block_until_ready()
    t1 = time.perf_counter()
    compile_ms = (t1 - t0) * 1000.0

    warmup_times_ms = []
    for _ in range(warmup_rounds):
        s = time.perf_counter()
        y = fn(*args)
        y.block_until_ready()
        e = time.perf_counter()
        warmup_times_ms.append((e - s) * 1000.0)

    times_ms = []
    for _ in range(timed_rounds):
        s = time.perf_counter()
        y = fn(*args)
        y.block_until_ready()
        e = time.perf_counter()
        times_ms.append((e - s) * 1000.0)

    stats = summarize_times(times_ms)
    avg_s = stats["average_runtime_ms"] / 1000.0
    return {
        "expression": expression,
        "label_slug": label_slug,
        "compile_ms": compile_ms,
        "warmup_times_ms": warmup_times_ms,
        "times_ms": times_ms,
        **stats,
        "effective_tflops": gemm_tflops(n, gemm_count, avg_s),
        "gemm_count": gemm_count,
        "status": "ok",
    }


def make_base_result(
    config: BenchmarkConfig,
    devices: list[Any],
    mesh_shape: tuple[int, int],
    mesh_axis_names: tuple[str, str],
    expressions: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "created_at_utc": iso_now(),
        "created_unix_s": time.time(),
        "name": config.name,
        "mode": config.mode,
        "dim": config.n,
        "dtype": config.dtype_name,
        "devices": [d.id for d in devices],
        "topology": config.topology,
        "mesh_shape": list(mesh_shape),
        "mesh_axis_names": list(mesh_axis_names),
        "warmup_rounds": config.warmup_rounds,
        "timed_rounds": config.timed_rounds,
        "expressions": expressions,
    }
    if extra:
        result.update(extra)
    return result


def print_run_summary(result: dict[str, Any], output_path: Path, total_runs: int) -> None:
    expr = result["expressions"][0]
    metric_label_width = 20
    print("=" * 72)
    print(f"8-GPU JAX Benchmark ({result['name']})")
    print("=" * 72)
    print(f"Saved stats file : {output_path}")
    print(f"Total runs saved : {total_runs}")
    print(
        f"Config           : n={result['dim']}, dtype={result['dtype']}, "
        f"warmup={result['warmup_rounds']}, timed={result['timed_rounds']}"
    )
    print(f"Devices          : {result['devices']}")
    print("-" * 72)
    print(f"{'Compile time (ms)':<{metric_label_width}}: {expr['compile_ms']:.3f}")
    print(
        f"{'Avg runtime (ms)':<{metric_label_width}}: {expr['average_runtime_ms']:.3f}"
    )
    print(f"{'Median runtime (ms)':<{metric_label_width}}: {expr['median_ms']:.3f}")
    print(f"{'Stddev (ms)':<{metric_label_width}}: {expr['stdev_ms']:.3f}")
    print(
        f"{'Min/Max (ms)':<{metric_label_width}}: "
        f"{expr['min_ms']:.3f} / {expr['max_ms']:.3f}"
    )
    print(
        f"{'Effective TFLOPS':<{metric_label_width}}: {expr['effective_tflops']:.3f}"
    )
    print("=" * 72)

