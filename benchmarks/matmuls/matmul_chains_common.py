import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    n: int
    dtype_name: str
    warmup_rounds: int
    timed_rounds: int
    topology: str
    mode: str = "jax_jit_named_sharding"


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    metric_label_width = 20
    print("=" * 72)
    print(f"JAX Benchmark ({result['name']})")
    print("=" * 72)
    print(f"Saved stats file : {output_path}")
    print(f"Total runs saved : {total_runs}")
    print(
        f"Config           : n={result['dim']}, dtype={result['dtype']}, "
        f"warmup={result['warmup_rounds']}, timed={result['timed_rounds']}"
    )
    print(f"Devices          : {result['devices']}")
    for expr in result["expressions"]:
        print("-" * 72)
        print(f"Expression       : {expr['expression']} ({expr['label_slug']})")
        print(f"{'Compile time (ms)':<{metric_label_width}}: {expr['compile_ms']:.3f}")
        print(
            f"{'Avg runtime (ms)':<{metric_label_width}}: "
            f"{expr['average_runtime_ms']:.3f}"
        )
        print(f"{'Median runtime (ms)':<{metric_label_width}}: {expr['median_ms']:.3f}")
        print(f"{'Stddev (ms)':<{metric_label_width}}: {expr['stdev_ms']:.3f}")
        print(
            f"{'Min/Max (ms)':<{metric_label_width}}: "
            f"{expr['min_ms']:.3f} / {expr['max_ms']:.3f}"
        )
        print(
            f"{'Effective TFLOPS':<{metric_label_width}}: "
            f"{expr['effective_tflops']:.3f}"
        )
    print("=" * 72)


def require_gpus(gpu_count):
    devices = jax.devices()
    if len(devices) < gpu_count:
        visible_devices = ",".join(str(i) for i in range(gpu_count))
        raise RuntimeError(
            f"Expected at least {gpu_count} visible GPU(s), got {len(devices)}. "
            f"Run with CUDA_VISIBLE_DEVICES={visible_devices}."
        )
    return devices[:gpu_count]


def prepare_for_profile(fn, args, warmup_rounds):
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

    return compile_ms, warmup_times_ms


def run_matmul_chain_benchmark(
    script_file,
    script_name,
    gpu_count,
    mesh_shape,
    topology,
    n=8192,
    dtype=jnp.float16,
    warmup_rounds=5,
    output_json=None,
    trace_dir=None,
    accelerator=None,
):
    output_json = output_json or f"{script_name}_stats.json"
    trace_dir = trace_dir or f"{script_name}_trace"
    devices = require_gpus(gpu_count)

    mesh_devices = np.array(devices).reshape(mesh_shape)
    mesh = Mesh(mesh_devices, axis_names=("island", "lane"))

    a_pspec = P("island", None)
    b_pspec = P(None, "lane")
    c_pspec = P("island", None)
    c_abc_pspec = P(None, "lane")
    d_pspec = P(None, "lane")
    e_pspec = P("island", None)
    f_pspec = P(None, "lane")
    g_pspec = P("island", None)
    h_pspec = P(None, "lane")
    out_pspec = P("island", "lane")

    a_sharding = NamedSharding(mesh, a_pspec)
    b_sharding = NamedSharding(mesh, b_pspec)
    c_sharding = NamedSharding(mesh, c_pspec)
    c_abc_sharding = NamedSharding(mesh, c_abc_pspec)
    d_sharding = NamedSharding(mesh, d_pspec)
    e_sharding = NamedSharding(mesh, e_pspec)
    f_sharding = NamedSharding(mesh, f_pspec)
    g_sharding = NamedSharding(mesh, g_pspec)
    h_sharding = NamedSharding(mesh, h_pspec)
    out_sharding = NamedSharding(mesh, out_pspec)

    @partial(
        jax.jit,
        in_shardings=(a_sharding, b_sharding),
        out_shardings=out_sharding,
    )
    def expr_ab(a, b):
        return jnp.linalg.multi_dot((a, b))

    @partial(
        jax.jit,
        in_shardings=(a_sharding, b_sharding, c_abc_sharding),
        out_shardings=out_sharding,
    )
    def expr_abc(a, b, c):
        return jnp.linalg.multi_dot((a, b, c))

    @partial(
        jax.jit,
        in_shardings=(a_sharding, b_sharding, c_sharding, d_sharding),
        out_shardings=out_sharding,
    )
    def expr_abcd(a, b, c, d):
        return jnp.linalg.multi_dot((a, b, c, d))

    @partial(
        jax.jit,
        in_shardings=(
            a_sharding,
            b_sharding,
            c_sharding,
            d_sharding,
            e_sharding,
            f_sharding,
            g_sharding,
            h_sharding,
        ),
        out_shardings=out_sharding,
    )
    def expr_abcdefgh(a, b, c, d, e, f, g, h):
        return jnp.linalg.multi_dot((a, b, c, d, e, f, g, h))

    with mesh:
        arrays = (
            make_sharded_array((n, n), a_sharding, seed=0, dtype=dtype),
            make_sharded_array((n, n), b_sharding, seed=1, dtype=dtype),
            make_sharded_array((n, n), c_sharding, seed=2, dtype=dtype),
            make_sharded_array((n, n), d_sharding, seed=3, dtype=dtype),
            make_sharded_array((n, n), e_sharding, seed=4, dtype=dtype),
            make_sharded_array((n, n), f_sharding, seed=5, dtype=dtype),
            make_sharded_array((n, n), g_sharding, seed=6, dtype=dtype),
            make_sharded_array((n, n), h_sharding, seed=7, dtype=dtype),
        )
        abc_arrays = (
            arrays[0],
            arrays[1],
            make_sharded_array((n, n), c_abc_sharding, seed=2, dtype=dtype),
        )

        specs = [
            ("a@b", "ab", expr_ab, arrays[:2], 1),
            ("a@b@c", "abc", expr_abc, abc_arrays, 2),
            ("a@b@c@d", "abcd", expr_abcd, arrays[:4], 3),
            ("a@b@c@d@e@f@g@h", "abcdefgh", expr_abcdefgh, arrays, 7),
        ]

        prepared = []
        for expression, label_slug, fn, args, gemm_count in specs:
            compile_ms, warmup_times_ms = prepare_for_profile(
                fn, args, warmup_rounds
            )
            prepared.append(
                {
                    "expression": expression,
                    "label_slug": label_slug,
                    "fn": fn,
                    "args": args,
                    "gemm_count": gemm_count,
                    "compile_ms": compile_ms,
                    "warmup_times_ms": warmup_times_ms,
                }
            )

        output_path = make_data_output_path(script_file, output_json)
        trace_path = output_path.parent / trace_dir
        trace_path.mkdir(parents=True, exist_ok=True)

        jax.profiler.start_trace(str(trace_path))
        expression_results = []
        for step_num, item in enumerate(prepared):
            s = time.perf_counter()
            with jax.profiler.StepTraceAnnotation(
                f"{item['label_slug']}_profiled_step", step_num=step_num
            ):
                y = item["fn"](*item["args"])
                y.block_until_ready()
            e = time.perf_counter()

            times_ms = [(e - s) * 1000.0]
            stats = summarize_times(times_ms)
            avg_s = stats["average_runtime_ms"] / 1000.0
            expression_results.append(
                {
                    "expression": item["expression"],
                    "label_slug": item["label_slug"],
                    "compile_ms": item["compile_ms"],
                    "warmup_times_ms": item["warmup_times_ms"],
                    "times_ms": times_ms,
                    **stats,
                    "effective_tflops": gemm_tflops(n, item["gemm_count"], avg_s),
                    "gemm_count": item["gemm_count"],
                    "status": "ok",
                    "profile_trace_dir": str(trace_path),
                }
            )
        jax.profiler.stop_trace()

    config = BenchmarkConfig(
        name=script_name,
        n=n,
        dtype_name=str(np.dtype(dtype).name),
        warmup_rounds=warmup_rounds,
        timed_rounds=1,
        topology=topology,
    )
    tags = ["matmul_chains", "v2-3", "flat", f"{gpu_count}gpu", "profile"]
    if accelerator:
        tags.append(accelerator.lower())

    extra = {
        "tags": tags,
        "jax_settings": {
            "order": [d.id for d in mesh.devices.flat],
            "a_sharding": str(a_pspec),
            "b_sharding": str(b_pspec),
            "c_sharding": str(c_pspec),
            "c_abc_sharding": str(c_abc_pspec),
            "d_sharding": str(d_pspec),
            "e_sharding": str(e_pspec),
            "f_sharding": str(f_pspec),
            "g_sharding": str(g_pspec),
            "h_sharding": str(h_pspec),
            "output_sharding": str(out_pspec),
        },
    }
    if accelerator:
        extra["accelerator"] = accelerator

    result = make_base_result(
        config=config,
        devices=devices,
        mesh_shape=mesh_shape,
        mesh_axis_names=("island", "lane"),
        expressions=expression_results,
        extra=extra,
    )

    merged = append_run_to_json(output_path, result)
    print_run_summary(result, output_path, len(merged.get("runs", [])))
    print(f"Profiler trace written to: {trace_path}")
    print(f"Open with: tensorboard --logdir {trace_path.parent}")
