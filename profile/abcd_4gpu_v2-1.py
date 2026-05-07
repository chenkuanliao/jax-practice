import argparse
from functools import partial
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("NCCL_DEBUG", "WARN")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from benchmark_api import BenchmarkConfig
from benchmark_api import append_run_to_json
from benchmark_api import gemm_tflops
from benchmark_api import make_base_result
from benchmark_api import make_data_output_path
from benchmark_api import make_sharded_array
from benchmark_api import print_run_summary
from benchmark_api import summarize_times


def require_4_gpus():
    devices = jax.devices()
    if len(devices) < 4:
        raise RuntimeError(
            f"Expected at least 4 visible GPUs, got {len(devices)}. "
            "Run with CUDA_VISIBLE_DEVICES=0,1,2,3."
        )
    return devices[:4]


def run_benchmark(
    n=8192,
    dtype=jnp.float32,
    warmup_rounds=5,
    output_json="abcd_4gpu_v2-1_stats.json",
    trace_dir="abcd_4gpu_v2-1_trace",
):
    devices = require_4_gpus()

    mesh_devices = np.array(devices).reshape(2, 2)
    mesh = Mesh(mesh_devices, axis_names=("island", "lane"))

    a_pspec = P("island", "lane")
    b_pspec = P("island", "lane")
    c_pspec = P("island", "lane")
    d_pspec = P("island", "lane")
    out_pspec = P("island", "lane")

    a_sharding = NamedSharding(mesh, a_pspec)
    b_sharding = NamedSharding(mesh, b_pspec)
    c_sharding = NamedSharding(mesh, c_pspec)
    d_sharding = NamedSharding(mesh, d_pspec)
    out_sharding = NamedSharding(mesh, out_pspec)

    @partial(
        jax.jit,
        in_shardings=(a_sharding, b_sharding, c_sharding, d_sharding),
        out_shardings=out_sharding,
    )
    def expr_abcd(a, b, c, d):
        x = a @ b
        x = jax.lax.with_sharding_constraint(x, out_pspec)
        x = x @ c
        x = jax.lax.with_sharding_constraint(x, out_pspec)
        x = x @ d
        return jax.lax.with_sharding_constraint(x, out_pspec)

    with mesh:
        A = make_sharded_array((n, n), a_sharding, seed=0, dtype=dtype)
        B = make_sharded_array((n, n), b_sharding, seed=1, dtype=dtype)
        C = make_sharded_array((n, n), c_sharding, seed=2, dtype=dtype)
        D = make_sharded_array((n, n), d_sharding, seed=3, dtype=dtype)
        t0 = time.perf_counter()
        y = expr_abcd(A, B, C, D)
        y.block_until_ready()
        t1 = time.perf_counter()
        compile_ms = (t1 - t0) * 1000.0

        warmup_times_ms = []
        for _ in range(warmup_rounds):
            s = time.perf_counter()
            y = expr_abcd(A, B, C, D)
            y.block_until_ready()
            e = time.perf_counter()
            warmup_times_ms.append((e - s) * 1000.0)

        output_path = make_data_output_path(__file__, output_json)
        trace_path = output_path.parent / trace_dir
        trace_path.mkdir(parents=True, exist_ok=True)

        jax.profiler.start_trace(str(trace_path))
        s = time.perf_counter()
        with jax.profiler.StepTraceAnnotation("abcd_profiled_step", step_num=0):
            y = expr_abcd(A, B, C, D)
            y.block_until_ready()
        e = time.perf_counter()
        jax.profiler.stop_trace()

        times_ms = [(e - s) * 1000.0]
        stats = summarize_times(times_ms)
        avg_s = stats["average_runtime_ms"] / 1000.0
        expression_result = {
            "expression": "a@b@c@d",
            "label_slug": "abcd",
            "compile_ms": compile_ms,
            "warmup_times_ms": warmup_times_ms,
            "times_ms": times_ms,
            **stats,
            "effective_tflops": gemm_tflops(n, 3, avg_s),
            "gemm_count": 3,
            "status": "ok",
            "profile_trace_dir": str(trace_path),
        }

    config = BenchmarkConfig(
        name="abcd_4gpu_v2-1",
        n=n,
        dtype_name=str(np.dtype(dtype).name),
        warmup_rounds=warmup_rounds,
        timed_rounds=1,
        topology="1_island_4gpu_v100",
    )
    result = make_base_result(
        config=config,
        devices=devices,
        mesh_shape=(2, 2),
        mesh_axis_names=("island", "lane"),
        expressions=[expression_result],
        extra={
            "jax_settings": {
                "order": [d.id for d in mesh.devices.flat],
                "a_sharding": str(a_pspec),
                "b_sharding": str(b_pspec),
                "c_sharding": str(c_pspec),
                "d_sharding": str(d_pspec),
                "output_sharding": str(out_pspec),
            }
        },
    )

    merged = append_run_to_json(output_path, result)
    print_run_summary(result, output_path, len(merged.get("runs", [])))
    print(f"Profiler trace written to: {trace_path}")
    print(f"Open with: tensorboard --logdir {trace_path.parent}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32", "float64"],
    )
    parser.add_argument("--json", type=str, default="abcd_4gpu_v2-1_stats.json")
    parser.add_argument("--trace-dir", type=str, default="abcd_4gpu_v2-1_trace")
    args = parser.parse_args()

    run_benchmark(
        n=args.n,
        dtype=getattr(jnp, args.dtype),
        warmup_rounds=args.warmup,
        output_json=args.json,
        trace_dir=args.trace_dir,
    )


if __name__ == "__main__":
    main()
