import argparse
from functools import partial
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("NCCL_DEBUG", "WARN")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from benchmark_api import BenchmarkConfig
from benchmark_api import append_run_to_json
from benchmark_api import make_base_result
from benchmark_api import make_data_output_path
from benchmark_api import make_sharded_array
from benchmark_api import print_run_summary
from benchmark_api import run_expression_benchmark


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
    warmup_rounds=2,
    timed_rounds=10,
    output_json="abcdefgh_4gpu_v2-3_stats.json",
):
    devices = require_4_gpus()

    mesh_devices = np.array(devices).reshape(2, 2)
    mesh = Mesh(mesh_devices, axis_names=("island", "lane"))

    a_pspec = P("island", None)
    b_pspec = P(None, "lane")
    c_pspec = P("island", None)
    d_pspec = P(None, "lane")
    e_pspec = P("island", None)
    f_pspec = P(None, "lane")
    g_pspec = P("island", None)
    h_pspec = P(None, "lane")
    out_pspec = P("island", "lane")

    a_sharding = NamedSharding(mesh, a_pspec)
    b_sharding = NamedSharding(mesh, b_pspec)
    c_sharding = NamedSharding(mesh, c_pspec)
    d_sharding = NamedSharding(mesh, d_pspec)
    e_sharding = NamedSharding(mesh, e_pspec)
    f_sharding = NamedSharding(mesh, f_pspec)
    g_sharding = NamedSharding(mesh, g_pspec)
    h_sharding = NamedSharding(mesh, h_pspec)
    out_sharding = NamedSharding(mesh, out_pspec)

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
        # ((a@b)@(c@d)) @ ((e@f)@(g@h))
        ab = a @ b
        ab = jax.lax.with_sharding_constraint(ab, out_pspec)
        cd = c @ d
        cd = jax.lax.with_sharding_constraint(cd, out_pspec)
        left = ab @ cd
        left = jax.lax.with_sharding_constraint(left, out_pspec)
        ef = e @ f
        ef = jax.lax.with_sharding_constraint(ef, out_pspec)
        gh = g @ h
        gh = jax.lax.with_sharding_constraint(gh, out_pspec)
        right = ef @ gh
        right = jax.lax.with_sharding_constraint(right, out_pspec)
        x = left @ right
        return jax.lax.with_sharding_constraint(x, out_pspec)

    with mesh:
        A = make_sharded_array((n, n), a_sharding, seed=0, dtype=dtype)
        B = make_sharded_array((n, n), b_sharding, seed=1, dtype=dtype)
        C = make_sharded_array((n, n), c_sharding, seed=2, dtype=dtype)
        D = make_sharded_array((n, n), d_sharding, seed=3, dtype=dtype)
        E = make_sharded_array((n, n), e_sharding, seed=4, dtype=dtype)
        F = make_sharded_array((n, n), f_sharding, seed=5, dtype=dtype)
        G = make_sharded_array((n, n), g_sharding, seed=6, dtype=dtype)
        H = make_sharded_array((n, n), h_sharding, seed=7, dtype=dtype)
        expression_result = run_expression_benchmark(
            fn=expr_abcdefgh,
            args=(A, B, C, D, E, F, G, H),
            expression="((a@b)@(c@d))@((e@f)@(g@h))",
            label_slug="abcdefgh_nested",
            gemm_count=7,
            warmup_rounds=warmup_rounds,
            timed_rounds=timed_rounds,
            n=n,
        )

    config = BenchmarkConfig(
        name="abcdefgh_4gpu_v2-3",
        n=n,
        dtype_name=str(np.dtype(dtype).name),
        warmup_rounds=warmup_rounds,
        timed_rounds=timed_rounds,
        topology="1_island_4gpu_v100",
    )
    result = make_base_result(
        config=config,
        devices=devices,
        mesh_shape=(2, 2),
        mesh_axis_names=("island", "lane"),
        expressions=[expression_result],
        extra={
            "tags": ["abcdefgh", "v2-3", "nested", "4gpu"],
            "jax_settings": {
                "order": [d.id for d in mesh.devices.flat],
                "a_sharding": str(a_pspec),
                "b_sharding": str(b_pspec),
                "c_sharding": str(c_pspec),
                "d_sharding": str(d_pspec),
                "e_sharding": str(e_pspec),
                "f_sharding": str(f_pspec),
                "g_sharding": str(g_pspec),
                "h_sharding": str(h_pspec),
                "output_sharding": str(out_pspec),
            }
        },
    )

    output_path = make_data_output_path(__file__, output_json)
    merged = append_run_to_json(output_path, result)
    print_run_summary(result, output_path, len(merged.get("runs", [])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32", "float64"],
    )
    parser.add_argument("--json", type=str, default="abcdefgh_4gpu_v2-3_stats.json")
    args = parser.parse_args()

    run_benchmark(
        n=args.n,
        dtype=getattr(jnp, args.dtype),
        warmup_rounds=args.warmup,
        timed_rounds=args.steps,
        output_json=args.json,
    )


if __name__ == "__main__":
    main()
