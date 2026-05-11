import argparse
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("NCCL_DEBUG", "WARN")

import jax.numpy as jnp

from matmul_chains_common import run_matmul_chain_benchmark


GPU_COUNT = 1
MESH_SHAPE = (1, 1)
TOPOLOGY = "1_gpu"
SCRIPT_NAME = "matmul_1gpu"


def run_benchmark(
    n=8192,
    dtype=jnp.float16,
    warmup_rounds=5,
    output_json=f"{SCRIPT_NAME}_stats.json",
    trace_dir=f"{SCRIPT_NAME}_trace",
    accelerator=None,
):
    run_matmul_chain_benchmark(
        script_file=__file__,
        script_name=SCRIPT_NAME,
        gpu_count=GPU_COUNT,
        mesh_shape=MESH_SHAPE,
        topology=TOPOLOGY,
        n=n,
        dtype=dtype,
        warmup_rounds=warmup_rounds,
        output_json=output_json,
        trace_dir=trace_dir,
        accelerator=accelerator,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32", "float64"],
    )
    parser.add_argument("--json", type=str, default=f"{SCRIPT_NAME}_stats.json")
    parser.add_argument("--trace-dir", type=str, default=f"{SCRIPT_NAME}_trace")
    parser.add_argument("--accelerator", type=str, default=None)
    args = parser.parse_args()

    run_benchmark(
        n=args.n,
        dtype=getattr(jnp, args.dtype),
        warmup_rounds=args.warmup,
        output_json=args.json,
        trace_dir=args.trace_dir,
        accelerator=args.accelerator,
    )


if __name__ == "__main__":
    main()
