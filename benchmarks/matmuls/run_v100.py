import argparse
import subprocess
import sys
from pathlib import Path


ACCELERATOR = "V100"
BENCHMARKS = ("matmul_1gpu.py", "matmul_4gpu.py", "matmul_8gpu.py")


def run_all(n, warmup, dtype):
    script_dir = Path(__file__).resolve().parent
    label = ACCELERATOR.lower()

    for benchmark in BENCHMARKS:
        stem = Path(benchmark).stem
        cmd = [
            sys.executable,
            str(script_dir / benchmark),
            "--n",
            str(n),
            "--warmup",
            str(warmup),
            "--dtype",
            dtype,
            "--accelerator",
            ACCELERATOR,
            "--json",
            f"{stem}_{label}_stats.json",
            "--trace-dir",
            f"{stem}_{label}_trace",
        ]
        print(f"Running {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)


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
    args = parser.parse_args()

    run_all(n=args.n, warmup=args.warmup, dtype=args.dtype)


if __name__ == "__main__":
    main()
