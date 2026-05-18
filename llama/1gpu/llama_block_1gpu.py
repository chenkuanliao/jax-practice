#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("NCCL_DEBUG", "WARN")

import jax
import jax.numpy as jnp


EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark one fp16 Llama block on one GPU.")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-sequences", type=int, default=1)
    parser.add_argument(
        "--batch-chunk",
        type=int,
        default=0,
        help=(
            "Number of packed sequences to process per compiled block chunk. "
            "0 chooses automatically; useful for large packed batches that would OOM."
        ),
    )
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--ffn-inter", type=int, default=14336)
    parser.add_argument("--ffn-gate-up", type=int, default=28672)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--json", type=str, default="llama_block_1gpu.json")
    parser.add_argument(
        "--attention-implementation",
        choices=["auto", "xla", "cudnn"],
        default="auto",
        help="Use auto unless you want to force XLA or cuDNN SDPA.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be positive.")
    if args.batch_sequences <= 0:
        raise ValueError("--batch-sequences must be positive.")
    if args.batch_chunk < 0:
        raise ValueError("--batch-chunk cannot be negative.")
    if args.batch_chunk > 0 and args.batch_sequences % args.batch_chunk != 0:
        raise ValueError("--batch-sequences must be divisible by --batch-chunk.")
    if args.hidden != args.q_heads * args.head_dim:
        raise ValueError("--hidden must equal --q-heads * --head-dim.")
    if args.q_heads % args.kv_heads != 0:
        raise ValueError("--q-heads must be divisible by --kv-heads for GQA.")
    if args.head_dim % 2 != 0:
        raise ValueError("--head-dim must be even for RoPE.")
    if args.ffn_gate_up != 2 * args.ffn_inter:
        raise ValueError("--ffn-gate-up must equal 2 * --ffn-inter for fused SiLU * up.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative.")


def rmsnorm(x: Any, gamma: Any, eps: float) -> Any:
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(variance + eps) * gamma


def rope(x: Any, cos: Any, sin: Any) -> Any:
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rot_even = even * cos[None, :, None, :] - odd * sin[None, :, None, :]
    rot_odd = even * sin[None, :, None, :] + odd * cos[None, :, None, :]
    return jnp.stack((rot_even, rot_odd), axis=-1).reshape(x.shape)


def fused_silu_mul(gate_up: Any) -> Any:
    gate, up = jnp.split(gate_up, 2, axis=-1)
    return jax.nn.silu(gate) * up


def make_rope_tables(seq: int, head_dim: int, dtype: Any) -> tuple[Any, Any]:
    pos = jnp.arange(seq, dtype=jnp.float32)[:, None]
    idx = jnp.arange(0, head_dim, 2, dtype=jnp.float32)[None, :]
    inv_freq = 1.0 / (10000.0 ** (idx / head_dim))
    angles = pos * inv_freq
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def make_array(shape: tuple[int, ...], seed: int, dtype: Any, device: Any) -> Any:
    key = jax.random.PRNGKey(seed)
    value = jax.random.normal(key, shape, dtype=dtype)
    return jax.device_put(value, device)


def flop_counts(args: argparse.Namespace) -> dict[str, int]:
    total_tokens = args.batch_sequences * args.seq_len
    projection = (
        2 * total_tokens * args.hidden * args.q_heads * args.head_dim
        + 2 * total_tokens * args.hidden * args.kv_heads * args.head_dim
        + 2 * total_tokens * args.hidden * args.kv_heads * args.head_dim
        + 2 * total_tokens * args.q_heads * args.head_dim * args.hidden
    )
    attention = 4 * args.batch_sequences * args.q_heads * args.seq_len * args.seq_len * args.head_dim
    ffn = (
        2 * total_tokens * args.hidden * args.ffn_gate_up
        + 2 * total_tokens * args.ffn_inter * args.hidden
    )
    return {
        "projection": projection,
        "attention": attention,
        "ffn": ffn,
        "total": projection + attention + ffn,
    }


def summarize(times_ms: list[float]) -> dict[str, float]:
    return {
        "average_runtime_ms": sum(times_ms) / len(times_ms),
        "median_ms": statistics.median(times_ms),
        "stdev_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }


def benchmark(fn: Any, fn_args: tuple[Any, ...], warmup: int, steps: int, flops: int) -> dict[str, Any]:
    compile_start = time.perf_counter()
    output = fn(*fn_args)
    output.block_until_ready()
    compile_ms = (time.perf_counter() - compile_start) * 1000.0

    warmup_times_ms = []
    for _ in range(warmup):
        start = time.perf_counter()
        output = fn(*fn_args)
        output.block_until_ready()
        warmup_times_ms.append((time.perf_counter() - start) * 1000.0)

    times_ms = []
    for _ in range(steps):
        start = time.perf_counter()
        output = fn(*fn_args)
        output.block_until_ready()
        times_ms.append((time.perf_counter() - start) * 1000.0)

    stats = summarize(times_ms)
    avg_s = stats["average_runtime_ms"] / 1000.0
    return {
        "compile_ms": compile_ms,
        "warmup_times_ms": warmup_times_ms,
        "times_ms": times_ms,
        **stats,
        "estimated_flops": flops,
        "effective_tflops": flops / avg_s / 1e12,
    }


def write_result(result: dict[str, Any], output_name: str) -> Path:
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / Path(output_name).name
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    validate_args(args)
    dtype = jnp.float16
    devices = jax.devices("gpu")
    if not devices:
        raise RuntimeError("No GPU is visible to JAX. This script expects CUDA_VISIBLE_DEVICES=0.")
    device = devices[0]
    attention_implementation = (
        None if args.attention_implementation == "auto" else args.attention_implementation
    )
    batch_chunk = args.batch_chunk
    if batch_chunk == 0:
        batch_chunk = 1 if args.batch_sequences >= 32 else args.batch_sequences
    num_batch_chunks = args.batch_sequences // batch_chunk

    @jax.jit
    def llama_block(x, gamma_a, gamma_f, wq, wk, wv, wo, w_gu, w_down, cos, sin):
        def block_chunk(x_chunk):
            norm_x = rmsnorm(x_chunk, gamma_a, EPS)
            q_3d = jnp.einsum("bld,dhm->blhm", norm_x, wq)
            k_3d = jnp.einsum("bld,dhm->blhm", norm_x, wk)
            v_3d = jnp.einsum("bld,dhm->blhm", norm_x, wv)
            q_r = rope(q_3d, cos, sin)
            k_r = rope(k_3d, cos, sin)
            attn_ctx = jax.nn.dot_product_attention(
                q_r,
                k_r,
                v_3d,
                is_causal=True,
                implementation=attention_implementation,
            )
            attn_out = jnp.einsum("blhm,hmd->bld", attn_ctx, wo)
            x_after = x_chunk + attn_out
            norm2 = rmsnorm(x_after, gamma_f, EPS)
            gate_up = norm2 @ w_gu
            intermed = fused_silu_mul(gate_up)
            ffn_out = intermed @ w_down
            return x_after + ffn_out

        if num_batch_chunks == 1:
            return block_chunk(x)

        x_chunks = x.reshape((num_batch_chunks, batch_chunk, args.seq_len, args.hidden))

        def scan_body(_, x_chunk):
            return None, block_chunk(x_chunk)

        _, output_chunks = jax.lax.scan(scan_body, None, x_chunks)
        return output_chunks.reshape(x.shape)

    fn_args = (
        make_array((args.batch_sequences, args.seq_len, args.hidden), 0, dtype, device),
        make_array((args.hidden,), 1, dtype, device),
        make_array((args.hidden,), 2, dtype, device),
        make_array((args.hidden, args.q_heads, args.head_dim), 3, dtype, device),
        make_array((args.hidden, args.kv_heads, args.head_dim), 4, dtype, device),
        make_array((args.hidden, args.kv_heads, args.head_dim), 5, dtype, device),
        make_array((args.q_heads, args.head_dim, args.hidden), 6, dtype, device),
        make_array((args.hidden, args.ffn_gate_up), 7, dtype, device),
        make_array((args.ffn_inter, args.hidden), 8, dtype, device),
        *[jax.device_put(x, device) for x in make_rope_tables(args.seq_len, args.head_dim, dtype)],
    )
    counts = flop_counts(args)
    timing = benchmark(llama_block, fn_args, args.warmup, args.steps, counts["total"])
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "name": "llama_block_1gpu",
        "framework": "jax",
        "jax_version": jax.__version__,
        "dtype": "float16",
        "devices": [
            {
                "id": device.id,
                "platform": device.platform,
                "device_kind": device.device_kind,
            }
        ],
        "topology": "single_gpu",
        "attention_implementation": args.attention_implementation,
        "shape": {
            "batch_sequences": args.batch_sequences,
            "seq_len": args.seq_len,
            "total_tokens": args.batch_sequences * args.seq_len,
            "packed_equal_length_sequences": args.batch_sequences > 1,
            "batch_chunk": batch_chunk,
            "num_batch_chunks": num_batch_chunks,
            "hidden": args.hidden,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "ffn_inter": args.ffn_inter,
            "ffn_gate_up": args.ffn_gate_up,
        },
        "rounds": {
            "warmup": args.warmup,
            "timed": args.steps,
        },
        "estimated_flop_counts": counts,
        "timing": timing,
    }
    output_path = write_result(result, args.json)
    print(f"Average runtime over {args.steps} runs: {timing['average_runtime_ms']:.3f} ms")
    print(f"JSON written to: {output_path}")


if __name__ == "__main__":
    main()
