#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("NCCL_DEBUG", "WARN")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


EPS = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark one fp16 Llama block on GPUs 0-3.")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-sequences", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--ffn-inter", type=int, default=14336)
    parser.add_argument("--ffn-gate-up", type=int, default=28672)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--json", type=str, default="llama_block_4gpu.json")
    parser.add_argument(
        "--strategy",
        choices=["tensor_parallel", "seq_model_parallel"],
        default="tensor_parallel",
        help="tensor_parallel keeps sequence replicated; seq_model_parallel shards sequence and heads.",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=["auto", "xla", "cudnn"],
        default="auto",
        help="Auto uses cuDNN SDPA on A100 and XLA elsewhere unless forced.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be positive.")
    if args.batch_sequences <= 0:
        raise ValueError("--batch-sequences must be positive.")
    if args.hidden != args.q_heads * args.head_dim:
        raise ValueError("--hidden must equal --q-heads * --head-dim.")
    if args.q_heads % args.kv_heads != 0:
        raise ValueError("--q-heads must be divisible by --kv-heads for GQA.")
    model_shards = 4 if args.strategy == "tensor_parallel" else 2
    if args.q_heads % model_shards != 0:
        raise ValueError(f"--q-heads must be divisible by {model_shards} for this strategy.")
    if args.kv_heads % model_shards != 0:
        raise ValueError(f"--kv-heads must be divisible by {model_shards} for this strategy.")
    if args.head_dim % 2 != 0:
        raise ValueError("--head-dim must be even for RoPE.")
    if args.ffn_gate_up != 2 * args.ffn_inter:
        raise ValueError("--ffn-gate-up must equal 2 * --ffn-inter for fused SiLU * up.")
    if args.ffn_inter % model_shards != 0 or args.ffn_gate_up % model_shards != 0:
        raise ValueError(f"--ffn-inter and --ffn-gate-up must be divisible by {model_shards}.")
    if args.strategy == "seq_model_parallel":
        can_shard_batch = args.batch_sequences > 1 and args.batch_sequences % 2 == 0
        if not can_shard_batch and args.seq_len % 2 != 0:
            raise ValueError("--seq-len must be divisible by 2 when sharding sequence length.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative.")


def resolve_attention_implementation(args: argparse.Namespace, devices: list[Any]) -> str | None:
    if args.attention_implementation != "auto":
        return args.attention_implementation
    if any("A100" in getattr(device, "device_kind", "") for device in devices):
        return "cudnn"
    return None


def rmsnorm(x: Any, gamma: Any, eps: float) -> Any:
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(variance + eps) * gamma


def rope(x: Any, cos: Any, sin: Any) -> Any:
    half = x.shape[-1] // 2
    first = x[..., :half]
    second = x[..., half:]
    cos_b = cos[:, :, None, :]
    sin_b = sin[:, :, None, :]
    rot_first = first * cos_b[..., :half] - second * sin_b[..., :half]
    rot_second = first * sin_b[..., half:] + second * cos_b[..., half:]
    return jnp.concatenate((rot_first, rot_second), axis=-1)


def fused_silu_mul(gate_up: Any) -> Any:
    gate, up = jnp.split(gate_up, 2, axis=-1)
    return jax.nn.silu(gate) * up


def make_rope_tables(batch_sequences: int, seq: int, head_dim: int, dtype: Any) -> tuple[Any, Any]:
    total_tokens = batch_sequences * seq
    pos = jnp.arange(total_tokens, dtype=jnp.float32)[:, None]
    idx = jnp.arange(0, head_dim, 2, dtype=jnp.float32)[None, :]
    inv_freq = 1.0 / (10000.0 ** (idx / head_dim))
    angles = pos * inv_freq
    cos = jnp.concatenate((jnp.cos(angles), jnp.cos(angles)), axis=-1)
    sin = jnp.concatenate((jnp.sin(angles), jnp.sin(angles)), axis=-1)
    return (
        cos.reshape((batch_sequences, seq, head_dim)).astype(dtype),
        sin.reshape((batch_sequences, seq, head_dim)).astype(dtype),
    )


def make_array(shape: tuple[int, ...], sharding: Any, seed: int, dtype: Any) -> Any:
    key = jax.random.PRNGKey(seed)
    value = jax.random.normal(key, shape, dtype=dtype)
    return jax.device_put(value, sharding)


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
    if len(devices) < 4:
        raise RuntimeError(
            f"Expected at least 4 visible GPUs, got {len(devices)}. "
            "Run with CUDA_VISIBLE_DEVICES=0,1,2,3."
        )
    devices = devices[:4]
    if args.strategy == "tensor_parallel":
        mesh = Mesh(np.array(devices), axis_names=("model",))
        x_spec = P(None, None, None)
        cos_sin_spec = P(None, None, None)
        q_spec = P(None, None, "model", None)
        kv_spec = P(None, None, "model", None)
        attn_ctx_spec = P(None, None, "model", None)
        ffn_inter_spec = P(None, None, "model")
        mesh_shape = [4]
        mesh_axis_names = ["model"]
        strategy_description = "4-way tensor parallelism with replicated sequence"
    else:
        mesh = Mesh(np.array(devices).reshape((2, 2)), axis_names=("seq", "model"))
        if args.batch_sequences > 1 and args.batch_sequences % 2 == 0:
            x_spec = P("seq", None, None)
            q_spec = P("seq", None, "model", None)
            kv_spec = P("seq", None, "model", None)
            attn_ctx_spec = P("seq", None, "model", None)
            ffn_inter_spec = P("seq", None, "model")
            seq_parallel_axis = "batch_sequences"
        else:
            x_spec = P(None, "seq", None)
            q_spec = P(None, "seq", "model", None)
            kv_spec = P(None, "seq", "model", None)
            attn_ctx_spec = P(None, "seq", "model", None)
            ffn_inter_spec = P(None, "seq", "model")
            seq_parallel_axis = "seq_len"
        cos_sin_spec = P(None, None, None)
        mesh_shape = [2, 2]
        mesh_axis_names = ["seq", "model"]
        strategy_description = f"2D {seq_parallel_axis} and tensor parallelism"
    attention_implementation = resolve_attention_implementation(args, devices)

    shardings = {
        "x": NamedSharding(mesh, x_spec),
        "gamma": NamedSharding(mesh, P(None)),
        "q_weight": NamedSharding(mesh, P(None, "model", None)),
        "kv_weight": NamedSharding(mesh, P(None, "model", None)),
        "o_weight": NamedSharding(mesh, P("model", None, None)),
        "gu_weight": NamedSharding(mesh, P(None, "model")),
        "down_weight": NamedSharding(mesh, P("model", None)),
        "cos_sin": NamedSharding(mesh, cos_sin_spec),
        "x_spec": x_spec,
        "cos_sin_spec": cos_sin_spec,
        "q": q_spec,
        "kv": kv_spec,
        "attn_ctx": attn_ctx_spec,
        "ffn_inter": ffn_inter_spec,
    }

    @partial(
        jax.jit,
        in_shardings=(
            shardings["x"],
            shardings["gamma"],
            shardings["gamma"],
            shardings["q_weight"],
            shardings["kv_weight"],
            shardings["kv_weight"],
            shardings["o_weight"],
            shardings["gu_weight"],
            shardings["down_weight"],
            shardings["cos_sin"],
            shardings["cos_sin"],
        ),
        out_shardings=shardings["x"],
    )
    def llama_block(x, gamma_a, gamma_f, wq, wk, wv, wo, w_gu, w_down, cos, sin):
        norm_x = rmsnorm(x, gamma_a, EPS)
        norm_x = jax.lax.with_sharding_constraint(norm_x, shardings["x_spec"])
        q_3d = jnp.einsum("bld,dhm->blhm", norm_x, wq)
        k_3d = jnp.einsum("bld,dhm->blhm", norm_x, wk)
        v_3d = jnp.einsum("bld,dhm->blhm", norm_x, wv)
        q_3d = jax.lax.with_sharding_constraint(q_3d, shardings["q"])
        k_3d = jax.lax.with_sharding_constraint(k_3d, shardings["kv"])
        v_3d = jax.lax.with_sharding_constraint(v_3d, shardings["kv"])
        q_r = rope(q_3d, cos, sin)
        k_r = rope(k_3d, cos, sin)
        q_r = jax.lax.with_sharding_constraint(q_r, shardings["q"])
        k_r = jax.lax.with_sharding_constraint(k_r, shardings["kv"])
        attn_ctx = jax.nn.dot_product_attention(
            q_r,
            k_r,
            v_3d,
            is_causal=True,
            implementation=attention_implementation,
        )
        attn_ctx = jax.lax.with_sharding_constraint(attn_ctx, shardings["attn_ctx"])
        attn_out = jnp.einsum("blhm,hmd->bld", attn_ctx, wo)
        attn_out = jax.lax.with_sharding_constraint(attn_out, shardings["x_spec"])
        x_after = x + attn_out
        x_after = jax.lax.with_sharding_constraint(x_after, shardings["x_spec"])
        norm2 = rmsnorm(x_after, gamma_f, EPS)
        norm2 = jax.lax.with_sharding_constraint(norm2, shardings["x_spec"])
        gate_up = norm2 @ w_gu
        gate_up = jax.lax.with_sharding_constraint(gate_up, shardings["ffn_inter"])
        intermed = fused_silu_mul(gate_up)
        intermed = jax.lax.with_sharding_constraint(intermed, shardings["ffn_inter"])
        ffn_out = intermed @ w_down
        ffn_out = jax.lax.with_sharding_constraint(ffn_out, shardings["x_spec"])
        return x_after + ffn_out

    with mesh:
        fn_args = (
            make_array((args.batch_sequences, args.seq_len, args.hidden), shardings["x"], 0, dtype),
            make_array((args.hidden,), shardings["gamma"], 1, dtype),
            make_array((args.hidden,), shardings["gamma"], 2, dtype),
            make_array((args.hidden, args.q_heads, args.head_dim), shardings["q_weight"], 3, dtype),
            make_array((args.hidden, args.kv_heads, args.head_dim), shardings["kv_weight"], 4, dtype),
            make_array((args.hidden, args.kv_heads, args.head_dim), shardings["kv_weight"], 5, dtype),
            make_array((args.q_heads, args.head_dim, args.hidden), shardings["o_weight"], 6, dtype),
            make_array((args.hidden, args.ffn_gate_up), shardings["gu_weight"], 7, dtype),
            make_array((args.ffn_inter, args.hidden), shardings["down_weight"], 8, dtype),
            *[
                jax.device_put(x, shardings["cos_sin"])
                for x in make_rope_tables(args.batch_sequences, args.seq_len, args.head_dim, dtype)
            ],
        )
        counts = flop_counts(args)
        timing = benchmark(llama_block, fn_args, args.warmup, args.steps, counts["total"])

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "name": f"llama_block_4gpu_{args.strategy}",
        "framework": "jax",
        "jax_version": jax.__version__,
        "dtype": "float16",
        "devices": [
            {
                "id": device.id,
                "platform": device.platform,
                "device_kind": device.device_kind,
            }
            for device in devices
        ],
        "topology": "first_4_gpus_single_numa_nvlink_island",
        "mesh": {
            "shape": mesh_shape,
            "axis_names": mesh_axis_names,
            "device_ids_row_major": [device.id for device in np.array(devices).flat],
        },
        "sharding": {
            "strategy": strategy_description,
            "x": str(x_spec),
            "q_weight": "PartitionSpec(None, 'model', None)",
            "kv_weight": "PartitionSpec(None, 'model', None)",
            "o_weight": "PartitionSpec('model', None, None)",
            "gu_weight": "PartitionSpec(None, 'model')",
            "down_weight": "PartitionSpec('model', None)",
            "cos_sin": str(cos_sin_spec),
            "qkv_activations": str(q_spec),
            "attention_context": str(attn_ctx_spec),
            "ffn_intermediate": str(ffn_inter_spec),
        },
        "attention_implementation": attention_implementation or "xla",
        "attention_implementation_requested": args.attention_implementation,
        "shape": {
            "batch_sequences": args.batch_sequences,
            "seq_len": args.seq_len,
            "total_tokens": args.batch_sequences * args.seq_len,
            "packed_equal_length_sequences": args.batch_sequences > 1,
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
