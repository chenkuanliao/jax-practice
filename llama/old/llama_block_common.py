#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
from typing import Any

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
from benchmark_api import require_4_gpus
from benchmark_api import require_8_gpus
from benchmark_api import run_expression_benchmark


EPS = 1e-6


@dataclass(frozen=True)
class LlamaShape:
    seq: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int


STRATEGY_LABELS = {
    "v1": "user_baseline",
    "v2": "minimal_manual_constraints",
}


def add_common_args(parser: argparse.ArgumentParser, default_json: str) -> None:
    parser.add_argument("--seq", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--ffn", type=int, default=11008)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32", "float64"],
    )
    parser.add_argument("--json", type=str, default=default_json)


def validate_shape(shape: LlamaShape) -> None:
    if shape.heads % shape.kv_heads != 0:
        raise ValueError("--heads must be divisible by --kv-heads for GQA.")
    if shape.hidden != shape.heads * shape.head_dim:
        raise ValueError("--hidden must equal --heads * --head-dim.")
    if shape.head_dim % 2 != 0:
        raise ValueError("--head-dim must be even for RoPE.")


def rmsnorm(x: Any, gamma: Any, eps: float = EPS) -> Any:
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(variance + eps) * gamma


def rope(x: Any, cos: Any, sin: Any) -> Any:
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rot_even = even * cos[:, None, :] - odd * sin[:, None, :]
    rot_odd = even * sin[:, None, :] + odd * cos[:, None, :]
    return jnp.stack((rot_even, rot_odd), axis=-1).reshape(x.shape)


def causal_gqa_attention(q: Any, k: Any, v: Any) -> Any:
    repeat = q.shape[1] // k.shape[1]
    k_heads = jnp.repeat(k, repeat, axis=1)
    v_heads = jnp.repeat(v, repeat, axis=1)
    scale = jnp.asarray(q.shape[-1], dtype=q.dtype) ** -0.5
    scores = jnp.einsum("shm,thm->hst", q, k_heads) * scale
    mask = jnp.tril(jnp.ones((q.shape[0], q.shape[0]), dtype=bool))
    scores = jnp.where(mask[None, :, :], scores, jnp.asarray(-1.0e30, dtype=q.dtype))
    probs = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("hst,thm->shm", probs, v_heads)


def fused_silu_mul(gate_up: Any) -> Any:
    gate, up = jnp.split(gate_up, 2, axis=-1)
    return jax.nn.silu(gate) * up


def make_rope_tables(seq: int, head_dim: int, dtype: Any) -> tuple[Any, Any]:
    pos = jnp.arange(seq, dtype=jnp.float32)[:, None]
    idx = jnp.arange(0, head_dim, 2, dtype=jnp.float32)[None, :]
    inv_freq = 1.0 / (10000.0 ** (idx / head_dim))
    angles = pos * inv_freq
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def flop_counts(shape: LlamaShape) -> dict[str, int]:
    q_proj = 2 * shape.seq * shape.hidden * shape.heads * shape.head_dim
    k_proj = 2 * shape.seq * shape.hidden * shape.kv_heads * shape.head_dim
    v_proj = 2 * shape.seq * shape.hidden * shape.kv_heads * shape.head_dim
    attn_scores = 2 * shape.heads * shape.seq * shape.seq * shape.head_dim
    attn_mix = 2 * shape.heads * shape.seq * shape.seq * shape.head_dim
    out_proj = 2 * shape.seq * shape.heads * shape.head_dim * shape.hidden
    ffn_up = 2 * shape.seq * shape.hidden * (2 * shape.ffn)
    ffn_down = 2 * shape.seq * shape.ffn * shape.hidden
    projection = q_proj + k_proj + v_proj + out_proj
    attention = attn_scores + attn_mix
    ffn = ffn_up + ffn_down
    return {
        "projection": projection,
        "attention": attention,
        "ffn": ffn,
        "total": projection + attention + ffn,
    }


def _pspecs(variant: str, y_size: int, kv_heads: int) -> dict[str, Any]:
    kv_y = P(None, "y", None) if kv_heads % y_size == 0 else P(None, None, None)
    kv_activation = P("x", "y", None) if kv_heads % y_size == 0 else P("x", None, None)
    base = {
        "x": P("x", None),
        "gamma": P(None),
        "q_weight": P(None, "y", None),
        "kv_weight": kv_y,
        "o_weight": P("y", None, None),
        "gu_weight": P(None, "y"),
        "down_weight": P("y", None),
        "cos_sin": P("x", None),
        "q": P("x", "y", None),
        "kv": kv_activation,
        "rope_q": P("x", "y", None),
        "rope_k": kv_activation,
        "attn_context": P("x", "y", None),
        "attn_out": P("x", None),
        "ffn_intermediate": P("x", "y"),
        "output": P("x", None),
        "fallback": None if kv_heads % y_size == 0 else "replicated_kv_heads",
    }
    return base


def _settings_from_pspecs(pspecs: dict[str, Any], mesh: Mesh) -> dict[str, Any]:
    return {
        "order": [d.id for d in mesh.devices.flat],
        "activation_sharding": str(pspecs["x"]),
        "q_sharding": str(pspecs["q"]),
        "kv_sharding": str(pspecs["kv"]),
        "rope_q_sharding": str(pspecs["rope_q"]),
        "rope_k_sharding": str(pspecs["rope_k"]),
        "attention_context_sharding": str(pspecs["attn_context"]),
        "attention_output_sharding": str(pspecs["attn_out"]),
        "ffn_intermediate_sharding": str(pspecs["ffn_intermediate"]),
        "final_output_sharding": str(pspecs["output"]),
        "weight_shardings": {
            "wq": str(pspecs["q_weight"]),
            "wk": str(pspecs["kv_weight"]),
            "wv": str(pspecs["kv_weight"]),
            "wo": str(pspecs["o_weight"]),
            "w_gu": str(pspecs["gu_weight"]),
            "w_down": str(pspecs["down_weight"]),
            "gamma_a": str(pspecs["gamma"]),
            "gamma_f": str(pspecs["gamma"]),
        },
        "fallback": pspecs["fallback"],
    }


def run_llama_benchmark(
    *,
    gpu_count: int,
    variant: str,
    script_file: str,
    output_json: str,
    shape: LlamaShape,
    dtype: Any,
    dtype_name: str,
    warmup_rounds: int,
    timed_rounds: int,
) -> None:
    validate_shape(shape)
    devices = require_4_gpus() if gpu_count == 4 else require_8_gpus()
    mesh_shape = (2, 2) if gpu_count == 4 else (2, 4)
    topology = "1_island_4gpu_v100" if gpu_count == 4 else "2_islands_8gpu_v100"
    mesh = Mesh(np.array(devices).reshape(mesh_shape), axis_names=("x", "y"))
    pspecs = _pspecs(variant, mesh_shape[1], shape.kv_heads)
    shardings = {name: NamedSharding(mesh, spec) for name, spec in pspecs.items() if name != "fallback"}

    minimal_constraints = variant == "v2"

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
        out_shardings=shardings["output"],
    )
    def llama_block(x, gamma_a, gamma_f, wq, wk, wv, wo, w_gu, w_down, cos, sin):
        norm_x = rmsnorm(x, gamma_a, EPS)
        if not minimal_constraints:
            norm_x = jax.lax.with_sharding_constraint(norm_x, pspecs["x"])

        q_3d = jnp.einsum("sd,dhm->shm", norm_x, wq)
        k_3d = jnp.einsum("sd,dhm->shm", norm_x, wk)
        v_3d = jnp.einsum("sd,dhm->shm", norm_x, wv)
        if not minimal_constraints:
            q_3d = jax.lax.with_sharding_constraint(q_3d, pspecs["q"])
            k_3d = jax.lax.with_sharding_constraint(k_3d, pspecs["kv"])
            v_3d = jax.lax.with_sharding_constraint(v_3d, pspecs["kv"])

        q_r = rope(q_3d, cos, sin)
        k_r = rope(k_3d, cos, sin)
        if not minimal_constraints:
            q_r = jax.lax.with_sharding_constraint(q_r, pspecs["rope_q"])
            k_r = jax.lax.with_sharding_constraint(k_r, pspecs["rope_k"])

        attn_ctx = causal_gqa_attention(q_r, k_r, v_3d)
        if not minimal_constraints:
            attn_ctx = jax.lax.with_sharding_constraint(attn_ctx, pspecs["attn_context"])

        attn_out = jnp.einsum("shm,hmd->sd", attn_ctx, wo)
        attn_out = jax.lax.with_sharding_constraint(attn_out, pspecs["attn_out"])
        x_after = x + attn_out
        if not minimal_constraints:
            x_after = jax.lax.with_sharding_constraint(x_after, pspecs["x"])

        norm2 = rmsnorm(x_after, gamma_f, EPS)
        if not minimal_constraints:
            norm2 = jax.lax.with_sharding_constraint(norm2, pspecs["x"])

        gate_up = norm2 @ w_gu
        if not minimal_constraints:
            gate_up = jax.lax.with_sharding_constraint(gate_up, pspecs["ffn_intermediate"])
        intermed = fused_silu_mul(gate_up)
        if not minimal_constraints:
            intermed = jax.lax.with_sharding_constraint(intermed, pspecs["ffn_intermediate"])

        ffn_out = intermed @ w_down
        ffn_out = jax.lax.with_sharding_constraint(ffn_out, pspecs["output"])
        output = x_after + ffn_out
        return jax.lax.with_sharding_constraint(output, pspecs["output"])

    with mesh:
        args = (
            make_sharded_array((shape.seq, shape.hidden), shardings["x"], seed=0, dtype=dtype),
            make_sharded_array((shape.hidden,), shardings["gamma"], seed=1, dtype=dtype),
            make_sharded_array((shape.hidden,), shardings["gamma"], seed=2, dtype=dtype),
            make_sharded_array(
                (shape.hidden, shape.heads, shape.head_dim),
                shardings["q_weight"],
                seed=3,
                dtype=dtype,
            ),
            make_sharded_array(
                (shape.hidden, shape.kv_heads, shape.head_dim),
                shardings["kv_weight"],
                seed=4,
                dtype=dtype,
            ),
            make_sharded_array(
                (shape.hidden, shape.kv_heads, shape.head_dim),
                shardings["kv_weight"],
                seed=5,
                dtype=dtype,
            ),
            make_sharded_array(
                (shape.heads, shape.head_dim, shape.hidden),
                shardings["o_weight"],
                seed=6,
                dtype=dtype,
            ),
            make_sharded_array((shape.hidden, 2 * shape.ffn), shardings["gu_weight"], seed=7, dtype=dtype),
            make_sharded_array((shape.ffn, shape.hidden), shardings["down_weight"], seed=8, dtype=dtype),
        )
        cos, sin = make_rope_tables(shape.seq, shape.head_dim, dtype)
        rope_args = (
            jax.device_put(cos, shardings["cos_sin"]),
            jax.device_put(sin, shardings["cos_sin"]),
        )
        counts = flop_counts(shape)
        expression_result = run_expression_benchmark(
            fn=llama_block,
            args=args + rope_args,
            expression="llama_decoder_block",
            label_slug="llama_block",
            estimated_flops=counts["total"],
            warmup_rounds=warmup_rounds,
            timed_rounds=timed_rounds,
        )

    name = f"llama_{gpu_count}gpu_{variant}"
    config = BenchmarkConfig(
        name=name,
        seq=shape.seq,
        hidden=shape.hidden,
        heads=shape.heads,
        kv_heads=shape.kv_heads,
        head_dim=shape.head_dim,
        ffn=shape.ffn,
        dtype_name=dtype_name,
        warmup_rounds=warmup_rounds,
        timed_rounds=timed_rounds,
        topology=topology,
        strategy=STRATEGY_LABELS[variant],
    )
    result = make_base_result(
        config=config,
        devices=devices,
        mesh_shape=mesh_shape,
        mesh_axis_names=("x", "y"),
        expressions=[expression_result],
        flop_counts=counts,
        jax_settings=_settings_from_pspecs(pspecs, mesh),
    )
    output_path = make_data_output_path(script_file, output_json)
    merged = append_run_to_json(output_path, result)
    print_run_summary(result, output_path, len(merged.get("runs", [])))


def main_for_variant(gpu_count: int, variant: str, script_file: str) -> None:
    default_json = f"llama_{gpu_count}gpu_{variant}_stats.json"
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_json)
    args = parser.parse_args()
    shape = LlamaShape(
        seq=args.seq,
        hidden=args.hidden,
        heads=args.heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        ffn=args.ffn,
    )
    run_llama_benchmark(
        gpu_count=gpu_count,
        variant=variant,
        script_file=script_file,
        output_json=args.json,
        shape=shape,
        dtype=getattr(jnp, args.dtype),
        dtype_name=args.dtype,
        warmup_rounds=args.warmup,
        timed_rounds=args.steps,
    )
