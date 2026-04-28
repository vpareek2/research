"""
Lightweight full-context inference benchmark helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import tiktoken

from config import ModelConfig, dtype_from_name
from kv_cache import init_kv_cache
from model import Model


MODE = "kv_cache_decode_loop_prefill"
DEFAULT_PROMPTS = [
    "Once upon a time, a small machine learned to tell a careful story about",
    "The capital of France is Paris. The next useful fact to remember is",
    "def add_numbers(a, b):\n    return a + b\n\n# Now write a function that",
]


@dataclass(frozen=True)
class InferenceBenchConfig:
    prompt_tokens: int
    decode_tokens: int
    batch_size: int = 1
    warmup_decode_tokens: int = 1


def default_bench_config(model_config: ModelConfig) -> InferenceBenchConfig:
    prompt_tokens = min(512, max(1, model_config.seq_len // 2), model_config.seq_len - 1)
    return InferenceBenchConfig(
        prompt_tokens=prompt_tokens,
        decode_tokens=min(128, max(1, model_config.seq_len - prompt_tokens)),
    )


@nnx.jit
def prefill_forward(model: Model, input_ids: jax.Array, cache) -> tuple[jax.Array, Any]:
    return model.prefill(input_ids, cache)


@nnx.jit
def decode_next(model: Model, input_ids: jax.Array, cache) -> tuple[jax.Array, jax.Array, Any]:
    logits, cache = model.decode_one(input_ids, cache)
    next_token = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    return next_token[:, None], next_token, cache


def benchmark_inference(
    model: Model,
    tokenizer: tiktoken.Encoding,
    model_config: ModelConfig,
    *,
    run_dir: str | Path,
    checkpoint_step: int,
    prompts: list[str] | None = None,
    bench_config: InferenceBenchConfig | None = None,
) -> dict[str, Any]:
    prompts = prompts or DEFAULT_PROMPTS
    bench_config = bench_config or default_bench_config(model_config)
    input_batches = [
        _make_input_ids(prompt, tokenizer, bench_config.prompt_tokens)
        for prompt in prompts
    ]

    _warmup(model, input_batches[0], model_config, bench_config.warmup_decode_tokens)

    prefill_elapsed = 0.0
    decode_elapsed = 0.0
    first_decode_elapsed = 0.0
    ttft_values = []
    for input_ids in input_batches:
        cache = init_kv_cache(model_config, bench_config.batch_size, dtype_from_name(model.precision.compute_dtype))
        start = time.perf_counter()
        logits, cache = prefill_forward(model, input_ids, cache)
        logits.block_until_ready()
        prompt_prefill_sec = time.perf_counter() - start
        prefill_elapsed += prompt_prefill_sec

        next_input_ids = jnp.argmax(logits, axis=-1).astype(jnp.int32)[:, None]
        start = time.perf_counter()
        next_input_ids, token, cache = decode_next(model, next_input_ids, cache)
        token.block_until_ready()
        first_decode_sec = time.perf_counter() - start
        first_decode_elapsed += first_decode_sec
        decode_elapsed += first_decode_sec
        ttft_values.append(prompt_prefill_sec + first_decode_sec)

        start = time.perf_counter()
        for _ in range(max(0, bench_config.decode_tokens - 1)):
            next_input_ids, token, cache = decode_next(model, next_input_ids, cache)
        token.block_until_ready()
        decode_elapsed += time.perf_counter() - start

    prompt_count = len(input_batches)
    total_prefill_tokens = prompt_count * bench_config.batch_size * bench_config.prompt_tokens
    total_decode_tokens = prompt_count * bench_config.batch_size * bench_config.decode_tokens
    return {
        "run_dir": str(run_dir),
        "checkpoint_step": checkpoint_step,
        "mode": MODE,
        "batch_size": bench_config.batch_size,
        "prompt_count": prompt_count,
        "prompt_tokens": bench_config.prompt_tokens,
        "decode_tokens": bench_config.decode_tokens,
        "seq_len": model_config.seq_len,
        "prefill_elapsed_sec": prefill_elapsed,
        "decode_elapsed_sec": decode_elapsed,
        "first_decode_sec": first_decode_elapsed / prompt_count if prompt_count else 0.0,
        "ttft_sec": sum(ttft_values) / len(ttft_values) if ttft_values else 0.0,
        "prefill_tokens_per_sec": total_prefill_tokens / prefill_elapsed if prefill_elapsed > 0 else 0.0,
        "decode_tokens_per_sec": total_decode_tokens / decode_elapsed if decode_elapsed > 0 else 0.0,
        "memory_used_bytes": None,
        "memory_peak_bytes": None,
    }


def write_inference_artifacts(run_dir: str | Path, checkpoint_step: int, metrics: dict[str, Any]) -> tuple[Path, Path]:
    eval_dir = Path(run_dir) / "evals" / f"step_{checkpoint_step}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = eval_dir / "inference_metrics.json"
    summary_path = eval_dir / "inference_summary.md"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(format_inference_summary(metrics), encoding="utf-8")
    return metrics_path, summary_path


def format_inference_summary(metrics: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Inference Benchmark",
            "",
            f"- run: `{metrics['run_dir']}`",
            f"- checkpoint_step: `{metrics['checkpoint_step']}`",
            f"- mode: `{metrics['mode']}`",
            f"- batch_size: `{metrics['batch_size']}`",
            f"- prompt_count: `{metrics['prompt_count']}`",
            f"- prompt_tokens: `{metrics['prompt_tokens']}`",
            f"- decode_tokens: `{metrics['decode_tokens']}`",
            f"- prefill_tokens_per_sec: `{metrics['prefill_tokens_per_sec']:.2f}`",
            f"- decode_tokens_per_sec: `{metrics['decode_tokens_per_sec']:.2f}`",
            f"- ttft_sec: `{metrics['ttft_sec']:.6f}`",
            f"- first_decode_sec: `{metrics.get('first_decode_sec', 0.0):.6f}`",
        ]
    ) + "\n"


def _make_input_ids(prompt: str, tokenizer: tiktoken.Encoding, prompt_tokens: int) -> jax.Array:
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        token_ids = [tokenizer.eot_token]
    token_ids = token_ids[-prompt_tokens:]
    if len(token_ids) < prompt_tokens:
        token_ids = [tokenizer.eot_token] * (prompt_tokens - len(token_ids)) + token_ids
    return jnp.asarray(token_ids, dtype=jnp.int32)[None, :]


def _warmup(model: Model, input_ids: jax.Array, model_config: ModelConfig, decode_tokens: int):
    cache = init_kv_cache(model_config, input_ids.shape[0], dtype_from_name(model.precision.compute_dtype))
    logits, cache = prefill_forward(model, input_ids, cache)
    logits.block_until_ready()
    next_input_ids = jnp.argmax(logits, axis=-1).astype(jnp.int32)[:, None]
    for _ in range(decode_tokens):
        next_input_ids, token, cache = decode_next(model, next_input_ids, cache)
    token.block_until_ready()
