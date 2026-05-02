"""
Autoregressive sampling helpers.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import tiktoken

from research.config import ModelConfig, SamplingConfig
from research.model import Model


def _sample_next_token(logits: jax.Array, key: jax.Array, temperature: float, top_k: int | None) -> jax.Array:
    if temperature == 0.0:
        return jnp.argmax(logits).astype(jnp.int32)

    logits = logits / temperature
    if top_k is not None:
        k = min(top_k, logits.shape[-1])
        values, _ = jax.lax.top_k(logits, k)
        threshold = values[-1]
        logits = jnp.where(logits < threshold, -jnp.inf, logits)

    return jax.random.categorical(key, logits).astype(jnp.int32)


def generate(
    model: Model,
    model_config: ModelConfig,
    sampling_config: SamplingConfig,
    tokenizer: tiktoken.Encoding,
    key: jax.Array,
) -> str:
    token_ids = tokenizer.encode(sampling_config.prompt)
    if not token_ids:
        token_ids = [tokenizer.eot_token]

    pad_token = tokenizer.eot_token
    tokens = jnp.asarray(token_ids, dtype=jnp.int32)
    for _ in range(sampling_config.max_new_tokens):
        key, sample_key = jax.random.split(key)
        context = tokens[-model_config.seq_len :]
        pad_len = model_config.seq_len - context.shape[0]
        if pad_len > 0:
            # Keep the model input shape fixed during sampling. Without this,
            # JAX recompiles for prompt length, prompt length + 1, and so on.
            padding = jnp.full((pad_len,), pad_token, dtype=jnp.int32)
            context = jnp.concatenate([padding, context], axis=0)

        logits = model(context[None, :])[0, -1]
        next_token = _sample_next_token(
            logits,
            sample_key,
            sampling_config.temperature,
            sampling_config.top_k,
        )
        tokens = jnp.concatenate([tokens, next_token[None]], axis=0)

    return tokenizer.decode([int(token) for token in tokens])


def write_sample(run_dir: str | Path, step: int, prompt: str, text: str) -> Path:
    path = Path(run_dir) / "samples" / f"sample_step_{step:06d}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"step: {step}\n"
        f"prompt:\n{prompt}\n"
        f"{'=' * 80}\n"
        f"sample:\n{text}\n",
        encoding="utf-8",
    )
    return path
