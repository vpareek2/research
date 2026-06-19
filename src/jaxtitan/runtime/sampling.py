"""Post-hoc token sampling from retained checkpoints."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jaxtitan.errors import ContractError
from jaxtitan.infer import generate_tokens, restore_inference_checkpoint
from jaxtitan.services import LocalArtifactWriter
from jaxtitan.specs.generation import GenerationSpec


def sample_checkpoint(
    run_dir: str | Path,
    checkpoint: str,
    prompt_ids: str | Sequence[int],
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Restore one retained checkpoint, generate token ids, and append a sample artifact."""

    prompt = parse_prompt_ids(prompt_ids)
    generation = GenerationSpec(max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    restored = restore_inference_checkpoint(run_dir, checkpoint)
    if restored.run_spec.parallelism.context_parallel:
        raise ContractError("checkpoint sampling is not supported for context-parallel runs until CP KV-cache support lands")
    model_spec = restored.run_spec.model
    _validate_prompt_ids(prompt, vocab_size=model_spec.vocab_size)
    if generation.top_k is not None and generation.top_k > model_spec.vocab_size:
        raise ContractError(f"generation.top_k={generation.top_k} exceeds vocab size {model_spec.vocab_size}")

    result = generate_tokens(
        restored.graph,
        restored.state,
        model_spec,
        jnp.asarray([prompt], dtype=jnp.int32),
        generation,
    )
    payload = _normalize(
        {
            "schema_version": 1,
            "status": "completed",
            "created_at": _utc_now(),
            "run_id": restored.metadata.run_id,
            "checkpoint": {
                "selector": checkpoint,
                "step": restored.metadata.checkpoint_step,
                "path": restored.metadata.checkpoint_path.as_posix(),
                "tokens_seen": restored.metadata.tokens_seen,
                "runtime_fingerprint": restored.metadata.runtime_fingerprint,
            },
            "model": {
                "name": model_spec.name,
                "variant": model_spec.variant,
                "vocab_size": model_spec.vocab_size,
                "max_seq_len": model_spec.max_seq_len,
            },
            "sampling": {
                "max_new_tokens": generation.max_new_tokens,
                "temperature": generation.temperature,
                "top_k": generation.top_k,
                "top_p": None,
            },
            "prompt_ids": prompt,
            "generated_ids": jax.device_get(result.generated_ids)[0],
            "full_ids": jax.device_get(result.full_ids)[0],
            "logprobs": jax.device_get(result.logprobs)[0],
        }
    )
    LocalArtifactWriter(run_dir).append_checkpoint_sample(restored.metadata.checkpoint_step, payload)
    return payload


def parse_prompt_ids(value: str | Sequence[int]) -> list[int]:
    """Parse strict comma-separated token ids or validate an integer sequence."""

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ContractError("prompt_ids must contain at least one token id")
        parts = text.split(",")
        if any(not part.strip() for part in parts):
            raise ContractError(f"prompt_ids must be comma-separated integers, got {value!r}")
        parsed = []
        for part in parts:
            try:
                token_id = int(part.strip(), 10)
            except ValueError as exc:
                raise ContractError(f"prompt_ids must be comma-separated integers, got {value!r}") from exc
            parsed.append(token_id)
        return _validate_prompt_id_values(parsed)
    return _validate_prompt_id_values(list(value))


def checkpoint_sample_to_json(payload: Mapping[str, Any]) -> str:
    """Serialize checkpoint sample payload as canonical JSON."""

    return _canonical_json(payload)


def format_checkpoint_sample(payload: Mapping[str, Any]) -> str:
    """Format checkpoint sample result for humans."""

    checkpoint = _require_mapping(payload.get("checkpoint"), "checkpoint")
    sampling = _require_mapping(payload.get("sampling"), "sampling")
    return (
        f"sampled checkpoint step={checkpoint['step']} path={checkpoint['path']} "
        f"generated={sampling['max_new_tokens']} top_k={sampling['top_k']}"
    )


def _validate_prompt_id_values(prompt_ids: Sequence[int]) -> list[int]:
    if not prompt_ids:
        raise ContractError("prompt_ids must contain at least one token id")
    parsed = []
    for token_id in prompt_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ContractError("prompt_ids must be integers")
        if token_id < 0:
            raise ContractError(f"prompt_ids must be non-negative, got {token_id}")
        parsed.append(token_id)
    return parsed


def _validate_prompt_ids(prompt_ids: Sequence[int], *, vocab_size: int) -> None:
    for token_id in prompt_ids:
        if token_id >= vocab_size:
            raise ContractError(f"prompt id {token_id} is outside model vocab_size={vocab_size}")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be a JSON object")
    return value


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(_normalize(value), sort_keys=True, separators=(",", ":"))


def _normalize(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_normalize(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.item() if value.shape == () else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, jax.Array):
        array = np.asarray(jax.device_get(value))
        return array.item() if array.shape == () else array.tolist()
    return value
