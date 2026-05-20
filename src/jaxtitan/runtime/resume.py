"""Deterministic resume compatibility metadata."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np

from jaxtitan.data import data_pipeline_compat_payload, dataset_manifest_sha256
from jaxtitan.errors import ContractError
from jaxtitan.services import CheckpointRestore
from jaxtitan.specs.run import RunSpec

RESUME_METADATA_SCHEMA_VERSION = 1
RESUME_COMPAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class ResumeCompatibility:
    """Canonical compatibility payload and fingerprint for resume checks."""

    payload: dict[str, Any]
    runtime_fingerprint: str


def build_resume_compat(spec: RunSpec) -> ResumeCompatibility:
    """Build the deterministic compatibility payload for an effective runtime spec."""

    payload = {
        "seed": spec.seed,
        "model": _normalize(spec.model),
        "optimizer": _normalize(spec.optimizer),
        "mesh": _normalize(spec.mesh),
        "data": {
            "train_manifest": spec.data.train_manifest.as_posix(),
            "train_manifest_sha256": dataset_manifest_sha256(spec.data.train_manifest),
            "tokenizer_id": spec.data.tokenizer_id,
            "validation_manifest": None
            if spec.data.validation_manifest is None
            else spec.data.validation_manifest.as_posix(),
            "validation_manifest_sha256": None
            if spec.data.validation_manifest is None
            else dataset_manifest_sha256(spec.data.validation_manifest),
            "training_pipeline": data_pipeline_compat_payload(
                spec.data.train_manifest,
                tokenizer_id=spec.data.tokenizer_id,
                split="train",
                seq_len=spec.training.seq_len,
                batch_size=spec.training.global_batch_size,
                order=spec.data.order,
                shuffle_seed=spec.data.shuffle_seed,
                worker_count=spec.data.worker_count,
                worker_buffer_size=spec.data.worker_buffer_size,
                prefetch=spec.data.prefetch,
                document_buffer_size=spec.data.document_buffer_size,
                document_refill_size=spec.data.document_refill_size,
            ),
        },
        "training": {
            "precision": spec.training.precision,
            "seq_len": spec.training.seq_len,
            "global_batch_size": spec.training.global_batch_size,
            "gradient_accumulation_steps": spec.training.gradient_accumulation_steps,
            "eval_every_steps": spec.training.eval_every_steps,
            "grad_clip_norm": spec.training.grad_clip_norm,
        },
    }
    payload = _normalize(payload)
    return ResumeCompatibility(payload=payload, runtime_fingerprint=_hash(payload))


def checkpoint_metadata(
    spec: RunSpec,
    row: Mapping[str, Any],
    *,
    reason: str,
    eval_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build canonical metadata saved with a checkpoint."""

    compat = build_resume_compat(spec)
    step = _required_row_int(row, "step")
    tokens_seen = _required_row_int(row, "tokens_seen")
    return {
        "schema_version": RESUME_METADATA_SCHEMA_VERSION,
        "compat_version": RESUME_COMPAT_VERSION,
        "run_id": spec.run_id,
        "checkpoint": {
            "step": step,
            "tokens_seen": tokens_seen,
            "reason": reason,
        },
        "metrics": {
            "train_loss": _required_row_float(row, "loss"),
            "eval_loss": None if eval_row is None else _required_row_float(eval_row, "loss"),
        },
        "runtime_fingerprint": compat.runtime_fingerprint,
        "compatibility": compat.payload,
        "mutable_controls": {
            "target_tokens": spec.training.target_tokens,
            "log_every_steps": spec.training.log_every_steps,
            "checkpoint_every_steps": spec.training.checkpoint_every_steps,
        },
    }


def validate_resume_metadata(metadata: Mapping[str, Any], current_spec: RunSpec) -> None:
    """Validate checkpoint metadata against the current effective runtime spec."""

    _require_int_equal(metadata, "schema_version", RESUME_METADATA_SCHEMA_VERSION)
    _require_int_equal(metadata, "compat_version", RESUME_COMPAT_VERSION)
    run_id = metadata.get("run_id")
    if run_id != current_spec.run_id:
        raise ContractError(f"resume metadata run_id mismatch: checkpoint={run_id!r} current={current_spec.run_id!r}")

    current = build_resume_compat(current_spec)
    stored_payload = _require_mapping(metadata.get("compatibility"), "compatibility")
    stored_fingerprint = metadata.get("runtime_fingerprint")
    if not isinstance(stored_fingerprint, str) or not stored_fingerprint:
        raise ContractError("resume metadata runtime_fingerprint must be a non-empty string")
    expected_fingerprint = _hash(_normalize(stored_payload))
    if stored_fingerprint != expected_fingerprint:
        raise ContractError("resume metadata runtime_fingerprint does not match compatibility payload")
    if stored_fingerprint != current.runtime_fingerprint:
        mismatch = _preferred_mismatch(stored_payload, current.payload)
        raise ContractError(f"resume compatibility mismatch at {mismatch}")

    checkpoint = _require_mapping(metadata.get("checkpoint"), "checkpoint")
    _required_int(checkpoint, "step", "checkpoint")
    _required_int(checkpoint, "tokens_seen", "checkpoint")


def validate_resume_compat(restored: CheckpointRestore, current_spec: RunSpec) -> None:
    """Validate a restored checkpoint before accepting it into runtime."""

    validate_resume_metadata(restored.metadata, current_spec)
    checkpoint = _require_mapping(restored.metadata.get("checkpoint"), "checkpoint")
    metadata_step = _required_int(checkpoint, "step", "checkpoint")
    metadata_tokens = _required_int(checkpoint, "tokens_seen", "checkpoint")
    state_step = _scalar_int(restored.train_state.step)
    state_tokens = _scalar_int(restored.train_state.tokens_seen)
    if restored.step != metadata_step:
        raise ContractError(
            f"resume checkpoint step mismatch: manager={restored.step} metadata={metadata_step}"
        )
    if state_step != metadata_step:
        raise ContractError(
            f"resume checkpoint step mismatch: train_state={state_step} metadata={metadata_step}"
        )
    if state_tokens != metadata_tokens:
        raise ContractError(
            f"resume checkpoint tokens_seen mismatch: train_state={state_tokens} metadata={metadata_tokens}"
        )
    if restored.host_state.dataset != restored.dataset_state:
        raise ContractError("resume checkpoint host_state.dataset does not match dataset_state")


def _required_row_int(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if not isinstance(value, int):
        raise ContractError(f"checkpoint metadata row {key} must be an integer")
    return value


def _required_row_float(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if not isinstance(value, int | float):
        raise ContractError(f"checkpoint metadata row {key} must be numeric")
    return float(value)


def _require_int_equal(raw: Mapping[str, Any], key: str, expected: int) -> None:
    value = raw.get(key)
    if value != expected:
        raise ContractError(f"resume metadata {key} must be {expected}, got {value!r}")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"resume metadata {name} must be a JSON object")
    return value


def _required_int(raw: Mapping[str, Any], key: str, name: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int):
        raise ContractError(f"resume metadata {name}.{key} must be an integer")
    return value


def _first_mismatch(left: Any, right: Any, path: str = "compatibility") -> str:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        left_keys = set(left)
        right_keys = set(right)
        missing = sorted(right_keys - left_keys)
        if missing:
            return f"{path}.{missing[0]}"
        extra = sorted(left_keys - right_keys)
        if extra:
            return f"{path}.{extra[0]}"
        for key in sorted(left_keys):
            mismatch = _first_mismatch(left[key], right[key], f"{path}.{key}")
            if mismatch:
                return mismatch
        return ""
    if isinstance(left, list) and isinstance(right, list):
        for idx, (left_item, right_item) in enumerate(zip(left, right, strict=False)):
            mismatch = _first_mismatch(left_item, right_item, f"{path}[{idx}]")
            if mismatch:
                return mismatch
        if len(left) != len(right):
            return f"{path}.length"
        return ""
    return "" if left == right else path


def _preferred_mismatch(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    for key in ("seed", "model", "optimizer", "mesh", "training", "data"):
        if left.get(key) != right.get(key):
            return _first_mismatch(left.get(key), right.get(key), f"compatibility.{key}")
    return _first_mismatch(left, right)


def _hash(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


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
    return value


def _scalar_int(value: Any) -> int:
    return int(np.asarray(jax.device_get(value)).item())
