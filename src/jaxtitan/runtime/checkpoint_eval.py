"""Standalone deterministic checkpoint validation eval."""

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np

from jaxtitan.data import dataset_manifest_sha256
from jaxtitan.errors import ContractError
from jaxtitan.infer import restore_inference_checkpoint
from jaxtitan.runtime.training import (
    _build_validation_eval_data,
    _validation_eval_row,
    _validation_eval_spec,
)
from jaxtitan.steps import make_eval_step


def evaluate_checkpoint(run_dir: str | Path, selector: str) -> dict[str, Any]:
    """Restore a retained checkpoint and run deterministic validation eval."""

    run_dir = Path(run_dir)
    restored = restore_inference_checkpoint(run_dir, selector)
    runtime_spec = restored.run_spec
    eval_spec = _validation_eval_spec(runtime_spec)
    if eval_spec is None:
        raise ContractError("checkpoint eval requires [[evals]] name = 'validation'")

    train_row = {
        "step": restored.metadata.checkpoint_step,
        "tokens_seen": restored.metadata.tokens_seen,
    }
    data = _build_validation_eval_data(runtime_spec)
    row = _validation_eval_row(
        eval_spec,
        data,
        make_eval_step(
            restored.graph,
            sharding=restored.sharding,
            state_template=restored.state.model,
            expected_batch_shape=(runtime_spec.training.global_batch_size, runtime_spec.training.seq_len),
        ),
        restored.sharding,
        restored.state,
        train_row,
    )
    manifest_path = runtime_spec.data.validation_manifest or runtime_spec.data.train_manifest
    payload = {
        "schema_version": 1,
        "status": "completed",
        "created_at": _utc_now(),
        "run_id": runtime_spec.run_id,
        "checkpoint": {
            "step": restored.metadata.checkpoint_step,
            "path": restored.metadata.checkpoint_path.as_posix(),
            "tokens_seen": restored.metadata.tokens_seen,
            "runtime_fingerprint": restored.metadata.runtime_fingerprint,
        },
        "eval": row,
        "data": {
            "manifest_path": manifest_path.as_posix(),
            "manifest_sha256": dataset_manifest_sha256(manifest_path),
        },
    }
    _write_eval_artifact(run_dir, restored.metadata.checkpoint_step, payload)
    return _normalize(payload)


def checkpoint_eval_to_json(payload: Mapping[str, Any]) -> str:
    """Serialize checkpoint eval payload as canonical JSON."""

    return _canonical_json(payload)


def format_checkpoint_eval(payload: Mapping[str, Any]) -> str:
    """Format checkpoint eval result for humans."""

    checkpoint = _require_mapping(payload.get("checkpoint"), "checkpoint")
    eval_row = _require_mapping(payload.get("eval"), "eval")
    return (
        f"evaluated checkpoint step={checkpoint['step']} path={checkpoint['path']} "
        f"loss={eval_row['loss']} tokens={eval_row['token_count']} batches={eval_row['num_batches']}"
    )


def _write_eval_artifact(run_dir: Path, step: int, payload: Mapping[str, Any]) -> None:
    path = run_dir / "evals" / "checkpoints" / f"{step:06d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(payload) + "\n")


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
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, jax.Array):
        array = np.asarray(jax.device_get(value))
        return array.item() if array.shape == () else array.tolist()
    return value
