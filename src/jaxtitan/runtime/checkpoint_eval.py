"""Standalone deterministic checkpoint validation eval."""

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass, replace
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np

from jaxtitan.config import load_config, load_resolved_config
from jaxtitan.data import dataset_manifest_sha256
from jaxtitan.errors import ContractError
from jaxtitan.mesh import build_mesh_context, build_sharding_plan, place_replicated
from jaxtitan.models import build_model
from jaxtitan.optim import build_optimizer
from jaxtitan.runtime.checkpoint_index import CheckpointRecord, load_checkpoint_index
from jaxtitan.runtime.resume import validate_resume_compat, validate_resume_metadata
from jaxtitan.runtime.training import (
    _build_validation_eval_data,
    _validation_eval_row,
    _validation_eval_spec,
    _with_runtime_schedule_steps,
)
from jaxtitan.services import LocalOrbaxCheckpointService
from jaxtitan.specs.run import RunSpec
from jaxtitan.steps import initialize_train_state, make_eval_step


def evaluate_checkpoint(run_dir: str | Path, selector: str) -> dict[str, Any]:
    """Restore a retained checkpoint and run deterministic validation eval."""

    run_dir = Path(run_dir)
    spec = _load_run_spec(run_dir)
    runtime_spec = _with_runtime_schedule_steps(spec)
    eval_spec = _validation_eval_spec(runtime_spec)
    if eval_spec is None:
        raise ContractError("checkpoint eval requires [[evals]] name = 'validation'")
    index = load_checkpoint_index(run_dir)
    record = _select_checkpoint(index.records, selector, run_dir)

    service = LocalOrbaxCheckpointService(run_dir)
    try:
        metadata = service.restore_metadata(record.step)
        validate_resume_metadata(metadata, runtime_spec)
        context = build_mesh_context(runtime_spec.mesh)
        sharding = build_sharding_plan(context)
        model = build_model(runtime_spec.model, seed=runtime_spec.seed)
        optimizer = build_optimizer(runtime_spec.optimizer, model.state, model.metadata)
        model_state = place_replicated(model.state, sharding)
        template_train_state = initialize_train_state(model_state, optimizer.transform, seed=runtime_spec.seed)
        eval_step = make_eval_step(model.graph)
        restored = service.restore(record.step, template_train_state)
        validate_resume_compat(restored, runtime_spec)
    finally:
        service.close()

    checkpoint = _require_mapping(restored.metadata.get("checkpoint"), "checkpoint")
    train_row = {
        "step": _required_int(checkpoint, "step", "checkpoint"),
        "tokens_seen": _required_int(checkpoint, "tokens_seen", "checkpoint"),
    }
    data = _build_validation_eval_data(runtime_spec)
    row = _validation_eval_row(eval_spec, data, eval_step, sharding, restored.train_state, train_row)
    manifest_path = runtime_spec.data.validation_manifest or runtime_spec.data.train_manifest
    payload = {
        "schema_version": 1,
        "status": "completed",
        "created_at": _utc_now(),
        "run_id": runtime_spec.run_id,
        "checkpoint": {
            "step": restored.step,
            "path": record.checkpoint_path.as_posix(),
            "tokens_seen": train_row["tokens_seen"],
            "runtime_fingerprint": restored.metadata["runtime_fingerprint"],
        },
        "eval": row,
        "data": {
            "manifest_path": manifest_path.as_posix(),
            "manifest_sha256": dataset_manifest_sha256(manifest_path),
        },
    }
    _write_eval_artifact(run_dir, restored.step, payload)
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


def _load_run_spec(run_dir: Path) -> RunSpec:
    resolved_path = run_dir / "config" / "resolved.json"
    source_path = run_dir / "config" / "source.toml"
    if resolved_path.is_file():
        spec = load_resolved_config(resolved_path)
    elif source_path.is_file():
        spec = load_config(source_path)
    else:
        raise ContractError(f"missing run config artifact under {run_dir / 'config'}")
    if spec.run_id != run_dir.name:
        raise ContractError(f"run config id {spec.run_id!r} does not match run directory {run_dir.name!r}")
    return replace(spec, output_dir=run_dir.parent)


def _select_checkpoint(records: tuple[CheckpointRecord, ...], selector: str, run_dir: Path) -> CheckpointRecord:
    retained = [record for record in records if record.retained and (run_dir / record.checkpoint_path).is_dir()]
    if selector == "latest":
        if not retained:
            raise ContractError("checkpoint index has no retained latest checkpoint")
        return max(retained, key=lambda record: record.step)
    if selector == "best":
        candidates = [record for record in retained if record.eval_loss is not None]
        if not candidates:
            raise ContractError("checkpoint index has no retained best validation checkpoint")
        return min(candidates, key=lambda record: (record.eval_loss, record.step))
    try:
        step = int(selector)
    except ValueError as exc:
        raise ContractError(f"checkpoint selector must be 'best', 'latest', or a step, got {selector!r}") from exc
    for record in retained:
        if record.step == step:
            return record
    raise ContractError(f"checkpoint step {step} is not retained in checkpoint index")


def _write_eval_artifact(run_dir: Path, step: int, payload: Mapping[str, Any]) -> None:
    path = run_dir / "evals" / "checkpoints" / f"{step:06d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(payload) + "\n")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be a JSON object")
    return value


def _required_int(raw: Mapping[str, Any], key: str, name: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int):
        raise ContractError(f"{name}.{key} must be an integer")
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
