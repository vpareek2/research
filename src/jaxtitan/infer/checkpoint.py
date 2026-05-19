"""Checkpoint restore into inference-only state."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from jaxtitan.config import load_config, load_resolved_config
from jaxtitan.errors import ContractError
from jaxtitan.infer.core import InferenceMetadata, InferenceState, inference_from_train_state
from jaxtitan.mesh import ShardingPlan, build_mesh_context, build_sharding_plan, place_replicated
from jaxtitan.models import build_model
from jaxtitan.optim import build_optimizer
from jaxtitan.runtime.checkpoint_index import CheckpointRecord, load_checkpoint_index
from jaxtitan.runtime.resume import validate_resume_compat, validate_resume_metadata
from jaxtitan.runtime.training import _with_runtime_schedule_steps
from jaxtitan.services import LocalOrbaxCheckpointService
from jaxtitan.specs.run import RunSpec
from jaxtitan.steps import initialize_train_state


@dataclass(frozen=True, slots=True)
class InferenceRestore:
    """Restored inference state plus static runtime context."""

    graph: Any
    state: InferenceState
    metadata: InferenceMetadata
    run_spec: RunSpec
    sharding: ShardingPlan


def restore_inference_checkpoint(run_dir: str | Path, checkpoint: str) -> InferenceRestore:
    """Restore a retained checkpoint as inference-only state."""

    run_dir = Path(run_dir)
    runtime_spec = _with_runtime_schedule_steps(_load_run_spec(run_dir))
    index = load_checkpoint_index(run_dir)
    record = _select_checkpoint(index.records, checkpoint, run_dir)

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
        restored = service.restore(record.step, template_train_state)
        validate_resume_compat(restored, runtime_spec)
    finally:
        service.close()

    checkpoint_metadata = _require_mapping(restored.metadata.get("checkpoint"), "checkpoint")
    inference_state = inference_from_train_state(restored.train_state)
    inference_metadata = InferenceMetadata(
        run_id=runtime_spec.run_id,
        checkpoint_step=restored.step,
        checkpoint_path=record.checkpoint_path,
        tokens_seen=_required_int(checkpoint_metadata, "tokens_seen", "checkpoint"),
        model_spec=runtime_spec.model,
        mesh_spec=runtime_spec.mesh,
        runtime_fingerprint=_required_str(restored.metadata, "runtime_fingerprint", "metadata"),
    )
    return InferenceRestore(
        graph=model.graph,
        state=inference_state,
        metadata=inference_metadata,
        run_spec=runtime_spec,
        sharding=sharding,
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


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be a JSON object")
    return value


def _required_int(raw: dict[str, Any], key: str, name: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int):
        raise ContractError(f"{name}.{key} must be an integer")
    return value


def _required_str(raw: dict[str, Any], key: str, name: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name}.{key} must be a non-empty string")
    return value
