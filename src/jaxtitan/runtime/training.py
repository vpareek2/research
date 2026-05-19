"""Minimal host-side training loop."""

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import math
from pathlib import Path
import time
from typing import Any

import jax
import numpy as np

from jaxtitan.config import load_config
from jaxtitan.data import PreparedDataService
from jaxtitan.errors import ContractError
from jaxtitan.mesh import build_mesh_context, build_sharding_plan, place_batch, place_replicated
from jaxtitan.models import build_model
from jaxtitan.optim import build_optimizer
from jaxtitan.runtime.resume import checkpoint_metadata, validate_resume_compat, validate_resume_metadata
from jaxtitan.services import LocalArtifactWriter, LocalOrbaxCheckpointService, initialize_run
from jaxtitan.specs.run import RunSpec
from jaxtitan.state import DatasetState, HostState, TrainState
from jaxtitan.steps import initialize_train_state, make_train_step


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Final result of a local training run."""

    run_id: str
    run_dir: Path
    status: str
    steps: int
    tokens_seen: int
    target_tokens: int
    final_loss: float


def run_training(config_path: str | Path, *, resume: bool = False) -> RunSummary:
    """Run the minimal local Jaxtitan training path for one TOML config."""

    spec = load_config(config_path)
    if resume:
        if not spec.dirs.run_dir.exists():
            raise ContractError(f"cannot resume missing run directory: {spec.dirs.run_dir}")
        writer = LocalArtifactWriter(spec.dirs.run_dir)
    else:
        manifest = initialize_run(config_path)
        writer = LocalArtifactWriter(manifest.run_dir)
    try:
        writer.append_event({**_event("training_started", spec), "resume": resume})
        summary = _run_training_initialized(spec, writer, resume=resume)
        writer.append_event(
            {
                **_event("training_completed", spec),
                "steps": summary.steps,
                "tokens_seen": summary.tokens_seen,
                "target_tokens": summary.target_tokens,
                "final_loss": summary.final_loss,
            }
        )
        writer.write_summary(asdict(summary))
        return summary
    except Exception as exc:
        writer.append_event(
            {
                **_event("training_failed", spec),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        raise


def _run_training_initialized(spec: RunSpec, writer: LocalArtifactWriter, *, resume: bool) -> RunSummary:
    runtime_spec = _with_runtime_schedule_steps(spec)
    data = PreparedDataService.from_manifest(
        runtime_spec.data.train_manifest,
        tokenizer_id=runtime_spec.data.tokenizer_id,
        split="train",
        seq_len=runtime_spec.training.seq_len,
        batch_size=runtime_spec.training.global_batch_size,
    )
    dataset_state = data.initial_state()
    host_state = HostState(
        dataset=dataset_state,
        last_checkpoint_step=0,
        wallclock_start_ns=time.monotonic_ns(),
        run_id=runtime_spec.run_id,
    )
    context = build_mesh_context(runtime_spec.mesh)
    sharding = build_sharding_plan(context)
    model = build_model(runtime_spec.model, seed=runtime_spec.seed)
    optimizer = build_optimizer(runtime_spec.optimizer, model.state, model.metadata)
    model_state = place_replicated(model.state, sharding)
    train_state = initialize_train_state(model_state, optimizer.transform, seed=runtime_spec.seed)
    train_step = make_train_step(model.graph, optimizer)
    checkpoint_service = LocalOrbaxCheckpointService(runtime_spec.dirs.run_dir)

    try:
        if resume:
            try:
                validate_resume_metadata(checkpoint_service.restore_latest_metadata(), runtime_spec)
                restored = checkpoint_service.restore_latest(train_state)
                validate_resume_compat(restored, runtime_spec)
            except Exception as exc:
                writer.append_event(
                    {
                        **_event("checkpoint_restore_failed", runtime_spec),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                raise
            train_state = restored.train_state
            dataset_state = restored.dataset_state
            host_state = restored.host_state
            writer.append_event(
                {
                    **_event("training_resumed", runtime_spec),
                    "checkpoint_step": restored.step,
                    "checkpoint_path": restored.path,
                    "compat_checked": True,
                    "runtime_fingerprint": restored.metadata["runtime_fingerprint"],
                    "step": _scalar_int(train_state.step),
                    "tokens_seen": _scalar_int(train_state.tokens_seen),
                    "dataset_token_offset": dataset_state.token_offset,
                }
            )

        last_row: dict[str, Any] | None = None
        last_logged_step = -1
        while _scalar_int(train_state.tokens_seen) < runtime_spec.training.target_tokens:
            try:
                batch, dataset_state, provenance = data.next_batch(dataset_state, repeat=False)
            except StopIteration as exc:
                raise ContractError(
                    "prepared train split ended before training.target_tokens was reached; "
                    f"tokens_seen={_scalar_int(train_state.tokens_seen)} target_tokens={runtime_spec.training.target_tokens}"
                ) from exc
            placed_batch = place_batch(batch, sharding)
            train_state, metrics = train_step(train_state, placed_batch)
            row = _metrics_row(train_state, metrics, provenance)
            last_row = row
            host_state = replace(host_state, dataset=dataset_state)
            if _should_log(row["step"], runtime_spec.training.log_every_steps, runtime_spec.training.target_tokens, row["tokens_seen"]):
                writer.append_train_metrics(row)
                last_logged_step = row["step"]
            if row["step"] % runtime_spec.training.checkpoint_every_steps == 0:
                host_state = _save_checkpoint(
                    checkpoint_service,
                    writer,
                    runtime_spec,
                    train_state,
                    dataset_state,
                    host_state,
                    row,
                    reason="interval",
                )

        if last_row is None:
            raise ContractError("training.target_tokens was already satisfied before the first train step")
        if last_logged_step != last_row["step"]:
            writer.append_train_metrics(last_row)
        if checkpoint_service.latest_step() != last_row["step"]:
            _save_checkpoint(
                checkpoint_service,
                writer,
                runtime_spec,
                train_state,
                dataset_state,
                host_state,
                last_row,
                reason="final",
            )

        summary = RunSummary(
            run_id=runtime_spec.run_id,
            run_dir=runtime_spec.dirs.run_dir,
            status="completed",
            steps=last_row["step"],
            tokens_seen=last_row["tokens_seen"],
            target_tokens=runtime_spec.training.target_tokens,
            final_loss=last_row["loss"],
        )
        return summary
    finally:
        checkpoint_service.close()


def _with_runtime_schedule_steps(spec: RunSpec) -> RunSpec:
    schedule = spec.optimizer.schedule
    if schedule.name in {"cosine", "wsd"} and schedule.total_steps is None:
        batch_tokens = spec.training.global_batch_size * spec.training.seq_len
        total_steps = math.ceil(spec.training.target_tokens / batch_tokens)
        schedule = replace(schedule, total_steps=total_steps)
        optimizer = replace(spec.optimizer, schedule=schedule)
        return replace(spec, optimizer=optimizer)
    return spec


def _metrics_row(train_state: TrainState, metrics: Any, provenance: Any) -> dict[str, Any]:
    token_count = _scalar_int(metrics.token_count)
    loss_sum = _scalar_float(metrics.loss_sum)
    return {
        "schema_version": 1,
        "step": _scalar_int(train_state.step),
        "tokens_seen": _scalar_int(train_state.tokens_seen),
        "loss_sum": loss_sum,
        "token_count": token_count,
        "loss": loss_sum / token_count,
        "lr": _scalar_float(metrics.lr),
        "grad_norm": _optional_scalar_float(metrics.grad_norm),
        "param_norm": _optional_scalar_float(metrics.param_norm),
        "update_norm": _optional_scalar_float(metrics.update_norm),
        "epoch": provenance.epoch,
        "token_start": provenance.token_start,
        "token_end": provenance.token_end,
        "examples": provenance.examples,
        "target_tokens": provenance.target_tokens,
    }


def _save_checkpoint(
    checkpoint_service: LocalOrbaxCheckpointService,
    writer: LocalArtifactWriter,
    spec: RunSpec,
    train_state: TrainState,
    dataset_state: DatasetState,
    host_state: HostState,
    row: dict[str, Any],
    *,
    reason: str,
) -> HostState:
    step = row["step"]
    next_host_state = HostState(
        dataset=dataset_state,
        last_checkpoint_step=step,
        wallclock_start_ns=host_state.wallclock_start_ns,
        run_id=host_state.run_id,
    )
    checkpoint_service.save(
        step,
        train_state,
        dataset_state,
        next_host_state,
        checkpoint_metadata(spec, row, reason=reason),
    )
    writer.append_event(
        {
            **_event("checkpoint_saved", spec),
            "step": step,
            "tokens_seen": row["tokens_seen"],
            "checkpoint_path": checkpoint_service.latest_path(),
            "reason": reason,
        }
    )
    return next_host_state


def _should_log(step: int, log_every_steps: int, target_tokens: int, tokens_seen: int) -> bool:
    return step % log_every_steps == 0 or tokens_seen >= target_tokens


def _event(event_type: str, spec: RunSpec) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "type": event_type,
        "run_id": spec.run_id,
        "created_at": _utc_now(),
    }


def _optional_scalar_float(value: Any) -> float | None:
    return None if value is None else _scalar_float(value)


def _scalar_float(value: Any) -> float:
    return float(np.asarray(jax.device_get(value)).item())


def _scalar_int(value: Any) -> int:
    return int(np.asarray(jax.device_get(value)).item())


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
