"""Minimal host-side training loop."""

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
import math
from pathlib import Path
import time
from typing import Any

import jax
import numpy as np

from jaxtitan.batch import Batch
from jaxtitan.config import load_config
from jaxtitan.data import (
    DATA_PIPELINE_BACKEND,
    HF_STREAMING_BACKEND,
    BatchProvenance,
    DataPipelineState,
    PreparedTokenGrainPipeline,
    TrainingDataPipeline,
    build_hf_streaming_pipeline,
    build_prepared_token_pipeline,
)
from jaxtitan.errors import ContractError
from jaxtitan.kernels import require_kernel_plan_supported
from jaxtitan.mesh import (
    build_mesh_context,
    build_sharding_plan,
    place_accumulated_batch,
    place_batch,
    place_model_state,
    place_optimizer_init_state,
    require_single_process_runtime,
    validate_runtime_mesh_spec,
)
from jaxtitan.models import build_model
from jaxtitan.optim import build_optimizer
from jaxtitan.runtime.checkpoint_index import CheckpointIndex, load_checkpoint_index, record_checkpoint
from jaxtitan.runtime.diagnostics import (
    ARTIFACT_WRITER,
    METRICS_SCOPE,
    PhaseTimer,
    build_runtime_diagnostics,
    enrich_eval_metrics,
    enrich_train_metrics,
    sample_device_telemetry,
    sync_and_time,
    training_diagnostics_summary,
    runtime_execution_mode,
)
from jaxtitan.runtime.profiling import ProfilingManager
from jaxtitan.runtime.resume import checkpoint_metadata, validate_resume_compat, validate_resume_metadata
from jaxtitan.services import (
    ArtifactWriter,
    LocalArtifactWriter,
    LocalOrbaxCheckpointService,
    build_artifact_writer,
    initialize_run,
)
from jaxtitan.specs.eval import EvalSpec
from jaxtitan.specs.run import RunSpec
from jaxtitan.state import HostState, TrainState
from jaxtitan.steps import initialize_train_state, make_eval_step, make_train_step

TrainingProgress = Callable[[str, dict[str, Any]], None]


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
    final_eval_loss: float | None = None
    final_eval_token_count: int | None = None
    final_eval_num_batches: int | None = None
    latest_checkpoint_path: str | None = None
    best_eval_step: int | None = None
    best_eval_loss: float | None = None
    best_checkpoint_path: str | None = None
    total_wall_sec: float | None = None
    avg_train_tokens_per_sec: float | None = None
    final_train_tokens_per_sec: float | None = None
    steady_train_tokens_per_sec: float | None = None
    avg_mfu: float | None = None
    final_mfu: float | None = None
    device_kind: str | None = None
    device_count: int | None = None
    runtime_diagnostics_path: str | None = None
    execution_mode: str | None = None
    metrics_scope: str | None = None
    artifact_writer: str | None = None
    model_remat: str | None = None
    data_axis_size: int | None = None
    global_batch_size: int | None = None
    per_device_batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    effective_global_batch_size: int | None = None
    micro_tokens_per_step: int | None = None
    effective_tokens_per_step: int | None = None
    selected_device_count: int | None = None
    global_device_count: int | None = None
    process_count: int | None = None
    process_index: int | None = None
    single_process: bool | None = None
    data_pipeline_backend: str | None = None
    data_pipeline_version: str | None = None
    data_pipeline_order: str | None = None
    data_pipeline_shuffle_seed: int | None = None
    data_pipeline_worker_count: int | None = None
    data_pipeline_worker_buffer_size: int | None = None
    data_pipeline_prefetch: bool | None = None
    data_pipeline_state_schema_version: int | None = None
    data_document_aware: bool | None = None
    data_document_count: int | None = None
    data_document_offsets_path: str | None = None
    data_document_offsets_sha256: str | None = None
    data_document_buffer_size: int | None = None
    data_document_refill_size: int | None = None
    final_batch_het: float | None = None
    avg_batch_het: float | None = None
    moe_balance_policy: str | None = None
    z_loss_weight: float | None = None
    final_moe_router_layers: list[dict[str, Any]] | None = None
    final_router_max_vio: float | None = None
    final_router_mean_load_cv: float | None = None
    final_router_dead_experts_count: float | None = None
    final_router_mean_importance_cv: float | None = None
    final_router_mean_importance_entropy: float | None = None
    final_optimizer_groups: list[dict[str, Any]] | None = None
    final_optimizer_grad_norm_max_group: str | None = None
    final_optimizer_update_norm_max_group: str | None = None
    final_optimizer_update_param_ratio_max: float | None = None
    final_optimizer_update_param_ratio_mean: float | None = None
    final_optimizer_groups_with_zero_grad: int | None = None
    final_optimizer_groups_with_zero_update: int | None = None
    final_optimizer_route_backend_counts: dict[str, int] | None = None


@dataclass(frozen=True, slots=True)
class EvalRunResult:
    """Host-normalized result for one validation eval pass."""

    row: dict[str, Any]


def run_training(
    config_path: str | Path,
    *,
    resume: bool = False,
    progress: TrainingProgress | None = None,
) -> RunSummary:
    """Run the minimal local Jaxtitan training path for one TOML config."""

    spec = load_config(config_path)
    if progress is not None:
        progress("start", {"spec": spec, "resume": resume, "config_path": Path(config_path)})
    require_single_process_runtime()
    validate_runtime_mesh_spec(spec.mesh)
    if resume:
        if not spec.dirs.run_dir.exists():
            raise ContractError(f"cannot resume missing run directory: {spec.dirs.run_dir}")
        local_writer = LocalArtifactWriter(spec.dirs.run_dir)
        manifest = None
    else:
        manifest = initialize_run(config_path)
        local_writer = LocalArtifactWriter(manifest.run_dir)
    writer = build_artifact_writer(spec=spec, local=local_writer, manifest=manifest, resume=resume)
    try:
        writer.append_event(
            {
                **_event("training_started", spec),
                "resume": resume,
                "execution_mode": runtime_execution_mode(spec),
                "parallelism_mode": spec.parallelism.mode,
                "metrics_scope": METRICS_SCOPE,
                "artifact_writer": ARTIFACT_WRITER,
                "model_remat": spec.model.remat,
                "gradient_accumulation_steps": spec.training.gradient_accumulation_steps,
                "data_pipeline_backend": _data_pipeline_backend_name(spec),
                "data_pipeline_order": spec.data.order,
                "data_pipeline_shuffle_seed": spec.data.shuffle_seed,
                "data_pipeline_worker_count": spec.data.worker_count,
                "data_pipeline_worker_buffer_size": spec.data.worker_buffer_size,
                "data_pipeline_prefetch": spec.data.prefetch,
                "data_document_buffer_size": spec.data.document_buffer_size,
                "data_document_refill_size": spec.data.document_refill_size,
                "moe_balance_policy": _moe_balance_name(spec),
                "z_loss_weight": spec.training.loss.z_loss_weight,
            }
        )
        summary = _run_training_initialized(spec, writer, resume=resume, progress=progress)
        writer.append_event(
            {
                **_event("training_completed", spec),
                "steps": summary.steps,
                "tokens_seen": summary.tokens_seen,
                "target_tokens": summary.target_tokens,
                "final_loss": summary.final_loss,
                "total_wall_sec": summary.total_wall_sec,
                "final_train_tokens_per_sec": summary.final_train_tokens_per_sec,
                "avg_train_tokens_per_sec": summary.avg_train_tokens_per_sec,
                "steady_train_tokens_per_sec": summary.steady_train_tokens_per_sec,
                "final_mfu": summary.final_mfu,
                "avg_mfu": summary.avg_mfu,
                "final_batch_het": summary.final_batch_het,
                "avg_batch_het": summary.avg_batch_het,
                "execution_mode": summary.execution_mode,
                "metrics_scope": summary.metrics_scope,
                "artifact_writer": summary.artifact_writer,
                "model_remat": summary.model_remat,
                "data_axis_size": summary.data_axis_size,
                "gradient_accumulation_steps": summary.gradient_accumulation_steps,
                "effective_global_batch_size": summary.effective_global_batch_size,
                "effective_tokens_per_step": summary.effective_tokens_per_step,
                "per_device_batch_size": summary.per_device_batch_size,
                "selected_device_count": summary.selected_device_count,
                "data_pipeline_backend": summary.data_pipeline_backend,
                "data_pipeline_order": summary.data_pipeline_order,
                "data_pipeline_shuffle_seed": summary.data_pipeline_shuffle_seed,
                "data_pipeline_worker_count": summary.data_pipeline_worker_count,
                "data_pipeline_worker_buffer_size": summary.data_pipeline_worker_buffer_size,
                "data_pipeline_prefetch": summary.data_pipeline_prefetch,
                "data_document_buffer_size": summary.data_document_buffer_size,
                "data_document_refill_size": summary.data_document_refill_size,
                "moe_balance_policy": summary.moe_balance_policy,
                "z_loss_weight": summary.z_loss_weight,
                "final_moe_router_layers": summary.final_moe_router_layers,
                "final_router_max_vio": summary.final_router_max_vio,
                "final_router_mean_load_cv": summary.final_router_mean_load_cv,
                "final_router_dead_experts_count": summary.final_router_dead_experts_count,
                "final_router_mean_importance_cv": summary.final_router_mean_importance_cv,
                "final_router_mean_importance_entropy": summary.final_router_mean_importance_entropy,
                "final_optimizer_groups": summary.final_optimizer_groups,
                "final_optimizer_grad_norm_max_group": summary.final_optimizer_grad_norm_max_group,
                "final_optimizer_update_norm_max_group": summary.final_optimizer_update_norm_max_group,
                "final_optimizer_update_param_ratio_max": summary.final_optimizer_update_param_ratio_max,
                "final_optimizer_update_param_ratio_mean": summary.final_optimizer_update_param_ratio_mean,
                "final_optimizer_groups_with_zero_grad": summary.final_optimizer_groups_with_zero_grad,
                "final_optimizer_groups_with_zero_update": summary.final_optimizer_groups_with_zero_update,
                "final_optimizer_route_backend_counts": summary.final_optimizer_route_backend_counts,
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
    finally:
        writer.close()


def _run_training_initialized(
    spec: RunSpec,
    writer: ArtifactWriter,
    *,
    resume: bool,
    progress: TrainingProgress | None,
) -> RunSummary:
    run_started_at = time.perf_counter()
    runtime_spec = _with_runtime_schedule_steps(spec)
    data = _build_train_data_pipeline(runtime_spec)
    dataset_state = data.initial_state()
    host_state = HostState(
        dataset=dataset_state,
        last_checkpoint_step=0,
        wallclock_start_ns=time.monotonic_ns(),
        run_id=runtime_spec.run_id,
    )
    context = build_mesh_context(runtime_spec.mesh)
    model = build_model(runtime_spec.model, seed=runtime_spec.seed)
    sharding = build_sharding_plan(
        context,
        parallelism=runtime_spec.parallelism,
        param_layouts=model.param_layouts,
        expert_layouts=model.expert_layouts,
    )
    model_state = place_model_state(model.state, sharding)
    optimizer_init_state = place_optimizer_init_state(model.state, sharding)
    optimizer = build_optimizer(runtime_spec.optimizer, optimizer_init_state, model.metadata)
    runtime_diagnostics = build_runtime_diagnostics(
        runtime_spec,
        context,
        model.metadata,
        optimizer=optimizer,
        sharding=sharding,
        data_pipeline=data.describe(),
        wandb=getattr(writer, "wandb_metadata", None),
    )
    writer.write_runtime_diagnostics(runtime_diagnostics.payload)
    require_kernel_plan_supported(runtime_diagnostics.payload["kernels"])
    profiler = ProfilingManager(spec=runtime_spec, writer=writer)
    profiler.write_initial_diagnostics()
    train_state = initialize_train_state(
        model_state,
        optimizer.transform,
        seed=runtime_spec.seed,
        optimizer_init_model_state=optimizer_init_state,
        moe_balance_spec=_moe_balance_spec(runtime_spec),
    )
    expected_train_shape = (
        runtime_spec.training.gradient_accumulation_steps,
        runtime_spec.training.global_batch_size,
        runtime_spec.training.seq_len,
    )
    expected_eval_shape = (runtime_spec.training.global_batch_size, runtime_spec.training.seq_len)
    train_step = make_train_step(
        model.graph,
        optimizer,
        sharding=sharding,
        state_template=train_state,
        donate_state=True,
        expected_batch_shape=expected_train_shape,
        loss=runtime_spec.training.loss,
    )
    eval_spec = _validation_eval_spec(runtime_spec)
    eval_step = None if eval_spec is None else make_eval_step(
        model.graph,
        sharding=sharding,
        state_template=train_state.model,
        expected_batch_shape=expected_eval_shape,
    )
    checkpoint_service = LocalOrbaxCheckpointService(runtime_spec.dirs.run_dir)
    checkpoint_index = load_checkpoint_index(runtime_spec.dirs.run_dir)
    checkpoint_service.set_protected_steps(checkpoint_index.protected_steps())
    if progress is not None:
        progress(
            "initialized",
            {
                "spec": runtime_spec,
                "diagnostics": runtime_diagnostics.payload,
                "optimizer": optimizer.description,
                "run_dir": runtime_spec.dirs.run_dir,
                "resume": resume,
            },
        )

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
            profiler.validate_resume_step(_scalar_int(train_state.step))
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
                    "data_pipeline_backend": dataset_state.backend,
                    "data_pipeline_order": dataset_state.order,
                    "data_pipeline_shuffle_seed": dataset_state.shuffle_seed,
                }
            )
            if progress is not None:
                progress(
                    "resumed",
                    {
                        "step": _scalar_int(train_state.step),
                        "tokens_seen": _scalar_int(train_state.tokens_seen),
                        "checkpoint_step": restored.step,
                        "checkpoint_path": restored.path,
                    },
                )

        last_row: dict[str, Any] | None = None
        last_eval: EvalRunResult | None = None
        last_eval_step = -1
        last_logged_step = -1
        logged_rows: list[dict[str, Any]] = []
        train_compiled = False
        eval_compiled = False
        while _scalar_int(train_state.tokens_seen) < runtime_spec.training.target_tokens:
            next_step = _scalar_int(train_state.step) + 1
            with profiler.step(next_step):
                timer = PhaseTimer()
                with timer.phase("data"), profiler.annotation("data"):
                    try:
                        batch, dataset_state, provenance = _next_accumulated_train_batch(
                            data,
                            dataset_state,
                            runtime_spec.training.gradient_accumulation_steps,
                        )
                    except StopIteration as exc:
                        raise ContractError(_data_exhausted_message(runtime_spec, data, train_state)) from exc
                with timer.phase("placement"), profiler.annotation("placement"):
                    placed_batch = place_accumulated_batch(batch, sharding)
                if not train_compiled and progress is not None:
                    progress("compile_start", {"phase": "train"})
                with timer.phase("train_dispatch"), profiler.annotation("train_step"):
                    train_state, metrics = train_step(train_state, placed_batch)
                train_compiled = True
                with profiler.annotation("metrics_sync"):
                    metrics_sync_sec = sync_and_time(_train_sync_target(train_state, metrics))
                timer.add("metrics_sync", metrics_sync_sec)
                base_row = _metrics_row(train_state, metrics, provenance, runtime_spec=runtime_spec, context=context)
                row = enrich_train_metrics(
                    base_row,
                    timings={
                        "data_sec": timer.seconds("data"),
                        "placement_sec": timer.seconds("placement"),
                        "train_dispatch_sec": timer.seconds("train_dispatch"),
                        "metrics_sync_sec": timer.seconds("metrics_sync"),
                        "train_step_sec": timer.seconds("train_dispatch") + timer.seconds("metrics_sync"),
                        "step_sec": timer.total_sec(),
                    },
                    runtime=runtime_diagnostics,
                    telemetry=sample_device_telemetry(context.devices),
                )
                last_row = row
                host_state = replace(host_state, dataset=dataset_state)
                if _should_log(
                    row["step"],
                    runtime_spec.training.log_every_steps,
                    runtime_spec.training.target_tokens,
                    row["tokens_seen"],
                ):
                    writer.append_train_metrics(row)
                    logged_rows.append(row)
                    last_logged_step = row["step"]
                    if progress is not None:
                        progress("train", {"row": row})
                checkpoint_due = row["step"] % runtime_spec.training.checkpoint_every_steps == 0
                eval_due = eval_spec is not None and row["step"] % eval_spec.every_steps == 0
                if eval_due or (eval_spec is not None and checkpoint_due):
                    if not eval_compiled and progress is not None:
                        progress("compile_start", {"phase": "eval"})
                    with profiler.annotation("eval"):
                        last_eval = run_validation_eval(
                            writer,
                            runtime_spec,
                            eval_spec,
                            eval_step,
                            sharding,
                            train_state,
                            row,
                            progress=progress,
                        )
                    last_eval_step = row["step"]
                    eval_compiled = True
                if checkpoint_due:
                    with profiler.annotation("checkpoint"):
                        host_state, checkpoint_index = _save_checkpoint(
                            checkpoint_service,
                            writer,
                            runtime_spec,
                            train_state,
                            dataset_state,
                            host_state,
                            checkpoint_index,
                            row,
                            last_eval,
                            reason="interval",
                            progress=progress,
                        )

        profiler.close()
        if last_row is None:
            raise ContractError("training.target_tokens was already satisfied before the first train step")
        if last_logged_step != last_row["step"]:
            writer.append_train_metrics(last_row)
            logged_rows.append(last_row)
            if progress is not None:
                progress("train", {"row": last_row})
        if eval_spec is not None and last_eval_step != last_row["step"]:
            if not eval_compiled and progress is not None:
                progress("compile_start", {"phase": "eval"})
            last_eval = run_validation_eval(
                writer,
                runtime_spec,
                eval_spec,
                eval_step,
                sharding,
                train_state,
                last_row,
                progress=progress,
            )
            last_eval_step = last_row["step"]
            eval_compiled = True
        if checkpoint_service.latest_step() != last_row["step"]:
            host_state, checkpoint_index = _save_checkpoint(
                checkpoint_service,
                writer,
                runtime_spec,
                train_state,
                dataset_state,
                host_state,
                checkpoint_index,
                last_row,
                last_eval,
                reason="final",
                progress=progress,
            )
        else:
            writer.write_checkpoint_index(checkpoint_index.to_dict())

        latest = checkpoint_index.latest_record
        best = checkpoint_index.best_record
        diagnostic_summary = training_diagnostics_summary(
            logged_rows,
            total_wall_sec=time.perf_counter() - run_started_at,
            runtime=runtime_diagnostics,
        )
        runtime_summary = runtime_diagnostics.payload
        mesh_summary = runtime_summary["mesh"]
        jax_summary = runtime_summary["jax"]
        data_pipeline_summary = runtime_summary["data_pipeline"]

        summary = RunSummary(
            run_id=runtime_spec.run_id,
            run_dir=runtime_spec.dirs.run_dir,
            status="completed",
            steps=last_row["step"],
            tokens_seen=last_row["tokens_seen"],
            target_tokens=runtime_spec.training.target_tokens,
            final_loss=last_row["loss"],
            final_eval_loss=None if last_eval is None else last_eval.row["loss"],
            final_eval_token_count=None if last_eval is None else last_eval.row["token_count"],
            final_eval_num_batches=None if last_eval is None else last_eval.row["num_batches"],
            latest_checkpoint_path=None if latest is None else latest.checkpoint_path.as_posix(),
            best_eval_step=None if best is None else best.step,
            best_eval_loss=None if best is None else best.eval_loss,
            best_checkpoint_path=None if best is None else best.checkpoint_path.as_posix(),
            total_wall_sec=diagnostic_summary["total_wall_sec"],
            avg_train_tokens_per_sec=diagnostic_summary["avg_train_tokens_per_sec"],
            final_train_tokens_per_sec=diagnostic_summary["final_train_tokens_per_sec"],
            steady_train_tokens_per_sec=diagnostic_summary["steady_train_tokens_per_sec"],
            avg_mfu=diagnostic_summary["avg_mfu"],
            final_mfu=diagnostic_summary["final_mfu"],
            device_kind=diagnostic_summary["device_kind"],
            device_count=diagnostic_summary["device_count"],
            runtime_diagnostics_path=diagnostic_summary["runtime_diagnostics_path"],
            execution_mode=diagnostic_summary["execution_mode"],
            metrics_scope=diagnostic_summary["metrics_scope"],
            artifact_writer=diagnostic_summary["artifact_writer"],
            model_remat=runtime_summary["model"]["remat"],
            data_axis_size=mesh_summary["data_axis_size"],
            global_batch_size=mesh_summary["global_batch_size"],
            per_device_batch_size=mesh_summary["per_device_batch_size"],
            gradient_accumulation_steps=mesh_summary["gradient_accumulation_steps"],
            effective_global_batch_size=mesh_summary["effective_global_batch_size"],
            micro_tokens_per_step=mesh_summary["micro_tokens_per_step"],
            effective_tokens_per_step=mesh_summary["effective_tokens_per_step"],
            selected_device_count=mesh_summary["selected_device_count"],
            global_device_count=jax_summary["global_device_count"],
            process_count=jax_summary["process_count"],
            process_index=jax_summary["process_index"],
            single_process=jax_summary["process_count"] == 1,
            data_pipeline_backend=data_pipeline_summary["backend"],
            data_pipeline_version=data_pipeline_summary["backend_version"],
            data_pipeline_order=data_pipeline_summary["order"],
            data_pipeline_shuffle_seed=data_pipeline_summary["shuffle_seed"],
            data_pipeline_worker_count=data_pipeline_summary["worker_count"],
            data_pipeline_worker_buffer_size=data_pipeline_summary["worker_buffer_size"],
            data_pipeline_prefetch=data_pipeline_summary["prefetch"],
            data_pipeline_state_schema_version=data_pipeline_summary["state_schema_version"],
            data_document_aware=data_pipeline_summary["document_aware"],
            data_document_count=data_pipeline_summary["document_count"],
            data_document_offsets_path=data_pipeline_summary["document_offsets_path"],
            data_document_offsets_sha256=data_pipeline_summary["document_offsets_sha256"],
            data_document_buffer_size=data_pipeline_summary.get("document_buffer_size"),
            data_document_refill_size=data_pipeline_summary.get("document_refill_size"),
            final_batch_het=diagnostic_summary["final_batch_het"],
            avg_batch_het=diagnostic_summary["avg_batch_het"],
            moe_balance_policy=runtime_summary["model"]["moe_balance"]["name"],
            z_loss_weight=runtime_summary["training"]["loss"]["z_loss_weight"],
            final_moe_router_layers=last_row.get("moe_router_layers"),
            final_router_max_vio=last_row["router_max_vio"],
            final_router_mean_load_cv=last_row["router_mean_load_cv"],
            final_router_dead_experts_count=last_row["router_dead_experts_count"],
            final_router_mean_importance_cv=last_row["router_mean_importance_cv"],
            final_router_mean_importance_entropy=last_row["router_mean_importance_entropy"],
            final_optimizer_groups=last_row.get("optimizer_groups"),
            final_optimizer_grad_norm_max_group=last_row.get("optimizer_grad_norm_max_group"),
            final_optimizer_update_norm_max_group=last_row.get("optimizer_update_norm_max_group"),
            final_optimizer_update_param_ratio_max=last_row.get("optimizer_update_param_ratio_max"),
            final_optimizer_update_param_ratio_mean=last_row.get("optimizer_update_param_ratio_mean"),
            final_optimizer_groups_with_zero_grad=last_row.get("optimizer_groups_with_zero_grad"),
            final_optimizer_groups_with_zero_update=last_row.get("optimizer_groups_with_zero_update"),
            final_optimizer_route_backend_counts=last_row.get("optimizer_route_backend_counts"),
        )
        if progress is not None:
            progress("completed", {"summary": summary})
        return summary
    finally:
        profiler.close()
        data.close()
        checkpoint_service.close()


def _with_runtime_schedule_steps(spec: RunSpec) -> RunSpec:
    schedule = spec.optimizer.schedule
    if schedule.name in {"cosine", "wsd"} and schedule.total_steps is None:
        batch_tokens = (
            spec.training.global_batch_size
            * spec.training.seq_len
            * spec.training.gradient_accumulation_steps
        )
        total_steps = math.ceil(spec.training.target_tokens / batch_tokens)
        schedule = replace(schedule, total_steps=total_steps)
        optimizer = replace(spec.optimizer, schedule=schedule)
        return replace(spec, optimizer=optimizer)
    return spec


def _validation_eval_spec(spec: RunSpec) -> EvalSpec | None:
    if not spec.evals:
        return None
    if len(spec.evals) > 1:
        raise ContractError("runtime supports exactly one eval entry for now")
    eval_spec = spec.evals[0]
    if eval_spec.name != "validation":
        raise ContractError(f"runtime supports only eval name 'validation', got {eval_spec.name!r}")
    return eval_spec


def _moe_balance_spec(spec: RunSpec) -> Any | None:
    trinity = spec.model.trinity
    if trinity is None or trinity.moe is None:
        return None
    return trinity.moe.balance


def _moe_balance_name(spec: RunSpec) -> str:
    balance = _moe_balance_spec(spec)
    return "none" if balance is None else balance.name


def _build_train_data_pipeline(spec: RunSpec) -> TrainingDataPipeline:
    if spec.data.mode == "hf_streaming":
        if spec.data.hf_streaming is None:
            raise ContractError("data.hf_streaming is required when data.mode='hf_streaming'")
        return build_hf_streaming_pipeline(
            spec.data.hf_streaming,
            tokenizer_id=spec.data.tokenizer_id,
            seq_len=spec.training.seq_len,
            batch_size=spec.training.global_batch_size,
            order=spec.data.order,
            shuffle_seed=spec.data.shuffle_seed,
            worker_count=spec.data.worker_count,
            worker_buffer_size=spec.data.worker_buffer_size,
            prefetch=spec.data.prefetch,
        )
    if spec.data.train_manifest is None:
        raise ContractError("data.train_manifest is required when data.mode='prepared'")
    return build_prepared_token_pipeline(
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
    )


def _build_validation_eval_data(spec: RunSpec) -> PreparedTokenGrainPipeline:
    if spec.data.mode == "hf_streaming" and spec.data.validation_manifest is None:
        raise ContractError("data.validation_manifest is required for eval when data.mode='hf_streaming'")
    manifest = spec.data.validation_manifest if spec.data.validation_manifest is not None else spec.data.train_manifest
    if manifest is None:
        raise ContractError("validation eval requires a prepared manifest")
    return PreparedTokenGrainPipeline.from_manifest(
        manifest,
        tokenizer_id=spec.data.tokenizer_id,
        split="val",
        seq_len=spec.training.seq_len,
        batch_size=spec.training.global_batch_size,
        order="sequential",
        shuffle_seed=None,
        worker_count=spec.data.worker_count,
        worker_buffer_size=spec.data.worker_buffer_size,
        prefetch=spec.data.prefetch,
    )


def _next_accumulated_train_batch(
    data: TrainingDataPipeline,
    dataset_state: DataPipelineState,
    accumulation_steps: int,
) -> tuple[Batch, DataPipelineState, BatchProvenance]:
    batches = []
    provenances = []
    next_state = dataset_state
    for _idx in range(accumulation_steps):
        result = data.next_batch(next_state)
        batches.append(result.batch)
        provenances.append(result.provenance)
        next_state = result.state
    return _stack_accumulated_batches(batches), next_state, _combine_provenance(provenances)


def _data_pipeline_backend_name(spec: RunSpec) -> str:
    return HF_STREAMING_BACKEND if spec.data.mode == "hf_streaming" else DATA_PIPELINE_BACKEND


def _data_exhausted_message(spec: RunSpec, data: TrainingDataPipeline, train_state: TrainState) -> str:
    prefix = "prepared train split" if spec.data.mode == "prepared" else f"{data.describe()['backend']} train data"
    return (
        f"{prefix} ended before training.target_tokens was reached; "
        f"tokens_seen={_scalar_int(train_state.tokens_seen)} target_tokens={spec.training.target_tokens}"
    )


def _stack_accumulated_batches(batches: list[Batch]) -> Batch:
    if not batches:
        raise ContractError("gradient accumulation requires at least one microbatch")
    has_doc_ids = batches[0].doc_ids is not None
    if any((batch.doc_ids is not None) != has_doc_ids for batch in batches):
        raise ContractError("gradient accumulation batches must consistently include or omit doc_ids")
    return Batch(
        input_ids=np.stack([batch.input_ids for batch in batches]),
        target_ids=np.stack([batch.target_ids for batch in batches]),
        loss_mask=np.stack([batch.loss_mask for batch in batches]),
        doc_ids=None if not has_doc_ids else np.stack([batch.doc_ids for batch in batches]),
    )


def _combine_provenance(provenances: list[BatchProvenance]) -> BatchProvenance:
    if not provenances:
        raise ContractError("gradient accumulation requires at least one microbatch provenance")
    first = provenances[0]
    has_doc_ids = first.row_doc_ids is not None
    if any((provenance.row_doc_ids is not None) != has_doc_ids for provenance in provenances):
        raise ContractError("gradient accumulation provenance must consistently include or omit document ids")
    return BatchProvenance(
        split=first.split,
        epoch=first.epoch,
        token_start=min(provenance.token_start for provenance in provenances),
        token_end=max(provenance.token_end for provenance in provenances),
        examples=sum(provenance.examples for provenance in provenances),
        target_tokens=sum(provenance.target_tokens for provenance in provenances),
        row_start_offsets=tuple(
            offset
            for provenance in provenances
            for offset in provenance.row_start_offsets
        ),
        row_doc_ids=None if not has_doc_ids else tuple(
            doc_id
            for provenance in provenances
            for doc_id in provenance.row_doc_ids
        ),
    )


def run_validation_eval(
    writer: ArtifactWriter,
    spec: RunSpec,
    eval_spec: EvalSpec,
    eval_step: Any,
    sharding: Any,
    train_state: TrainState,
    train_row: dict[str, Any],
    *,
    progress: TrainingProgress | None = None,
) -> EvalRunResult:
    """Run one deterministic validation eval pass from the validation split start."""

    eval_started_at = time.perf_counter()
    writer.append_event(
        {
            **_event("eval_started", spec),
            "step": train_row["step"],
            "tokens_seen": train_row["tokens_seen"],
            "eval_name": eval_spec.name,
            "num_batches": eval_spec.num_batches,
        }
    )
    data: PreparedTokenGrainPipeline | None = None
    try:
        data = _build_validation_eval_data(spec)
        row = _validation_eval_row(eval_spec, data, eval_step, sharding, train_state, train_row)
        row = enrich_eval_metrics(row, eval_sec=time.perf_counter() - eval_started_at)
        writer.append_eval_metrics(row)
        if progress is not None:
            progress("eval", {"row": row})
        writer.append_event(
            {
                **_event("eval_completed", spec),
                "step": row["step"],
                "tokens_seen": row["tokens_seen"],
                "eval_name": eval_spec.name,
                "loss": row["loss"],
                "token_count": row["token_count"],
                "num_batches": row["num_batches"],
                "eval_sec": row["eval_sec"],
            }
        )
        return EvalRunResult(row=row)
    except Exception as exc:
        writer.append_event(
            {
                **_event("eval_failed", spec),
                "step": train_row["step"],
                "tokens_seen": train_row["tokens_seen"],
                "eval_name": eval_spec.name,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "eval_sec": time.perf_counter() - eval_started_at,
            }
        )
        raise
    finally:
        if data is not None:
            data.close()


def _validation_eval_row(
    eval_spec: EvalSpec,
    data: TrainingDataPipeline,
    eval_step: Any,
    sharding: Any,
    train_state: TrainState,
    train_row: dict[str, Any],
) -> dict[str, Any]:
    dataset_state = data.initial_state()
    loss_sum = 0.0
    token_count = 0
    first_token_start: int | None = None
    token_end = dataset_state.token_offset
    examples = 0
    target_tokens = 0
    row_doc_ids: list[int] = []
    saw_document_ids = False
    for _idx in range(eval_spec.num_batches):
        try:
            result = data.next_batch(dataset_state)
        except StopIteration as exc:
            raise ContractError(
                f"validation split ended before eval.num_batches={eval_spec.num_batches} completed"
            ) from exc
        batch = result.batch
        dataset_state = result.state
        provenance = result.provenance
        metrics = eval_step(train_state.model, place_batch(batch, sharding))
        loss_sum += _scalar_float(metrics.loss_sum)
        token_count += _scalar_int(metrics.token_count)
        first_token_start = provenance.token_start if first_token_start is None else first_token_start
        token_end = provenance.token_end
        examples += provenance.examples
        target_tokens += provenance.target_tokens
        if provenance.row_doc_ids is not None:
            saw_document_ids = True
            row_doc_ids.extend(provenance.row_doc_ids)
    row = {
        "schema_version": 1,
        "step": train_row["step"],
        "tokens_seen": train_row["tokens_seen"],
        "eval_name": eval_spec.name,
        "loss_sum": loss_sum,
        "token_count": token_count,
        "loss": loss_sum / token_count,
        "num_batches": eval_spec.num_batches,
        "token_start": first_token_start,
        "token_end": token_end,
        "examples": examples,
        "target_tokens": target_tokens,
    }
    row.update(_document_metric_fields(tuple(row_doc_ids) if saw_document_ids else None))
    return row


def _metrics_row(
    train_state: TrainState,
    metrics: Any,
    provenance: Any,
    *,
    runtime_spec: RunSpec | None = None,
    context: Any | None = None,
) -> dict[str, Any]:
    token_count = _scalar_int(metrics.token_count)
    loss_sum = _scalar_float(metrics.loss_sum)
    row = {
        "schema_version": 1,
        "step": _scalar_int(train_state.step),
        "tokens_seen": _scalar_int(train_state.tokens_seen),
        "loss_sum": loss_sum,
        "token_count": token_count,
        "loss": loss_sum / token_count,
        "lm_loss": loss_sum / token_count,
        "lr": _scalar_float(metrics.lr),
        "grad_norm": _optional_scalar_float(metrics.grad_norm),
        "param_norm": _optional_scalar_float(metrics.param_norm),
        "update_norm": _optional_scalar_float(metrics.update_norm),
        "total_loss": _optional_scalar_float(metrics.total_loss),
        "z_loss": _optional_scalar_float(metrics.z_loss),
        "moe_aux_loss": _optional_scalar_float(metrics.moe_aux_loss),
        "aux_loss": _optional_scalar_float(metrics.aux_loss),
        "router_max_vio": _optional_scalar_float(metrics.router_max_vio),
        "router_load_min": _optional_scalar_float(metrics.router_load_min),
        "router_load_max": _optional_scalar_float(metrics.router_load_max),
        "router_load_entropy": _optional_scalar_float(metrics.router_load_entropy),
        "router_mean_load_cv": _optional_scalar_float(metrics.router_mean_load_cv),
        "router_std_load_cv": _optional_scalar_float(metrics.router_std_load_cv),
        "router_mean_load_entropy": _optional_scalar_float(metrics.router_mean_load_entropy),
        "router_min_load_entropy": _optional_scalar_float(metrics.router_min_load_entropy),
        "router_dead_experts_count": _optional_scalar_float(metrics.router_dead_experts_count),
        "router_experts_active_mean": _optional_scalar_float(metrics.router_experts_active_mean),
        "router_mean_importance_cv": _optional_scalar_float(metrics.router_mean_importance_cv),
        "router_mean_importance_entropy": _optional_scalar_float(metrics.router_mean_importance_entropy),
        "smebu_bias_norm": _optional_scalar_float(metrics.smebu_bias_norm),
        "smebu_momentum_norm": _optional_scalar_float(metrics.smebu_momentum_norm),
        "microbatch_loss_mean": _optional_scalar_float(metrics.microbatch_loss_mean),
        "microbatch_loss_max": _optional_scalar_float(metrics.microbatch_loss_max),
        "batch_het": _optional_scalar_float(metrics.batch_het),
        "epoch": provenance.epoch,
        "token_start": provenance.token_start,
        "token_end": provenance.token_end,
        "examples": provenance.examples,
        "target_tokens": provenance.target_tokens,
    }
    router_layers = _router_layers_payload(metrics.router_expert_counts, metrics.router_importance)
    if router_layers is not None:
        row["moe_router_layers"] = router_layers
    optimizer_groups = _optimizer_groups_payload(
        metrics.optimizer_group_specs,
        metrics.optimizer_group_grad_norms,
        metrics.optimizer_group_update_norms,
        metrics.optimizer_group_param_norms,
    )
    row.update(_optimizer_health_fields(optimizer_groups))
    if optimizer_groups is not None:
        row["optimizer_groups"] = optimizer_groups
    row.update(_document_metric_fields(provenance.row_doc_ids))
    if runtime_spec is not None and context is not None:
        data_axis_size = context.data_axis_size
        per_device_batch_size = runtime_spec.training.global_batch_size // data_axis_size
        micro_tokens_per_step = runtime_spec.training.global_batch_size * runtime_spec.training.seq_len
        effective_global_batch_size = (
            runtime_spec.training.global_batch_size
            * runtime_spec.training.gradient_accumulation_steps
        )
        row.update(
            {
                "data_axis_size": data_axis_size,
                "global_batch_size": runtime_spec.training.global_batch_size,
                "per_device_batch_size": per_device_batch_size,
                "gradient_accumulation_steps": runtime_spec.training.gradient_accumulation_steps,
                "micro_global_batch_size": runtime_spec.training.global_batch_size,
                "effective_global_batch_size": effective_global_batch_size,
                "micro_tokens_per_step": micro_tokens_per_step,
                "effective_tokens_per_step": provenance.target_tokens,
                "global_target_tokens": provenance.target_tokens,
                "per_device_target_tokens": provenance.target_tokens // data_axis_size,
                "data_pipeline_backend": _data_pipeline_backend_name(runtime_spec),
                "data_order": runtime_spec.data.order,
                "data_worker_count": runtime_spec.data.worker_count,
                "data_prefetch": runtime_spec.data.prefetch,
                "data_worker_buffer_size": runtime_spec.data.worker_buffer_size,
            }
        )
    return row


def _document_metric_fields(row_doc_ids: tuple[int, ...] | None) -> dict[str, Any]:
    if row_doc_ids is None:
        return {
            "document_aware": False,
            "documents_touched": None,
            "document_min": None,
            "document_max": None,
        }
    if not row_doc_ids:
        raise ContractError("document-aware provenance must include at least one document id")
    return {
        "document_aware": True,
        "documents_touched": len(set(row_doc_ids)),
        "document_min": min(row_doc_ids),
        "document_max": max(row_doc_ids),
    }


def _save_checkpoint(
    checkpoint_service: LocalOrbaxCheckpointService,
    writer: ArtifactWriter,
    spec: RunSpec,
    train_state: TrainState,
    dataset_state: DataPipelineState,
    host_state: HostState,
    checkpoint_index: CheckpointIndex,
    row: dict[str, Any],
    eval_result: EvalRunResult | None,
    *,
    reason: str,
    progress: TrainingProgress | None = None,
) -> tuple[HostState, CheckpointIndex]:
    checkpoint_started_at = time.perf_counter()
    step = row["step"]
    next_host_state = HostState(
        dataset=dataset_state,
        last_checkpoint_step=step,
        wallclock_start_ns=host_state.wallclock_start_ns,
        run_id=host_state.run_id,
    )
    checkpoint_service.set_protected_steps(checkpoint_index.protected_steps())
    checkpoint_service.save(
        step,
        train_state,
        dataset_state,
        next_host_state,
        checkpoint_metadata(spec, row, reason=reason, eval_row=None if eval_result is None else eval_result.row),
    )
    checkpoint_path = checkpoint_service.latest_path()
    next_index = record_checkpoint(
        checkpoint_index,
        spec.dirs.run_dir,
        step=step,
        tokens_seen=row["tokens_seen"],
        checkpoint_path=checkpoint_path or spec.dirs.checkpoints_dir / f"{step:06d}",
        reason=reason,
        train_loss=row["loss"],
        eval_loss=None if eval_result is None else eval_result.row["loss"],
    )
    checkpoint_service.set_protected_steps(next_index.protected_steps())
    writer.write_checkpoint_index(next_index.to_dict())
    checkpoint_sec = time.perf_counter() - checkpoint_started_at
    writer.append_event(
        {
            **_event("checkpoint_saved", spec),
            "step": step,
            "tokens_seen": row["tokens_seen"],
            "checkpoint_path": checkpoint_path,
            "reason": reason,
            "checkpoint_sec": checkpoint_sec,
            "data_pipeline_backend": dataset_state.backend,
            "data_pipeline_order": dataset_state.order,
            "data_pipeline_shuffle_seed": dataset_state.shuffle_seed,
        }
    )
    if progress is not None:
        progress(
            "checkpoint",
            {
                "step": step,
                "tokens_seen": row["tokens_seen"],
                "checkpoint_path": checkpoint_path,
                "reason": reason,
                "checkpoint_sec": checkpoint_sec,
                "eval_loss": None if eval_result is None else eval_result.row["loss"],
            },
        )
    return next_host_state, next_index


def _should_log(step: int, log_every_steps: int, target_tokens: int, tokens_seen: int) -> bool:
    return step % log_every_steps == 0 or tokens_seen >= target_tokens


def _event(event_type: str, spec: RunSpec) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "type": event_type,
        "run_id": spec.run_id,
        "created_at": _utc_now(),
    }


def _train_sync_target(train_state: TrainState, metrics: Any) -> tuple[Any, ...]:
    values = [
        train_state.step,
        train_state.tokens_seen,
        metrics.loss_sum,
        metrics.token_count,
        metrics.lr,
    ]
    for value in (
        metrics.grad_norm,
        metrics.param_norm,
        metrics.update_norm,
        metrics.optimizer_group_grad_norms,
        metrics.optimizer_group_update_norms,
        metrics.optimizer_group_param_norms,
        metrics.overflow,
        metrics.objective,
        metrics.aux_loss,
        metrics.z_loss,
        metrics.moe_aux_loss,
        metrics.total_loss,
        metrics.router_expert_counts,
        metrics.router_importance,
        metrics.router_max_vio,
        metrics.router_load_min,
        metrics.router_load_max,
        metrics.router_load_entropy,
        metrics.router_mean_load_cv,
        metrics.router_std_load_cv,
        metrics.router_mean_load_entropy,
        metrics.router_min_load_entropy,
        metrics.router_dead_experts_count,
        metrics.router_experts_active_mean,
        metrics.router_mean_importance_cv,
        metrics.router_mean_importance_entropy,
        metrics.smebu_bias_norm,
        metrics.smebu_momentum_norm,
        metrics.microbatch_loss_mean,
        metrics.microbatch_loss_max,
        metrics.batch_het,
    ):
        if value is not None:
            values.append(value)
    return tuple(values)


def _optional_scalar_float(value: Any) -> float | None:
    return None if value is None else _scalar_float(value)


def _optimizer_groups_payload(
    specs: Any,
    grad_norms: Any,
    update_norms: Any,
    param_norms: Any,
) -> list[dict[str, Any]] | None:
    if not specs or grad_norms is None or update_norms is None or param_norms is None:
        return None
    grad_array = np.asarray(jax.device_get(grad_norms), dtype=np.float64)
    update_array = np.asarray(jax.device_get(update_norms), dtype=np.float64)
    param_array = np.asarray(jax.device_get(param_norms), dtype=np.float64)
    rows = []
    for idx, spec in enumerate(specs):
        param_norm = float(param_array[idx])
        grad_norm = float(grad_array[idx])
        update_norm = float(update_array[idx])
        rows.append(
            {
                "group": spec["group"],
                "tag": spec["tag"],
                "backend": spec["backend"],
                "parameter_count": int(spec["parameter_count"]),
                "leaf_count": int(spec["leaf_count"]),
                "grad_norm": grad_norm,
                "update_norm": update_norm,
                "param_norm": param_norm,
                "grad_param_ratio": _safe_ratio(grad_norm, param_norm),
                "update_param_ratio": _safe_ratio(update_norm, param_norm),
                "weight_decay_enabled_count": int(spec["weight_decay_enabled_count"]),
                "auto_resolved_count": int(spec["auto_resolved_count"]),
            }
        )
    return rows


def _optimizer_health_fields(groups: list[dict[str, Any]] | None) -> dict[str, Any]:
    empty = {
        "optimizer_grad_norm_max_group": None,
        "optimizer_update_norm_max_group": None,
        "optimizer_update_param_ratio_max": None,
        "optimizer_update_param_ratio_mean": None,
        "optimizer_groups_with_zero_grad": None,
        "optimizer_groups_with_zero_update": None,
        "optimizer_route_backend_counts": None,
    }
    if not groups:
        return empty
    max_grad = max(groups, key=lambda item: item["grad_norm"])
    max_update = max(groups, key=lambda item: item["update_norm"])
    ratios = [float(item["update_param_ratio"]) for item in groups]
    backend_counts: dict[str, int] = {}
    for item in groups:
        backend = item["backend"]
        backend_counts[backend] = backend_counts.get(backend, 0) + int(item["leaf_count"])
    return {
        "optimizer_grad_norm_max_group": max_grad["group"],
        "optimizer_update_norm_max_group": max_update["group"],
        "optimizer_update_param_ratio_max": max(ratios),
        "optimizer_update_param_ratio_mean": float(np.mean(ratios)),
        "optimizer_groups_with_zero_grad": sum(1 for item in groups if item["grad_norm"] <= 0.0),
        "optimizer_groups_with_zero_update": sum(1 for item in groups if item["update_norm"] <= 0.0),
        "optimizer_route_backend_counts": dict(sorted(backend_counts.items())),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _router_layers_payload(counts: Any, importance: Any) -> list[dict[str, Any]] | None:
    if counts is None or importance is None:
        return None
    counts_array = np.asarray(jax.device_get(counts), dtype=np.float64)
    importance_array = np.asarray(jax.device_get(importance), dtype=np.float64)
    if counts_array.ndim != 2 or counts_array.shape[0] == 0:
        return None
    rows = []
    for layer_index, (layer_counts, layer_importance) in enumerate(
        zip(counts_array, importance_array, strict=True)
    ):
        load_percent = _percent_distribution(layer_counts)
        importance_percent = _percent_distribution(layer_importance)
        rows.append(
            {
                "layer_index": layer_index,
                "total_assignments": float(np.sum(layer_counts)),
                "expert_counts": [int(value) for value in layer_counts.tolist()],
                "load_min": _quantile(load_percent, 0.0),
                "load_p10": _quantile(load_percent, 0.10),
                "load_p50": _quantile(load_percent, 0.50),
                "load_p90": _quantile(load_percent, 0.90),
                "load_max": _quantile(load_percent, 1.0),
                "load_cv": _cv(layer_counts),
                "load_entropy": _entropy(layer_counts),
                "experts_active": int(np.count_nonzero(layer_counts > 0.0)),
                "importance": [float(value) for value in layer_importance.tolist()],
                "importance_min": _quantile(importance_percent, 0.0),
                "importance_p10": _quantile(importance_percent, 0.10),
                "importance_p50": _quantile(importance_percent, 0.50),
                "importance_p90": _quantile(importance_percent, 0.90),
                "importance_max": _quantile(importance_percent, 1.0),
                "importance_cv": _cv(layer_importance),
                "importance_entropy": _entropy(layer_importance),
            }
        )
    return rows


def _percent_distribution(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    if total <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return values / total * 100.0


def _quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, q))


def _cv(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    if mean <= 0.0:
        return 0.0
    return float(np.std(values) / mean * 100.0)


def _entropy(values: np.ndarray) -> float:
    total = float(np.sum(values))
    if total <= 0.0:
        return 0.0
    probabilities = values / total
    probabilities = probabilities[probabilities > 0.0]
    return float(-np.sum(probabilities * np.log(probabilities)))


def _scalar_float(value: Any) -> float:
    return float(np.asarray(jax.device_get(value)).item())


def _scalar_int(value: Any) -> int:
    return int(np.asarray(jax.device_get(value)).item())


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
