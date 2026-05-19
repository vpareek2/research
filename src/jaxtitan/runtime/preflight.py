"""Runtime readiness preflight checks."""

from collections.abc import Mapping
from dataclasses import dataclass, is_dataclass, asdict
import json
import math
from pathlib import Path
from typing import Any

import jax
import numpy as np

from jaxtitan.config import load_config
from jaxtitan.data import PreparedDataService
from jaxtitan.errors import ContractError
from jaxtitan.mesh import build_mesh_context, build_sharding_plan, place_batch, place_replicated
from jaxtitan.models import build_model, count_parameters
from jaxtitan.optim import build_optimizer
from jaxtitan.runtime.training import (
    _build_validation_eval_data,
    _metrics_row,
    _validation_eval_row,
    _validation_eval_spec,
    _with_runtime_schedule_steps,
)
from jaxtitan.steps import initialize_train_state, make_eval_step, make_train_step


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """Stable runtime readiness report."""

    payload: dict[str, Any]


def run_preflight(config_path: str | Path) -> PreflightReport:
    """Run a side-effect-free preflight over the local training path."""

    spec = load_config(config_path)
    runtime_spec = _with_runtime_schedule_steps(spec)
    if runtime_spec.dirs.run_dir.exists():
        raise ContractError(f"run directory already exists: {runtime_spec.dirs.run_dir}")

    train_data = PreparedDataService.from_manifest(
        runtime_spec.data.train_manifest,
        tokenizer_id=runtime_spec.data.tokenizer_id,
        split="train",
        seq_len=runtime_spec.training.seq_len,
        batch_size=runtime_spec.training.global_batch_size,
    )
    train_dataset_state = train_data.initial_state()
    train_batch, _next_dataset_state, train_provenance = train_data.next_batch(train_dataset_state, repeat=False)

    context = build_mesh_context(runtime_spec.mesh)
    sharding = build_sharding_plan(context)
    model = build_model(runtime_spec.model, seed=runtime_spec.seed)
    optimizer = build_optimizer(runtime_spec.optimizer, model.state, model.metadata)
    model_state = place_replicated(model.state, sharding)
    train_state = initialize_train_state(model_state, optimizer.transform, seed=runtime_spec.seed)
    train_step = make_train_step(model.graph, optimizer)

    next_train_state, train_metrics = train_step(train_state, place_batch(train_batch, sharding))
    jax.block_until_ready(next_train_state.step)
    jax.block_until_ready(train_metrics.loss_sum)
    train_row = _metrics_row(next_train_state, train_metrics, train_provenance)

    eval_spec = _validation_eval_spec(runtime_spec)
    eval_report = None
    if eval_spec is not None:
        eval_data = _build_validation_eval_data(runtime_spec)
        eval_step = make_eval_step(model.graph)
        eval_row = _validation_eval_row(
            eval_spec,
            eval_data,
            eval_step,
            sharding,
            train_state,
            {"step": 0, "tokens_seen": 0},
        )
        eval_report = {
            "name": eval_spec.name,
            "every_steps": eval_spec.every_steps,
            "num_batches": eval_spec.num_batches,
            "manifest": (
                runtime_spec.data.validation_manifest
                if runtime_spec.data.validation_manifest is not None
                else runtime_spec.data.train_manifest
            ),
            "split": "val",
            "token_start": eval_row["token_start"],
            "token_end": eval_row["token_end"],
            "examples": eval_row["examples"],
            "target_tokens": eval_row["target_tokens"],
            "loss": eval_row["loss"],
            "compile": "passed",
        }

    batch_tokens = runtime_spec.training.global_batch_size * runtime_spec.training.seq_len
    report = {
        "schema_version": 1,
        "status": "passed",
        "run_id": runtime_spec.run_id,
        "config_path": Path(config_path),
        "run_dir": runtime_spec.dirs.run_dir,
        "devices": {
            "local_device_count": context.local_device_count,
            "selected_device_count": len(context.devices),
            "platforms": sorted({str(getattr(device, "platform", "unknown")) for device in context.devices}),
            "kinds": sorted({str(getattr(device, "device_kind", "unknown")) for device in context.devices}),
        },
        "mesh": {
            "axis_names": runtime_spec.mesh.axis_names,
            "axis_sizes": runtime_spec.mesh.axis_sizes,
            "data_axis_size": context.data_axis_size,
        },
        "data": {
            "train_manifest": runtime_spec.data.train_manifest,
            "tokenizer_id": runtime_spec.data.tokenizer_id,
            "train_split_start": train_data.split_start,
            "train_split_end": train_data.split_end,
            "train_split_tokens": train_data.split_end - train_data.split_start,
            "first_batch": {
                "token_start": train_provenance.token_start,
                "token_end": train_provenance.token_end,
                "examples": train_provenance.examples,
                "target_tokens": train_provenance.target_tokens,
            },
        },
        "model": {
            "name": runtime_spec.model.name,
            "variant": runtime_spec.model.variant,
            "parameters": count_parameters(model.metadata),
            "parameter_leaves": len(model.metadata),
            "param_dtype": runtime_spec.model.param_dtype,
            "compute_dtype": runtime_spec.model.compute_dtype,
        },
        "optimizer": {
            "name": runtime_spec.optimizer.name,
            "schedule": runtime_spec.optimizer.schedule.name,
            "peak_lr": runtime_spec.optimizer.schedule.peak_lr,
            "total_steps": runtime_spec.optimizer.schedule.total_steps,
            "description": optimizer.description,
        },
        "training": {
            "seq_len": runtime_spec.training.seq_len,
            "global_batch_size": runtime_spec.training.global_batch_size,
            "target_tokens": runtime_spec.training.target_tokens,
            "batch_tokens": batch_tokens,
            "estimated_steps": math.ceil(runtime_spec.training.target_tokens / batch_tokens),
            "log_every_steps": runtime_spec.training.log_every_steps,
            "checkpoint_every_steps": runtime_spec.training.checkpoint_every_steps,
            "compile": "passed",
            "first_step_loss": train_row["loss"],
        },
        "eval": eval_report,
    }
    return PreflightReport(payload=_normalize(report))


def preflight_report_to_json(report: PreflightReport) -> str:
    """Serialize a preflight report as canonical JSON."""

    return _canonical_json(report.payload)


def format_preflight_report(report: PreflightReport) -> str:
    """Format a preflight report for humans."""

    payload = report.payload
    training = payload["training"]
    devices = payload["devices"]
    mesh = payload["mesh"]
    data = payload["data"]
    model = payload["model"]
    optimizer = payload["optimizer"]
    lines = [
        f"preflight: {payload['status']}",
        f"run: {payload['run_id']}",
        f"run_dir: {payload['run_dir']}",
        (
            "devices: "
            f"selected={devices['selected_device_count']} local={devices['local_device_count']} "
            f"platforms={','.join(devices['platforms'])}"
        ),
        f"mesh: axes={mesh['axis_names']} sizes={mesh['axis_sizes']}",
        (
            "train data: "
            f"manifest={data['train_manifest']} tokens={data['train_split_tokens']} "
            f"first_batch={data['first_batch']['target_tokens']}"
        ),
        (
            "model: "
            f"{model['name']} variant={model['variant']} parameters={model['parameters']} "
            f"param_dtype={model['param_dtype']} compute_dtype={model['compute_dtype']}"
        ),
        (
            "optimizer: "
            f"{optimizer['name']} schedule={optimizer['schedule']} "
            f"peak_lr={optimizer['peak_lr']} total_steps={optimizer['total_steps']}"
        ),
        (
            "training: "
            f"estimated_steps={training['estimated_steps']} batch_tokens={training['batch_tokens']} "
            f"log_every={training['log_every_steps']} checkpoint_every={training['checkpoint_every_steps']} "
            f"compile={training['compile']}"
        ),
    ]
    if payload["eval"] is None:
        lines.append("eval: skipped")
    else:
        eval_report = payload["eval"]
        lines.append(
            "eval: "
            f"name={eval_report['name']} every={eval_report['every_steps']} "
            f"batches={eval_report['num_batches']} compile={eval_report['compile']}"
        )
    return "\n".join(lines)


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
