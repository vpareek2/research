#!/usr/bin/env python3
"""Audit persistent distributed-Muon checkpoint replicas without timing training."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jaxtitan.config import load_config, load_resolved_config
from jaxtitan.errors import ContractError
from jaxtitan.mesh import (
    build_mesh_context,
    build_sharding_plan,
    gradient_shardings_like,
    place_model_state,
    place_optimizer_init_state,
)
from jaxtitan.models import build_model
from jaxtitan.optim import build_optimizer
from jaxtitan.runtime.checkpoint_index import CheckpointRecord, load_checkpoint_index
from jaxtitan.runtime.resume import validate_resume_compat, validate_resume_metadata
from jaxtitan.runtime.training import _moe_balance_spec, _with_runtime_schedule_steps
from jaxtitan.services import LocalOrbaxCheckpointService
from jaxtitan.steps import initialize_train_state

AUDIT_SCHEMA_VERSION = 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Completed Jaxtitan run directory.")
    parser.add_argument("--checkpoint", default="latest", help="Retained checkpoint selector.")
    parser.add_argument("--json-out", help="Optional output path; stdout is always JSON.")
    args = parser.parse_args()

    payload = audit_checkpoint(args.run_dir, checkpoint=args.checkpoint)
    encoded = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if payload["overall_gate"] else 1


def audit_checkpoint(
    run_dir: str | Path,
    *,
    checkpoint: str = "latest",
) -> dict[str, Any]:
    """Restore one full train state and audit persistent model/optimizer replicas."""

    run_dir = Path(run_dir)
    runtime_spec = _with_runtime_schedule_steps(_load_run_spec(run_dir))
    record = _select_checkpoint(load_checkpoint_index(run_dir).records, checkpoint, run_dir)

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
    optimizer = build_optimizer(
        runtime_spec.optimizer,
        optimizer_init_state,
        model.metadata,
        runtime_parameter_state=model_state,
        gradient_shardings=gradient_shardings_like(model_state, sharding),
    )
    template = initialize_train_state(
        model_state,
        optimizer.transform,
        seed=runtime_spec.seed,
        optimizer_init_model_state=optimizer_init_state,
        moe_balance_spec=_moe_balance_spec(runtime_spec),
    )

    service = LocalOrbaxCheckpointService(run_dir)
    try:
        metadata = service.restore_metadata(record.step)
        validate_resume_metadata(metadata, runtime_spec)
        restored = service.restore(record.step, template)
        validate_resume_compat(restored, runtime_spec)
    finally:
        service.close()

    sections = {
        "model": audit_tree(restored.train_state.model),
        "optimizer": audit_tree(restored.train_state.opt_state),
    }
    overall_gate = all(section["gate"] for section in sections.values())
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "run_id": runtime_spec.run_id,
        "checkpoint": {
            "selector": checkpoint,
            "step": restored.step,
            "path": str(record.checkpoint_path),
            "runtime_fingerprint": restored.metadata["runtime_fingerprint"],
        },
        "sections": sections,
        "array_count": sum(section["array_count"] for section in sections.values()),
        "replicated_array_count": sum(
            section["replicated_array_count"] for section in sections.values()
        ),
        "finite": all(section["finite"] for section in sections.values()),
        "max_replica_abs_diff": max(
            section["max_replica_abs_diff"] for section in sections.values()
        ),
        "overall_gate": overall_gate,
    }


def audit_tree(tree: Any) -> dict[str, Any]:
    """Audit finite values and physically duplicated shards in one PyTree."""

    array_count = 0
    replicated_array_count = 0
    nonfinite_paths = []
    disagreement_paths = []
    maximum_difference = 0.0
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        shards = tuple(getattr(value, "addressable_shards", ()))
        if not shards:
            continue
        array_count += 1
        path_text = jax.tree_util.keystr(path)
        if not _shards_finite(shards):
            nonfinite_paths.append(path_text)
        grouped: dict[str, list[Any]] = {}
        for shard in shards:
            grouped.setdefault(str(shard.index), []).append(shard.data)
        has_replicas = any(len(replicas) > 1 for replicas in grouped.values())
        if has_replicas:
            replicated_array_count += 1
        leaf_difference = 0.0
        for replicas in grouped.values():
            if len(replicas) < 2:
                continue
            reference = replicas[0]
            for replica in replicas[1:]:
                if not _arrays_equal(reference, replica):
                    leaf_difference = max(
                        leaf_difference,
                        _array_max_abs_difference(reference, replica),
                    )
        if leaf_difference != 0.0:
            disagreement_paths.append(path_text)
            maximum_difference = max(maximum_difference, leaf_difference)
    finite = not nonfinite_paths
    replicas_equal = not disagreement_paths
    return {
        "array_count": array_count,
        "replicated_array_count": replicated_array_count,
        "finite": finite,
        "nonfinite_paths": nonfinite_paths,
        "replicas_equal": replicas_equal,
        "max_replica_abs_diff": maximum_difference,
        "replica_disagreement_paths": disagreement_paths,
        "gate": array_count > 0 and finite and replicas_equal,
    }


def _shards_finite(shards: tuple[Any, ...]) -> bool:
    return all(
        bool(np.asarray(jax.device_get(jnp.all(jnp.isfinite(shard.data)))).item())
        for shard in shards
    )


def _arrays_equal(left: Any, right: Any) -> bool:
    right = jax.device_put(right, left.device)
    return bool(np.asarray(jax.device_get(jnp.array_equal(left, right))).item())


def _array_max_abs_difference(left: Any, right: Any) -> float:
    right = jax.device_put(right, left.device)
    difference = jnp.max(
        jnp.abs(left.astype(jnp.float32) - right.astype(jnp.float32)),
        initial=jnp.asarray(0.0, dtype=jnp.float32),
    )
    value = float(np.asarray(jax.device_get(difference)).item())
    return value if math.isfinite(value) else math.inf


def _load_run_spec(run_dir: Path):
    resolved_path = run_dir / "config" / "resolved.json"
    source_path = run_dir / "config" / "source.toml"
    if resolved_path.is_file():
        spec = load_resolved_config(resolved_path)
    elif source_path.is_file():
        spec = load_config(source_path)
    else:
        raise ContractError(f"missing run config artifact under {run_dir / 'config'}")
    if spec.run_id != run_dir.name:
        raise ContractError(
            f"run config id {spec.run_id!r} does not match run directory {run_dir.name!r}"
        )
    return replace(spec, output_dir=run_dir.parent)


def _select_checkpoint(
    records: tuple[CheckpointRecord, ...],
    selector: str,
    run_dir: Path,
) -> CheckpointRecord:
    retained = [
        record
        for record in records
        if record.retained and (run_dir / record.checkpoint_path).is_dir()
    ]
    if selector == "latest":
        if not retained:
            raise ContractError("checkpoint index has no retained latest checkpoint")
        return max(retained, key=lambda record: record.step)
    try:
        step = int(selector)
    except ValueError as exc:
        raise ContractError(
            f"checkpoint selector must be 'latest' or a step, got {selector!r}"
        ) from exc
    for record in retained:
        if record.step == step:
            return record
    raise ContractError(f"checkpoint step {step} is not retained in checkpoint index")


if __name__ == "__main__":
    raise SystemExit(main())
