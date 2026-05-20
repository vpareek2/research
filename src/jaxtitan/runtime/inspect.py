"""Read-only local run artifact inspection."""

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from jaxtitan.errors import ContractError
from jaxtitan.runtime.checkpoint_index import load_checkpoint_index


@dataclass(frozen=True, slots=True)
class RunInspection:
    """Stable read-only summary of one local run directory."""

    payload: dict[str, Any]


def inspect_run(run_dir: str | Path) -> RunInspection:
    """Inspect canonical local artifacts for a completed run."""

    run_dir = Path(run_dir)
    manifest = _read_required_json(run_dir / "manifest.json", "run manifest")
    final = _read_required_json(run_dir / "summaries" / "final.json", "final summary")
    raw_index = _read_required_json(run_dir / "checkpoints" / "index.json", "checkpoint index")
    diagnostics = _read_optional_json(run_dir / "diagnostics" / "runtime.json", "runtime diagnostics")
    index = load_checkpoint_index(run_dir)
    _validate_checkpoint_paths(run_dir, raw_index)

    latest = index.latest_record
    best = index.best_record
    payload = {
        "schema_version": 1,
        "run_dir": run_dir.as_posix(),
        "run_id": _required_str(manifest, "run_id", "run manifest"),
        "status": final.get("status"),
        "final": {
            "step": final.get("steps"),
            "tokens_seen": final.get("tokens_seen"),
            "target_tokens": final.get("target_tokens"),
            "loss": final.get("final_loss"),
            "eval_loss": final.get("final_eval_loss"),
            "eval_token_count": final.get("final_eval_token_count"),
            "eval_num_batches": final.get("final_eval_num_batches"),
        },
        "latest_checkpoint": None if latest is None else _checkpoint_summary(run_dir, latest),
        "best_checkpoint": None if best is None else _checkpoint_summary(run_dir, best),
        "checkpoints": [_checkpoint_summary(run_dir, record) for record in index.records],
        "diagnostics": _diagnostics_summary(diagnostics),
        "recent_train_metrics": _read_recent_jsonl(run_dir / "metrics" / "train.jsonl"),
        "recent_eval_metrics": _read_recent_jsonl(run_dir / "metrics" / "eval.jsonl"),
    }
    return RunInspection(payload=payload)


def run_inspection_to_json(inspection: RunInspection) -> str:
    """Serialize an inspection payload as canonical JSON."""

    return _canonical_json(inspection.payload)


def format_run_inspection(inspection: RunInspection) -> str:
    """Format a run inspection for humans."""

    payload = inspection.payload
    final = payload["final"]
    lines = [
        f"run: {payload['run_id']}",
        f"status: {payload['status']}",
        f"final: step={final['step']} tokens={final['tokens_seen']}/{final['target_tokens']} loss={final['loss']}",
    ]
    if final["eval_loss"] is not None:
        lines.append(
            f"validation: loss={final['eval_loss']} tokens={final['eval_token_count']} batches={final['eval_num_batches']}"
        )
    diagnostics = payload["diagnostics"]
    if diagnostics is not None:
        lines.append(
            "runtime: "
            f"backend={diagnostics['jax_backend']} device={diagnostics['device_kind']} "
            f"devices={diagnostics['device_count']}"
        )
        parallelism = diagnostics.get("parallelism")
        if parallelism is not None:
            batch = parallelism["batch"]
            mesh = parallelism["mesh"]
            lines.append(
                "parallelism: "
                f"mode={parallelism['execution_mode']} metrics={parallelism['metrics_scope']} "
                f"artifacts={parallelism['artifact_writer']} data_axis={mesh['data_axis_size']} "
                f"global_batch={batch['global_batch_size']} per_device_batch={batch['per_device_batch_size']}"
            )
        data_pipeline = diagnostics.get("data_pipeline")
        if data_pipeline is not None:
            lines.append(
                "data pipeline: "
                f"backend={data_pipeline['backend']} version={data_pipeline['backend_version']} "
                f"order={data_pipeline['order']} shuffle_seed={data_pipeline['shuffle_seed']} "
                f"workers={data_pipeline['worker_count']} prefetch={data_pipeline['prefetch']}"
            )
    latest = payload["latest_checkpoint"]
    best = payload["best_checkpoint"]
    lines.append(f"latest checkpoint: {_format_checkpoint_ref(latest)}")
    lines.append(f"best checkpoint: {_format_checkpoint_ref(best)}")
    lines.append("checkpoints:")
    checkpoints = payload["checkpoints"]
    if not checkpoints:
        lines.append("  none")
    for record in checkpoints:
        lines.append(
            "  "
            f"step={record['step']} tokens={record['tokens_seen']} reason={record['reason']} "
            f"train_loss={record['train_loss']} eval_loss={record['eval_loss']} "
            f"retained={record['retained']} path={record['checkpoint_path']}"
        )
    return "\n".join(lines)


def _checkpoint_summary(run_dir: Path, record: Any) -> dict[str, Any]:
    path = record.checkpoint_path
    exists = (run_dir / path).is_dir()
    return {
        "step": record.step,
        "tokens_seen": record.tokens_seen,
        "checkpoint_path": path.as_posix(),
        "reason": record.reason,
        "train_loss": record.train_loss,
        "eval_loss": record.eval_loss,
        "retained": record.retained,
        "exists": exists,
    }


def _format_checkpoint_ref(record: Mapping[str, Any] | None) -> str:
    if record is None:
        return "none"
    eval_loss = record["eval_loss"]
    suffix = "" if eval_loss is None else f" eval_loss={eval_loss}"
    return f"step={record['step']} path={record['checkpoint_path']}{suffix}"


def _validate_checkpoint_paths(run_dir: Path, raw_index: Mapping[str, Any]) -> None:
    records = raw_index.get("records")
    if not isinstance(records, list):
        raise ContractError("checkpoint index records must be a list")
    for idx, item in enumerate(records):
        if not isinstance(item, Mapping):
            raise ContractError(f"checkpoint index record {idx} must be a JSON object")
        if item.get("retained") is True:
            path = item.get("checkpoint_path")
            if not isinstance(path, str) or not path:
                raise ContractError(f"checkpoint index record {idx} checkpoint_path must be a non-empty string")
            if not (run_dir / path).is_dir():
                raise ContractError(f"checkpoint index retained path does not exist: {path}")


def _diagnostics_summary(raw: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    performance = raw.get("performance")
    jax_info = raw.get("jax")
    parallelism = raw.get("parallelism")
    sharding = raw.get("sharding")
    data_pipeline = raw.get("data_pipeline")
    if not isinstance(performance, Mapping) or not isinstance(jax_info, Mapping):
        raise ContractError("runtime diagnostics must include performance and jax objects")
    if parallelism is not None and not isinstance(parallelism, Mapping):
        raise ContractError("runtime diagnostics parallelism must be an object")
    if sharding is not None and not isinstance(sharding, Mapping):
        raise ContractError("runtime diagnostics sharding must be an object")
    if data_pipeline is not None and not isinstance(data_pipeline, Mapping):
        raise ContractError("runtime diagnostics data_pipeline must be an object")
    return {
        "path": "diagnostics/runtime.json",
        "jax_backend": jax_info.get("backend"),
        "process_count": jax_info.get("process_count"),
        "process_index": jax_info.get("process_index"),
        "device_kind": performance.get("device_kind"),
        "device_count": performance.get("device_count"),
        "flops_per_token": performance.get("flops_per_token"),
        "peak_flops_per_device": performance.get("peak_flops_per_device"),
        "peak_flops_total": performance.get("peak_flops_total"),
        "parallelism": None if parallelism is None else _parallelism_summary(parallelism),
        "sharding": sharding,
        "data_pipeline": None if data_pipeline is None else _data_pipeline_summary(data_pipeline),
    }


def _parallelism_summary(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "execution_mode": raw.get("execution_mode"),
        "metrics_scope": raw.get("metrics_scope"),
        "artifact_writer": raw.get("artifact_writer"),
        "single_process": raw.get("single_process"),
        "process": raw.get("process"),
        "devices": raw.get("devices"),
        "mesh": raw.get("mesh"),
        "batch": raw.get("batch"),
        "host_artifacts": raw.get("host_artifacts"),
    }


def _data_pipeline_summary(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "backend": raw.get("backend"),
        "backend_version": raw.get("backend_version"),
        "state_schema_version": raw.get("state_schema_version"),
        "split": raw.get("split"),
        "order": raw.get("order"),
        "shuffle_seed": raw.get("shuffle_seed"),
        "worker_count": raw.get("worker_count"),
        "worker_buffer_size": raw.get("worker_buffer_size"),
        "prefetch": raw.get("prefetch"),
        "batch_size": raw.get("batch_size"),
        "seq_len": raw.get("seq_len"),
        "num_records": raw.get("num_records"),
        "manifest_sha256": raw.get("manifest_sha256"),
        "tokenizer_id": raw.get("tokenizer_id"),
    }


def _read_required_json(path: Path, name: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ContractError(f"missing {name}: {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ContractError(f"failed to parse {name} {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ContractError(f"{name} must be a JSON object")
    return raw


def _read_optional_json(path: Path, name: str) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    return _read_required_json(path, name)


def _read_recent_jsonl(path: Path, *, limit: int = 5) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"failed to parse JSONL row {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ContractError(f"JSONL row {path}:{line_number} must be an object")
        rows.append(row)
    return rows[-limit:]


def _required_str(raw: Mapping[str, Any], key: str, name: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name}.{key} must be a non-empty string")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
