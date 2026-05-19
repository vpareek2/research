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
