"""Local checkpoint index artifacts."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass, replace
import json
from pathlib import Path
from typing import Any

from jaxtitan.errors import ContractError

CHECKPOINT_INDEX_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    """One local checkpoint record."""

    step: int
    tokens_seen: int
    checkpoint_path: Path
    reason: str
    train_loss: float
    eval_loss: float | None = None
    retained: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        if self.step < 0:
            raise ContractError(f"checkpoint index step must be non-negative, got {self.step}")
        if self.tokens_seen < 0:
            raise ContractError(f"checkpoint index tokens_seen must be non-negative, got {self.tokens_seen}")
        if not self.reason:
            raise ContractError("checkpoint index reason must be non-empty")


@dataclass(frozen=True, slots=True)
class CheckpointIndex:
    """Canonical local checkpoint index."""

    records: tuple[CheckpointRecord, ...] = ()
    schema_version: int = CHECKPOINT_INDEX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_INDEX_SCHEMA_VERSION:
            raise ContractError(
                f"checkpoint index schema_version must be {CHECKPOINT_INDEX_SCHEMA_VERSION}, got {self.schema_version}"
            )
        object.__setattr__(self, "records", tuple(sorted(self.records, key=lambda record: record.step)))

    @property
    def retained_records(self) -> tuple[CheckpointRecord, ...]:
        return tuple(record for record in self.records if record.retained)

    @property
    def latest_record(self) -> CheckpointRecord | None:
        retained = self.retained_records
        return None if not retained else max(retained, key=lambda record: record.step)

    @property
    def best_record(self) -> CheckpointRecord | None:
        candidates = [record for record in self.retained_records if record.eval_loss is not None]
        if not candidates:
            return None
        return min(candidates, key=lambda record: (record.eval_loss, record.step))

    def protected_steps(self) -> set[int]:
        protected = set()
        latest = self.latest_record
        best = self.best_record
        if latest is not None:
            protected.add(latest.step)
        if best is not None:
            protected.add(best.step)
        return protected

    def to_dict(self) -> dict[str, Any]:
        latest = self.latest_record
        best = self.best_record
        return {
            "schema_version": self.schema_version,
            "latest_step": None if latest is None else latest.step,
            "latest_checkpoint_path": None if latest is None else latest.checkpoint_path.as_posix(),
            "best_eval_step": None if best is None else best.step,
            "best_eval_loss": None if best is None else best.eval_loss,
            "best_checkpoint_path": None if best is None else best.checkpoint_path.as_posix(),
            "records": [_normalize(record) for record in self.records],
        }


def load_checkpoint_index(run_dir: str | Path) -> CheckpointIndex:
    """Load and refresh a checkpoint index from a run directory."""

    run_dir = Path(run_dir)
    path = run_dir / "checkpoints" / "index.json"
    if not path.exists():
        return CheckpointIndex()
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ContractError(f"failed to parse checkpoint index {path}: {exc}") from exc
    return refresh_checkpoint_index(_index_from_mapping(_require_mapping(raw, "checkpoint index")), run_dir)


def record_checkpoint(
    index: CheckpointIndex,
    run_dir: str | Path,
    *,
    step: int,
    tokens_seen: int,
    checkpoint_path: str | Path,
    reason: str,
    train_loss: float,
    eval_loss: float | None,
) -> CheckpointIndex:
    """Return an updated checkpoint index with one checkpoint record."""

    run_dir = Path(run_dir)
    relative_path = _relative_checkpoint_path(run_dir, checkpoint_path)
    next_record = CheckpointRecord(
        step=step,
        tokens_seen=tokens_seen,
        checkpoint_path=relative_path,
        reason=reason,
        train_loss=train_loss,
        eval_loss=eval_loss,
        retained=(run_dir / relative_path).is_dir(),
    )
    records = [record for record in index.records if record.step != step]
    records.append(next_record)
    return refresh_checkpoint_index(CheckpointIndex(records=tuple(records)), run_dir)


def refresh_checkpoint_index(index: CheckpointIndex, run_dir: str | Path) -> CheckpointIndex:
    """Refresh retained flags from local checkpoint directories."""

    run_dir = Path(run_dir)
    records = tuple(
        replace(record, retained=(run_dir / record.checkpoint_path).is_dir())
        for record in index.records
    )
    return CheckpointIndex(records=records)


def checkpoint_index_to_json(index: CheckpointIndex) -> str:
    """Serialize a checkpoint index as canonical JSON."""

    return json.dumps(index.to_dict(), sort_keys=True, separators=(",", ":"))


def _index_from_mapping(raw: Mapping[str, Any]) -> CheckpointIndex:
    schema_version = _required_int(raw, "schema_version", "checkpoint index")
    records_raw = raw.get("records", [])
    if not isinstance(records_raw, list):
        raise ContractError("checkpoint index records must be a list")
    return CheckpointIndex(
        schema_version=schema_version,
        records=tuple(_record_from_mapping(_require_mapping(item, "checkpoint record")) for item in records_raw),
    )


def _record_from_mapping(raw: Mapping[str, Any]) -> CheckpointRecord:
    return CheckpointRecord(
        step=_required_int(raw, "step", "checkpoint record"),
        tokens_seen=_required_int(raw, "tokens_seen", "checkpoint record"),
        checkpoint_path=Path(_required_str(raw, "checkpoint_path", "checkpoint record")),
        reason=_required_str(raw, "reason", "checkpoint record"),
        train_loss=_required_float(raw, "train_loss", "checkpoint record"),
        eval_loss=_optional_float(raw, "eval_loss", "checkpoint record"),
        retained=_required_bool(raw, "retained", "checkpoint record"),
    )


def _relative_checkpoint_path(run_dir: Path, checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path)
    if path.is_absolute():
        try:
            path = path.relative_to(run_dir.resolve())
        except ValueError:
            raise ContractError(f"checkpoint path {path} is outside run directory {run_dir}") from None
    else:
        try:
            path = path.relative_to(run_dir)
        except ValueError:
            pass
    return path


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be a JSON object")
    return value


def _required_int(raw: Mapping[str, Any], key: str, name: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int):
        raise ContractError(f"{name}.{key} must be an integer")
    return value


def _required_float(raw: Mapping[str, Any], key: str, name: str) -> float:
    value = raw.get(key)
    if not isinstance(value, int | float):
        raise ContractError(f"{name}.{key} must be numeric")
    return float(value)


def _optional_float(raw: Mapping[str, Any], key: str, name: str) -> float | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise ContractError(f"{name}.{key} must be numeric or null")
    return float(value)


def _required_bool(raw: Mapping[str, Any], key: str, name: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise ContractError(f"{name}.{key} must be a boolean")
    return value


def _required_str(raw: Mapping[str, Any], key: str, name: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name}.{key} must be a non-empty string")
    return value


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
