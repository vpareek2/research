"""Stable RunSpec serialization and hashing."""

from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from jaxtitan.specs.run import RunSpec


def run_spec_to_dict(spec: RunSpec) -> dict[str, Any]:
    """Convert a RunSpec into JSON-compatible data."""

    return _normalize(asdict(spec))


def run_spec_to_json(spec: RunSpec) -> str:
    """Serialize a RunSpec as canonical JSON."""

    return json.dumps(run_spec_to_dict(spec), sort_keys=True, separators=(",", ":"))


def resolved_config_sha256(spec: RunSpec) -> str:
    """Return the SHA256 hash of canonical resolved RunSpec JSON."""

    return sha256(run_spec_to_json(spec).encode("utf-8")).hexdigest()


def source_config_sha256(path: str | Path) -> str:
    """Return the SHA256 hash of source TOML bytes."""

    return sha256(Path(path).read_bytes()).hexdigest()


def _normalize(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_normalize(item) for item in value]
    return value
