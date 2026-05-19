"""Artifact service contracts and local artifact writer."""

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
from typing import Any, Protocol
from uuid import uuid4

from jaxtitan import __version__
from jaxtitan.config import load_config, resolved_config_sha256, run_spec_to_json, source_config_sha256
from jaxtitan.data import dataset_manifest_summary, validate_dataset_manifest
from jaxtitan.errors import ContractError
from jaxtitan.specs.run import RunManifest, RunSpec


class ArtifactWriter(Protocol):
    """Host-side writer for canonical local artifacts."""

    def write_config(self, source_toml: str, resolved: RunSpec) -> None: ...

    def append_event(self, event: Mapping[str, Any]) -> None: ...

    def append_train_metrics(self, row: Mapping[str, Any]) -> None: ...

    def append_eval_metrics(self, row: Mapping[str, Any]) -> None: ...

    def write_checkpoint_index(self, index: Mapping[str, Any]) -> None: ...

    def write_summary(self, summary: Mapping[str, Any]) -> None: ...


def initialize_run(config_path: str | Path) -> RunManifest:
    """Create the canonical local artifact skeleton for a config."""

    source_path = Path(config_path)
    spec = load_config(source_path)
    source_toml = source_path.read_text()
    dataset_manifest = validate_dataset_manifest(spec.data.train_manifest, tokenizer_id=spec.data.tokenizer_id)
    created_at = _utc_now()
    source_hash = source_config_sha256(source_path)
    resolved_hash = resolved_config_sha256(spec)

    manifest = RunManifest(
        schema_version=1,
        artifact_layout_version=1,
        run_id=spec.run_id,
        created_at=created_at,
        source_config_path=source_path,
        source_config_sha256=source_hash,
        resolved_config_sha256=resolved_hash,
        package={"name": "jaxtitan", "version": __version__},
        directories={
            "config": "config",
            "metrics": "metrics",
            "checkpoints": "checkpoints",
            "evals": "evals",
            "samples": "samples",
            "summaries": "summaries",
        },
        run_dir=spec.dirs.run_dir,
        data=dataset_manifest_summary(dataset_manifest),
    )
    LocalArtifactWriter.initialize(source_toml=source_toml, resolved=spec, manifest=manifest)
    return manifest


class LocalArtifactWriter:
    """Concrete local writer for canonical run artifacts."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)

    @classmethod
    def initialize(cls, *, source_toml: str, resolved: RunSpec, manifest: RunManifest) -> "LocalArtifactWriter":
        run_dir = manifest.run_dir
        if run_dir.exists():
            raise ContractError(f"run directory already exists: {run_dir}")

        root = run_dir.parent
        root.mkdir(parents=True, exist_ok=True)
        tmp_dir = root / f".{run_dir.name}.tmp-{uuid4().hex}"
        writer = cls(tmp_dir)
        try:
            writer._create_layout()
            writer.write_config(source_toml, resolved)
            writer.write_manifest(manifest)
            writer.append_event(
                {
                    "schema_version": 1,
                    "type": "run_initialized",
                    "run_id": manifest.run_id,
                    "created_at": manifest.created_at,
                    "source_config_sha256": manifest.source_config_sha256,
                    "resolved_config_sha256": manifest.resolved_config_sha256,
                }
            )
            tmp_dir.replace(run_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        return cls(run_dir)

    def write_config(self, source_toml: str, resolved: RunSpec) -> None:
        (self.run_dir / "config" / "source.toml").write_text(source_toml)
        (self.run_dir / "config" / "resolved.json").write_text(run_spec_to_json(resolved) + "\n")

    def write_manifest(self, manifest: RunManifest) -> None:
        manifest_row = _normalize(manifest)
        manifest_row.pop("run_dir", None)
        self._write_json(self.run_dir / "manifest.json", manifest_row)

    def append_event(self, event: Mapping[str, Any]) -> None:
        self._append_jsonl(self.run_dir / "events.jsonl", event)

    def append_train_metrics(self, row: Mapping[str, Any]) -> None:
        self._append_jsonl(self.run_dir / "metrics" / "train.jsonl", row)

    def append_eval_metrics(self, row: Mapping[str, Any]) -> None:
        self._append_jsonl(self.run_dir / "metrics" / "eval.jsonl", row)

    def write_checkpoint_index(self, index: Mapping[str, Any]) -> None:
        self._write_json(self.run_dir / "checkpoints" / "index.json", index)

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        self._write_json(self.run_dir / "summaries" / "final.json", summary)

    def _create_layout(self) -> None:
        for path in (
            self.run_dir / "config",
            self.run_dir / "metrics",
            self.run_dir / "checkpoints",
            self.run_dir / "evals",
            self.run_dir / "samples",
            self.run_dir / "summaries",
        ):
            path.mkdir(parents=True, exist_ok=False)

    def _write_json(self, path: Path, value: Any) -> None:
        path.write_text(_canonical_json(value) + "\n")

    def _append_jsonl(self, path: Path, row: Mapping[str, Any]) -> None:
        with path.open("a") as handle:
            handle.write(_canonical_json(dict(row)) + "\n")


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
    return value


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
