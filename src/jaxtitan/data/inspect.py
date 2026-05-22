"""Read-only inspection helpers for prepared-token data artifacts."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from jaxtitan.data.manifest import PreparedDatasetManifest, prepared_dataset_manifest_to_dict, validate_dataset_manifest
from jaxtitan.errors import ContractError


@dataclass(frozen=True, slots=True)
class DataInspection:
    """Stable read-only summary of a prepared-token manifest."""

    manifest: PreparedDatasetManifest
    raw_manifest: dict[str, Any]
    seq_len: int | None = None


def inspect_dataset_manifest(
    path: str | Path,
    *,
    tokenizer_id: str | None = None,
    verify_checksums: bool = False,
    seq_len: int | None = None,
) -> DataInspection:
    """Validate and summarize a prepared-token manifest."""

    if seq_len is not None and seq_len <= 0:
        raise ContractError(f"data inspect --seq-len must be positive, got {seq_len}")
    manifest = validate_dataset_manifest(path, tokenizer_id=tokenizer_id, verify_checksums=verify_checksums)
    raw_manifest = _read_manifest_json(manifest.manifest_path)
    return DataInspection(manifest=manifest, raw_manifest=raw_manifest, seq_len=seq_len)


def data_inspection_to_dict(inspection: DataInspection) -> dict[str, Any]:
    """Convert a data inspection to stable JSON-compatible data."""

    manifest = inspection.manifest
    raw = inspection.raw_manifest
    source = raw.get("source")
    tokenizer = raw.get("tokenizer")
    shard_bytes = sum(shard.bytes for shard in manifest.shards)
    records = None
    if inspection.seq_len is not None:
        records = {
            "seq_len": inspection.seq_len,
            "train": _record_count(manifest.train_tokens, inspection.seq_len),
            "val": _record_count(manifest.val_tokens, inspection.seq_len),
        }
    return {
        "manifest": prepared_dataset_manifest_to_dict(manifest),
        "source": source if isinstance(source, dict) else None,
        "tokenizer": tokenizer if isinstance(tokenizer, dict) else {"name": manifest.tokenizer_id},
        "tokens": {
            "total": manifest.num_tokens,
            "train": manifest.train_tokens,
            "val": manifest.val_tokens,
        },
        "documents": {
            "count": None if manifest.documents is None else manifest.documents.count,
            "offsets_path": None if manifest.documents is None else manifest.documents.path.as_posix(),
            "offsets_sha256": None if manifest.documents is None else manifest.documents.sha256,
        },
        "shards": {
            "count": manifest.shard_count,
            "bytes": shard_bytes,
            "checksums_present": all(shard.sha256 is not None for shard in manifest.shards),
        },
        "records": records,
        "data_config_toml": data_config_snippet(manifest),
    }


def data_inspection_to_json(inspection: DataInspection) -> str:
    """Serialize a data inspection as canonical JSON."""

    return json.dumps(data_inspection_to_dict(inspection), sort_keys=True, separators=(",", ":"))


def format_data_inspection(inspection: DataInspection) -> str:
    """Format a prepared-token manifest summary for humans."""

    payload = data_inspection_to_dict(inspection)
    source = payload["source"] or {}
    tokenizer = payload["tokenizer"]
    records = payload["records"]
    lines = [
        f"manifest: {inspection.manifest.manifest_path.as_posix()}",
        f"source: {_format_source(source)}",
        f"tokenizer: {tokenizer.get('name')} append_eot={tokenizer.get('append_eot')} "
        f"eot={tokenizer.get('eot_token')}",
        f"tokens: total={_format_int(inspection.manifest.num_tokens)} "
        f"train={_format_int(inspection.manifest.train_tokens)} val={_format_int(inspection.manifest.val_tokens)}",
        f"documents: count={payload['documents']['count']} offsets={payload['documents']['offsets_path']}",
        f"shards: count={payload['shards']['count']} bytes={_format_int(payload['shards']['bytes'])} "
        f"checksums={payload['shards']['checksums_present']}",
    ]
    if records is not None:
        lines.append(
            f"records: seq_len={records['seq_len']} train={_format_int(records['train'])} "
            f"val={_format_int(records['val'])}"
        )
    lines.extend(["", "training config:", payload["data_config_toml"]])
    return "\n".join(lines)


def data_config_snippet(manifest: PreparedDatasetManifest) -> str:
    """Return a paste-ready TOML [data] block for this prepared manifest."""

    lines = [
        "[data]",
        f'train_manifest = "{manifest.manifest_path.as_posix()}"',
        f'tokenizer_id = "{manifest.tokenizer_id}"',
    ]
    if manifest.documents is None:
        lines.append('order = "sequential"')
    else:
        lines.extend(
            [
                'order = "document_buffer"',
                "shuffle_seed = 123",
                "document_buffer_size = 8",
                "document_refill_size = 8",
            ]
        )
    return "\n".join(lines)


def _record_count(tokens: int, seq_len: int) -> int:
    return max(0, (tokens - 1) // seq_len)


def _read_manifest_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContractError(f"failed to read prepared dataset manifest {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"failed to parse prepared dataset manifest {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ContractError(f"prepared dataset manifest must be a JSON object: {path}")
    return raw


def _format_int(value: int) -> str:
    return f"{value:,}"


def _format_source(source: dict[str, Any]) -> str:
    if source.get("type") == "hf":
        return (
            f"type=hf dataset={source.get('dataset')} name={source.get('name')} "
            f"split={source.get('split')} text_column={source.get('text_column')}"
        )
    if source:
        return (
            f"type={source.get('type')} files={source.get('resolved_file_count')} "
            f"bytes={source.get('resolved_total_bytes')} text_column={source.get('text_column')}"
        )
    return "type=None dataset=None name=None split=None text_column=None"
