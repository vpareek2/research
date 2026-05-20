"""Prepared-token dataset manifest validation."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from jaxtitan.errors import ContractError

UINT32_BYTES = 4
UINT64_BYTES = 8


@dataclass(frozen=True, slots=True)
class TokenShard:
    """Validated prepared-token shard metadata."""

    path: Path
    start: int
    end: int
    tokens: int
    bytes: int
    sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True, slots=True)
class TokenSplit:
    """Validated token split interval."""

    start: int
    end: int
    tokens: int


@dataclass(frozen=True, slots=True)
class DocumentTable:
    """Validated prepared-token document offset metadata."""

    path: Path
    dtype: str
    count: int
    bytes: int
    sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True, slots=True)
class PreparedDatasetManifest:
    """Validated prepared training-token manifest."""

    manifest_path: Path
    schema_version: int
    kind: str
    dtype: str
    tokenizer_id: str
    num_tokens: int
    train: TokenSplit
    val: TokenSplit
    shards: tuple[TokenShard, ...]
    token_bytes_path: Path
    token_bytes_sha256: str | None
    documents: DocumentTable | None
    manifest_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(self, "shards", tuple(self.shards))
        object.__setattr__(self, "token_bytes_path", Path(self.token_bytes_path))

    @property
    def shard_count(self) -> int:
        return len(self.shards)

    @property
    def train_tokens(self) -> int:
        return self.train.tokens

    @property
    def val_tokens(self) -> int:
        return self.val.tokens


def load_dataset_manifest(path: str | Path) -> PreparedDatasetManifest:
    """Load and validate a prepared training-token manifest."""

    return validate_dataset_manifest(path)


def validate_dataset_manifest(
    path: str | Path,
    *,
    tokenizer_id: str | None = None,
    verify_checksums: bool = False,
) -> PreparedDatasetManifest:
    """Validate a prepared training-token manifest and its local files."""

    manifest_path = Path(path)
    raw = _load_manifest_json(manifest_path)
    data_dir = manifest_path.parent

    schema_version = _required_int(raw, "schema_version", "manifest")
    _expect(schema_version > 0, f"prepared dataset manifest schema_version must be positive, got {schema_version}")
    _expect(raw.get("kind") == "training_tokens", f"prepared dataset manifest kind must be training_tokens, got {raw.get('kind')!r}")
    _expect(raw.get("dtype") == "uint32", f"prepared dataset manifest dtype must be uint32, got {raw.get('dtype')!r}")

    manifest_tokenizer = _mapping(raw.get("tokenizer"), "tokenizer").get("name")
    _expect(isinstance(manifest_tokenizer, str) and manifest_tokenizer, "prepared dataset manifest tokenizer.name is required")
    if tokenizer_id is not None:
        _expect(
            manifest_tokenizer == tokenizer_id,
            f"prepared dataset tokenizer {manifest_tokenizer!r} does not match config tokenizer {tokenizer_id!r}",
        )

    raw_shards = raw.get("shards")
    _expect(isinstance(raw_shards, list) and bool(raw_shards), "prepared dataset manifest must contain non-empty shards")
    shards = _validate_shards(data_dir, raw_shards, verify_checksums=verify_checksums)

    num_tokens = _required_int(raw, "num_tokens", "manifest")
    shard_tokens = sum(shard.tokens for shard in shards)
    _expect(num_tokens == shard_tokens, f"prepared dataset num_tokens={num_tokens} does not match shard total={shard_tokens}")

    splits = _mapping(raw.get("splits"), "splits")
    train = _validate_split(splits, "train", num_tokens)
    val = _validate_split(splits, "val", num_tokens)
    _expect(
        not _intervals_overlap(train, val),
        f"prepared dataset train/val splits overlap: train={train}, val={val}",
    )

    files = _mapping(raw.get("files"), "files")
    token_bytes_info = _mapping(files.get("token_bytes"), "files.token_bytes")
    token_bytes_rel = _required_str(token_bytes_info, "path", "files.token_bytes")
    token_bytes_path = _resolve_relative_file(data_dir, token_bytes_rel, "token_bytes")
    token_bytes_sha = token_bytes_info.get("sha256")
    if token_bytes_sha is not None:
        _expect(isinstance(token_bytes_sha, str) and bool(token_bytes_sha), "files.token_bytes.sha256 must be non-empty")
    if verify_checksums and token_bytes_sha:
        actual_sha = _sha256(token_bytes_path)
        _expect(actual_sha == token_bytes_sha, f"prepared dataset token_bytes checksum mismatch for {token_bytes_path}")

    documents = _validate_documents(
        data_dir,
        raw,
        files,
        num_tokens=num_tokens,
        verify_checksums=verify_checksums,
    )

    return PreparedDatasetManifest(
        manifest_path=manifest_path,
        schema_version=schema_version,
        kind="training_tokens",
        dtype="uint32",
        tokenizer_id=manifest_tokenizer,
        num_tokens=num_tokens,
        train=train,
        val=val,
        shards=tuple(shards),
        token_bytes_path=Path(token_bytes_rel),
        token_bytes_sha256=token_bytes_sha,
        documents=documents,
        manifest_sha256=dataset_manifest_sha256(manifest_path),
    )


def dataset_manifest_sha256(path: str | Path) -> str:
    """Return the SHA256 hash of prepared dataset manifest bytes."""

    return _sha256(Path(path))


def dataset_manifest_summary(manifest: PreparedDatasetManifest) -> dict[str, Any]:
    """Return the run-manifest data section for a prepared dataset."""

    return {
        "manifest_path": manifest.manifest_path.as_posix(),
        "manifest_sha256": manifest.manifest_sha256,
        "tokenizer_id": manifest.tokenizer_id,
        "total_tokens": manifest.num_tokens,
        "train_tokens": manifest.train_tokens,
        "val_tokens": manifest.val_tokens,
        "shard_count": manifest.shard_count,
        "token_bytes_path": manifest.token_bytes_path.as_posix(),
        "document_aware": manifest.documents is not None,
        "document_count": None if manifest.documents is None else manifest.documents.count,
        "document_offsets_path": None if manifest.documents is None else manifest.documents.path.as_posix(),
        "document_offsets_sha256": None if manifest.documents is None else manifest.documents.sha256,
    }


def prepared_dataset_manifest_to_dict(manifest: PreparedDatasetManifest) -> dict[str, Any]:
    """Convert a prepared dataset manifest to JSON-compatible data."""

    return _normalize(asdict(manifest))


def prepared_dataset_manifest_to_json(manifest: PreparedDatasetManifest) -> str:
    """Serialize a prepared dataset manifest as canonical JSON."""

    return json.dumps(prepared_dataset_manifest_to_dict(manifest), sort_keys=True, separators=(",", ":"))


def _load_manifest_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ContractError(f"prepared dataset manifest does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except OSError as exc:
        raise ContractError(f"failed to read prepared dataset manifest {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"failed to parse prepared dataset manifest {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ContractError(f"prepared dataset manifest must be a JSON object: {path}")
    return raw


def _validate_shards(data_dir: Path, raw_shards: list[Any], *, verify_checksums: bool) -> list[TokenShard]:
    shards = []
    expected_start = 0
    for idx, raw_shard in enumerate(raw_shards):
        shard = _mapping(raw_shard, f"shards[{idx}]")
        path_value = _required_str(shard, "path", f"shards[{idx}]")
        shard_path = _resolve_relative_file(data_dir, path_value, f"shards[{idx}]")
        start = _required_int(shard, "start", f"shards[{idx}]")
        end = _required_int(shard, "end", f"shards[{idx}]")
        tokens = _required_int(shard, "tokens", f"shards[{idx}]")
        byte_count = _required_int(shard, "bytes", f"shards[{idx}]")

        if start != expected_start or end < start or tokens != end - start:
            raise ContractError(f"prepared dataset shards must be contiguous and correctly bounded; bad shard={idx}")

        actual_bytes = shard_path.stat().st_size
        if actual_bytes % UINT32_BYTES != 0:
            raise ContractError(f"prepared dataset shard size is not divisible by uint32 size: {shard_path}")
        actual_tokens = actual_bytes // UINT32_BYTES
        if actual_tokens != tokens:
            raise ContractError(
                f"prepared dataset shard {path_value} tokens={tokens} does not match file length={actual_tokens}"
            )
        if byte_count != actual_bytes:
            raise ContractError(
                f"prepared dataset shard {path_value} bytes={byte_count} does not match file bytes={actual_bytes}"
            )

        expected_sha = shard.get("sha256")
        if expected_sha is not None:
            _expect(isinstance(expected_sha, str) and bool(expected_sha), f"shards[{idx}].sha256 must be non-empty")
        if verify_checksums and expected_sha:
            actual_sha = _sha256(shard_path)
            _expect(actual_sha == expected_sha, f"prepared dataset checksum mismatch for {shard_path}")

        shards.append(
            TokenShard(
                path=Path(path_value),
                start=start,
                end=end,
                tokens=tokens,
                bytes=byte_count,
                sha256=expected_sha,
            )
        )
        expected_start = end
    return shards


def _validate_split(splits: Mapping[str, Any], name: str, num_tokens: int) -> TokenSplit:
    split = _mapping(splits.get(name), f"splits.{name}")
    start = _required_int(split, "start", f"splits.{name}")
    end = _required_int(split, "end", f"splits.{name}")
    tokens = _required_int(split, "tokens", f"splits.{name}")
    if not 0 <= start <= end <= num_tokens:
        raise ContractError(
            f"prepared dataset split {name} has invalid bounds start={start}, end={end}, num_tokens={num_tokens}"
        )
    if tokens != end - start:
        raise ContractError(f"prepared dataset split {name} tokens={tokens} does not equal end-start={end - start}")
    return TokenSplit(start=start, end=end, tokens=tokens)


def _validate_documents(
    data_dir: Path,
    raw: Mapping[str, Any],
    files: Mapping[str, Any],
    *,
    num_tokens: int,
    verify_checksums: bool,
) -> DocumentTable | None:
    has_documents = raw.get("documents") is not None
    has_offsets = files.get("document_offsets") is not None
    if not has_documents and not has_offsets:
        return None
    _expect(has_documents == has_offsets, "prepared dataset documents metadata requires files.document_offsets")

    documents = _mapping(raw.get("documents"), "documents")
    count = _required_int(documents, "count", "documents")
    _expect(count > 0, f"prepared dataset documents.count must be positive, got {count}")

    offset_info = _mapping(files.get("document_offsets"), "files.document_offsets")
    dtype = _required_str(offset_info, "dtype", "files.document_offsets")
    _expect(dtype == "uint64", f"prepared dataset document_offsets dtype must be uint64, got {dtype!r}")
    path_value = _required_str(offset_info, "path", "files.document_offsets")
    offsets_path = _resolve_relative_file(data_dir, path_value, "document_offsets")
    declared_bytes = _required_int(offset_info, "bytes", "files.document_offsets")
    expected_bytes = (count + 1) * UINT64_BYTES
    actual_bytes = offsets_path.stat().st_size
    _expect(
        declared_bytes == actual_bytes,
        f"prepared dataset document_offsets bytes={declared_bytes} does not match file bytes={actual_bytes}",
    )
    _expect(
        actual_bytes == expected_bytes,
        f"prepared dataset document_offsets file bytes={actual_bytes} does not match documents.count={count}",
    )

    offsets = np.memmap(offsets_path, dtype="<u8", mode="r")
    _expect(int(offsets[0]) == 0, "prepared dataset document_offsets first offset must be 0")
    _expect(
        int(offsets[-1]) == num_tokens,
        f"prepared dataset document_offsets final offset={int(offsets[-1])} must equal num_tokens={num_tokens}",
    )
    _expect(
        bool(np.all(offsets[1:] > offsets[:-1])),
        "prepared dataset document_offsets must be strictly increasing",
    )

    expected_sha = offset_info.get("sha256")
    if expected_sha is not None:
        _expect(isinstance(expected_sha, str) and bool(expected_sha), "files.document_offsets.sha256 must be non-empty")
    if verify_checksums and expected_sha:
        actual_sha = _sha256(offsets_path)
        _expect(actual_sha == expected_sha, f"prepared dataset document_offsets checksum mismatch for {offsets_path}")

    return DocumentTable(
        path=Path(path_value),
        dtype="uint64",
        count=count,
        bytes=declared_bytes,
        sha256=expected_sha,
    )


def _resolve_relative_file(data_dir: Path, path_value: str, label: str) -> Path:
    rel = Path(path_value)
    if rel.is_absolute():
        raise ContractError(f"prepared dataset {label} path must be relative, got {path_value!r}")
    full_path = data_dir / rel
    if not full_path.resolve().is_relative_to(data_dir.resolve()):
        raise ContractError(f"prepared dataset {label} path must stay inside the manifest directory, got {path_value!r}")
    if not full_path.is_file():
        raise ContractError(f"prepared dataset {label} file does not exist: {full_path}")
    return full_path


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"prepared dataset manifest {label} must be an object")
    return value


def _required_str(raw: Mapping[str, Any], key: str, label: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ContractError(f"prepared dataset manifest {label}.{key} must be a non-empty string")
    return value


def _required_int(raw: Mapping[str, Any], key: str, label: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ContractError(f"prepared dataset manifest {label}.{key} must be an integer")
    return value


def _intervals_overlap(left: TokenSplit, right: TokenSplit) -> bool:
    return left.start < right.end and right.start < left.end


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


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
