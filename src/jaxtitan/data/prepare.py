"""Offline preparation of token datasets for Jaxtitan training."""

from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
import glob
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any
import tomllib

import numpy as np
import tiktoken
from tqdm.auto import tqdm

from jaxtitan.data.inspect import data_config_snippet
from jaxtitan.data.manifest import PreparedDatasetManifest, prepared_dataset_manifest_to_dict, validate_dataset_manifest
from jaxtitan.errors import ConfigError, ContractError

TOKEN_BYTES_FILENAME = "token_bytes.bin"
DOCUMENT_OFFSETS_FILENAME = "document_offsets.u64"
SOURCE_TYPES = frozenset({"hf", "parquet", "jsonl", "text"})


@dataclass(frozen=True, slots=True)
class PrepareSourceConfig:
    """Raw dataset source for offline preparation."""

    type: str
    dataset: str | None = None
    split: str | None = None
    paths: tuple[str, ...] = ()
    name: str | None = None
    data_dir: str | None = None
    text_column: str = "text"
    streaming: bool = True

    def __post_init__(self) -> None:
        if self.type not in SOURCE_TYPES:
            supported = "', '".join(sorted(SOURCE_TYPES))
            raise ConfigError(f"source.type must be one of '{supported}', got {self.type!r}")
        object.__setattr__(self, "paths", tuple(self.paths))
        if self.type == "hf" and not self.dataset:
            raise ConfigError("source.dataset must be a non-empty string")
        if self.type == "hf" and not self.split:
            raise ConfigError("source.split must be a non-empty string")
        if self.type != "hf" and not self.paths:
            raise ConfigError("source.paths must be a non-empty list of paths")
        if any(not isinstance(path, str) or not path for path in self.paths):
            raise ConfigError("source.paths must contain non-empty strings")
        if not self.text_column:
            raise ConfigError("source.text_column must be a non-empty string")
        if not isinstance(self.streaming, bool):
            raise ConfigError("source.streaming must be a boolean")


@dataclass(frozen=True, slots=True)
class PrepareTokenizerConfig:
    """Tokenizer settings for prepared-token output."""

    name: str
    append_eot: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("tokenizer.name must be a non-empty string")
        if not isinstance(self.append_eot, bool):
            raise ConfigError("tokenizer.append_eot must be a boolean")


@dataclass(frozen=True, slots=True)
class PrepareOutputConfig:
    """Output layout for prepared-token artifacts."""

    path: Path
    max_tokens: int
    val_fraction: float = 0.001
    shard_tokens: int = 128_000_000
    dtype: str = "uint32"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.dtype != "uint32":
            raise ConfigError(f"output.dtype must be 'uint32', got {self.dtype!r}")
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool) or self.max_tokens <= 0:
            raise ConfigError("output.max_tokens must be a positive integer")
        if not isinstance(self.shard_tokens, int) or isinstance(self.shard_tokens, bool) or self.shard_tokens <= 0:
            raise ConfigError("output.shard_tokens must be a positive integer")
        if not isinstance(self.val_fraction, int | float) or isinstance(self.val_fraction, bool):
            raise ConfigError("output.val_fraction must be numeric")
        if not 0.0 < float(self.val_fraction) < 1.0:
            raise ConfigError(f"output.val_fraction must be between 0 and 1, got {self.val_fraction}")


@dataclass(frozen=True, slots=True)
class PrepareTokenizationConfig:
    """Tokenization worker policy."""

    workers: int | str = "auto"
    batch_docs: int = 256
    queue_batches: int = 8

    def __post_init__(self) -> None:
        if self.workers != "auto" and (
            not isinstance(self.workers, int) or isinstance(self.workers, bool) or self.workers <= 0
        ):
            raise ConfigError("tokenization.workers must be 'auto' or a positive integer")
        if not isinstance(self.batch_docs, int) or isinstance(self.batch_docs, bool) or self.batch_docs <= 0:
            raise ConfigError("tokenization.batch_docs must be a positive integer")
        if not isinstance(self.queue_batches, int) or isinstance(self.queue_batches, bool) or self.queue_batches <= 0:
            raise ConfigError("tokenization.queue_batches must be a positive integer")


@dataclass(frozen=True, slots=True)
class PrepareConfig:
    """Resolved offline data preparation config."""

    source: PrepareSourceConfig
    tokenizer: PrepareTokenizerConfig
    output: PrepareOutputConfig
    tokenization: PrepareTokenizationConfig = PrepareTokenizationConfig()


@dataclass(frozen=True, slots=True)
class PrepareResult:
    """Completed prepare operation result."""

    manifest: PreparedDatasetManifest
    raw_manifest: Mapping[str, Any]


def load_prepare_config(path: str | Path) -> PrepareConfig:
    """Load a Jaxtitan data-prepare TOML config."""

    config_path = Path(path)
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"failed to read data prepare config {config_path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"failed to parse data prepare config {config_path}: {exc}") from exc

    return prepare_config_from_mapping(raw)


def prepare_config_from_mapping(raw: Mapping[str, Any]) -> PrepareConfig:
    """Resolve a TOML-like mapping into a data prepare config."""

    source = _source_section(_required_mapping(raw, "source"))
    tokenizer = _tokenizer_section(_required_mapping(raw, "tokenizer"))
    output = _output_section(_required_mapping(raw, "output"))
    tokenization = _tokenization_section(_optional_mapping(raw, "tokenization"))
    return PrepareConfig(source=source, tokenizer=tokenizer, output=output, tokenization=tokenization)


def prepare_dataset(config_or_path: PrepareConfig | str | Path, *, overwrite: bool = False, quiet: bool = False) -> PrepareResult:
    """Prepare a text dataset into Jaxtitan token artifacts."""

    config = load_prepare_config(config_or_path) if isinstance(config_or_path, str | Path) else config_or_path
    tokenizer = _load_tokenizer(config.tokenizer.name)
    output_dir = config.output.path
    _prepare_output_dir(output_dir, overwrite=overwrite)

    started = time.perf_counter()
    texts, source_manifest = load_source_texts(config.source)
    token_writer = _ShardWriter(output_dir, shard_tokens=config.output.shard_tokens)
    document_offsets = [0]
    docs_seen = 0
    docs_written = 0

    progress = tqdm(
        total=config.output.max_tokens,
        desc="Tokenizing tokens",
        unit="tok",
        disable=quiet,
    )
    try:
        for tokenized_docs in _tokenized_batches(
            texts,
            tokenizer_name=config.tokenizer.name,
            append_eot=config.tokenizer.append_eot,
            tokenization=config.tokenization,
        ):
            for tokens in tokenized_docs:
                docs_seen += 1
                remaining = config.output.max_tokens - token_writer.total_tokens
                if remaining <= 0:
                    break
                if not tokens:
                    continue
                tokens = tokens[:remaining]
                token_writer.write(tokens)
                docs_written += 1
                document_offsets.append(token_writer.total_tokens)
                progress.update(len(tokens))
            if token_writer.total_tokens >= config.output.max_tokens:
                break
    finally:
        progress.close()
        token_writer.close()

    if token_writer.total_tokens <= 0:
        raise ContractError("data prepare wrote zero tokens")
    if docs_written <= 0 or len(document_offsets) < 2:
        raise ContractError("data prepare wrote zero documents")

    token_bytes_path = output_dir / TOKEN_BYTES_FILENAME
    document_offsets_path = output_dir / DOCUMENT_OFFSETS_FILENAME
    build_token_bytes(tokenizer).tofile(token_bytes_path)
    np.asarray(document_offsets, dtype="<u8").tofile(document_offsets_path)

    elapsed = time.perf_counter() - started
    train_end = int(token_writer.total_tokens * (1.0 - float(config.output.val_fraction)))
    raw_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "kind": "training_tokens",
        "source": source_manifest,
        "tokenizer": {
            "name": config.tokenizer.name,
            "append_eot": config.tokenizer.append_eot,
            "eot_token": tokenizer.eot_token,
        },
        "dtype": "uint32",
        "val_fraction": float(config.output.val_fraction),
        "max_tokens": config.output.max_tokens,
        "shard_tokens": config.output.shard_tokens,
        "num_tokens": token_writer.total_tokens,
        "num_docs": docs_written,
        "source_docs_seen": docs_seen,
        "elapsed_sec": elapsed,
        "tokens_per_sec": token_writer.total_tokens / elapsed if elapsed > 0.0 else None,
        "tokenization": {
            "workers": _resolve_workers(config.tokenization.workers),
            "batch_docs": config.tokenization.batch_docs,
            "queue_batches": config.tokenization.queue_batches,
        },
        "splits": {
            "train": {"start": 0, "end": train_end, "tokens": train_end},
            "val": {
                "start": train_end,
                "end": token_writer.total_tokens,
                "tokens": token_writer.total_tokens - train_end,
            },
        },
        "train_tokens": train_end,
        "val_tokens": token_writer.total_tokens - train_end,
        "documents": {"count": docs_written},
        "shards": token_writer.shards,
        "files": {
            "token_bytes": {
                "path": TOKEN_BYTES_FILENAME,
                "sha256": _sha256(token_bytes_path),
                "bytes": token_bytes_path.stat().st_size,
                "dtype": "uint16",
            },
            "document_offsets": {
                "path": DOCUMENT_OFFSETS_FILENAME,
                "sha256": _sha256(document_offsets_path),
                "bytes": document_offsets_path.stat().st_size,
                "dtype": "uint64",
            },
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(raw_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = validate_dataset_manifest(manifest_path, tokenizer_id=config.tokenizer.name)
    return PrepareResult(manifest=manifest, raw_manifest=raw_manifest)


def load_hf_texts(config: PrepareSourceConfig) -> Iterable[str]:
    """Load a Hugging Face dataset as a stream of text documents."""

    _configure_hf_imports()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ContractError("Hugging Face dataset preparation requires the 'datasets' package") from exc

    kwargs: dict[str, Any] = {
        "split": config.split,
        "streaming": config.streaming,
    }
    if config.data_dir is not None:
        kwargs["data_dir"] = config.data_dir
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        kwargs["token"] = token

    try:
        if config.name is None:
            dataset = load_dataset(config.dataset, **kwargs)
        else:
            dataset = load_dataset(config.dataset, config.name, **kwargs)
    except Exception as exc:
        raise ContractError(f"failed to load Hugging Face dataset {config.dataset!r}: {exc}") from exc

    return _TextColumnIterable(dataset, config.text_column)


def load_source_texts(config: PrepareSourceConfig) -> tuple[Iterable[str], dict[str, Any]]:
    """Load a configured source as text documents plus source provenance."""

    if config.type == "hf":
        return load_hf_texts(config), _source_manifest(config)
    files = _resolve_source_files(config.paths)
    return _local_texts(config, files), _source_manifest(config, files)


def _local_texts(config: PrepareSourceConfig, files: Sequence[Path]) -> Iterable[str]:
    if config.type == "parquet":
        return _parquet_texts(files, config.text_column)
    if config.type == "jsonl":
        return _jsonl_texts(files, config.text_column)
    if config.type == "text":
        return _plain_texts(files)
    raise ConfigError(f"unsupported source.type {config.type!r}")


def _parquet_texts(files: Sequence[Path], text_column: str) -> Iterator[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ContractError("parquet data preparation requires the 'pyarrow' package") from exc

    for path in files:
        try:
            parquet_file = pq.ParquetFile(path)
            if text_column not in parquet_file.schema_arrow.names:
                raise ContractError(f"parquet source {path} is missing text column {text_column!r}")
            for batch in parquet_file.iter_batches(columns=[text_column]):
                for value in batch.column(0).to_pylist():
                    yield _ensure_text_value(value, source=path, text_column=text_column)
        except ContractError:
            raise
        except Exception as exc:
            raise ContractError(f"failed to read parquet source {path}: {exc}") from exc


def _jsonl_texts(files: Sequence[Path], text_column: str) -> Iterator[str]:
    for path in files:
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ContractError(f"failed to parse JSONL source {path}:{line_number}: {exc}") from exc
                    if not isinstance(item, Mapping):
                        raise ContractError(f"JSONL source {path}:{line_number} must contain JSON objects")
                    if text_column not in item:
                        raise ContractError(f"JSONL source {path}:{line_number} is missing text column {text_column!r}")
                    yield _ensure_text_value(item[text_column], source=path, text_column=text_column, line=line_number)
        except OSError as exc:
            raise ContractError(f"failed to read JSONL source {path}: {exc}") from exc


def _plain_texts(files: Sequence[Path]) -> Iterator[str]:
    for path in files:
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    yield line.rstrip("\r\n")
        except OSError as exc:
            raise ContractError(f"failed to read text source {path}: {exc}") from exc


def _ensure_text_value(value: Any, *, source: Path, text_column: str, line: int | None = None) -> str:
    if isinstance(value, str):
        return value
    location = f"{source}:{line}" if line is not None else str(source)
    raise ContractError(f"text column {text_column!r} in {location} must contain strings")


def _configure_hf_imports() -> None:
    # Jaxtitan only needs dataset rows here; disabling Torch avoids importing
    # incompatible local CUDA Torch builds during text preparation.
    os.environ.setdefault("USE_TORCH", "0")
    _patch_multiprocess_resource_tracker()


def _patch_multiprocess_resource_tracker() -> None:
    try:
        from multiprocess import resource_tracker
    except Exception:
        return
    current = resource_tracker.ResourceTracker.__del__
    if getattr(current, "__name__", "") == "_jaxtitan_resource_tracker_del":
        return

    def _jaxtitan_resource_tracker_del(self) -> None:
        try:
            self._stop(use_blocking_lock=False)
        except AttributeError as exc:
            if "_recursion_count" not in str(exc):
                raise

    resource_tracker.ResourceTracker.__del__ = _jaxtitan_resource_tracker_del


def build_token_bytes(tokenizer: tiktoken.Encoding) -> np.ndarray:
    """Build a per-token byte-length table for diagnostics and throughput accounting."""

    token_bytes = np.zeros(tokenizer.n_vocab, dtype=np.uint16)
    for token_id in range(tokenizer.n_vocab):
        try:
            token_bytes[token_id] = len(tokenizer.decode_single_token_bytes(token_id))
        except KeyError:
            token_bytes[token_id] = 0
    token_bytes[tokenizer.eot_token] = 0
    return token_bytes


def prepare_result_to_dict(result: PrepareResult) -> dict[str, Any]:
    """Return a stable machine-readable prepare summary."""

    raw = result.raw_manifest
    return {
        "manifest": prepared_dataset_manifest_to_dict(result.manifest),
        "source": dict(raw["source"]),
        "tokenization": dict(raw["tokenization"]),
        "documents": {"count": raw["documents"]["count"]},
        "elapsed_sec": raw["elapsed_sec"],
        "tokens_per_sec": raw["tokens_per_sec"],
    }


def prepare_result_to_json(result: PrepareResult) -> str:
    """Serialize a prepare result as canonical JSON."""

    return json.dumps(prepare_result_to_dict(result), sort_keys=True, separators=(",", ":"))


def format_prepare_result(result: PrepareResult) -> str:
    """Format a completed prepare result for humans."""

    manifest = result.manifest
    raw = result.raw_manifest
    source = raw["source"]
    check_command = (
        f"uv run jaxtitan data check {manifest.manifest_path.as_posix()} "
        f"--tokenizer {manifest.tokenizer_id} --verify-checksums"
    )
    return (
        f"source: {_format_source_summary(source)}\n"
        f"output: {manifest.manifest_path.parent.as_posix()}\n"
        f"tokens: total={manifest.num_tokens:,} train={manifest.train_tokens:,} val={manifest.val_tokens:,}\n"
        f"documents: {0 if manifest.documents is None else manifest.documents.count:,}\n"
        f"shards: {manifest.shard_count}\n"
        f"tokens_per_sec: {_format_float(raw['tokens_per_sec'])}\n"
        f"manifest: {manifest.manifest_path.as_posix()}\n"
        "\n"
        f"check: {check_command}\n"
        "\n"
        "training config:\n"
        f"{data_config_snippet(manifest)}"
    )


class _TextColumnIterable:
    def __init__(self, dataset: Any, text_column: str) -> None:
        self.dataset = dataset
        self.text_column = text_column

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[str]:
        for item in self.dataset:
            yield self._text_from_item(item)

    def _text_from_item(self, item: Any) -> str:
        if not isinstance(item, Mapping):
            raise ContractError(f"HF dataset rows must be mappings, got {type(item).__name__}")
        if self.text_column not in item:
            raise ContractError(f"HF dataset row is missing text column {self.text_column!r}")
        value = item[self.text_column]
        if not isinstance(value, str):
            raise ContractError(f"HF dataset text column {self.text_column!r} must contain strings")
        return value


class _ShardWriter:
    def __init__(self, output_dir: Path, *, shard_tokens: int) -> None:
        self.output_dir = output_dir
        self.shard_tokens = shard_tokens
        self.shards: list[dict[str, Any]] = []
        self.total_tokens = 0
        self._current_path: Path | None = None
        self._current_file: Any | None = None
        self._current_hash: Any | None = None
        self._current_start = 0
        self._current_tokens = 0

    def write(self, tokens: Sequence[int]) -> None:
        offset = 0
        while offset < len(tokens):
            self._ensure_open()
            remaining = self.shard_tokens - self._current_tokens
            take = min(remaining, len(tokens) - offset)
            data = np.asarray(tokens[offset : offset + take], dtype=np.uint32).tobytes()
            self._current_file.write(data)
            self._current_hash.update(data)
            self._current_tokens += take
            self.total_tokens += take
            offset += take
            if self._current_tokens >= self.shard_tokens:
                self.close_current()

    def close(self) -> None:
        self.close_current()

    def close_current(self) -> None:
        if self._current_file is None:
            return
        self._current_file.close()
        path = self._current_path
        tokens = self._current_tokens
        if tokens > 0:
            self.shards.append(
                {
                    "path": path.name,
                    "start": self._current_start,
                    "end": self._current_start + tokens,
                    "tokens": tokens,
                    "bytes": path.stat().st_size,
                    "sha256": self._current_hash.hexdigest(),
                }
            )
        elif path.exists():
            path.unlink()
        self._current_path = None
        self._current_file = None
        self._current_hash = None
        self._current_tokens = 0

    def _ensure_open(self) -> None:
        if self._current_file is not None:
            return
        shard_idx = len(self.shards)
        self._current_path = self.output_dir / f"tokens-{shard_idx:05d}.bin"
        self._current_file = self._current_path.open("wb")
        self._current_hash = sha256()
        self._current_start = self.total_tokens
        self._current_tokens = 0


_WORKER_TOKENIZER: tiktoken.Encoding | None = None
_WORKER_APPEND_EOT = True


def _init_tokenizer_worker(tokenizer_name: str, append_eot: bool) -> None:
    global _WORKER_TOKENIZER, _WORKER_APPEND_EOT
    _WORKER_TOKENIZER = tiktoken.get_encoding(tokenizer_name)
    _WORKER_APPEND_EOT = append_eot


def _tokenize_batch_worker(texts: list[str]) -> list[list[int]]:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("tokenizer worker was not initialized")
    return [_tokenize(text, _WORKER_TOKENIZER, _WORKER_APPEND_EOT) for text in texts]


def _tokenized_batches(
    texts: Iterable[str],
    *,
    tokenizer_name: str,
    append_eot: bool,
    tokenization: PrepareTokenizationConfig,
) -> Iterator[list[list[int]]]:
    workers = _resolve_workers(tokenization.workers)
    batches = _batched(texts, tokenization.batch_docs)
    if workers == 1:
        tokenizer = _load_tokenizer(tokenizer_name)
        for batch in batches:
            yield [_tokenize(text, tokenizer, append_eot) for text in batch]
        return

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_tokenizer_worker,
        initargs=(tokenizer_name, append_eot),
    ) as pool:
        pending = deque()
        max_pending = max(1, workers * tokenization.queue_batches)

        def submit_next() -> bool:
            try:
                batch = next(batches)
            except StopIteration:
                return False
            pending.append(pool.submit(_tokenize_batch_worker, batch))
            return True

        for _idx in range(max_pending):
            if not submit_next():
                break

        while pending:
            future = pending.popleft()
            yield future.result()
            submit_next()


def _tokenize(text: str, tokenizer: tiktoken.Encoding, append_eot: bool) -> list[int]:
    tokens = tokenizer.encode(text)
    if append_eot:
        tokens.append(tokenizer.eot_token)
    return tokens


def _batched(texts: Iterable[str], batch_docs: int) -> Iterator[list[str]]:
    batch = []
    for text in texts:
        if not isinstance(text, str):
            raise ContractError(f"data prepare text rows must be strings, got {type(text).__name__}")
        batch.append(text)
        if len(batch) >= batch_docs:
            yield batch
            batch = []
    if batch:
        yield batch


def _load_tokenizer(name: str) -> tiktoken.Encoding:
    try:
        return tiktoken.get_encoding(name)
    except Exception as exc:
        raise ConfigError(f"unsupported tiktoken tokenizer {name!r}: {exc}") from exc


def _resolve_workers(workers: int | str) -> int:
    if workers == "auto":
        return max(1, (os.cpu_count() or 2) - 2)
    return int(workers)


def _prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise ContractError(f"data prepare output already exists: {path}; rerun with --overwrite to replace it")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True)


def _source_manifest(source: PrepareSourceConfig, files: Sequence[Path] | None = None) -> dict[str, Any]:
    if source.type == "hf":
        return {
            "type": source.type,
            "dataset": source.dataset,
            "name": source.name,
            "data_dir": source.data_dir,
            "split": source.split,
            "text_column": source.text_column,
            "streaming": source.streaming,
        }
    if files is None:
        raise ContractError(f"local source {source.type!r} requires resolved files")
    return {
        "type": source.type,
        "paths": list(source.paths),
        "text_column": None if source.type == "text" else source.text_column,
        "resolved_file_count": len(files),
        "resolved_total_bytes": sum(path.stat().st_size for path in files),
    }


def _resolve_source_files(paths: Sequence[str]) -> tuple[Path, ...]:
    resolved: list[Path] = []
    for raw_path in paths:
        expanded = os.path.expanduser(raw_path)
        matches = sorted(glob.glob(expanded)) if glob.has_magic(expanded) else [expanded]
        if not matches:
            raise ContractError(f"source.paths entry did not match any files: {raw_path!r}")
        for match in matches:
            path = Path(match).expanduser()
            if not path.exists():
                raise ContractError(f"source path does not exist: {raw_path!r}")
            if path.is_dir():
                raise ContractError(f"source path must be a file, got directory: {path}")
            resolved.append(path.resolve())
    unique = sorted(dict.fromkeys(resolved), key=lambda path: path.as_posix())
    if not unique:
        raise ContractError("source.paths did not match any files")
    return tuple(unique)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_float(value: float | None) -> str:
    if value is None:
        return "null"
    return f"{value:.3f}"


def _format_source_summary(source: Mapping[str, Any]) -> str:
    if source.get("type") == "hf":
        return (
            f"type=hf dataset={source.get('dataset')} name={source.get('name')} "
            f"split={source.get('split')}"
        )
    return (
        f"type={source.get('type')} files={source.get('resolved_file_count')} "
        f"bytes={source.get('resolved_total_bytes')} text_column={source.get('text_column')}"
    )


def _source_section(raw: Mapping[str, Any]) -> PrepareSourceConfig:
    return PrepareSourceConfig(
        type=_required_str(raw, "type", "source"),
        dataset=_optional_str(raw, "dataset", "source"),
        split=_optional_str(raw, "split", "source"),
        paths=_optional_str_sequence(raw, "paths", "source"),
        name=_optional_str(raw, "name", "source"),
        data_dir=_optional_str(raw, "data_dir", "source"),
        text_column=_optional_str(raw, "text_column", "source") or "text",
        streaming=_optional_bool(raw, "streaming", "source", default=True),
    )


def _tokenizer_section(raw: Mapping[str, Any]) -> PrepareTokenizerConfig:
    return PrepareTokenizerConfig(
        name=_required_str(raw, "name", "tokenizer"),
        append_eot=_optional_bool(raw, "append_eot", "tokenizer", default=True),
    )


def _output_section(raw: Mapping[str, Any]) -> PrepareOutputConfig:
    return PrepareOutputConfig(
        path=Path(_required_str(raw, "path", "output")),
        max_tokens=_required_int(raw, "max_tokens", "output"),
        val_fraction=float(raw.get("val_fraction", 0.001)),
        shard_tokens=_optional_int_with_default(raw, "shard_tokens", "output", default=128_000_000),
        dtype=_optional_str(raw, "dtype", "output") or "uint32",
    )


def _tokenization_section(raw: Mapping[str, Any]) -> PrepareTokenizationConfig:
    return PrepareTokenizationConfig(
        workers=raw.get("workers", "auto"),
        batch_docs=_optional_int_with_default(raw, "batch_docs", "tokenization", default=256),
        queue_batches=_optional_int_with_default(raw, "queue_batches", "tokenization", default=8),
    )


def _required_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in raw:
        raise ConfigError(f"missing required [{key}] section")
    return _ensure_mapping(raw[key], key)


def _optional_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return _ensure_mapping(raw.get(key, {}), key)


def _ensure_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"[{name}] must be a TOML table")
    return value


def _required_str(raw: Mapping[str, Any], key: str, section: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{section}.{key} must be a non-empty string")
    return value


def _optional_str(raw: Mapping[str, Any], key: str, section: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{section}.{key} must be a non-empty string")
    return value


def _optional_str_sequence(raw: Mapping[str, Any], key: str, section: str) -> tuple[str, ...]:
    value = raw.get(key, ())
    if value == ():
        return ()
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{section}.{key} must be a non-empty list of strings")
    if any(not isinstance(item, str) or not item for item in value):
        raise ConfigError(f"{section}.{key} must contain non-empty strings")
    return tuple(value)


def _required_int(raw: Mapping[str, Any], key: str, section: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{section}.{key} must be an integer")
    return value


def _optional_int_with_default(raw: Mapping[str, Any], key: str, section: str, *, default: int) -> int:
    value = raw.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{section}.{key} must be an integer")
    return value


def _optional_bool(raw: Mapping[str, Any], key: str, section: str, *, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{section}.{key} must be a boolean")
    return value
