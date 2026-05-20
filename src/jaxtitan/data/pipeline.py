"""Grain-backed prepared-token training data pipeline."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
import json
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Literal, Protocol

import grain.python as grain
import numpy as np

from jaxtitan.batch import Batch
from jaxtitan.data.manifest import PreparedDatasetManifest, validate_dataset_manifest
from jaxtitan.data.tokens import read_token_range
from jaxtitan.errors import ContractError
from jaxtitan.state import DataPipelineState

SplitName = Literal["train", "val"]

DATA_PIPELINE_STATE_SCHEMA_VERSION = 1
DATA_PIPELINE_BACKEND = "grain"
DATA_PIPELINE_ORDER = "sequential"
DATA_PIPELINE_WORKER_COUNT = 0
DATA_PIPELINE_PREFETCH = False
DATA_PIPELINE_NUM_EPOCHS = 1
DATA_PIPELINE_DROP_REMAINDER = True
PREPARED_TOKEN_SOURCE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class BatchProvenance:
    """Host-side provenance for one deterministic token batch."""

    split: str
    epoch: int
    token_start: int
    token_end: int
    examples: int
    target_tokens: int
    row_start_offsets: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PipelineBatch:
    """One pipeline result: model batch, next state, and provenance."""

    batch: Batch
    state: DataPipelineState
    provenance: BatchProvenance


class TrainingDataPipeline(Protocol):
    """Runtime data boundary used by training, eval, and preflight."""

    def initial_state(self) -> DataPipelineState: ...

    def next_batch(self, state: DataPipelineState) -> PipelineBatch: ...

    def state_from_json(self, raw: Mapping[str, Any]) -> DataPipelineState: ...

    def state_to_json(self, state: DataPipelineState) -> dict[str, Any]: ...

    def describe(self) -> dict[str, Any]: ...


class PreparedTokenDataSource(grain.RandomAccessDataSource):
    """Random-access Grain source where one record is one LM example."""

    def __init__(self, *, manifest: PreparedDatasetManifest, split: SplitName, seq_len: int) -> None:
        if seq_len <= 0:
            raise ContractError(f"seq_len must be positive, got {seq_len}")
        if split not in {"train", "val"}:
            raise ContractError(f"split must be 'train' or 'val', got {split!r}")
        token_split = manifest.train if split == "train" else manifest.val
        self.manifest = manifest
        self.split = split
        self.seq_len = int(seq_len)
        self.split_start = int(token_split.start)
        self.split_end = int(token_split.end)
        self.split_tokens = int(token_split.tokens)
        self.num_records = max(0, (self.split_tokens - 1) // self.seq_len)
        self.manifest_sha256 = manifest.manifest_sha256
        self.tokenizer_id = manifest.tokenizer_id
        self.source_summary = _source_summary(self)

    @classmethod
    def from_manifest(
        cls,
        path: str | Path,
        *,
        tokenizer_id: str | None,
        split: SplitName,
        seq_len: int,
    ) -> "PreparedTokenDataSource":
        manifest = validate_dataset_manifest(path, tokenizer_id=tokenizer_id)
        return cls(manifest=manifest, split=split, seq_len=seq_len)

    def __len__(self) -> int:
        return self.num_records

    def __getitem__(self, record_key: int) -> dict[str, Any]:
        if isinstance(record_key, np.integer):
            record_key = int(record_key)
        if not isinstance(record_key, int):
            raise TypeError(f"record_key must be an integer, got {type(record_key).__name__}")
        if record_key < 0 or record_key >= self.num_records:
            raise IndexError(f"record_key {record_key} is outside {self.split} records [0, {self.num_records})")
        token_start = self.split_start + record_key * self.seq_len
        tokens = read_token_range(self.manifest, token_start, token_start + self.seq_len + 1)
        input_ids = np.asarray(tokens[:-1], dtype=np.int32)
        target_ids = np.asarray(tokens[1:], dtype=np.int32)
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "loss_mask": np.ones((self.seq_len,), dtype=np.bool_),
            "token_start": np.asarray(token_start, dtype=np.int64),
            "token_end": np.asarray(token_start + self.seq_len, dtype=np.int64),
            "record_key": np.asarray(record_key, dtype=np.int64),
        }

    def __repr__(self) -> str:
        return (
            "PreparedTokenDataSource("
            f"schema_version={PREPARED_TOKEN_SOURCE_SCHEMA_VERSION}, "
            f"manifest_sha256='{self.manifest_sha256}', "
            f"split='{self.split}', "
            f"seq_len={self.seq_len}, "
            f"tokenizer_id='{self.tokenizer_id}', "
            f"records={self.num_records}"
            ")"
        )


class PreparedTokenGrainPipeline:
    """Canonical Grain-backed pipeline for prepared-token LM batches."""

    def __init__(
        self,
        *,
        source: PreparedTokenDataSource,
        batch_size: int,
    ) -> None:
        if batch_size <= 0:
            raise ContractError(f"batch_size must be positive, got {batch_size}")
        if len(source) < batch_size:
            required = batch_size * source.seq_len + 1
            raise ContractError(
                f"{source.split} split has {source.split_tokens} tokens, but one batch requires at least {required}"
            )
        self.source = source
        self.batch_size = int(batch_size)
        self.split = source.split
        self.seq_len = source.seq_len
        self.split_start = source.split_start
        self.split_end = source.split_end
        self.split_tokens = source.split_tokens
        self.manifest = source.manifest
        self.manifest_path = source.manifest.manifest_path
        self.manifest_sha256 = source.manifest_sha256
        self.tokenizer_id = source.tokenizer_id
        self.backend_version = grain_version()
        self.sampler = grain.IndexSampler(
            num_records=len(source),
            shard_options=grain.NoSharding(),
            shuffle=False,
            num_epochs=DATA_PIPELINE_NUM_EPOCHS,
            seed=None,
        )
        self.sampler_summary = repr(self.sampler)
        self.source_summary = repr(source)
        self.loader = grain.DataLoader(
            data_source=source,
            sampler=self.sampler,
            operations=[grain.Batch(batch_size=self.batch_size, drop_remainder=DATA_PIPELINE_DROP_REMAINDER)],
            worker_count=DATA_PIPELINE_WORKER_COUNT,
            worker_buffer_size=1,
            shard_options=grain.NoSharding(),
        )
        self._iterator: Any | None = None
        self._current_state: DataPipelineState | None = None

    @classmethod
    def from_manifest(
        cls,
        path: str | Path,
        *,
        tokenizer_id: str | None,
        split: SplitName,
        seq_len: int,
        batch_size: int,
    ) -> "PreparedTokenGrainPipeline":
        source = PreparedTokenDataSource.from_manifest(
            path,
            tokenizer_id=tokenizer_id,
            split=split,
            seq_len=seq_len,
        )
        return cls(source=source, batch_size=batch_size)

    def initial_state(self) -> DataPipelineState:
        self._iterator = iter(self.loader)
        grain_state = _decode_grain_state(self._iterator.get_state())
        state = self._state_from_components(
            grain_state=grain_state,
            next_record_index=0,
            token_offset=self.split_start,
            epoch=0,
        )
        self._current_state = state
        return state

    def next_batch(self, state: DataPipelineState) -> PipelineBatch:
        self._validate_state(state)
        iterator = self._iterator_for_state(state)
        try:
            raw = next(iterator)
        except StopIteration as exc:
            raise StopIteration(f"not enough records left in {self.split} split for one full batch") from exc
        batch, provenance = self._batch_from_raw(raw, epoch=state.epoch)
        next_record_index = state.next_record_index + provenance.examples
        next_state = self._state_from_components(
            grain_state=_decode_grain_state(iterator.get_state()),
            next_record_index=next_record_index,
            token_offset=self.split_start + next_record_index * self.seq_len,
            epoch=state.epoch,
        )
        self._current_state = next_state
        return PipelineBatch(batch=batch, state=next_state, provenance=provenance)

    def state_to_json(self, state: DataPipelineState) -> dict[str, Any]:
        self._validate_state(state)
        return data_pipeline_state_to_dict(state)

    def state_from_json(self, raw: Mapping[str, Any]) -> DataPipelineState:
        state = data_pipeline_state_from_mapping(raw)
        self._validate_state(state)
        return state

    def describe(self) -> dict[str, Any]:
        return _normalize(
            {
                "schema_version": 1,
                "backend": DATA_PIPELINE_BACKEND,
                "backend_version": self.backend_version,
                "state_schema_version": DATA_PIPELINE_STATE_SCHEMA_VERSION,
                "split": self.split,
                "order": DATA_PIPELINE_ORDER,
                "shuffle": False,
                "num_epochs": DATA_PIPELINE_NUM_EPOCHS,
                "worker_count": DATA_PIPELINE_WORKER_COUNT,
                "prefetch": DATA_PIPELINE_PREFETCH,
                "drop_remainder": DATA_PIPELINE_DROP_REMAINDER,
                "batch_size": self.batch_size,
                "seq_len": self.seq_len,
                "num_records": len(self.source),
                "manifest_path": self.manifest_path,
                "manifest_sha256": self.manifest_sha256,
                "tokenizer_id": self.tokenizer_id,
                "split_start": self.split_start,
                "split_end": self.split_end,
                "split_tokens": self.split_tokens,
                "source_summary": self.source_summary,
                "sampler_summary": self.sampler_summary,
            }
        )

    def close(self) -> None:
        if self._iterator is not None and hasattr(self._iterator, "close"):
            self._iterator.close()

    def _iterator_for_state(self, state: DataPipelineState) -> Any:
        if self._iterator is None:
            self._iterator = iter(self.loader)
            self._iterator.set_state(_encode_grain_state(state.grain_state))
        elif self._current_state != state:
            self._iterator.set_state(_encode_grain_state(state.grain_state))
        self._current_state = state
        return self._iterator

    def _batch_from_raw(self, raw: Mapping[str, Any], *, epoch: int) -> tuple[Batch, BatchProvenance]:
        if not isinstance(raw, Mapping):
            raise ContractError("Grain batch must be a mapping")
        input_ids = np.asarray(raw.get("input_ids"), dtype=np.int32)
        target_ids = np.asarray(raw.get("target_ids"), dtype=np.int32)
        loss_mask = np.asarray(raw.get("loss_mask"), dtype=np.bool_)
        row_starts = tuple(int(value) for value in np.asarray(raw.get("token_start")).tolist())
        row_ends = tuple(int(value) for value in np.asarray(raw.get("token_end")).tolist())
        expected_shape = (self.batch_size, self.seq_len)
        if input_ids.shape != expected_shape or target_ids.shape != expected_shape or loss_mask.shape != expected_shape:
            raise ContractError(
                "Grain batch has unexpected shape: "
                f"input={input_ids.shape} target={target_ids.shape} mask={loss_mask.shape} expected={expected_shape}"
            )
        if len(row_starts) != self.batch_size or len(row_ends) != self.batch_size:
            raise ContractError("Grain batch provenance does not match batch size")
        return (
            Batch(input_ids=input_ids, target_ids=target_ids, loss_mask=loss_mask),
            BatchProvenance(
                split=self.split,
                epoch=epoch,
                token_start=row_starts[0],
                token_end=row_ends[-1],
                examples=self.batch_size,
                target_tokens=self.batch_size * self.seq_len,
                row_start_offsets=row_starts,
            ),
        )

    def _state_from_components(
        self,
        *,
        grain_state: Mapping[str, Any],
        next_record_index: int,
        token_offset: int,
        epoch: int,
    ) -> DataPipelineState:
        return DataPipelineState(
            schema_version=DATA_PIPELINE_STATE_SCHEMA_VERSION,
            backend=DATA_PIPELINE_BACKEND,
            backend_version=self.backend_version,
            split=self.split,
            order=DATA_PIPELINE_ORDER,
            worker_count=DATA_PIPELINE_WORKER_COUNT,
            prefetch=DATA_PIPELINE_PREFETCH,
            manifest_path=self.manifest_path,
            manifest_sha256=self.manifest_sha256,
            tokenizer_id=self.tokenizer_id,
            seq_len=self.seq_len,
            batch_size=self.batch_size,
            num_records=len(self.source),
            next_record_index=next_record_index,
            token_offset=token_offset,
            epoch=epoch,
            sampler_summary=self.sampler_summary,
            source_summary=self.source_summary,
            grain_state=dict(grain_state),
        )

    def _validate_state(self, state: DataPipelineState) -> None:
        expected = self._state_from_components(
            grain_state=state.grain_state,
            next_record_index=state.next_record_index,
            token_offset=state.token_offset,
            epoch=state.epoch,
        )
        checks = (
            ("schema_version", state.schema_version, expected.schema_version),
            ("backend", state.backend, expected.backend),
            ("backend_version", state.backend_version, expected.backend_version),
            ("split", state.split, expected.split),
            ("order", state.order, expected.order),
            ("worker_count", state.worker_count, expected.worker_count),
            ("prefetch", state.prefetch, expected.prefetch),
            ("manifest_path", state.manifest_path, expected.manifest_path),
            ("manifest_sha256", state.manifest_sha256, expected.manifest_sha256),
            ("tokenizer_id", state.tokenizer_id, expected.tokenizer_id),
            ("seq_len", state.seq_len, expected.seq_len),
            ("batch_size", state.batch_size, expected.batch_size),
            ("num_records", state.num_records, expected.num_records),
            ("sampler_summary", state.sampler_summary, expected.sampler_summary),
            ("source_summary", state.source_summary, expected.source_summary),
        )
        for name, actual, wanted in checks:
            if actual != wanted:
                raise ContractError(f"data pipeline state {name} mismatch: state={actual!r} pipeline={wanted!r}")
        if state.next_record_index < 0 or state.next_record_index > len(self.source):
            raise ContractError(
                f"data pipeline next_record_index={state.next_record_index} is outside [0, {len(self.source)}]"
            )
        expected_offset = self.split_start + state.next_record_index * self.seq_len
        if state.token_offset != expected_offset:
            raise ContractError(
                f"data pipeline token_offset={state.token_offset} does not match next_record_index "
                f"{state.next_record_index} expected_offset={expected_offset}"
            )
        if state.epoch != 0:
            raise ContractError("Grain prepared-token pipeline supports only epoch=0 in this slice")


def data_pipeline_compat_payload(
    path: str | Path,
    *,
    tokenizer_id: str | None,
    split: SplitName,
    seq_len: int,
    batch_size: int,
) -> dict[str, Any]:
    """Return the canonical compatibility payload for a Grain data pipeline."""

    pipeline = PreparedTokenGrainPipeline.from_manifest(
        path,
        tokenizer_id=tokenizer_id,
        split=split,
        seq_len=seq_len,
        batch_size=batch_size,
    )
    try:
        return pipeline.describe()
    finally:
        pipeline.close()


def grain_version() -> str | None:
    try:
        return importlib_metadata.version("grain")
    except importlib_metadata.PackageNotFoundError:
        return None


def data_pipeline_state_to_dict(state: DataPipelineState) -> dict[str, Any]:
    return _normalize(asdict(state))


def data_pipeline_state_from_mapping(raw: Mapping[str, Any]) -> DataPipelineState:
    return DataPipelineState(
        schema_version=_required_int(raw, "schema_version", "data pipeline state"),
        backend=_required_str(raw, "backend", "data pipeline state"),
        backend_version=_optional_str(raw, "backend_version", "data pipeline state"),
        split=_required_str(raw, "split", "data pipeline state"),
        order=_required_str(raw, "order", "data pipeline state"),
        worker_count=_required_int(raw, "worker_count", "data pipeline state"),
        prefetch=_required_bool(raw, "prefetch", "data pipeline state"),
        manifest_path=Path(_required_str(raw, "manifest_path", "data pipeline state")),
        manifest_sha256=_required_str(raw, "manifest_sha256", "data pipeline state"),
        tokenizer_id=_required_str(raw, "tokenizer_id", "data pipeline state"),
        seq_len=_required_int(raw, "seq_len", "data pipeline state"),
        batch_size=_required_int(raw, "batch_size", "data pipeline state"),
        num_records=_required_int(raw, "num_records", "data pipeline state"),
        next_record_index=_required_int(raw, "next_record_index", "data pipeline state"),
        token_offset=_required_int(raw, "token_offset", "data pipeline state"),
        epoch=_required_int(raw, "epoch", "data pipeline state"),
        sampler_summary=_required_str(raw, "sampler_summary", "data pipeline state"),
        source_summary=_required_str(raw, "source_summary", "data pipeline state"),
        grain_state=dict(_require_mapping(raw.get("grain_state"), "data pipeline state.grain_state")),
    )


def _source_summary(source: PreparedTokenDataSource) -> dict[str, Any]:
    return _normalize(
        {
            "schema_version": PREPARED_TOKEN_SOURCE_SCHEMA_VERSION,
            "manifest_path": source.manifest.manifest_path,
            "manifest_sha256": source.manifest_sha256,
            "tokenizer_id": source.tokenizer_id,
            "split": source.split,
            "seq_len": source.seq_len,
            "split_start": source.split_start,
            "split_end": source.split_end,
            "split_tokens": source.split_tokens,
            "num_records": source.num_records,
        }
    )


def _decode_grain_state(state: bytes) -> dict[str, Any]:
    try:
        raw = json.loads(state.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"failed to decode Grain iterator state: {exc}") from exc
    if not isinstance(raw, dict):
        raise ContractError("Grain iterator state must decode to a JSON object")
    return raw


def _encode_grain_state(state: Mapping[str, Any]) -> bytes:
    return json.dumps(state, indent=4, sort_keys=True).encode("utf-8")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be a JSON object")
    return value


def _required_int(raw: Mapping[str, Any], key: str, name: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ContractError(f"{name}.{key} must be an integer")
    return value


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


def _optional_str(raw: Mapping[str, Any], key: str, name: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name}.{key} must be a non-empty string or null")
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
    if isinstance(value, np.generic):
        return value.item()
    return value
