"""Grain-backed prepared-token training data pipeline."""

import bisect
from collections.abc import Mapping, Sequence
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

DATA_PIPELINE_STATE_SCHEMA_VERSION = 2
DATA_PIPELINE_BACKEND = "grain"
DATA_PIPELINE_DEFAULT_ORDER = "sequential"
DATA_PIPELINE_DEFAULT_WORKER_COUNT = 0
DATA_PIPELINE_DEFAULT_WORKER_BUFFER_SIZE = 1
DATA_PIPELINE_DEFAULT_PREFETCH = False
DATA_PIPELINE_NUM_EPOCHS = 1
DATA_PIPELINE_DROP_REMAINDER = True
DATA_PIPELINE_DEFAULT_DOCUMENT_BUFFER_SIZE = 8
DATA_PIPELINE_DEFAULT_DOCUMENT_REFILL_SIZE = 8


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
    row_doc_ids: tuple[int, ...] | None = None


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
        self.document_offsets = _open_document_offsets(manifest)
        self.document_aware = self.document_offsets is not None
        self.document_count = None if manifest.documents is None else manifest.documents.count
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
        record = {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "loss_mask": np.ones((self.seq_len,), dtype=np.bool_),
            "token_start": np.asarray(token_start, dtype=np.int64),
            "token_end": np.asarray(token_start + self.seq_len, dtype=np.int64),
            "record_key": np.asarray(record_key, dtype=np.int64),
        }
        if self.document_offsets is not None:
            record["doc_id"] = np.asarray(_document_id_for_token(self.document_offsets, token_start), dtype=np.int32)
        return record

    def __repr__(self) -> str:
        return (
            "PreparedTokenDataSource("
            f"manifest_sha256='{self.manifest_sha256}', "
            f"split='{self.split}', "
            f"seq_len={self.seq_len}, "
            f"tokenizer_id='{self.tokenizer_id}', "
            f"document_aware={self.document_aware}, "
            f"document_count={self.document_count}, "
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
        order: str = DATA_PIPELINE_DEFAULT_ORDER,
        shuffle_seed: int | None = None,
        worker_count: int = DATA_PIPELINE_DEFAULT_WORKER_COUNT,
        worker_buffer_size: int = DATA_PIPELINE_DEFAULT_WORKER_BUFFER_SIZE,
        prefetch: bool = DATA_PIPELINE_DEFAULT_PREFETCH,
    ) -> None:
        if batch_size <= 0:
            raise ContractError(f"batch_size must be positive, got {batch_size}")
        _validate_loader_policy(
            order=order,
            shuffle_seed=shuffle_seed,
            worker_count=worker_count,
            worker_buffer_size=worker_buffer_size,
            prefetch=prefetch,
        )
        if len(source) < batch_size:
            required = batch_size * source.seq_len + 1
            raise ContractError(
                f"{source.split} split has {source.split_tokens} tokens, but one batch requires at least {required}"
            )
        self.source = source
        self.batch_size = int(batch_size)
        self.order = order
        self.shuffle_seed = shuffle_seed
        self.worker_count = int(worker_count)
        self.worker_buffer_size = int(worker_buffer_size)
        self.prefetch = bool(prefetch)
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
            shuffle=order == "shuffle",
            num_epochs=DATA_PIPELINE_NUM_EPOCHS,
            seed=shuffle_seed,
        )
        self.sampler_summary = repr(self.sampler)
        self.source_summary = repr(source)
        self.loader = grain.DataLoader(
            data_source=source,
            sampler=self.sampler,
            operations=[grain.Batch(batch_size=self.batch_size, drop_remainder=DATA_PIPELINE_DROP_REMAINDER)],
            worker_count=self.worker_count,
            worker_buffer_size=self.worker_buffer_size,
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
        order: str = DATA_PIPELINE_DEFAULT_ORDER,
        shuffle_seed: int | None = None,
        worker_count: int = DATA_PIPELINE_DEFAULT_WORKER_COUNT,
        worker_buffer_size: int = DATA_PIPELINE_DEFAULT_WORKER_BUFFER_SIZE,
        prefetch: bool = DATA_PIPELINE_DEFAULT_PREFETCH,
    ) -> "PreparedTokenGrainPipeline":
        source = PreparedTokenDataSource.from_manifest(
            path,
            tokenizer_id=tokenizer_id,
            split=split,
            seq_len=seq_len,
        )
        return cls(
            source=source,
            batch_size=batch_size,
            order=order,
            shuffle_seed=shuffle_seed,
            worker_count=worker_count,
            worker_buffer_size=worker_buffer_size,
            prefetch=prefetch,
        )

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
                "order": self.order,
                "shuffle": self.order == "shuffle",
                "shuffle_seed": self.shuffle_seed,
                "num_epochs": DATA_PIPELINE_NUM_EPOCHS,
                "worker_count": self.worker_count,
                "worker_buffer_size": self.worker_buffer_size,
                "prefetch": self.prefetch,
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
                "document_aware": self.source.document_aware,
                "document_count": self.source.document_count,
                "document_offsets_path": None
                if self.manifest.documents is None
                else self.manifest.documents.path.as_posix(),
                "document_offsets_sha256": None if self.manifest.documents is None else self.manifest.documents.sha256,
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
        raw_doc_ids = raw.get("doc_id")
        if self.source.document_aware and raw_doc_ids is None:
            raise ContractError("document-aware Grain batch must include doc_ids")
        if not self.source.document_aware and raw_doc_ids is not None:
            raise ContractError("token-only Grain batch must not include doc_ids")
        doc_ids = None if raw_doc_ids is None else np.asarray(raw_doc_ids, dtype=np.int32)
        if doc_ids is not None and doc_ids.shape != (self.batch_size,):
            raise ContractError(f"Grain batch doc_ids shape={doc_ids.shape} expected={(self.batch_size,)}")
        return (
            Batch(input_ids=input_ids, target_ids=target_ids, loss_mask=loss_mask, doc_ids=doc_ids),
            BatchProvenance(
                split=self.split,
                epoch=epoch,
                token_start=min(row_starts),
                token_end=max(row_ends),
                examples=self.batch_size,
                target_tokens=self.batch_size * self.seq_len,
                row_start_offsets=row_starts,
                row_doc_ids=None if doc_ids is None else tuple(int(value) for value in doc_ids.tolist()),
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
            order=self.order,
            shuffle_seed=self.shuffle_seed,
            worker_count=self.worker_count,
            worker_buffer_size=self.worker_buffer_size,
            prefetch=self.prefetch,
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
            ("shuffle_seed", state.shuffle_seed, expected.shuffle_seed),
            ("worker_count", state.worker_count, expected.worker_count),
            ("worker_buffer_size", state.worker_buffer_size, expected.worker_buffer_size),
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


class PreparedTokenDocumentBufferPipeline:
    """Deterministic document-buffer packer for prepared-token LM batches."""

    def __init__(
        self,
        *,
        manifest: PreparedDatasetManifest,
        split: SplitName,
        seq_len: int,
        batch_size: int,
        shuffle_seed: int,
        document_buffer_size: int,
        document_refill_size: int,
    ) -> None:
        if manifest.documents is None:
            raise ContractError("data.order='document_buffer' requires prepared manifest document offsets")
        if seq_len <= 0:
            raise ContractError(f"seq_len must be positive, got {seq_len}")
        if batch_size <= 0:
            raise ContractError(f"batch_size must be positive, got {batch_size}")
        if document_buffer_size <= 0:
            raise ContractError(f"document_buffer_size must be positive, got {document_buffer_size}")
        if document_refill_size <= 0:
            raise ContractError(f"document_refill_size must be positive, got {document_refill_size}")
        if split not in {"train", "val"}:
            raise ContractError(f"split must be 'train' or 'val', got {split!r}")
        token_split = manifest.train if split == "train" else manifest.val
        self.manifest = manifest
        self.split = split
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.shuffle_seed = int(shuffle_seed)
        self.document_buffer_size = int(document_buffer_size)
        self.document_refill_size = int(document_refill_size)
        self.split_start = int(token_split.start)
        self.split_end = int(token_split.end)
        self.split_tokens = int(token_split.tokens)
        self.manifest_path = manifest.manifest_path
        self.manifest_sha256 = manifest.manifest_sha256
        self.tokenizer_id = manifest.tokenizer_id
        self.backend_version = grain_version()
        self.document_offsets = _open_document_offsets(manifest)
        self.document_spans = _document_spans(self.document_offsets, self.split_start, self.split_end)
        self.document_ids = tuple(self.document_spans)
        self.document_count = manifest.documents.count
        self.num_records = max(0, self.split_tokens // (self.seq_len + 1))
        if len(self.document_ids) < 1:
            raise ContractError(f"{split} split has no document spans")
        if self.num_records < self.batch_size:
            required = self.batch_size * (self.seq_len + 1)
            raise ContractError(
                f"{split} split has {self.split_tokens} tokens, but one document-buffer batch requires at least {required}"
            )
        self.source_summary = repr(self)
        self.sampler_summary = (
            "DocumentBufferSampler("
            f"seed={self.shuffle_seed}, buffer_size={self.document_buffer_size}, "
            f"refill_size={self.document_refill_size}, documents={len(self.document_ids)}"
            ")"
        )

    @classmethod
    def from_manifest(
        cls,
        path: str | Path,
        *,
        tokenizer_id: str | None,
        split: SplitName,
        seq_len: int,
        batch_size: int,
        shuffle_seed: int,
        document_buffer_size: int,
        document_refill_size: int,
    ) -> "PreparedTokenDocumentBufferPipeline":
        manifest = validate_dataset_manifest(path, tokenizer_id=tokenizer_id)
        return cls(
            manifest=manifest,
            split=split,
            seq_len=seq_len,
            batch_size=batch_size,
            shuffle_seed=shuffle_seed,
            document_buffer_size=document_buffer_size,
            document_refill_size=document_refill_size,
        )

    def initial_state(self) -> DataPipelineState:
        rng = np.random.default_rng(self.shuffle_seed)
        document_order = [int(value) for value in rng.permutation(np.asarray(self.document_ids, dtype=np.int64)).tolist()]
        backend_state = {
            "rng_state": _normalize(rng.bit_generator.state),
            "document_order": document_order,
            "replacement_cursor": 0,
            "refill_queue": [],
            "active": [],
        }
        backend_state = self._fill_active(backend_state)
        return self._state_from_components(
            backend_state=backend_state,
            next_record_index=0,
            token_offset=self.split_start,
            epoch=0,
        )

    def next_batch(self, state: DataPipelineState) -> PipelineBatch:
        self._validate_state(state)
        backend_state = _copy_backend_state(state.grain_state)
        rng = np.random.default_rng()
        rng.bit_generator.state = backend_state["rng_state"]
        records = []
        for _idx in range(self.batch_size):
            try:
                records.append(self._next_record(backend_state, rng))
            except StopIteration as exc:
                raise StopIteration(f"not enough document-buffer tokens left in {self.split} split for one full batch") from exc
        backend_state["rng_state"] = _normalize(rng.bit_generator.state)
        input_ids = np.stack([record["input_ids"] for record in records])
        target_ids = np.stack([record["target_ids"] for record in records])
        loss_mask = np.stack([record["loss_mask"] for record in records])
        doc_ids = np.asarray([record["doc_id"] for record in records], dtype=np.int32)
        next_record_index = state.next_record_index + self.batch_size
        next_state = self._state_from_components(
            backend_state=backend_state,
            next_record_index=next_record_index,
            token_offset=self.split_start + next_record_index * self.seq_len,
            epoch=state.epoch,
        )
        row_starts = tuple(int(record["token_start"]) for record in records)
        row_ends = tuple(int(record["token_end"]) for record in records)
        return PipelineBatch(
            batch=Batch(input_ids=input_ids, target_ids=target_ids, loss_mask=loss_mask, doc_ids=doc_ids),
            state=next_state,
            provenance=BatchProvenance(
                split=self.split,
                epoch=state.epoch,
                token_start=min(row_starts),
                token_end=max(row_ends),
                examples=self.batch_size,
                target_tokens=self.batch_size * self.seq_len,
                row_start_offsets=row_starts,
                row_doc_ids=tuple(int(value) for value in doc_ids.tolist()),
            ),
        )

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
                "order": "document_buffer",
                "shuffle": True,
                "shuffle_seed": self.shuffle_seed,
                "num_epochs": DATA_PIPELINE_NUM_EPOCHS,
                "worker_count": DATA_PIPELINE_DEFAULT_WORKER_COUNT,
                "worker_buffer_size": DATA_PIPELINE_DEFAULT_WORKER_BUFFER_SIZE,
                "prefetch": DATA_PIPELINE_DEFAULT_PREFETCH,
                "drop_remainder": DATA_PIPELINE_DROP_REMAINDER,
                "batch_size": self.batch_size,
                "seq_len": self.seq_len,
                "num_records": self.num_records,
                "manifest_path": self.manifest_path,
                "manifest_sha256": self.manifest_sha256,
                "tokenizer_id": self.tokenizer_id,
                "split_start": self.split_start,
                "split_end": self.split_end,
                "split_tokens": self.split_tokens,
                "document_aware": True,
                "document_count": self.document_count,
                "document_offsets_path": self.manifest.documents.path.as_posix(),
                "document_offsets_sha256": self.manifest.documents.sha256,
                "document_buffer_size": self.document_buffer_size,
                "document_refill_size": self.document_refill_size,
                "source_summary": self.source_summary,
                "sampler_summary": self.sampler_summary,
            }
        )

    def close(self) -> None:
        return None

    def __repr__(self) -> str:
        return (
            "PreparedTokenDocumentBufferPipeline("
            f"manifest_sha256='{self.manifest_sha256}', "
            f"split='{self.split}', "
            f"seq_len={self.seq_len}, "
            f"tokenizer_id='{self.tokenizer_id}', "
            f"seed={self.shuffle_seed}, "
            f"buffer_size={self.document_buffer_size}, "
            f"refill_size={self.document_refill_size}, "
            f"documents={len(self.document_ids)}"
            ")"
        )

    def _next_record(self, backend_state: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
        token_values = []
        token_doc_ids = []
        token_offsets = []
        while len(token_values) < self.seq_len + 1:
            backend_state = self._fill_active(backend_state)
            active = backend_state["active"]
            if not active:
                raise StopIteration
            active_idx = int(rng.integers(0, len(active)))
            cursor = int(active[active_idx]["cursor"])
            end = int(active[active_idx]["end"])
            doc_id = int(active[active_idx]["doc_id"])
            take = min(self.seq_len + 1 - len(token_values), end - cursor)
            values = read_token_range(self.manifest, cursor, cursor + take)
            token_values.extend(int(value) for value in values.tolist())
            token_doc_ids.extend([doc_id] * take)
            token_offsets.extend(range(cursor, cursor + take))
            cursor += take
            if cursor >= end:
                del active[active_idx]
            else:
                active[active_idx] = {"doc_id": doc_id, "cursor": cursor, "end": end}
        input_ids = np.asarray(token_values[:-1], dtype=np.int32)
        target_ids = np.asarray(token_values[1:], dtype=np.int32)
        loss_mask = np.asarray(
            [token_doc_ids[idx] == token_doc_ids[idx + 1] for idx in range(self.seq_len)],
            dtype=np.bool_,
        )
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "loss_mask": loss_mask,
            "doc_id": int(token_doc_ids[0]),
            "token_start": int(token_offsets[0]),
            "token_end": int(max(token_offsets) + 1),
        }

    def _fill_active(self, backend_state: dict[str, Any]) -> dict[str, Any]:
        while len(backend_state["active"]) < min(self.document_buffer_size, len(self.document_ids)):
            next_doc = self._pop_replacement_doc(backend_state)
            if next_doc is None:
                break
            start, end = self.document_spans[next_doc]
            backend_state["active"].append({"doc_id": int(next_doc), "cursor": int(start), "end": int(end)})
        return backend_state

    def _pop_replacement_doc(self, backend_state: dict[str, Any]) -> int | None:
        if not backend_state["refill_queue"]:
            cursor = int(backend_state["replacement_cursor"])
            order = backend_state["document_order"]
            if cursor >= len(order):
                return None
            refill_end = min(cursor + self.document_refill_size, len(order))
            backend_state["refill_queue"] = [int(value) for value in order[cursor:refill_end]]
            backend_state["replacement_cursor"] = refill_end
        return int(backend_state["refill_queue"].pop(0))

    def _state_from_components(
        self,
        *,
        backend_state: Mapping[str, Any],
        next_record_index: int,
        token_offset: int,
        epoch: int,
    ) -> DataPipelineState:
        return DataPipelineState(
            schema_version=DATA_PIPELINE_STATE_SCHEMA_VERSION,
            backend=DATA_PIPELINE_BACKEND,
            backend_version=self.backend_version,
            split=self.split,
            order="document_buffer",
            shuffle_seed=self.shuffle_seed,
            worker_count=DATA_PIPELINE_DEFAULT_WORKER_COUNT,
            worker_buffer_size=DATA_PIPELINE_DEFAULT_WORKER_BUFFER_SIZE,
            prefetch=DATA_PIPELINE_DEFAULT_PREFETCH,
            manifest_path=self.manifest_path,
            manifest_sha256=self.manifest_sha256,
            tokenizer_id=self.tokenizer_id,
            seq_len=self.seq_len,
            batch_size=self.batch_size,
            num_records=self.num_records,
            next_record_index=next_record_index,
            token_offset=token_offset,
            epoch=epoch,
            sampler_summary=self.sampler_summary,
            source_summary=self.source_summary,
            grain_state=_normalize(backend_state),
        )

    def _validate_state(self, state: DataPipelineState) -> None:
        expected = self._state_from_components(
            backend_state=state.grain_state,
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
            ("shuffle_seed", state.shuffle_seed, expected.shuffle_seed),
            ("worker_count", state.worker_count, expected.worker_count),
            ("worker_buffer_size", state.worker_buffer_size, expected.worker_buffer_size),
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
        if state.next_record_index < 0 or state.next_record_index > self.num_records:
            raise ContractError(
                f"data pipeline next_record_index={state.next_record_index} is outside [0, {self.num_records}]"
            )
        expected_offset = self.split_start + state.next_record_index * self.seq_len
        if state.token_offset != expected_offset:
            raise ContractError(
                f"data pipeline token_offset={state.token_offset} does not match next_record_index "
                f"{state.next_record_index} expected_offset={expected_offset}"
            )
        if state.epoch != 0:
            raise ContractError("document-buffer pipeline supports only epoch=0 in this slice")
        _require_mapping(state.grain_state.get("rng_state"), "data pipeline state.rng_state")
        _require_int_list(state.grain_state.get("document_order"), "data pipeline state.document_order")
        _required_int(state.grain_state, "replacement_cursor", "data pipeline state")
        _require_int_list(state.grain_state.get("refill_queue"), "data pipeline state.refill_queue")
        active = state.grain_state.get("active")
        if not isinstance(active, list):
            raise ContractError("data pipeline state.active must be a list")
        for idx, item in enumerate(active):
            active_doc = _require_mapping(item, f"data pipeline state.active[{idx}]")
            doc_id = _required_int(active_doc, "doc_id", f"data pipeline state.active[{idx}]")
            cursor = _required_int(active_doc, "cursor", f"data pipeline state.active[{idx}]")
            end = _required_int(active_doc, "end", f"data pipeline state.active[{idx}]")
            if doc_id not in self.document_spans:
                raise ContractError(f"data pipeline active doc_id={doc_id} is not in split documents")
            start, expected_end = self.document_spans[doc_id]
            if end != expected_end or not start <= cursor <= end:
                raise ContractError(f"data pipeline active document cursor is invalid for doc_id={doc_id}")


def data_pipeline_compat_payload(
    path: str | Path,
    *,
    tokenizer_id: str | None,
    split: SplitName,
    seq_len: int,
    batch_size: int,
    order: str = DATA_PIPELINE_DEFAULT_ORDER,
    shuffle_seed: int | None = None,
    worker_count: int = DATA_PIPELINE_DEFAULT_WORKER_COUNT,
    worker_buffer_size: int = DATA_PIPELINE_DEFAULT_WORKER_BUFFER_SIZE,
    prefetch: bool = DATA_PIPELINE_DEFAULT_PREFETCH,
    document_buffer_size: int | None = None,
    document_refill_size: int | None = None,
) -> dict[str, Any]:
    """Return the canonical compatibility payload for a Grain data pipeline."""

    pipeline = build_prepared_token_pipeline(
        path,
        tokenizer_id=tokenizer_id,
        split=split,
        seq_len=seq_len,
        batch_size=batch_size,
        order=order,
        shuffle_seed=shuffle_seed,
        worker_count=worker_count,
        worker_buffer_size=worker_buffer_size,
        prefetch=prefetch,
        document_buffer_size=document_buffer_size,
        document_refill_size=document_refill_size,
    )
    try:
        return pipeline.describe()
    finally:
        pipeline.close()


def build_prepared_token_pipeline(
    path: str | Path,
    *,
    tokenizer_id: str | None,
    split: SplitName,
    seq_len: int,
    batch_size: int,
    order: str = DATA_PIPELINE_DEFAULT_ORDER,
    shuffle_seed: int | None = None,
    worker_count: int = DATA_PIPELINE_DEFAULT_WORKER_COUNT,
    worker_buffer_size: int = DATA_PIPELINE_DEFAULT_WORKER_BUFFER_SIZE,
    prefetch: bool = DATA_PIPELINE_DEFAULT_PREFETCH,
    document_buffer_size: int | None = None,
    document_refill_size: int | None = None,
) -> TrainingDataPipeline:
    if order == "document_buffer":
        if shuffle_seed is None:
            raise ContractError("data.shuffle_seed is required when data.order='document_buffer'")
        if worker_count != 0 or worker_buffer_size != 1 or prefetch:
            raise ContractError(
                "data.order='document_buffer' requires worker_count=0, worker_buffer_size=1, and prefetch=false"
            )
        return PreparedTokenDocumentBufferPipeline.from_manifest(
            path,
            tokenizer_id=tokenizer_id,
            split=split,
            seq_len=seq_len,
            batch_size=batch_size,
            shuffle_seed=shuffle_seed,
            document_buffer_size=document_buffer_size or DATA_PIPELINE_DEFAULT_DOCUMENT_BUFFER_SIZE,
            document_refill_size=document_refill_size or DATA_PIPELINE_DEFAULT_DOCUMENT_REFILL_SIZE,
        )
    return PreparedTokenGrainPipeline.from_manifest(
        path,
        tokenizer_id=tokenizer_id,
        split=split,
        seq_len=seq_len,
        batch_size=batch_size,
        order=order,
        shuffle_seed=shuffle_seed,
        worker_count=worker_count,
        worker_buffer_size=worker_buffer_size,
        prefetch=prefetch,
    )


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
        shuffle_seed=_optional_int(raw, "shuffle_seed", "data pipeline state"),
        worker_count=_required_int(raw, "worker_count", "data pipeline state"),
        worker_buffer_size=_required_int(raw, "worker_buffer_size", "data pipeline state"),
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
            "manifest_path": source.manifest.manifest_path,
            "manifest_sha256": source.manifest_sha256,
            "tokenizer_id": source.tokenizer_id,
            "split": source.split,
            "seq_len": source.seq_len,
            "split_start": source.split_start,
            "split_end": source.split_end,
            "split_tokens": source.split_tokens,
            "num_records": source.num_records,
            "document_aware": source.document_aware,
            "document_count": source.document_count,
            "document_offsets_path": None if source.manifest.documents is None else source.manifest.documents.path,
            "document_offsets_sha256": None if source.manifest.documents is None else source.manifest.documents.sha256,
        }
    )


def _open_document_offsets(manifest: PreparedDatasetManifest) -> np.memmap | None:
    if manifest.documents is None:
        return None
    return np.memmap(manifest.manifest_path.parent / manifest.documents.path, dtype="<u8", mode="r")


def _document_spans(offsets: Sequence[int], split_start: int, split_end: int) -> dict[int, tuple[int, int]]:
    spans = {}
    for doc_id in range(len(offsets) - 1):
        start = max(int(offsets[doc_id]), split_start)
        end = min(int(offsets[doc_id + 1]), split_end)
        if start < end:
            spans[doc_id] = (start, end)
    return spans


def _document_id_for_token(offsets: np.memmap, token_offset: int) -> int:
    return bisect.bisect_right(offsets, token_offset) - 1


def _copy_backend_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(_normalize(state), sort_keys=True))


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


def _require_int_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise ContractError(f"{name} must be a list of integers")
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


def _optional_int(raw: Mapping[str, Any], key: str, name: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ContractError(f"{name}.{key} must be an integer or null")
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


def _validate_loader_policy(
    *,
    order: str,
    shuffle_seed: int | None,
    worker_count: int,
    worker_buffer_size: int,
    prefetch: bool,
) -> None:
    if order not in {"sequential", "shuffle"}:
        raise ContractError(f"data pipeline order must be 'sequential' or 'shuffle', got {order!r}")
    if order == "shuffle" and shuffle_seed is None:
        raise ContractError("data pipeline shuffle_seed is required when order='shuffle'")
    if order == "sequential" and shuffle_seed is not None:
        raise ContractError("data pipeline shuffle_seed must be null when order='sequential'")
    if shuffle_seed is not None and shuffle_seed < 0:
        raise ContractError(f"data pipeline shuffle_seed must be non-negative, got {shuffle_seed}")
    if worker_count < 0:
        raise ContractError(f"data pipeline worker_count must be non-negative, got {worker_count}")
    if worker_buffer_size <= 0:
        raise ContractError(f"data pipeline worker_buffer_size must be positive, got {worker_buffer_size}")
    if not isinstance(prefetch, bool):
        raise ContractError("data pipeline prefetch must be boolean")


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
