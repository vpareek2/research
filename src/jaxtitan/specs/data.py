"""Data specs and dataset manifest contracts."""

from dataclasses import dataclass
from pathlib import Path

from jaxtitan.errors import ContractError


@dataclass(frozen=True, slots=True)
class HFStreamingSpec:
    """Hugging Face streaming source contract for runtime training."""

    dataset: str
    split: str
    revision: str
    name: str | None = None
    data_dir: str | None = None
    text_column: str = "text"
    append_eot: bool = True

    def __post_init__(self) -> None:
        if not self.dataset:
            raise ContractError("data.hf_streaming.dataset must be a non-empty string")
        if not self.split:
            raise ContractError("data.hf_streaming.split must be a non-empty string")
        if not self.revision:
            raise ContractError("data.hf_streaming.revision must be a non-empty pinned revision")
        if self.name is not None and not self.name:
            raise ContractError("data.hf_streaming.name must be non-empty when provided")
        if self.data_dir is not None and not self.data_dir:
            raise ContractError("data.hf_streaming.data_dir must be non-empty when provided")
        if not self.text_column:
            raise ContractError("data.hf_streaming.text_column must be a non-empty string")
        if not isinstance(self.append_eot, bool):
            raise ContractError("data.hf_streaming.append_eot must be a boolean")


@dataclass(frozen=True, slots=True)
class ShardInfo:
    """Prepared token shard metadata."""

    path: Path
    num_tokens: int
    sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.num_tokens <= 0:
            raise ContractError(f"data shard num_tokens must be positive, got {self.num_tokens}")
        if self.sha256 is not None and not self.sha256:
            raise ContractError("data shard sha256 must be non-empty when provided")


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Prepared-token dataset manifest shape."""

    dataset_id: str
    tokenizer_id: str
    token_dtype: str
    num_tokens: int
    shards: tuple[ShardInfo, ...]
    split: str = "train"
    schema_version: int = 1
    hash: str | None = None

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ContractError("dataset_manifest.dataset_id must be non-empty")
        if not self.tokenizer_id:
            raise ContractError("dataset_manifest.tokenizer_id must be non-empty")
        if not self.token_dtype:
            raise ContractError("dataset_manifest.token_dtype must be non-empty")
        if self.num_tokens <= 0:
            raise ContractError(f"dataset_manifest.num_tokens must be positive, got {self.num_tokens}")
        object.__setattr__(self, "shards", tuple(self.shards))
        if not self.shards:
            raise ContractError("dataset_manifest.shards must be non-empty")
        if sum(shard.num_tokens for shard in self.shards) != self.num_tokens:
            raise ContractError("dataset_manifest.num_tokens must equal the sum of shard token counts")
        if self.schema_version <= 0:
            raise ContractError(f"dataset_manifest.schema_version must be positive, got {self.schema_version}")


@dataclass(frozen=True, slots=True)
class DataSpec:
    """Static data source contract for a run."""

    mode: str = "prepared"
    train_manifest: Path | None = None
    tokenizer_id: str | None = None
    validation_manifest: Path | None = None
    hf_streaming: HFStreamingSpec | None = None
    order: str = "sequential"
    shuffle_seed: int | None = None
    worker_count: int = 0
    worker_buffer_size: int = 1
    prefetch: bool = False
    document_buffer_size: int | None = None
    document_refill_size: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"prepared", "hf_streaming"}:
            raise ContractError(f"data.mode must be 'prepared' or 'hf_streaming', got {self.mode!r}")
        if self.mode == "prepared":
            if self.train_manifest is None:
                raise ContractError("data.train_manifest is required when data.mode='prepared'")
            object.__setattr__(self, "train_manifest", Path(self.train_manifest))
            if self.hf_streaming is not None:
                raise ContractError("data.hf_streaming is only valid when data.mode='hf_streaming'")
        else:
            if self.train_manifest is not None:
                object.__setattr__(self, "train_manifest", Path(self.train_manifest))
                raise ContractError("data.train_manifest must be omitted when data.mode='hf_streaming'")
            if self.hf_streaming is None:
                raise ContractError("data.hf_streaming is required when data.mode='hf_streaming'")
            if self.tokenizer_id is None:
                raise ContractError("data.tokenizer_id is required when data.mode='hf_streaming'")
        if self.validation_manifest is not None:
            object.__setattr__(self, "validation_manifest", Path(self.validation_manifest))
        if self.tokenizer_id is not None and not self.tokenizer_id:
            raise ContractError("data.tokenizer_id must be non-empty when provided")
        if self.order not in {"sequential", "shuffle", "document_buffer"}:
            raise ContractError(
                f"data.order must be 'sequential', 'shuffle', or 'document_buffer', got {self.order!r}"
            )
        if self.mode == "hf_streaming":
            if self.order != "sequential":
                raise ContractError("data.mode='hf_streaming' supports only data.order='sequential'")
            if self.shuffle_seed is not None:
                raise ContractError("data.shuffle_seed must be null or omitted when data.mode='hf_streaming'")
            if self.worker_count != 0 or self.worker_buffer_size != 1 or self.prefetch:
                raise ContractError(
                    "data.mode='hf_streaming' requires worker_count=0, worker_buffer_size=1, and prefetch=false"
                )
            if self.document_buffer_size is not None or self.document_refill_size is not None:
                raise ContractError("document buffer settings are not supported when data.mode='hf_streaming'")
        if self.order in {"shuffle", "document_buffer"} and self.shuffle_seed is None:
            raise ContractError(f"data.shuffle_seed is required when data.order={self.order!r}")
        if self.order == "sequential" and self.shuffle_seed is not None:
            raise ContractError("data.shuffle_seed must be null or omitted when data.order='sequential'")
        if self.shuffle_seed is not None and (
            not isinstance(self.shuffle_seed, int) or isinstance(self.shuffle_seed, bool)
        ):
            raise ContractError("data.shuffle_seed must be an integer when provided")
        if not isinstance(self.worker_count, int) or isinstance(self.worker_count, bool):
            raise ContractError("data.worker_count must be an integer")
        if not isinstance(self.worker_buffer_size, int) or isinstance(self.worker_buffer_size, bool):
            raise ContractError("data.worker_buffer_size must be an integer")
        if not isinstance(self.prefetch, bool):
            raise ContractError("data.prefetch must be a boolean")
        if self.shuffle_seed is not None and self.shuffle_seed < 0:
            raise ContractError(f"data.shuffle_seed must be non-negative, got {self.shuffle_seed}")
        if self.worker_count < 0:
            raise ContractError(f"data.worker_count must be non-negative, got {self.worker_count}")
        if self.worker_buffer_size <= 0:
            raise ContractError(f"data.worker_buffer_size must be positive, got {self.worker_buffer_size}")
        if self.document_buffer_size is not None and (
            not isinstance(self.document_buffer_size, int)
            or isinstance(self.document_buffer_size, bool)
            or self.document_buffer_size <= 0
        ):
            raise ContractError("data.document_buffer_size must be a positive integer when provided")
        if self.document_refill_size is not None and (
            not isinstance(self.document_refill_size, int)
            or isinstance(self.document_refill_size, bool)
            or self.document_refill_size <= 0
        ):
            raise ContractError("data.document_refill_size must be a positive integer when provided")
        if self.order == "document_buffer":
            if self.document_buffer_size is None:
                raise ContractError("data.document_buffer_size is required when data.order='document_buffer'")
            if self.document_refill_size is None:
                raise ContractError("data.document_refill_size is required when data.order='document_buffer'")
            if self.worker_count != 0 or self.worker_buffer_size != 1 or self.prefetch:
                raise ContractError(
                    "data.order='document_buffer' requires worker_count=0, worker_buffer_size=1, and prefetch=false"
                )
        elif self.document_buffer_size is not None or self.document_refill_size is not None:
            raise ContractError("document buffer settings require data.order='document_buffer'")
