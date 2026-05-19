"""Data specs and dataset manifest contracts."""

from dataclasses import dataclass
from pathlib import Path

from jaxtitan.errors import ContractError


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

    train_manifest: Path
    tokenizer_id: str | None = None
    validation_manifest: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_manifest", Path(self.train_manifest))
        if self.validation_manifest is not None:
            object.__setattr__(self, "validation_manifest", Path(self.validation_manifest))
        if self.tokenizer_id is not None and not self.tokenizer_id:
            raise ContractError("data.tokenizer_id must be non-empty when provided")
