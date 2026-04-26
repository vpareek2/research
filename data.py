"""
Data loading utilities.
"""

from pathlib import Path
from typing import Iterator
import json

import grain
import jax
import numpy as np
import tiktoken

from config import DataConfig, TrainConfig


Batch = dict[str, jax.Array]


class TokenDataset:
    """
    Fixed-length non-overlapping token chunks with provenance metadata.
    """

    def __init__(self, tokens: np.ndarray, seq_len: int, token_offset: int = 0):
        self.tokens = tokens
        self.seq_len = seq_len
        self.token_offset = token_offset
        self.n_examples = len(tokens) // seq_len

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | int]:
        local_start = idx * self.seq_len
        local_end = local_start + self.seq_len
        start = self.token_offset + local_start
        end = self.token_offset + local_end
        return {
            "input_ids": np.asarray(self.tokens[local_start:local_end], dtype=np.int32),
            "chunk_idx": idx,
            "token_start": start,
            "token_end": end,
        }


class TokenMemmapDataset(TokenDataset):
    """
    Fixed-length chunks backed by a uint32 token .bin file.
    """

    def __init__(
        self,
        path: str | Path,
        seq_len: int,
        *,
        start: int = 0,
        end: int | None = None,
        dtype: str = "uint32",
    ):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Token file does not exist: {self.path}")
        if dtype != "uint32":
            raise ValueError(f"Only uint32 token files are supported, got {dtype}")
        tokens = np.memmap(self.path, dtype=np.uint32, mode="r")
        if end is None:
            end = len(tokens)
        if not 0 <= start <= end <= len(tokens):
            raise ValueError(f"Invalid token window start={start}, end={end}, len={len(tokens)}")
        super().__init__(tokens[start:end], seq_len, token_offset=start)


def load_tokens(config: DataConfig) -> np.ndarray:
    if config.source != "text":
        raise ValueError(f"load_tokens only supports text data, got source={config.source}")

    path = Path(config.path)
    if not path.exists():
        raise FileNotFoundError(f"Data file does not exist: {path}")

    text = path.read_text(encoding="utf-8")
    tokenizer = tiktoken.get_encoding(config.tokenizer)
    tokens = tokenizer.encode(text)
    return np.asarray(tokens, dtype=np.int32)


def load_token_file(path: str | Path, dtype: str = "uint32") -> np.memmap:
    if dtype != "uint32":
        raise ValueError(f"Only uint32 token files are supported, got {dtype}")
    return np.memmap(path, dtype=np.uint32, mode="r")


def load_token_manifest(path: str | Path) -> dict:
    with (Path(path) / "manifest.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def split_tokens(tokens: np.ndarray, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    split_idx = int(len(tokens) * (1.0 - val_fraction))
    return tokens[:split_idx], tokens[split_idx:]


def build_datasets(data_config: DataConfig, train_config: TrainConfig) -> tuple[TokenDataset, TokenDataset]:
    if data_config.source == "text":
        tokens = load_tokens(data_config)
        train_tokens, val_tokens = split_tokens(tokens, data_config.val_fraction)
        return TokenDataset(train_tokens, train_config.seq_len), TokenDataset(val_tokens, train_config.seq_len)

    if data_config.source == "tokens":
        data_dir = Path(data_config.path)
        manifest = load_token_manifest(data_dir)
        token_path = data_dir / manifest["files"]["tokens"]["path"]
        train_split = manifest["splits"]["train"]
        val_split = manifest["splits"]["val"]
        return (
            TokenMemmapDataset(token_path, train_config.seq_len, start=train_split["start"], end=train_split["end"]),
            TokenMemmapDataset(token_path, train_config.seq_len, start=val_split["start"], end=val_split["end"]),
        )

    raise ValueError(f"Unknown data source: {data_config.source}")


def _make_iter(
    dataset: TokenDataset,
    batch_size: int,
    *,
    seed: int | None,
    shuffle: bool,
    repeat: bool,
) -> Iterator[Batch]:
    ds = grain.MapDataset.source(dataset)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=True)
    return iter(ds.to_iter_dataset())


def _check_datasets(train_dataset: TokenDataset, val_dataset: TokenDataset, train_config: TrainConfig):
    if len(train_dataset) == 0:
        raise ValueError(f"Not enough train tokens for seq_len={train_config.seq_len}")
    if len(val_dataset) == 0:
        raise ValueError(f"Not enough val tokens for seq_len={train_config.seq_len}")

    required_val_examples = train_config.eval_steps * train_config.batch_size
    if len(val_dataset) < required_val_examples:
        raise ValueError(
            f"Validation set has {len(val_dataset)} examples, but eval requires "
            f"{required_val_examples} examples ({train_config.eval_steps=} * "
            f"{train_config.batch_size=})"
        )


def make_dataloaders(
    data_config: DataConfig,
    train_config: TrainConfig,
) -> tuple[Iterator[Batch], Iterator[Batch]]:
    train_dataset, val_dataset = build_datasets(data_config, train_config)
    _check_datasets(train_dataset, val_dataset, train_config)

    train_iter = _make_iter(
        train_dataset,
        train_config.batch_size,
        seed=train_config.seed,
        shuffle=True,
        repeat=True,
    )
    val_iter = _make_iter(
        val_dataset,
        train_config.batch_size,
        seed=None,
        shuffle=False,
        repeat=False,
    )
    return train_iter, val_iter


def make_val_dataloader(
    data_config: DataConfig,
    train_config: TrainConfig,
) -> Iterator[Batch]:
    train_dataset, val_dataset = build_datasets(data_config, train_config)
    _check_datasets(train_dataset, val_dataset, train_config)

    return _make_iter(
        val_dataset,
        train_config.batch_size,
        seed=None,
        shuffle=False,
        repeat=False,
    )
