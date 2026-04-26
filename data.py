"""
Data loading utilities.
"""

from pathlib import Path
from typing import Iterator

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

    def __init__(self, tokens: np.ndarray, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n_examples = len(tokens) // seq_len

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | int]:
        start = idx * self.seq_len
        end = start + self.seq_len
        return {
            "input_ids": self.tokens[start:end],
            "chunk_idx": idx,
            "token_start": start,
            "token_end": end,
        }


def load_tokens(config: DataConfig) -> np.ndarray:
    path = Path(config.path)
    if not path.exists():
        raise FileNotFoundError(f"Data file does not exist: {path}")

    text = path.read_text(encoding="utf-8")
    tokenizer = tiktoken.get_encoding(config.tokenizer)
    tokens = tokenizer.encode(text)
    return np.asarray(tokens, dtype=np.int32)


def split_tokens(tokens: np.ndarray, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    split_idx = int(len(tokens) * (1.0 - val_fraction))
    return tokens[:split_idx], tokens[split_idx:]


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


def make_dataloaders(
    data_config: DataConfig,
    train_config: TrainConfig,
) -> tuple[Iterator[Batch], Iterator[Batch]]:
    tokens = load_tokens(data_config)
    train_tokens, val_tokens = split_tokens(tokens, data_config.val_fraction)

    train_dataset = TokenDataset(train_tokens, train_config.seq_len)
    val_dataset = TokenDataset(val_tokens, train_config.seq_len)

    if len(train_dataset) == 0:
        raise ValueError(
            f"Not enough train tokens ({len(train_tokens)}) for seq_len={train_config.seq_len}"
        )
    if len(val_dataset) == 0:
        raise ValueError(
            f"Not enough val tokens ({len(val_tokens)}) for seq_len={train_config.seq_len}"
        )

    required_val_examples = train_config.eval_steps * train_config.batch_size
    if len(val_dataset) < required_val_examples:
        raise ValueError(
            f"Validation set has {len(val_dataset)} examples, but eval requires "
            f"{required_val_examples} examples ({train_config.eval_steps=} * "
            f"{train_config.batch_size=})"
        )

    train_iter = _make_iter(
        train_dataset,
        train_config.batch_size,
        seed=train_config.seed,
        shuffle=True,
        repeat=True,
    )
    val_iter = make_val_dataloader(data_config, train_config)
    return train_iter, val_iter


def make_val_dataloader(
    data_config: DataConfig,
    train_config: TrainConfig,
) -> Iterator[Batch]:
    tokens = load_tokens(data_config)
    _, val_tokens = split_tokens(tokens, data_config.val_fraction)
    val_dataset = TokenDataset(val_tokens, train_config.seq_len)

    required_val_examples = train_config.eval_steps * train_config.batch_size
    if len(val_dataset) < required_val_examples:
        raise ValueError(
            f"Validation set has {len(val_dataset)} examples, but eval requires "
            f"{required_val_examples} examples ({train_config.eval_steps=} * "
            f"{train_config.batch_size=})"
        )

    return _make_iter(
        val_dataset,
        train_config.batch_size,
        seed=None,
        shuffle=False,
        repeat=False,
    )
