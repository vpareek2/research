"""
Data loading utilities.
"""

from pathlib import Path
from typing import Iterator

import grain
import jax
import jax.numpy as jnp
import numpy as np
import tiktoken

from config import DataConfig, TrainConfig


class TokenDataset:
    """
    Fixed-length non-overlapping token chunks.
    """

    def __init__(self, tokens: np.ndarray, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n_examples = len(tokens) // seq_len

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int) -> np.ndarray:
        start = idx * self.seq_len
        end = start + self.seq_len
        return self.tokens[start:end]


def load_tokens(config: DataConfig) -> np.ndarray:
    path = Path(config.path)
    if not path.exists():
        raise FileNotFoundError(f"Data file does not exist: {path}")

    text = path.read_text(encoding="utf-8")
    tokenizer = tiktoken.get_encoding(config.tokenizer)
    tokens = tokenizer.encode(text)
    return np.asarray(tokens, dtype=np.int32)


def make_dataloader(
    data_config: DataConfig,
    train_config: TrainConfig,
) -> Iterator[jax.Array]:
    tokens = load_tokens(data_config)
    dataset = TokenDataset(tokens, train_config.seq_len)
    if len(dataset) == 0:
        raise ValueError(
            f"Not enough tokens ({len(tokens)}) for seq_len={train_config.seq_len}"
        )

    ds = grain.MapDataset.source(dataset)
    ds = ds.shuffle(seed=train_config.seed)
    ds = ds.repeat()
    ds = ds.batch(train_config.batch_size, drop_remainder=True)

    for batch in ds.to_iter_dataset():
        yield jnp.asarray(batch)
