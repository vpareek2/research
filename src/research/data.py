"""
Data loading utilities.
"""

from pathlib import Path
from typing import Iterator
import json
import hashlib

import grain
import jax
import numpy as np
import tiktoken

from research.config import DataConfig, EvalConfig, TrainConfig


Batch = dict[str, jax.Array]
TOKEN_BYTES_FILENAME = "token_bytes.bin"
DEFAULT_EVAL_DOMAIN_ROOT = Path("data") / "eval_domains"
REQUIRED_EVAL_DOMAINS = ("web", "knowledge", "books", "news", "code", "math", "reasoning", "docs", "dialogue")


def tokenizer_path_name(tokenizer_name: str) -> str:
    return tokenizer_name.replace("/", "__")


def default_eval_domain_root(tokenizer_name: str) -> Path:
    return DEFAULT_EVAL_DOMAIN_ROOT / tokenizer_path_name(tokenizer_name)


def eval_domain_root(eval_config: EvalConfig, data_config: DataConfig) -> Path:
    if eval_config.domain_root is not None:
        return Path(eval_config.domain_root)
    return default_eval_domain_root(data_config.tokenizer)


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


def build_token_bytes(tokenizer: tiktoken.Encoding) -> np.ndarray:
    token_bytes = np.zeros(tokenizer.n_vocab, dtype=np.uint16)
    for token_id in range(tokenizer.n_vocab):
        try:
            token_bytes[token_id] = len(tokenizer.decode_single_token_bytes(token_id))
        except KeyError:
            token_bytes[token_id] = 0
    token_bytes[tokenizer.eot_token] = 0
    return token_bytes


def load_token_bytes(data_config: DataConfig) -> np.ndarray:
    if data_config.source == "tokens":
        data_dir = Path(data_config.path)
        manifest = load_validated_token_manifest(data_config)
        token_bytes_path = manifest.get("files", {}).get("token_bytes", {}).get("path")
        if token_bytes_path is not None:
            return np.fromfile(data_dir / token_bytes_path, dtype=np.uint16)

    tokenizer = tiktoken.get_encoding(data_config.tokenizer)
    return build_token_bytes(tokenizer)


def load_token_manifest(path: str | Path) -> dict:
    with (Path(path) / "manifest.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_eval_domain_manifest(eval_root: str | Path) -> dict:
    with (Path(eval_root) / "manifest.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_eval_domain_pack(eval_config: EvalConfig, data_config: DataConfig) -> dict:
    eval_root = eval_domain_root(eval_config, data_config)
    manifest = load_eval_domain_manifest(eval_root)
    if manifest.get("kind") != "eval_domains":
        raise ValueError(f"Eval domain manifest kind must be eval_domains, got {manifest.get('kind')!r}")

    manifest_tokenizer = manifest.get("tokenizer", {}).get("name")
    if manifest_tokenizer != data_config.tokenizer:
        raise ValueError(
            f"Eval domain tokenizer {manifest_tokenizer!r} does not match data tokenizer {data_config.tokenizer!r}"
        )

    domains = manifest.get("domains", {})
    domain_names = set(domains)
    required = set(REQUIRED_EVAL_DOMAINS)
    if domain_names != required:
        missing = sorted(required - domain_names)
        extra = sorted(domain_names - required)
        raise ValueError(f"Eval domain pack must contain exactly {sorted(required)}; missing={missing}, extra={extra}")

    for name in REQUIRED_EVAL_DOMAINS:
        domain = domains[name]
        domain_path = eval_root / domain["path"]
        token_path = domain_path / "tokens.bin"
        manifest_path = domain_path / "manifest.json"
        if not token_path.exists():
            raise FileNotFoundError(f"Eval domain token file does not exist: {token_path}")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Eval domain manifest does not exist: {manifest_path}")

    return manifest


def load_eval_domain_token_bytes(eval_config: EvalConfig, data_config: DataConfig) -> np.ndarray:
    manifest = validate_eval_domain_pack(eval_config, data_config)
    token_bytes_path = manifest.get("files", {}).get("token_bytes", {}).get("path")
    if token_bytes_path is not None:
        return np.fromfile(eval_domain_root(eval_config, data_config) / token_bytes_path, dtype=np.uint16)

    tokenizer = tiktoken.get_encoding(data_config.tokenizer)
    return build_token_bytes(tokenizer)


def validate_token_manifest(
    data_dir: str | Path,
    manifest: dict,
    data_config: DataConfig,
    *,
    verify_checksum: bool = False,
) -> dict:
    data_dir = Path(data_dir)
    if manifest.get("dtype") != "uint32":
        raise ValueError(f"Prepared token manifest dtype must be uint32, got {manifest.get('dtype')}")

    manifest_tokenizer = manifest.get("tokenizer", {}).get("name")
    if manifest_tokenizer != data_config.tokenizer:
        raise ValueError(
            f"Prepared token manifest tokenizer {manifest_tokenizer!r} does not "
            f"match config tokenizer {data_config.tokenizer!r}"
        )

    token_file = manifest.get("files", {}).get("tokens", {})
    token_path_value = token_file.get("path")
    if not token_path_value:
        raise ValueError("Prepared token manifest is missing files.tokens.path")
    token_path = data_dir / token_path_value
    if not token_path.exists():
        raise FileNotFoundError(f"Prepared token file does not exist: {token_path}")

    itemsize = np.dtype(np.uint32).itemsize
    token_file_bytes = token_path.stat().st_size
    if token_file_bytes % itemsize != 0:
        raise ValueError(f"Prepared token file size is not divisible by uint32 size: {token_path}")
    file_tokens = token_file_bytes // itemsize
    manifest_tokens = manifest.get("num_tokens")
    if manifest_tokens != file_tokens:
        raise ValueError(f"Prepared token manifest num_tokens={manifest_tokens} does not match token file length={file_tokens}")

    splits = manifest.get("splits", {})
    train_split = _validate_manifest_split(splits, "train", file_tokens)
    val_split = _validate_manifest_split(splits, "val", file_tokens)
    if train_split["start"] < val_split["end"] and val_split["start"] < train_split["end"]:
        raise ValueError(f"Prepared token train/val splits overlap: train={train_split}, val={val_split}")

    expected_sha256 = token_file.get("sha256")
    if verify_checksum and expected_sha256:
        actual_sha256 = _sha256(token_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"Prepared token checksum mismatch for {token_path}")

    return manifest


def load_validated_token_manifest(
    data_config: DataConfig,
    *,
    verify_checksum: bool = False,
) -> dict:
    data_dir = Path(data_config.path)
    manifest = load_token_manifest(data_dir)
    return validate_token_manifest(data_dir, manifest, data_config, verify_checksum=verify_checksum)


def _validate_manifest_split(splits: dict, name: str, num_tokens: int) -> dict:
    if name not in splits:
        raise ValueError(f"Prepared token manifest is missing splits.{name}")
    split = splits[name]
    start = split.get("start")
    end = split.get("end")
    tokens = split.get("tokens")
    if not isinstance(start, int) or not isinstance(end, int) or not isinstance(tokens, int):
        raise ValueError(f"Prepared token split {name} must contain integer start, end, and tokens")
    if not 0 <= start <= end <= num_tokens:
        raise ValueError(f"Prepared token split {name} has invalid bounds start={start}, end={end}, num_tokens={num_tokens}")
    if tokens != end - start:
        raise ValueError(f"Prepared token split {name} tokens={tokens} does not equal end-start={end - start}")
    return split


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        manifest = load_validated_token_manifest(data_config)
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


def domain_eval_steps(eval_config: EvalConfig, train_config: TrainConfig) -> int:
    return eval_config.domain_eval_steps or train_config.eval_steps


def make_eval_domain_dataloaders(
    eval_config: EvalConfig,
    data_config: DataConfig,
    train_config: TrainConfig,
) -> dict[str, Iterator[Batch]]:
    manifest = validate_eval_domain_pack(eval_config, data_config)
    eval_root = eval_domain_root(eval_config, data_config)
    eval_steps = domain_eval_steps(eval_config, train_config)
    required_examples = eval_steps * train_config.batch_size
    dataloaders = {}

    for name in REQUIRED_EVAL_DOMAINS:
        domain = manifest["domains"][name]
        domain_path = eval_root / domain["path"]
        domain_manifest = load_token_manifest(domain_path)
        token_path = domain_path / domain_manifest["files"]["tokens"]["path"]
        num_tokens = int(domain_manifest["num_tokens"])
        dataset = TokenMemmapDataset(token_path, train_config.seq_len, start=0, end=num_tokens)
        if len(dataset) < required_examples:
            raise ValueError(
                f"Eval domain {name!r} has {len(dataset)} examples, but eval requires "
                f"{required_examples} examples ({eval_steps=} * {train_config.batch_size=})"
            )
        dataloaders[name] = _make_iter(dataset, train_config.batch_size, seed=None, shuffle=False, repeat=False)

    return dataloaders
