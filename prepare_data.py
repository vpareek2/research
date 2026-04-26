"""
Offline dataset preparation for mmap-backed token training.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import getpass
import hashlib
import os
import json
from pathlib import Path
import tomllib
from typing import Iterable, Sequence

import numpy as np
import tiktoken
from tqdm.auto import tqdm


@dataclass
class SourceConfig:
    type: str
    dataset: str | None = None
    split: str | None = None
    text_column: str = "text"
    path: str | None = None


@dataclass
class TokenizerConfig:
    name: str
    append_eot: bool = True


@dataclass
class HfConfig:
    prompt_for_token: bool = True
    token_env: str = "HF_TOKEN"


@dataclass
class OutputConfig:
    path: str
    dtype: str = "uint32"
    val_fraction: float = 0.001

    def __post_init__(self):
        if self.dtype != "uint32":
            raise ValueError(f"Only uint32 prepared token dtype is supported, got {self.dtype}")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be between 0 and 1, got {self.val_fraction}")


@dataclass
class PrepareConfig:
    source: SourceConfig
    tokenizer: TokenizerConfig
    output: OutputConfig
    hf: HfConfig = field(default_factory=HfConfig)


class TextColumnSequence:
    def __init__(self, dataset, text_column: str):
        self.dataset = dataset
        self.text_column = text_column

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> str:
        return self.dataset[idx][self.text_column]


def load_prepare_config(path: str | Path) -> PrepareConfig:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    return PrepareConfig(
        source=SourceConfig(**data["source"]),
        tokenizer=TokenizerConfig(**data["tokenizer"]),
        output=OutputConfig(**data["output"]),
        hf=HfConfig(**data.get("hf", {})),
    )


def load_texts(config: SourceConfig) -> Iterable[str]:
    if config.type == "hf":
        token, _ = resolve_hf_token(HfConfig())
        return load_hf_texts(config, token=token)

    if config.type == "text":
        if config.path is None:
            raise ValueError("Text source requires path")
        return Path(config.path).read_text(encoding="utf-8").splitlines()

    raise ValueError(f"Unknown source type: {config.type}")


def load_hf_texts(config: SourceConfig, *, token: str | None) -> Iterable[str]:
    if config.dataset is None or config.split is None:
        raise ValueError("HF source requires dataset and split")
    from datasets import load_dataset

    dataset = load_dataset(config.dataset, split=config.split, token=token)
    return TextColumnSequence(dataset, config.text_column)


def resolve_hf_token(config: HfConfig) -> tuple[str | None, str]:
    token = os.environ.get(config.token_env) or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token, "env"

    token = _get_saved_hf_token()
    if token:
        return token, "saved"

    if config.prompt_for_token:
        token = getpass.getpass("HF token (blank for anonymous): ").strip()
        if token:
            _save_hf_token(token)
            return token, "prompt"

    return None, "anonymous"


def _get_saved_hf_token() -> str | None:
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None

    return get_token()


def _save_hf_token(token: str):
    try:
        from huggingface_hub import login
    except ImportError:
        return

    login(token=token, skip_if_logged_in=True)


def prepare_dataset(config: PrepareConfig):
    print("Preparing dataset")
    print(f"source: {config.source.type}")
    if config.source.type == "hf":
        print(f"dataset: {config.source.dataset} split={config.source.split}")
    else:
        print(f"path: {config.source.path}")
    print(f"output: {config.output.path}")
    print(f"tokenizer: {config.tokenizer.name} append_eot={config.tokenizer.append_eot}")

    hf_auth = "not_applicable"
    if config.source.type == "hf":
        token, hf_auth = resolve_hf_token(config.hf)
        print(f"hf_auth: {hf_auth}")
        texts = load_hf_texts(config.source, token=token)
    else:
        texts = load_texts(config.source)
    return prepare_texts(texts, config, hf_auth=hf_auth)


def prepare_texts(texts: Iterable[str], config: PrepareConfig, *, hf_auth: str = "not_applicable") -> dict:
    if not isinstance(texts, Sequence):
        texts = list(texts)

    tokenizer = tiktoken.get_encoding(config.tokenizer.name)
    output_dir = Path(config.output.path)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = output_dir / "tokens.bin"

    print("Writing tokens...")
    token_count = _write_tokens_bin(
        texts,
        tokenizer=tokenizer,
        append_eot=config.tokenizer.append_eot,
        tokens_path=tokens_path,
    )
    split_idx = int(token_count * (1.0 - config.output.val_fraction))
    print(f"tokens: {token_count} train={split_idx} val={token_count - split_idx}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "type": config.source.type,
            "dataset": config.source.dataset,
            "split": config.source.split,
            "text_column": config.source.text_column,
            "path": config.source.path,
        },
        "tokenizer": {
            "name": config.tokenizer.name,
            "append_eot": config.tokenizer.append_eot,
            "eot_token": tokenizer.eot_token,
        },
        "hf_auth": hf_auth,
        "dtype": config.output.dtype,
        "val_fraction": config.output.val_fraction,
        "num_tokens": token_count,
        "splits": {
            "train": {"start": 0, "end": split_idx, "tokens": split_idx},
            "val": {"start": split_idx, "end": token_count, "tokens": token_count - split_idx},
        },
        "files": {
            "tokens": {"path": "tokens.bin", "sha256": _sha256(tokens_path, desc="Hashing tokens.bin")},
        },
    }
    manifest["train_tokens"] = manifest["splits"]["train"]["tokens"]
    manifest["val_tokens"] = manifest["splits"]["val"]["tokens"]

    print("Writing manifest...")
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    return manifest


def _tokenize(text: str, tokenizer: tiktoken.Encoding, append_eot: bool) -> list[int]:
    tokens = tokenizer.encode(text)
    if append_eot:
        tokens.append(tokenizer.eot_token)
    return tokens


def _write_tokens_bin(
    texts: Iterable[str],
    *,
    tokenizer: tiktoken.Encoding,
    append_eot: bool,
    tokens_path: Path,
) -> int:
    token_count = 0
    with tokens_path.open("wb") as f:
        for text in tqdm(texts, total=_safe_len(texts), desc="Tokenizing docs", unit="doc"):
            tokens = np.asarray(_tokenize(text, tokenizer, append_eot), dtype=np.uint32)
            if len(tokens) == 0:
                continue
            tokens.tofile(f)
            token_count += len(tokens)
    return token_count


def _safe_len(value) -> int | None:
    try:
        return len(value)
    except TypeError:
        return None


def _sha256(path: Path, desc: str) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    with path.open("rb") as f, tqdm(total=total, desc=desc, unit="B", unit_scale=True) as progress:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
            progress.update(len(chunk))
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to a data preparation TOML config.")
    args = parser.parse_args()

    config = load_prepare_config(args.config)
    manifest = prepare_dataset(config)
    print(f"wrote {config.output.path}")
    print(f"tokens: {manifest['num_tokens']} train={manifest['train_tokens']} val={manifest['val_tokens']}")


if __name__ == "__main__":
    main()
