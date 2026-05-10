"""
Offline dataset preparation for mmap-backed token training.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import getpass
import hashlib
import os
import json
from pathlib import Path
import tomllib
import time
from typing import Iterable, Iterator

import numpy as np
import tiktoken
from tqdm.auto import tqdm

from research.data import REQUIRED_EVAL_DOMAINS, TOKEN_BYTES_FILENAME, build_token_bytes


@dataclass
class SourceConfig:
    type: str
    dataset: str | None = None
    subset: str | None = None
    data_dir: str | None = None
    split: str | None = None
    text_column: str = "text"
    path: str | None = None
    streaming: bool = False


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
    max_tokens: int | None = None
    tokens_per_domain: int | None = None
    shard_tokens: int = 128_000_000
    seed: int = 0

    def __post_init__(self):
        if self.dtype != "uint32":
            raise ValueError(f"Only uint32 prepared token dtype is supported, got {self.dtype}")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be between 0 and 1, got {self.val_fraction}")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.tokens_per_domain is not None and self.tokens_per_domain <= 0:
            raise ValueError(f"tokens_per_domain must be positive, got {self.tokens_per_domain}")
        if self.shard_tokens <= 0:
            raise ValueError(f"shard_tokens must be positive, got {self.shard_tokens}")


@dataclass
class TokenizationConfig:
    workers: int | str = "auto"
    batch_docs: int = 256
    queue_batches: int = 8

    def __post_init__(self):
        if self.workers != "auto" and (not isinstance(self.workers, int) or self.workers <= 0):
            raise ValueError(f"tokenization.workers must be 'auto' or a positive integer, got {self.workers!r}")
        if self.batch_docs <= 0:
            raise ValueError(f"tokenization.batch_docs must be positive, got {self.batch_docs}")
        if self.queue_batches <= 0:
            raise ValueError(f"tokenization.queue_batches must be positive, got {self.queue_batches}")


@dataclass
class DomainConfig:
    name: str
    source: SourceConfig


@dataclass
class PrepareConfig:
    tokenizer: TokenizerConfig
    output: OutputConfig
    source: SourceConfig | None = None
    kind: str = "dataset"
    domains: list[DomainConfig] = field(default_factory=list)
    hf: HfConfig = field(default_factory=HfConfig)
    tokenization: TokenizationConfig = field(default_factory=TokenizationConfig)

    def __post_init__(self):
        if self.kind not in {"dataset", "eval_domains"}:
            raise ValueError(f"Unknown prepare-data kind: {self.kind}")
        if self.kind == "dataset" and self.source is None:
            raise ValueError("Dataset preparation requires [source]")
        if self.kind == "eval_domains":
            _validate_eval_domain_configs(self.domains)


class TextColumnSequence:
    def __init__(self, dataset, text_column: str):
        self.dataset = dataset
        self.text_column = text_column

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        try:
            size = len(self.dataset)
        except TypeError:
            for item in self.dataset:
                yield item[self.text_column]
        else:
            for idx in range(size):
                yield self[idx]

    def __getitem__(self, idx: int) -> str:
        return self.dataset[idx][self.text_column]


def load_prepare_config(path: str | Path) -> PrepareConfig:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    domains = [
        DomainConfig(name=item["name"], source=SourceConfig(**item["source"]))
        for item in data.get("domain", [])
    ]
    return PrepareConfig(
        kind=data.get("kind", "dataset"),
        source=SourceConfig(**data["source"]) if "source" in data else None,
        tokenizer=TokenizerConfig(**data["tokenizer"]),
        output=OutputConfig(**data["output"]),
        domains=domains,
        hf=HfConfig(**data.get("hf", {})),
        tokenization=TokenizationConfig(**data.get("tokenization", {})),
    )


def load_texts(config: SourceConfig) -> Iterable[str]:
    if config.type == "hf":
        token, _ = resolve_hf_token(HfConfig())
        return load_hf_texts(config, token=token)

    if config.type == "text":
        if config.path is None:
            raise ValueError("Text source requires path")
        return _iter_text_lines(Path(config.path))

    raise ValueError(f"Unknown source type: {config.type}")


def load_hf_texts(config: SourceConfig, *, token: str | None) -> Iterable[str]:
    if config.dataset is None or config.split is None:
        raise ValueError("HF source requires dataset and split")
    from datasets import load_dataset

    kwargs = {
        "split": config.split,
        "token": token,
        "streaming": config.streaming,
    }
    if config.data_dir is not None:
        kwargs["data_dir"] = config.data_dir

    if config.subset is None:
        dataset = load_dataset(config.dataset, **kwargs)
    else:
        dataset = load_dataset(config.dataset, config.subset, **kwargs)
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
    if config.kind == "eval_domains":
        return prepare_eval_domains(config)

    assert config.source is not None
    print("Preparing dataset")
    print(f"source: {config.source.type}")
    if config.source.type == "hf":
        print(f"dataset: {config.source.dataset} split={config.source.split}")
    else:
        print(f"path: {config.source.path}")
    print(f"output: {config.output.path}")
    print(f"tokenizer: {config.tokenizer.name} append_eot={config.tokenizer.append_eot}")
    print(f"max_tokens: {config.output.max_tokens}")
    print(f"shard_tokens: {config.output.shard_tokens}")
    print(f"tokenization: workers={config.tokenization.workers} batch_docs={config.tokenization.batch_docs}")

    hf_auth = "not_applicable"
    if config.source.type == "hf":
        token, hf_auth = resolve_hf_token(config.hf)
        print(f"hf_auth: {hf_auth}")
        texts = load_hf_texts(config.source, token=token)
    else:
        texts = load_texts(config.source)
    return prepare_texts(texts, config, hf_auth=hf_auth)


def prepare_texts(texts: Iterable[str], config: PrepareConfig, *, hf_auth: str = "not_applicable") -> dict:
    assert config.source is not None
    tokenizer = tiktoken.get_encoding(config.tokenizer.name)
    output_dir = Path(config.output.path)
    output_dir.mkdir(parents=True, exist_ok=True)
    token_bytes_path = output_dir / TOKEN_BYTES_FILENAME
    if config.output.max_tokens is None:
        raise ValueError("Training dataset preparation requires output.max_tokens")

    print("Writing tokens...")
    token_count, shards, docs_count, elapsed = _write_sharded_tokens(
        texts,
        tokenizer_name=config.tokenizer.name,
        append_eot=config.tokenizer.append_eot,
        output_dir=output_dir,
        max_tokens=config.output.max_tokens,
        shard_tokens=config.output.shard_tokens,
        tokenization=config.tokenization,
    )
    split_idx = int(token_count * (1.0 - config.output.val_fraction))
    print(f"tokens: {token_count} train={split_idx} val={token_count - split_idx}")

    print("Writing token byte table...")
    build_token_bytes(tokenizer).tofile(token_bytes_path)

    manifest = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "training_tokens",
        "source": _source_manifest(config.source),
        "tokenizer": {
            "name": config.tokenizer.name,
            "append_eot": config.tokenizer.append_eot,
            "eot_token": tokenizer.eot_token,
        },
        "hf_auth": hf_auth,
        "dtype": config.output.dtype,
        "val_fraction": config.output.val_fraction,
        "max_tokens": config.output.max_tokens,
        "shard_tokens": config.output.shard_tokens,
        "num_tokens": token_count,
        "num_docs": docs_count,
        "elapsed_sec": elapsed,
        "tokens_per_sec": token_count / elapsed if elapsed > 0.0 else None,
        "tokenization": {
            "workers": _resolve_workers(config.tokenization.workers),
            "batch_docs": config.tokenization.batch_docs,
            "queue_batches": config.tokenization.queue_batches,
        },
        "splits": {
            "train": {"start": 0, "end": split_idx, "tokens": split_idx},
            "val": {"start": split_idx, "end": token_count, "tokens": token_count - split_idx},
        },
        "shards": shards,
        "files": {
            "token_bytes": {"path": TOKEN_BYTES_FILENAME, "sha256": _sha256(token_bytes_path, desc=f"Hashing {TOKEN_BYTES_FILENAME}")},
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


def prepare_eval_domains(config: PrepareConfig) -> dict:
    if config.output.tokens_per_domain is None:
        raise ValueError("eval_domains preparation requires output.tokens_per_domain")

    print("Preparing eval domain pack")
    print(f"output: {config.output.path}")
    print(f"tokenizer: {config.tokenizer.name} append_eot={config.tokenizer.append_eot}")
    print(f"tokens_per_domain: {config.output.tokens_per_domain}")

    tokenizer = tiktoken.get_encoding(config.tokenizer.name)
    output_dir = Path(config.output.path)
    output_dir.mkdir(parents=True, exist_ok=True)
    token_bytes_path = output_dir / TOKEN_BYTES_FILENAME

    print("Writing token byte table...")
    build_token_bytes(tokenizer).tofile(token_bytes_path)

    hf_auth = "not_applicable"
    token = None
    if any(domain.source.type == "hf" for domain in config.domains):
        token, hf_auth = resolve_hf_token(config.hf)
        print(f"hf_auth: {hf_auth}")

    domains = {}
    domain_by_name = {domain.name: domain for domain in config.domains}
    for name in REQUIRED_EVAL_DOMAINS:
        domain = domain_by_name[name]
        domain_dir = output_dir / name
        domain_dir.mkdir(parents=True, exist_ok=True)
        tokens_path = domain_dir / "tokens.bin"
        print(f"Writing domain {name}...")
        texts = load_hf_texts(domain.source, token=token) if domain.source.type == "hf" else load_texts(domain.source)
        token_count = _write_limited_tokens_bin(
            texts,
            tokenizer=tokenizer,
            append_eot=config.tokenizer.append_eot,
            tokens_path=tokens_path,
            max_tokens=config.output.tokens_per_domain,
        )

        domain_manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "kind": "eval_domain",
            "domain": name,
            "source": _source_manifest(domain.source),
            "tokenizer": {
                "name": config.tokenizer.name,
                "append_eot": config.tokenizer.append_eot,
                "eot_token": tokenizer.eot_token,
            },
            "dtype": config.output.dtype,
            "num_tokens": token_count,
            "splits": {
                "train": {"start": 0, "end": 0, "tokens": 0},
                "val": {"start": 0, "end": token_count, "tokens": token_count},
            },
            "files": {
                "tokens": {"path": "tokens.bin", "sha256": _sha256(tokens_path, desc=f"Hashing {name}/tokens.bin")},
            },
        }
        with (domain_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(domain_manifest, f, indent=2, sort_keys=True)
            f.write("\n")

        domains[name] = {
            "path": name,
            "num_tokens": token_count,
            "source": _source_manifest(domain.source),
            "files": domain_manifest["files"],
        }

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "eval_domains",
        "required_domains": list(REQUIRED_EVAL_DOMAINS),
        "tokens_per_domain": config.output.tokens_per_domain,
        "seed": config.output.seed,
        "tokenizer": {
            "name": config.tokenizer.name,
            "append_eot": config.tokenizer.append_eot,
            "eot_token": tokenizer.eot_token,
        },
        "hf_auth": hf_auth,
        "dtype": config.output.dtype,
        "domains": domains,
        "files": {
            "token_bytes": {"path": TOKEN_BYTES_FILENAME, "sha256": _sha256(token_bytes_path, desc=f"Hashing {TOKEN_BYTES_FILENAME}")},
        },
    }

    print("Writing eval domain manifest...")
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    return manifest


def _iter_text_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")


def _tokenize(text: str, tokenizer: tiktoken.Encoding, append_eot: bool) -> list[int]:
    tokens = tokenizer.encode(text)
    if append_eot:
        tokens.append(tokenizer.eot_token)
    return tokens


_WORKER_TOKENIZER = None
_WORKER_APPEND_EOT = True


def _init_tokenizer_worker(tokenizer_name: str, append_eot: bool):
    global _WORKER_TOKENIZER, _WORKER_APPEND_EOT
    _WORKER_TOKENIZER = tiktoken.get_encoding(tokenizer_name)
    _WORKER_APPEND_EOT = append_eot


def _tokenize_batch_worker(texts: list[str]) -> list[list[int]]:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("Tokenizer worker was not initialized")
    return [_tokenize(text, _WORKER_TOKENIZER, _WORKER_APPEND_EOT) for text in texts]


def _batched(texts: Iterable[str], batch_docs: int) -> Iterator[list[str]]:
    batch = []
    for text in texts:
        batch.append(text)
        if len(batch) >= batch_docs:
            yield batch
            batch = []
    if batch:
        yield batch


def _resolve_workers(workers: int | str) -> int:
    if workers == "auto":
        return max(1, (os.cpu_count() or 2) - 2)
    return int(workers)


def _tokenized_batches(
    texts: Iterable[str],
    *,
    tokenizer_name: str,
    append_eot: bool,
    tokenization: TokenizationConfig,
) -> Iterator[list[list[int]]]:
    workers = _resolve_workers(tokenization.workers)
    batches = _batched(texts, tokenization.batch_docs)
    if workers == 1:
        tokenizer = tiktoken.get_encoding(tokenizer_name)
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

        for _ in range(max_pending):
            if not submit_next():
                break

        while pending:
            future = pending.popleft()
            yield future.result()
            submit_next()


class ShardWriter:
    def __init__(self, output_dir: Path, *, shard_tokens: int):
        self.output_dir = output_dir
        self.shard_tokens = shard_tokens
        self.shards = []
        self.total_tokens = 0
        self.current_path: Path | None = None
        self.current_file = None
        self.current_hash = None
        self.current_start = 0
        self.current_tokens = 0

    def write(self, tokens: list[int]):
        offset = 0
        while offset < len(tokens):
            self._ensure_open()
            remaining = self.shard_tokens - self.current_tokens
            take = min(remaining, len(tokens) - offset)
            array = np.asarray(tokens[offset : offset + take], dtype=np.uint32)
            data = array.tobytes()
            self.current_file.write(data)
            self.current_hash.update(data)
            self.current_tokens += take
            self.total_tokens += take
            offset += take
            if self.current_tokens >= self.shard_tokens:
                self.close_current()

    def close(self):
        self.close_current()

    def _ensure_open(self):
        if self.current_file is not None:
            return
        shard_idx = len(self.shards)
        self.current_path = self.output_dir / f"tokens-{shard_idx:05d}.bin"
        self.current_file = self.current_path.open("wb")
        self.current_hash = hashlib.sha256()
        self.current_start = self.total_tokens
        self.current_tokens = 0

    def close_current(self):
        if self.current_file is None:
            return
        self.current_file.close()
        path = self.current_path
        tokens = self.current_tokens
        if tokens > 0:
            self.shards.append(
                {
                    "path": path.name,
                    "start": self.current_start,
                    "end": self.current_start + tokens,
                    "tokens": tokens,
                    "bytes": path.stat().st_size,
                    "sha256": self.current_hash.hexdigest(),
                }
            )
        elif path.exists():
            path.unlink()
        self.current_path = None
        self.current_file = None
        self.current_hash = None
        self.current_tokens = 0


def _write_sharded_tokens(
    texts: Iterable[str],
    *,
    tokenizer_name: str,
    append_eot: bool,
    output_dir: Path,
    max_tokens: int,
    shard_tokens: int,
    tokenization: TokenizationConfig,
) -> tuple[int, list[dict], int, float]:
    start = time.perf_counter()
    docs_count = 0
    writer = ShardWriter(output_dir, shard_tokens=shard_tokens)
    with tqdm(total=max_tokens, desc="Tokenizing tokens", unit="tok") as progress:
        for tokenized_batch in _tokenized_batches(
            texts,
            tokenizer_name=tokenizer_name,
            append_eot=append_eot,
            tokenization=tokenization,
        ):
            for tokens in tokenized_batch:
                docs_count += 1
                remaining = max_tokens - writer.total_tokens
                if remaining <= 0:
                    writer.close()
                    return writer.total_tokens, writer.shards, docs_count, time.perf_counter() - start
                if not tokens:
                    continue
                tokens = tokens[:remaining]
                writer.write(tokens)
                progress.update(len(tokens))
    writer.close()
    return writer.total_tokens, writer.shards, docs_count, time.perf_counter() - start


def _write_limited_tokens_bin(
    texts: Iterable[str],
    *,
    tokenizer: tiktoken.Encoding,
    append_eot: bool,
    tokens_path: Path,
    max_tokens: int,
) -> int:
    token_count = 0
    with tokens_path.open("wb") as f:
        for text in tqdm(texts, total=_safe_len(texts), desc="Tokenizing docs", unit="doc"):
            remaining = max_tokens - token_count
            if remaining <= 0:
                break
            tokens = _tokenize(text, tokenizer, append_eot)
            if not tokens:
                continue
            tokens = tokens[:remaining]
            np.asarray(tokens, dtype=np.uint32).tofile(f)
            token_count += len(tokens)
    return token_count


def _validate_eval_domain_configs(domains: list[DomainConfig]):
    names = [domain.name for domain in domains]
    required = set(REQUIRED_EVAL_DOMAINS)
    found = set(names)
    duplicates = sorted(name for name in found if names.count(name) > 1)
    missing = sorted(required - found)
    extra = sorted(found - required)
    if duplicates or missing or extra:
        raise ValueError(f"eval_domains must contain exactly {sorted(required)}; missing={missing}, extra={extra}, duplicates={duplicates}")


def _source_manifest(source: SourceConfig) -> dict:
    return {
        "type": source.type,
        "dataset": source.dataset,
        "subset": source.subset,
        "data_dir": source.data_dir,
        "split": source.split,
        "text_column": source.text_column,
        "path": source.path,
        "streaming": source.streaming,
    }


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
    if config.kind == "eval_domains":
        print(f"domains: {', '.join(manifest['domains'])}")
    else:
        print(f"tokens: {manifest['num_tokens']} train={manifest['train_tokens']} val={manifest['val_tokens']}")


if __name__ == "__main__":
    main()
