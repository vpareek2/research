import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_force_host_platform_device_count" not in _xla_flags:
    os.environ["XLA_FLAGS"] = f"{_xla_flags} --xla_force_host_platform_device_count=4".strip()

from collections.abc import Callable, Sequence
from hashlib import sha256
import json
from pathlib import Path
import struct

import pytest

TOKENIZER_ID = "toy-tokenizer"


def minimal_config_text(train_manifest: Path | str, *, tokenizer_id: str = TOKENIZER_ID) -> str:
    return f"""
[run]
id = "smoke"
seed = 11
output_dir = "runs"

[model]
name = "decoder"
variant = "tiny"
vocab_size = 32000
hidden_size = 128
intermediate_size = 512
num_layers = 2
num_heads = 4
max_seq_len = 64

[optimizer]
name = "adamw"
weight_decay = 0.1

[optimizer.schedule]
name = "constant"
peak_lr = 0.001

[data]
train_manifest = "{Path(train_manifest).as_posix()}"
tokenizer_id = "{tokenizer_id}"

[training]
seq_len = 64
global_batch_size = 2
target_tokens = 128
log_every_steps = 1
checkpoint_every_steps = 10

[mesh]
axis_names = ["data"]
axis_sizes = [1]
"""


def write_prepared_dataset(
    root: Path,
    *,
    tokenizer_id: str = TOKENIZER_ID,
    shard_token_groups: Sequence[Sequence[int]] | None = None,
    train_tokens: int | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    token_bytes = root / "token_bytes.bin"
    token_bytes.write_bytes(b"\x00\x00\x01\x00")

    if shard_token_groups is None:
        shard_token_groups = ([1, 2, 3, 4], [5, 6, 7, 8])

    shards = []
    start = 0
    for idx, tokens in enumerate(shard_token_groups):
        name = f"tokens-{idx:05d}.bin"
        path = root / name
        path.write_bytes(struct.pack(f"<{len(tokens)}I", *tokens))
        end = start + len(tokens)
        shards.append(
            {
                "path": name,
                "start": start,
                "end": end,
                "tokens": len(tokens),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
        start = end

    num_tokens = sum(len(tokens) for tokens in shard_token_groups)
    if train_tokens is None:
        train_tokens = min(6, num_tokens)
    val_tokens = num_tokens - train_tokens

    manifest = {
        "schema_version": 2,
        "kind": "training_tokens",
        "dtype": "uint32",
        "tokenizer": {"name": tokenizer_id, "append_eot": True, "eot_token": 0},
        "num_tokens": num_tokens,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "splits": {
            "train": {"start": 0, "end": train_tokens, "tokens": train_tokens},
            "val": {"start": train_tokens, "end": num_tokens, "tokens": val_tokens},
        },
        "files": {
            "token_bytes": {
                "path": "token_bytes.bin",
                "sha256": _sha256(token_bytes),
                "bytes": token_bytes.stat().st_size,
                "dtype": "uint16",
            }
        },
        "shards": shards,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest_path


@pytest.fixture
def prepared_dataset(tmp_path: Path) -> Path:
    return write_prepared_dataset(tmp_path / "data" / "train")


@pytest.fixture
def prepared_dataset_factory(tmp_path: Path) -> Callable[..., Path]:
    def make(name: str = "train", **kwargs) -> Path:
        return write_prepared_dataset(tmp_path / "data" / name, **kwargs)

    return make


@pytest.fixture
def minimal_config(prepared_dataset: Path) -> str:
    return minimal_config_text(prepared_dataset)


@pytest.fixture
def minimal_config_builder() -> Callable[[Path | str], str]:
    return minimal_config_text


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
