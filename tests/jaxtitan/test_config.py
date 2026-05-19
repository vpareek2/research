from __future__ import annotations

from pathlib import Path

import pytest

from jaxtitan.config import load_config
from jaxtitan.errors import ConfigError

MINIMAL_CONFIG = """
[run]
id = "smoke"
seed = 11
output_dir = "runs"

[model]
name = "decoder"
variant = "tiny"
vocab_size = 32000
hidden_size = 128
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
train_manifest = "data/train/manifest.json"
tokenizer_id = "toy-tokenizer"

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


def test_load_config_resolves_minimal_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(MINIMAL_CONFIG)

    spec = load_config(config_path)

    assert spec.run_id == "smoke"
    assert spec.seed == 11
    assert spec.model.hidden_size == 128
    assert spec.optimizer.schedule.peak_lr == 0.001
    assert spec.data.train_manifest == Path("data/train/manifest.json")
    assert spec.training.target_tokens == 128
    assert spec.mesh.axis_names == ("data",)
    assert spec.artifacts.root == Path("runs")


def test_load_config_rejects_cross_spec_mismatch(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.toml"
    config_path.write_text(MINIMAL_CONFIG.replace("max_seq_len = 64", "max_seq_len = 32"))

    with pytest.raises(ConfigError, match="max_seq_len"):
        load_config(config_path)
