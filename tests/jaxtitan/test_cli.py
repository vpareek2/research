from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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


def test_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "jaxtitan.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "config" in result.stdout


def test_cli_config_check(tmp_path: Path) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(MINIMAL_CONFIG)

    result = subprocess.run(
        [sys.executable, "-m", "jaxtitan.cli", "config", "check", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "valid: smoke"
