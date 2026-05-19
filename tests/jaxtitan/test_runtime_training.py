import json
import subprocess
import sys
from pathlib import Path

import pytest

from jaxtitan.errors import ContractError
from jaxtitan.runtime import run_training


def test_run_training_writes_artifacts_metrics_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "loop",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=16, log_every_steps=1))

    summary = run_training(config_path)

    run_dir = tmp_path / "runs" / "loop"
    assert summary.run_id == "loop"
    assert summary.run_dir == Path("runs/loop")
    assert summary.status == "completed"
    assert summary.steps == 2
    assert summary.tokens_seen == 16
    assert summary.target_tokens == 16
    assert run_dir.is_dir()
    assert (run_dir / "config" / "source.toml").is_file()
    assert (run_dir / "config" / "resolved.json").is_file()
    assert (run_dir / "manifest.json").is_file()

    events = _jsonl(run_dir / "events.jsonl")
    metrics = _jsonl(run_dir / "metrics" / "train.jsonl")
    final = json.loads((run_dir / "summaries" / "final.json").read_text())

    assert [event["type"] for event in events] == ["run_initialized", "training_started", "training_completed"]
    assert [row["step"] for row in metrics] == [1, 2]
    assert metrics[-1]["tokens_seen"] == 16
    assert metrics[-1]["token_count"] == 8
    assert metrics[-1]["loss"] == pytest.approx(metrics[-1]["loss_sum"] / metrics[-1]["token_count"])
    assert metrics[-1]["epoch"] == 0
    assert metrics[-1]["token_start"] == 8
    assert metrics[-1]["token_end"] == 16
    assert metrics[-1]["examples"] == 2
    assert metrics[-1]["target_tokens"] == 8
    assert final["status"] == "completed"
    assert final["steps"] == metrics[-1]["step"]
    assert final["tokens_seen"] == metrics[-1]["tokens_seen"]
    assert final["final_loss"] == pytest.approx(metrics[-1]["loss"])


def test_run_training_logs_final_row_even_when_not_on_log_interval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "final-row",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=16, log_every_steps=5))

    run_training(config_path)

    metrics = _jsonl(tmp_path / "runs" / "loop" / "metrics" / "train.jsonl")
    assert [row["step"] for row in metrics] == [2]
    assert metrics[-1]["tokens_seen"] == 16


def test_run_training_stops_after_crossing_target_token_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "cross-target",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=10, log_every_steps=1))

    summary = run_training(config_path)

    assert summary.steps == 2
    assert summary.tokens_seen == 16
    assert summary.target_tokens == 10


def test_run_training_records_failure_when_dataset_exhausts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "exhaust",
        shard_token_groups=(tuple(range(0, 20)),),
        train_tokens=17,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=24, log_every_steps=1))

    with pytest.raises(ContractError, match="prepared train split ended"):
        run_training(config_path)

    run_dir = tmp_path / "runs" / "loop"
    events = _jsonl(run_dir / "events.jsonl")
    metrics = _jsonl(run_dir / "metrics" / "train.jsonl")
    assert events[-1]["type"] == "training_failed"
    assert events[-1]["error_type"] == "ContractError"
    assert [row["step"] for row in metrics] == [1, 2]
    assert not (run_dir / "summaries" / "final.json").exists()


def test_cli_run_train_succeeds_for_tiny_run(
    tmp_path: Path,
    prepared_dataset_factory,
) -> None:
    manifest = prepared_dataset_factory(
        "cli-loop",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=8, log_every_steps=1))

    result = subprocess.run(
        [sys.executable, "-m", "jaxtitan.cli", "run", "train", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "runs/loop"
    assert (tmp_path / "runs" / "loop" / "metrics" / "train.jsonl").is_file()
    assert (tmp_path / "runs" / "loop" / "summaries" / "final.json").is_file()


def test_cli_run_train_invalid_data_fails_cleanly(tmp_path: Path) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(tmp_path / "missing" / "manifest.json", target_tokens=8, log_every_steps=1))

    result = subprocess.run(
        [sys.executable, "-m", "jaxtitan.cli", "run", "train", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 2
    assert "manifest does not exist" in result.stderr
    assert "Traceback" not in result.stderr


def _training_config(train_manifest: Path | str, *, target_tokens: int, log_every_steps: int) -> str:
    return f"""
[run]
id = "loop"
seed = 7
output_dir = "runs"

[model]
name = "decoder"
variant = "tiny"
vocab_size = 64
hidden_size = 8
intermediate_size = 16
num_layers = 1
num_heads = 2
n_kv_heads = 1
max_seq_len = 4
compute_dtype = "float32"

[optimizer]
name = "adamw"
weight_decay = 0.0

[optimizer.schedule]
name = "constant"
peak_lr = 0.001

[data]
train_manifest = "{Path(train_manifest).as_posix()}"
tokenizer_id = "toy-tokenizer"

[training]
seq_len = 4
global_batch_size = 2
target_tokens = {target_tokens}
log_every_steps = {log_every_steps}
checkpoint_every_steps = 10

[mesh]
axis_names = ["data"]
axis_sizes = [1]
"""


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]
