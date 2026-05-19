import json
import subprocess
import sys
from pathlib import Path

import pytest

from jaxtitan.errors import ContractError
from jaxtitan.runtime import run_training
from jaxtitan.services import initialize_run


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

    assert [event["type"] for event in events] == [
        "run_initialized",
        "training_started",
        "checkpoint_saved",
        "training_completed",
    ]
    assert events[-2]["step"] == 2
    assert events[-2]["reason"] == "final"
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
    assert (run_dir / "checkpoints" / "000002").is_dir()


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


def test_run_training_saves_interval_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "checkpoint-interval",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(manifest, target_tokens=16, log_every_steps=1, checkpoint_every_steps=1)
    )

    run_training(config_path)

    events = _jsonl(tmp_path / "runs" / "loop" / "events.jsonl")
    checkpoint_events = [event for event in events if event["type"] == "checkpoint_saved"]
    assert [event["step"] for event in checkpoint_events] == [1, 2]
    assert [event["reason"] for event in checkpoint_events] == ["interval", "interval"]
    assert (tmp_path / "runs" / "loop" / "checkpoints" / "000001").is_dir()
    assert (tmp_path / "runs" / "loop" / "checkpoints" / "000002").is_dir()


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


def test_run_training_resume_continues_from_latest_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "resume",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(manifest, target_tokens=8, log_every_steps=1, checkpoint_every_steps=1)
    )
    first = run_training(config_path)
    config_path.write_text(
        _training_config(manifest, target_tokens=16, log_every_steps=5, checkpoint_every_steps=2)
    )

    resumed = run_training(config_path, resume=True)

    run_dir = tmp_path / "runs" / "loop"
    events = _jsonl(run_dir / "events.jsonl")
    metrics = _jsonl(run_dir / "metrics" / "train.jsonl")
    assert first.steps == 1
    assert resumed.steps == 2
    assert resumed.tokens_seen == 16
    assert [row["step"] for row in metrics] == [1, 2]
    assert metrics[-1]["token_start"] == 8
    assert metrics[-1]["token_end"] == 16
    assert "training_resumed" in [event["type"] for event in events]
    resumed_event = next(event for event in events if event["type"] == "training_resumed")
    assert resumed_event["checkpoint_step"] == 1
    assert resumed_event["compat_checked"] is True
    assert resumed_event["runtime_fingerprint"]
    assert resumed_event["dataset_token_offset"] == 8
    assert (run_dir / "checkpoints" / "000002").is_dir()


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"hidden_size": 16}, r"compatibility\.model\.hidden_size"),
        ({"weight_decay": 0.2}, r"compatibility\.optimizer\.weight_decay"),
        ({"seq_len": 2}, r"compatibility\.training\.seq_len"),
        ({"global_batch_size": 1}, r"compatibility\.training\.global_batch_size"),
        ({"axis_sizes": (2,)}, r"compatibility\.mesh\.axis_sizes"),
        ({"seed": 9}, r"compatibility\.seed"),
    ],
)
def test_run_training_resume_rejects_unsafe_config_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
    change: dict,
    match: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "unsafe-resume",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(manifest, target_tokens=8, log_every_steps=1, checkpoint_every_steps=1)
    )
    run_training(config_path)
    config_path.write_text(
        _training_config(manifest, target_tokens=16, log_every_steps=1, checkpoint_every_steps=1, **change)
    )

    with pytest.raises(ContractError, match=match):
        run_training(config_path, resume=True)

    events = _jsonl(tmp_path / "runs" / "loop" / "events.jsonl")
    assert events[-2]["type"] == "checkpoint_restore_failed"
    assert events[-1]["type"] == "training_failed"


def test_run_training_resume_rejects_data_manifest_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    first_manifest = prepared_dataset_factory(
        "data-resume-first",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    second_manifest = prepared_dataset_factory(
        "data-resume-second",
        shard_token_groups=(tuple(range(10, 40)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(first_manifest, target_tokens=8, log_every_steps=1, checkpoint_every_steps=1)
    )
    run_training(config_path)
    config_path.write_text(
        _training_config(second_manifest, target_tokens=16, log_every_steps=1, checkpoint_every_steps=1)
    )

    with pytest.raises(ContractError, match=r"compatibility\.data\.train_manifest"):
        run_training(config_path, resume=True)


def test_run_training_resume_rejects_tokenizer_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "tokenizer-first",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(manifest, target_tokens=8, log_every_steps=1, checkpoint_every_steps=1)
    )
    run_training(config_path)
    manifest_json = json.loads(manifest.read_text())
    manifest_json["tokenizer"]["name"] = "other-tokenizer"
    manifest.write_text(json.dumps(manifest_json, sort_keys=True))
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=16,
            log_every_steps=1,
            checkpoint_every_steps=1,
            tokenizer_id="other-tokenizer",
        )
    )

    with pytest.raises(ContractError, match=r"compatibility\.data\.tokenizer_id"):
        run_training(config_path, resume=True)


def test_run_training_resume_rejects_auto_cosine_schedule_target_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "cosine-auto-resume",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=8,
            log_every_steps=1,
            checkpoint_every_steps=1,
            schedule_name="cosine",
        )
    )
    run_training(config_path)
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=16,
            log_every_steps=1,
            checkpoint_every_steps=1,
            schedule_name="cosine",
        )
    )

    with pytest.raises(ContractError, match=r"compatibility\.optimizer\.schedule\.total_steps"):
        run_training(config_path, resume=True)


def test_run_training_resume_allows_explicit_cosine_schedule_target_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "cosine-explicit-resume",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=8,
            log_every_steps=1,
            checkpoint_every_steps=1,
            schedule_name="cosine",
            total_steps=2,
        )
    )
    run_training(config_path)
    config_path.write_text(
        _training_config(
            manifest,
            target_tokens=16,
            log_every_steps=1,
            checkpoint_every_steps=1,
            schedule_name="cosine",
            total_steps=2,
        )
    )

    resumed = run_training(config_path, resume=True)

    assert resumed.steps == 2
    assert resumed.tokens_seen == 16


def test_run_training_resume_records_restore_failure_without_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory(
        "restore-failure",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=16, log_every_steps=1))
    initialize_run(config_path)

    with pytest.raises(ContractError, match="no checkpoints"):
        run_training(config_path, resume=True)

    events = _jsonl(tmp_path / "runs" / "loop" / "events.jsonl")
    assert [event["type"] for event in events] == [
        "run_initialized",
        "training_started",
        "checkpoint_restore_failed",
        "training_failed",
    ]
    assert events[1]["resume"] is True


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


def test_cli_run_train_resume_succeeds(
    tmp_path: Path,
    prepared_dataset_factory,
) -> None:
    manifest = prepared_dataset_factory(
        "cli-resume",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        _training_config(manifest, target_tokens=8, log_every_steps=1, checkpoint_every_steps=1)
    )
    first = subprocess.run(
        [sys.executable, "-m", "jaxtitan.cli", "run", "train", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    config_path.write_text(
        _training_config(manifest, target_tokens=16, log_every_steps=1, checkpoint_every_steps=1)
    )

    resumed = subprocess.run(
        [sys.executable, "-m", "jaxtitan.cli", "run", "train", "--resume", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    metrics = _jsonl(tmp_path / "runs" / "loop" / "metrics" / "train.jsonl")
    assert first.returncode == 0
    assert resumed.returncode == 0
    assert resumed.stdout.strip() == "runs/loop"
    assert [row["step"] for row in metrics] == [1, 2]


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


def _training_config(
    train_manifest: Path | str,
    *,
    target_tokens: int,
    log_every_steps: int,
    checkpoint_every_steps: int = 10,
    seed: int = 7,
    hidden_size: int = 8,
    weight_decay: float = 0.0,
    schedule_name: str = "constant",
    total_steps: int | None = None,
    tokenizer_id: str = "toy-tokenizer",
    seq_len: int = 4,
    global_batch_size: int = 2,
    axis_sizes: tuple[int, ...] = (1,),
) -> str:
    total_steps_line = "" if total_steps is None else f"total_steps = {total_steps}\n"
    return f"""
[run]
id = "loop"
seed = {seed}
output_dir = "runs"

[model]
name = "decoder"
variant = "tiny"
vocab_size = 64
hidden_size = {hidden_size}
intermediate_size = 16
num_layers = 1
num_heads = 2
n_kv_heads = 1
max_seq_len = 4
compute_dtype = "float32"

[optimizer]
name = "adamw"
weight_decay = {weight_decay}

[optimizer.schedule]
name = "{schedule_name}"
peak_lr = 0.001
{total_steps_line}

[data]
train_manifest = "{Path(train_manifest).as_posix()}"
tokenizer_id = "{tokenizer_id}"

[training]
seq_len = {seq_len}
global_batch_size = {global_batch_size}
target_tokens = {target_tokens}
log_every_steps = {log_every_steps}
checkpoint_every_steps = {checkpoint_every_steps}

[mesh]
axis_names = ["data"]
axis_sizes = [{", ".join(str(size) for size in axis_sizes)}]
"""


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]
