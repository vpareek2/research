import json
import subprocess
import sys
from pathlib import Path

import pytest

from jaxtitan.errors import ContractError
from jaxtitan.runtime import run_training
from jaxtitan.runtime.sampling import sample_checkpoint


def test_sample_latest_checkpoint_writes_jsonl_without_mutating_training_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("latest-sample", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=16, checkpoint_every_steps=1))
    run_training(config_path)
    run_dir = tmp_path / "runs" / "loop"
    train_before = (run_dir / "metrics" / "train.jsonl").read_text()
    eval_before = (run_dir / "metrics" / "eval.jsonl").read_text()
    summary_before = (run_dir / "summaries" / "final.json").read_text()
    index_before = (run_dir / "checkpoints" / "index.json").read_text()

    payload = sample_checkpoint(run_dir, "latest", "1,2", max_new_tokens=2, top_k=1)

    artifact_path = run_dir / "samples" / "checkpoints" / "000002.jsonl"
    rows = _jsonl(artifact_path)
    assert rows == [payload]
    assert payload["status"] == "completed"
    assert payload["run_id"] == "loop"
    assert payload["checkpoint"]["selector"] == "latest"
    assert payload["checkpoint"]["step"] == 2
    assert payload["checkpoint"]["path"] == "checkpoints/000002"
    assert payload["model"]["vocab_size"] == 64
    assert payload["sampling"] == {"max_new_tokens": 2, "temperature": 1.0, "top_k": 1, "top_p": None}
    assert payload["prompt_ids"] == [1, 2]
    assert len(payload["generated_ids"]) == 2
    assert len(payload["full_ids"]) == 4
    assert len(payload["logprobs"]) == 2
    assert (run_dir / "metrics" / "train.jsonl").read_text() == train_before
    assert (run_dir / "metrics" / "eval.jsonl").read_text() == eval_before
    assert (run_dir / "summaries" / "final.json").read_text() == summary_before
    assert (run_dir / "checkpoints" / "index.json").read_text() == index_before


def test_sample_best_explicit_and_repeated_latest_are_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("selector-sample", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=16, checkpoint_every_steps=1))
    run_training(config_path)
    run_dir = tmp_path / "runs" / "loop"
    index = json.loads((run_dir / "checkpoints" / "index.json").read_text())

    latest_a = sample_checkpoint(run_dir, "latest", [1, 2], max_new_tokens=2, top_k=3)
    latest_b = sample_checkpoint(run_dir, "latest", [1, 2], max_new_tokens=2, top_k=3)
    best = sample_checkpoint(run_dir, "best", [1, 2], max_new_tokens=2, top_k=1)
    explicit = sample_checkpoint(run_dir, "000001", [1, 2], max_new_tokens=2, top_k=1)

    assert latest_a["checkpoint"]["step"] == 2
    assert latest_a["generated_ids"] == latest_b["generated_ids"]
    assert latest_a["logprobs"] == pytest.approx(latest_b["logprobs"])
    assert best["checkpoint"]["step"] == index["best_eval_step"]
    assert explicit["checkpoint"]["step"] == 1
    latest_rows = _jsonl(run_dir / "samples" / "checkpoints" / "000002.jsonl")
    assert len(latest_rows) >= 2


def test_sample_checkpoint_rejects_invalid_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("bad-sample", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=8, checkpoint_every_steps=1))
    run_training(config_path)
    run_dir = tmp_path / "runs" / "loop"

    with pytest.raises(ContractError, match="at least one"):
        sample_checkpoint(run_dir, "latest", "", max_new_tokens=1)
    with pytest.raises(ContractError, match="comma-separated integers"):
        sample_checkpoint(run_dir, "latest", "1,,2", max_new_tokens=1)
    with pytest.raises(ContractError, match="non-negative"):
        sample_checkpoint(run_dir, "latest", "-1", max_new_tokens=1)
    with pytest.raises(ContractError, match="outside model vocab_size"):
        sample_checkpoint(run_dir, "latest", "64", max_new_tokens=1)
    with pytest.raises(ContractError, match="top_k=65 exceeds vocab size"):
        sample_checkpoint(run_dir, "latest", "1", max_new_tokens=1, top_k=65)
    with pytest.raises(ContractError, match="max_seq_len"):
        sample_checkpoint(run_dir, "latest", "1,2,3", max_new_tokens=2, top_k=1)
    with pytest.raises(ContractError, match="checkpoint selector"):
        sample_checkpoint(run_dir, "middle", "1", max_new_tokens=1, top_k=1)


def test_sample_checkpoint_rejects_context_parallel_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("cp-sample", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=8, checkpoint_every_steps=1, context_parallel=True))
    run_training(config_path)

    with pytest.raises(ContractError, match="context-parallel"):
        sample_checkpoint(tmp_path / "runs" / "loop", "latest", "1,2", max_new_tokens=1, top_k=1)


def test_cli_sample_checkpoint_json_success_and_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("cli-sample", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=8, checkpoint_every_steps=1))
    run_training(config_path)
    run_dir = tmp_path / "runs" / "loop"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jaxtitan.cli",
            "sample",
            "checkpoint",
            str(run_dir),
            "--checkpoint",
            "latest",
            "--prompt-ids",
            "1,2",
            "--max-new-tokens",
            "2",
            "--top-k",
            "1",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    rows = _jsonl(run_dir / "samples" / "checkpoints" / "000001.jsonl")
    assert result.returncode == 0
    assert payload == rows[-1]
    assert payload["checkpoint"]["step"] == 1
    assert payload["generated_ids"]

    failure = subprocess.run(
        [
            sys.executable,
            "-m",
            "jaxtitan.cli",
            "sample",
            "checkpoint",
            str(run_dir),
            "--checkpoint",
            "latest",
            "--prompt-ids",
            "",
            "--max-new-tokens",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert failure.returncode == 2
    assert "prompt_ids must contain at least one token id" in failure.stderr
    assert "Traceback" not in failure.stderr


def _training_config(
    train_manifest: Path,
    *,
    target_tokens: int,
    checkpoint_every_steps: int,
    context_parallel: bool = False,
) -> str:
    axis_names = '["data", "cp"]' if context_parallel else '["data"]'
    axis_sizes = "[1, 2]" if context_parallel else "[1]"
    parallelism = "\n[parallelism]\ncontext_parallel = true\n" if context_parallel else ""
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
train_manifest = "{train_manifest.as_posix()}"
tokenizer_id = "toy-tokenizer"

[training]
seq_len = 4
global_batch_size = 2
target_tokens = {target_tokens}
log_every_steps = 1
checkpoint_every_steps = {checkpoint_every_steps}

[mesh]
axis_names = {axis_names}
axis_sizes = {axis_sizes}
{parallelism}

[[evals]]
name = "validation"
every_steps = 1
num_batches = 1
"""


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]
