import json
import subprocess
import sys
from pathlib import Path

import pytest

from jaxtitan.errors import ContractError
from jaxtitan.runtime import run_training
from jaxtitan.runtime.checkpoint_eval import evaluate_checkpoint


def test_evaluate_latest_checkpoint_writes_posthoc_artifact_without_mutating_training_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("latest-eval", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=16, checkpoint_every_steps=1))
    run_training(config_path)
    run_dir = tmp_path / "runs" / "loop"
    metrics_before = (run_dir / "metrics" / "eval.jsonl").read_text()
    summary_before = (run_dir / "summaries" / "final.json").read_text()
    index_before = (run_dir / "checkpoints" / "index.json").read_text()

    payload = evaluate_checkpoint(run_dir, "latest")

    artifact = json.loads((run_dir / "evals" / "checkpoints" / "000002.json").read_text())
    training_eval = _jsonl(run_dir / "metrics" / "eval.jsonl")[-1]
    assert payload == artifact
    assert payload["status"] == "completed"
    assert payload["checkpoint"]["step"] == 2
    assert payload["checkpoint"]["path"] == "checkpoints/000002"
    assert payload["eval"]["loss"] == pytest.approx(training_eval["loss"])
    assert payload["eval"]["token_count"] == training_eval["token_count"]
    assert payload["data"]["manifest_sha256"]
    assert (run_dir / "metrics" / "eval.jsonl").read_text() == metrics_before
    assert (run_dir / "summaries" / "final.json").read_text() == summary_before
    assert (run_dir / "checkpoints" / "index.json").read_text() == index_before


def test_evaluate_best_and_explicit_checkpoint_selectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("selector-eval", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=16, checkpoint_every_steps=1))
    run_training(config_path)
    run_dir = tmp_path / "runs" / "loop"
    index = json.loads((run_dir / "checkpoints" / "index.json").read_text())

    best = evaluate_checkpoint(run_dir, "best")
    explicit = evaluate_checkpoint(run_dir, "000002")

    assert best["checkpoint"]["step"] == index["best_eval_step"]
    assert explicit["checkpoint"]["step"] == 2


def test_evaluate_checkpoint_rejects_invalid_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("bad-selector", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=8, checkpoint_every_steps=1))
    run_training(config_path)

    with pytest.raises(ContractError, match="checkpoint selector"):
        evaluate_checkpoint(tmp_path / "runs" / "loop", "middle")


def test_evaluate_checkpoint_rejects_missing_best_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("missing-best", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=8, checkpoint_every_steps=1))
    run_training(config_path)
    index_path = tmp_path / "runs" / "loop" / "checkpoints" / "index.json"
    index = json.loads(index_path.read_text())
    index["best_eval_step"] = None
    index["best_eval_loss"] = None
    index["best_checkpoint_path"] = None
    for record in index["records"]:
        record["eval_loss"] = None
    index_path.write_text(json.dumps(index, sort_keys=True))

    with pytest.raises(ContractError, match="best validation checkpoint"):
        evaluate_checkpoint(tmp_path / "runs" / "loop", "best")


def test_evaluate_checkpoint_rejects_unsupported_eval_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("unsupported-eval", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=8, checkpoint_every_steps=1))
    run_training(config_path)
    resolved_path = tmp_path / "runs" / "loop" / "config" / "resolved.json"
    resolved = json.loads(resolved_path.read_text())
    resolved["evals"][0]["name"] = "perplexity"
    resolved_path.write_text(json.dumps(resolved, sort_keys=True))

    with pytest.raises(ContractError, match="validation"):
        evaluate_checkpoint(tmp_path / "runs" / "loop", "latest")


def test_evaluate_checkpoint_rejects_compatibility_mismatch_before_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("compat-eval", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=8, checkpoint_every_steps=1))
    run_training(config_path)
    resolved_path = tmp_path / "runs" / "loop" / "config" / "resolved.json"
    resolved = json.loads(resolved_path.read_text())
    resolved["model"]["hidden_size"] = 16
    resolved_path.write_text(json.dumps(resolved, sort_keys=True))

    with pytest.raises(ContractError, match=r"compatibility\.model\.hidden_size"):
        evaluate_checkpoint(tmp_path / "runs" / "loop", "latest")


def test_cli_eval_checkpoint_json_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("cli-eval", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=8, checkpoint_every_steps=1))
    run_training(config_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jaxtitan.cli",
            "eval",
            "checkpoint",
            str(tmp_path / "runs" / "loop"),
            "--checkpoint",
            "latest",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["checkpoint"]["step"] == 1
    assert payload["eval"]["eval_name"] == "validation"
    assert (tmp_path / "runs" / "loop" / "evals" / "checkpoints" / "000001.json").is_file()


def _training_config(train_manifest: Path, *, target_tokens: int, checkpoint_every_steps: int) -> str:
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
axis_names = ["data"]
axis_sizes = [1]

[[evals]]
name = "validation"
every_steps = 1
num_batches = 1
"""


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]
