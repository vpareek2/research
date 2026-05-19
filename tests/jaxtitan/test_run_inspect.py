import json
import subprocess
import sys
from pathlib import Path

import pytest

from jaxtitan.errors import ContractError
from jaxtitan.runtime import run_training
from jaxtitan.runtime.inspect import format_run_inspection, inspect_run, run_inspection_to_json


def test_inspect_run_reports_summary_checkpoints_and_recent_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("inspect", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest))
    run_training(config_path)

    inspection = inspect_run(tmp_path / "runs" / "loop")
    payload = inspection.payload
    text = format_run_inspection(inspection)

    assert payload["run_id"] == "loop"
    assert payload["status"] == "completed"
    assert payload["final"]["step"] == 2
    assert payload["latest_checkpoint"]["checkpoint_path"] == "checkpoints/000002"
    assert payload["best_checkpoint"]["eval_loss"] is not None
    assert payload["checkpoints"][0]["retained"] is True
    assert payload["recent_train_metrics"][-1]["step"] == 2
    assert payload["recent_eval_metrics"][-1]["eval_name"] == "validation"
    assert "run: loop" in text
    assert "best checkpoint:" in text
    assert json.loads(run_inspection_to_json(inspection))["run_id"] == "loop"


def test_inspect_run_missing_required_artifact_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("missing-index", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest))
    run_training(config_path)
    (tmp_path / "runs" / "loop" / "checkpoints" / "index.json").unlink()

    with pytest.raises(ContractError, match="missing checkpoint index"):
        inspect_run(tmp_path / "runs" / "loop")


def test_inspect_run_missing_retained_checkpoint_path_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("stale-index", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest))
    run_training(config_path)
    checkpoint_dir = tmp_path / "runs" / "loop" / "checkpoints" / "000002"
    for path in sorted(checkpoint_dir.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    checkpoint_dir.rmdir()

    with pytest.raises(ContractError, match="retained path does not exist"):
        inspect_run(tmp_path / "runs" / "loop")


def test_cli_run_inspect_json_does_not_import_jax(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "fake"
    (run_dir / "summaries").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "metrics").mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": "fake"}))
    (run_dir / "summaries" / "final.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "steps": 0,
                "tokens_seen": 0,
                "target_tokens": 0,
                "final_loss": None,
                "final_eval_loss": None,
                "final_eval_token_count": None,
                "final_eval_num_batches": None,
            }
        )
    )
    (run_dir / "checkpoints" / "index.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "latest_step": None,
                "latest_checkpoint_path": None,
                "best_eval_step": None,
                "best_eval_loss": None,
                "best_checkpoint_path": None,
                "records": [],
            }
        )
    )
    script = (
        "import sys\n"
        "from jaxtitan.cli import main\n"
        f"code = main(['run', 'inspect', {str(run_dir)!r}, '--json'])\n"
        "print('JAX_LOADED', 'jax' in sys.modules)\n"
        "raise SystemExit(code)\n"
    )

    result = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)

    assert result.returncode == 0
    assert '"run_id":"fake"' in result.stdout
    assert "JAX_LOADED False" in result.stdout


def _training_config(train_manifest: Path) -> str:
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
target_tokens = 16
log_every_steps = 1
checkpoint_every_steps = 10

[mesh]
axis_names = ["data"]
axis_sizes = [1]

[[evals]]
name = "validation"
every_steps = 1
num_batches = 1
"""
