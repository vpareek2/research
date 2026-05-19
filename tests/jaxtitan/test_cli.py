import subprocess
import sys
import json
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


def test_cli_config_check_json(tmp_path: Path) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(MINIMAL_CONFIG)

    result = subprocess.run(
        [sys.executable, "-m", "jaxtitan.cli", "config", "check", str(config_path), "--json"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout)["run_id"] == "smoke"


def test_cli_run_init(tmp_path: Path, minimal_config: str) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(minimal_config)

    result = subprocess.run(
        [sys.executable, "-m", "jaxtitan.cli", "run", "init", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    run_dir = tmp_path / "runs" / "smoke"
    assert result.returncode == 0
    assert result.stdout.strip() == "runs/smoke"
    assert run_dir.is_dir()
    assert (run_dir / "manifest.json").is_file()


def test_cli_run_init_existing_dir_fails_cleanly(tmp_path: Path, minimal_config: str) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(minimal_config)
    (tmp_path / "runs" / "smoke").mkdir(parents=True)

    result = subprocess.run(
        [sys.executable, "-m", "jaxtitan.cli", "run", "init", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 2
    assert "run directory already exists" in result.stderr


def test_cli_run_preflight(tmp_path: Path, prepared_dataset_factory) -> None:
    manifest = prepared_dataset_factory(
        "preflight-cli",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(manifest))

    result = subprocess.run(
        [sys.executable, "-m", "jaxtitan.cli", "run", "preflight", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert "preflight: passed" in result.stdout
    assert "run: cli" in result.stdout
    assert not (tmp_path / "runs" / "cli").exists()


def test_cli_run_preflight_json(tmp_path: Path, prepared_dataset_factory) -> None:
    manifest = prepared_dataset_factory(
        "preflight-cli-json",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(manifest))

    result = subprocess.run(
        [sys.executable, "-m", "jaxtitan.cli", "run", "preflight", str(config_path), "--json"],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["status"] == "passed"
    assert payload["run_id"] == "cli"
    assert payload["training"]["compile"] == "passed"
    assert not (tmp_path / "runs" / "cli").exists()


def test_cli_missing_config_fails_cleanly(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "jaxtitan.cli", "config", "check", str(tmp_path / "missing.toml")],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "failed to read config" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_data_check_json(tmp_path: Path, prepared_dataset: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jaxtitan.cli",
            "data",
            "check",
            str(prepared_dataset),
            "--tokenizer",
            "toy-tokenizer",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["tokenizer_id"] == "toy-tokenizer"
    assert payload["num_tokens"] == 8


def test_cli_data_check_tokenizer_mismatch_fails_cleanly(tmp_path: Path, prepared_dataset: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jaxtitan.cli",
            "data",
            "check",
            str(prepared_dataset),
            "--tokenizer",
            "wrong-tokenizer",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 2
    assert "does not match config tokenizer" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_run_preflight_invalid_data_fails_cleanly(tmp_path: Path) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_preflight_config(tmp_path / "missing" / "manifest.json"))

    result = subprocess.run(
        [sys.executable, "-m", "jaxtitan.cli", "run", "preflight", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 2
    assert "manifest does not exist" in result.stderr
    assert "Traceback" not in result.stderr


def _preflight_config(train_manifest: Path) -> str:
    return f"""
[run]
id = "cli"
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
"""
