import subprocess
import sys
import json
from pathlib import Path

import pytest

import jaxtitan.cli as cli

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


def test_cli_data_prepare_human_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("jaxtitan.data.prepare.load_hf_texts", lambda source: ["hello", "world"])
    config_path = tmp_path / "prepare.toml"
    output = tmp_path / "prepared"
    config_path.write_text(_prepare_data_config(output), encoding="utf-8")

    code = cli.main(["data", "prepare", str(config_path)])
    captured = capsys.readouterr()

    assert code == 0
    assert "manifest:" in captured.out
    assert "tokens:" in captured.out
    assert "uv run jaxtitan data check" in captured.out
    assert "[data]" in captured.out
    assert 'order = "document_buffer"' in captured.out
    assert (output / "manifest.json").is_file()


def test_cli_data_prepare_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("jaxtitan.data.prepare.load_hf_texts", lambda source: ["hello"])
    config_path = tmp_path / "prepare.toml"
    config_path.write_text(_prepare_data_config(tmp_path / "prepared-json"), encoding="utf-8")

    code = cli.main(["data", "prepare", str(config_path), "--json"])
    captured = capsys.readouterr()

    payload = json.loads(captured.out)
    assert code == 0
    assert payload["manifest"]["tokenizer_id"] == "gpt2"
    assert payload["source"]["dataset"] == "HuggingFaceFW/fineweb"
    assert captured.err == ""


def test_cli_data_prepare_existing_output_fails_cleanly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    config_path = tmp_path / "prepare.toml"
    config_path.write_text(_prepare_data_config(output), encoding="utf-8")

    code = cli.main(["data", "prepare", str(config_path), "--json"])
    captured = capsys.readouterr()

    assert code == 2
    assert "already exists" in captured.err
    assert "Traceback" not in captured.err


def test_cli_data_prepare_bad_text_row_fails_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("jaxtitan.data.prepare.load_hf_texts", lambda source: [object()])
    config_path = tmp_path / "prepare.toml"
    config_path.write_text(_prepare_data_config(tmp_path / "bad-text"), encoding="utf-8")

    code = cli.main(["data", "prepare", str(config_path), "--json"])
    captured = capsys.readouterr()

    assert code == 2
    assert "text rows must be strings" in captured.err
    assert "Traceback" not in captured.err


def test_cli_data_inspect_human_output(
    tmp_path: Path,
    prepared_dataset: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = cli.main(["data", "inspect", str(prepared_dataset), "--tokenizer", "toy-tokenizer", "--seq-len", "4"])
    captured = capsys.readouterr()

    assert code == 0
    assert "manifest:" in captured.out
    assert "records: seq_len=4 train=1 val=0" in captured.out
    assert "training config:" in captured.out
    assert 'train_manifest = "' in captured.out


def test_cli_data_inspect_json(
    prepared_dataset: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = cli.main(["data", "inspect", str(prepared_dataset), "--tokenizer", "toy-tokenizer", "--json"])
    captured = capsys.readouterr()

    payload = json.loads(captured.out)
    assert code == 0
    assert payload["manifest"]["tokenizer_id"] == "toy-tokenizer"
    assert payload["data_config_toml"].startswith("[data]")
    assert captured.err == ""


def test_cli_data_inspect_invalid_seq_len_fails_cleanly(
    prepared_dataset: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = cli.main(["data", "inspect", str(prepared_dataset), "--seq-len", "0"])
    captured = capsys.readouterr()

    assert code == 2
    assert "seq-len" in captured.err
    assert "Traceback" not in captured.err


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


def _prepare_data_config(output: Path) -> str:
    return f"""
[source]
type = "hf"
dataset = "HuggingFaceFW/fineweb"
name = "sample-10BT"
split = "train"
text_column = "text"
streaming = true

[tokenizer]
name = "gpt2"
append_eot = true

[output]
path = "{output.as_posix()}"
max_tokens = 32
val_fraction = 0.25
shard_tokens = 8

[tokenization]
workers = 1
batch_docs = 2
queue_batches = 1
"""
