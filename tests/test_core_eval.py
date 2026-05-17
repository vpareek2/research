import json
import sys
import urllib.request

import jax.numpy as jnp
import pytest

from research.utils import core_eval, run_summary


def write_core_bundle(root):
    (root / "eval_data").mkdir(parents=True)
    (root / "core.yaml").write_text(
        """
icl_tasks:
  - label: toy_mc
    icl_task_type: multiple_choice
    dataset_uri: toy_mc.jsonl
    num_fewshot: [0]
    continuation_delimiter: " "
  - label: toy_lm
    icl_task_type: language_modeling
    dataset_uri: toy_lm.jsonl
    num_fewshot: [0]
    continuation_delimiter: " "
""",
        encoding="utf-8",
    )
    (root / "eval_meta_data.csv").write_text(
        "Eval Task,Random baseline\n"
        "toy_mc,50\n"
        "toy_lm,0\n",
        encoding="utf-8",
    )
    (root / "eval_data" / "toy_mc.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"query": "Q0", "choices": ["A", "B"], "gold": 0}),
                json.dumps({"query": "Q1", "choices": ["C", "D"], "gold": 1}),
                json.dumps({"query": "Q2", "choices": ["E", "F"], "gold": 0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "eval_data" / "toy_lm.jsonl").write_text(
        json.dumps({"context": "hello ", "continuation": "world"}) + "\n",
        encoding="utf-8",
    )


def write_run(run_dir):
    run_dir.mkdir(parents=True)
    (run_dir / "config.toml").write_text(
        f"""
[experiment]
name = "unit"
out_dir = "{run_dir.parent}"

[model]
vocab_size = 128
hidden_size = 32
intermediate_size = 64
n_layers = 1
n_heads = 4
n_kv_heads = 1
seq_len = 8
theta = 10000.0
eps = 0.000001
tied = false

[distributed]
enabled = false
device_count = "auto"
axis_name = "data"

[train]
seed = 0
batch_size = 2
seq_len = 8
steps = 2
log_every = 1
eval_every = 1
eval_steps = 1
checkpoint_every = 1
keep_last = 2

[optimizer]
name = "muon"
lr = 0.001
weight_decay = 0.1

[data]
source = "text"
path = "input.txt"
tokenizer = "gpt2"
val_fraction = 0.25
""",
        encoding="utf-8",
    )
    rows = [
        {"step": 0, "train/loss": 2.0, "val/loss": 2.5, "health/nan_count": 0},
        {"step": 1, "train/loss": 1.5, "val/loss": 2.0, "health/nan_count": 0, "train/tokens_seen": 32},
    ]
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_prompt_rendering_and_batch_spans():
    item = {"query": "Question?", "choices": [" yes", " no"], "gold": 1}
    prompts = core_eval.render_prompts_mc(item, " ")
    assert prompts == ["Question?  yes", "Question?  no"]

    tokenizer = core_eval.CoreTokenizer("gpt2")
    tokens, starts, ends = core_eval.batch_sequences_mc(tokenizer, prompts)
    assert starts == [core_eval.find_common_length(tokens, direction="left")] * 2
    assert ends == [len(tokens[0]), len(tokens[1])]

    schema = {"context_options": ["A", "B"], "continuation": " C", "gold": 0}
    schema_prompts = core_eval.render_prompts_schema(schema, " ")
    schema_tokens, schema_starts, schema_ends = core_eval.batch_sequences_schema(tokenizer, schema_prompts)
    suffix_length = core_eval.find_common_length(schema_tokens, direction="right")
    assert schema_starts == [end - suffix_length for end in schema_ends]

    lm = {"context": "hello  ", "continuation": "world"}
    without, with_continuation = core_eval.render_prompts_lm(lm, " ")
    assert without == "hello"
    assert with_continuation.endswith("world")


def test_crop_sequences_adjusts_continuation_spans():
    tokens, starts, ends = core_eval.crop_sequences([[1, 2, 3, 4, 5]], [3], [5], 4)
    assert tokens == [[2, 3, 4, 5]]
    assert starts == [2]
    assert ends == [4]


def test_crop_sequences_returns_none_when_continuation_is_cropped_away():
    assert core_eval.crop_sequences([[1, 2, 3, 4, 5]], [1], [2], 3) is None


def test_scoring_helpers_and_centered_formula():
    losses = jnp.asarray([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]])
    assert core_eval.score_multiple_choice(losses, [1, 1], [4, 4], 0)
    assert not core_eval.score_multiple_choice(losses, [1, 1], [4, 4], 1)

    input_ids = jnp.asarray([[10, 11, 12, 13]])
    predictions = jnp.asarray([[11, 12, 13, 0]])
    assert core_eval.score_language_modeling(input_ids, predictions, [1], [4])

    assert core_eval.centered_score(0.5, 25.0) == pytest.approx((0.5 - 0.25) / 0.75)


def test_bundle_loading_and_deterministic_subsample(tmp_path):
    bundle_dir = tmp_path / "eval_bundle"
    write_core_bundle(bundle_dir)

    tasks = core_eval.load_core_tasks(bundle_dir)
    assert [task.label for task in tasks] == ["toy_mc", "toy_lm"]
    assert tasks[0].random_baseline == 50.0

    first = core_eval.load_task_data(bundle_dir, tasks[0], max_per_task=2)
    second = core_eval.load_task_data(bundle_dir, tasks[0], max_per_task=2)
    assert first == second
    assert len(first) == 2


def test_ensure_eval_bundle_skips_existing_bundle(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "eval_bundle"
    bundle_dir.mkdir()
    (bundle_dir / "core.yaml").write_text("icl_tasks: []\n", encoding="utf-8")

    def fail_download(*args, **kwargs):
        raise AssertionError("download should not run")

    monkeypatch.setattr(urllib.request, "urlretrieve", fail_download)
    assert core_eval.ensure_eval_bundle(bundle_dir) == bundle_dir


def test_eval_core_cli_writes_artifacts_and_refreshes_summary(tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / "unit"
    write_run(run_dir)
    metrics = {
        "run_dir": str(run_dir),
        "checkpoint_step": 1,
        "max_per_task": 5,
        "core": 0.25,
        "tasks": {
            "toy_mc": {
                "accuracy": 0.5,
                "centered": 0.25,
                "random_baseline": 25.0,
                "examples": 5,
                "elapsed_sec": 0.1,
            }
        },
        "elapsed_sec": 0.1,
    }

    def fake_run_core_eval(run_path, *, step, max_per_task, bundle_dir, run_inference_bench):
        assert run_path == run_dir
        assert step == 1
        assert max_per_task == 5
        assert not run_inference_bench
        return 1, metrics, None

    monkeypatch.setattr(core_eval, "run_core_eval", fake_run_core_eval)
    monkeypatch.setattr(sys, "argv", ["eval-core", str(run_dir), "--step", "1", "--max-per-task", "5"])

    core_eval.main()

    eval_dir = run_dir / "evals" / "step_1"
    assert (eval_dir / "core_metrics.json").exists()
    assert (eval_dir / "core.csv").exists()
    assert (eval_dir / "core_summary.md").exists()
    assert (run_dir / "summary" / "run_summary.json").exists()
    summary = run_summary.summarize_run(run_dir)
    assert summary["benchmark_core"]["latest"]["core"] == 0.25


def test_eval_core_cli_inference_bench_flags(tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / "unit"
    write_run(run_dir)
    metrics = {
        "run_dir": str(run_dir),
        "checkpoint_step": 1,
        "max_per_task": -1,
        "core": 0.25,
        "tasks": {},
        "elapsed_sec": 0.1,
    }
    inference_metrics = {
        "run_dir": str(run_dir),
        "checkpoint_step": 1,
        "mode": "full_context_no_kv_cache",
        "batch_size": 1,
        "prompt_count": 3,
        "prompt_tokens": 4,
        "decode_tokens": 4,
        "seq_len": 8,
        "prefill_elapsed_sec": 0.1,
        "decode_elapsed_sec": 0.2,
        "ttft_sec": 0.01,
        "prefill_tokens_per_sec": 120.0,
        "decode_tokens_per_sec": 60.0,
        "memory_used_bytes": None,
        "memory_peak_bytes": None,
    }
    seen = []

    def fake_run_core_eval(run_path, *, step, max_per_task, bundle_dir, run_inference_bench):
        seen.append(run_inference_bench)
        return 1, metrics | {"max_per_task": max_per_task}, inference_metrics if run_inference_bench else None

    monkeypatch.setattr(core_eval, "run_core_eval", fake_run_core_eval)

    core_eval.main([str(run_dir)])
    assert seen[-1] is True
    assert (run_dir / "evals" / "step_1" / "inference_metrics.json").exists()

    (run_dir / "evals" / "step_1" / "inference_metrics.json").unlink()
    core_eval.main([str(run_dir), "--max-per-task", "5"])
    assert seen[-1] is False
    assert not (run_dir / "evals" / "step_1" / "inference_metrics.json").exists()

    core_eval.main([str(run_dir), "--max-per-task", "5", "--inference-bench"])
    assert seen[-1] is True
    assert (run_dir / "evals" / "step_1" / "inference_metrics.json").exists()

    (run_dir / "evals" / "step_1" / "inference_metrics.json").unlink()
    core_eval.main([str(run_dir), "--no-inference-bench"])
    assert seen[-1] is False
    assert not (run_dir / "evals" / "step_1" / "inference_metrics.json").exists()
