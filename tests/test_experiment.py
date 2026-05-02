from pathlib import Path
from types import SimpleNamespace

import pytest

from research.utils import experiment


def test_experiment_runs_train_evals_scores_and_registers(monkeypatch, tmp_path):
    commands = []
    preflight_calls = []
    prepare_calls = []
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")
    run_dir = tmp_path / "runs" / "exp"
    summary = aligned_summary(run_dir)
    scored_summary = {**summary, "score": {"eligible": True, "final_score": 25.0}}

    monkeypatch.setattr(
        experiment,
        "load_config",
        lambda path: SimpleNamespace(experiment=SimpleNamespace(out_dir=str(tmp_path / "runs"), name="exp")),
    )
    monkeypatch.setattr(experiment, "run_preflight", lambda path, **kwargs: preflight_calls.append((path, kwargs)) or "preflight")
    monkeypatch.setattr(experiment, "prepare_missing_artifacts", lambda preflight: prepare_calls.append(preflight))
    monkeypatch.setattr(experiment, "_run", lambda command: commands.append(command))
    monkeypatch.setattr(experiment, "summarize_run", lambda path: summary)
    monkeypatch.setattr(experiment, "select_baseline_summary", lambda current, **kwargs: current)
    monkeypatch.setattr(experiment, "attach_score", lambda current, baseline: scored_summary)
    monkeypatch.setattr(experiment, "registry_record", lambda current: {"run_name": "exp"})
    monkeypatch.setattr(experiment, "format_scorecard", lambda current: "# scorecard\n")
    monkeypatch.setattr(
        experiment,
        "write_summary_artifacts",
        lambda path, current, markdown: (Path(path) / "summary" / "run_summary.json", Path(path) / "summary" / "scorecard.md"),
    )
    registry_path = tmp_path / "registry.jsonl"
    chart_path = tmp_path / "registry.html"
    readme_chart_path = tmp_path / "chart.svg"
    monkeypatch.setattr(experiment, "register_summary", lambda current, path: registry_path)
    monkeypatch.setattr(experiment, "write_registry_charts", lambda registry, chart=None: chart_path)
    monkeypatch.setattr(experiment, "write_readme_chart", lambda registry, chart=None: readme_chart_path)

    experiment.main([str(config_path), "--registry-path", str(registry_path)])

    assert preflight_calls == [
        (config_path, {}),
        (config_path, {"require_ready": True}),
    ]
    assert prepare_calls == ["preflight"]
    assert [command[2] for command in commands] == ["research.pretrain", "research.utils.eval_checkpoint", "research.utils.core_eval"]
    assert commands[0][-1] == str(config_path)
    assert commands[1][-1] == str(run_dir)
    assert commands[2][-1] == str(run_dir)


def aligned_summary(run_dir, *, step=4):
    return {
        "run": {"name": "exp", "run_dir": str(run_dir)},
        "training": {"steps_completed": step},
        "checkpoint_evals": {"latest": {"checkpoint_step": step}},
        "benchmark_core": {"latest": {"checkpoint_step": step}},
        "inference_benchmark": {"latest": {"checkpoint_step": step}},
    }


def test_validate_final_eval_alignment_accepts_matching_final_artifacts(tmp_path):
    experiment.validate_final_eval_alignment(aligned_summary(tmp_path / "run", step=7))


@pytest.mark.parametrize(
    ("path", "label"),
    [
        (["checkpoint_evals", "latest", "checkpoint_step"], "checkpoint eval"),
        (["benchmark_core", "latest", "checkpoint_step"], "CORE eval"),
        (["inference_benchmark", "latest", "checkpoint_step"], "inference benchmark"),
    ],
)
def test_validate_final_eval_alignment_rejects_stale_artifacts(tmp_path, path, label):
    summary = aligned_summary(tmp_path / "run", step=7)
    target = summary
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = 5

    with pytest.raises(RuntimeError, match=f"expected checkpoint step 7.*{label}=5"):
        experiment.validate_final_eval_alignment(summary)


def test_validate_final_eval_alignment_rejects_missing_artifacts(tmp_path):
    summary = aligned_summary(tmp_path / "run", step=7)
    summary["inference_benchmark"] = {"latest": None}

    with pytest.raises(RuntimeError, match="inference benchmark=missing"):
        experiment.validate_final_eval_alignment(summary)
