from pathlib import Path
from types import SimpleNamespace

from utils import experiment


def test_experiment_runs_train_evals_scores_and_registers(monkeypatch, tmp_path):
    commands = []
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")
    run_dir = tmp_path / "runs" / "exp"
    summary = {"run": {"name": "exp", "run_dir": str(run_dir)}}
    scored_summary = {**summary, "score": {"eligible": True, "final_score": 25.0}}

    monkeypatch.setattr(
        experiment,
        "load_config",
        lambda path: SimpleNamespace(experiment=SimpleNamespace(out_dir=str(tmp_path / "runs"), name="exp")),
    )
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

    assert [command[2] for command in commands] == ["pretrain", "utils.eval_checkpoint", "utils.core_eval"]
    assert commands[0][-1] == str(config_path)
    assert commands[1][-1] == str(run_dir)
    assert commands[2][-1] == str(run_dir)
