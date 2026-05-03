"""
End-to-end experiment runner.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from research.config import load_config
from research.preflight import prepare_missing_artifacts, run_preflight
from research.utils.run_score import attach_score, select_baseline_summary
from research.utils.run_summary import (
    DEFAULT_REGISTRY_PATH,
    format_scorecard,
    register_summary,
    registry_record,
    summarize_run,
    write_summary_artifacts,
)
from research.utils.run_viz import DEFAULT_README_CHART_PATH, write_readme_chart, write_registry_charts


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run pretraining, evals, scoring, registry, and charts.")
    parser.add_argument("config", help="Training config TOML.")
    parser.add_argument("--registry-path", default=str(DEFAULT_REGISTRY_PATH), help="Registry JSONL path.")
    parser.add_argument("--chart-path", default=None, help="Output HTML chart path. Defaults beside registry.")
    parser.add_argument(
        "--readme-chart-path",
        default=str(DEFAULT_README_CHART_PATH),
        help="Tracked SVG chart path for the README.",
    )
    parser.add_argument("--baseline-run", default=None, help="Optional baseline run directory for relative scoring.")
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Skip preflight, training, and eval; score/register existing run artifacts.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = load_config(config_path)
    run_dir = Path(config.experiment.out_dir) / config.experiment.name

    if not args.finalize_only:
        preflight = run_preflight(config_path)
        prepare_missing_artifacts(preflight)
        run_preflight(config_path, require_ready=True)

        _run([sys.executable, "-m", "research.pretrain", str(config_path)])
        _run([sys.executable, "-m", "research.utils.eval_checkpoint", str(run_dir)])
        _run([sys.executable, "-m", "research.utils.core_eval", str(run_dir)])

    summary = summarize_run(run_dir)
    validate_final_eval_alignment(summary)
    baseline = select_baseline_summary(
        summary,
        registry_path=args.registry_path,
        baseline_run=args.baseline_run,
    )
    summary = attach_score(summary, baseline)
    summary["registry_record"] = registry_record(summary)
    markdown = format_scorecard(summary)
    json_path, md_path = write_summary_artifacts(run_dir, summary, markdown)
    registry_path = register_summary(summary, args.registry_path)
    chart_path = write_registry_charts(registry_path, args.chart_path)
    readme_chart_path = write_readme_chart(registry_path, args.readme_chart_path)

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {registry_path}")
    print(f"wrote {chart_path}")
    print(f"wrote {readme_chart_path}")


def _run(command: list[str]) -> None:
    print(f"running: {' '.join(command)}")
    subprocess.run(command, check=True)


def validate_final_eval_alignment(summary: dict) -> None:
    expected_step = _nested(summary, ["training", "steps_completed"])
    artifacts = {
        "checkpoint eval": _nested(summary, ["checkpoint_evals", "latest", "checkpoint_step"]),
        "CORE eval": _nested(summary, ["benchmark_core", "latest", "checkpoint_step"]),
        "inference benchmark": _nested(summary, ["inference_benchmark", "latest", "checkpoint_step"]),
    }
    stale = {
        name: step
        for name, step in artifacts.items()
        if step != expected_step
    }
    if stale:
        details = ", ".join(
            f"{name}={step if step is not None else 'missing'}"
            for name, step in stale.items()
        )
        raise RuntimeError(
            f"Refusing to score run with stale eval artifacts: expected checkpoint step "
            f"{expected_step}, got {details}"
        )


def _nested(row: dict, path: list[str]):
    value = row
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


if __name__ == "__main__":
    main()
