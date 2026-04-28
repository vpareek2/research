"""
End-to-end experiment runner.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from config import load_config
from utils.run_score import attach_score, select_baseline_summary
from utils.run_summary import (
    DEFAULT_REGISTRY_PATH,
    format_scorecard,
    register_summary,
    registry_record,
    summarize_run,
    write_summary_artifacts,
)
from utils.run_viz import DEFAULT_README_CHART_PATH, write_readme_chart, write_registry_charts


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
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = load_config(config_path)
    run_dir = Path(config.experiment.out_dir) / config.experiment.name

    _run([sys.executable, "-m", "pretrain", str(config_path)])
    _run([sys.executable, "-m", "utils.eval_checkpoint", str(run_dir)])
    _run([sys.executable, "-m", "utils.core_eval", str(run_dir)])

    summary = summarize_run(run_dir)
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


if __name__ == "__main__":
    main()
