"""
Run registry CLI helpers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from utils.run_summary import DEFAULT_REGISTRY_PATH, register_summary, summarize_and_write


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Register a summarized run.")
    parser.add_argument("run_dir", help="Run directory containing config.toml and metrics.jsonl.")
    parser.add_argument("--registry-path", default=str(DEFAULT_REGISTRY_PATH), help="Registry JSONL path.")
    args = parser.parse_args(argv)

    summary, json_path, md_path, _ = summarize_and_write(args.run_dir)
    registry_path = register_summary(summary, args.registry_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {registry_path}")


def list_main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="List registered runs.")
    parser.add_argument("--registry-path", default=str(DEFAULT_REGISTRY_PATH), help="Registry JSONL path.")
    args = parser.parse_args(argv)

    rows = _load_registry(Path(args.registry_path))
    print(_format_registry(rows))


def _load_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _format_registry(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "no registered runs"

    header = (
        f"{'run':<28} {'status':<10} {'tokens':>12} {'best_val':>10} "
        f"{'ckpt_loss':>10} {'domain':>10} {'core':>8} {'mfu':>8} {'tok/gpu-hr':>12}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row.get('run_name', ''):<28} "
            f"{row.get('status', ''):<10} "
            f"{_fmt(row.get('tokens_seen')):>12} "
            f"{_fmt(row.get('best_val_loss')):>10} "
            f"{_fmt(row.get('latest_checkpoint_loss')):>10} "
            f"{_fmt(row.get('latest_domain_mean_loss')):>10} "
            f"{_fmt(row.get('latest_core')):>8} "
            f"{_fmt(row.get('avg_mfu')):>8} "
            f"{_fmt(row.get('train_tokens_per_gpu_hour')):>12}"
        )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if abs(value) >= 1000.0:
            return f"{value:.0f}"
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
