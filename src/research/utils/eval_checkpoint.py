"""
Evaluate a saved training checkpoint on its configured validation split.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from flax import nnx
import jax

from research.checkpoint import create_checkpoint_manager, restore_model_checkpoint
from research.config import load_config
from research.data import (
    REQUIRED_EVAL_DOMAINS,
    domain_eval_steps,
    load_eval_domain_token_bytes,
    load_token_bytes,
    make_eval_domain_dataloaders,
    make_val_dataloader,
)
from research.distributed import create_distributed_context, place_replicated_model
from research.evals import LossEvalResult, evaluate_domain_losses, evaluate_loss
from research.model import Model
from research.utils.run_summary import summarize_and_write


def write_eval_artifacts(
    run_dir: Path,
    checkpoint_step: int,
    result: LossEvalResult,
    domains: dict[str, LossEvalResult] | None = None,
) -> tuple[Path, Path]:
    eval_dir = run_dir / "evals" / f"step_{checkpoint_step}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "run_dir": str(run_dir),
        "checkpoint_step": checkpoint_step,
        **result.to_dict(),
    }
    if domains is not None:
        metrics["domains"] = {name: domain_result.to_dict() for name, domain_result in domains.items()}
    metrics_path = eval_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")

    summary_path = eval_dir / "summary.md"
    summary_path.write_text(_format_eval_summary(run_dir, checkpoint_step, result, domains), encoding="utf-8")
    return metrics_path, summary_path


def _format_eval_summary(
    run_dir: Path,
    checkpoint_step: int,
    result: LossEvalResult,
    domains: dict[str, LossEvalResult] | None,
) -> str:
    lines = [
        "# Checkpoint Eval",
        "",
        f"- run: `{run_dir}`",
        f"- checkpoint_step: `{checkpoint_step}`",
        "",
        "## Native Validation",
        "",
        f"- eval_steps: `{result.eval_steps}`",
        f"- examples: `{result.examples}`",
        f"- tokens: `{result.tokens}`",
        f"- loss: `{result.loss:.6f}`",
        f"- ppl: `{result.ppl:.6f}`",
        f"- bpb: `{result.bpb:.6f}`",
        f"- bytes: `{result.bytes}`",
        f"- elapsed_sec: `{result.elapsed_sec:.6f}`",
        f"- tokens_per_sec: `{result.tokens_per_sec:.2f}`",
    ]
    if domains:
        lines.extend(
            [
                "",
                "## Domain Validation",
                "",
                f"{'domain':<12} {'loss':>10} {'ppl':>10} {'bpb':>10} {'tokens':>10}",
                "-" * 58,
            ]
        )
        for name in _ordered_domain_names(domains):
            domain_result = domains[name]
            lines.append(
                f"{name:<12} {domain_result.loss:>10.6f} {domain_result.ppl:>10.6f} "
                f"{domain_result.bpb:>10.6f} {domain_result.tokens:>10}"
            )
    return "\n".join(lines) + "\n"


def _ordered_domain_names(domains: dict[str, LossEvalResult]) -> list[str]:
    known = [name for name in REQUIRED_EVAL_DOMAINS if name in domains]
    extras = sorted(name for name in domains if name not in REQUIRED_EVAL_DOMAINS)
    return known + extras


def run_eval(run_dir: Path, *, step: int | None, eval_steps: int | None) -> tuple[int, LossEvalResult, dict[str, LossEvalResult]]:
    config = load_config(run_dir / "config.toml")
    train_config = config.train
    if eval_steps is not None:
        if eval_steps <= 0:
            raise ValueError(f"--eval-steps must be positive, got {eval_steps}")
        train_config = replace(train_config, eval_steps=eval_steps)

    distributed = create_distributed_context(config.distributed, train_config)
    model = Model(config.model, precision=config.precision, rngs=nnx.Rngs(train_config.seed))
    manager = create_checkpoint_manager(run_dir, train_config.keep_last)
    checkpoint_step = restore_model_checkpoint(manager, model=model, step=step)
    place_replicated_model(model, distributed)
    token_bytes = jax.device_put(load_token_bytes(config.data), distributed.replicated_sharding)
    domain_token_bytes = jax.device_put(load_eval_domain_token_bytes(config.eval, config.data), distributed.replicated_sharding)

    val_iter = make_val_dataloader(config.data, train_config)
    result = evaluate_loss(
        model,
        val_iter,
        train_config.eval_steps,
        distributed,
        tokens_per_example=train_config.seq_len,
        token_bytes=token_bytes,
    )
    domain_results = evaluate_domain_losses(
        model,
        make_eval_domain_dataloaders(config.eval, config.data, train_config),
        domain_eval_steps(config.eval, train_config),
        distributed,
        tokens_per_example=train_config.seq_len,
        token_bytes=domain_token_bytes,
    )
    return checkpoint_step, result, domain_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Run directory containing config.toml and checkpoints/.")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step to evaluate. Defaults to latest.")
    parser.add_argument("--eval-steps", type=int, default=None, help="Override config train.eval_steps for this eval.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoint_step, result, domains = run_eval(run_dir, step=args.step, eval_steps=args.eval_steps)
    metrics_path, summary_path = write_eval_artifacts(run_dir, checkpoint_step, result, domains)
    _, summary_json_path, scorecard_path, _ = summarize_and_write(run_dir)
    print(f"wrote {metrics_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {summary_json_path}")
    print(f"wrote {scorecard_path}")


if __name__ == "__main__":
    main()
