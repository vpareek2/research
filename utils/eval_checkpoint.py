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

from checkpoint import create_checkpoint_manager, restore_model_checkpoint
from config import load_config
from data import load_token_bytes, make_val_dataloader
from distributed import create_distributed_context, place_replicated_model
from evals import LossEvalResult, evaluate_loss
from model import Model


def write_eval_artifacts(run_dir: Path, checkpoint_step: int, result: LossEvalResult) -> tuple[Path, Path]:
    eval_dir = run_dir / "evals" / f"step_{checkpoint_step}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "run_dir": str(run_dir),
        "checkpoint_step": checkpoint_step,
        **result.to_dict(),
    }
    metrics_path = eval_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")

    summary_path = eval_dir / "summary.md"
    summary_path.write_text(
        "# Checkpoint Eval\n\n"
        f"- run: `{run_dir}`\n"
        f"- checkpoint_step: `{checkpoint_step}`\n"
        f"- eval_steps: `{result.eval_steps}`\n"
        f"- examples: `{result.examples}`\n"
        f"- tokens: `{result.tokens}`\n"
        f"- loss: `{result.loss:.6f}`\n"
        f"- ppl: `{result.ppl:.6f}`\n"
        f"- bpb: `{result.bpb:.6f}`\n"
        f"- bytes: `{result.bytes}`\n"
        f"- elapsed_sec: `{result.elapsed_sec:.6f}`\n"
        f"- tokens_per_sec: `{result.tokens_per_sec:.2f}`\n",
        encoding="utf-8",
    )
    return metrics_path, summary_path


def run_eval(run_dir: Path, *, step: int | None, eval_steps: int | None) -> tuple[int, LossEvalResult]:
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

    val_iter = make_val_dataloader(config.data, train_config)
    result = evaluate_loss(
        model,
        val_iter,
        train_config.eval_steps,
        distributed,
        tokens_per_example=train_config.seq_len,
        token_bytes=token_bytes,
    )
    return checkpoint_step, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Run directory containing config.toml and checkpoints/.")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step to evaluate. Defaults to latest.")
    parser.add_argument("--eval-steps", type=int, default=None, help="Override config train.eval_steps for this eval.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoint_step, result = run_eval(run_dir, step=args.step, eval_steps=args.eval_steps)
    metrics_path, summary_path = write_eval_artifacts(run_dir, checkpoint_step, result)
    print(f"wrote {metrics_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
