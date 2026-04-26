"""
Training budget and epoch-step utilities.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from config import RunConfig, load_config
from data import load_validated_token_manifest


@dataclass(frozen=True)
class Budget:
    tokens_per_step: int
    configured_steps: int
    configured_tokens: int
    train_tokens: int | None
    steps_per_epoch: int | None
    usable_epoch_tokens: int | None
    target_tokens: int | None
    target_steps: int | None

    @property
    def configured_epochs(self) -> float | None:
        if self.train_tokens is None:
            return None
        return self.configured_tokens / self.train_tokens

    @property
    def target_epochs(self) -> float | None:
        if self.train_tokens is None or self.target_tokens is None:
            return None
        return self.target_tokens / self.train_tokens


def tokens_per_step(config: RunConfig) -> int:
    return config.train.batch_size * config.train.seq_len


def steps_for_tokens(token_count: int, step_tokens: int) -> int:
    if token_count < 0:
        raise ValueError(f"token_count must be non-negative, got {token_count}")
    if step_tokens <= 0:
        raise ValueError(f"step_tokens must be positive, got {step_tokens}")
    return math.ceil(token_count / step_tokens)


def steps_per_epoch(train_tokens: int, step_tokens: int) -> int:
    if train_tokens < 0:
        raise ValueError(f"train_tokens must be non-negative, got {train_tokens}")
    if step_tokens <= 0:
        raise ValueError(f"step_tokens must be positive, got {step_tokens}")
    return train_tokens // step_tokens


def infer_train_tokens(config: RunConfig) -> int | None:
    if config.data.source != "tokens":
        return None

    manifest = load_validated_token_manifest(config.data)
    train_split = manifest["splits"]["train"]
    return int(train_split.get("tokens", train_split["end"] - train_split["start"]))


def build_budget(config: RunConfig, *, target_tokens: int | None = None) -> Budget:
    step_tokens = tokens_per_step(config)
    train_tokens = infer_train_tokens(config)
    epoch_steps = steps_per_epoch(train_tokens, step_tokens) if train_tokens is not None else None
    target_steps = steps_for_tokens(target_tokens, step_tokens) if target_tokens is not None else None
    return Budget(
        tokens_per_step=step_tokens,
        configured_steps=config.train.steps,
        configured_tokens=config.train.steps * step_tokens,
        train_tokens=train_tokens,
        steps_per_epoch=epoch_steps,
        usable_epoch_tokens=epoch_steps * step_tokens if epoch_steps is not None else None,
        target_tokens=target_tokens,
        target_steps=target_steps,
    )


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,}"


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"


def format_budget(config: RunConfig, budget: Budget) -> str:
    lines = [
        "Training Budget",
        f"  config:            {config.experiment.name}",
        f"  batch_size:        {_fmt_int(config.train.batch_size)}",
        f"  seq_len:           {_fmt_int(config.train.seq_len)}",
        f"  tokens_per_step:   {_fmt_int(budget.tokens_per_step)}",
        "",
        f"configured_steps:    {_fmt_int(budget.configured_steps)}",
        f"configured_tokens:   {_fmt_int(budget.configured_tokens)}",
        f"configured_epochs:   {_fmt_float(budget.configured_epochs)}",
    ]

    if budget.target_tokens is not None:
        lines.extend(
            [
                "",
                f"target_tokens:       {_fmt_int(budget.target_tokens)}",
                f"target_steps:        {_fmt_int(budget.target_steps)}",
                f"target_epochs:       {_fmt_float(budget.target_epochs)}",
            ]
        )

    if budget.train_tokens is not None:
        lines.extend(
            [
                "",
                f"train_tokens:        {_fmt_int(budget.train_tokens)}",
                f"steps_per_epoch:     {_fmt_int(budget.steps_per_epoch)}",
                f"usable_epoch_tokens: {_fmt_int(budget.usable_epoch_tokens)}",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "train_tokens:        n/a (only inferred for prepared token datasets)",
                "steps_per_epoch:     n/a",
            ]
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compute training steps, tokens, and epochs from a run config.")
    parser.add_argument("config", help="Path to a run config TOML file.")
    parser.add_argument("--tokens", type=int, help="Optional target token count to convert into training steps.")
    args = parser.parse_args()

    config = load_config(args.config)
    budget = build_budget(config, target_tokens=args.tokens)
    print(format_budget(config, budget))


if __name__ == "__main__":
    main()
