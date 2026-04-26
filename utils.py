"""
Small command-line utilities for run inspection.
"""

import argparse
import json
from pathlib import Path

import tiktoken

from config import load_config


def _load_batch_record(run_dir: Path, step: int) -> dict:
    batches_path = run_dir / "batches.jsonl"
    if not batches_path.exists():
        raise FileNotFoundError(f"Batch provenance file does not exist: {batches_path}")

    with batches_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["step"] == step:
                return record

    raise ValueError(f"No batch provenance found for step {step} in {batches_path}")


def _format_batch_text(run_dir: Path, step: int) -> str:
    config = load_config(run_dir / "config.toml")
    record = _load_batch_record(run_dir, step)

    text = Path(config.data.path).read_text(encoding="utf-8")
    tokenizer = tiktoken.get_encoding(config.data.tokenizer)
    tokens = tokenizer.encode(text)

    lines = [
        f"run: {run_dir}",
        f"step: {step}",
        f"batch_size: {len(record['chunk_idx'])}",
    ]

    for i, (chunk_idx, start, end) in enumerate(
        zip(record["chunk_idx"], record["token_start"], record["token_end"])
    ):
        decoded = tokenizer.decode(tokens[start:end])
        lines.extend(
            [
                "=" * 80,
                f"example: {i}",
                f"chunk_idx: {chunk_idx}",
                f"token_start: {start}",
                f"token_end: {end}",
                "-" * 80,
                decoded,
            ]
        )

    return "\n".join(lines) + "\n"


def inspect_batch(run_dir: str | Path, step: int, out_file: str | Path | None = None) -> Path:
    run_dir = Path(run_dir)
    if out_file is None:
        out_file = run_dir / "samples" / f"batch_step_{step:06d}.txt"
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(_format_batch_text(run_dir, step), encoding="utf-8")
    return out_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Path to a run directory.")
    parser.add_argument("step", type=int, help="Training step to inspect.")
    parser.add_argument(
        "out_file",
        nargs="?",
        help="Output text file. Defaults to <run_dir>/samples/batch_step_<step>.txt.",
    )
    args = parser.parse_args()

    out_file = inspect_batch(args.run_dir, args.step, args.out_file)
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
