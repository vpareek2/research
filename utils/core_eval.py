"""
CORE benchmark evaluation for saved checkpoints.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
import urllib.request
import zipfile

from flax import nnx
import jax
import jax.numpy as jnp
from jinja2 import Template
import optax
import tiktoken
from tqdm import tqdm
import yaml

from checkpoint import create_checkpoint_manager, restore_model_checkpoint
from config import load_config
from distributed import create_distributed_context, place_replicated_model
from model import Model
from utils.inference_bench import benchmark_inference, write_inference_artifacts
from utils.run_summary import summarize_and_write


EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
DEFAULT_BUNDLE_DIR = Path("data") / "eval_bundle"


@dataclass(frozen=True)
class CoreTask:
    label: str
    task_type: str
    dataset_uri: str
    num_fewshot: int
    continuation_delimiter: str
    random_baseline: float


class CoreTokenizer:
    def __init__(self, name: str):
        self.encoding = tiktoken.get_encoding(name)
        self.bos_token_id = self.encoding.eot_token

    def __call__(self, texts: list[str], *, prepend: int | None = None) -> list[list[int]]:
        prefix = [] if prepend is None else [prepend]
        return [prefix + self.encoding.encode(text) for text in texts]

    def get_bos_token_id(self) -> int:
        return self.bos_token_id


def render_prompts_mc(item: dict, continuation_delimiter: str, fewshot_examples: list[dict] | None = None) -> list[str]:
    template = Template(
        """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}
{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}
""".strip()
    )
    context = {
        "fewshot_examples": fewshot_examples or [],
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    return [template.render(choice=choice, **context) for choice in item["choices"]]


def render_prompts_schema(item: dict, continuation_delimiter: str, fewshot_examples: list[dict] | None = None) -> list[str]:
    template = Template(
        """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}
{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}
""".strip()
    )
    context = {
        "fewshot_examples": fewshot_examples or [],
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    return [template.render(context=context_option, **context) for context_option in item["context_options"]]


def render_prompts_lm(item: dict, continuation_delimiter: str, fewshot_examples: list[dict] | None = None) -> list[str]:
    template = Template(
        """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}
{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}
""".strip()
    )
    context = {
        "fewshot_examples": fewshot_examples or [],
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    prompt_without = template.render(include_continuation=False, **context).strip()
    prompt_with = template.render(include_continuation=True, **context)
    return [prompt_without, prompt_with]


def find_common_length(token_sequences: list[list[int]], direction: str = "left") -> int:
    min_len = min(len(seq) for seq in token_sequences)
    indices = range(min_len) if direction == "left" else range(-1, -min_len - 1, -1)
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens: list[list[int]], pad_token_id: int, seq_len: int | None = None) -> jax.Array:
    seq_len = max(len(seq) for seq in tokens) if seq_len is None else seq_len
    input_ids = jnp.full((len(tokens), seq_len), pad_token_id, dtype=jnp.int32)
    for i, seq in enumerate(tokens):
        input_ids = input_ids.at[i, : len(seq)].set(jnp.asarray(seq, dtype=jnp.int32))
    return input_ids


def batch_sequences_mc(tokenizer: CoreTokenizer, prompts: list[str]) -> tuple[list[list[int]], list[int], list[int]]:
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    answer_start_idx = find_common_length(tokens, direction="left")
    return tokens, [answer_start_idx] * len(prompts), [len(seq) for seq in tokens]


def batch_sequences_schema(tokenizer: CoreTokenizer, prompts: list[str]) -> tuple[list[list[int]], list[int], list[int]]:
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    suffix_length = find_common_length(tokens, direction="right")
    end_indices = [len(seq) for seq in tokens]
    start_indices = [end_idx - suffix_length for end_idx in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer: CoreTokenizer, prompts: list[str]) -> tuple[list[list[int]], list[int], list[int]]:
    tokens_without, tokens_with = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert start_idx < end_idx, "prompt without is supposed to be a prefix of prompt with"
    assert tokens_without == tokens_with[:start_idx], "prompt without is supposed to be a prefix of prompt with"
    return [tokens_with], [start_idx], [end_idx]


def crop_sequences(
    tokens: list[list[int]],
    start_indices: list[int],
    end_indices: list[int],
    max_tokens: int,
) -> tuple[list[list[int]], list[int], list[int]] | None:
    new_tokens = []
    new_start_indices = []
    new_end_indices = []
    for seq, start_idx, end_idx in zip(tokens, start_indices, end_indices):
        if len(seq) > max_tokens:
            num_to_crop = len(seq) - max_tokens
            new_start_idx = start_idx - num_to_crop
            new_end_idx = end_idx - num_to_crop
            if new_start_idx < 1 or new_end_idx <= new_start_idx:
                return None
            new_tokens.append(seq[-max_tokens:])
            new_start_indices.append(new_start_idx)
            new_end_indices.append(new_end_idx)
        else:
            if start_idx < 1 or end_idx <= start_idx:
                return None
            new_tokens.append(seq)
            new_start_indices.append(start_idx)
            new_end_indices.append(end_idx)
    return new_tokens, new_start_indices, new_end_indices


@nnx.jit
def core_forward(model: Model, input_ids: jax.Array) -> tuple[jax.Array, jax.Array]:
    logits = model(input_ids)
    losses = optax.softmax_cross_entropy_with_integer_labels(
        logits[:, :-1, :].astype(model.loss_dtype),
        input_ids[:, 1:],
    )
    predictions = jnp.argmax(logits, axis=-1)
    return losses, predictions


def score_multiple_choice(losses: jax.Array, start_indices: list[int], end_indices: list[int], gold: int) -> bool:
    host_losses = jax.device_get(losses)
    mean_losses = [
        float(host_losses[i, start_idx - 1 : end_idx - 1].mean())
        for i, (start_idx, end_idx) in enumerate(zip(start_indices, end_indices))
    ]
    return mean_losses.index(min(mean_losses)) == gold


def score_language_modeling(
    input_ids: jax.Array,
    predictions: jax.Array,
    start_indices: list[int],
    end_indices: list[int],
) -> bool:
    start_idx = start_indices[0]
    end_idx = end_indices[0]
    host_input_ids = jax.device_get(input_ids)
    host_predictions = jax.device_get(predictions)
    predicted_tokens = host_predictions[0, start_idx - 1 : end_idx - 1]
    actual_tokens = host_input_ids[0, start_idx:end_idx]
    return bool((predicted_tokens == actual_tokens).all())


def evaluate_example(idx: int, model: Model, tokenizer: CoreTokenizer, data: list[dict], task: CoreTask, max_seq_len: int) -> bool | None:
    item = data[idx]
    fewshot_examples = []
    if task.num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_count = min(task.num_fewshot, len(available_indices))
        fewshot_indices = rng.sample(available_indices, fewshot_count)
        fewshot_examples = [data[i] for i in fewshot_indices]

    if task.task_type == "multiple_choice":
        prompts = render_prompts_mc(item, task.continuation_delimiter, fewshot_examples)
        tokens, start_indices, end_indices = batch_sequences_mc(tokenizer, prompts)
    elif task.task_type == "schema":
        prompts = render_prompts_schema(item, task.continuation_delimiter, fewshot_examples)
        tokens, start_indices, end_indices = batch_sequences_schema(tokenizer, prompts)
    elif task.task_type == "language_modeling":
        prompts = render_prompts_lm(item, task.continuation_delimiter, fewshot_examples)
        tokens, start_indices, end_indices = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported CORE task type: {task.task_type}")

    cropped = crop_sequences(tokens, start_indices, end_indices, max_seq_len)
    if cropped is None:
        return None
    tokens, start_indices, end_indices = cropped
    input_ids = stack_sequences(tokens, tokenizer.get_bos_token_id(), seq_len=max_seq_len)
    losses, predictions = core_forward(model, input_ids)

    if task.task_type == "language_modeling":
        return score_language_modeling(input_ids, predictions, start_indices, end_indices)
    return score_multiple_choice(losses, start_indices, end_indices, int(item["gold"]))


def centered_score(accuracy: float, random_baseline: float) -> float:
    return (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)


def load_core_tasks(bundle_dir: str | Path) -> list[CoreTask]:
    bundle_dir = Path(bundle_dir)
    config = yaml.safe_load((bundle_dir / "core.yaml").read_text(encoding="utf-8"))
    baselines = {}
    with (bundle_dir / "eval_meta_data.csv").open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            baselines[row["Eval Task"]] = float(row["Random baseline"])

    tasks = []
    for task in config["icl_tasks"]:
        label = task["label"]
        tasks.append(
            CoreTask(
                label=label,
                task_type=task["icl_task_type"],
                dataset_uri=task["dataset_uri"],
                num_fewshot=int(task["num_fewshot"][0]),
                continuation_delimiter=task.get("continuation_delimiter", " "),
                random_baseline=baselines[label],
            )
        )
    return tasks


def load_task_data(bundle_dir: str | Path, task: CoreTask, max_per_task: int = -1) -> list[dict]:
    data_path = Path(bundle_dir) / "eval_data" / task.dataset_uri
    data = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(1337)
    rng.shuffle(data)
    if max_per_task > 0:
        data = data[:max_per_task]
    return data


def evaluate_core(model: Model, tokenizer: CoreTokenizer, bundle_dir: str | Path, *, max_per_task: int, max_seq_len: int) -> dict:
    tasks = load_core_tasks(bundle_dir)
    task_results = {}
    start = time.perf_counter()
    for task in tqdm(tasks, desc="CORE tasks", unit="task"):
        task_start = time.perf_counter()
        data = load_task_data(bundle_dir, task, max_per_task=max_per_task)
        correct = 0
        scored = 0
        skipped = 0
        for idx in tqdm(range(len(data)), desc=task.label, unit="ex", leave=False):
            result = evaluate_example(idx, model, tokenizer, data, task, max_seq_len)
            if result is None:
                skipped += 1
                continue
            scored += 1
            correct += int(result)
        accuracy = correct / scored if scored else 0.0
        centered = centered_score(accuracy, task.random_baseline)
        task_results[task.label] = {
            "accuracy": accuracy,
            "centered": centered,
            "random_baseline": task.random_baseline,
            "examples": scored,
            "skipped_examples": skipped,
            "elapsed_sec": time.perf_counter() - task_start,
        }

    core = sum(task["centered"] for task in task_results.values()) / len(task_results) if task_results else 0.0
    return {
        "core": core,
        "tasks": task_results,
        "elapsed_sec": time.perf_counter() - start,
    }


def ensure_eval_bundle(bundle_dir: str | Path = DEFAULT_BUNDLE_DIR) -> Path:
    bundle_dir = Path(bundle_dir)
    if (bundle_dir / "core.yaml").exists():
        return bundle_dir

    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = bundle_dir.with_suffix(".lock")
    _acquire_lock(lock_path)
    try:
        if (bundle_dir / "core.yaml").exists():
            return bundle_dir
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            zip_path = tmp_path / "eval_bundle.zip"
            urllib.request.urlretrieve(EVAL_BUNDLE_URL, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmp_path)
            extracted = tmp_path / "eval_bundle"
            if bundle_dir.exists():
                shutil.rmtree(bundle_dir)
            shutil.move(str(extracted), bundle_dir)
    finally:
        _release_lock(lock_path)
    return bundle_dir


def _acquire_lock(lock_path: Path):
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return
        except FileExistsError:
            time.sleep(0.25)


def _release_lock(lock_path: Path):
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def write_core_artifacts(run_dir: Path, checkpoint_step: int, metrics: dict) -> tuple[Path, Path, Path]:
    eval_dir = run_dir / "evals" / f"step_{checkpoint_step}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = eval_dir / "core_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = eval_dir / "core.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Task", "Accuracy", "Centered", "Random baseline", "Examples", "Skipped", "Elapsed sec"])
        for task_name, task_metrics in metrics["tasks"].items():
            writer.writerow(
                [
                    task_name,
                    f"{task_metrics['accuracy']:.6f}",
                    f"{task_metrics['centered']:.6f}",
                    f"{task_metrics['random_baseline']:.6f}",
                    task_metrics["examples"],
                    task_metrics.get("skipped_examples", 0),
                    f"{task_metrics['elapsed_sec']:.6f}",
                ]
            )
        writer.writerow(["CORE", "", f"{metrics['core']:.6f}", "", "", "", f"{metrics['elapsed_sec']:.6f}"])

    summary_path = eval_dir / "core_summary.md"
    summary_path.write_text(format_core_summary(metrics), encoding="utf-8")
    return metrics_path, csv_path, summary_path


def format_core_summary(metrics: dict) -> str:
    lines = [
        "# CORE Eval",
        "",
        f"- run: `{metrics['run_dir']}`",
        f"- checkpoint_step: `{metrics['checkpoint_step']}`",
        f"- max_per_task: `{metrics['max_per_task']}`",
        f"- core: `{metrics['core']:.6f}`",
        f"- elapsed_sec: `{metrics['elapsed_sec']:.2f}`",
        "",
        f"{'task':<35} {'acc':>10} {'centered':>10} {'baseline':>10} {'examples':>10} {'skipped':>10}",
        "-" * 92,
    ]
    for task_name, task_metrics in metrics["tasks"].items():
        lines.append(
            f"{task_name:<35} "
            f"{task_metrics['accuracy']:>10.6f} "
            f"{task_metrics['centered']:>10.6f} "
            f"{task_metrics['random_baseline']:>10.2f} "
            f"{task_metrics['examples']:>10} "
            f"{task_metrics.get('skipped_examples', 0):>10}"
        )
    return "\n".join(lines) + "\n"


def run_core_eval(
    run_dir: Path,
    *,
    step: int | None,
    max_per_task: int,
    bundle_dir: str | Path,
    run_inference_bench: bool,
) -> tuple[int, dict, dict | None]:
    bundle_dir = ensure_eval_bundle(bundle_dir)
    config = load_config(run_dir / "config.toml")
    distributed = create_distributed_context(config.distributed, config.train)
    model = Model(config.model, precision=config.precision, rngs=nnx.Rngs(config.train.seed))
    manager = create_checkpoint_manager(run_dir, config.train.keep_last)
    checkpoint_step = restore_model_checkpoint(manager, model=model, step=step)
    place_replicated_model(model, distributed)

    tokenizer = CoreTokenizer(config.data.tokenizer)
    result = evaluate_core(
        model,
        tokenizer,
        bundle_dir,
        max_per_task=max_per_task,
        max_seq_len=config.model.seq_len,
    )
    metrics = {
        "run_dir": str(run_dir),
        "checkpoint_step": checkpoint_step,
        "max_per_task": max_per_task,
        **result,
    }
    inference_metrics = None
    if run_inference_bench:
        inference_metrics = benchmark_inference(
            model,
            tokenizer.encoding,
            config.model,
            run_dir=run_dir,
            checkpoint_step=checkpoint_step,
        )
    return checkpoint_step, metrics, inference_metrics


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on nanochat CORE.")
    parser.add_argument("run_dir", help="Run directory containing config.toml and checkpoints/.")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step to evaluate. Defaults to latest.")
    parser.add_argument("--max-per-task", type=int, default=-1, help="Max examples per CORE task. -1 evaluates all.")
    parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR), help="Path to nanochat eval_bundle directory.")
    inference_group = parser.add_mutually_exclusive_group()
    inference_group.add_argument("--inference-bench", action="store_true", help="Run inference benchmark even for partial CORE.")
    inference_group.add_argument("--no-inference-bench", action="store_true", help="Skip inference benchmark even for full CORE.")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    run_inference_bench = args.inference_bench or (args.max_per_task == -1 and not args.no_inference_bench)
    checkpoint_step, metrics, inference_metrics = run_core_eval(
        run_dir,
        step=args.step,
        max_per_task=args.max_per_task,
        bundle_dir=args.bundle_dir,
        run_inference_bench=run_inference_bench,
    )
    metrics_path, csv_path, core_summary_path = write_core_artifacts(run_dir, checkpoint_step, metrics)
    inference_paths = ()
    if inference_metrics is not None:
        inference_paths = write_inference_artifacts(run_dir, checkpoint_step, inference_metrics)
    _, summary_json_path, scorecard_path, _ = summarize_and_write(run_dir)
    print(f"wrote {metrics_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {core_summary_path}")
    for path in inference_paths:
        print(f"wrote {path}")
    print(f"wrote {summary_json_path}")
    print(f"wrote {scorecard_path}")


if __name__ == "__main__":
    main()
