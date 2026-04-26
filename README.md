# LM Research

A small JAX/Flax NNX opinionated language-model research repo focused on reproducible experiments, fast iteration, and easy inspection when training behavior looks strange. Goal: Best model trained with only 2B tokens.

## What It Does

- Trains a decoder-only language model.
- Uses JAX + Flax NNX for the model and training step.
- Uses Optax Muon for hidden 2D matrices, with vocab matrices routed to AdamW.
- Uses Grain for deterministic data loading and checkpointable train iteration.
- Supports local text data and offline prepared token datasets from Hugging Face.
- Logs local run artifacts first: metrics, batch provenance, samples, configs, and checkpoints.
- Optionally mirrors scalar metrics and generated samples to W&B.

Local artifacts are the source of truth. W&B is only a dashboard and comparison layer.

## Setup

Install dependencies with uv:

```bash
uv sync
```

The smoke config expects Tiny Shakespeare at:

```text
data/tiny_shakespeare/input.txt
```

One way to download it:

```bash
mkdir -p data/tiny_shakespeare
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
  -o data/tiny_shakespeare/input.txt
```

## Configs

Experiment configs live in `configs/` so the repo root stays clean. The default smoke run is:

```text
configs/smoke.toml
```

Main sections:

- `[experiment]`: run name and output directory
- `[model]`: model size, heads, context length, RoPE theta
- `[train]`: batch size, steps, peak learning rate, eval/checkpoint cadence
- `[train.lr_schedule]`: optional LR schedule; defaults to linear warmup + cosine decay
- `[data]`: either raw text data or prepared token data
- `[sampling]`: optional deterministic prompt sampling during eval
- `[wandb]`: optional online logging

Learning-rate schedules are step-based. `train.lr` is the peak LR. If
`[train.lr_schedule]` is omitted, training uses cosine decay with
`warmup_ratio = 0.01` and `min_lr_ratio = 0.1`.

Cosine schedule:

```toml
[train.lr_schedule]
type = "cosine"
warmup_ratio = 0.01
min_lr_ratio = 0.1
```

Warmup-stable-decay schedule:

```toml
[train.lr_schedule]
type = "wsd"
warmup_ratio = 0.01
stable_ratio = 0.80
min_lr_ratio = 0.1
```

## Data

Raw text configs use a single local text file and split tokens deterministically:

```toml
[data]
source = "text"
path = "data/tiny_shakespeare/input.txt"
tokenizer = "gpt2"
val_fraction = 0.1
```

Prepared token configs read offline `.bin` files with `np.memmap`:

```toml
[data]
source = "tokens"
path = "data/tinystories_gpt2"
tokenizer = "gpt2"
```

Prepared token directories contain:

```text
data/<name>/
  tokens.bin
  manifest.json
```

To prepare a Hugging Face dataset once, then train fully offline:

```bash
uv run prepare-data configs/data/tinystories.toml
```

The prep step tokenizes the dataset once, inserts EOT between documents by default, writes one `uint32` `tokens.bin`, and records train/val split offsets plus source/tokenizer metadata in `manifest.json`. For HF datasets, `prepare-data` uses `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` if set, then falls back to saved Hugging Face Hub credentials from `hf auth login`, and only prompts if no token is available. Blank prompt input uses anonymous downloads. Tokens are never written to the manifest.

## Run Training

```bash
uv run pretrain configs/smoke.toml
```

Startup prints the run summary, initializes optional W&B logging, then compiles the first JAX step before printing the metrics table.

A run writes to:

```text
runs/<experiment.name>/
```

Important files:

- `config.toml`: copied config used for that run
- `metadata.json`: environment metadata
- `metrics.jsonl`: scalar metrics
- `batches.jsonl`: per-step batch provenance
- `samples/`: generated samples and inspected batch text
- `checkpoints/`: Orbax checkpoints with model, optimizer, metadata, and Grain iterator state

If the run directory already exists, startup fails by design. Use a new `[experiment].name` or resume.

## Count Parameters

Use `param-count` to size model configs before launching a run:

```bash
uv run param-count configs/smoke.toml
```

You can override individual dimensions from a config:

```bash
uv run param-count configs/smoke.toml --hidden-size 768 --layers 12 --heads 12 --kv-heads 4
```

Or provide the model shape directly:

```bash
uv run param-count \
  --vocab-size 50257 \
  --hidden-size 768 \
  --intermediate-size 2048 \
  --layers 12 \
  --heads 12 \
  --kv-heads 4 \
  --seq-len 2048
```

The report includes total parameters plus splits for token embeddings, LM head, attention, MLP, and norms.

## Plan Training Budget

Use `train-budget` to convert config batch shape into steps, tokens, and approximate epochs:

```bash
uv run train-budget configs/smoke.toml --tokens 2000000000
```

The utility always reports `tokens_per_step`, configured steps, and configured tokens. For prepared token datasets, it reads `manifest.json` and also reports train tokens, steps per epoch, usable epoch tokens, and configured/target epochs:

```bash
uv run train-budget configs/tinystories_smoke.toml
```

Raw text configs do not infer dataset token counts; pass `--tokens` when you only need target-step math.

## Resume

```bash
uv run pretrain configs/smoke.toml --resume
```

Resume restores the latest checkpoint and Grain iterator state, then continues from the saved next step.

## Inspect A Training Batch

Every training step logs token spans to `batches.jsonl`, so a suspicious metric spike can be traced back to the exact data seen at that step.

```bash
uv run inspect-batch runs/tiny_shakespeare_smoke 84
```

This writes:

```text
runs/tiny_shakespeare_smoke/samples/batch_step_000084.txt
```

You can also choose the output path:

```bash
uv run inspect-batch runs/tiny_shakespeare_smoke 84 runs/tiny_shakespeare_smoke/samples/my_batch.txt
```

## W&B

Enable W&B in a config file, for example `configs/smoke.toml`:

```toml
[wandb]
enabled = true
project = "data-research"
entity = ""
tags = []
```

If `WANDB_API_KEY` is set or W&B already has saved credentials, training starts without prompting. Otherwise, startup asks for the W&B API key.

W&B receives scalar metrics and a text table of generated samples. Local files are still written either way.

## Tests

```bash
uv run pytest -q
```

Compile check:

```bash
uv run python -m py_compile config.py data.py model.py prepare_data.py pretrain.py logs.py checkpoint.py sample.py inspect_batch.py param_count.py lr_schedule.py train_budget.py
```
