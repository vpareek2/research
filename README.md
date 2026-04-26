# Random LM Research

A small JAX/Flax NNX language-model research repo focused on reproducible experiments, fast iteration, and easy inspection when training behavior looks strange.

## What It Does

- Trains a decoder-only language model.
- Uses JAX + Flax NNX for the model and training step.
- Uses Optax Muon for hidden 2D matrices, with vocab matrices routed to AdamW.
- Uses Grain for deterministic data loading and checkpointable train iteration.
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
- `[train]`: batch size, steps, learning rate, eval/checkpoint cadence
- `[data]`: local text path, tokenizer, validation split
- `[sampling]`: optional deterministic prompt sampling during eval
- `[wandb]`: optional online logging

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
uv run python -m py_compile config.py data.py model.py pretrain.py run.py checkpoint.py sample.py utils.py
```
