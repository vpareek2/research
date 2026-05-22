# Jaxtitan Data

Jaxtitan trains from explicit prepared-token manifests. The usual flow is:

1. Prepare a Hugging Face or local text dataset into local token shards.
2. Inspect and validate the manifest.
3. Paste the emitted `[data]` block into a training config.
4. Run preflight, then train.

## Prepare

```bash
uv run jaxtitan data prepare configs/data/tinystories_gpt2_smoke.toml
```

Use `--overwrite` to replace an existing output directory:

```bash
uv run jaxtitan data prepare configs/data/tinystories_gpt2_smoke.toml --overwrite
```

Preparation writes:

- `manifest.json`
- `tokens-00000.bin`, etc.
- `token_bytes.bin`
- `document_offsets.u64`

The command also prints the exact `data check` command and a paste-ready `[data]` block.

Supported sources:

```toml
[source]
type = "hf"
dataset = "roneneldan/TinyStories"
split = "train"
text_column = "text"
streaming = true
```

```toml
[source]
type = "parquet"
paths = ["~/datasets/my_corpus/*.parquet"]
text_column = "text"
```

```toml
[source]
type = "jsonl"
paths = ["~/datasets/my_corpus/*.jsonl"]
text_column = "text"
```

```toml
[source]
type = "text"
paths = ["~/datasets/my_corpus/*.txt"]
```

For local sources, path globs are expanded deterministically and one parquet/jsonl row or text line is one document. Local source preparation is still offline preparation; it does not enable runtime streaming training.

## Inspect

```bash
uv run jaxtitan data inspect data/tinystories_gpt2_smoke/manifest.json --tokenizer gpt2 --seq-len 512
```

Use JSON for scripts:

```bash
uv run jaxtitan data inspect data/tinystories_gpt2_smoke/manifest.json --tokenizer gpt2 --seq-len 512 --json
```

## Validate

```bash
uv run jaxtitan data check data/tinystories_gpt2_smoke/manifest.json --tokenizer gpt2 --verify-checksums
```

## Train Config

Prepared manifests are the training input. Training does not silently prepare data.

```toml
[data]
train_manifest = "data/tinystories_gpt2_smoke/manifest.json"
tokenizer_id = "gpt2"
order = "document_buffer"
shuffle_seed = 123
document_buffer_size = 8
document_refill_size = 8
```

Then run:

```bash
uv run jaxtitan run preflight configs/jaxtitan/your_run.toml
uv run jaxtitan run train configs/jaxtitan/your_run.toml
```

`source.streaming = true` in a data-prepare config means Hugging Face streaming during offline preparation. Runtime streaming training is a separate future data pipeline.
