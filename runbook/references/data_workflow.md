# Data Workflow Reference

Prepared manifests are the canonical training input for reproducible runs.

## Prepare HF Data

```sh
cd /home/veer/Master/projects/research
uv run jaxtitan data prepare --overwrite configs/data/tinystories_gpt2_smoke.toml
uv run jaxtitan data inspect data/tinystories_gpt2_smoke/manifest.json \
  --tokenizer gpt2 \
  --verify-checksums \
  --seq-len 512
uv run jaxtitan data check data/tinystories_gpt2_smoke/manifest.json \
  --tokenizer gpt2 \
  --verify-checksums
```

## Prepare Local Sources

Use `source.type = "parquet"`, `"jsonl"`, or `"text"` in a data config.
Training still consumes the generated manifest.

## Training Data Block

```toml
[data]
train_manifest = "data/<prepared>/manifest.json"
tokenizer_id = "gpt2"
order = "document_buffer"
shuffle_seed = 123
document_buffer_size = 8
document_refill_size = 8
```

## Streaming

HF streaming training is available but strict. Use it when local preparation is
not appropriate, and record the pinned revision/config in the stream.

Prepared data remains preferred for scored comparisons.
