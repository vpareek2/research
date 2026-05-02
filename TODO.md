# TODO

## 1. Replace slow, monolithic data preparation

Current dataset preparation is not acceptable for real cloud iteration.

Observed failure mode:
- `experiment` was allowed to start cloud prep with a FineWeb source that would have tokenized the full `sample-10BT` split.
- `target.tokens` originally controlled training length but not dataset preparation length.
- Even after adding `output.max_tokens`, prep is still single-process and monolithic:
  - HF streaming iterator
  - single-process `tiktoken`
  - one huge `tokens.bin`
  - full-file hash after writing

Required direction:
- Sharded prepared datasets, not one huge file.
- Parallel tokenization.
- Explicit token budget/cap required for HF sources.
- Preflight must reject uncapped or obviously excessive prep.
- Manifest should record source, cap, shard list, token counts, hashes, and train/val split.
- Training dataloader should read shards cleanly.
- Cloud UX should make data prep time predictable before launch.

Target UX:

```bash
uv run preflight configs/0_baseline.toml
uv run experiment configs/0_baseline.toml
```

That flow must never silently tokenize far more data than the experiment target requires.
