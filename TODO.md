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

## 2. Fix MFU and throughput accounting

Status: implemented. `metrics.jsonl` remains raw; `run_summary`, scorecard, and
registry now derive corrected steady-state and wall-clock throughput/MFU fields.

The old scorecard reported `avg_mfu = 1.85%` for `0_baseline`, but the run-level wall-clock estimate is much higher:

- configured tokens: `2,000,027,648`
- final elapsed: `26,970.5s`
- wall throughput: about `74.2k tokens/sec`
- wall MFU using current code peak (`419 TFLOP/s`): about `12.4%`
- wall MFU using NVIDIA BF16 peak (`1 PFLOP/s`): about `5.2%`

The logged per-step MFU is likely undercounted because JAX async execution and `log_every=10` charge work to logged sync points unevenly.

Implemented direction:
- Added post-processed wall-clock throughput and MFU to `run_summary`.
- Separated corrected steady training throughput, wall-clock throughput, and raw logged throughput.
- Excluded eval/sample/checkpoint rows when computing steady-state training estimates.
- Updated RTX PRO 6000 Blackwell peak FLOP denominator to `1e15`.
- Added derived fields to scorecard and registry so W&B/log-step timing does not dominate run-level comparisons.
