# Run Scoring

The run score is baseline-relative and displayed on an early-stage progression
scale. The baseline run starts at:

```text
BASE_SCORE = 25.0
```

Future runs are scored against that baseline so architecture, kernel, and data
changes show up as a progression ladder without implying that the first baseline
is already a strong or mature model.

The normal scoring entrypoint is:

```bash
uv run experiment configs/x.toml
```

This runs training, checkpoint validation, full CORE plus inference, writes the
scored summary, upserts `runs/registry.jsonl`, and regenerates
`runs/registry.html` plus the tracked README chart at
`docs/run_score_progression.svg`.

```text
RunScore =
BASE_SCORE * (
  0.40 * Quality
+ 0.25 * TrainingEfficiency
+ 0.20 * InferenceEfficiency
+ 0.15 * Health
)
```

`Quality`, `TrainingEfficiency`, and `InferenceEfficiency` are baseline-relative
multipliers centered around `1.0`. `Health` is an absolute reliability score in
`[0.0, 1.0]`.

## Quality Score

Quality measures whether the trained model is better, using capability first and
language-model fit second:

```text
Quality =
  0.50 * core_score
+ 0.23 * native_val_bpb_score
+ 0.17 * domain_mean_bpb_score
+ 0.05 * domain_worst_bpb_score
+ 0.05 * epiplexity_score
```

CORE is the primary quality signal because it is the hardest heldout capability
benchmark. Validation BPB and domain BPB keep the score grounded in language
model fit and catch regressions that CORE alone may miss.

## Component Scores

CORE can be negative, so compare it with a shifted ratio:

```text
core_score = (run_core + 1.0) / (baseline_core + 1.0)
```

BPB is lower-is-better:

```text
native_val_bpb_score = baseline_final_val_bpb / run_final_val_bpb
domain_mean_bpb_score = baseline_domain_mean_bpb / run_domain_mean_bpb
domain_worst_bpb_score = baseline_domain_worst_bpb / run_domain_worst_bpb
```

Epiplexity is higher-is-better, using a lightweight prequential proxy:

```text
epiplexity_score =
  run_train_epiplexity_bpb_auc_per_byte
  /
  baseline_train_epiplexity_bpb_auc_per_byte
```

The proxy is weakly weighted because it can reward slower learning if read
without final quality metrics.

Each component is clamped before weighting:

```text
component = clamp(component, 0.5, 1.5)
```

## Why BPB

Token loss and perplexity are not comparable across tokenizers because they are
measured per token. BPB is measured per raw byte:

```text
BPB = nats / (ln(2) * bytes)
```

That makes BPB the right scoring metric when tokenizer experiments are allowed.
Loss and PPL remain useful diagnostics, but they should not drive the
cross-tokenizer quality score.

## Training Efficiency Score

Training efficiency measures whether the run used hardware and estimated model
compute well. MFU is the key indicator:

```text
TrainingEfficiency =
  0.50 * mfu_score
+ 0.30 * quality_per_compute_score
+ 0.20 * tokens_per_peak_flop_score
```

MFU is higher-is-better:

```text
mfu_score = run_avg_mfu / baseline_avg_mfu
```

Quality per compute uses estimated model training FLOPs:

```text
train_compute_flops = flops_per_token * tokens_seen

quality_per_compute_score =
  (run_quality / run_train_compute_flops)
  /
  (baseline_quality / baseline_train_compute_flops)
```

Tokens per peak FLOP measures actual token throughput normalized by hardware
size:

```text
tokens_per_peak_flop = avg_train_tokens_per_sec / peak_flops_total

tokens_per_peak_flop_score =
  run_tokens_per_peak_flop / baseline_tokens_per_peak_flop
```

Each component is clamped before weighting:

```text
component = clamp(component, 0.5, 1.5)
```

Because MFU and quality-per-compute depend on the FLOPs estimate, architecture
changes that meaningfully change compute must update `estimate_flops_per_token`.
If `avg_mfu`, `flops_per_token`, or `peak_flops_total` is missing, the training
efficiency score and overall run score should be marked unscored instead of
silently filling neutral values.

## Inference Efficiency Score

Inference efficiency measures whether the trained model is usable at inference
time. In v1 this uses the repo's simple KV-cache decode benchmark:

```text
InferenceEfficiency =
  0.55 * decode_score
+ 0.30 * prefill_score
+ 0.15 * ttft_score
```

Decode and prefill throughput are higher-is-better:

```text
decode_score = run_decode_tokens_per_sec / baseline_decode_tokens_per_sec
prefill_score = run_prefill_tokens_per_sec / baseline_prefill_tokens_per_sec
```

Time to first token is lower-is-better:

```text
ttft_score = baseline_ttft_sec / run_ttft_sec
```

Each component is clamped before weighting:

```text
component = clamp(component, 0.5, 1.5)
```

If inference benchmark metrics are missing, the inference efficiency score and
overall run score should be marked unscored. When KV-cache support lands, this
section should continue using cached decode throughput as the primary decode
metric, and update only if the serving path changes materially.

## Health Score

Health measures whether the run is trustworthy enough to compare or scale. It is
not baseline-relative.

Start with:

```text
Health = 1.0
```

Apply hard gates:

```text
if status == "failed":
  Health = 0.0

if status == "incomplete":
  Health = min(Health, 0.25)

if status == "unstable":
  Health = min(Health, 0.75)

if nan_count > 0:
  Health = 0.0
```

Then subtract bounded penalties:

```text
loss_spike_penalty = min(0.25, 0.05 * loss_spike_count)
grad_spike_penalty = min(0.15, 0.03 * grad_norm_spike_count)

val_regression =
  max(0, final_val_bpb - best_val_bpb) / best_val_bpb

val_regression_penalty = min(0.20, val_regression)
```

Final health:

```text
Health =
  Health
- loss_spike_penalty
- grad_spike_penalty
- val_regression_penalty

Health = clamp(Health, 0.0, 1.0)
```

Failed runs should be marked unscored even if partial metrics look good.

## Scoring Eligibility

A run is eligible for a final score only if it has:

```text
full CORE eval, not a --max-per-task subset
native validation BPB
domain validation BPB
avg MFU
flops_per_token
peak_flops_total
inference benchmark metrics
health metrics
train epiplexity proxy
```

If required metrics are missing, `final_score` should be `null`, but available
partial category scores may still be reported for debugging.

For validation metrics, use standalone checkpoint eval artifacts when present.
If no checkpoint eval exists, fall back to final training-time validation metrics.

## Current Caveats

CORE is intentionally dominant inside Quality because it is the hardest heldout
capability benchmark. A real gain on CORE should meaningfully move the ladder.

Inference scoring is still provisional because the current benchmark uses a
simple single-request KV-cache decode loop, not a production serving scheduler.

Epiplexity currently uses a prequential BPB-AUC proxy from the run loss curve.
The fuller requential teacher/student KL estimate is deferred.
