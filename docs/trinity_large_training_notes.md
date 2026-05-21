# Trinity Large Training Notes

These notes capture training details from the Arcee Trinity Large technical report so we can map the useful pieces into Jaxtitan deliberately. This is reference material, not a claim that Jaxtitan should copy the whole recipe.

Sources:

- Arcee Trinity Large Technical Report: https://arxiv.org/abs/2602.17004
- Arcee Trinity Large blog: https://www.arcee.ai/blog/trinity-large

## Model Recipe

Trinity Large is a decoder-only sparse MoE model:

- 400B total parameters.
- 13B active parameters per token.
- 60 transformer layers.
- 6 initial dense layers, followed by MoE layers.
- Hidden size: 3072.
- Dense FFN intermediate size: 12288.
- Attention heads: 48 query heads, 8 KV heads.
- Head dimension: 128.
- Pretraining sequence length: 8192.
- Local attention window: 4096.
- Attention pattern: 3 local sliding-window RoPE layers followed by 1 global NoPE layer, repeated.
- Attention features: GQA, QK norm, gated attention.
- MoE: 1 shared expert, 256 routed experts, 4 activated routed experts per token.
- Routed expert size: 3072.
- Route scale: 2.448.
- Initialization: truncated normal with sigma 0.009 for Trinity Large.
- Embedding activations are scaled by sqrt(hidden_size).

The stability-relevant architecture choices were the dense prefix, QK norm, gated attention, depth-scaled sandwich norm, sigmoid routing, shared experts, and expert bias routing.

## Optimizer And Schedule

The report uses Muon for hidden layers and AdamW for embedding/output layers.

Important difference from our current local Muon baseline: Trinity does not use RMS matching against AdamW. It uses the Keller Jordan-style width transfer adjustment:

```text
adjusted_lr = lr * sqrt(max(1, fan_out / fan_in))
```

Trinity Large training hyperparameters:

- Warmup: 2000 steps.
- Peak Muon LR: 8e-4.
- Peak AdamW LR: 2e-4.
- Initial global batch: 12288 sequences.
- Sequence length: 8192.
- Batch increased to 16384 sequences after 4.9T tokens.
- Decay phase: cosine decay to one tenth of peak LR.
- Context extension: continue cosine decay from one tenth to one twentieth of peak LR.

For Jaxtitan this implies we should eventually separate:

- optimizer intent: Muon for hidden matrices, AdamW fallback;
- Muon LR adjustment policy;
- per-backend peak LR;
- schedule shape and phase transitions.

## Data Recipe

Trinity Large was trained on 17T tokens sampled from a 20T-token mix organized into three phases with a 13:4:3 phase ratio. Later phases boost higher-quality and domain-specific data, especially code, math, STEM, and reasoning.

The report emphasizes synthetic data at large scale:

- over 8T synthetic tokens total;
- about 6.5T synthetic web tokens;
- about 1T synthetic multilingual tokens;
- about 800B synthetic code tokens.

The data recipe matters to Jaxtitan less as a direct source list and more as a runtime requirement: data phase, document provenance, domain mixture, and sampling policy need to be observable and checkpoint-compatible.

## Data Loading And RSDB

Trinity uses sequence packing and later introduces the Random Sequential Document Buffer to reduce inter-batch correlation from long documents.

RSDB behavior:

- Keep a buffer of active tokenized documents.
- Each document has a read head.
- To fill one sequence, repeatedly sample a random active document, read sequential tokens from its read head, advance that read head, and continue until the sequence is full.
- Refill active documents in bulk.
- Use an internal buffer twice the configured user-facing buffer size.
- Trinity Large phase 3 used a user buffer size of 4096 per GPU, internal size 8192 per GPU, and 4 workers per GPU.

Reported effects:

- BatchHet reduced by 4.23x after RSDB was introduced.
- Step-to-step loss variance reduced by 2.4x.
- In small experiments, RSDB reduced BatchHet by 46%.
- Matching RSDB BatchHet reportedly required around 7x larger batch size in the baseline.

Jaxtitan already has document-aware buffering. The next maturity step is to make the document boundary data useful to training, not just sampling:

- document-aware attention masks;
- per-document or per-sequence provenance;
- BatchHet as an artifact metric, not a console headline;
- deterministic buffer state in checkpoints.

## MoE Routing

The report uses sigmoid routing with expert bias:

- Router logits are converted to sigmoid scores.
- Expert selection uses `score + expert_bias`.
- Gating weights use the unbiased score, not the biased score.
- Selected scores are normalized across the selected experts.
- Route scale is applied after normalization.

This matches the AFMoE semantics we aligned in Jaxtitan.

## SMEBU

SMEBU means Soft-clamped Momentum Expert Bias Updates. It is the main Trinity Large-specific MoE training algorithm.

For each expert:

```text
mean_load = average expert load
violation_i = (mean_load - load_i) / mean_load
soft_violation_i = tanh(kappa * violation_i)
delta_i = lambda * soft_violation_i
delta_i = delta_i - mean(delta)
momentum_i = beta * momentum_i + (1 - beta) * delta_i
expert_bias_i = expert_bias_i + momentum_i
```

Trinity Large constants:

- load-balance learning rate `lambda = 5e-4`;
- momentum `beta = 0.5`;
- clamp scale `kappa = 2`.

The report’s motivation is that sign-based aux-loss-free expert bias updates can oscillate near a good bias value, especially as expert count grows. SMEBU replaces the sign step with a continuous bounded update and momentum.

For Jaxtitan, this should be implemented as explicit non-gradient router state:

- expert bias remains model/router state but has no gradients and no weight decay;
- SMEBU momentum is training/runtime state;
- load counts and MaxVio are metrics;
- checkpoint/resume must include bias and momentum exactly;
- compatibility fingerprints must include the balancing policy and constants.

## Sequence-Wise Auxiliary Loss

Trinity Large also adds a sequence-wise load-balance auxiliary loss with coefficient 1e-4. This is separate from SMEBU.

The purpose is to encourage expert balance inside each sequence, not only across the whole batch. This means our current model-output/loss contract should support auxiliary losses as first-class named terms, with numerator/denominator or scalar accounting that stays compatible with dense models.

For Jaxtitan this points to:

- model forward returns logits plus optional auxiliary terms;
- train metrics record causal loss and total loss separately;
- eval can choose whether aux losses are included or reported only;
- aux losses must be explicit in config and resume compatibility.

## Z-Loss And Logit Diagnostics

The report calls out LM-head logit scale as a stability issue. They adopted z-loss and logged logit statistics.

Practical note from the report:

- intended z-loss weight was 1e-4;
- when introduced mid-run, larger values destabilized the model;
- they used 1e-6 in that context;
- z-loss stabilized maximum and mean logit trends.

For Jaxtitan:

- z-loss should be implemented in the causal LM loss path, not inside the model;
- logit max/mean should be training diagnostics;
- loss rows should expose `lm_loss`, `z_loss`, `aux_loss`, and `total_loss` distinctly.

## Intra-Document Attention Masking

Trinity Large adopted intra-document masking so tokens cannot attend to tokens from other documents in packed sequences.

This matters for Jaxtitan because our prepared-data path already has document-aware sampling, but training still needs attention masks that represent document boundaries. Loss masks are not enough; attention itself must respect document boundaries for packed samples when enabled.

Likely implementation shape:

- data pipeline emits document segment IDs or a block attention mask contract;
- full forward and KV-cache prefill honor attention masks;
- train/eval compile contracts include mask shapes;
- document masking is part of runtime compatibility.

## Systems And Distributed Training

Trinity used a modified TorchTitan stack.

Distributed setup:

- Nano and Mini: HSDP on 512 H200 GPUs.
- Large: 2048 B300 GPUs.
- FSDP group size: 128.
- Expert parallel group size for Large: 8.
- Context parallelism degree for context extension: 4.

Distributed optimizer detail:

- They used a Dion-derived distributed Muon implementation.
- Expert gradients are orthogonalized in batched form without flattening the rank-3 expert gradient tensor.
- This avoids gathering across the expert-parallel group for expert tensors.

For Jaxtitan this is a warning: rank-3 MoE expert optimizer behavior should not be treated as normal rank-2 Muon. We currently route rank-3 expert tensors to AdamW fallback, which is conservative. If we want Trinity-style distributed Muon for experts, it should be a separate tested optimizer path.

## Stability Bundle

The report says the final stable Trinity Large run applied multiple changes together, so the individual effects are not ablated. The bundle was:

- SMEBU;
- BF16 fallback instead of MXFP8 kernels for linears/grouped GEMMs;
- z-loss;
- sequence-wise load-balance aux loss;
- dense prefix increased from 3 to 6;
- intra-document attention masking.

We should not assume any single item is sufficient. For Jaxtitan, the safe implementation order is:

1. router/load metrics and MaxVio;
2. SMEBU state and deterministic bias updates;
3. auxiliary loss plumbing;
4. z-loss and logit diagnostics;
5. document-aware attention masks;
6. Trinity-style Muon LR adjustment;
7. rank-3 expert optimizer work, only after distributed MoE layout is clearer.

## Immediate Jaxtitan Follow-Up

The next implementation slice should probably be the MoE training-state/loss foundation, not expert parallelism yet:

- add model auxiliary-output contract;
- expose router counts, route probabilities, MaxVio, and entropy;
- add SMEBU state/update after each train step;
- checkpoint SMEBU momentum and expert bias;
- add sequence-wise aux loss and z-loss;
- keep all terms visible in local metrics.

That gives us the training algorithm surface the report says mattered, while keeping the distributed expert-kernel problem separate.
