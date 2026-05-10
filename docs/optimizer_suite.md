# Optimizer Suite Refactor

## Goal

Replace the hardcoded `optax.contrib.muon` path with repo-owned optimizer implementations that can support Muon, Aurora, SOAP, NorMuon, and matrix-iteration ablations without changing the training loop.

The first implementation keeps the Optax `GradientTransformation` interface so `flax.nnx.Optimizer`, checkpointing, and distributed state placement keep working. Optax remains a protocol dependency, not the owner of optimizer algorithms.

## V1 Interface

Run configs use an explicit optimizer section:

```toml
[optimizer]
name = "muon"
lr = 0.001
weight_decay = 0.1

[optimizer.adamw]
b1 = 0.9
b2 = 0.95
eps = 1e-8

[optimizer.muon]
beta = 0.95
nesterov = true
ns_steps = 5
ns_coeffs = [3.4445, -4.775, 2.0315]
eps = 1e-8
```

`train.lr` and `train.decay` are removed. The LR schedule shape remains under `[train.lr_schedule]`, but its peak value is `optimizer.lr`.

## Routing

Routing is generic, not tied to the current transformer block names:

- `matrix`: hidden 2D weights that should receive matrix optimizers such as Muon.
- `embedding`: input vocab table.
- `output`: vocab output projection.
- `vector`: norm scales, biases, and other 1D leaves.
- `other`: any remaining trainable leaf.

Path-derived tags such as `attention`, `mlp`, `norm`, `embedding`, and `output` are metadata for future overrides. Unknown parameters should still receive a generic class and be visible in tests.

## V1 Algorithms

- AdamW is implemented in-repo and used for non-matrix leaves.
- Muon is implemented in-repo and used for `matrix` leaves when `[optimizer].name = "muon"`.
- Muon v1 uses first-moment momentum, optional Nesterov direction, Newton-Schulz orthogonalization, width scaling, decoupled weight decay, and scheduled LR.

Aurora, SOAP, NorMuon, and PolarExpress are follow-up modules behind the same factory and routing interface.

## Implementation Steps

1. Add optimizer config dataclasses and update config loading.
2. Add `research.optimizers` with routing, AdamW, Muon, and factory code.
3. Replace `pretrain.py` optimizer construction with the factory.
4. Update configs, README references, and tests to the new optimizer schema.
5. Add focused tests for config validation, routing, optimizer finite updates, train-step integration, checkpointing, and distributed state placement.

## Acceptance

- `pretrain.py` no longer imports or calls `optax.contrib.muon`.
- Repo training optimizers do not call `optax.adamw`.
- Current MHA/MQA configs load with `[optimizer]`.
- Custom Muon trains a tiny model for one step.
- Existing checkpoint and distributed placement tests still pass with the new optimizer state.
