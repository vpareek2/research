# TorchTitan Deep Dive

This document summarizes the local TorchTitan checkout under `ref/torchtitan`.
It is descriptive only. It does not decide which pieces belong in this repo.

Local reference inspected:

- Path: `ref/torchtitan`
- Branch: `main`
- Commit: `a670699c [graph_trainer] Skip dense numerics tests due to upstream DTensor regression (#3372)`

## Purpose

TorchTitan is a PyTorch-native platform for large-scale generative model
training. It is built as a clean-room stack around PyTorch distributed
features, especially Llama-style pretraining. Its public README describes the
project as a platform for rapid experimentation and large-scale training of
generative AI models.

The codebase is organized around a configurable `Trainer` that assembles the
training runtime from typed component configs:

- model specification
- tokenizer
- dataloader
- loss
- optimizer
- learning-rate scheduler
- distributed mesh and parallelism setup
- activation checkpointing
- compile settings
- checkpoint manager
- validator
- metrics processor
- profiler
- debug and communication settings

## Top-Level Execution Flow

Primary entrypoint:

- `torchtitan/train.py`

Typical shell launcher:

- `run_train.sh`

The launcher defaults to:

```bash
MODULE=llama3
CONFIG=llama3_debugmodel
NGPU=8
```

Normal launch path:

```bash
torchrun --nproc_per_node=${NGPU} -m torchtitan.train --module ${MODULE} --config ${CONFIG}
```

Debug communication modes are supported through `COMM_MODE`:

- `fake_backend`: configuration validation without real communication
- `local_tensor`: single-GPU simulation of multi-GPU behavior
- `torchcomms`: torchcomms-based communicators
- default: normal distributed execution

`train.py` performs the top-level lifecycle:

1. Initialize logging.
2. Parse config through `ConfigManager`.
3. Initialize structured logging.
4. Build the configured trainer with `config.build()`.
5. Create a seed checkpoint if requested, otherwise run `trainer.train()`.
6. Close trainer resources.
7. Destroy the process group if initialized.

## Configuration System

Key files:

- `torchtitan/config/configurable.py`
- `torchtitan/config/configs.py`
- `torchtitan/config/manager.py`
- model config registries such as `torchtitan/models/llama3/config_registry.py`

TorchTitan uses Python dataclasses rather than static TOML or YAML as the
canonical config source. Each configurable component defines a nested
`Config` dataclass. `Configurable.__init_subclass__` auto-wires the config's
`build()` method so that a config can construct its owning class.

The config manager requires:

```bash
--module <module_name>
--config <config_function_name>
```

For example:

```bash
python -m torchtitan.train --module llama3 --config llama3_debugmodel
```

The manager loads `torchtitan.models.<module>.config_registry` or
`torchtitan.experiments.<module>.config_registry`, calls the selected function,
then applies CLI overrides using `tyro`.

Configuration precedence:

1. CLI arguments
2. Python config registry defaults

Top-level trainer config fields include:

- `model_spec`
- `hf_assets_path`
- `dump_folder`
- `profiler`
- `metrics`
- `tokenizer`
- `dataloader`
- `optimizer`
- `lr_scheduler`
- `training`
- `parallelism`
- `checkpoint`
- `activation_checkpoint`
- `compile`
- `comm`
- `validator`
- `debug`
- `loss`

## Trainer Lifecycle

Key file:

- `torchtitan/trainer.py`

`Trainer` is the central runtime owner. It inherits from:

- `torch.distributed.checkpoint.stateful.Stateful`
- `Configurable`

The constructor performs the following setup:

1. Select the local device from `LOCAL_RANK`.
2. Initialize distributed state and build parallel meshes.
3. Log or save config if requested.
4. Determine batch mesh degree and rank.
5. Initialize garbage-collection control.
6. Set determinism and RNG behavior.
7. Build tokenizer.
8. Build dataloader.
9. Build model config from `model_spec`.
10. Build the model on the PyTorch `meta` device.
11. Verify the model satisfies TorchTitan's module protocol.
12. Build metrics processor.
13. Estimate model parameter count and FLOPs per token.
14. Apply parallelism, activation checkpointing, and compile.
15. Materialize model parameters on CPU or accelerator.
16. Initialize model weights and buffers.
17. Wire chunked loss to `lm_head` when applicable.
18. Build optimizer.
19. Build LR scheduler.
20. Initialize train state.
21. Build checkpoint manager.
22. Build train context.
23. Build validator if enabled.

The training loop:

1. Load checkpoint if available or requested.
2. Build a batch generator.
3. Run until `self.step == training.steps`.
4. Increment step.
5. Run garbage collection policy.
6. Run `train_step`.
7. Save checkpoint if policy says to save.
8. Run validation if enabled.
9. Step the profiler.
10. Tighten process-group timeouts after the first resumed or fresh step.

Trainer state saved in checkpoints:

- `step`
- `ntokens_seen`

## Training Step

`Trainer.train_step` coordinates gradient accumulation, token accounting,
forward/backward, optimization, and metrics.

Step structure:

1. Zero optimizer gradients.
2. Capture current LR for logging.
3. Fetch all microbatches for the accumulation step on CPU.
4. Count local valid tokens using `IGNORE_INDEX`.
5. All-reduce valid-token count over the batch mesh when data parallelism is enabled.
6. Move each microbatch to device.
7. Run forward/backward for each microbatch.
8. Clip gradient norm.
9. Wait for checkpoint staging if necessary.
10. Step optimizer.
11. Step LR scheduler.
12. Sum accumulated losses.
13. Collect distributed loss metrics.
14. Log metrics if the current step should log.

Loss normalization is token based. Cross entropy uses sum reduction and is
divided by global valid tokens. This keeps loss accounting explicit under data
parallelism, context parallelism, masking, and gradient accumulation.

## Post-Dataloading Processing

`Trainer.post_dataloading_process` converts raw dataloader output into the
model call signature.

Input dataloader output:

- `input_dict`
- `labels`

Expected primary input key:

- `input`

Optional fields:

- `positions`
- attention-related auxiliary tensors

The method separates:

- `inputs`: main token tensor
- `labels`: target labels
- `extra_inputs`: tensors only needed by the first pipeline stage
- `extra_kwargs`: arguments forwarded across pipeline stages, such as positions
  and attention masks

For decoder models, positions are derived or validated depending on the
attention mask type. Block-causal attention requires per-document positions from
the dataloader.

## Distributed Mesh Model

Key files:

- `torchtitan/distributed/parallel_dims.py`
- `torchtitan/distributed/utils.py`

TorchTitan represents distributed layout through `ParallelDims`.

Primary dimensions:

- `dp_replicate`: data-parallel replication degree
- `dp_shard`: data-parallel sharding degree
- `cp`: context parallel degree
- `tp`: tensor parallel degree
- `pp`: pipeline parallel degree
- `ep`: expert parallel degree
- `world_size`: total distributed process count

Validation requires:

```text
dp_replicate * dp_shard * cp * tp * pp == world_size
```

If `dp_shard == -1`, it is inferred from the remaining world-size after other
parallel dimensions are assigned.

Derived meshes include:

- `pp`
- `batch`
- `loss`
- `dp_replicate`
- `fsdp`
- `cp`
- `tp`
- `ep`
- `efsdp`

The mesh builder creates several global mesh views:

- dataloading mesh: `["pp", "batch", "cp", "tp"]`
- loss mesh: flattened `batch` and `cp`
- dense mesh: `["pp", "dp_replicate", "fsdp", "tp"]`
- sparse mesh: `["pp", "dp_replicate", "efsdp", "ep"]`

Convenience flags expose which dimensions are active:

- `dp_enabled`
- `dp_replicate_enabled`
- `dp_shard_enabled`
- `cp_enabled`
- `fsdp_enabled`
- `tp_enabled`
- `pp_enabled`
- `ep_enabled`

`seq_len_divisor` is computed as:

```text
tp * (cp * 2)
```

This reflects sequence-parallel and context-parallel divisibility constraints.

## Parallelism Application

Llama-specific parallelization lives in:

- `torchtitan/models/llama3/parallelize.py`

The Llama path applies:

1. Context-parallel attention wrapping when CP is enabled.
2. Tensor parallelism when TP is enabled.
3. Optional async tensor parallelism when compile is enabled.
4. Activation checkpointing.
5. Per-block `torch.compile`.
6. FSDP or HSDP through `fully_shard`.
7. CPU offload if requested.

FSDP setup:

- Builds a `MixedPrecisionPolicy`.
- Applies FSDP to embeddings, output/norm, each transformer block, and the
  full model.
- Disables FSDP automatic gradient division because TorchTitan scales loss
  explicitly by global valid-token count.

Pipeline parallelism lives in:

- `torchtitan/distributed/pipeline_parallel.py`

It:

- Computes virtual stage metadata.
- Splits model modules into pipeline parts.
- Applies SPMD-style parallelism to each model part.
- Builds a pipeline schedule.
- Returns schedule, model parts, and booleans for first/last stage ownership.

Tensor-parallel helpers live in:

- `torchtitan/distributed/tensor_parallel.py`

They include custom `ParallelStyle` wrappers for replicated computation,
explicit gradient placement, and async TP enablement.

## Data Loading

Key files:

- `torchtitan/components/dataloader.py`
- `torchtitan/hf_datasets/text_datasets.py`

TorchTitan dataloaders are stateful. `BaseDataLoader` inherits from:

- `torch.distributed.checkpoint.stateful.Stateful`
- `Configurable`

`ParallelAwareDataloader` wraps `torchdata.stateful_dataloader.StatefulDataLoader`
and stores dataloader state per data-parallel rank.

The main text dataset path is `HuggingFaceTextDataset`. It:

- Loads a configured Hugging Face dataset.
- Splits data by data-parallel node.
- Tokenizes text with BOS/EOS.
- Packs tokens into fixed-length training examples.
- Emits `input`, `positions`, and labels.
- Tracks sample index, epoch, input buffer, and position buffer.
- Supports checkpoint save/load for resume.
- Can loop infinitely, with deterministic reshuffling for map-style datasets.

Built-in text datasets include:

- `c4`
- `c4_test`
- `c4_validation`

There is also a chat/instruction-tuning dataset path with masked labels using
`IGNORE_INDEX`.

## Tokenizer

Tokenizers are configurable components. Trainer builds the tokenizer with:

```python
config.tokenizer.build(tokenizer_path=config.hf_assets_path)
```

The default path expects local Hugging Face assets. The config manager validates
whether `hf_assets_path` exists and contains the expected tokenizer format.

## Loss System

Key file:

- `torchtitan/components/loss.py`

Loss classes inherit from `BaseLoss`, which provides a common call signature:

```python
loss(pred, labels, global_valid_tokens)
```

Implemented losses include:

- `CrossEntropyLoss`
- `MSELoss`
- `ChunkedCELoss`

Cross entropy:

- flattens prediction and label tensors
- casts logits to fp32
- uses sum reduction
- ignores labels equal to `IGNORE_INDEX`
- divides by global valid-token count when provided

`ChunkedCELoss` avoids materializing the full `[batch, seq, vocab]` logits
tensor at once. It expects the model to return hidden states with `_skip_lm_head`
enabled, then applies `lm_head` chunk by chunk along the sequence dimension.

The documented flow is:

1. Model forward returns hidden states.
2. Hidden states are split along sequence dimension.
3. Each chunk is projected through `lm_head`.
4. Cross entropy is computed per chunk.
5. Chunk gradients are accumulated.
6. Backward is applied to the full hidden-state tensor.

## Optimizers

Key file:

- `torchtitan/components/optimizer.py`

TorchTitan wraps optimizers in containers so the training loop does not need to
know whether there is one optimizer or many.

`OptimizersContainer`:

- wraps one optimizer per model part
- supports Adam and AdamW
- supports implementation modes:
  - `for-loop`
  - `foreach`
  - `fused`
  - `fused_opt_states_bf16`
- supports regex-based parameter groups
- flattens optimizer state dicts for distributed checkpointing

`OptimizersInBackwardContainer`:

- creates one optimizer per parameter
- registers post-accumulate-grad hooks
- steps parameters during backward
- does not support pipeline parallelism or expert parallelism

There is also a MoE load-balancing optimizer hook that updates expert bias
before optimizer steps when configured.

## Learning-Rate Scheduler

Key file:

- `torchtitan/components/lr_scheduler.py`

`LRSchedulersContainer` wraps one or more PyTorch schedulers. The schedule is a
warmup-stable-decay form implemented through `LambdaLR`.

Config fields:

- `warmup_steps`
- `total_steps`
- `decay_ratio`
- `decay_type`
- `min_lr_factor`

Supported decay types:

- `linear`
- `sqrt`
- `cosine`

The container saves only the first scheduler state because it assumes all
schedulers are identical.

## Checkpointing

Key file:

- `torchtitan/components/checkpoint.py`

`CheckpointManager` owns checkpoint save/load policy for:

- model
- optimizer
- LR scheduler
- dataloader
- trainer state

Checkpoint config fields include:

- `enable`
- `folder`
- `interval`
- `initial_load_path`
- `initial_load_model_only`
- `initial_load_in_hf`
- `last_save_model_only`
- `last_save_in_hf`
- `export_dtype`
- `async_mode`
- `keep_latest_k`
- `load_step`
- `exclude_from_loading`
- `enable_first_step_checkpoint`
- `create_seed_checkpoint`
- `load_only`

Async modes:

- `disabled`
- `async`
- `async_with_pinned_mem`

Checkpoint IDs use:

```text
step-<step>
```

Loading behavior:

- If the checkpoint folder does not exist, optional initial load paths are used.
- If the checkpoint folder exists, the latest or requested step is loaded.
- Step `0` is treated as model-only.
- Exclusion lists can omit state sections from loading.

Final save behavior:

- Can save model-only.
- Can export to selected dtype.
- Can optionally save in Hugging Face safetensors format through a state-dict
  adapter.

The manager also supports stale-checkpoint purging with a background thread.

## Metrics

Key file:

- `torchtitan/components/metrics.py`

`MetricsProcessor` owns console and optional external metrics logging.

Backends:

- no-op logger
- TensorBoard
- Weights & Biases
- logger container for multiple backends

Rank selection:

- non-pipeline runs log from rank `0`
- pipeline runs usually log from the first rank of the last pipeline stage
- `save_for_all_ranks` can override this

Logged training metrics include:

- global average loss
- global max loss
- gradient norm
- throughput in tokens/sec
- TFLOPs
- MFU
- end-to-end step time
- data-loading time
- data-loading percentage
- max active memory
- max reserved memory
- allocation retries
- OOM count
- LR and token counts from trainer-provided extra metrics

MFU uses:

```text
100 * flops_per_token * tokens_per_sec / gpu_peak_flops
```

If quantization is active, MFU is skipped.

The metrics processor also tracks validation loss and validation throughput.

## Device Memory Monitoring

`DeviceMemoryMonitor` records:

- device name
- device memory capacity
- peak active memory
- peak reserved memory
- allocation retries
- OOM count

It resets peak memory stats after logging.

## Profiling

Key file:

- `torchtitan/tools/profiler.py`

Profiler features:

- PyTorch profiler traces
- memory snapshots
- periodic trace export
- optional trace post-processing hook
- OOM-triggered memory snapshot on exit

Trace output defaults:

```text
profiling/traces/iteration_<step>/rank<rank>_trace.json
```

Memory snapshot output defaults:

```text
profiling/memory_snapshot/step_<step>/<rank>_step_<step>.pickle
```

The trainer activates the profiler as a context manager around the training
loop and calls `profiler.step()` once per training iteration.

## Validation

Key file:

- `torchtitan/components/validate.py`

Validation is a configurable component. `Validator` builds its own validation
dataloader, moves batches to device, computes global valid-token count, runs
forward/loss, handles pipeline-parallel evaluation when needed, and logs
validation metrics through `MetricsProcessor`.

Validator config fields:

- `enable`
- `freq`
- `steps`
- `dataloader`

Validation can consume a fixed number of steps or the whole validation dataset.
The code warns that consuming all data can hang if ranks have unequal sample
counts.

## Model System

Key files:

- `torchtitan/protocols/module.py`
- `torchtitan/protocols/model.py`
- `torchtitan/protocols/model_spec.py`
- `torchtitan/models/common/*`
- `torchtitan/models/llama3/*`

TorchTitan model modules inherit from `Module`, which combines:

- `torch.nn.Module`
- `Configurable`

`Module` provides:

- recursive `init_states`
- parameter initialization through config-provided `param_init`
- buffer initialization hook
- sharding config support
- recursive `parallelize`
- optional local-map wrapping
- input/output redistribution around forward

`BaseModel` adds:

- `init_weights` alias for `init_states`
- protocol verification for submodules
- abstract config methods:
  - `update_from_config`
  - `get_nparams_and_flops`

`ModelSpec` bundles:

- model name
- flavor
- selected model config
- parallelization function
- pipelining function
- post-optimizer hook
- state-dict adapter

## Common Model Components

Common components include:

- `Linear`
- `Embedding`
- `RMSNorm`
- `FeedForward`
- `RoPE`
- attention implementations
- decoder base class
- MoE components
- sharding helpers

`Linear`, `Embedding`, and `RMSNorm` use multiple inheritance with PyTorch
native modules plus TorchTitan `Module`, keeping the module tree flat while
adding config/build/init behavior.

`FeedForward` implements SwiGLU:

```text
w2(silu(w1(x)) * w3(x))
```

`compute_ffn_hidden_dim` implements Llama-style 2/3 FFN scaling and rounding to
a configured multiple.

`RoPE` supports:

- complex backend
- cos/sin backend
- no scaling
- Llama scaling
- YaRN scaling

## Decoder Model

Key file:

- `torchtitan/models/common/decoder.py`

`Decoder` is a shared autoregressive decoder-only base. Its config contains:

- model dimension
- vocab size
- token embedding config
- RoPE config
- transformer layer configs
- final norm config
- LM head config

Forward path:

1. Embed input tokens if embeddings are present.
2. Iterate transformer layers.
3. Apply final norm if present.
4. Return hidden states if `_skip_lm_head` is set.
5. Otherwise apply `lm_head`.

Attention masks are generated for FlexAttention and VarlenAttention from
per-token positions. Block-causal masks use document IDs derived from position
resets.

## Llama3 Model

Key files:

- `torchtitan/models/llama3/model.py`
- `torchtitan/models/llama3/__init__.py`
- `torchtitan/models/llama3/parallelize.py`
- `torchtitan/models/llama3/state_dict_adapter.py`

`Llama3TransformerBlock` contains:

- attention
- feed-forward network
- attention RMSNorm
- FFN RMSNorm

Forward:

```text
h = x + attention(attention_norm(x), ...)
out = h + feed_forward(ffn_norm(h))
```

`Llama3Model` extends `Decoder` and supports optional weight tying.

`Llama3Model.Config.update_from_config` syncs model settings with trainer
settings:

- updates RoPE max sequence length
- validates context parallelism with attention backend
- validates tensor parallel degree divides attention head counts
- rejects weight tying with pipeline parallelism
- applies Llama3 sharding config

`Llama3Model.Config.get_nparams_and_flops` delegates dense model parameter and
FLOPs estimation to `get_dense_model_nparams_and_flops`.

Llama3 model registry flavors include:

- `debugmodel`
- `debugmodel_fused_qkv`
- `1B`
- `3B`
- `8B`
- `70B`
- `405B`

Registry construction returns a `ModelSpec` with the selected config and
Llama-specific parallelization, pipelining, and state-dict adapter functions.

## Attention System

Key file:

- `torchtitan/models/common/attention.py`

Attention backends include:

- scaled dot-product attention
- FlexAttention
- VarlenAttention

Attention masking helpers include:

- causal mask
- document mask from position resets
- fixed block masks
- sliding window masks

`VarlenAttention` packs `[batch, seq, heads, dim]` into varlen layout and uses
`torch.nn.attention.varlen.varlen_attn`.

`FlexAttention` compiles `flex_attention` as a class variable and accepts block
masks, score modifiers, kernel options, GQA enablement, and optional LSE return.

`ScaledDotProductAttention` calls `torch.nn.functional.scaled_dot_product_attention`
with a priority list of SDPA backends.

GQA-related configs can use separate Q and KV projections or fused QKV
projection.

## Activation Checkpointing And Compile

Config fields live in:

- `torchtitan/config/configs.py`

Activation checkpoint modes:

- `selective`
- `full`
- `memory_budget`
- `none`

The config also contains controls for early stop, memory budget visualization,
RNG preservation, determinism checks, and debugging.

Compile config:

- `enable`
- `components`
- `backend`

The Llama parallelization path applies activation checkpointing before
`torch.compile`, then applies FSDP after compile.

## Debug And Determinism

Debug config fields include:

- seed
- deterministic mode
- deterministic warn-only mode
- MoE force load balance
- autograd anomaly detection
- batch-invariant mode
- print config
- save config file
- structured logging enablement

Distributed determinism setup:

- can enable deterministic algorithms
- sets CuDNN deterministic behavior
- sets `CUBLAS_WORKSPACE_CONFIG`
- broadcasts a seed across ranks when needed
- offsets seeds along selected mesh dimensions such as pipeline parallelism
- seeds DTensor RNG tracker for SPMD parallelism

Batch-invariant mode sets several NCCL and matmul flags to reduce
batch-composition-dependent numeric drift.

## Structured Logging

TorchTitan uses `torchtitan.observability.structured_logger` to record trace
spans, instants, step state, and scalars. The trainer and components annotate
major phases such as:

- distributed init
- model parallelism init
- dataloading
- forward/backward
- optimizer
- checkpoint save/load
- eval
- profiler step

Structured logging can be disabled with debug config.

## Experiments Tree

TorchTitan includes an `experiments` namespace. In the inspected checkout it
contains areas such as:

- fault-tolerance experiments
- graph trainer
- reinforcement learning
- transformers modeling backend

These experiments use the same general project conventions but are separate
from the core Llama training path.

## Tests And Local Environment Note

Running this repo's unscoped pytest command from the parent workspace collected
TorchTitan tests under `ref/torchtitan`. Collection failed because the parent
JAX workspace environment does not install TorchTitan dependencies such as
`torch` and `tokenizers`.

The scoped parent repo tests passed:

```text
uv run pytest -q tests
305 passed, 1 skipped
```

This note describes local test discovery behavior in the parent workspace, not
TorchTitan upstream test status.
