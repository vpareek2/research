# LM Research: What We're Building

We want to build a small, sharp language-model research stack for reproducible 2B-token experiments.

Your goal is to build production-quality research infrastructure: model architecture, optimizers, data preparation, distributed JAX training, logging, diagnostics, evals, scoring, and registry tracking that make experiments easy to trust and compare.

This means: no hidden fallbacks, no vague metrics, no untracked experiment state. Local artifacts are the source of truth; W&B is only a dashboard and comparison layer.

## Public Repo Contract

- No internal-only paths, hostnames, IPs, tokens, or cloud runbooks in tracked files.
- The supported execution path is `uv` + TOML configs + local run artifacts under `runs/`.
- Keep surfaces small: no second stacks for the same use-case.
- `runs/registry.jsonl` is the scored comparison ledger. Changes to `master` should either add/update a scored registry row or be explicitly labeled `no-run-required`.

## Execution Environment

- Assume hardware-sensitive behavior. Verify actual JAX devices, GPU memory, driver/runtime state, and prepared data before making claims.
- Do not assume W&B is complete or canonical. Check local files first: `metrics.jsonl`, summaries, eval artifacts, checkpoints, and registry rows.
- Do not introduce a second dependency manager or install path. Use `uv`.

## Agent Protocol (Failure Prevention)

This section exists because principles alone do not prevent repeated failures. These rules are the operational contract.

### Session Start (Always)

- Read `AGENTS.md`, then `AGENTS.md.local` if present.
- Confirm mode and print it at the top of every response: `Mode: No-Edits` or `Mode: Execution`.
- If executing: confirm branch (`git branch --show-current`) and working tree (`git status -sb`).

### Mode Gates (Hard)

**No-Edits Mode**

- Trigger: user says "do not make edits/changes", "review only", or "planning/brainstorming".
- In this mode: do not modify tracked files, do not install deps, do not run destructive git ops, and do not change GitHub/cloud state; only read/inspect/analyze.
- Exit only when the user explicitly authorizes execution ("proceed", "implement", "make the changes", "do it").

**Execution Mode**

- Default when the user asks to implement/fix/build.
- If the user says "just X", do X immediately with minimal narration.

### Scope Lock (Before Editing)

- Before editing: list the exact files you will modify.
- Do not touch out-of-scope files; stable/working code is read-only unless explicitly told otherwise.
- For ports/refactors: preserve semantics by default; call out intentional semantic deltas and get approval.
- After fixing a bug pattern: search for other occurrences (prefer `rg`) and fix them in-scope.

### Don't Guess (Ever)

- Never guess at environment state, config values, run status, benchmark results, or file contents. Verify via files, diffs, logs, commands, and measurements.
- Never claim a performance improvement without measured run artifacts.

### Git Safety (High Severity)

- Do not run `git checkout`, `git restore`, `git reset`, or `git clean` without explicit approval; explain what will be lost first.
- Never `git push` unless explicitly asked; only commit when explicitly asked.
- `master` is protected. `dev` is the normal integration branch for active research work.

### Build And Run Discipline (High Severity)

- Only regenerate data, rerun expensive training, or launch cloud runs when the change actually requires it.
- Run narrow tests for the touched surface; run broader tests when changing shared training, logging, summary, config, or registry behavior.
- Do not use W&B-only evidence to debug or score a run.

### Output Completeness

- If asked for "full diff/log output", do not truncate.
- Provide complete copy/paste-ready commands, including `cd`, env vars, config paths, and flags.

### Ultrathink Protocol (When Asked)

1. Verified facts (code/config/logs/diffs)
2. Unknowns / gaps
3. Hypotheses + predicted outcomes
4. Minimal experiments, then conclusions from results

Our ethos: do one thing, exceedingly well - reproducible small-LM research with honest scoring - and nothing else. Elegant minimalism is disciplined intent plus impeccable execution.

Principles

- One clear path per use-case: each supported mode (data prep, preflight, training, full scored experiment, eval, registry) has one explicit way to run.
- Small, sharp surfaces: tiny modules with crisp responsibilities; few public knobs; declarative TOML config is the source of truth.
- Explicit over magical: no hidden background machinery or side effects; contracts and control flow are obvious.
- Hot paths first: training step, data loading, metrics sync, and eval loops are lean, predictable, and measured.
- Fail fast, fail loud: specific guardrails with actionable remedies. No silent downshifts.
- Minimal dependencies: new layers must improve both clarity and experiment quality.
- One source of truth: one config format, one prepared-data manifest schema, one metrics schema, one registry ledger.
- Test what matters: deterministic data/resume behavior, metric correctness, scoring compatibility, and config validation.
- Reproducibility over folklore: record config, environment, artifacts, and score inputs so a result can be rerun months later.
- Documentation that guides, not overwhelms: precise runbooks and remedies; zero fluff.

Craftsmanship rubric for any change

- Intent: Does this improve model quality, throughput, stability, observability, or reproducibility?
- Uniqueness: Are we creating a second way to do something? If yes, why?
- Surface: Did we add a new public knob? Could it be expressed via existing TOML?
- Hot path: If training/data/eval changed, where is the measured tokens/sec, MFU, loss, or scoring delta?
- Invariants: Are token budgets, prepared-data manifests, eval artifacts, and registry requirements enforced or clarified?
- Blast radius: Did deps or coupling increase?
- Repro: Is config/provenance captured to rerun months later?
- Elegance: Is the code visibly simpler afterward?

## Research Workflow

This repo is designed to be a one-stop shop for architecture and optimizer experiments on a fixed small-LM training budget. It should stay streamlined, opinionated, and easy to inspect when training behavior looks strange.

**Supported scope** (intentionally narrow):

- Decoder-only LM pretraining in JAX/Flax NNX.
- TOML-defined model, data, precision, distributed, training, eval, sampling, and W&B settings.
- Prepared offline token datasets, especially FineWeb sample data and tokenizer-derived eval domain packs.
- Muon/AdamW-style optimizer experiments and architecture ablations.
- Local metrics, summaries, checkpoint evals, CORE evals, inference benchmarks, scoring, and registry comparison.

**Typical research flow:**

1. **Baseline or mechanism branch** - Start from `dev` unless explicitly fixing `master`.
2. **Config and preflight** - Add/update TOML, prepare data if needed, and run `uv run preflight configs/your_experiment.toml`.
3. **Training run** - Use `uv run experiment configs/your_experiment.toml` for runs intended to enter the comparison ladder.
4. **Artifact inspection** - Verify local metrics, summaries, checkpoint evals, CORE/inference artifacts, and W&B only as a mirror.
5. **Registry update** - Register the completed scored run in `runs/registry.jsonl`.
6. **Promotion** - Merge to `master` only after the mechanism has a recorded score, unless the PR is maintenance-only and labeled `no-run-required`.

When you work on optimizers: preserve comparability unless the experiment is explicitly about schedule or optimizer hyperparameters.
When you work on metrics: remember JAX dispatch is asynchronous; scoring and live stats must use corrected interval-based throughput where appropriate.
When you work on data: manifests and hashes matter; do not silently accept mismatched prepared artifacts.

## Contract (Principles)

- Config: TOML-only for experiment configuration. Avoid ad hoc flags once a workflow stabilizes.
- Data: no hardcoded data locations; use config paths and prepared manifests.
- Distributed: keep the replicated data-parallel path explicit and tested for one or more local JAX devices.
- Metrics/Tracking: local artifacts are canonical; W&B mirrors scalar metrics and samples.
- Registry: `runs/registry.jsonl` is the durable comparison record and should remain machine-readable JSONL.

## Common Commands

```bash
uv sync
uv run pytest -q
uv run prepare-data configs/data/fineweb_sample10bt.toml
uv run prepare-data configs/data/eval_domains.toml
uv run preflight configs/your_experiment.toml
uv run experiment configs/your_experiment.toml
uv run summarize-run runs/your_run --register
uv run list-runs
```

## Important Paths

- `configs/`: experiment and data-prep configs.
- `src/research/pretrain.py`: training loop and optimizer setup.
- `src/research/model.py`: model definition.
- `src/research/prepare_data.py`: prepared dataset writer.
- `src/research/logs.py`: local metrics and W&B logging.
- `src/research/utils/experiment.py`: full train/eval/score/register flow.
- `src/research/utils/run_summary.py`: summary and registry record generation.
- `src/research/utils/registry_required.py`: PR gate requiring scored registry updates into `master`.
- `runs/registry.jsonl`: scored comparison ledger.
