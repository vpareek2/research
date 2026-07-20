# LM Research: What We're Building

We want to build a small, sharp language-model research stack for reproducible 2B-token experiments.

Your goal is to build production-quality research infrastructure: model architecture, optimizers, data preparation, distributed JAX training, logging, diagnostics, evals, scoring, and registry tracking that make experiments easy to trust and compare.

This means: no hidden fallbacks, no vague metrics, no untracked experiment state. Local artifacts are the source of truth; W&B is only a dashboard and comparison layer.

## Public Repo Contract

- No internal-only paths, hostnames, IPs, tokens, or cloud runbooks in tracked files.
- The supported execution path is `uv` + TOML configs + local run artifacts under `runs/`.
- Keep surfaces small: no second stacks for the same use-case.
- `runs/registry.jsonl` is the scored comparison ledger. Changes intended to alter scored model quality or comparison claims must add/update a scored registry row. Maintenance and correctness-infrastructure PRs use `no-run-required`; docs/runbook-only maintenance may land directly on `master` without a registry row.

## Execution Environment

- Assume hardware-sensitive behavior. Verify actual JAX devices, GPU memory, driver/runtime state, and prepared data before making claims.
- Do not assume W&B is complete or canonical. Check local files first: `metrics.jsonl`, summaries, eval artifacts, checkpoints, and registry rows.
- Do not introduce a second dependency manager or install path. Use `uv`.

## Agent Protocol (Failure Prevention)

This section exists because principles alone do not prevent repeated failures. These rules are the operational contract.

### Session Start (Always)

- Read `AGENTS.md`, then `AGENTS.md.local` if present.
- Read `runbook/index.md` before non-trivial Jaxtitan work. If the task maps to
  an active stream or reference, read that file too.
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
- Never force-push, rewrite published history, or delete branches/tags without explicit approval.
- `master` is the single integration base. Do not invent or depend on a long-lived `dev` branch.
- An explicit request to implement, fix, build, update, or otherwise execute a change authorizes the normal task-scoped publication workflow described below. Do not stop for separate commit/push permission when the worktree is clean and the intended scope is already clear.
- Docs/runbook-only maintenance may be committed and pushed directly to `master` after link/diff validation when the user requested the change and the worktree is clean. If the work already lives on a task branch or PR, finish it there instead of moving it back to `master`.
- Code, config, test, dependency, workflow, and experiment changes use a short-lived branch from current `master`. After proportionate checks pass, commit, push, and open a draft PR as part of completing the implementation request.
- Scored model/optimizer mechanisms require a registry row before promotion. Correctness infrastructure and maintenance PRs use `no-run-required`.
- Merging a PR remains user-owned unless the user explicitly asks the agent to merge it.
- Ask before publication only when scope is mixed or ambiguous, the target is not the user's repository, a direct non-docs push to `master` would be required, or the operation is destructive/history-changing.

### Build And Run Discipline (High Severity)

- Only regenerate data, rerun expensive training, or launch cloud runs when the change actually requires it.
- Run narrow tests for the touched surface; run broader tests when changing shared training, logging, summary, config, or registry behavior.
- Do not use W&B-only evidence to debug or score a run.

### Output Completeness

- If asked for "full diff/log output", do not truncate.
- Provide complete copy/paste-ready commands, including `cd`, env vars, config paths, and flags.

### Runbook Protocol

- The tracked `runbook/` directory is shared project memory, not private scratch.
- Update the relevant `runbook/streams/*.md` file when work changes run status,
  validation results, artifact locations, known constraints, or next actions.
- Use headers like `## YYYY-MM-DD [codex] short title`.
- Write terse, factual entries. Prefer bullets and fenced commands over prose.
- Every substantive entry should include:
  - `Context`: what changed or what was being investigated.
  - `Commands`: exact commands run, with `cd`, env vars, config paths, and flags.
  - `Artifacts`: run dirs, config paths, manifest paths/hashes, profile paths,
    checkpoint selectors, eval/sample outputs, or commit hashes.
  - `Result`: observed pass/fail status, metrics, error messages, or measured
    numbers. Do not summarize a run as "works" without artifact evidence.
  - `Next`: the next concrete action or the reason no action remains.
- If no command was run, say that explicitly and record the source of the
  conclusion, such as code inspection, doc update, or design decision.
- When logging experiments, include local artifact evidence first:
  `run inspect`, `final.json`, `metrics/train.jsonl`, checkpoint eval/sample,
  profiler metadata, and registry rows when relevant. W&B links are optional
  mirrors and never sufficient on their own.
- When logging failures, include the failing command, the exact error line, the
  suspected cause, and the retry/remediation plan.
- When creating a new workstream, add it to `runbook/index.md` and seed it with
  the current objective, known state, and immediate next action.
- Do not record secrets, cloud IPs, hostnames, SSH keys, provider account
  details, or tokens.
- Move completed streams to `runbook/archive/YYYY/` only when the stream is
  genuinely done, and update the archive index.

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

1. **Baseline or mechanism branch** - Start a short-lived branch from current `master`.
2. **Config and preflight** - Add/update TOML, prepare data if needed, and run `uv run preflight configs/your_experiment.toml`.
3. **Training run** - Use `uv run experiment configs/your_experiment.toml` for runs intended to enter the comparison ladder.
4. **Artifact inspection** - Verify local metrics, summaries, checkpoint evals, CORE/inference artifacts, and W&B only as a mirror.
5. **Registry update** - Register the completed scored run in `runs/registry.jsonl`.
6. **Promotion** - Merge scored mechanisms to `master` only after they have a recorded score. Correctness infrastructure and maintenance PRs use `no-run-required`; docs/runbook-only maintenance may land directly on `master`.

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
