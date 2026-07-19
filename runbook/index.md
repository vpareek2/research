# Jaxtitan Runbook Index

This runbook is the shared memory for humans and agents working on Jaxtitan.
It records active investigations, reusable workflows, run artifacts, and
current conclusions. Keep it factual and command-oriented.

## Rules

- Read this index before starting non-trivial Jaxtitan work.
- Update the relevant stream when work changes project state, run status, or
  conclusions.
- Use actor/date headers: `## YYYY-MM-DD [codex] short title`.
- Prefer local artifact paths, config names, commit hashes, and exact commands.
- Do not write secrets, cloud IPs, hostnames, SSH keys, tokens, or private
  account details.
- W&B is a mirror. Local artifacts, configs, checkpoints, summaries, and
  registry rows are the source of truth.
- If a stream is complete, move it to `archive/YYYY/` and update the archive
  index.

## Active Streams

- [Tensor parallel completion](streams/tp_completion.md)
- [MoE parallelism](streams/moe_parallelism.md)
- [Cloud validation queue](streams/cloud_validation_queue.md)
- [Distributed Muon handoff](streams/distributed_muon_handoff.md)
- [Kernels and automatic backend](streams/kernels.md)
- [Data and experiment UX](streams/data_and_experiment_ux.md)

## References

- [Cloud validation](references/cloud_validation.md)
- [Data workflow](references/data_workflow.md)
- [Profiling](references/profiling.md)
- [Registry and scoring](references/registry_scoring.md)

## Recently Completed

- `9a046a4` added semantic data-axis RDEP and dense tensor-parallel semantics,
  including vocab-parallel LM head and exact sharded loss.

## Archive

- [2026 archive index](archive/2026/index.md)
