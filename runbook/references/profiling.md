# Profiling Reference

Jaxtitan supports programmatic JAX profiling through `[profiling]` in run TOML.
Trace artifacts are local run artifacts and are not uploaded to W&B.

## Config Pattern

```toml
[profiling]
enabled = true
trace_start_step = 4
trace_steps = 2
create_perfetto_trace = true
create_perfetto_link = false
```

## Commands

```sh
cd /home/veer/Master/projects/research
uv run jaxtitan run preflight configs/jaxtitan/<config>.toml
uv run jaxtitan run train --overwrite configs/jaxtitan/<config>.toml
uv run jaxtitan run inspect runs/<run_id>
cat runs/<run_id>/diagnostics/profiling.json
```

## What To Inspect

- `diagnostics/profiling.json`;
- `profiles/plugins/profile/.../perfetto_trace.json.gz`;
- train step annotations: data, placement, train step, metrics sync, eval,
  checkpoint;
- unexpected large collectives or replication;
- checkpoint stalls and metrics sync stalls.
