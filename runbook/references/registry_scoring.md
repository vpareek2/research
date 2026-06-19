# Registry And Scoring Reference

Local artifacts are canonical. W&B is a mirror.

## Research Run Checklist

Before a result should enter the comparison ladder:

- config is committed or reproducibly identified;
- data manifest exists and passes checksum validation;
- run completed with local `final.json`;
- eval artifacts are present;
- checkpoint eval/sample evidence exists when relevant;
- scoring command was run;
- `runs/registry.jsonl` has the scored row or the change is explicitly
  maintenance-only / no-run-required.

## Local First

Inspect local artifacts before looking at W&B:

```sh
cd /home/veer/Master/projects/research
uv run jaxtitan run inspect runs/<run_id>
cat runs/<run_id>/final.json
tail -n 1 runs/<run_id>/metrics/train.jsonl
```

Use W&B only to compare mirrored scalars after local artifacts are verified.
