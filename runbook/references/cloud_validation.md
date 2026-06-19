# Cloud Validation Reference

Use this workflow for GPU validation. Do not commit cloud IPs, hostnames, SSH
details, tokens, or provider account metadata.

## Setup

```sh
sudo apt-get update
sudo apt-get install -y git curl build-essential pkg-config

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

git clone <repo-url> research
cd research
git checkout jaxtitan
uv sync
```

Verify devices:

```sh
cd ~/research
uv run python - <<'PY'
import jax
print("backend:", jax.default_backend())
print("process_count:", jax.process_count())
print("process_index:", jax.process_index())
for device in jax.devices():
    print(device)
PY
```

## Data

Prefer preparing or streaming data on the cloud host.

```sh
cd ~/research
uv run jaxtitan data prepare --overwrite configs/data/tinystories_gpt2_smoke.toml
uv run jaxtitan data inspect data/tinystories_gpt2_smoke/manifest.json \
  --tokenizer gpt2 \
  --verify-checksums \
  --seq-len 512
uv run jaxtitan data check data/tinystories_gpt2_smoke/manifest.json \
  --tokenizer gpt2 \
  --verify-checksums
```

If preflight reports the validation split is too small, increase
`val_fraction` in the data config and regenerate the manifest.

## Run Pattern

```sh
cd ~/research
uv run jaxtitan config check configs/jaxtitan/<config>.toml
uv run jaxtitan run preflight configs/jaxtitan/<config>.toml
uv run jaxtitan run train --overwrite configs/jaxtitan/<config>.toml
uv run jaxtitan run inspect runs/<run_id>
uv run jaxtitan eval checkpoint runs/<run_id> --checkpoint latest --json
uv run jaxtitan sample checkpoint runs/<run_id> \
  --checkpoint latest \
  --prompt-ids "15496,11" \
  --max-new-tokens 8 \
  --top-k 1 \
  --json
```

## What To Record In Streams

- commit hash;
- config path and run id;
- hardware type and `jax.devices()` summary, without host/IP details;
- data manifest path and manifest hash;
- preflight result;
- final inspect output;
- checkpoint eval JSON;
- sample JSON;
- profiling artifact paths if profiling was enabled;
- any errors and exact remediation.
