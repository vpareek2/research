I did a repo pass and ran the tests: `44 passed in 9.77s`. No files changed.

Main things that look missing for a serious ~100M / 2B-token pretraining repo:

1. **A real scale config** Done
   Only smoke configs exist. I’d add a checked-in `configs/100m_2b.toml` with model size, global batch, context length, total steps, checkpoint cadence, and expected tokens. Right now the core config shape is there in [config.py](/home/veer/Master/projects/research/config.py:35), but no production-ish example.

2. **LR schedule / warmup**
   Training uses a constant `train.lr` through Muon in [pretrain.py](/home/veer/Master/projects/research/pretrain.py:130), and logs it statically at [pretrain.py](/home/veer/Master/projects/research/pretrain.py:168). For 2B tokens, I’d expect warmup + cosine/linear decay + min LR.

3. **Precision controls**
   There’s no config for `bf16`/`fp32`, parameter dtype, activation dtype, or matmul precision. The model path in [model.py](/home/veer/Master/projects/research/model.py:93) appears to default to normal JAX/Flax behavior. That is fine for smoke, but expensive and ambiguous for actual runs.

4. **Multi-device / sharding story**
   Training is single-process/simple JIT via [pretrain.py](/home/veer/Master/projects/research/pretrain.py:55). If you intend 2B tokens on multiple GPUs, you’ll want explicit data parallel/sharding guidance or implementation.

5. **Data scale ergonomics**
   Prepared token mmap support is good, and provenance logging is strong. But `prepare_data.py` can materialize text collections before writing tokens in [prepare_data.py](/home/veer/Master/projects/research/prepare_data.py:167), which may become painful at larger scale. I’d also add manifest validation on load: dtype, tokenizer name, checksum optional, split sanity.

6. **Run reproducibility metadata**
   Metadata logs library versions/devices, but not git commit, command line, hostname, env knobs, or lockfile hash. That would strengthen [logs.py](/home/veer/Master/projects/research/logs.py:136) for comparing research runs.

7. **Config validation**
   `model.seq_len` and `train.seq_len` can diverge. The model only rejects overlong sequences at runtime in [model.py](/home/veer/Master/projects/research/model.py:112). I’d validate consistency at config load.

8. **Evaluation beyond val loss**
   Current eval is just mean LM loss in [pretrain.py](/home/veer/Master/projects/research/pretrain.py:71). For small models, even a lean harness for fixed prompts, TinyStories-style eval, or a couple lightweight downstream probes would help.

9. **CI**
   Tests are solid for repo size, but I don’t see CI files. A minimal GitHub Actions workflow for `uv sync`, `pytest`, and `py_compile` would make this safer to iterate on.

The strongest parts already present are local-first artifacts, batch provenance via [data.py](/home/veer/Master/projects/research/data.py:20), checkpointable Grain iteration, and focused tests. The biggest practical gap is moving from “smoke trainer” to “reproducible actual pretraining run”: scale config, LR schedule, dtype, and metadata.