# Inference Benchmark

- run: `runs/1_mha`
- checkpoint_step: `30518`
- mode: `kv_cache_decode_loop_prefill`
- batch_size: `1`
- prompt_count: `3`
- prompt_tokens: `512`
- decode_tokens: `128`
- prefill_tokens_per_sec: `1145.06`
- decode_tokens_per_sec: `115.46`
- ttft_sec: `0.453049`
- first_decode_sec: `0.005909`
