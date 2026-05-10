# Run Scorecard

- run: `1_mha`
- run_dir: `runs/1_mha`
- status: `unstable`
- decision_hint: `inspect`
- score: `24.2572`
- score_eligible: `True`
- baseline: `0_baseline`

## Run Score

- final_score: `24.2572`
- quality: `1.01596`
- training_efficiency: `1.05889`
- inference_efficiency: `1.04592`
- health: `0.599984`

## Model

- params: `125294976`
- layers: `12`
- hidden_size: `640`
- heads: `10`
- kv_heads: `10`
- seq_len: `2048`
- vocab_size: `50257`

## Training

- final_step: `30510`
- steps_completed: `30518`
- tokens_seen: `1999568896`
- configured_tokens: `2000027648`

## Training Native Validation

- final_train_loss: `3.40497`
- best_train_loss: `3.22583`
- first_val_loss: `11.3252`
- final_val_loss: `3.40646`
- best_val_loss: `3.40646`
- final_val_bpb: `1.10034`
- best_val_bpb: `1.10034`

## Health

- nan_count: `0`
- loss_spike_count: `0`
- grad_norm_spike_count: `5`
- final_train_val_gap: `0.0216722`
- final_train_loss_slope: `0.000124622`
- final_val_loss_slope: `-0.00101721`

## Speed

- avg_tokens_per_sec: `87888.7`
- avg_train_tokens_per_sec: `87888.7`
- wall_tokens_per_sec: `79404.5`
- logged_avg_train_tokens_per_sec: `11427.8`
- final_elapsed_sec: `25182.1`

## Performance

- final_mfu: `0.726546`
- logged_final_mfu: `0.726546`
- avg_mfu: `6.56899`
- wall_mfu: `5.93486`
- logged_avg_mfu: `0.854139`
- flops_per_token: `747421440`
- avg_train_tokens_per_gpu_hour: `3.16399e+08`
- peak_gpu_memory_bytes: `78419394560`
- avg_gpu_utilization_pct: `96.9528`
- avg_gpu_power_w: `437.98`

## Epiplexity Proxy

- train_bpb_auc: `1.20787e+09`
- train_bpb_auc_per_byte: `0.135855`
- val_bpb_auc: `4.61006e+07`
- val_bpb_auc_per_byte: `0.128259`

## Checkpoint Native Validation

- count: `1`
- latest_step: `30518`
- latest_loss: `3.4065`
- latest_bpb: `1.10035`
- best_step: `30518`
- best_loss: `3.4065`
- best_bpb: `1.10035`

## Training Domain Validation

domain         first_loss   final_loss    best_loss        delta    final_bpb     best_bpb
------------------------------------------------------------------------------------------
books              11.299       3.7821       3.7808     -7.51688      1.32194      1.32149
code              11.3942      2.41455      2.41455     -8.97965       1.4898       1.4898
dialogue          11.3102      3.26675      3.26631     -8.04341      1.45949      1.45938
docs              11.4959      2.02059      1.99296     -9.47532      1.43636      1.41668
knowledge         11.6753       2.4979      2.37354      -9.1774      1.31408      1.24866
math              11.3556      3.53198      3.52334     -7.82362      1.43489      1.43136
news               11.317      3.31537      3.31537     -8.00159      1.04183      1.04183
reasoning         11.3935      2.84058      2.82963     -8.55291      1.16526      1.16084
web               11.3356      3.30475      3.30475     -8.03084      1.08898      1.08898

## Checkpoint Domain Validation

domain        latest_step  latest_loss    best_loss   latest_bpb     best_bpb
-----------------------------------------------------------------------------
books               30518      3.77982      3.77982      1.32114      1.32114
code                30518      2.41608      2.41608      1.49074      1.49074
dialogue            30518      3.27136      3.27136      1.46161      1.46161
docs                30518      2.02001      2.02001      1.43594      1.43594
knowledge           30518      2.46365      2.46365      1.29599      1.29599
math                30518      3.52837      3.52837       1.4334       1.4334
news                30518      3.31589      3.31589      1.04202      1.04202
reasoning           30518      2.83352      2.83352      1.16236      1.16236
web                 30518      3.30526      3.30526      1.08914      1.08914

## Benchmark CORE

- count: `1`
- latest_step: `30518`
- latest_core: `0.0976392`
- best_step: `30518`
- best_core: `0.0976392`

task                                  accuracy   centered   baseline   examples
---------------------------------------------------------------------------------
agi_eval_lsat_ar                      0.252174  0.0652174         20        230
arc_challenge                         0.211604 -0.0511945         25       1172
arc_easy                              0.360269   0.147026         25       2376
bigbench_cs_algorithms                0.382576   0.382576          0       1320
bigbench_dyck_languages                  0.026      0.026          0       1000
bigbench_language_identification        0.2524   0.177558        9.1      10000
bigbench_operators                   0.0761905  0.0761905          0        210
bigbench_qa_wikidata                  0.246002   0.246002          0      20321
bigbench_repeat_copy_logic                   0          0          0         32
boolq                                 0.580428  -0.104136         62       3270
commonsense_qa                         0.28665   0.108313         20       1221
copa                                      0.61       0.22         50        100
coqa                                 0.0656395  0.0656395          0       7983
hellaswag                             0.290679  0.0542389         25      10042
hellaswag_zeroshot                    0.292173  0.0562305         25      10042
jeopardy                            0.000472367 0.000472367          0       2117
lambada_openai                        0.282748   0.282748          0       5153
openbook_qa                              0.278  0.0373333         25        500
piqa                                  0.630316   0.260632         50       1834
squad                               0.00473037 0.00473037          0      10570
winograd                              0.534799  0.0695971         50        273
winogrande                            0.511444  0.0228887         50       1267

## Inference Benchmark

- latest_step: `30518`
- mode: `kv_cache_decode_loop_prefill`
- decode_tokens_per_sec: `115.462`
- prefill_tokens_per_sec: `1145.06`
- ttft_sec: `0.453049`
