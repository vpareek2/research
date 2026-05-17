# Run Scorecard

- run: `2_mqa`
- run_dir: `runs/2_mqa`
- status: `unstable`
- decision_hint: `inspect`
- score: `23.7949`
- score_eligible: `True`
- baseline: `0_baseline`

## Run Score

- final_score: `23.7949`
- quality: `0.985518`
- training_efficiency: `1.02366`
- inference_efficiency: `1.05842`
- health: `0.599939`

## Model

- params: `116447616`
- layers: `12`
- hidden_size: `640`
- heads: `10`
- kv_heads: `1`
- seq_len: `2048`
- vocab_size: `50257`

## Training

- final_step: `30510`
- steps_completed: `30518`
- tokens_seen: `1999568896`
- configured_tokens: `2000027648`

## Training Native Validation

- final_train_loss: `3.44871`
- best_train_loss: `3.27605`
- first_val_loss: `11.2522`
- final_val_loss: `3.45121`
- best_val_loss: `3.45121`
- final_val_bpb: `1.11478`
- best_val_bpb: `1.11478`

## Health

- nan_count: `0`
- loss_spike_count: `0`
- grad_norm_spike_count: `7`
- final_train_val_gap: `0.0256851`
- final_train_loss_slope: `0.000104108`
- final_val_loss_slope: `-0.00103548`

## Speed

- avg_tokens_per_sec: `86942.5`
- avg_train_tokens_per_sec: `86942.5`
- wall_tokens_per_sec: `78351.3`
- logged_avg_train_tokens_per_sec: `11336.6`
- final_elapsed_sec: `25520.6`

## Performance

- final_mfu: `0.669852`
- logged_final_mfu: `0.669852`
- avg_mfu: `6.03674`
- wall_mfu: `5.44022`
- logged_avg_mfu: `0.787145`
- flops_per_token: `694337280`
- avg_train_tokens_per_gpu_hour: `3.12993e+08`
- peak_gpu_memory_bytes: `78419394560`
- avg_gpu_utilization_pct: `94.4089`
- avg_gpu_power_w: `431.333`

## Epiplexity Proxy

- train_bpb_auc: `1.18088e+09`
- train_bpb_auc_per_byte: `0.132819`
- val_bpb_auc: `4.47767e+07`
- val_bpb_auc_per_byte: `0.124576`

## Checkpoint Native Validation

- count: `1`
- latest_step: `30518`
- latest_loss: `3.45138`
- latest_bpb: `1.11484`
- best_step: `30518`
- best_loss: `3.45138`
- best_bpb: `1.11484`

## Training Domain Validation

domain         first_loss   final_loss    best_loss        delta    final_bpb     best_bpb
------------------------------------------------------------------------------------------
books             11.2345      3.83517      3.83303     -7.39933      1.34049      1.33974
code              11.7896      2.65039      2.53519     -9.13924      1.63535      1.56426
dialogue          11.3395      3.31282      3.31225     -8.02669        1.481      1.48058
docs              11.6026       2.3748      2.12426     -9.22782      1.68814      1.51003
knowledge          11.645       2.9168      2.44123     -8.72817      1.53512      1.28474
math               11.293      3.64698       3.6273     -7.64607      1.48166      1.47365
news               11.245      3.35717      3.35717     -7.88783      1.05504      1.05504
reasoning         11.2937      2.96452      2.94271     -8.32919      1.21587        1.207
web               11.2646      3.34821      3.34821     -7.91637      1.10326      1.10326

## Checkpoint Domain Validation

domain        latest_step  latest_loss    best_loss   latest_bpb     best_bpb
-----------------------------------------------------------------------------
books               30518      3.83924      3.83924      1.34191      1.34191
code                30518      2.66522      2.66522       1.6445       1.6445
dialogue            30518      3.31887      3.31887      1.48367      1.48367
docs                30518      2.39257      2.39257      1.70077      1.70077
knowledge           30518       2.9401       2.9401       1.5474       1.5474
math                30518       3.6476       3.6476       1.4819       1.4819
news                30518      3.35757      3.35757      1.05516      1.05516
reasoning           30518      2.96365      2.96365      1.21554      1.21554
web                 30518      3.34845      3.34845      1.10333      1.10333

## Benchmark CORE

- count: `1`
- latest_step: `30518`
- latest_core: `0.0803978`
- best_step: `30518`
- best_core: `0.0803978`

task                                  accuracy   centered   baseline   examples
---------------------------------------------------------------------------------
agi_eval_lsat_ar                      0.291304    0.11413         20        230
arc_challenge                         0.217577 -0.0432309         25       1172
arc_easy                              0.361111   0.148148         25       2376
bigbench_cs_algorithms                0.381818   0.381818          0       1320
bigbench_dyck_languages                  0.012      0.012          0       1000
bigbench_language_identification         0.253   0.178218        9.1      10000
bigbench_operators                   0.0428571  0.0428571          0        210
bigbench_qa_wikidata                  0.208159   0.208159          0      20321
bigbench_repeat_copy_logic                   0          0          0         32
boolq                                 0.479817  -0.368904         62       3270
commonsense_qa                        0.268632  0.0857903         20       1221
copa                                      0.59       0.18         50        100
coqa                                 0.0424652  0.0424652          0       7983
hellaswag                             0.289584  0.0527783         25      10042
hellaswag_zeroshot                    0.290878  0.0545044         25      10042
jeopardy                            0.000472367 0.000472367          0       2117
lambada_openai                        0.264506   0.264506          0       5153
openbook_qa                              0.272  0.0293333         25        500
piqa                                  0.612323   0.224646         50       1834
squad                               0.00567644 0.00567644          0      10570
winograd                              0.556777   0.113553         50        273
winogrande                            0.520916  0.0418311         50       1267

## Inference Benchmark

- latest_step: `30518`
- mode: `kv_cache_decode_loop_prefill`
- decode_tokens_per_sec: `114.989`
- prefill_tokens_per_sec: `1182.42`
- ttft_sec: `0.438982`
