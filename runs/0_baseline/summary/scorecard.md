# Run Scorecard

- run: `0_baseline`
- run_dir: `runs/0_baseline`
- status: `unstable`
- decision_hint: `inspect`
- score: `23.4997`
- score_eligible: `True`
- baseline: `0_baseline`

## Run Score

- final_score: `23.4997`
- quality: `1`
- training_efficiency: `1`
- inference_efficiency: `1`
- health: `0.599931`

## Model

- params: `117430656`
- layers: `12`
- hidden_size: `640`
- heads: `10`
- kv_heads: `2`
- seq_len: `2048`
- vocab_size: `50257`

## Training

- final_step: `30510`
- steps_completed: `30518`
- tokens_seen: `1999568896`
- configured_tokens: `2000027648`

## Training Native Validation

- final_train_loss: `3.43303`
- best_train_loss: `3.25957`
- first_val_loss: `11.3295`
- final_val_loss: `3.43368`
- best_val_loss: `3.43368`
- final_val_bpb: `1.10911`
- best_val_bpb: `1.10911`

## Health

- nan_count: `0`
- loss_spike_count: `0`
- grad_norm_spike_count: `6`
- final_train_val_gap: `0.0242109`
- final_train_loss_slope: `0.000131279`
- final_val_loss_slope: `-0.00101368`

## Speed

- avg_tokens_per_sec: `83382.9`
- avg_train_tokens_per_sec: `83382.9`
- wall_tokens_per_sec: `74139.2`
- logged_avg_train_tokens_per_sec: `11095.5`
- final_elapsed_sec: `26970.5`

## Performance

- final_mfu: `0.65083`
- logged_final_mfu: `1.55329`
- avg_mfu: `5.83877`
- wall_mfu: `5.19149`
- logged_avg_mfu: `1.85428`
- flops_per_token: `700235520`
- avg_train_tokens_per_gpu_hour: `3.00178e+08`
- peak_gpu_memory_bytes: `78417297408`
- avg_gpu_utilization_pct: `96.7536`
- avg_gpu_power_w: `408.197`

## Epiplexity Proxy

- train_bpb_auc: `1.18689e+09`
- train_bpb_auc_per_byte: `0.133495`
- val_bpb_auc: `4.52415e+07`
- val_bpb_auc_per_byte: `0.125869`

## Checkpoint Native Validation

- count: `1`
- latest_step: `30518`
- latest_loss: `3.4339`
- latest_bpb: `1.10919`
- best_step: `30518`
- best_loss: `3.4339`
- best_bpb: `1.10919`

## Training Domain Validation

domain         first_loss   final_loss    best_loss        delta    final_bpb     best_bpb
------------------------------------------------------------------------------------------
books             11.3139      3.80901       3.8083     -7.50485      1.33135       1.3311
code              11.1687      2.59028      2.54947     -8.57842      1.59831      1.57313
dialogue          11.3267      3.29617      3.28662     -8.03049      1.47279      1.46876
docs              11.1999      2.13059      2.11185     -9.06935      1.51452       1.5012
knowledge         11.1179      2.52596      2.41982     -8.59189       1.3293      1.27338
math              11.3412      3.66098      3.63861     -7.68017       1.4873      1.47824
news              11.3232      3.34305      3.34305     -7.98016      1.05056      1.05056
reasoning         11.3255       2.9271      2.91299     -8.39844      1.20055      1.19484
web               11.3484      3.33311      3.33311     -8.01532      1.09833      1.09833

## Checkpoint Domain Validation

domain        latest_step  latest_loss    best_loss   latest_bpb     best_bpb
-----------------------------------------------------------------------------
books               30518      3.81242      3.81242      1.33254      1.33254
code                30518      2.59604      2.59604      1.60187      1.60187
dialogue            30518      3.29899      3.29899      1.47413      1.47413
docs                30518      2.14309      2.14309      1.52341      1.52341
knowledge           30518      2.53268      2.53268      1.33286      1.33286
math                30518      3.65886      3.65886      1.48643      1.48643
news                30518      3.34333      3.34333      1.05067      1.05067
reasoning           30518      2.92448      2.92448       1.1995       1.1995
web                 30518      3.33198      3.33198      1.09796      1.09796

## Benchmark CORE

- count: `1`
- latest_step: `30518`
- latest_core: `0.0885269`
- best_step: `30518`
- best_core: `0.0885269`

task                                  accuracy   centered   baseline   examples
---------------------------------------------------------------------------------
agi_eval_lsat_ar                      0.230435  0.0380435         20        230
arc_challenge                         0.208191 -0.0557452         25       1172
arc_easy                              0.356481   0.141975         25       2376
bigbench_cs_algorithms                0.366667   0.366667          0       1320
bigbench_dyck_languages                  0.004      0.004          0       1000
bigbench_language_identification        0.2588   0.184598        9.1      10000
bigbench_operators                         0.1        0.1          0        210
bigbench_qa_wikidata                  0.195906   0.195906          0      20321
bigbench_repeat_copy_logic                   0          0          0         32
boolq                                 0.619266 -0.00193143         62       3270
commonsense_qa                        0.232596  0.0407453         20       1221
copa                                      0.54       0.08         50        100
coqa                                 0.0437179  0.0437179          0       7983
hellaswag                             0.289982  0.0533094         25      10042
hellaswag_zeroshot                    0.291077    0.05477         25      10042
jeopardy                            0.000944733 0.000944733          0       2117
lambada_openai                        0.270134   0.270134          0       5153
openbook_qa                              0.248 -0.00266667         25        500
piqa                                  0.618866   0.237732         50       1834
squad                               0.00350047 0.00350047          0      10570
winograd                              0.586081   0.172161         50        273
winogrande                            0.509866  0.0197316         50       1267

## Inference Benchmark

- latest_step: `30518`
- mode: `kv_cache_decode_loop_prefill`
- decode_tokens_per_sec: `108.289`
- prefill_tokens_per_sec: `1121.5`
- ttft_sec: `0.462662`
