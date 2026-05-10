# CORE Eval

- run: `runs/1_mha`
- checkpoint_step: `30518`
- max_per_task: `-1`
- core: `0.097639`
- elapsed_sec: `2614.40`

task                                       acc   centered   baseline   examples    skipped
--------------------------------------------------------------------------------------------
hellaswag_zeroshot                    0.292173   0.056230      25.00      10042          0
jeopardy                              0.000472   0.000472       0.00       2117          0
bigbench_qa_wikidata                  0.246002   0.246002       0.00      20321          0
arc_easy                              0.360269   0.147026      25.00       2376          0
arc_challenge                         0.211604  -0.051195      25.00       1172          0
copa                                  0.610000   0.220000      50.00        100          0
commonsense_qa                        0.286650   0.108313      20.00       1221          0
piqa                                  0.630316   0.260632      50.00       1834          4
openbook_qa                           0.278000   0.037333      25.00        500          0
lambada_openai                        0.282748   0.282748       0.00       5153          0
hellaswag                             0.290679   0.054239      25.00      10042          0
winograd                              0.534799   0.069597      50.00        273          0
winogrande                            0.511444   0.022889      50.00       1267          0
bigbench_dyck_languages               0.026000   0.026000       0.00       1000          0
agi_eval_lsat_ar                      0.252174   0.065217      20.00        230          0
bigbench_cs_algorithms                0.382576   0.382576       0.00       1320          0
bigbench_operators                    0.076190   0.076190       0.00        210          0
bigbench_repeat_copy_logic            0.000000   0.000000       0.00         32          0
squad                                 0.004730   0.004730       0.00      10570          0
coqa                                  0.065639   0.065639       0.00       7983          0
boolq                                 0.580428  -0.104136      62.00       3270          0
bigbench_language_identification      0.252400   0.177558       9.10      10000          0
