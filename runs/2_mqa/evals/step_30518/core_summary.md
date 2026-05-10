# CORE Eval

- run: `runs/2_mqa`
- checkpoint_step: `30518`
- max_per_task: `-1`
- core: `0.080398`
- elapsed_sec: `2838.40`

task                                       acc   centered   baseline   examples    skipped
--------------------------------------------------------------------------------------------
hellaswag_zeroshot                    0.290878   0.054504      25.00      10042          0
jeopardy                              0.000472   0.000472       0.00       2117          0
bigbench_qa_wikidata                  0.208159   0.208159       0.00      20321          0
arc_easy                              0.361111   0.148148      25.00       2376          0
arc_challenge                         0.217577  -0.043231      25.00       1172          0
copa                                  0.590000   0.180000      50.00        100          0
commonsense_qa                        0.268632   0.085790      20.00       1221          0
piqa                                  0.612323   0.224646      50.00       1834          4
openbook_qa                           0.272000   0.029333      25.00        500          0
lambada_openai                        0.264506   0.264506       0.00       5153          0
hellaswag                             0.289584   0.052778      25.00      10042          0
winograd                              0.556777   0.113553      50.00        273          0
winogrande                            0.520916   0.041831      50.00       1267          0
bigbench_dyck_languages               0.012000   0.012000       0.00       1000          0
agi_eval_lsat_ar                      0.291304   0.114130      20.00        230          0
bigbench_cs_algorithms                0.381818   0.381818       0.00       1320          0
bigbench_operators                    0.042857   0.042857       0.00        210          0
bigbench_repeat_copy_logic            0.000000   0.000000       0.00         32          0
squad                                 0.005676   0.005676       0.00      10570          0
coqa                                  0.042465   0.042465       0.00       7983          0
boolq                                 0.479817  -0.368904      62.00       3270          0
bigbench_language_identification      0.253000   0.178218       9.10      10000          0
