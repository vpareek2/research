# Checkpoint Eval

- run: `runs/1_mha`
- checkpoint_step: `30518`

## Native Validation

- eval_steps: `10`
- examples: `320`
- tokens: `655360`
- loss: `3.406501`
- ppl: `30.159527`
- bpb: `1.100353`
- bytes: `2922225`
- elapsed_sec: `14.744572`
- tokens_per_sec: `44447.54`

## Domain Validation

domain             loss        ppl        bpb     tokens
----------------------------------------------------------
web            3.305259  27.255606   1.089145      65536
knowledge      2.463650  11.747615   1.295986      65536
books          3.779820  43.808167   1.321144      65536
news           3.315887  27.546829   1.042016      65536
code           2.416076  11.201814   1.490740      65536
math           3.528365  34.068233   1.433399      65536
reasoning      2.833518  17.005180   1.162362      65536
docs           2.020013   7.538420   1.435943      65536
dialogue       3.271360  26.347151   1.461605      65536
