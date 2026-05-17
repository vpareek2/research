# Checkpoint Eval

- run: `runs/2_mqa`
- checkpoint_step: `30518`

## Native Validation

- eval_steps: `10`
- examples: `320`
- tokens: `655360`
- loss: `3.451382`
- ppl: `31.543945`
- bpb: `1.114844`
- bytes: `2922225`
- elapsed_sec: `15.511082`
- tokens_per_sec: `42251.08`

## Domain Validation

domain             loss        ppl        bpb     tokens
----------------------------------------------------------
web            3.348453  28.458677   1.103330      65536
knowledge      2.940099  18.917719   1.547400      65536
books          3.839238  46.490055   1.341912      65536
news           3.357574  28.719433   1.055163      65536
code           2.665222  14.371146   1.644504      65536
math           3.647596  38.382271   1.481902      65536
reasoning      2.963646  19.368464   1.215539      65536
docs           2.392566  10.941539   1.700775      65536
dialogue       3.318868  27.629061   1.483673      65536
