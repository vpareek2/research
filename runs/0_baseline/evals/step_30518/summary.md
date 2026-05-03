# Checkpoint Eval

- run: `runs/0_baseline`
- checkpoint_step: `30518`

## Native Validation

- eval_steps: `10`
- examples: `320`
- tokens: `655360`
- loss: `3.433897`
- ppl: `30.997189`
- bpb: `1.109187`
- bytes: `2922225`
- elapsed_sec: `16.958864`
- tokens_per_sec: `38644.10`

## Domain Validation

domain             loss        ppl        bpb     tokens
----------------------------------------------------------
web            3.331980  27.993713   1.097956      65536
knowledge      2.532681  12.587205   1.332857      65536
books          3.812421  45.259895   1.332539      65536
news           3.343330  28.313250   1.050673      65536
code           2.596045  13.410588   1.601867      65536
math           3.658864  38.817211   1.486435      65536
reasoning      2.924478  18.624508   1.199497      65536
docs           2.143092   8.525763   1.523409      65536
dialogue       3.298990  27.085276   1.474132      65536
