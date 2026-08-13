# Hints · p20 DPO loss

Read one at a time.

## Level 1

Compute policy and reference chosen-minus-rejected log-ratios.

## Level 2

The margin is `(pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)`.

## Level 3

Return `-logsigmoid(beta * margin).mean()`; zero margin must give exactly log(2).
