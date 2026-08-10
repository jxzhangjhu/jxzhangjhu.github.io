# Hints · p05 RMSNorm

Read one at a time. If the first is enough, stop.

## Level 1

RMS(x) = sqrt(mean(x^2) + eps) over the last dimension, keepdim=True.

## Level 2

There is no mean subtraction and no bias term — that is the whole difference from LayerNorm.

## Level 3

Compute the reduction in fp32 (`x.float()`) and cast back with `.type_as(x)`; a bf16 sum over 64+ squared values loses too much precision.
