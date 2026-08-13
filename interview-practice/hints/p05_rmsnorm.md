# Hints · p05 RMSNorm

Read one at a time. If the first is enough, stop.

## Level 1

RMS(x) = sqrt(mean(x^2) + eps) over the last dimension, keepdim=True.

## Level 2

There is no mean subtraction and no bias term — that is the whole difference from LayerNorm.

## Level 3

Promote fp16/bf16 reductions to fp32, but do not demote float64; cast the normalised activations back to the input dtype.
