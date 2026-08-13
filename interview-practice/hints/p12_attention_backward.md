# Hints · p12 Attention backward by hand

Read one at a time.

## Level 1

From `O = P V`: `dV = P^T dO`, `dP = dO V^T`.

## Level 2

Softmax VJP: `dS = P * (dP - rowsum(dP * P))`.

## Level 3

From `S = scale QK^T`: `dQ = scale dS K`, `dK = scale dS^T Q`. Masked entries already have P=0.
