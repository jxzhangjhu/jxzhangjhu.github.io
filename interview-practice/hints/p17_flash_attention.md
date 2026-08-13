# Hints · p17 Tiled FlashAttention forward

Read one at a time.

## Level 1

Outer-loop over query tiles, inner-loop over KV tiles; keep `m`, `l`, and `acc` per query row.

## Level 2

Use the recurrence from p16 and place V in the numerator update. For fp16/bf16 inputs, compute block scores and keep `m`, `l`, and `acc` in float32.

## Level 3

For causality, skip tiles wholly above the diagonal and element-mask overlapping tiles; guard the `-inf - -inf` fully-masked case, then cast the output back to the input dtype.
