# Hints · d07 Attention-scale failure

Read one level at a time.

## Level 1

Ask which dimension is summed inside one query-key dot product after heads are split.

## Level 2

The variance of that dot product grows with the per-head width, not the model width.

## Level 3

Divide scores by `math.sqrt(d_head)`, where `d_head = d_model // n_heads`.
