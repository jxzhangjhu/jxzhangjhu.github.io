# Hints · p04 Rotary position embeddings

Read one at a time. If the first is enough, stop.

## Level 1

Frequencies are `base ** (-arange(0, d, 2) / d)`; the angle at position m is m * freq.

## Level 2

Rotating the pair (x0, x1) by theta gives (x0*cos - x1*sin, x0*sin + x1*cos).

## Level 3

Slice the pairs with `x[..., 0::2]` and `x[..., 1::2]`, then interleave back with `torch.stack([...], -1).flatten(-2)`.
