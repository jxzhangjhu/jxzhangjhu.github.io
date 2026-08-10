# Hints · p18 LoRA with a lossless merge

Read one at a time. If the first is enough, stop.

## Level 1

W' = W + (alpha/r) * B @ A, with A of shape (r, in) and B of shape (out, r).

## Level 2

Initialise A randomly (kaiming) and B to zeros, so B@A = 0 and the adapter starts as a no-op.

## Level 3

Freeze the base with `base.weight.requires_grad_(False)`; the merge is just `W + (alpha/r) * B @ A`.
