# Hints · d02 Head-merge failure

Read one level at a time.

## Level 1

Inspect both shape and stride before and after swapping the head and time axes.

## Level 2

`transpose` returns a non-contiguous view. `view` cannot reinterpret that storage layout.

## Level 3

Transpose to `(B,T,H,Dh)`, then call `.contiguous().view(B,T,H*Dh)` (or use
`.reshape(B,T,H*Dh)`).
