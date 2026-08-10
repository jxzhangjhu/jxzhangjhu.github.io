# Hints · p01 Causal multi-head attention

Read one at a time. If the first is enough, stop.

## Level 1

The shape journey is (B,T,C) -> (B,T,n_heads,d_head) -> (B,n_heads,T,d_head), and back again at the end.

## Level 2

Mask before the softmax, additively, with -inf. Multiplying by zero after the softmax leaves the masked positions in the denominator.

## Level 3

After `att @ v` you have (B,n_heads,T,d_head). `.transpose(1,2)` makes it non-contiguous, so `.view()` raises — call `.contiguous()` first, or use `.reshape()`.
