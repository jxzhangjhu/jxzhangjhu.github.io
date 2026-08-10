# Hints · p12 Attention backward by hand

Read one at a time.

## Level 1

d_v = P^T d_out and d_p = d_out V^T are the easy two.

## Level 2

The softmax VJP is d_s = P * (d_p - rowsum(d_p * P)).

## Level 3

Masked positions have P = 0, so their gradient is zeroed automatically — no need to re-apply the mask.
