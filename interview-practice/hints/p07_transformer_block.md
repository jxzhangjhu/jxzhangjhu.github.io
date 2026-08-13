# Hints · p07 A full pre-norm block

Read one at a time.

## Level 1

Complete p01, p05, and p06 first; this exercise deliberately reuses those modules.

## Level 2

Two residual lines: `x = x + attn(norm1(x))`, then `x = x + mlp(norm2(x))`.

## Level 3

Pre-norm normalises each sublayer input. A complete language model also needs a final norm before `lm_head`.
