# Hints · p07 A full pre-norm block

Read one at a time.

## Level 1

Two lines of forward, each a residual around a normalised sublayer.

## Level 2

x = x + attn(norm1(x)); x = x + mlp(norm2(x)).

## Level 3

Pre-norm normalises the sublayer input; a full model also needs a final norm before lm_head.
