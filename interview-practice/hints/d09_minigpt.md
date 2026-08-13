# Hints · d09 Debug miniGPT

Read one level at a time and rerun the tests after every change.

## Level 1

Check four invariants independently: absolute positions vary across time; attention
matches PyTorch SDPA; one training step changes parameters; cached and uncached logits
match. Print shape suffixes (`BHTD`, `BTD`) at attention boundaries.

## Level 2

Localise the failures to position indices, mask/softmax ordering, the head-to-time layout,
and the optimiser lifecycle. For the cache follow-up, derive the new token's absolute
position from cache length.

## Level 3

Use `arange(T)` for positions; apply additive `-inf` masking before softmax; merge heads
with `transpose(1,2).contiguous().view(...)`; call `opt.step()` after `backward()`.
In `forward_cached`, cache K/V per layer, embed positions from `past:past+T`, and use a
rectangular causal mask with `diagonal=T_full-T`.
