# Hints · p02 KV cache and incremental decode

Read one at a time. If the first is enough, stop.

## Level 1

The query is only for the new token; keys and values are the concatenation of the cache and the new step.

## Level 2

During decode there is exactly one query row attending to all t keys, so no causal mask is needed at all.

## Level 3

For T_new > 1 with a non-empty cache the mask must start at row offset T_full - T_new: `torch.tril(ones(T_new, T_full), diagonal=T_full - T_new)`.
