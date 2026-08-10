# Hints · p03 Grouped-query attention

Read one at a time.

## Level 1

Q gets n_heads, K and V get n_kv_heads. Only the projection output sizes differ.

## Level 2

repeat_interleave(n_rep, dim=1) expands the KV heads to match the query heads before the matmul.

## Level 3

With a cache the query block starts at T_full - T, so the mask needs diagonal=T_full - T.
