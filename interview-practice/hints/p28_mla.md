# Hints · p28 Multi-head latent attention and compressed cache

Read one at a time.

## Level 1

The content cache is `c = w_down(x)` with shape (B,T,kv_rank); reconstruct non-positional K and V from c only when needed.

## Level 2

Use a separate shared `k_rope` of shape (B,1,T,rope_dim). Split Q into `q_nope` and `q_rope`, rotate only the RoPE parts, and concatenate before QK^T.

## Level 3

For cached decode, rotate new Q/K with positions `past:past+T`, append c along axis 1 and k_rope along axis 2, then use a causal mask with diagonal=T_full-T.
