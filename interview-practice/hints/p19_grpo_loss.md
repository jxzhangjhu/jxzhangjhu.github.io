# Hints · p19 GRPO objective

Read one at a time.

## Level 1

Advantage: reshape rewards to (-1, G), standardise within the group with population std (`correction=0`), reshape back, and broadcast over tokens.

## Level 2

ratio = (logp - logp_old).exp(); the clipped surrogate is -min(ratio*adv, clamp(ratio)*adv).

## Level 3

k3 KL: with log_ratio = logp_ref - logp, it is log_ratio.exp() - log_ratio - 1. Average valid tokens within each completion, then average valid completions so length does not change example weight.
