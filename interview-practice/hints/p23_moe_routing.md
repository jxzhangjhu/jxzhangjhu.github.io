# Hints · p23 Top-1 MoE routing with capacity

Read one at a time.

## Level 1

Softmax router logits, then take each token's maximum probability and expert index.

## Level 2

For each expert, keep at most `capacity` assigned tokens — the most confident ones, not merely the first ones.

## Level 3

Balance loss is `E * sum(fraction_routed.detach-like * mean_gate_probability)`; hard routing supplies load while gradients flow through probabilities.
