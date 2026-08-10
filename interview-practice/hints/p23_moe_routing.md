# Hints · p23 Top-1 MoE routing with capacity

Read one at a time.

## Level 1

Softmax the router logits, take the argmax, and assign in order until an expert is full.

## Level 2

Overflowing tokens skip the layer entirely and pass through the residual.

## Level 3

The Switch loss is E * sum_e f_e * p_e, minimised at uniform routing where it equals 1.
