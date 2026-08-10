# Hints · p13 MLP backward by hand

Read one at a time.

## Level 1

Work backwards through down-projection, activation, up-projection.

## Level 2

Each gradient has the shape of the tensor it belongs to; that fixes every contraction.

## Level 3

The bias gradient sums over the batch, because broadcasting forward means summing backward.
