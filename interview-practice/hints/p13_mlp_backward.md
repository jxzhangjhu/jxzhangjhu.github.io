# Hints · p13 MLP backward by hand

Read one at a time.

## Level 1

Work backwards through the down projection, ReLU, then the up projection.

## Level 2

Every gradient must have the shape of the tensor it belongs to; use that to determine each transpose.

## Level 3

A broadcast bias becomes a sum over the broadcast batch dimension in backward.
