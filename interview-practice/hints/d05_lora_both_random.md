# Hints · d05 LoRA initialisation failure

Read one level at a time.

## Level 1

At step zero, the adapter contribution must be exactly zero while at least one factor
still receives a gradient.

## Level 2

The contribution is `B @ A`. Zeroing both factors kills the first-step gradient; making
both random changes the base model.

## Level 3

Initialise `A` randomly and `B` to zeros. Then `B @ A == 0`, while the loss gradient can
immediately update `B`.
