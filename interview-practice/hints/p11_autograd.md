# Hints · p11 A 40-line autograd

Read one at a time.

## Level 1

Each node stores data, grad, its children, and a closure that pushes gradient to them.

## Level 2

Accumulate with += , not = : a node used twice receives gradient from both paths.

## Level 3

backward() needs a reverse topological order, built with a DFS post-order.
