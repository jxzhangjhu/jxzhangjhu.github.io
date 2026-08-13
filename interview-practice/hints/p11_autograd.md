# Hints · p11 A minimal scalar autograd

Read one at a time.

## Level 1

Each result stores its parents and a closure that pushes `out.grad` into them.

## Level 2

Use `+=`, not `=`: one node may receive gradient through several graph paths.

## Level 3

`backward()` builds a DFS post-order, seeds the output gradient with 1, then executes closures in reverse topological order.
