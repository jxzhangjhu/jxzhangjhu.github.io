# Hints · p24 1-NN in pure NumPy, no loops

Read one at a time.

## Level 1

Expand the square: ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2.

## Level 2

The cross term is a single matmul, test_x @ train_x.T, of shape (m, n).

## Level 3

Add the two norm vectors with explicit broadcasting: (m,1) + (1,n). You never need sqrt, because argmin is invariant to it.
