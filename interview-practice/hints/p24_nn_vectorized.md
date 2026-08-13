# Hints · p24 1-NN in pure NumPy, no loops

Read one at a time.

## Level 1

Broadcast direct differences to shape (m, n, d): `test_x[:, None, :] - train_x[None, :, :]`.

## Level 2

Square those differences and sum over the feature axis. This avoids cancellation from expanding two large nearby squared norms.

## Level 3

You never need sqrt because argmin is invariant to it. This teaching version materialises (m,n,d); production code can chunk rows or use a vetted distance kernel.
