# Hints · d04 Cached-mask failure

Read one level at a time.

## Level 1

Label query rows by absolute position. With two new queries and five total keys, they
live at positions 3 and 4, not 0 and 1.

## Level 2

A rectangular lower triangle needs its diagonal shifted by the number of cached tokens.

## Level 3

Build `torch.tril(torch.ones(n_new, n_total, dtype=torch.bool),
diagonal=n_total - n_new)`.
