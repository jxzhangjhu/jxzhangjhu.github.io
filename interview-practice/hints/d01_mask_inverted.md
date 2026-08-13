# Hints · d01 Causal-mask failure

Read one level at a time.

## Level 1

Draw the 4×4 mask. Query row 0 may see only key 0; query row 3 may see keys 0 through 3.

## Level 2

Check the boolean convention at `masked_fill`: it replaces entries where its mask is `True`.

## Level 3

`allowed` marks legal entries, so fill its complement:
`scores.masked_fill(~allowed, float("-inf"))`.
