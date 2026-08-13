# Hints · d06 Softmax stability failure

Read one level at a time.

## Level 1

Softmax is unchanged when every logit in a row is shifted by the same constant.

## Level 2

Choose that constant so every exponent is non-positive and at least one is zero.

## Level 3

Compute `e = (x - x.max(dim=-1, keepdim=True).values).exp()`, then divide by the row sum.
