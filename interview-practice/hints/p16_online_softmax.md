# Hints · p16 Streaming softmax

Read one at a time.

## Level 1

Carry a running max `m`, denominator `l`, and weighted numerator `acc`.

## Level 2

When the max changes, rescale old state with `exp(m_old - m_new)`.

## Level 3

Apply that correction to both `l` and `acc`; initialise tensors from `scores`/`values` so device and dtype are preserved.
