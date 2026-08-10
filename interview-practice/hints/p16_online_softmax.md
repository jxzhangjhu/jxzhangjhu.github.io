# Hints · p16 Streaming softmax

Read one at a time.

## Level 1

Carry a running max, a running denominator, and a running numerator.

## Level 2

When a block reveals a larger max, rescale everything so far by exp(m_old - m_new).

## Level 3

Both the denominator AND the accumulator need the correction — forgetting the accumulator is the classic bug.
