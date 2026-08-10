# Hints · p15 Speculative decoding accept/reject

Read one at a time.

## Level 1

Accept with probability min(1, p(x)/q(x)).

## Level 2

On rejection, sample from the normalised residual max(0, p - q).

## Level 3

This is exact: the emitted distribution is provably p. Verify it by sampling a few hundred thousand times.
