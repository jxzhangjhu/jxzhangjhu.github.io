# Hints · d03 Nucleus-support failure

Read one level at a time.

## Level 1

For probabilities `[.50, .30, .15, .05]` and `p=.90`, write down the shortest prefix
whose mass reaches the threshold.

## Level 2

The token that crosses the threshold belongs to the nucleus. Compare inclusive and
exclusive cumulative mass at each token.

## Level 3

Use `drop = cum - probs >= top_p`, then mask `drop`. Force the first sorted token to
survive if you choose an equivalent shifted-mask implementation.
