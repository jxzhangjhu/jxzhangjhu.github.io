# Hints · p14 Temperature, top-k, top-p

Read one at a time. If the first is enough, stop.

## Level 1

Apply temperature first — it changes the distribution the truncations then act on.

## Level 2

For top-p, sort descending, take the cumulative sum, and keep the shortest prefix whose mass reaches p.

## Level 3

The exclusive cumulative sum is `cum - probs`; drop where that is already >= top_p, which keeps the token that crosses the threshold.
