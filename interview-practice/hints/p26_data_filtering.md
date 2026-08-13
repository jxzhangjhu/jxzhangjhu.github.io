# Hints · p26 Filter bad human annotations

Read one at a time.

## Level 1

Take the non-None per-item majority, then measure each annotator's agreement with it.

## Level 2

Only flag below-threshold annotators who labelled at least `min_items`; otherwise the estimate is too noisy.

## Level 3

Recompute each retained item label after removing flagged annotators, and skip rows with no remaining votes.
