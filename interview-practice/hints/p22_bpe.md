# Hints · p22 Byte-pair encoding

Read one at a time. If the first is enough, stop.

## Level 1

Start from raw bytes (`text.encode('utf-8')`), so there is never an out-of-vocabulary case.

## Level 2

Each round: count adjacent pairs, take the most frequent, replace every occurrence with a fresh id starting at 256.

## Level 3

Encoding applies merges in the order they were LEARNED, not by frequency in the string being encoded. Iterate the merge dict in insertion order.
