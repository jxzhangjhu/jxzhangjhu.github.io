# Hints · p09 SFT loss masking and packing

Read one at a time.

## Level 1

There are three independent invariants: only response tokens are labels; position ids restart at each packed segment; attention is causal and cannot cross a segment boundary.

## Level 2

For labels, intersect response_mask with attention_mask. For attention, first identify contiguous segment runs, then compare each query/key run and AND with a lower-triangular mask and validity on both axes.

## Level 3

Reset an offset whenever the segment id changes. Build run ids from boundary indicators and return `same_run & causal & query_valid & key_valid`; padding gets position zero and no allowed row or column.
