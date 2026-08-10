# Hints · p09 SFT loss masking and packing

Read one at a time.

## Level 1

Mask each example's own prompt length, not a slice of the batch dimension.

## Level 2

labels[i, :prompt_lens[i]] = -100, in a loop over the batch.

## Level 3

Padding must be masked too: labels[attention_mask == 0] = -100.
