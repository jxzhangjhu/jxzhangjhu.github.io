# Hints · d08 SFT-objective failure

Read one level at a time.

## Level 1

After next-token shifting, map each target column back to the input token it predicts.

## Level 2

For `prompt_len = 4`, shifted target columns 0, 1, and 2 are still prompt tokens and
must use `ignore_index`.

## Level 3

Clone targets and set `targets[:, :prompt_len - 1] = -100` before calling cross entropy.
