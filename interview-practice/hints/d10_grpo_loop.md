# Hints · d10 Debug a GRPO loop

Read one level at a time and rerun the tests after every change.

## Level 1

Write three invariants before reading code: sampled frequencies match the policy within
sampling error; tied rewards produce finite zero advantages; when new and old policies
are identical, each importance ratio is one.

## Level 2

Inspect the input contract of `torch.multinomial`, the zero-variance and group-size-one
cases in standardisation, and the conversion from a log-probability difference to a
probability ratio.

## Level 3

Sample from `softmax(logits)`; use `std(..., correction=0) + eps`; compute
`ratio = exp(logp - old_logp)` before applying PPO-style clipping. Then verify that the
optimizer receives finite gradients on an all-tied group.
