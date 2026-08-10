# Hints · p20 DPO loss

Read one at a time.

## Level 1

The margin is (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected).

## Level 2

Loss is -logsigmoid(beta * margin), averaged.

## Level 3

Sanity check: at the reference policy the margin is 0 and the loss is exactly log 2.
