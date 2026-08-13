# Hints · p21 Generalised advantage estimation

Read one at a time.

## Level 1

`delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)`.

## Level 2

`A_t = delta_t + gamma * lambda * A_{t+1}`, so iterate backwards.

## Level 3

Return both advantages and `advantages + values`; preserve the input device and dtype.
