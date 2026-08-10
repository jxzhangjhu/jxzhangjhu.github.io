# Hints · p21 Generalised advantage estimation

Read one at a time.

## Level 1

delta_t = r_t + gamma * V(s_{t+1}) - V(s_t).

## Level 2

A_t = delta_t + gamma * lambda * A_{t+1}, so the loop runs backwards.

## Level 3

Assert the limits: lambda=1 is Monte Carlo, lambda=0 is one-step TD.
