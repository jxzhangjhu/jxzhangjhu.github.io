# Hints · p10 Overfit a tiny batch

Read one at a time. If the first is enough, stop.

## Level 1

Fixed random inputs and targets, a two-layer MLP, cross entropy, Adam. No dropout, no shuffling.

## Level 2

The three lines that must be in this order every step: `opt.zero_grad()`, `loss.backward()`, `opt.step()`.

## Level 3

Call `.item()` outside any `torch.no_grad()` block, and remember the loss you return should come from the final forward pass, not a stale variable.
