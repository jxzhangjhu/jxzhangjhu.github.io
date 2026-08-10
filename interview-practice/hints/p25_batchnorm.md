# Hints · p25 BatchNorm forward, backward, eval mode

Read one at a time.

## Level 1

Train mode uses batch statistics; eval mode uses the running ones. They compute different functions.

## Level 2

register_buffer, not nn.Parameter — they move with .to(device) and are saved, but get no gradient.

## Level 3

PyTorch normalises with the biased variance (/n) but accumulates the unbiased one (/(n-1)). Mismatch this and only eval mode diverges.
