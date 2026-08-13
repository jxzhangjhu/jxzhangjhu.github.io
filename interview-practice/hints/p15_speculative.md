# Hints · p15 Speculative decoding accept/reject

Read one at a time.

## Level 1

Require `q_draft[token] > 0`, then accept exactly when `u < min(1, p_target[token] / q_draft[token])`; the strict inequality matters at a zero threshold.

## Level 2

On rejection, form `clamp(p_target - q_draft, min=0)` and normalise it.

## Level 3

Sample from that residual with `torch.multinomial`; the accepted mass plus residual mass equals the target distribution exactly.
