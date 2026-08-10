# Hints · p06 SwiGLU feed-forward

Read one at a time.

## Level 1

Gate, up, down: three projections, not two.

## Level 2

forward is w_down(silu(w_gate(x)) * w_up(x)).

## Level 3

d_ff = 8*d_model/3 keeps the parameter count equal to a 4x ReLU FFN.
