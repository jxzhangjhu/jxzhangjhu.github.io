# Hints · p08 Cross entropy with log-sum-exp

Read one at a time. If the first is enough, stop.

## Level 1

log_softmax(x)[t] = x[t] - logsumexp(x). Never build probabilities and then take a log.

## Level 2

logsumexp needs the max subtracted first: m + log(sum(exp(x - m))).

## Level 3

For ignore_index, average only kept rows. If none remain, return `logits.sum() * 0.0`: finite zero, still attached to the graph.
