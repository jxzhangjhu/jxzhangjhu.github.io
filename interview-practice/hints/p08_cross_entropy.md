# Hints · p08 Cross entropy with log-sum-exp

Read one at a time. If the first is enough, stop.

## Level 1

log_softmax(x)[t] = x[t] - logsumexp(x). Never build probabilities and then take a log.

## Level 2

logsumexp needs the max subtracted first: m + log(sum(exp(x - m))).

## Level 3

For ignore_index, build a boolean keep-mask, gather only the kept rows, and divide by the number kept — not by N.
