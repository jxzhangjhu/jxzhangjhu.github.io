"""d04 · Cached-mask failure   —   budget 5 min

Symptom: cached decode cannot attend to legal history

One line in this file is wrong. Run:  python -m pytest tests/test_d04_cache_mask_offset.py -q
Stuck? Read hints/d04_cache_mask_offset.md one level at a time.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def cached_causal_mask(n_new, n_total):
    """True where a query in the last n_new positions may attend to a key in n_total."""
    return torch.tril(torch.ones(n_new, n_total, dtype=torch.bool),
                      diagonal=n_total - n_new)
