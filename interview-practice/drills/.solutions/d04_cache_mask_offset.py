"""d04 · Cached decode with an unshifted mask   —   budget 5 min

During incremental decode the query block is short but the key block is the whole
cache, so a plain lower triangle points at the wrong keys.

One line in this file is wrong. Run:  python -m pytest tests/test_d04_cache_mask_offset.py -q
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def cached_causal_mask(n_new, n_total):
    """True where a query in the last n_new positions may attend to a key in n_total."""
    return torch.tril(torch.ones(n_new, n_total, dtype=torch.bool),
                      diagonal=n_total - n_new)
