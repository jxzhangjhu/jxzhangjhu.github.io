"""p15 · Speculative decoding accept/reject   —   budget 20 min

Implement one exact speculative-decoding accept/reject step.

Fill in the body. Run:  python run.py p15
Stuck? hints/p15_speculative.md has three levels.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def speculative_accept(p_target, q_draft, token, u):
    """Return (emitted_token, accepted_draft: bool)."""
    raise NotImplementedError
