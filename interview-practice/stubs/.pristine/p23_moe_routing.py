"""p23 · Top-1 MoE routing with capacity   —   budget 20 min

Implement top-1 MoE routing with capacity and the Switch balancing loss.

Fill in the body. Run:  python run.py p23
Stuck? hints/p23_moe_routing.md has three levels.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def top1_route(logits, capacity):
    """Return (expert_index, selected_gate, kept_mask), each with leading token axis."""
    raise NotImplementedError


def load_balancing_loss(logits):
    """Switch loss E * sum_e fraction_e * mean_probability_e."""
    raise NotImplementedError
