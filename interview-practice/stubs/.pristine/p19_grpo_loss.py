"""p19 · GRPO objective   —   budget 20 min  [cold-start set]  [reported: OpenAI + Anthropic 4+]

The GRPO objective: group-relative advantage, clipped ratio, per-token k3 KL.

Fill in the body. Run:  python run.py p19
Stuck? hints/p19_grpo_loss.md has three levels.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def grpo_loss(logp, logp_old, logp_ref, rewards, mask, group_size, clip_eps=0.2, beta=0.04):
    """logp/logp_old/logp_ref: (B, L). rewards: (B,). mask: (B, L). Returns a scalar."""
    raise NotImplementedError
