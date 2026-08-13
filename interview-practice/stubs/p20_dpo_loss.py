"""p20 · DPO loss   —   budget 15 min

Implement DPO from four sequence log-probabilities.

Fill in the body. Run:  python run.py p20
Stuck? hints/p20_dpo_loss.md has three levels.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1):
    """All inputs have shape (B,); return a scalar mean."""
    raise NotImplementedError
