"""p20 · DPO loss   —   budget 15 min

The DPO loss from four log-probabilities.

Fill in the body. Run:  python run.py p20
Stuck? hints/p20_dpo_loss.md has three levels.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def dpo_loss(*args, **kwargs):
    """Signature matches reference.dpo_loss."""
    raise NotImplementedError
