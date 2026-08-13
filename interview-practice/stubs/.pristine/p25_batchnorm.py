"""p25 · BatchNorm forward, gradients, and eval mode   —   budget 20 min  [reported: Personal anecdotal report: Datadog]

BatchNorm1d from scratch: forward, running statistics, and eval mode.

Fill in the body. Run:  python run.py p25
Stuck? hints/p25_batchnorm.md has three levels.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchNorm1dScratch(nn.Module):
    def __init__(self, d, eps=1e-5, momentum=0.1):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        """x: (B, d) -> (B, d)"""
        raise NotImplementedError
