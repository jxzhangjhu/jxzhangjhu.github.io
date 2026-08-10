"""p09 · SFT loss masking and packing   —   budget 10 min

SFT label masking over a padded batch.

Fill in the body. Run:  python run.py p09
Stuck? hints/p09_loss_masking.md has three levels.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_sft_labels(*args, **kwargs):
    """Signature matches reference.build_sft_labels."""
    raise NotImplementedError
