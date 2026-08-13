"""p09 · SFT loss masking and packing   —   budget 20 min

Build SFT labels plus position ids and a block-diagonal mask for packing.

Fill in the body. Run:  python run.py p09
Stuck? hints/p09_loss_masking.md has three levels.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_sft_labels(input_ids, prompt_lens, attention_mask, ignore_index=-100):
    """Return labels with prompts and padding replaced by ignore_index."""
    raise NotImplementedError


def build_packed_sft_labels(input_ids, response_mask, attention_mask, ignore_index=-100):
    """Label response tokens in packed rows; mask prompts, separators, and padding."""
    raise NotImplementedError


def build_packed_attention(segment_ids, attention_mask):
    """Return (position_ids, allowed_mask) with positions reset at each segment."""
    raise NotImplementedError
