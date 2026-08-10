"""d08 · SFT loss counting the prompt   —   budget 4 min

Supervised fine-tuning should only score the completion. Scoring the prompt too
trains the model to generate the instructions it is supposed to follow.

One line in this file is wrong. Run:  python -m pytest tests/test_d08_prompt_not_masked.py -q
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def sft_loss(logits, input_ids, prompt_len):
    """logits: (B, T, V), input_ids: (B, T). Only completion tokens should be scored."""
    logits, targets = logits[:, :-1], input_ids[:, 1:]
    B, Tm1, V = logits.shape
    targets = targets.clone()
    positions = torch.arange(Tm1).expand(B, Tm1)
    targets[positions < prompt_len - 1] = -100
    return F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100)
