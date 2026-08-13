"""d08 · SFT objective failure   —   budget 4 min

Symptom: changing prompt tokens changes the completion loss

One line in this file is wrong. Run:  python -m pytest tests/test_d08_prompt_not_masked.py -q
Stuck? Read hints/d08_prompt_not_masked.md one level at a time.
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
    return F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100)
