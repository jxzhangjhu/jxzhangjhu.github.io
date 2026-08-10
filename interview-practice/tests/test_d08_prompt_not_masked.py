"""Tests for d08 · SFT loss counting the prompt. Run: python -m pytest tests/test_d08_prompt_not_masked.py -q"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drills import d08_prompt_not_masked as d  # noqa: E402

def test_prompt_tokens_do_not_affect_the_loss():
    torch.manual_seed(0)
    logits = torch.randn(2, 8, 50)
    ids = torch.randint(0, 50, (2, 8))
    base = d.sft_loss(logits, ids, prompt_len=4)

    moved = ids.clone()
    moved[:, :3] = (moved[:, :3] + 7) % 50        # rewrite prompt tokens only
    assert torch.allclose(base, d.sft_loss(logits, moved, prompt_len=4)), \
        "changing prompt tokens changed the loss, so the prompt is still being scored"


def test_matches_a_hand_masked_reference():
    torch.manual_seed(0)
    logits, ids = torch.randn(2, 8, 50), torch.randint(0, 50, (2, 8))
    lg, tg = logits[:, :-1], ids[:, 1:].clone()
    tg[:, :3] = -100                               # prompt_len=4 -> targets 0..2 are prompt
    want = F.cross_entropy(lg.reshape(-1, 50), tg.reshape(-1), ignore_index=-100)
    assert torch.allclose(d.sft_loss(logits, ids, prompt_len=4), want)
