"""p26 · Filter bad human annotations   —   budget 20 min  [reported: Anecdotal report: OpenAI]

Filter unreliable annotators, while protecting sparse but plausible annotators.

Fill in the body. Run:  python run.py p26
Stuck? hints/p26_data_filtering.md has three levels.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def filter_annotations(labels, annotators, min_agreement=0.6, min_items=3):
    """Return (clean_items, flagged_annotators)."""
    raise NotImplementedError
