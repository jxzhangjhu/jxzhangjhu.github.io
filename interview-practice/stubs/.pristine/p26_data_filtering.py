"""p26 · Filter bad human annotations   —   budget 20 min  [reported: OpenAI 2+]

Filter unreliable human annotations.

Fill in the body. Run:  python run.py p26
Stuck? hints/p26_data_filtering.md has three levels.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def filter_annotations(*args, **kwargs):
    """Signature matches reference.filter_annotations."""
    raise NotImplementedError
