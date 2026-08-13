"""p24 · 1-NN in pure NumPy, no loops   —   budget 15 min  [cold-start set]  [reported: Anecdotal report: OpenAI]

1-nearest-neighbour classification in NumPy. No Python loops over test points.

Fill in the body. Run:  python run.py p24
Stuck? hints/p24_nn_vectorized.md has three levels.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def nearest_neighbour(train_x, train_y, test_x):
    """train_x (n, d), train_y (n,), test_x (m, d) -> predicted labels (m,)."""
    raise NotImplementedError
