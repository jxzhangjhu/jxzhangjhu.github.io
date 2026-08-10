"""p27 · Spinning light source -> Cauchy   —   budget 20 min  [reported: OpenAI]

Simulate a spinning light source hitting a wall, then verify the distribution.

Fill in the body. Run:  python run.py p27
Stuck? hints/p27_cauchy_simulation.md has three levels.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def light_source_samples(n, seed=0):
    """A lamp at distance 1 from an infinite wall, pointing uniformly at random.

    Return n sample positions along the wall.
    """
    raise NotImplementedError


def cauchy_pdf(x):
    """The analytic density the samples should follow."""
    raise NotImplementedError
