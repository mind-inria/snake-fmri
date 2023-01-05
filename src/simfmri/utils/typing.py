"""Typing utilities."""
from __future__ import annotations
from typing import Union

import numpy as np

RngType = Union[int, np.random.Generator]
"""Type characterising a random number generator.

A random generator is either reprensented by its seed (int), or a numpy.random.Generator.
"""


Shape2d3d = Union[tuple[int, int], tuple[int, int, int]]
"""Type for a 2D or 3D shape."""
