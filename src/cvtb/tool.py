import numpy as np
from typing import Literal


def normalize(x: np.ndarray, axis: int = -1):
    return x / np.linalg.norm(x, axis=axis, keepdims=True)
