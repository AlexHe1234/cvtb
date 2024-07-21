import numpy as np
from typing import Literal


def normalize(x: np.ndarray, axis: int = -1):
    return x / np.linalg.norm(x, axis=axis, keepdims=True)


def to_unit_cube(pcd: np.ndarray, padding: 0.1):
    assert len(pcd.shape) == 2
    center = np.mean(pcd, axis=0, keepdims=True)
    pcd = pcd - center
    pcd = pcd * (1. - padding) / np.max(np.abs(pcd))
    return (pcd + 1.) * 0.5


def place_pcds(pcds, unit = 2.):
    # place a list of pcd so that they don't overlap
    def place(x, current):
        # unit = 2.
        x[..., 2] += unit * current
        return x
    
    ret = []
    for i, pcd in enumerate(pcds):
        p_shape = pcd.shape
        pcd = pcd.reshape(-1, 3)
        pcd = to_unit_cube(pcd, padding=0.1)
        pcd = pcd.reshape(*p_shape)
        pcd = place(pcd, i)
        ret.append(pcd)
    
    return ret
