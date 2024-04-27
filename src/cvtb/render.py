import numpy as np
from typing import Union
from .utils.render_utils import *
from .utils.flags import *


def pcd(
    pcd: np.ndarray,  # N, 3
    color: np.ndarray,  # N, 3
    radius: Union[np.ndarray, int],  # N, or int
    
    k: np.ndarray,  # 3, 3
    r: np.ndarray,  # 3, 3
    t: np.ndarray,  # 3,
    
    H: int,  # int
    W: int,  # int
    
    flag: int,  # SHAPE, DEPTH, RGB
) -> np.ndarray:  # H, W, 3
    if flag == RENDER_PCD_SHAPE:
        if isinstance(radius, np.ndarray):
            raise TypeError('Radius must be of type int for this flag')
        return pcd_shape_vanilla(pcd, radius, k, r, t, H, W)
    elif flag == RENDER_PCD_DEPTH:
        raise NotImplementedError()
    elif flag == RENDER_PCD_RGB:
        raise NotImplementedError()
    else:
        raise TypeError()


def mesh(
    pts: np.ndarray,  # N, 3
    color: np.ndarray,  # N, 3
    radius: Union[np.ndarray, int],  # N, or int
    
    k: np.ndarray,  # 3, 3
    r: np.ndarray,  # 3, 3
    t: np.ndarray,  # 3,
    
    H: int,  # int
    W: int,  # int
    
    flag: int,
) -> np.ndarray:
    
    if flag == RENDER_MESH_SHAPE:
        raise NotImplementedError()
    elif flag == RENDER_MESH_DEPTH:
        raise NotImplementedError()
    elif flag == RENDER_MESH_RGB:
        raise NotImplementedError()
    else:
        raise TypeError()
