import cv2
import numpy as np
from typing import Literal, Union


def hsv_to_rgb_opencv(hsv_array):
    # Ensure that the input array has the correct shape (N, 3)
    assert hsv_array.shape[1] == 3, "Input array must have shape (N, 3)"
    # Convert HSV array to RGB array using OpenCV
    hsv_array_uint8 = (hsv_array * 255).astype(np.uint8)  # Convert to uint8
    rgb_array_uint8 = cv2.cvtColor(hsv_array_uint8[None], cv2.COLOR_HSV2RGB)[0]
    # Normalize back to [0, 1]
    rgb_array = rgb_array_uint8 / 255.0
    return rgb_array


# generate #num of as contrast as possible colors  
def generate_discrete_color(num: int, fixed: bool = True):
    if fixed:
        np.random.seed(42)
    hue = np.arange(num) / num
    hue += np.random.uniform()
    hue %= 1
    hsv = np.ones((num, 3))
    hsv[:, 0] = hue
    rgb = hsv_to_rgb_opencv(hsv)
    return rgb


# Map the 0-1 value to the Hue component in HSV
def _generate_gradient_color(value: np.ndarray,
                            start=0.4,
                            end=0.65,
                            ):
    N = value.shape[0]
    value = (value - value.min()) / (value.max() - value.min())
    value = start + (end - start) * value
    hsv = np.ones((N, 3), dtype=np.float32)
    # breakpoint()
    hsv[:, 0] = hsv[:, 0] * value
    rgb = hsv_to_rgb_opencv(hsv)

    return rgb


def generate_gradient_color(pcd: np.ndarray, 
                            start=0.4, 
                            end=0.65, 
                            use_index: bool = False, 
                            use_axis: Union[Literal['x', 'y', 'z'], None] = 'z',
                            ):
    # use_index would simply use index for coloring while False analysis point direction and use that
    if len(pcd.shape) == 2:
        N = pcd.shape[0]
    elif len(pcd.shape) == 3:
        N = pcd.shape[1]
        pcd = pcd[0]
    else:
        raise ValueError(f'Tell me why ~~~ would a point cloud have shape {pcd.shape}?')
    
    if use_index:
        return _generate_gradient_color(np.arange(N), start=start, end=end)
    else:
        if use_axis is None:
            # get main direction est.
            center = pcd.mean(axis=0)
            print('Finding main direction, this could be time consuming')
            *_, v = np.linalg.svd(pcd - center)
            vec = v[0]
        else:
            if use_axis == 'x':
                vec = np.array([1., 0., 0.])
            elif use_axis == 'y':
                vec = np.array([0., 1., 0.])
            elif use_axis == 'z':
                vec = np.array([0., 0., 1.])
            else:
                raise ValueError(f'Use axis got an unexpected value')
        align_value = pcd @ vec.T
        align_ordering = np.argsort(np.argsort(align_value))
        return _generate_gradient_color(align_ordering.astype(np.float32), start=start, end=end)
