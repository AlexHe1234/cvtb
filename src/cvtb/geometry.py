import numpy as np
from .tool import *


def generate_synthetic_point_cloud(num_points):
    return np.random.rand(num_points, 3) * 2 - 1


# generate dotted lines
def generate_lines(start: np.ndarray,  # N, 3 or 3,
                   end: np.ndarray,  # N, 3 or 3,
                   num_points: int = 50):
    # make sure points cover start and end spot
    if len(start.shape) == 1:
        flat = True
        start = start[None]
        end = end[None]
    else:
        flat = False
    direction = end - start  # N, 3
    dist_norm = np.arange(num_points) / (num_points - 1.)
    dist = direction[:, None, :] * dist_norm.reshape(1, num_points, 1)  # N, P, 3
    line = start[:, None, :] + dist  # N, P, 3
    
    if not flat:
        return line
    else:
        return line[0]
    

# TODO: batchify
def generate_circle(center: np.ndarray,  # 3
                    direction: np.ndarray,  # 3
                    radius: np.ndarray,  # 3
                    num_points: int,
                    sample_strat: Literal['uniform', 'random'] = 'uniform'):
    direction = normalize(direction)
    
    random_vec = np.random.rand(3)
    random_radius = normalize(np.cross(direction, random_vec))
    
    x = random_radius
    z = direction
    y = normalize(np.cross(z, x))
    
    if sample_strat == 'uniform':
        theta = np.arange(num_points) * 2 * np.pi / num_points
    else:
        theta = np.random.uniform(0., np.pi * 2, size=(num_points))
        theta = np.sort(theta)    
    
    theta = theta.reshape(num_points, 1)
    x_coord = np.cos(theta) * radius
    y_coord = np.sin(theta) * radius
    z_coord = np.zeros_like(x_coord)
    
    coord = x_coord * x + y_coord * y + z_coord * z + center
    
    return coord
    

# TODO: batchify
# direction points to top
# angle is full angle in degree
# length is corner length
def generate_cone(top: np.ndarray,  # 3,
                  direction: np.ndarray,  # 3,
                  angle: float,
                  length: int,
                  num_points: int = 200,
                  num_strides: int = 4):
    radius = length * np.sin(np.deg2rad(angle) / 2)
    direction = normalize(direction) * length * np.cos(np.deg2rad(angle) / 2)
    center = top - direction
    circle = generate_circle(center, direction, radius, num_strides, 'uniform')  # N, 3
    lines = generate_lines(circle, top[None].repeat(num_strides, 0), num_points // num_strides).reshape(-1, 3)
    return lines


# TODO: batchify
# generate dotted arrow from 
#   1. origin 3
#   2. vector 3
# return P, 3
def generate_arrow(origin: np.ndarray,
                   vector: np.ndarray, 
                   num_points: int = 100,
                   cone_angle: float = 60,
                   length_ratio: float = 0.2,
                   cone_ratio: float = 0.5,
                   num_stride: int = 4):
    # we're hoping to save a template and the scale it
    line = generate_lines(origin, 
                          origin + vector, 
                          int(num_points * (1 - cone_ratio))).reshape(-1, 3)
    cone = generate_cone(origin + vector, 
                         vector, 
                         cone_angle, 
                         length_ratio * np.linalg.norm(vector),
                         int(num_points * cone_ratio),
                         num_stride).reshape(-1, 3)
    ret = np.concatenate([line, cone], axis=0)
    return ret
    