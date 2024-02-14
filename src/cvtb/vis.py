import numpy as np
from .utils.vis_utils import *
     
        
def pcd(point_clouds,  # F, N, 3
        fps=24,
        color=None,
        point_size=5):
    player = Canvas(point_clouds=point_clouds,
                     fps=fps,
                     color=color,
                     point_size=point_size)
    player.show(run=True)


def pcd_demo(num_points=1000,
             frames=100):
    pcd_init = generate_synthetic_point_cloud(num_points)[None]  # 1, N, 3
    rand_diff = (np.random.rand(frames, num_points, 3) - 0.5) * 2. / frames
    traj = np.cumsum(rand_diff, axis=0)  # F, N, 3
    point_clouds_sequence = pcd_init + traj
    point_size = np.arange(frames) / frames * 5 + 5.
    point_size = np.stack([point_size] * num_points).transpose()

    pcd(point_clouds_sequence, 
         fps=24, 
         color=generate_gradient_color_from_coords(pcd_init),
         point_size=point_size)
        
        
def pcd_static(point_clouds,  # N, 3 
               color=None,  # N, 3
               point_size=5):
    player = Canvas(point_clouds=point_clouds[None],
                     fps=1.,
                     color=color[None] if color is not None else None,
                     point_size=point_size)
    player.show(run=True)
    

def pcd_static_demo(num_points=1000):
    pcd_init = generate_synthetic_point_cloud(num_points)  # N, 3
    pcd_static(pcd_init, color=generate_gradient_color_from_coords(pcd_init))
