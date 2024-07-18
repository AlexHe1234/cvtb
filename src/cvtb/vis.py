import numpy as np
from typing import List, Union

from .tool import *
from .color import *
from .utils.vis_utils import *


def pcd_static(p,  # N, 3 
               c=None,  # N, 3
               r=5,
               show_dir=False):
    if c is None:
        c = generate_gradient_color(p)
    
    player = Canvas(point_clouds=p[None],
                    fps=1.,
                    color=c[None] if c is not None else None,
                    point_size=r,
                    show_directions=show_dir)
    player.show(run=True)
     
        
def pcd(p,  # F, N, 3 or N, 3
        c=None,
        r=5,
        fps=24):
    assert len(p.shape) == 2 or len(p.shape) == 3
    
    if c is None:
        c = generate_gradient_color(p)
        
    if len(p.shape) == 2:
        pcd_static(p, c, r)
    else:
        player = Canvas(point_clouds=p,
                        fps=fps,
                        color=c,
                        point_size=r)
        player.show(run=True)
        
        
def pcds(ps: List,  # list of Ni, 3 or F, Ni, 3 for i in list index
         cs: List = None,  # list of Ni, 3 or F, Ni, 3 or None for i in list index
         rs: Union[int, List] = 5,  # list of Ni, or F, Ni for i in list index
         fps: int = 24,
         overlay: bool = True,
         overlay_gap: float = 2.):
    num_pcd = len(ps)
    
    assert len(ps) == len(cs) if cs is not None else True
    assert len(ps) == len(rs) if isinstance(rs, list) else True
    
    if not isinstance(rs, int) and not isinstance(rs, float):
        raise NotImplementedError()
    
    if len(ps[0].shape) == 2:
        if cs is None:
            cs = []
            seg_c = generate_discrete_color(num_pcd)  # num, 3
            seg_c_list = np.split(seg_c, len(seg_c))
            for i, p in enumerate(ps):
                cs.append(np.broadcast_to(seg_c_list[i], p.shape))
        elif len(cs[0]) != len(ps[0]):
            cs_ = []
            for i, p in enumerate(ps):
                cs_.append(np.broadcast_to(cs[i].reshape(3), p.shape))
            cs = cs_
        if not overlay:
            ps = place_pcds(ps, overlay_gap)
        p = np.concatenate(ps, axis=0)
        c = np.concatenate(cs, axis=0)
        p = p.copy()
        c = c.copy()
        pcd_static(p, c, r=rs)
    else:  # dynamic
        raise NotImplementedError()
    
    
def pcd_demo(num_points=1000,
             frames=100):
    pcd_init = generate_synthetic_point_cloud(num_points)[None]  # 1, N, 3
    rand_diff = (np.random.rand(frames, num_points, 3) - 0.5) * 2. / frames
    traj = np.cumsum(rand_diff, axis=0)  # F, N, 3
    point_clouds_sequence = pcd_init + traj
    point_size = np.arange(frames) / frames * 5 + 5.
    point_size = np.stack([point_size] * num_points).transpose()

    pcd(point_clouds_sequence, 
        fps = 24, 
        c = generate_gradient_color(pcd_init, use_axis = 'z'),
        r = point_size)
    

def pcd_static_demo(num_points=1000):
    pcd_init = generate_synthetic_point_cloud(num_points)  # N, 3
    pcd_static(pcd_init, c = generate_gradient_color(pcd_init))
    
    
def pcds_demo(num_pcds=10, max_points=100, min_points=5000):
    ps = []
    distance = 5.
    sigma = distance / (num_pcds * 6)  # 3-sigma principle
    miu = -distance / 2. + distance * np.arange(num_pcds) / (num_pcds + 1) + distance / num_pcds
    for num in range(num_pcds):
        num_point = np.random.uniform(min_points, max_points)
        p = np.random.randn(int(num_point), 3)
        p[:, 0] = p[:, 0] * sigma + miu[num]
        ps.append(p)
    pcds(ps)
