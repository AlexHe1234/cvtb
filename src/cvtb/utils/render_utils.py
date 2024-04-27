import numpy as np
import cv2
from typing import Union


def pcd_depth():
    pass


def pcd_shape_vanilla(
    # renders point clouds with minimum dependency, no color and no independent radius
        pcd: str,  # N, 3
        radius: int,  # int
        
        k: np.ndarray,  # 3, 3
        r: np.ndarray,  # 3, 3
        t: np.ndarray,  # 3,

        height: int,  # int
        width: int,  # int
    ) -> np.ndarray:
    
    color = [[255, 255, 255]]

    if isinstance(pcd, str):
        if '.ply' in pcd:
            import open3d as o3d
            mesh = o3d.io.read_point_cloud(pcd)
            pts = np.asarray(mesh.points)
        elif '.npy' in pcd:
            pts = np.load(pcd)
            if not (len(pts.shape) == 2 and pts.shape[1] == 3):
                raise IndexError
        else:
            raise NotImplementedError
    elif isinstance(pcd, np.ndarray):
        assert pcd.shape[1] == 3, 'Point cloud needs to be [N, 3] in shape'
        pts = pcd
    else:
        raise NotImplementedError

    t = t.reshape(3, 1)
    coord_cam = r.dot(pts.T) + t
    coord_img = k.dot(coord_cam)
    coord_img /= coord_img[2, :]
    pixels = coord_img[:2, :].T
    
    img = np.zeros((height, width, 3))
    
    for i in range(pixels.shape[0]):
        pixel = pixels[i]
        if not (0 <= pixel[1] < height and 0 <= pixel[0] < width):
            continue
        img = cv2.circle(img, [int(pixel[0]), int(pixel[1])], radius, color=color[i % 7], thickness=2*radius)
    return img.astype(np.uint8)


def mesh_depth(
    pts: np.ndarray,  # N, 3
    color: np.ndarray,  # N, 3
    radius: Union[np.ndarray, int],  # N, or int
    
    k: np.ndarray,  # 3, 3
    r: np.ndarray,  # 3, 3
    t: np.ndarray,  # 3,
    
    H: int,  # int
    W: int,  # int
) -> np.ndarray:
    pass
