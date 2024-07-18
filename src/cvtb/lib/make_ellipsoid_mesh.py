import trimesh
import numpy as np
from typing import List, Union
from trimesh.transformations import quaternion_matrix, translation_matrix


def make_ellipsoid_mesh(
    scaling: Union[np.ndarray, List[float]],  # sx,sy,sz half-axis length
    rotation: Union[np.ndarray, List[float]],  # q0,q1,q2,q3 quaternion
    center: Union[np.ndarray, List[float]],  # x,y,z center
    color: Union[np.ndarray, List[float]],  # r,g,b union color for the mesh
    subdivision: int = 2,
) -> trimesh.Trimesh:
    # first create a unit sphere
    unit_sphere = trimesh.creation.icosphere(subdivisions=subdivision, radius=1.0)
    
    # rescale it
    ellipsoid = unit_sphere.copy()
    ellipsoid.apply_scale(scaling)

    # rotate it
    rot_matrix = quaternion_matrix(rotation)
    ellipsoid.apply_transform(rot_matrix)
    
    # recenter it
    trans_matrix = translation_matrix(center)
    ellipsoid.apply_transform(trans_matrix)
    
    if color is not None:
        ellipsoid.visual.face_colors = color
        
    return ellipsoid


if __name__ == '__main__':
    mesh = make_ellipsoid_mesh([1.,2.,3.], [1.,0.,0.,0.], [0.,0.,0.], [0.5,0.,0.])
    mesh.show()
