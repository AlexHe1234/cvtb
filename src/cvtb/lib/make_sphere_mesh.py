import trimesh
import numpy as np
from typing import List, Union
from trimesh.transformations import translation_matrix


def make_sphere_mesh(
    radius: float,
    center: Union[np.ndarray, List[float]],  # x,y,z center
    color: Union[np.ndarray, List[float]],  # r,g,b union color for the mesh
    subdivision: int = 2,
) -> trimesh.Trimesh:
    # first create a unit sphere
    sphere = trimesh.creation.icosphere(subdivisions=subdivision, radius=radius)
    
    # recenter it
    trans_matrix = translation_matrix(center)
    sphere.apply_transform(trans_matrix)
    
    if color is not None:
        sphere.visual.face_colors = color
        
    return sphere


if __name__ == '__main__':
    mesh = make_sphere_mesh(2.0, [1,2,3], [0.5,0.,0.])
    mesh.show()
