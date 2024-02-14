import numpy as np
import open3d as o3d


def pcd_obj(file_path):
    vertices = []

    with open(file_path, 'r') as obj_file:
        for line in obj_file:
            if line.startswith('v '):
                vertices.append(list(map(float, line.split()[1:])))

    return np.array(vertices)


def pcd(file_path):  # return N, 3 pcd
    if '.obj' in file_path:
        return pcd_obj(file_path)
    if '.ply' in file_path:
        return np.asarray(o3d.io.read_point_cloud(file_path).points)
    if '.npy' in file_path:
        return np.load(file_path)
    raise TypeError(f'Extension format not supported for opening point cloud')
