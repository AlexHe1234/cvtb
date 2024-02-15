import numpy as np


# lift single image points into camera space 3d points WITHOUT scale
def lift_vanilla(
        k: np.ndarray,    # [3, 3] intrinsic matrix
        img: np.ndarray,  # [H, W, 3] image
        dpt: np.ndarray,  # [H, W] depth
        msk: np.ndarray   # [H, W] bool mask
    ) -> np.ndarray:      # [N, 3] pcd
    H, W = img.shape[:2]
    msk = msk.astype(bool)

    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    pixel_coords = np.stack([x, y]).transpose((1, 2, 0))
    
    masked_pixel_coords = pixel_coords[msk]
    masked_dpt = dpt[msk]

    inverse_mat = np.linalg.inv(k)
    masked_pixel_coords = np.concatenate([masked_pixel_coords, np.ones((masked_pixel_coords.shape[0], 1))], axis=-1)
    pixel_coords_3d = masked_pixel_coords @ inverse_mat.T

    pixel_coords_3d = pixel_coords_3d / np.linalg.norm(pixel_coords_3d, axis=-1, keepdims=True)
    return pixel_coords_3d * masked_dpt[:, None]
