import numpy as np


# lift image points into camera space 3d points
def lift(
        k: np.ndarray,    # [3, 3] intrinsic matrix
        img: np.ndarray,  # [H, W, 3] image
        dpt: np.ndarray,  # [H, W] depth
        msk: np.ndarray   # [H, W] bool mask
    ) -> np.ndarray:      # [N, 3] pcd
    H, W = img.shape[:2]
    msk = msk.astype(bool).flatten()

    pixel_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    pixel_coords = np.stack(pixel_coords).transpose((1, 2, 0))
    
    masked_pixel_coords = pixel_coords.reshape(-1, 2)[msk]
    masked_dpt = dpt[msk]

    inverse_mat = np.linalg.inv(k)
    masked_pixel_coords = np.concatenate([masked_pixel_coords, np.ones((masked_pixel_coords.shape[0], 1))], axis=-1)
    pixel_coords_3d = masked_pixel_coords @ inverse_mat.T

    breakpoint()
    

if __name__ == '__main__':
    import cv2

    k = np.array([[517.,   0., 320.],
                  [  0., 517., 240.],
                  [  0.,   0.,   1.]])
    img_path = '/home/idarc/hgz/datasets/killing_fusion/Alex/color_000000.png'
    dpt_path = '/home/idarc/hgz/datasets/killing_fusion/Alex/depth_000000.png'

    img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    dpt = cv2.imread(dpt_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)


    msk = np.ones_like(dpt, dtype=bool)

    pcd = lift(k, img, dpt, msk)

