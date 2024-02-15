import numpy as np


# lift image points into camera space 3d points
def lift(k: np.ndarray, img: np.ndarray, dpt: np.ndarray, msk: np.ndarray) -> np.ndarray:
    pass
    

if __name__ == '__main__':
    import cv2

    k = np.array([[517.,   0., 320.],
                  [  0., 517., 240.],
                  [  0.,   0.,   1.]])
    img_path = '/home/idarc/hgz/datasets/killing_fusion/Alex/color_000000.png'
    dpt_path = '/home/idarc/hgz/datasets/killing_fusion/Alex/color_000000.png'

    img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    dpt = cv2.cvtColor(cv2.imread(dpt_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)

    msk = None

    pcd = lift(k, img, dpt, msk)
