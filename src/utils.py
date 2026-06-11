import numpy as np
def load_intrinsics(path):
    M = np.loadtxt(path)

    if M.shape == (3, 4):
        K = M[:, :3]
    elif M.shape == (3, 3):
        K = M
    else:
        raise ValueError(f"Unexpected intrinsic matrix shape: {M.shape}")

    return K

def load_extrinsics(path):
    M = np.loadtxt(path)

    if M.shape != (3, 4):
        raise ValueError(f"Unexpected extrinsic matrix shape: {M.shape}")

    R = M[:, :3]
    T = M[:, 3].reshape(3, 1)

    return R, T