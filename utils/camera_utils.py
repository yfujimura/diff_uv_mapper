import numpy as np

def get_perspective(fovy, aspect=1, near=0.01, far=1000):
    """construct a perspective matrix from fovy.

    Args:
        fovy (float): field of view in degree along y-axis.
        aspect (int, optional): aspect ratio. Defaults to 1.
        near (float, optional): near clip plane. Defaults to 0.01.
        far (int, optional): far clip plane. Defaults to 1000.

    Returns:
        np.ndarray: perspective matrix, float [4, 4]
    """
    # fovy: field of view in degree.
    
    y = np.tan(np.deg2rad(fovy) / 2)
    return np.array(
        [
            [1 / (y * aspect), 0, 0, 0],
            [0, -1 / y, 0, 0],
            [
                0,
                0,
                -(far + near) / (far - near),
                -(2 * far * near) / (far - near),
            ],
            [0, 0, -1, 0],
        ],
        dtype=np.float32,
    )