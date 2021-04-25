import numpy as np
from cv2 import projectPoints
from .options import cfg, K


def project_xy(xy_coords, pvec):
    """
    Get cubic polynomial coefficients given by:

      f(0) = 0, f'(0) = alpha
      f(1) = 0, f'(1) = beta
    """
    alpha, beta = tuple(pvec[cfg.proj_opts.CUBIC_IDX])
    poly = np.array([alpha + beta, -2 * alpha - beta, alpha, 0])

    xy_coords = xy_coords.reshape((-1, 2))
    z_coords = np.polyval(poly, xy_coords[:, 0])

    objpoints = np.hstack((xy_coords, z_coords.reshape((-1, 1))))
    image_points, _ = projectPoints(
        objpoints,
        pvec[cfg.proj_opts.RVEC_IDX],
        pvec[cfg.proj_opts.TVEC_IDX],
        K(),
        np.zeros(5),
    )
    return image_points
