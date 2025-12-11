"""JAX-based optimization backend for page dewarping.

This module provides accelerated optimization using JAX's automatic
differentiation for computing gradients, enabling efficient L-BFGS-B
optimization.
"""

from __future__ import annotations

import warnings
from datetime import datetime as dt

import jax
import jax.numpy as jnp
import numpy as np
from cv2 import Rodrigues
from scipy.optimize import minimize

from ..debug_utils import debug_show
from ..device import get_device
from ..keypoints import make_keypoint_index, project_keypoints
from ..options import cfg
from ._base import draw_correspondences, make_objective


__all__ = ["optimise_params_jax"]


def _rodrigues_jax(rvec):
    """Convert rotation vector to rotation matrix (Rodrigues formula)."""
    theta = jnp.linalg.norm(rvec)

    # For small angles, OpenCV uses a Taylor expansion
    # R = I + K + 0.5*K^2 when theta is small
    # For larger angles: R = I + sin(theta)*K + (1-cos(theta))*K^2

    # Avoid division by zero
    theta_safe = jnp.where(theta < 1e-10, 1.0, theta)
    k = rvec / theta_safe

    # Skew-symmetric matrix
    K_mat = jnp.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
    )
    K_mat_sq = K_mat @ K_mat

    # Standard Rodrigues formula
    R_standard = jnp.eye(3) + jnp.sin(theta) * K_mat + (1.0 - jnp.cos(theta)) * K_mat_sq

    # Small angle approximation: sin(theta) ≈ theta, 1-cos(theta) ≈ theta^2/2
    R_small = jnp.eye(3) + theta * K_mat + 0.5 * theta * theta * K_mat_sq

    return jnp.where(theta < 1e-8, R_small, R_standard)


def _project_xy_jax(xy_coords, pvec, focal_length):
    """JAX version of project_xy.

    project_xy does:

    1. Extract alpha, beta from pvec[6:8], clip to [-0.5, 0.5]
    2. Build cubic polynomial coeffs
    3. Evaluate z = polyval(poly, x)
    4. Build 3D objpoints
    5. Call projectPoints(objpoints, rvec, tvec, K, dist_coeffs=0)
    """
    alpha = pvec[6]
    beta = pvec[7]

    alpha = jnp.clip(alpha, -0.5, 0.5)
    beta = jnp.clip(beta, -0.5, 0.5)

    # poly = [alpha + beta, -2*alpha - beta, alpha, 0]
    # polyval evaluates: poly[0]*x^3 + poly[1]*x^2 + poly[2]*x + poly[3]
    x = xy_coords[:, 0]
    z_coords = (alpha + beta) * x**3 + (-2.0 * alpha - beta) * x**2 + alpha * x

    objpoints = jnp.column_stack([xy_coords, z_coords])

    # projectPoints does:
    # 1. R = Rodrigues(rvec)
    # 2. transformed = R @ point + tvec
    # 3. x' = transformed[0] / transformed[2]
    # 4. y' = transformed[1] / transformed[2]
    # 5. u = fx * x' + cx  (cx = 0 in our K)
    # 6. v = fy * y' + cy  (cy = 0 in our K)

    rvec = pvec[0:3]
    tvec = pvec[3:6]

    R = _rodrigues_jax(rvec)

    # Transform each point
    transformed = (R @ objpoints.T).T + tvec

    # Perspective divide
    z = transformed[:, 2]
    x_proj = transformed[:, 0] / z
    y_proj = transformed[:, 1] / z

    # Apply camera matrix (just focal length since cx=cy=0)
    u = focal_length * x_proj
    v = focal_length * y_proj

    return jnp.column_stack([u, v])


def _project_keypoints_jax(pvec, keypoint_index, focal_length):
    """JAX version of project_keypoints.

    1. xy_coords = pvec[keypoint_index]  # shape (N, 2)
    2. xy_coords[0, :] = 0  # first point at origin
    3. return project_xy(xy_coords, pvec)
    """
    xy_coords = pvec[keypoint_index]
    xy_coords = xy_coords.at[0, :].set(0.0)
    return _project_xy_jax(xy_coords, pvec, focal_length)


def _make_objective_jax(dstpoints_flat, keypoint_index, focal_length, shear_cost):
    """Create JIT-compiled objective matching the original."""

    @jax.jit
    def objective(pvec):
        projected = _project_keypoints_jax(pvec, keypoint_index, focal_length)
        error = jnp.sum((dstpoints_flat - projected) ** 2)

        # Shear penalty
        if shear_cost > 0.0:
            rvec = pvec[0:3]
            error = error + shear_cost * rvec[0] ** 2

        return error

    return objective


def _run_jax_lbfgsb(
    dstpoints: np.ndarray,
    keypoint_index: np.ndarray,
    params: np.ndarray,
) -> minimize:
    """Run optimization using JAX autodiff + L-BFGS-B."""
    # Match original dtype (float32 for K matrix, but params are float64)
    dstpoints_flat = jnp.array(dstpoints.reshape(-1, 2))
    keypoint_index_jax = jnp.array(keypoint_index, dtype=jnp.int32)
    focal_length = cfg.FOCAL_LENGTH
    shear_cost = cfg.SHEAR_COST

    objective_jax = _make_objective_jax(
        dstpoints_flat,
        keypoint_index_jax,
        focal_length,
        shear_cost,
    )
    objective_and_grad = jax.value_and_grad(objective_jax)

    def objective_with_grad_np(p):
        p_jax = jnp.array(p)
        val, grad = objective_and_grad(p_jax)
        val_np = float(val)
        grad_np = np.array(grad, dtype=np.float64)
        if not np.isfinite(val_np):
            return 1e10, np.zeros_like(p)
        grad_np = np.nan_to_num(grad_np, nan=0.0, posinf=0.0, neginf=0.0)
        return val_np, grad_np

    result = minimize(
        objective_with_grad_np,
        params,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": cfg.OPT_MAX_ITER, "maxcor": cfg.MAX_CORR},
    )
    return result


def optimise_params_jax(
    name: str,
    small: np.ndarray,
    dstpoints: np.ndarray,
    span_counts: list[int],
    params: np.ndarray,
    debug_lvl: int,
) -> np.ndarray:
    """Refine the parameter vector using JAX-accelerated L-BFGS-B optimization.

    Args:
        name: Image name for debug output.
        small: Downscaled image for visualization.
        dstpoints: Target points to match.
        span_counts: Number of keypoints per span.
        params: Initial parameter vector.
        debug_lvl: Debug verbosity level.

    Returns:
        Optimized parameter vector.

    """
    device = get_device(cfg.DEVICE)
    keypoint_index = make_keypoint_index(span_counts)

    # Use the base objective for initial value display
    objective = make_objective(
        dstpoints,
        keypoint_index,
        cfg.SHEAR_COST,
        slice(*cfg.RVEC_IDX),
    )

    print("  initial objective is", objective(params))
    if debug_lvl >= 1:
        projpts = project_keypoints(params, keypoint_index)
        display = draw_correspondences(small, dstpoints, projpts)
        debug_show(name, 4, "keypoints before", display)

    print(f"  optimizing {len(params)} parameters on {device.device_kind.upper()}...")

    start = dt.now()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with jax.default_device(device):
            result = _run_jax_lbfgsb(dstpoints, keypoint_index, params)
    elapsed = (dt.now() - start).total_seconds()
    print(
        f"  optimization (L-BFGS-B + JAX autodiff) took {elapsed:.2f}s, {result.nfev} evals",
    )

    print(f"  final objective is {result.fun:.6f}")
    params = result.x

    if debug_lvl >= 1:
        _print_diagnostics(params, keypoint_index, small, dstpoints, name)

    return params


def _print_diagnostics(
    params: np.ndarray,
    keypoint_index: np.ndarray,
    small: np.ndarray,
    dstpoints: np.ndarray,
    name: str,
) -> None:
    """Print parameter diagnostics and show debug visualization."""
    projpts = project_keypoints(params, keypoint_index)
    display = draw_correspondences(small, dstpoints, projpts)
    debug_show(name, 5, "keypoints after", display)

    rvec = params[slice(*cfg.RVEC_IDX)]
    tvec = params[slice(*cfg.TVEC_IDX)]
    alpha, beta = params[slice(*cfg.CUBIC_IDX)]

    print("  === Parameter Diagnostics ===")
    print(f"  Rotation vector: {rvec}")
    print(f"  Rotation angles (degrees): {np.degrees(rvec)}")
    print(f"  Translation vector: {tvec}")
    print(f"  Cubic params - alpha: {alpha}, beta: {beta}")

    R, _ = Rodrigues(rvec)
    print(f"  Rotation matrix determinant: {np.linalg.det(R)}")
    print(f"  Rotation matrix condition number: {np.linalg.cond(R)}")
