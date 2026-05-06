"""Parallel optimization using JAX's native L-BFGS with vmap."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime as dt

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.scipy.optimize._lbfgs import _minimize_lbfgs

from ..device import get_device
from ..keypoints import make_keypoint_index
from ..logging_config import get_logger
from ..options import cfg
from ._base import make_objective
from ._jax import _project_xy_jax


__all__ = ["OptimizationProblem", "optimise_params_vmap"]

logger = get_logger("optimise.jax_vmap")


@dataclass
class OptimizationProblem:
    """Data for a single optimization problem."""

    name: str
    dstpoints: np.ndarray
    span_counts: list[int]
    params: np.ndarray
    small: np.ndarray | None = None
    corners: np.ndarray | None = None
    rough_dims: tuple[float, float] | None = None


def optimise_params_vmap(
    problems: list[OptimizationProblem],
    debug_lvl: int = 0,
) -> list[np.ndarray]:
    """Run parallel independent optimizations using JAX vmap + L-BFGS.

    Args:
        problems: List of optimization problems.
        debug_lvl: Debug verbosity level.

    Returns:
        List of optimized parameter vectors.

    """
    if not problems:
        return []

    if len(problems) == 1:
        from ._jax import optimise_params_jax

        p = problems[0]
        return [
            optimise_params_jax(
                p.name,
                p.small,
                p.dstpoints,
                p.span_counts,
                p.params,
                debug_lvl,
            ),
        ]

    device = get_device(cfg.DEVICE)
    focal_length = float(cfg.FOCAL_LENGTH)
    shear_cost = float(cfg.SHEAR_COST)
    maxcor = cfg.MAX_CORR
    maxiter = cfg.OPT_MAX_ITER

    # Prepare all problems
    keypoint_indices = [make_keypoint_index(p.span_counts) for p in problems]

    n_points_list = [len(p.dstpoints) for p in problems]
    n_params_list = [len(p.params) for p in problems]
    max_points = max(n_points_list)
    max_params = max(n_params_list)
    max_kp_idx = max(len(kp) for kp in keypoint_indices)
    batch_size = len(problems)

    # Pad arrays
    dstpoints_padded = np.zeros((batch_size, max_points, 2), dtype=np.float64)
    keypoint_index_padded = np.zeros((batch_size, max_kp_idx, 2), dtype=np.int32)
    params_padded = np.zeros((batch_size, max_params), dtype=np.float64)
    point_valid_mask = np.zeros((batch_size, max_points), dtype=np.float64)
    n_valid_points = np.zeros(batch_size, dtype=np.int32)

    for i, p in enumerate(problems):
        n_pts = n_points_list[i]
        n_par = n_params_list[i]
        n_kp = len(keypoint_indices[i])

        dstpoints_padded[i, :n_pts] = p.dstpoints.reshape(-1, 2)
        keypoint_index_padded[i, :n_kp] = keypoint_indices[i]
        params_padded[i, :n_par] = p.params
        point_valid_mask[i, :n_pts] = 1.0
        n_valid_points[i] = n_pts

    # Log initial objectives
    for i, p in enumerate(problems):
        obj = make_objective(
            p.dstpoints,
            keypoint_indices[i],
            shear_cost,
            slice(*cfg.RVEC_IDX),
        )
        logger.info(
            "Optimization starting",
            extra={
                "file": p.name,
                "initial_objective": round(obj(p.params), 6),
            },
        )

    # Convert to JAX arrays
    dstpoints_jax = jnp.array(dstpoints_padded)
    kp_idx_jax = jnp.array(keypoint_index_padded)
    params_jax = jnp.array(params_padded)
    point_mask_jax = jnp.array(point_valid_mask)
    n_valid_jax = jnp.array(n_valid_points)

    def single_objective(pvec, dstpoints, kp_idx, point_mask, n_valid):
        """Objective for one problem."""
        xy_coords = pvec[kp_idx]
        xy_coords = xy_coords.at[0, :].set(0.0)
        projected = _project_xy_jax(xy_coords, pvec, focal_length)

        diff = dstpoints - projected
        sq_error = jnp.sum(diff**2, axis=1)
        error = jnp.sum(sq_error * point_mask)

        if shear_cost > 0.0:
            rvec = pvec[0:3]
            error = error + shear_cost * rvec[0] ** 2

        return error

    def optimize_single(pvec, dstpoints, kp_idx, point_mask, n_valid):
        """Run L-BFGS on one problem."""

        def obj(p):
            return single_objective(p, dstpoints, kp_idx, point_mask, n_valid)

        result = _minimize_lbfgs(
            obj,
            pvec,
            maxiter=maxiter,
            maxcor=maxcor,
            gtol=1e-5,
            ftol=2.220446049250313e-09,
        )
        return result.x_k

    optimize_batch = jax.vmap(optimize_single)

    logger.info(
        "Parallel optimization starting",
        extra={
            "n_problems": len(problems),
            "device": device.device_kind,
        },
    )

    start = dt.now()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with jax.default_device(device):
            optimize_batch_jit = jax.jit(optimize_batch)
            results_padded = optimize_batch_jit(
                params_jax,
                dstpoints_jax,
                kp_idx_jax,
                point_mask_jax,
                n_valid_jax,
            )
            results_padded.block_until_ready()

    elapsed = (dt.now() - start).total_seconds()

    logger.info(
        "Parallel optimization complete",
        extra={
            "elapsed_s": round(elapsed, 2),
            "n_problems": len(problems),
        },
    )

    # Extract results
    results = []
    for i, n_par in enumerate(n_params_list):
        opt_params = np.array(results_padded[i, :n_par], dtype=np.float64)
        results.append(opt_params)

    # Log final objectives
    for i, (p, opt_params) in enumerate(zip(problems, results)):
        obj = make_objective(
            p.dstpoints,
            keypoint_indices[i],
            shear_cost,
            slice(*cfg.RVEC_IDX),
        )
        logger.info(
            "Optimization result",
            extra={
                "file": p.name,
                "final_objective": round(obj(opt_params), 6),
            },
        )

    return results
