# src/page_dewarp/optimise/_jax_batched.py
"""Batched JAX optimization for multiple images simultaneously.

This module provides GPU-efficient batched optimization by running
multiple independent L-BFGS-B optimizations in parallel using JAX's
vectorization capabilities.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime as dt

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from ..device import get_device
from ..keypoints import make_keypoint_index
from ..options import cfg
from ._jax import _project_xy_jax


__all__ = ["OptimizationProblem", "optimise_params_batched"]


@dataclass
class OptimizationProblem:
    """Data for a single optimization problem in a batch."""

    name: str
    dstpoints: np.ndarray  # (N, 1, 2) target points
    span_counts: list[int]
    params: np.ndarray  # Initial parameters
    # These are needed for post-processing but not optimization
    small: np.ndarray | None = None
    corners: np.ndarray | None = None
    rough_dims: tuple[float, float] | None = None


def _project_keypoints_jax_single(pvec, keypoint_index, focal_length):
    """JAX version of project_keypoints for a single problem."""
    xy_coords = pvec[keypoint_index]
    xy_coords = xy_coords.at[0, :].set(0.0)
    return _project_xy_jax(xy_coords, pvec, focal_length)


def _make_single_objective_jax(
    dstpoints_flat,
    keypoint_index,
    focal_length,
    shear_cost,
):
    """Create objective for a single problem (used in vmap)."""

    def objective(pvec, valid_mask):
        projected = _project_keypoints_jax_single(pvec, keypoint_index, focal_length)
        # Mask out invalid (padded) points
        diff = dstpoints_flat - projected
        error = jnp.sum(diff**2 * valid_mask[:, None])

        if shear_cost > 0.0:
            rvec = pvec[0:3]
            error = error + shear_cost * rvec[0] ** 2

        return error

    return objective


def optimise_params_batched(
    problems: list[OptimizationProblem],
    debug_lvl: int = 0,
) -> list[np.ndarray]:
    """Run batched optimization for multiple images on GPU.

    This function batches multiple independent optimization problems
    and runs them together, leveraging GPU parallelism for gradient
    computation.

    Args:
        problems: List of OptimizationProblem instances.
        debug_lvl: Debug verbosity level.

    Returns:
        List of optimized parameter arrays, one per input problem.

    """
    from ..debug_utils import debug_show
    from ..keypoints import project_keypoints
    from ._base import draw_correspondences, make_objective

    if not problems:
        return []

    # For single problem, use regular optimization (no batching overhead)
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
    focal_length = cfg.FOCAL_LENGTH
    shear_cost = cfg.SHEAR_COST

    # Prepare data for each problem
    keypoint_indices = []
    dstpoints_list = []
    params_list = []
    n_points_list = []
    n_params_list = []

    for p in problems:
        kp_idx = make_keypoint_index(p.span_counts)
        keypoint_indices.append(kp_idx)
        dstpoints_list.append(p.dstpoints.reshape(-1, 2))
        params_list.append(p.params)
        n_points_list.append(len(p.dstpoints))
        n_params_list.append(len(p.params))

    # Find max sizes for padding
    max_points = max(n_points_list)
    max_params = max(n_params_list)
    max_kp_idx = max(len(kp) for kp in keypoint_indices)

    batch_size = len(problems)

    # Pad everything to uniform size
    dstpoints_padded = np.zeros((batch_size, max_points, 2), dtype=np.float32)
    keypoint_index_padded = np.zeros((batch_size, max_kp_idx, 2), dtype=np.int32)
    params_padded = np.zeros((batch_size, max_params), dtype=np.float64)
    valid_point_mask = np.zeros((batch_size, max_points), dtype=np.float32)
    valid_param_mask = np.zeros((batch_size, max_params), dtype=np.float32)

    for i, (dst, kp_idx, params, n_pts, n_par) in enumerate(
        zip(
            dstpoints_list,
            keypoint_indices,
            params_list,
            n_points_list,
            n_params_list,
        ),
    ):
        dstpoints_padded[i, :n_pts] = dst
        keypoint_index_padded[i, : len(kp_idx)] = kp_idx
        params_padded[i, :n_par] = params
        valid_point_mask[i, :n_pts] = 1.0
        valid_param_mask[i, :n_par] = 1.0

    # Print initial objectives
    for i, p in enumerate(problems):
        kp_idx = keypoint_indices[i]
        obj = make_objective(p.dstpoints, kp_idx, shear_cost, slice(*cfg.RVEC_IDX))
        print(f"  [{p.name}] initial objective is {obj(p.params):.6f}")

    # Convert to JAX arrays
    dstpoints_jax = jnp.array(dstpoints_padded)
    keypoint_index_jax = jnp.array(keypoint_index_padded)
    valid_point_mask_jax = jnp.array(valid_point_mask)

    # Create batched objective using vmap
    def single_problem_objective(pvec, dstpoints, kp_idx, valid_mask):
        """Objective for one problem."""
        xy_coords = pvec[kp_idx]
        xy_coords = xy_coords.at[0, :].set(0.0)
        projected = _project_xy_jax(xy_coords, pvec, focal_length)

        diff = dstpoints - projected
        error = jnp.sum(diff**2 * valid_mask[:, None])

        if shear_cost > 0.0:
            error = error + shear_cost * pvec[0] ** 2

        return error

    @jax.jit
    def batched_objective(params_flat):
        """Total objective across all problems."""
        params_batch = params_flat.reshape(batch_size, max_params)

        # vmap over batch dimension
        errors = jax.vmap(single_problem_objective)(
            params_batch,
            dstpoints_jax,
            keypoint_index_jax,
            valid_point_mask_jax,
        )
        return jnp.sum(errors)

    objective_and_grad = jax.value_and_grad(batched_objective)

    def objective_with_grad_np(p):
        p_jax = jnp.array(p)
        val, grad = objective_and_grad(p_jax)
        val_np = float(val)
        grad_np = np.array(grad, dtype=np.float64)

        if not np.isfinite(val_np):
            return 1e10, np.zeros_like(p)

        grad_np = np.nan_to_num(grad_np, nan=0.0, posinf=0.0, neginf=0.0)

        # Zero out gradients for padded parameters
        grad_np = grad_np.reshape(batch_size, max_params)
        grad_np = grad_np * valid_param_mask
        grad_np = grad_np.flatten()

        return val_np, grad_np

    # Run optimization
    params_flat = params_padded.flatten()
    total_params = batch_size * max_params

    print(
        f"  optimizing {batch_size} problems ({total_params} total params) "
        f"on {device.device_kind.upper()}...",
    )

    start = dt.now()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with jax.default_device(device):
            # Warm up JIT
            _ = objective_with_grad_np(params_flat)

            result = minimize(
                objective_with_grad_np,
                params_flat,
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": cfg.OPT_MAX_ITER, "maxcor": cfg.MAX_CORR},
            )

    elapsed = (dt.now() - start).total_seconds()
    print(
        f"  batched optimization (L-BFGS-B + JAX autodiff) took {elapsed:.2f}s, "
        f"{result.nfev} evals",
    )

    # Unpack results
    results_batch = result.x.reshape(batch_size, max_params)
    optimized_params = []

    for i, (p, n_par) in enumerate(zip(problems, n_params_list)):
        params = results_batch[i, :n_par]
        optimized_params.append(params)

        # Print final objective
        kp_idx = keypoint_indices[i]
        obj = make_objective(p.dstpoints, kp_idx, shear_cost, slice(*cfg.RVEC_IDX))
        print(f"  [{p.name}] final objective is {obj(params):.6f}")

        # Debug visualization
        if debug_lvl >= 1 and p.small is not None:
            projpts = project_keypoints(params, kp_idx)
            display = draw_correspondences(p.small, p.dstpoints, projpts)
            debug_show(p.name, 5, "keypoints after", display)

    return optimized_params
