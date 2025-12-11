# src/page_dewarp/optimise/_jax_vmap.py
"""Truly parallel optimization using JAX's native BFGS with vmap."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime as dt

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.optimize import minimize as jax_minimize

from ..device import get_device
from ..keypoints import make_keypoint_index
from ..options import cfg
from ._base import make_objective
from ._jax import _project_xy_jax


__all__ = ["OptimizationProblem", "optimise_params_vmap"]


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
    """Run parallel independent optimizations using JAX vmap + BFGS.

    Uses JAX's native BFGS optimizer which can be vmapped for true parallelism.
    Each problem gets its own independent Hessian approximation.
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
    focal_length = cfg.FOCAL_LENGTH
    shear_cost = cfg.SHEAR_COST

    # Prepare all problems - need to pad to same size
    keypoint_indices = [make_keypoint_index(p.span_counts) for p in problems]

    n_points_list = [len(p.dstpoints) for p in problems]
    n_params_list = [len(p.params) for p in problems]
    max_points = max(n_points_list)
    max_params = max(n_params_list)
    max_kp_idx = max(len(kp) for kp in keypoint_indices)
    batch_size = len(problems)

    # Pad arrays
    dstpoints_padded = np.zeros((batch_size, max_points, 2), dtype=np.float32)
    keypoint_index_padded = np.zeros((batch_size, max_kp_idx, 2), dtype=np.int32)
    params_padded = np.zeros((batch_size, max_params), dtype=np.float64)
    valid_mask = np.zeros((batch_size, max_points), dtype=np.float32)

    for i, p in enumerate(problems):
        n_pts = n_points_list[i]
        n_par = n_params_list[i]
        dstpoints_padded[i, :n_pts] = p.dstpoints.reshape(-1, 2)
        keypoint_index_padded[i, : len(keypoint_indices[i])] = keypoint_indices[i]
        params_padded[i, :n_par] = p.params
        valid_mask[i, :n_pts] = 1.0

    # Print initial objectives
    for i, p in enumerate(problems):
        obj = make_objective(
            p.dstpoints,
            keypoint_indices[i],
            shear_cost,
            slice(*cfg.RVEC_IDX),
        )
        print(f"  [{p.name}] initial objective is {obj(p.params):.6f}")

    # Convert to JAX
    dstpoints_jax = jnp.array(dstpoints_padded)
    kp_idx_jax = jnp.array(keypoint_index_padded)
    params_jax = jnp.array(params_padded)
    valid_jax = jnp.array(valid_mask)

    def single_objective(pvec, dstpoints, kp_idx, valid):
        """Objective for one problem."""
        xy_coords = pvec[kp_idx]
        xy_coords = xy_coords.at[0, :].set(0.0)
        projected = _project_xy_jax(xy_coords, pvec, focal_length)

        diff = dstpoints - projected
        error = jnp.sum(diff**2 * valid[:, None])

        if shear_cost > 0.0:
            error = error + shear_cost * pvec[0] ** 2

        return error

    def optimize_single(pvec, dstpoints, kp_idx, valid):
        """Run BFGS on one problem."""

        def obj(p):
            return single_objective(p, dstpoints, kp_idx, valid)

        result = jax_minimize(obj, pvec, method="BFGS")
        return result.x

    # vmap over batch dimension
    optimize_batch = jax.vmap(optimize_single)

    print(
        f"\n  Running {len(problems)} parallel BFGS optimizations "
        f"on {device.device_kind.upper()}...",
    )

    start = dt.now()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with jax.default_device(device):
            # JIT compile the vmapped optimizer
            optimize_batch_jit = jax.jit(optimize_batch)

            # Run all optimizations in parallel
            results_padded = optimize_batch_jit(
                params_jax,
                dstpoints_jax,
                kp_idx_jax,
                valid_jax,
            )
            results_padded.block_until_ready()

    elapsed = (dt.now() - start).total_seconds()
    print(f"  parallel BFGS optimization took {elapsed:.2f}s")

    # Extract results (trim padding)
    results = []
    for i, n_par in enumerate(n_params_list):
        opt_params = np.array(results_padded[i, :n_par])
        results.append(opt_params)

    # Print final objectives
    for i, (p, opt_params) in enumerate(zip(problems, results)):
        obj = make_objective(
            p.dstpoints,
            keypoint_indices[i],
            shear_cost,
            slice(*cfg.RVEC_IDX),
        )
        print(f"  [{p.name}] final objective is {obj(opt_params):.6f}")

    return results
