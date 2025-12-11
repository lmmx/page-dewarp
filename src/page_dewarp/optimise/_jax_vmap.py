# src/page_dewarp/optimise/_jax_vmap.py
"""Truly parallel optimization using JAX's native L-BFGS with vmap."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime as dt

import jax
import jax.numpy as jnp
import numpy as np

from ..device import get_device
from ..keypoints import make_keypoint_index
from ..options import cfg
from ._base import make_objective
from ._jax import _project_xy_jax


__all__ = ["OptimizationProblem", "optimise_params_vmap"]


# Import the internal L-BFGS implementation directly
from jax._src.scipy.optimize._lbfgs import _minimize_lbfgs


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

    Uses JAX's internal L-BFGS optimizer which can be vmapped for true parallelism.
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
    maxcor = cfg.MAX_CORR

    # Prepare all problems - need to pad to same size
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
    valid_mask = np.zeros((batch_size, max_points), dtype=np.float64)
    param_mask = np.zeros((batch_size, max_params), dtype=np.float64)

    for i, p in enumerate(problems):
        n_pts = n_points_list[i]
        n_par = n_params_list[i]
        dstpoints_padded[i, :n_pts] = p.dstpoints.reshape(-1, 2)
        keypoint_index_padded[i, : len(keypoint_indices[i])] = keypoint_indices[i]
        params_padded[i, :n_par] = p.params
        valid_mask[i, :n_pts] = 1.0
        param_mask[i, :n_par] = 1.0

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
    param_mask_jax = jnp.array(param_mask)

    # Soft bounds for cubic params (indices 6, 7) to mimic L-BFGS-B bounds
    CUBIC_LO = -0.5
    CUBIC_HI = 0.5
    BOUND_PENALTY = 1000.0

    def make_objective_for_problem(dstpoints, kp_idx, valid, p_mask):
        """Create objective function for a single problem."""

        def objective(pvec):
            # Mask padded parameters
            pvec_masked = pvec * p_mask

            # Soft bound penalties on cubic params
            alpha = pvec_masked[6]
            beta = pvec_masked[7]

            alpha_penalty = jnp.where(
                alpha < CUBIC_LO,
                BOUND_PENALTY * (CUBIC_LO - alpha) ** 2,
                jnp.where(
                    alpha > CUBIC_HI,
                    BOUND_PENALTY * (alpha - CUBIC_HI) ** 2,
                    0.0,
                ),
            )
            beta_penalty = jnp.where(
                beta < CUBIC_LO,
                BOUND_PENALTY * (CUBIC_LO - beta) ** 2,
                jnp.where(
                    beta > CUBIC_HI,
                    BOUND_PENALTY * (beta - CUBIC_HI) ** 2,
                    0.0,
                ),
            )

            # Project keypoints
            xy_coords = pvec_masked[kp_idx]
            xy_coords = xy_coords.at[0, :].set(0.0)
            projected = _project_xy_jax(xy_coords, pvec_masked, focal_length)

            # Masked squared error
            diff = dstpoints - projected
            error = jnp.sum(diff**2 * valid[:, None])

            # Shear penalty
            if shear_cost > 0.0:
                error = error + shear_cost * pvec_masked[0] ** 2

            # Add bound penalties
            error = error + alpha_penalty + beta_penalty

            return error

        return objective

    def optimize_single(pvec, dstpoints, kp_idx, valid, p_mask):
        """Run L-BFGS on one problem."""
        obj = make_objective_for_problem(dstpoints, kp_idx, valid, p_mask)

        result = _minimize_lbfgs(
            obj,
            pvec,
            maxiter=cfg.OPT_MAX_ITER,
            maxcor=maxcor,
            gtol=1e-5,
            ftol=2.220446049250313e-09,  # Same as scipy default
        )

        # Mask output to zero padded params
        return result.x_k * p_mask

    # vmap over batch dimension
    optimize_batch = jax.vmap(optimize_single)

    print(
        f"\n  Running {len(problems)} parallel L-BFGS optimizations "
        f"on {device.device_kind.upper()}...",
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
                valid_jax,
                param_mask_jax,
            )
            results_padded.block_until_ready()

    elapsed = (dt.now() - start).total_seconds()
    print(f"  parallel L-BFGS optimization took {elapsed:.2f}s")

    # Extract results (trim padding) and clip cubic params
    results = []
    for i, n_par in enumerate(n_params_list):
        opt_params = np.array(results_padded[i, :n_par])
        opt_params[6] = np.clip(opt_params[6], CUBIC_LO, CUBIC_HI)
        opt_params[7] = np.clip(opt_params[7], CUBIC_LO, CUBIC_HI)
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
