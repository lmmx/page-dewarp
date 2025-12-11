# src/page_dewarp/optimise/_jax_vmap.py
"""Parallel optimization using JAX - runs independent L-BFGS-B optimizations."""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime as dt

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from ..device import get_device
from ..keypoints import make_keypoint_index
from ..options import cfg
from ._base import make_objective
from ._jax import _project_keypoints_jax


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


def _make_objective_and_grad(dstpoints, keypoint_index, focal_length, shear_cost):
    """Create JIT-compiled objective and gradient for scipy L-BFGS-B."""
    dstpoints_flat = jnp.array(dstpoints.reshape(-1, 2))
    keypoint_index_jax = jnp.array(keypoint_index, dtype=jnp.int32)

    @jax.jit
    def objective_jax(pvec):
        projected = _project_keypoints_jax(pvec, keypoint_index_jax, focal_length)
        error = jnp.sum((dstpoints_flat - projected) ** 2)
        if shear_cost > 0.0:
            error = error + shear_cost * pvec[0] ** 2
        return error

    objective_and_grad = jax.jit(jax.value_and_grad(objective_jax))

    def objective_with_grad_np(p):
        p_jax = jnp.array(p)
        val, grad = objective_and_grad(p_jax)
        val_np = float(val)
        grad_np = np.array(grad, dtype=np.float64)
        # NaN/Inf handling - critical for stability
        if not np.isfinite(val_np):
            return 1e10, np.zeros_like(p)
        grad_np = np.nan_to_num(grad_np, nan=0.0, posinf=0.0, neginf=0.0)
        return val_np, grad_np

    return objective_with_grad_np


def _optimize_single(args):
    """Optimize a single problem using scipy L-BFGS-B with JAX gradients."""
    (
        name,
        dstpoints,
        keypoint_index,
        params,
        focal_length,
        shear_cost,
        maxiter,
        maxcor,
    ) = args

    objective_with_grad = _make_objective_and_grad(
        dstpoints,
        keypoint_index,
        focal_length,
        shear_cost,
    )

    result = minimize(
        objective_with_grad,
        params,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "maxcor": maxcor},
    )

    return name, result.x, result.fun, result.nfev


def optimise_params_vmap(
    problems: list[OptimizationProblem],
    debug_lvl: int = 0,
) -> list[np.ndarray]:
    """Run parallel independent L-BFGS-B optimizations.

    Uses scipy's L-BFGS-B (which supports bounds and per-problem iteration)
    with JAX autodiff for gradients. Parallelizes across problems using threads
    since the actual computation happens on GPU via JAX.
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
    maxiter = cfg.OPT_MAX_ITER
    maxcor = cfg.MAX_CORR

    # Prepare keypoint indices
    keypoint_indices = [make_keypoint_index(p.span_counts) for p in problems]

    # Print initial objectives
    for i, p in enumerate(problems):
        obj = make_objective(
            p.dstpoints,
            keypoint_indices[i],
            shear_cost,
            slice(*cfg.RVEC_IDX),
        )
        print(f"  [{p.name}] initial objective is {obj(p.params):.6f}")

    # Prepare arguments for parallel execution
    opt_args = [
        (
            p.name,
            p.dstpoints,
            keypoint_indices[i],
            p.params,
            focal_length,
            shear_cost,
            maxiter,
            maxcor,
        )
        for i, p in enumerate(problems)
    ]

    print(
        f"\n  Running {len(problems)} parallel L-BFGS-B optimizations "
        f"on {device.device_kind.upper()}...",
    )

    start = dt.now()

    # Use ThreadPoolExecutor - JAX releases GIL during computation
    # so threads can run JAX ops in parallel on GPU
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with jax.default_device(device):
            # Warm up JIT compilation with first problem
            _ = _optimize_single(opt_args[0])

            # Run all in parallel
            with ThreadPoolExecutor(max_workers=len(problems)) as executor:
                results_raw = list(executor.map(_optimize_single, opt_args))

    elapsed = (dt.now() - start).total_seconds()
    print(f"  parallel L-BFGS-B optimization took {elapsed:.2f}s")

    # Extract results in order
    results = []
    for name, opt_params, final_obj, nfev in results_raw:
        print(f"  [{name}] final objective is {final_obj:.6f} ({nfev} evals)")
        results.append(opt_params)

    return results
