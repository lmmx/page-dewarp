"""Curl parameter estimation using Toeplitz matrix factorization.

This module provides functions to estimate the cubic curl parameters (α, β)
from span sample points, using the Carathéodory-Fejér theorem relating
positive semidefinite Toeplitz matrices to atomic measures on the unit circle.

The key insight: a page curl signal has characteristic low-frequency content.
The autocorrelation matrix of the curl residuals is Toeplitz, and its
near-null space reveals the dominant frequencies, which map to curl shape.
"""

import numpy as np
from scipy.linalg import toeplitz


__all__ = [
    "extract_curl_residuals",
    "build_autocorrelation_toeplitz",
    "find_unit_circle_roots",
    "estimate_curl_from_toeplitz",
    "estimate_cubic_params_from_spans",
]


def extract_curl_residuals(span_points_single: np.ndarray) -> np.ndarray | None:
    """Extract curl residuals from a single span by removing linear trend.

    Args:
        span_points_single: Array of shape (N, 1, 2) in normalized coordinates.

    Returns:
        1D array of residuals (y - linear_fit), or None if insufficient points.

    """
    pts = span_points_single.reshape((-1, 2))
    if len(pts) < 4:
        return None

    x_vals = pts[:, 0]
    y_vals = pts[:, 1]

    # Remove linear trend (baseline)
    coeffs = np.polyfit(x_vals, y_vals, 1)
    y_linear = np.polyval(coeffs, x_vals)
    residuals = y_vals - y_linear

    return residuals


def build_autocorrelation_toeplitz(
    residuals: np.ndarray,
    max_lag: int | None = None,
) -> np.ndarray:
    """Build a Toeplitz matrix from the autocorrelation of residuals.

    Args:
        residuals: 1D array of curl residuals.
        max_lag: Maximum lag to include. Defaults to min(len(residuals)//2, 10).

    Returns:
        A symmetric positive semidefinite Toeplitz matrix.

    """
    n = len(residuals)
    if max_lag is None:
        max_lag = min(n // 2, 10)

    # Center the signal
    residuals = residuals - np.mean(residuals)

    # Compute autocorrelation via correlation
    autocorr = np.correlate(residuals, residuals, mode="full")
    autocorr = autocorr[n - 1 :]  # Non-negative lags only

    # Normalize
    if autocorr[0] > 0:
        autocorr = autocorr / autocorr[0]
    else:
        autocorr = np.zeros_like(autocorr)

    # Build Toeplitz matrix
    return toeplitz(autocorr[:max_lag])


def find_unit_circle_roots(
    toeplitz_matrix: np.ndarray,
    tol: float = 1e-6,
    circle_tol: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find roots on or near the unit circle from Toeplitz null space.

    By the Carathéodory-Fejér theorem, if T is a positive semidefinite
    Toeplitz matrix of rank r, then there exist r distinct points on the
    unit circle such that T = V Λ V*, where V is Vandermonde.

    The null space of T gives a polynomial whose roots are these points.

    Args:
        toeplitz_matrix: A symmetric Toeplitz matrix.
        tol: Tolerance for identifying near-zero eigenvalues.
        circle_tol: Tolerance for roots being "on" the unit circle.

    Returns:
        A tuple (roots_on_circle, frequencies, eigenvalues) where:
            - roots_on_circle: Complex roots near |z| = 1
            - frequencies: Their angles (in radians)
            - eigenvalues: All eigenvalues of the Toeplitz matrix

    """
    eigenvalues, eigenvectors = np.linalg.eigh(toeplitz_matrix)

    # Sort by eigenvalue magnitude
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Find near-null space
    threshold = tol * max(eigenvalues.max(), 1e-10)
    null_mask = eigenvalues < threshold

    if np.any(null_mask):
        # Use the eigenvector for smallest eigenvalue
        null_vec = eigenvectors[:, 0]
    else:
        # Matrix is full rank; use smallest eigenvector anyway
        # (signal is approximately, not exactly, a sum of sinusoids)
        null_vec = eigenvectors[:, 0]

    # Form polynomial and find roots
    # Coefficients are the null vector entries
    if np.allclose(null_vec, 0):
        return np.array([]), np.array([]), eigenvalues

    roots = np.roots(null_vec[::-1])

    # Filter to roots near the unit circle
    distances_from_circle = np.abs(np.abs(roots) - 1)
    on_circle_mask = distances_from_circle < circle_tol
    roots_on_circle = roots[on_circle_mask]

    # Extract frequencies (angles)
    frequencies = np.angle(roots_on_circle)

    return roots_on_circle, frequencies, eigenvalues


def estimate_curl_from_toeplitz(
    residuals: np.ndarray,
    x_span: float = 1.0,
) -> tuple[float, float]:
    """Estimate (α, β) from curl residuals using Toeplitz analysis.

    The cubic model is f(x) = αx(1-x)² + βx²(1-x) on [0, 1].

    The spectral content of this cubic is concentrated at low frequencies.
    The asymmetry between α and β manifests as a shift in the spectral
    center of mass.

    Args:
        residuals: 1D array of detrended curl residuals.
        x_span: The x-extent of the span (for scaling).

    Returns:
        A tuple (alpha, beta) estimating the cubic parameters.

    """
    if len(residuals) < 4:
        return 0.0, 0.0

    # Build Toeplitz matrix from autocorrelation
    T = build_autocorrelation_toeplitz(residuals)

    if T.shape[0] < 2:
        return 0.0, 0.0

    # Find roots on unit circle
    roots, frequencies, eigenvalues = find_unit_circle_roots(T)

    # Estimate curvature magnitude from residual variance
    curvature_magnitude = np.std(residuals)

    if len(frequencies) == 0:
        # No clear spectral peaks; estimate from eigenvalue decay
        # Rapid decay suggests smooth (low-frequency) signal
        if len(eigenvalues) >= 2 and eigenvalues[-1] > 0:
            decay_rate = eigenvalues[0] / eigenvalues[-1]
            # Slow decay (ratio near 1) = noisy; fast decay = smooth curl
            smoothness = np.clip(1 - decay_rate, 0, 1)
            alpha = curvature_magnitude * smoothness * 5
            beta = alpha  # Assume symmetric if no frequency info
        else:
            alpha = beta = 0.0
        return alpha, beta

    # Find dominant low frequency
    # For a curl, we expect spectral mass near ω = 0
    abs_frequencies = np.abs(frequencies)
    dominant_idx = np.argmin(abs_frequencies)
    dominant_freq = frequencies[dominant_idx]

    # Map spectral signature to (α, β)
    #
    # Physical interpretation:
    # - Symmetric curl (α = β): spectral peak at exactly ω = 0
    # - α > β: curl peaks toward left edge, slight positive frequency shift
    # - β > α: curl peaks toward right edge, slight negative frequency shift
    #
    # The magnitude comes from the residual variance.

    # Normalize frequency to [-1, 1] range
    asymmetry = np.clip(dominant_freq / (np.pi / 4), -1, 1)

    # Scale factor (calibrated empirically; may need tuning)
    scale = curvature_magnitude * 8.0

    # Map to α, β
    # When asymmetry = 0: α = β (symmetric)
    # When asymmetry > 0: α > β
    # When asymmetry < 0: β > α
    alpha = scale * (1 + asymmetry * 0.5)
    beta = scale * (1 - asymmetry * 0.5)

    return alpha, beta


def estimate_cubic_params_from_spans(
    span_points: list[np.ndarray],
    ycoords: np.ndarray,
    page_width: float,
) -> tuple[float, float]:
    """Aggregate curl estimates from multiple spans.

    Each span provides an independent estimate of the page curl.
    We combine them using a weighted average, where weight is span length.

    Args:
        span_points: List of arrays, each of shape (N_i, 1, 2).
        ycoords: Array of mean y-positions for each span.
        page_width: Estimated page width (for scaling).

    Returns:
        A tuple (alpha, beta) giving the aggregated curl parameter estimates.

    """
    if not span_points:
        return 0.0, 0.0

    alpha_estimates = []
    beta_estimates = []
    weights = []

    for sp in span_points:
        residuals = extract_curl_residuals(sp)
        if residuals is None:
            continue

        # Estimate x-span from the points
        pts = sp.reshape((-1, 2))
        x_span = pts[:, 0].max() - pts[:, 0].min()
        if x_span < 0.01:
            continue

        alpha, beta = estimate_curl_from_toeplitz(residuals, x_span)

        # Weight by number of points (longer spans are more reliable)
        weight = len(residuals)

        alpha_estimates.append(alpha)
        beta_estimates.append(beta)
        weights.append(weight)

    if not alpha_estimates:
        return 0.0, 0.0

    # Weighted average
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()

    alpha = np.dot(alpha_estimates, weights)
    beta = np.dot(beta_estimates, weights)

    # Clamp to reasonable range (matching projection.py clipping)
    alpha = np.clip(alpha, -0.5, 0.5)
    beta = np.clip(beta, -0.5, 0.5)

    return float(alpha), float(beta)
