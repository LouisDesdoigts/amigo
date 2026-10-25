"""Image response and eigenmode selection for AMIOptics.visibility_model."""

import dLux.utils as dlu
import jax.numpy as np
import zodiax as zdx

from .vis_models import inject_vis


def visibility_jtj(
    intensity, knot_modes, coords, *, pad=2, downsample=1,
    batch_size=8, checkpoint=True,
):
    """Calculate unweighted image J.T @ J at zero visibility coefficients."""
    def image(coefficients):
        result = inject_vis(intensity, coefficients @ knot_modes, coords, pad=pad)
        return dlu.downsample(result, downsample, mean=False).reshape(-1)

    # Differentiate the image directly, without noise or PSF weighting.
    dimension = len(knot_modes)
    jtj = zdx.gauss_newton(
        image, np.zeros(dimension),
        nbatches=(dimension + batch_size - 1) // batch_size,
        checkpoint=checkpoint,
    ).matrix
    if not bool(np.all(np.isfinite(jtj))):
        raise FloatingPointError("visibility JTJ contains non-finite values")
    return jtj


def select_visibility_modes(jtj, knot_modes, *, n_modes=None, retained_trace=0.99):
    """Return leading complex modes and the full descending JTJ spectrum."""
    dimension = len(knot_modes)
    if n_modes is not None:
        if isinstance(n_modes, bool) or not isinstance(n_modes, int) or not 1 <= n_modes <= dimension:
            raise ValueError(f"n_modes must be between 1 and {dimension}")
    elif not 0 < retained_trace <= 1:
        raise ValueError("retained_trace must lie in (0, 1]")

    # Symmetrize roundoff, then order modes by image response.
    eigenvalues, vectors = np.linalg.eigh((jtj + jtj.T) / 2)
    eigenvalues, rows = eigenvalues[::-1], vectors[:, ::-1].T
    positive = np.maximum(eigenvalues, 0)
    if not bool(positive.sum() > 0):
        raise ValueError("optical model has no visibility response")
    if n_modes is None:
        cumulative = np.cumsum(positive) / positive.sum()
        n_modes = min(int(np.searchsorted(cumulative, retained_trace)) + 1, dimension)
    if not bool(np.all(eigenvalues[:n_modes] > 0)):
        raise ValueError("requested modes include a zero or negative JTJ eigenvalue")

    # Fix the arbitrary eigenvector sign before expressing modes on the UV grid.
    rows = rows[:n_modes]
    pivots = np.argmax(np.abs(rows), axis=1)
    rows = rows * np.sign(rows[np.arange(n_modes), pivots])[:, None]
    return rows @ knot_modes, eigenvalues
