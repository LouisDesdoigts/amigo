"""Construct visibility modes insensitive to selected parameter responses.

Response matrices have observables in rows and parameter derivatives in
columns. Reference models and derivative generation belong to the caller.
"""

import jax.numpy as np


_RANK_RTOL = 1e-12


def _response_matrix(response):
    """Require finite, real derivatives without discarding complex parts."""
    response = np.asarray(response)
    if response.ndim != 2 or np.iscomplexobj(response):
        raise ValueError("response must be a real, two-dimensional matrix")
    if not bool(np.all(np.isfinite(response))):
        raise ValueError("response contains non-finite values")
    return response


def fourier_null(null_jac):
    """Return K with K @ null_jac = 0.

    ``null_jac`` is the direct Fourier response already projected
    into the real visibility coefficients, with shape (n_visibility, n_null).
    Rows of K are an orthonormal basis of the retained coefficient space.
    """
    response = _response_matrix(null_jac)
    if response.shape[1] == 0:
        return np.eye(response.shape[0], dtype=response.dtype)

    # Remove the full independent span selected for nulling, including redundant columns.
    left, singular, _ = np.linalg.svd(response, full_matrices=True)
    rank = int(np.sum(singular > _RANK_RTOL * singular.max(initial=0)))
    return left[:, rank:].T


def image_null(
    vis_jac, null_jac, *, weighting=None,
):
    """Return the canonical image-response null acting on visibility coefficients.

    Supply UNWEIGHTED derivatives, with identical pixels in the same row order.
    ``weighting=None`` gives every pixel equal weight (the default).
    Supply a PSF intensity array as ``weighting`` to divide both Jacobians by
    its square root. Pass intensities, not precomputed inverse-variance weights.
    The PSF is flattened in C order to match the response rows. Pixels at or
    below 1e-14 of its peak are excluded from both matrices to avoid division
    by effectively zero intensity. This cutoff is only used for PSF weighting.

    ``vis_jac`` has shape (n_pixels, n_visibility) and must have
    full column rank. ``null_jac`` has shape (n_pixels, n_null).
    K nulls the least-squares coefficient response to the selected image perturbations;
    its normalization follows the existing canonical decomposition.
    """
    visibility = _response_matrix(vis_jac)
    null = _response_matrix(null_jac)
    if visibility.shape[0] != null.shape[0]:
        raise ValueError("visibility and null responses must use the same pixels")

    # Apply the chosen image metric identically to both derivative matrices.
    if weighting is not None:
        psf = np.asarray(weighting).reshape(-1)
        if np.iscomplexobj(psf) or psf.shape != (visibility.shape[0],):
            raise ValueError("weighting PSF must contain one real intensity per response row")
        if not bool(np.all(np.isfinite(psf)) & np.all(psf >= 0) & np.any(psf > 0)):
            raise ValueError("weighting PSF must be finite, nonnegative and nonzero")
        keep = psf > 1e-14 * psf.max()
        scale = 1 / np.sqrt(psf[keep])
        visibility = visibility[keep] * scale[:, None]
        null = null[keep] * scale[:, None]

    if not 0 < visibility.shape[1] <= visibility.shape[0]:
        raise ValueError("visibility response must have independent coefficient columns")

    # Express the two response spans in orthonormal image coordinates.
    q_visibility, singular, right = np.linalg.svd(visibility, full_matrices=False)
    if not bool(np.all(singular > _RANK_RTOL * singular[0])):
        raise ValueError("visibility response is rank deficient")
    model_whitener = singular[:, None] * right
    if null.shape[1] == 0:
        return model_whitener
    q_null, null_singular, _ = np.linalg.svd(null, full_matrices=False)
    null_rank = int(np.sum(
        null_singular > _RANK_RTOL * null_singular.max(initial=0)
    ))
    q_null = q_null[:, :null_rank]
    if null_rank == 0:
        return model_whitener

    # Null their overlap, then map back to the original visibility coefficients.
    overlap = q_visibility.T @ q_null
    left, correlations, _ = np.linalg.svd(overlap, full_matrices=True)
    # Canonical correlations are dimensionless and bounded by one; an absolute
    # tolerance also handles spans whose entire overlap is roundoff.
    rank = int(np.sum(correlations > _RANK_RTOL))
    return left[:, rank:].T @ model_whitener


def null_modes(
    null_jac, *, image_plane=False, vis_jac=None,
    weighting=None,
):
    """Build a visibility null with one switch between the two strategies.

    ``null_jac`` contains derivatives with respect to parameters selected for
    nulling. With ``image_plane=False``, its rows are visibility coefficients.
    With ``image_plane=True``, both Jacobians use the same unweighted pixels.
    For the image path,
    ``weighting=None`` uses uniform weighting; ``weighting=psf`` divides both
    Jacobians by sqrt(psf), with pixels ordered as documented in image_null.

    Both paths return K for ``nulled = K @ coefficients``. Propagate models
    and covariance through the same K, using ``K @ covariance @ K.T``.
    """
    if image_plane:
        if vis_jac is None:
            raise ValueError("image-plane nulling requires vis_jac")
        return image_null(
            vis_jac, null_jac,
            weighting=weighting,
        )
    if weighting is not None:
        raise ValueError("image weighting requires image_plane=True")
    return fourier_null(null_jac)
