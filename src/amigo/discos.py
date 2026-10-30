"""Separate Fourier and image nulling workflows for visibility coefficients.

Build one null at a chosen reference and reuse it for data in the same basis.
Parameter defaults belong to the pipeline. Jacobian rows are observables;
columns are selected parameters. Apply every returned K to predictions and
covariance too: K @ prediction and K @ covariance @ K.T.
"""

import jax.numpy as np
import zodiax as zdx
from .misc import spectral_weights

# Shared parameter selection and differentiation


def model_jac(model, params, fn, *, nbatches=1, checkpoint=False):
    """Return the unweighted Jacobian of ``fn(model)``.

    ``params`` contains explicit model paths, with no default selection.
    Columns follow the supplied path order, then C order within each array.
    The real array returned by ``fn`` is flattened in C order into rows.
    Other model params remain at their reference values. Existing gradient
    stops in the forward model are respected.

    Returns a Zodiax Jacobian labelled with the selected model paths. Use
    ``result.matrix`` when passing responses to the nulling functions.
    """
    paths = [params] if isinstance(params, str) else list(params)
    if not paths:
        raise ValueError("Select at least one parameter path")
    # Select physical parameter arrays without changing their units or scales.
    layout = zdx.TreeLayout.from_paths(model, paths)
    values = tuple(np.asarray(model.get(path)) for path in paths)

    # Zodiax owns flattening the params, differentiation and column batching.
    def output(selected):
        reference = model.set(paths, list(selected))
        return np.asarray(fn(reference)).reshape(-1)

    result = zdx.jacobian(output, values, nbatches=nbatches, checkpoint=checkpoint)
    return zdx.Jacobian(result.matrix, layout)


def _check_null_jac(jac):
    """Do not silently accept a selected parameter with no response."""
    for path, column in jac.columns().items():
        if not bool(np.any(column != 0)):
            raise ValueError(f"{path} has no response in the selected plane at this reference")


# Fourier workflow


class _VisReference(zdx.Base):
    """Keep the optical state and source spectrum independently differentiable."""

    optics: zdx.Base
    spectra: np.ndarray


def vis_jac(optics, vis_model, filt, params, *, spectra, nbatches=1, checkpoint=False):
    """Differentiate optical coefficients at one explicit, exposure-free reference.

    params selects paths within optics, plus optional "spectra" for the source
    spectral slope. Returned optical column labels are prefixed with "optics.".
    Position is zero and source flux is unity. The supplied visibility basis is
    held fixed; detector and fitted visibility parameters do not enter.
    """
    params = [params] if isinstance(params, str) else list(params)
    paths = [p if p == "spectra" else "optics." + p for p in params]
    reference = _VisReference(optics, np.asarray(spectra))

    def coefficients(ref):
        wavelengths, throughput = ref.optics.filters[filt]
        weights = spectral_weights(throughput, ref.spectra)
        intensity = ref.optics.propagate(
            wavelengths,
            weights=weights,
            return_intensity=True,
        )
        return vis_model.intensity_to_latent(intensity, filt)

    return model_jac(reference, paths, coefficients, nbatches=nbatches, checkpoint=checkpoint)


def vis_null(optics, vis_model, filt, null_params, *, spectra, nbatches=1, checkpoint=False):
    """Build one Fourier null to reuse for all data in this visibility basis.

    Choose a common optical state and spectrum explicitly. Do not regenerate K
    per exposure when combining or calibrating the projected measurements.
    """
    jac = vis_jac(
        optics,
        vis_model,
        filt,
        null_params,
        spectra=spectra,
        nbatches=nbatches,
        checkpoint=checkpoint,
    )
    _check_null_jac(jac)
    return vis_kernel(jac.matrix)


# Image workflow


def image_jacs(model, exp, null_params, *, nbatches=1, checkpoint=False):
    """Return (vis_jac, null_jac) through the existing exposure forward model.

    exp selects ONE reference exposure, including its recovered spectrum,
    visibility coefficients, detector/read response and valid pixel support.
    Rows follow exp.to_vec(exp(model)), flattened in C order. There is no extra
    normalization, background pedestal, sampling rule or statistical weighting.
    Only parameters wired into this forward model can have a response; its
    existing gradient stops are respected.
    """

    def pixels(reference):
        return exp.to_vec(exp(reference))

    vis_jac = model_jac(
        model, exp.map_param("vis"), pixels, nbatches=nbatches, checkpoint=checkpoint
    )
    null_jac = model_jac(model, null_params, pixels, nbatches=nbatches, checkpoint=checkpoint)
    return vis_jac, null_jac


def image_null(model, exp, null_params, *, weighting=None, nbatches=1, checkpoint=False):
    """Build one image-overlap null from a deliberately selected reference.

    Reuse K for all measurements in the same visibility basis. exp selects the
    reference for construction, not an instruction to build a null per exposure.
    weighting=None is uniform; an explicit nonnegative intensity array must
    match exp.to_vec(exp(model)) in shape/order (including group differences).
    """
    vis_jac, null_jac = image_jacs(
        model, exp, null_params, nbatches=nbatches, checkpoint=checkpoint
    )
    _check_null_jac(null_jac)
    return image_kernel(vis_jac.matrix, null_jac.matrix, weighting=weighting)


# Null modes

_RANK_RTOL = 1e-12


def _response_matrix(response):
    """Require finite, real derivatives without discarding complex parts."""
    response = np.asarray(response)
    if response.ndim != 2 or np.iscomplexobj(response):
        raise ValueError("response must be a real, two-dimensional matrix")
    if not bool(np.all(np.isfinite(response))):
        raise ValueError("response contains non-finite values")
    return response


def vis_kernel(null_jac):
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


def image_kernel(
    vis_jac,
    null_jac,
    *,
    weighting=None,
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
    null_rank = int(np.sum(null_singular > _RANK_RTOL * null_singular.max(initial=0)))
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


# Coefficient calibration


def calibrate(c, C, cal, Ccal):
    """Subtract independent calibrator coefficients in the same basis.

    c and cal are real coefficient vectors; C and Ccal are their full
    covariance matrices. Shared-calibrator correlations between different
    calibrated observations must be retained separately by the caller.
    """
    return c - cal, C + Ccal
