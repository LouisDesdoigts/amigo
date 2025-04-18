import jax.numpy as np
from jax import vmap, lax
from jax.scipy.stats import multivariate_normal as mvn
import pkg_resources as pkg


# Noise modelling
def get_read_cov(read_noise, ngroups):
    # Bind the read noise function
    # (Is this just an outer product?, find a friendlier syntax)
    raw_read_fn = lambda i, j: np.eye(ngroups) * read_noise[i, j] ** 2
    read_fn = vmap(vmap(raw_read_fn, (0, None), (2)), (None, 0), (2))

    # Get the read noise covariance matrix
    pix_idx = np.arange(read_noise.shape[-1])
    return read_fn(pix_idx, pix_idx)


# def build_cov(var):
#     Is = np.arange(len(var))
#     IJs = np.array(np.meshgrid(Is, Is))

#     vals = vmap(vmap(vmap(lambda ind: var[ind], 0), 1), 0)(IJs)
#     cov = np.min(vals, (0))
#     return cov


def get_slope_cov_mask(n_slope):
    tri = np.tri(n_slope, n_slope, 1)
    mask = (tri * tri.T) - np.eye(n_slope)
    return -mask
    # return -(read_noise**2) * mask


def build_cov(var, read_std):
    # Get the slope covariance matrix (diagonal)
    slope_cov = np.eye(len(var))[..., None, None] * var[None, ...]

    # Get the read noise covariance mask
    slope_cov_mask = get_slope_cov_mask(len(var))

    # Create the read noise covariance matrix
    read_cov = read_std[None, None, ...] * slope_cov_mask[..., None, None]

    # Return the combined covariance matrix
    return slope_cov + read_cov


# def log_likelihood(slope, exposure, read_noise=0, return_im=False):
def log_likelihood(slope, exposure, return_im=False):
    """
    Note we have the infrastructure for dealing with the slope read noise
    covariance, but it seems to give nan likelihoods when read_noise > ~6. As such
    we leave the _capability_ here but set the read_noise to default of zero.
    """
    # Get the model, data, and variances
    slope_vec = exposure.to_vec(slope)
    data_vec = exposure.to_vec(exposure.slopes)
    cov_vec = exposure.to_vec(exposure.cov)

    # # Get the covariance matrix from the data variance (diagonal)
    # cov_vec = np.eye(exposure.nslopes)[None, ...] * var_vec[..., None]

    # # Get the read noise covariance and combine with the data covariance
    # read_cov_mask = get_slope_cov_mask(exposure.nslopes)
    # cov_vec += (read_noise * read_cov_mask / exposure.nints)[None, ...]

    # Calculate per-pixel likelihood
    loglike_fn = vmap(lambda x, mu, cov: mvn.logpdf(x, mu, cov))
    loglike_vec = loglike_fn(slope_vec, data_vec, cov_vec)

    # Return image or vector
    if return_im:
        # NOTE: Adds nans to the empty spots
        return exposure.from_vec(loglike_vec)
    return loglike_vec


def prior(*args, **kwargs):
    return 0.0


def posterior(model, exposure, per_pix=True, return_im=False):
    # Get the model
    slopes = exposure(model)
    posterior_vec = prior(model, slopes, exposure) + log_likelihood(slopes, exposure)

    # return image
    if return_im:
        return exposure.from_vec(posterior_vec)

    # return per pixel or total
    if per_pix:
        return np.nanmean(posterior_vec)
    return np.nansum(posterior_vec)


# def reg_loss_fn(model, exposure, args):
#     return -np.array(posterior(model, exposure, per_pix=True)).sum()


def check_symmetric(mat):
    """Checks if a matrix is symmetric"""
    return np.allclose(mat, mat.T)


def check_positive_semi_definite(mat):
    """Checks if a matrix is positive semi-definite"""
    return lax.cond(
        np.isnan(mat).any(),
        lambda x: False,
        lambda x: np.all(np.linalg.eigvals(mat) >= 0),
        mat,
    )


def variance_model(model, exposure, true_read_noise=False, read_noise=10):
    """
    True read noise will use the CRDS read noise array, else it will use a constant
    value as determined by the input. true_read_noise therefore supersedes read_noise.
    Using a flat value of 10 seems to be more accurate that the CRDS array.

    That said I think the data has overly ambitious variances as a consequence of the
    sigma clipping that is performed. We could determine the variance analytically from
    the variance of the individual pixel values, but we will look at this later.
    """

    nan_mask = np.isnan(exposure.slopes)

    # Estimate the photon covariance
    slopes = exposure(model)  # .model(exposure)

    # TODO: Update with exposure.pixel_support
    slopes = slopes.at[np.where(nan_mask)].set(np.nan)
    variance = slopes / exposure.nints

    # Read noise covariance
    if true_read_noise:
        rn = np.load(pkg.resource_filename(__name__, "data/SUB80_readnoise.npy"))
    else:
        rn = read_noise
    read_variance = (rn**2) * np.ones((80, 80)) / exposure.nints
    variance += read_variance[None, ...]

    return slopes, variance


def covariance_model(model, exposure):
    """
    True read noise will use the CRDS read noise array, else it will use a constant
    value as determined by the input. true_read_noise therefore supersedes read_noise.
    Using a flat value of 10 seems to be more accurate that the CRDS array.

    That said I think the data has overly ambitious variances as a consequence of the
    sigma clipping that is performed. We could determine the variance analytically from
    the variance of the individual pixel values, but we will look at this later.
    """
    # Estimate the photon covariance
    slopes = exposure(model)

    # Pixel read noise
    read_std = np.load(pkg.resource_filename(__name__, "data/SUB80_readnoise.npy"))
    # read_std = np.load("../../amigo/src/amigo/data/SUB80_readnoise.npy")
    read_var = read_std**2

    # Add the 2x read noise to the variance
    variance = 2 * read_var + slopes

    # Build the covariance matrix
    cov = build_cov(variance, read_std)

    # Get the covariance matrix support - slightly more complex than it seems since the
    # the off diagonal terms are constructed from two different reads, which can both
    # have different support values. Here I simply take the mean support over both
    # reads, constructed in such a way to match the entries of the covariance matrix.
    support = exposure.slope_support
    cov_support = (support[None, ...] + support[:, None, ...]) / 2
    cov /= cov_support

    return slopes, cov
