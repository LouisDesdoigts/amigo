import jax.numpy as np
from jax import vmap, lax
from .detector_layers import model_amplifier


def total_read_noise(bias, one_on_fs):
    return bias[None, ...] + vmap(model_amplifier)(one_on_fs)


def total_amplifier_noise(one_on_fs):
    return vmap(model_amplifier)(one_on_fs)


# Noise modelling
def get_read_cov(read_noise, ngroups):
    # Bind the read noise function
    # (Is this just an outer product?, find a friendlier syntax)
    raw_read_fn = lambda i, j: np.eye(ngroups) * read_noise[i, j] ** 2
    read_fn = vmap(vmap(raw_read_fn, (0, None), (2)), (None, 0), (2))

    # Get the read noise covariance matrix
    pix_idx = np.arange(read_noise.shape[-1])
    return read_fn(pix_idx, pix_idx)


def build_cov(var):
    Is = np.arange(len(var))
    IJs = np.array(np.meshgrid(Is, Is))

    vals = vmap(vmap(vmap(lambda ind: var[ind], 0), 1), 0)(IJs)
    cov = np.min(vals, (0))
    return cov


import pkg_resources as pkg


def variance_model(model, exposure, true_read_noise=False, read_noise=10):
    """
    True read noise will use the CRDS read noise array, else it will use a constant
    value as determined by the input. true_read_noise therefore supersedes read_noise.
    Using a flat value of 10 seems to be more accurate that the CRDS array.

    That said I think the data has overly ambitious variances as a consequence of the
    sigma clipping that is performed. We could determine the variance analytically from
    the variance of the individual pixel values, but we will look at this later.
    """

    nan_mask = np.isnan(exposure.data)

    # Estimate the photon covariance
    psf = model_fn(model, exposure)

    psf = psf.at[np.where(nan_mask)].set(np.nan)
    variance = psf / exposure.nints

    # Read noise covariance
    if true_read_noise:
        rn = np.load(pkg.resource_filename(__name__, "data/SUB80_readnoise.npy"))
    else:
        rn = read_noise
    read_variance = (rn**2) * np.ones((80, 80)) / exposure.nints
    variance += read_variance[None, ...]

    return psf, variance


import jax.numpy as np
from jax.scipy.stats import norm
from amigo.FIM import FIM
from amigo.modelling import model_fn

# # Actual posterior
# # def posterior(model, exposure, model_fn, per_pix=True, zero_idx=-1, **kwargs):
# def posterior(model, exposure, per_pix=True, **kwargs):
#     slope = model_fn(model, exposure, **kwargs)
#     posterior_vec = exposure.loglike_vec(slope)
#     posterior = np.nansum(posterior_vec)
#     if per_pix:
#         return posterior / posterior_vec.size
#     return posterior


def log_likelihood(x, mean, var):
    return norm.logpdf(x, mean, np.sqrt(var))


def posterior(model, exposure, per_pix=True, as_psf=False, photon=False, return_image=False, **kwargs):
    to_vec = lambda x: exposure.to_vec(x)
    slopes = to_vec(model_fn(model, exposure, **kwargs))
    data = to_vec(exposure.data)
    var = to_vec(exposure.variance)

    # Probs dont need this anymore
    var = np.abs(var)

    # plt.title("Slopes")
    # plt.imshow(slopes.sum(0))
    # plt.colorbar()
    # plt.show()

    # plt.title("Data")
    # plt.imshow(data.sum(0))
    # plt.colorbar()
    # plt.show()

    # plt.title("Variance")
    # plt.imshow(var.sum(0))
    # plt.colorbar()
    # plt.show()

    if photon:
        posterior = log_likelihood(slopes.sum(0), data.sum(0), var.sum(0))
    else:
        posterior = log_likelihood(slopes, data, var)

    # plt.title("Posterior")
    # plt.imshow(-posterior)
    # plt.colorbar()
    # plt.show()

    if return_image:
        return posterior

    if per_pix:
        return np.nanmean(posterior)
    return np.nansum(posterior)


# def loss_fn(model, exposures, **kwargs):
def loss_fn(model, args, **kwargs):
    # exposures, step_mappers, model_fn = args
    exposures, step_mappers = args
    return -np.array([posterior(model, exp, per_pix=True, **kwargs) for exp in exposures]).sum()


# def build_covariance_matrix(var, read_noise=None, min_value=True):
#     """
#     The off-diagonal covariance terms cov(i, j), can be the minimum value of:
#         1. The value: min(var(i), var(j))
#         2. The index: var(min(i, j))

#     if min_value is True (default), then the minimum value is chosen, otherwise the
#     minimum index is chosen. Testing show min index results in some data sets being
#     majority nan, as the resulting covariance matrix is non symmetric or positive
#     semi-definite.

#     Read noise can optional be added to the diagonal terms.
#     """
#     Is = np.arange(len(var))
#     IJs = np.array(np.meshgrid(Is, Is))

#     if min_value:
#         vals = vmap(vmap(vmap(lambda ind: var[ind], 0), 1), 0)(IJs)
#         cov = np.min(vals, (0))
#     else:
#         inds = vmap(vmap(vmap(lambda ind: ind, 0), 1), 0)(IJs)
#         cov = var[np.min(inds, 0)]

#     if read_noise is not None:
#         cov += get_read_cov(read_noise, len(var))

#     return cov


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
