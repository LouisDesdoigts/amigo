import jax.numpy as np
from jax import vmap, lax
from jax.scipy.stats import multivariate_normal as mvn


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


def get_slope_cov(n_slope, read_noise):
    tri = np.tri(n_slope, n_slope, 1)
    mask = (tri * tri.T) - np.eye(n_slope)
    return -(read_noise**2) * mask


def log_likelihood(slope, exposure, read_noise=0):
    """
    Note we have the infrastructure for dealing with the slope read noise
    covariance, but it seems to give nan likelihoods when read_noise > ~6. As such
    we leave the _capability_ here but set the read_noise to default of zero.
    """

    # Get the model, data, and variances
    slope_vec = exposure.to_vec(slope)
    data_vec = exposure.to_vec(exposure.slopes)
    var_vec = exposure.to_vec(exposure.variance)

    # Get th build we need to deal with the covariance terms
    cov = get_slope_cov(exposure.nslopes, read_noise) / exposure.nints
    eye = np.eye(exposure.nslopes)

    # Bind the likelihood function
    loglike_fn = lambda x, mu, var: mvn.logpdf(x, mu, (eye * var) + cov)

    # Calculate per-pixel likelihood
    return vmap(loglike_fn, (0, 0, 0))(slope_vec, data_vec, var_vec)


def prior(*args):
    return 0.0


def posterior(model, exposure, per_pix=True, return_im=False):
    # Get the model
    slopes = model.model(exposure)  # , slopes=True)
    posterior_vec = prior(model, slopes, exposure) + log_likelihood(slopes, exposure)

    # return image
    if return_im:
        return exposure.from_vec(posterior_vec)

    # return per pixel or total
    if per_pix:
        return np.nanmean(posterior_vec)
    return np.nansum(posterior_vec)


def loss_fn(model, exposure):
    return -np.array(posterior(model, exposure, per_pix=True)).sum()


def batch_loss_fn(model, batch):
    return -np.array([posterior(model, exp, per_pix=True) for exp in batch]).sum()


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
