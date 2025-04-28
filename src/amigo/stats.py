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
    # 2x here to account for the two reads that contribute to the slope
    read_cov = 2 * read_std[None, None, ...] * slope_cov_mask[..., None, None]

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


def variance_model(model, exposure):
    slopes, cov = covariance_model(model, exposure)
    inds = np.arange(len(slopes))
    return slopes, cov[inds, inds]


def covariance_model(model, exposure):
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


from jax.flatten_util import ravel_pytree
import equinox as eqx


def batched_jacobian(X, fn, n_batch=1):
    Xs = np.array_split(X, n_batch)
    rebuild = lambda X_batch, index: X.at[index : index + len(X_batch)].set(X_batch)
    lens = np.cumsum(np.array([len(x) for x in Xs]))[:-1]
    starts = np.concatenate([np.array([0]), lens])

    @eqx.filter_jacfwd
    def batched_jac_fn(x, index):
        return eqx.filter_jit(fn)(rebuild(x, index))

    return np.concatenate([batched_jac_fn(x, index) for x, index in zip(Xs, starts)], axis=-1).T


def model_batched_jacobian(fn, model, exp, params, n_batch=1):
    key_fn = lambda param: exp.map_param(param)
    params = {key_fn(key): model.get(key_fn(key)) for key in params}
    X, unravel_fn = ravel_pytree(params)
    Xs = np.array_split(X, n_batch)

    rebuild = lambda X_batch, index: X.at[index : index + len(X_batch)].set(X_batch)
    lens = np.cumsum(np.array([len(x) for x in Xs]))[:-1]
    starts = np.concatenate([np.array([0]), lens])

    @eqx.filter_jacfwd
    def batched_jac_fn(x, index, model, exp):
        params = unravel_fn(rebuild(x, index))
        for param, value in params.items():
            model = model.set(param, value)
        return eqx.filter_jit(fn)(model, exp)

    return np.concatenate(
        [batched_jac_fn(x, index, model, exp) for x, index in zip(Xs, starts)], axis=-1
    )


def decompose(J, values):
    # Get the covariance matrix
    cov = np.eye(values.size) * values[..., None]

    # Get the hessian
    hess = J @ (cov @ J.T)

    # Get the eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(hess)
    eigvecs, eigvals = np.real(eigvecs).T, np.real(eigvals)

    #
    eigvals /= eigvals[0]
    return hess, eigvals, eigvecs
