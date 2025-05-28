import jax.numpy as np
import jax.scipy as jsp
from jax import vmap, lax
import equinox as eqx
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


def batched_jacobian(X, fn, n_batch=1):
    Xs = np.array_split(X, n_batch)
    rebuild = lambda X_batch, index: X.at[index : index + len(X_batch)].set(X_batch)
    lens = np.cumsum(np.array([len(x) for x in Xs]))[:-1]
    starts = np.concatenate([np.array([0]), lens])

    @eqx.filter_jacfwd
    def batched_jac_fn(x, index):
        return eqx.filter_jit(fn)(rebuild(x, index))

    return np.concatenate([batched_jac_fn(x, index) for x, index in zip(Xs, starts)], axis=-1).T


def gauss_hessian(J, cov):
    # Gauss-Newton hessian approximation under the assumption of a multivariate normal
    return J @ (np.linalg.inv(cov) @ J.T)


def mv_zscore(x, mu, cov):
    """Multivariate z-score, return identical gradients to normal log-likelihood"""
    return -0.5 * np.dot(x - mu, np.dot(np.linalg.inv(cov), x - mu))


def loglike(x, mu, cov):
    """Multivariate log-likelihood"""
    return jsp.stats.multivariate_normal.logpdf(x, mean=mu, cov=cov)


def svd(jacobian, normalise=True):
    u, s, vh = np.linalg.svd(jacobian, full_matrices=True)
    if normalise:
        s /= s[0]
    return u, s, vh


def decompose(hess, normalise=True):
    # Get the eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(hess)
    eigvecs, eigvals = eigvecs.real.T, eigvals.real

    # Normalise
    if normalise:
        eigvals /= eigvals[0]
    return eigvals, eigvecs


# def orthogonalise(x, cov):
#     eig_vals, eig_vecs = np.linalg.eig(cov)
#     eig_vals, eig_vecs = eig_vals.real, eig_vecs.real.T
#     ortho_cov = np.dot(eig_vecs, np.dot(cov, np.linalg.inv(eig_vecs)))
#     ortho_x = np.dot(eig_vecs, x)
#     return ortho_x, ortho_cov, eig_vecs, eig_vals


def orthogonalise(x, cov):
    # P = calc_projection(cov=cov, unit=normalise)[0]
    eig_vals, P = decompose(cov, normalise=False)
    ortho_cov = P @ (cov @ P.T)
    ortho_x = np.dot(P, x)
    return ortho_x, ortho_cov, P


def calc_projection(fmat=None, cov=None, unit=True):
    if fmat is None and cov is None:
        raise ValueError("Must provide either fmat or cov")
    mat = fmat if fmat is not None else cov

    # Dont normalise the values since we need them
    eig_vals, eig_vecs = decompose(mat, normalise=False)

    # Eigen values of cov and fisher matrices are inverse
    if fmat is not None:
        scale = eig_vals**0.5
    else:
        scale = eig_vals**-0.5

    # Project to orthogonal or ortho-normal space
    projection = eig_vecs
    if unit:
        projection *= scale[:, None]
    return projection, eig_vecs, eig_vals


def weighted_average(values, errors):
    var = errors**2
    weights = 1 / var
    average = np.sum(weights * values) / np.sum(weights)
    uncertainty = np.sqrt(np.sum(np.square(weights * errors))) / np.sum(weights)
    return average, uncertainty


def bin_data(x, y, bin_inds):
    return np.array([weighted_average(x[inds], y[inds]) for inds in bin_inds]).T


def chi2(x, pred, std, ddof):
    res = x - pred
    z_score = res / std
    chi2 = np.nansum(z_score**2) / ddof
    return chi2
