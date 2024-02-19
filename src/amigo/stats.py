import jax.numpy as np
from jax import vmap, lax
from detector_layers import model_amplifier


def total_read_noise(bias, one_on_fs):
    return bias[None, ...] + vmap(model_amplifier)(one_on_fs)


# Noise modelling
def get_read_cov(read_noise, ngroups):
    # Bind the read noise function
    # (Is this just an outer product?, find a friendlier syntax)
    raw_read_fn = lambda i, j: np.eye(ngroups) * read_noise[i, j] ** 2
    read_fn = vmap(vmap(raw_read_fn, (0, None), (2)), (None, 0), (2))

    # Get the read noise covariance matrix
    pix_idx = np.arange(read_noise.shape[-1])
    return read_fn(pix_idx, pix_idx)


def build_covariance_matrix(std, read_noise=None, min_value=True):
    """
    The off-diagonal covariance terms cov(i, j), can be the minimum value of:
        1. The value: min(var(i), var(j))
        2. The index: var(min(i, j))
    
    if min_value is True (default), then the minimum value is chosen, otherwise the 
    minimum index is chosen. Testing show min index results in some data sets being
    majority nan, as the resulting covariance matrix is non symmetric or positive 
    semi-definite.
    
    Read noise can optional be added to the diagonal terms.
    """
    var = std ** 2
    Is = np.arange(len(var))
    IJs = np.array(np.meshgrid(Is, Is))

    if min_value:
        vals = vmap(vmap(vmap(lambda ind: var[ind], 0), 1), 0)(IJs)
        cov = np.min(vals, (0))
    else:
        # inds = np.min(vmap(vmap(vmap(lambda ind: ind, 0), 1), 0)(IJs), (0))
        inds = vmap(vmap(vmap(lambda ind: ind, 0), 1), 0)(IJs)
        cov = var[np.min(inds, 0)]

    if read_noise is not None:
        cov += get_read_cov(read_noise, len(std))
    
    return cov

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
