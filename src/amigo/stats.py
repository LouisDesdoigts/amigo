import jax.numpy as np
from jax import vmap
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


def build_covariance_matrix(electrons):
    Is = np.arange(len(electrons))
    IJs = np.array(np.meshgrid(Is, Is))
    vals = vmap(vmap(vmap(lambda ind: electrons[ind], 0), 1), 0)(IJs)
    cov = np.min(vals, (0))
    return cov


def get_covariance_matrix(data_ramp, total_bias, read_noise):
    electron_ramp = data_ramp - total_bias
    electron_cov = build_covariance_matrix(electron_ramp)
    read_cov = get_read_cov(read_noise, len(data_ramp))
    return electron_cov + read_cov


def check_symmetric(mat):
    """Checks if a matrix is symmetric"""
    return np.allclose(mat, mat.T)


def check_positive_semi_definite(mat):
    """Checks if a matrix is positive semi-definite"""
    if np.isnan(mat).any():
        return False
    return np.all(np.linalg.eigvals(mat) >= 0)
