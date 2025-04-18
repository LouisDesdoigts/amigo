import jax.numpy as np
import optimistix as optx
from jax.scipy.stats import norm
import equinox as eqx
from jax import vmap
import dLux.utils as dlu
from drpangloss.models import BinaryModelCartesian

# def batched_fn(fn, array, n_batch=1):
#     print(array.shape)
#     batches = np.array_split(array, n_batch)
#     out = []
#     for batch in batches:
#         out.append(fn(batch))
#     out = np.concatenate(out)
#     print(out.shape)
#     return out.reshape(array.shape[1:])  # check the shapes output here


@eqx.filter_jit
def loglike(values, params, data_obj, model_class):
    """
    Abstract log-likelihood function for a given model class and data object, assuming
    Gaussian errors.

    Parameters
    ----------
    values : array-like
        Values of the model parameters.
    params : list
        List of parameter names.
    data_obj : OIData
        Object containing the data to be fitted.
    model_class : class
        Model class to be fitted to the data.

    Returns
    -------
    float
        Log-likelihood value.
    """
    model_data = data_obj.model(model_class(**dict(zip(params, values))))
    data, errors = data_obj.flatten_data()
    return norm.logpdf(model_data, loc=data, scale=errors).sum()


def likelihood_grid(data_obj, model_class, samples_dict, n_batch=50):
    """
    Function to vmap a likelihood function over a grid of parameter values provided in
    a dictionary.

    Parameters
    ----------
    data_obj : OIData
        Object containing the data to be fitted.
    model_class : class
        Model class to be fitted to the data.
    samples_dict: dict
        Dictionary of parameter names and values to be fitted to the data.

    Returns
    -------
    array-like
        Log-likelihood values over the grid of parameter values.
    """

    params = list(samples_dict.keys())
    samples = samples_dict.values()
    vals = np.array(np.meshgrid(*samples))
    vals_vec = vals.reshape((len(vals), -1)).T

    # NOTE: Is this dimensionally robust? How do we know what which output axis is
    # which? Seems it could be ordered by the dictionary key
    fn = eqx.filter_jit(vmap(lambda values: loglike(values, params, data_obj, model_class)))
    # return batched_fn(fn, vals_vec, n_batch=n_batch)

    batches = np.array_split(vals_vec, n_batch)
    out = []
    for batch in batches:
        out.append(fn(batch))
    out = np.concatenate(out)
    return out.reshape(vals.shape[1:])  # check the shapes output here


@eqx.filter_jit
def optimized_contrast_grid(data_obj, model_class, samples_dict, n_batch=50):
    """
    Function to optimize the contrast of a model over a grid of parameter values
    provided in a dictionary.

    Parameters
    ----------
    best_contrast_indices : array-like
        Indices of the best contrast values in a grid calculated with likelihood_grid.
    data_obj : OIData
        Object containing the data to be fitted.
    model_class : class
        Model class to be fitted to the data.
    samples_dict: dict
        Dictionary of parameter names and values to be fitted to the data.

    Returns
    -------
    array-like
        Optimized contrast values over the grid of parameter values.

    """

    params = list(samples_dict.keys())

    # first do a grid search to find a starting point
    # samples = samples_dict.values()
    # vals = np.array(np.meshgrid(*samples))
    # vals_vec = vals.reshape((len(vals), -1)).T

    # fn = vmap(lambda values: loglike(values, params, data_obj, model_class))
    # batches = np.array_split(vals_vec, n_batch)
    # out = []
    # for batch in batches:
    #     out.append(fn(batch))
    # out = np.concatenate(out)
    # loglike_im = out.reshape(vals.shape[1:])  # check the shapes output here

    loglike_im = likelihood_grid(data_obj, model_class, samples_dict, n_batch=n_batch)
    # loglike_im = fn(vals_vec).reshape(vals.shape[1:]) # check the shapes output here
    best_contrast_indices = np.argmax(loglike_im, axis=2)

    # then do optimization to fine tune the contrast
    coords = [samples_dict[key] for key in ["dra", "ddec"]]
    ras, decs = np.meshgrid(*coords)
    vals = np.array([samples_dict["flux"][best_contrast_indices].T, decs, ras])
    vals_vec = vals.reshape((len(vals), -1)).T

    to_optimize = lambda flux, dra_inp, ddec_inp: -loglike(
        [dra_inp, ddec_inp, flux], params, data_obj, model_class
    )
    bestcon = lambda flux, dra, ddec: optx.compat.minimize(
        to_optimize,
        x0=np.array([flux]),
        args=(np.array(dra), np.array(ddec)),
        method="BFGS",
        options={"maxiter": 100},
    ).x

    fn = eqx.filter_jit(vmap(lambda values: bestcon(*values)))
    # return batched_fn(fn, vals_vec, n_batch=n_batch).T  # check the shapes output here

    batches = np.array_split(vals_vec, n_batch)
    out = []
    for batch in batches:
        out.append(fn(batch))
    out = np.concatenate(out)
    return out.reshape(vals.shape[1:]).T  # check the shapes output here


from drpangloss.models import OIData


class AmigoOIData(OIData):
    Kphi: np.ndarray
    d_Kphi: np.ndarray
    Kphi_cov: np.ndarray
    vis_cov: np.ndarray
    phi_cov: np.ndarray
    Kphi_mat: np.ndarray
    amp_mat: np.ndarray
    phi_mat: np.ndarray
    parang: np.ndarray
    use_null_phase: bool = eqx.field(static=True)

    def __init__(self, oi_data, use_null_phase=False):
        """
        Default should have vis, phi as their _latent_ values
        """
        #
        self.u = np.array(oi_data["u"], dtype=float)
        self.v = np.array(oi_data["v"], dtype=float)
        self.wavel = np.array(oi_data["wavel"], dtype=float)
        self.parang = np.array(oi_data["parang"], dtype=float)

        #
        self.i_cps1 = None
        self.i_cps2 = None
        self.i_cps3 = None
        self.v2_flag = False
        self.cp_flag = False

        #
        self.vis = np.array(oi_data["vis"], dtype=float)
        self.phi = np.array(oi_data["phi"], dtype=float)
        self.Kphi = np.array(oi_data["Kphi"], dtype=float)

        #
        self.vis_cov = np.array(oi_data["vis_cov"], dtype=float)
        self.phi_cov = np.array(oi_data["phi_cov"], dtype=float)
        self.Kphi_cov = np.array(oi_data["Kphi_cov"], dtype=float)

        #
        self.d_vis = np.diag(self.vis_cov) ** 0.5
        self.d_phi = np.diag(self.phi_cov) ** 0.5
        self.d_Kphi = np.diag(self.Kphi_cov) ** 0.5

        # Operators and projection matrices
        # These projection matrices go from latent -> pixel, so we save the inverse
        self.amp_mat = np.linalg.pinv(np.array(oi_data["vis_mat"], dtype=float))
        self.phi_mat = np.linalg.pinv(np.array(oi_data["phi_mat"], dtype=float))

        #
        # The Kernel phase matrix goes from latent -> pixel, so we dont invert
        self.Kphi_mat = np.array(oi_data["Kphi_mat"], dtype=float)
        self.use_null_phase = bool(use_null_phase)

    def to_null(self, phi):
        """Project the phases to the null space"""
        return np.dot(self.Kphi_mat, phi)

    def to_latent(self, amps, phases):
        # Project our pixel parameters to latent amplitudes and phases
        return np.dot(amps, self.amp_mat), np.dot(phases, self.phi_mat)
        # return np.dot(self.amp_mat.T, amps), np.dot(self.phi_mat.T, phases)

    def flatten_data(self):
        """
        Flatten closure phases and uncertainties.
        """

        if self.use_null_phase:
            phi = self.Kphi
            d_phi = self.d_Kphi
        else:
            phi = self.phi
            d_phi = self.d_phi

        full_vis = np.concatenate([self.vis, phi])
        errs = np.concatenate([self.d_vis, d_phi])
        return full_vis, errs

    def flatten_model(self, cvis):
        """
        cvis: complex visibilities from model

        Flatten model visibilities and phases.
        """
        vis, phi = np.abs(cvis), np.angle(cvis)
        vis, phi = self.to_latent(vis, phi)
        if self.use_null_phase:
            phi = self.to_null(phi)
        return np.concatenate([vis, phi])

    # Overwrite this, it was a mistake from he beginning
    def __repr__(self):
        return eqx.Module.__repr__(self)


import jax.numpy as jnp


def calibrate_amplitudes(amp_1, amp_2, cov_1, cov_2):
    """Propagate covariance for element-wise amplitude ratio r = amp_1 / amp_2."""
    amp_1.shape[0]
    r = amp_1 / amp_2

    # Build Jacobians
    J1 = jnp.diag(1.0 / amp_2)
    J2 = jnp.diag(-amp_1 / amp_2**2)

    # Combine covariances
    cov_r = J1 @ cov_1 @ J1.T + J2 @ cov_2 @ J2.T
    return r, cov_r


def calibrate_phases(phase_1, phase_2, cov_1, cov_2):
    """Propagate covariance for element-wise phase difference Δφ = phase_1 - phase_2."""
    n = phase_1.shape[0]

    J1 = jnp.eye(n)
    J2 = -jnp.eye(n)

    cov_dphi = J1 @ cov_1 @ J1.T + J2 @ cov_2 @ J2.T
    dphi = phase_1 - phase_2
    return dphi, cov_dphi


def fit_oi_data(
    grid,
    oi_data,
    amp_scale=1.0,
    phi_scale=1.0,
    Kphi_scale=1.0,
    n_batch=50,
    use_null=False,
    obj=AmigoOIData,
):
    # Scale the errors
    oi_obj = obj(oi_data, use_null_phase=use_null)
    oi_obj = oi_obj.multiply("d_vis", amp_scale)
    oi_obj = oi_obj.multiply("d_phi", phi_scale)
    oi_obj = oi_obj.multiply("d_Kphi", Kphi_scale)

    # NOTE: The log-likelihood image seems to come out (Dec, RA), so we _dont_ need to
    # transpose the output
    # Log-likelihood
    loglike_im = likelihood_grid(oi_obj, BinaryModelCartesian, grid, n_batch=n_batch)
    loglike_im = loglike_im.max(axis=2)

    # NOTE: The Contrast image seems to come out (RA, Dec), so we transpose the output
    # Best contrast
    contrast_im = optimized_contrast_grid(oi_obj, BinaryModelCartesian, grid, n_batch=n_batch)
    contrast_im = np.where(contrast_im < 0, 0, contrast_im).T

    # # Get the FoVs
    r_bls = np.hypot(oi_data["u"], oi_data["v"])
    min_bls, max_bls = r_bls.min(), r_bls.max()
    min_fov = 1e3 * dlu.rad2arcsec(oi_data["wavel"] / (2 * max_bls))
    max_fov = 1e3 * dlu.rad2arcsec(oi_data["wavel"] / (2 * min_bls))

    # Mask the fov
    rs = np.hypot(*np.meshgrid(grid["dra"], grid["ddec"]))
    r_max = np.minimum(max_fov, 2 * grid["dra"].max())
    # rmask = (rs > min_fov / 2) & (rs <= r_max / 2)
    rmask = (rs > 200 / 2) & (rs <= r_max / 2)
    loglike_im = np.where(rmask, loglike_im, np.nan)
    contrast_im = np.where(rmask, contrast_im, np.nan)

    # Return the images
    return loglike_im, contrast_im, (min_fov, max_fov)
