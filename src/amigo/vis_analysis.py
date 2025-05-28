import jax
import jax.numpy as np
import jax.random as jr
import equinox as eqx
import zodiax as zdx
import jax.scipy as jsp
import optimistix as optx
from jax import jit, vmap
import dLux.utils as dlu
from amigo.misc import interp
from drpangloss.grid_fit import azimuthalAverage
from numpyro.distributions.util import gammaincinv
from amigo.stats import orthogonalise
from copy import deepcopy


class AmigoOIData(zdx.Base):
    # Coordinates
    u: np.ndarray
    v: np.ndarray
    wavel: np.ndarray
    parang: np.ndarray

    # Latent Kernel Visibilities
    vis: np.ndarray
    d_vis: np.ndarray

    # Latent Kernel Phases
    phi: np.ndarray
    d_phi: np.ndarray

    # Latent space projection matrices
    vis_mat: np.ndarray  # Ortho -> Pixel
    phi_mat: np.ndarray  # Ortho -> Pixel

    def __init__(self, oi_data):
        """
        Default should have vis, phi as their _latent_ values
        """
        self.u = -np.array(oi_data["u"], dtype=float)
        self.v = -np.array(oi_data["v"], dtype=float)
        self.wavel = np.array(oi_data["wavel"], dtype=float)
        self.parang = np.array(oi_data["parang"], dtype=float)

        # Visibilities
        self.vis = np.array(oi_data["O_vis"], dtype=float)
        self.phi = np.array(oi_data["O_phi"], dtype=float)

        # Uncertainties
        self.d_vis = np.diag(oi_data["O_vis_cov"]) ** 0.5
        self.d_phi = np.diag(oi_data["O_phi_cov"]) ** 0.5

        pix_to_latent = np.linalg.pinv(np.array(oi_data["vis_mat"], dtype=float))
        latent_to_kernel = np.array(oi_data["K_vis_mat"], dtype=float)
        kernel_to_ortho = np.array(oi_data["O_vis_mat"], dtype=float)
        proj_mat = np.dot(pix_to_latent, latent_to_kernel.T)
        proj_mat = np.dot(kernel_to_ortho, proj_mat.T)
        self.vis_mat = proj_mat.T

        pix_to_latent = np.linalg.pinv(np.array(oi_data["phi_mat"], dtype=float))
        latent_to_kernel = np.array(oi_data["K_phi_mat"], dtype=float)
        kernel_to_ortho = np.array(oi_data["O_phi_mat"], dtype=float)
        proj_mat = np.dot(pix_to_latent, latent_to_kernel.T)
        proj_mat = np.dot(kernel_to_ortho, proj_mat.T)
        self.phi_mat = proj_mat.T

    def flatten_data(self):
        """Flatten closure phases and uncertainties."""
        vis_vec = np.concatenate([self.vis, self.phi])
        vis_err = np.concatenate([self.d_vis, self.d_phi])
        return vis_vec, vis_err

    def flatten_model(self, cvis):
        """
        cvis: complex visibilities from model
        Flatten model visibilities and phases.
        """
        # Project the visibilities to the latent
        log_cvis = np.log(cvis)
        log_vis, phi = log_cvis.real, log_cvis.imag

        vis = np.dot(log_vis, self.vis_mat)
        phi = np.dot(phi, self.phi_mat)
        return np.concatenate([vis, phi])

    def model(self, model_object):
        """
        Compute the model visibilities and phases for the given model object.
        """
        cvis = model_object.model(self.u, self.v, self.wavel)
        return self.flatten_model(cvis)

    # Overwrite this, it was a mistake from the beginning
    def __repr__(self):
        return eqx.Module.__repr__(self)


rad2mas = 180.0 / np.pi * 3600.0 * 1000.0  # convert rad to mas
mas2rad = np.pi / 180.0 / 3600.0 / 1000.0  # convert mas to rad

dtor = np.pi / 180.0
i2pi = 1j * 2.0 * np.pi


def calc_phi(u, v, ddec, dra):
    # relative locations
    ddec = ddec * np.pi / (180.0 * 3600.0 * 1000.0)
    dra = dra * np.pi / (180.0 * 3600.0 * 1000.0)
    phi_r = np.cos(-2 * np.pi * (u * dra + v * ddec))
    phi_i = np.sin(-2 * np.pi * (u * dra + v * ddec))
    return phi_r + 1j * phi_i


def calc_fluxes(c):
    """
    Compute normalized fluxes [f0, f1, ..., fn] from contrast vector c = [c1, ..., cn],
    where each ci = fi / f0 for i > 0 and sum(fi) = 1.

    Args:
        c: Array of shape (n,), contrast values relative to source 0.

    Returns:
        f: Array of shape (n+1,), normalized fluxes [f0, f1, ..., fn].
    """
    c = np.atleast_1d(c)  # Ensure c is at least 1D
    denom = 1.0 + np.sum(c)
    f0 = 1.0 / denom
    f_rest = c * f0
    return np.concatenate([np.array([f0]), f_rest])


def cvis_multiple(u, v, ddecs, dras, contrasts):
    """
    This assume that all the companions are _low_ contrast
    """

    # Get the complex signals
    positions = [(0.0, 0.0)]  # primary star
    positions += [pos for pos in zip(ddecs, dras)]  # companions
    cphis = [calc_phi(u, v, ddec, dra) for ddec, dra in positions]

    # Get the fluxes
    fluxes = calc_fluxes(contrasts)

    # Get the complex visibilities
    civs = [lum * cphi for lum, cphi in zip(fluxes, cphis)]
    return np.array(civs).sum(axis=0)


def cvis_binary(u, v, ddec, dra, contrast):
    # adapted from pymask
    """Calculate the complex visibilities observed by an array on a binary star
    ----------------------------------------------------------------
    - ddec = ddec (mas)
    - dra = dra (mas)
    - planet = planet brightness
    - u,v: baseline coordinates (wavelengths)
    ----------------------------------------------------------------"""
    return cvis_multiple(u, v, [ddec], [dra], contrast)


class BinaryModelCartesian(zdx.Base):
    dra: jax.Array
    ddec: jax.Array
    flux: jax.Array
    log: bool

    def __init__(self, dra, ddec, flux, log=False):
        self.dra = np.asarray(dra, dtype=float)
        self.ddec = np.asarray(ddec, dtype=float)
        self.flux = np.asarray(flux, dtype=float)
        self.log = bool(log)

    def model(self, u, v, wavel):
        """
        Model for binary star system.
        """
        uu, vv = u / wavel, v / wavel
        flux = 10**self.flux if self.log else self.flux
        return cvis_binary(uu, vv, self.ddec, self.dra, flux)

    def __call__(self, oi_obj):
        return self.model(oi_obj.u, oi_obj.v, oi_obj.wavel)


class MultiModelCartesian(zdx.Base):
    dras: jax.Array
    ddecs: jax.Array
    fluxes: jax.Array
    log: bool

    def __init__(self, dras, ddecs, fluxes, log=False):
        self.dras = np.atleast_1d(np.asarray(dras, dtype=float))
        self.ddecs = np.atleast_1d(np.asarray(ddecs, dtype=float))
        self.fluxes = np.atleast_1d(np.asarray(fluxes, dtype=float))
        self.log = bool(log)

    def model(self, u, v, wavel):
        """
        Model for binary star system.
        """
        uu, vv = u / wavel, v / wavel
        fluxes = 10**self.fluxes if self.log else self.fluxes
        return cvis_multiple(uu, vv, self.ddecs, self.dras, fluxes)

    def __call__(self, oi_obj):
        return self.model(oi_obj.u, oi_obj.v, oi_obj.wavel)


def model_loglike(oi_model, oi_obj):
    civs = oi_model(oi_obj)
    model_data = oi_obj.flatten_model(civs)
    data, errors = oi_obj.flatten_data()
    return jsp.stats.norm.logpdf(model_data, loc=data, scale=errors).sum()


def chi2_loglike(oi_model, oi_obj):
    civs = oi_model(oi_obj)
    model_data = oi_obj.flatten_model(civs)
    data, errors = oi_obj.flatten_data()
    res2 = (model_data - data) ** 2
    var = errors**2
    return -0.5 * np.sum(res2 / var)


def batched_grid(fn, vals_grid, n_batch=1):
    out_shape = vals_grid.shape[1:]
    vals_vec = vals_grid.reshape((len(vals_grid), -1)).T
    batches = np.array_split(vals_vec, n_batch)
    out = []
    for batch in batches:
        out.append(fn(batch))

    if isinstance(out[0], jax.Array):
        return np.concatenate(out).reshape(out_shape)
    else:
        n_outputs = len(out[0])
        final_out = []
        for i in range(n_outputs):
            tmp_out = []
            for outs in out:
                tmp_out.append(outs[i])
            final_out.append(np.concatenate(tmp_out).reshape(out_shape))
        return final_out


def ruffio_upperlimit(mean, sigma, percentile, epsilon=1e-12):
    # eqn 8 from Ruffio+2018
    cdf = jsp.stats.norm.cdf(0, loc=mean, scale=sigma)
    x = np.clip(percentile + (1 - percentile) * cdf, 0, 1 - epsilon)
    limit = jsp.stats.norm.ppf(x, loc=mean, scale=sigma)
    return limit


def chi2ppf(p, df):
    return gammaincinv(df / 2.0, p) * 2


def nsigma(chi2r_test, chi2r_true, ndof):
    q = jax.scipy.stats.chi2.cdf(ndof * chi2r_test / chi2r_true, ndof)
    p = 1.0 - q
    nsigma = np.sqrt(chi2ppf(p, 1.0))
    return nsigma


def solve(fn, solver, x0, args, max_steps=512):
    sol = optx.minimise(fn, solver, x0, args, throw=False, max_steps=max_steps)
    return sol.state.f_info.f, sol.value


def analyse_vis(
    oi_obj,
    size=500,  # mas
    n_pts=100,  # number of points in the final grid
    n_grid=10,  # number of initial grid points
    n_batch=1,  # number of batches to calculate the likelihood
    min_flux=-6,  # minimum log flux range
    n_sigma=3.0,  # number of sigma for the upper limits
    max_steps=512,  # maximum number of steps for the optimizer
    tol=1e-8,  # tolerance for the optimizer
    log=False,  # whether to use log flux (needs testing)
):

    # Build wrapper for the model fit and data object
    binary = BinaryModelCartesian(0.0, 0.0, 0.0, log=log)
    # binary = MultiModelCartesian(0.0, 0.0, 0.0, log=log)
    fn_wrapper = lambda fn, values: fn(binary.set(["dra", "ddec", "flux"], list(values)), oi_obj)
    loglike_fn = lambda values: fn_wrapper(model_loglike, values)
    chi2_fn = lambda values: fn_wrapper(chi2_loglike, values)

    # Samples
    min_flux = min_flux if log else 10**min_flux
    RAs = np.linspace(size, -size, n_grid)
    Decs = np.linspace(-size, size, n_grid)
    flux = np.linspace(min_flux, 0, n_grid)
    param_grid = np.array(np.meshgrid(RAs, Decs, flux, indexing="xy"))

    # Get the log-likelihood grid
    loglike_grid = batched_grid(jit(vmap(loglike_fn)), param_grid, n_batch=n_batch)
    loglike_mle = loglike_grid.max(axis=2)
    contrast_mle = flux[np.argmax(loglike_grid, axis=2)]
    loglike_im = loglike_mle

    coords = np.array(np.meshgrid(RAs, Decs))
    mle_param_grid = np.array([*coords, contrast_mle])

    # Contrast fit function
    solver = optx.BFGS(rtol=tol, atol=tol)
    opt_fn = lambda flux, coords: -loglike_fn([*coords, flux])
    fit_fn = lambda values: solve(
        opt_fn, solver, values[2], (values[0], values[1]), max_steps=max_steps
    )
    loss, contrast_im = batched_grid(jit(vmap(fit_fn)), mle_param_grid, n_batch=n_batch)
    loglike_im = -loss

    # Sigma uncertainties
    params_grid = np.array([*coords, contrast_im])
    sigma_fn = lambda values: np.sqrt(1 / jax.hessian(opt_fn)(values[2], values[:2]))
    sigma_im = batched_grid(jit(vmap(sigma_fn)), params_grid, n_batch=n_batch)

    # Ruffio upper limits
    perc = jsp.stats.norm.cdf(n_sigma)
    contrast_clipped = np.clip(contrast_im, 0, min_flux)  # clip to min_flux
    ruffio_im = ruffio_upperlimit(contrast_clipped, sigma_im, perc)

    # Radial Ruffio upper limits
    avg_fn = lambda ruffio, **kwargs: azimuthalAverage(
        -2.5 * np.log10(ruffio), returnradii=True, binsize=2, **kwargs
    )
    rad_width_ruffio, avg_width_ruffio = avg_fn(ruffio_im, stddev=False)
    _, std_width_ruffio = avg_fn(ruffio_im, stddev=True)
    ruffio = [rad_width_ruffio, avg_width_ruffio, std_width_ruffio]

    # Absil upper limits
    # TODO: This should actually be ndof - num params (3 in this case, but not always)
    ndof = oi_obj.vis.size + oi_obj.phi.size  # number of degrees of freedom
    # chi2_bin = lambda values: -2 * zscore_fn(values) / ndof
    chi2_bin = lambda values: -2 * chi2_fn(values) / ndof
    chi2_null = chi2_bin([0.0, 0.0, 0.0])

    # Absil upper limits fit
    solver = optx.BFGS(rtol=tol, atol=tol)
    loss = lambda values: (nsigma(chi2_bin(values) / ndof, chi2_null / ndof, ndof) - n_sigma) ** 2
    opt_fn = lambda flux, coords: loss([*coords, flux])
    fit_fn = lambda values: solve(
        opt_fn, solver, values[2], (values[0], values[1]), max_steps=max_steps
    )
    loss, limits_im = batched_grid(jit(vmap(fit_fn)), mle_param_grid, n_batch=n_batch)

    # Upsample the log-likelihood and contrast images via interpolation
    knots = dlu.pixel_coords(n_grid, 2 * size)
    sample_pts = dlu.pixel_coords(n_pts, 2 * size)
    loglike_im = interp(loglike_im, knots, sample_pts, method="cubic", fill=np.nan)
    contrast_im = interp(contrast_im, knots, sample_pts, method="cubic", fill=np.nan)
    sigma_im = interp(sigma_im, knots, sample_pts, method="cubic", fill=np.nan)
    ruffio_im = interp(ruffio_im, knots, sample_pts, method="cubic", fill=np.nan)
    limits_im = interp(limits_im, knots, sample_pts, method="cubic", fill=np.nan)

    # # Clip the limits
    # limits_im = np.where(limits_im < 0, 1e-6, limits_im)  # clip to 0

    # Get the FoVs
    r_bls = np.hypot(oi_obj.u, oi_obj.v)
    min_bls, max_bls = r_bls.min(), r_bls.max()
    min_fov = 1e3 * dlu.rad2arcsec(oi_obj.wavel / (2 * max_bls))
    # max_fov = 1e3 * dlu.rad2arcsec(oi_obj.wavel / (2 * min_bls))
    max_fov = 1e3 * dlu.rad2arcsec(oi_obj.wavel / min_bls)
    # min_fov = 1e3 * dlu.rad2arcsec(oi_obj.wavel / (2*optics.diameter))

    # Mask the fov
    rs = np.hypot(*sample_pts)
    r_max = np.minimum(max_fov, size)
    rmask = (rs < min_fov) | (rs >= r_max)

    return {
        "oi_obj": oi_obj,
        "fov": (min_fov, max_fov),
        "extent": (RAs[0], RAs[-1], Decs[0], Decs[-1]),
        "loglike_im": np.where(rmask, np.nan, loglike_im),
        "contrast_im": np.where(rmask, np.nan, contrast_im),
        "sigma_im": np.where(rmask, np.nan, sigma_im),
        "ruffio_im": np.where(rmask, np.nan, ruffio_im),
        "limits_im": np.where(rmask, np.nan, limits_im),
        "ruffio": ruffio,
        "n_sigma": n_sigma,
    }


def generate_photon_data(vis_eig_vals, n_terms, n_phot, cal_vis_dict, key=jr.key(0)):
    # Get the eigenvalues of the vis basis
    amp_vals = vis_eig_vals["amplitude"][:n_terms]
    phase_vals = vis_eig_vals["phase"][:n_terms]

    # Convert the eigenvalues to variances by scaling by the number of photons
    # n_phot = data_photons[filt]
    amp_var = 1 / (n_phot * amp_vals)
    phase_var = 1 / (n_phot * phase_vals)

    # Double the variances to account for the error propagation through calibration
    amp_var = 2 * amp_var
    phase_var = 2 * phase_var

    # Convert the variances to a diagonal covariance matrix
    amp_cov = np.eye(n_terms) * amp_var[:, None]
    phase_cov = np.eye(n_terms) * phase_var[:, None]

    # Get the standard deviations
    amp_std = np.sqrt(np.diag(amp_cov))
    phase_std = np.sqrt(np.diag(phase_cov))

    # Generate random samples
    key1, key2 = jr.split(key)
    vis = amp_std * jr.normal(key1, amp_std.shape)
    phi = phase_std * jr.normal(key2, phase_std.shape)

    # Get the kernel matrices
    K_vis_mat = cal_vis_dict["K_vis_mat"]
    K_phi_mat = cal_vis_dict["K_phi_mat"]

    # Project values to the kernel space
    K_vis = np.dot(K_vis_mat, vis)
    K_phi = np.dot(K_phi_mat, phi)

    # Project the covariance matrices to the kernel space
    K_vis_cov = np.dot(K_vis_mat, np.dot(amp_cov, np.linalg.pinv(K_vis_mat)))
    K_phi_cov = np.dot(K_phi_mat, np.dot(phase_cov, np.linalg.pinv(K_phi_mat)))

    # Orthonormalise the visibilities
    o_vis, o_vis_cov, o_vis_mat = orthogonalise(K_vis, K_vis_cov, normalise=False)
    o_phi, o_phi_cov, o_phi_mat = orthogonalise(K_phi, K_phi_cov, normalise=False)

    phot_cal_vis_dict = deepcopy(cal_vis_dict)

    phot_cal_vis_dict["O_vis"] = o_vis
    phot_cal_vis_dict["O_phi"] = o_phi
    phot_cal_vis_dict["O_vis_cov"] = o_vis_cov
    phot_cal_vis_dict["O_phi_cov"] = o_phi_cov
    phot_cal_vis_dict["O_vis_mat"] = o_vis_mat
    phot_cal_vis_dict["O_phi_mat"] = o_phi_mat

    return phot_cal_vis_dict
