import jax.numpy as np
import pkg_resources as pkg
import dLux.utils as dlu
from .misc import find_position


def summarise_files(files, extra_keys=[]):
    main_keys = []

    main_keys += extra_keys
    for key in main_keys:
        values = set([f[0].header[key] for f in files])
        vals_str = ", ".join([f"{val}" for val in values])
        print(f"  {key}: {vals_str}")


def get_files(paths, ext, **kwargs):
    """

    data_path: Path to the data files
    ext: File extension to search for
    """
    import os
    from astropy.io import fits

    if isinstance(paths, str):
        paths = [paths]

    files = []
    for path in paths:
        file_names = os.listdir(path)

        checked = False
        for name in file_names:
            if name.endswith(f"{ext}.fits"):
                file = fits.open(path + name)
                h = file[0].header
                if not checked:
                    if not all([key in h.keys() for key in kwargs.keys()]):
                        raise KeyError(f"Header keys {kwargs.keys()} not found in file {name}")

                match = True
                for key, val in kwargs.items():
                    if isinstance(val, list):
                        if h[key] not in val:
                            match = False
                    elif h[key] != val:
                        match = False

                if match:
                    files.append(file)
    return files


def get_default_params(exposures, optics, amp_order=1):

    # These are the default parameters, they are _always_ present
    positions = {}
    fluxes = {}
    aberrations = {}
    one_on_fs = {}
    one_on_fs = {}
    reflectivity = {}
    for exp in exposures:

        im = exp.slopes[0]
        psf = np.where(np.isnan(im), 0.0, im)

        # Get pixel scale in arcseconds
        if hasattr(optics, "focal_length"):
            pixel_scale = dlu.rad2arcsec(1e-6 * optics.psf_pixel_scale / optics.focal_length)
        else:
            pixel_scale = optics.psf_pixel_scale
        position = find_position(psf, pixel_scale)
        positions[exp.fit.get_key(exp, "positions")] = position

        flux = np.log10(1.05 * exp.ngroups * np.nansum(exp.slopes[0]))
        fluxes[exp.fit.get_key(exp, "fluxes")] = flux

        abers = np.zeros_like(optics.pupil_mask.abb_coeffs)
        aberrations[exp.fit.get_key(exp, "aberrations")] = abers

        if optics.pupil_mask.amp_coeffs is not None:
            reflects = np.zeros_like(optics.pupil_mask.amp_coeffs)
            reflectivity[exp.fit.get_key(exp, "reflectivity")] = reflects

        one_on_f = np.zeros((exp.ngroups, 80, amp_order + 1))
        one_on_fs[exp.fit.get_key(exp, "one_on_fs")] = one_on_f

    return {
        "positions": positions,
        "fluxes": fluxes,
        "aberrations": aberrations,
        "reflectivity": reflectivity,
        "one_on_fs": one_on_fs,
    }


def initialise_vis(vis_model, exposures):
    """At present this assumes that we are fitting a spline visibility"""

    params = {
        "amplitudes": {},
        "phases": {},
    }
    n = vis_model.knots[0].size // 2
    for exp in exposures:
        params["amplitudes"][f"{exp.get_key('amplitudes')}"] = np.ones(n)
        params["phases"][f"{exp.get_key('phases')}"] = np.zeros(n)
    return params


def initialise_params(
    exposures,
    optics,
    fit_one_on_fs=True,
    fit_reflectivity=False,
    vis_model=None,
):
    """Assumes all exposures have the same fit"""
    params = get_default_params(exposures, optics)

    if not fit_one_on_fs:
        params.pop("one_on_fs")

    if not fit_reflectivity:
        params.pop("reflectivity")

    if vis_model is not None:
        # if isinstance(exposures[0].fit, SplineVisFit):
        params.update(initialise_vis(vis_model, exposures))

    return params


def prep_data(file, ms_thresh=None, as_psf=False):
    data = np.asarray(file["SCI"].data, float)
    var = np.asarray(file["SCI_VAR"].data, float)
    dq = np.asarray(file["PIXELDQ"].data > 0, bool)

    if ms_thresh is not None:
        dq = dq.at[np.mean(data, axis=0) <= ms_thresh].set(True)

    badpix = np.load(pkg.resource_filename(__name__, "data/badpix.npy"))
    dq = dq | badpix

    if as_psf:
        supp_mask = ~np.isnan(data) & ~dq
        support = np.where(supp_mask)
        data = data.at[~supp_mask].set(np.nan)
        var = var.at[~supp_mask].set(np.nan)
        return data, var, support

    supp_mask = ~np.isnan(data.sum(0)) & ~dq

    # Nan the bad pixels
    support = np.array(np.where(supp_mask))
    data = data.at[:, ~supp_mask].set(np.nan)
    var = var.at[..., ~supp_mask].set(np.nan)

    return data, var, support


def get_exposures(files, fit, ms_thresh=None, as_psf=False):
    from amigo.core_models import Exposure

    exposures = []
    for file in files:
        data, variance, support = prep_data(file, ms_thresh=ms_thresh, as_psf=as_psf)
        data = np.asarray(data, float)
        variance = np.asarray(variance, float)
        support = np.asarray(support, int)
        exposures.append(Exposure(file, data, variance, support, fit))  # key_fn, fit))
    return exposures


def repopulate(model, history, index=-1):

    # Populate parameters
    for param_key, value in history.params.items():
        if isinstance(value, dict):
            for exp_key, sub_value in value.items():
                try:
                    model = model.set(f"{param_key}.{exp_key}", sub_value[index])

                # Catch key not existing since we might not have a certain exposure here
                except KeyError:
                    pass
        else:
            model = model.set(param_key, value[index])

    return model
