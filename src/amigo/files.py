# import jax.numpy as np
# import dLux.utils as dlu
# from .misc import find_position
from .search_Teffs import get_Teffs


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


# def get_default_params(exposures, optics, amp_order=1):

#     # These are the default parameters, they are _always_ present
#     positions = {}
#     fluxes = {}
#     aberrations = {}
#     one_on_fs = {}
#     reflectivity = {}
#     biases = {}
#     separations = {}
#     contrasts = {}
#     position_angles = {}
#     for exp in exposures:

# im = np.where(exp.badpix, np.nan, exp.slopes[0])
# psf = np.where(np.isnan(im), 0.0, im)

# # Get pixel scale in arcseconds
# if hasattr(optics, "focal_length"):
#     pixel_scale = dlu.rad2arcsec(1e-6 * optics.psf_pixel_scale / optics.focal_length)
# else:
#     pixel_scale = optics.psf_pixel_scale
# position = find_position(psf, pixel_scale)
# positions[exp.fit.get_key(exp, "positions")] = position

# flux = np.log10(1.05 * exp.ngroups * np.nansum(exp.slopes[0]))
# fluxes[exp.fit.get_key(exp, "fluxes")] = flux

# abers = np.zeros_like(optics.pupil_mask.abb_coeffs)
# aberrations[exp.fit.get_key(exp, "aberrations")] = abers

# if optics.pupil_mask.amp_coeffs is not None:
#     reflects = np.zeros_like(optics.pupil_mask.amp_coeffs)
#     reflectivity[exp.fit.get_key(exp, "reflectivity")] = reflects

# one_on_f = np.zeros((exp.ngroups, 80, amp_order + 1))
# one_on_fs[exp.fit.get_key(exp, "one_on_fs")] = one_on_f
# biases[exp.fit.get_key(exp, "biases")] = np.zeros((80, 80))

# separations[exp.fit.get_key(exp, "separations")] = 0.15  # arcsec, ~2 pixels
# contrasts[exp.fit.get_key(exp, "contrasts")] = 2.0  # 100x contrast
# position_angles[exp.fit.get_key(exp, "position_angles")] = 0.0  # degrees

# return {
#     "positions": positions,
#     "fluxes": fluxes,
#     "aberrations": aberrations,
#     "reflectivity": reflectivity,
#     "one_on_fs": one_on_fs,
#     "biases": biases,
#     "separations": separations,
#     "contrasts": contrasts,
#     "position_angles": position_angles,
# }


# def initialise_vis(vis_model, exposures):
#     """At present this assumes that we are fitting a spline visibility"""

#     params = {
#         "amplitudes": {},
#         "phases": {},
#     }
#     # n = vis_model.knot_map.size // 2
#     n = vis_model.knot_inds.size
#     for exp in exposures:
#         params["amplitudes"][f"{exp.get_key('amplitudes')}"] = np.ones(n)
#         params["phases"][f"{exp.get_key('phases')}"] = np.zeros(n)
#     return params


def initialise_params(
    files,
    exposures,
    optics,
    vis_model=None,
    Teff_cache="../data/Teffs",
):
    # NOTE: This method should be improved to take the _average_ over params that are
    # constrained by multiple exposures

    params = {}
    for exp in exposures:
        if vis_model is not None:
            param_dict = exp.initialise_params(optics, vis_model=vis_model)
        else:
            param_dict = exp.initialise_params(optics)
        for param, (key, value) in param_dict.items():
            if param not in params.keys():
                params[param] = {}
            params[param][key] = value

    params["Teffs"] = get_Teffs(files, Teff_cache=Teff_cache)

    return params


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
