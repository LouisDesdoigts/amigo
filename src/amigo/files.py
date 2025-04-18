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

    # Remove the flat field files for the Teff search
    star_files = []
    for file in files:
        if file["PRIMARY"].header["EXP_TYPE"] == "NIS_AMI":
            star_files.append(file)
    # params["Teffs"] = get_Teffs(star_files, Teff_cache=Teff_cache)
    params["Teffs"] = get_Teffs(star_files, Teff_cache=Teff_cache)

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
