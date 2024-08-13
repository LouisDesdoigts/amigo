import os
import jax.numpy as np
import jax.scipy as jsp
from astroquery.simbad import Simbad
import pyia
import pkg_resources as pkg
import numpy as onp
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


def get_simbad_spectral_type(source_id):
    """Returns the spectral type and quality from Simbad for a given source ID."""
    Simbad.add_votable_fields("sptype", "sp_qual")
    table = Simbad.query_object(source_id)
    if table is None:
        return None, None
    return table["SP_TYPE"].tolist()[0], table["SP_QUAL"].tolist()[0]


def get_gaia_Teff(source_id, data_dr="dr3"):
    """Returns the Teff from Gaia for a given source ID."""
    result_table = Simbad.query_objectids(source_id)
    if result_table is None:
        return []
    ids = []
    for x in result_table:
        if f"gaia {data_dr}" in x["ID"].lower():
            ids.append(x["ID"].split(" ")[-1])

    # All the potential keys with a Teff
    teff_keys = [
        "teff_gspphot",
        "teff_gspspec",
        "teff_msc1",
        "teff_msc2",
        "teff_esphs",
        "teff_espucd",
        "teff_val",
    ]

    Teffs_out = []
    for obj_id in ids:
        data = pyia.GaiaData.from_source_id(obj_id, source_id_dr="dr3", data_dr=data_dr)
        for teff_type in teff_keys:
            if hasattr(data, teff_type):
                val = np.squeeze(np.array(getattr(data, teff_type).value))
                if not np.isnan(val):
                    Teffs_out.append(val)

    return Teffs_out


def get_Teff(targ_name):
    """Returns the Teff for a given target name."""
    # First check DR3
    dr3_teffs = get_gaia_Teff(targ_name, data_dr="dr3")
    if len(dr3_teffs) == 1:
        return dr3_teffs[0]
    elif len(dr3_teffs) > 1:
        print(f"Multiple Teffs for {targ_name} in DR3, returning mean")
        return np.array(dr3_teffs).mean()

    # Then check Simbad -> Mamajeck?? table
    # Skip this for now till I have time to make it work
    if False:
        spec_type, qual = get_simbad_spectral_type(targ_name)
        if spec_type is not None:
            return pyia.spectral_type_to_Teff(spec_type)
        # TODO: Use `MeanStars` to get Teff from spectral type

    # Finally, check DR2
    dr2_teffs = get_gaia_Teff(targ_name, data_dr="dr2")
    if len(dr2_teffs) == 1:
        return dr2_teffs[0]
    elif len(dr2_teffs) > 1:
        print(f"Multiple Teffs for {targ_name} in DR2, returning mean")
        return np.array(dr2_teffs).mean()

    # Return -1 as a flag for 'not found'
    return -1


def get_Teffs(files, default=4500, skip_search=False, Teff_cache="files/Teffs"):
    # def get_Teffs(exposures, default=4500, skip_search=False, Teff_cache="files/Teffs"):
    # Check whether the specified cache directory exists
    if not os.path.exists(Teff_cache):
        os.makedirs(Teff_cache)

    Teffs = {}
    for file in files:
        prop_name = file[0].header["TARGPROP"]

        # if os.exists(f"{Teff_cache}/{prop_name}.npy"):
        try:
            Teffs[prop_name] = np.load(f"{Teff_cache}/{prop_name}.npy")
            continue
        except FileNotFoundError:
            pass

        if prop_name in Teffs:
            continue

        # Temporary measure to get around gaia archive being dead
        if skip_search:
            Teffs[prop_name] = default
            print("Warning using default Teff")
            continue

        Teff = get_Teff(file[0].header["TARGNAME"])

        if Teff == -1:
            print(f"No Teff found for {prop_name}, defaulting to 4500K")
            Teffs[prop_name] = default
        else:
            Teffs[prop_name] = Teff
            np.save(f"{Teff_cache}/{prop_name}.npy", Teff)

    return Teffs


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
    fit_one_on_fs=False,
    fit_reflectivity=False,
    vis_model=None,
):
    # NOTE: At present this assume the _same_ fit is being applied to all the exposures

    params = get_default_params(exposures, optics)

    if not fit_one_on_fs:
        params.pop("one_on_fs")

    if not fit_reflectivity:
        params.pop("reflectivity")

    if vis_model is not None:
        params.update(initialise_vis(vis_model, exposures))

    return params


def get_filters(files, nwavels=9):
    filters = list(set([file[0].header["FILTER"] for file in files]))
    filter_dict = {}
    for filt in filters:
        filter_dict[filt] = calc_throughput(filt, nwavels=nwavels)
    return filter_dict


def calc_throughput(filt, nwavels=9):

    if filt not in ["F380M", "F430M", "F480M", "F277W"]:
        raise ValueError("Supported filters are F380M, F430M, F480M, F277W.")

    # filter_path = os.path.join()
    file_path = pkg.resource_filename(__name__, f"/data/filters/{filt}.dat")
    wl_array, throughput_array = np.array(onp.loadtxt(file_path, unpack=True))

    edges = np.linspace(wl_array.min(), wl_array.max(), nwavels + 1)
    wavels = np.linspace(wl_array.min(), wl_array.max(), 2 * nwavels + 1)[1::2]

    areas = []
    for i in range(nwavels):
        cond1 = edges[i] < wl_array
        cond2 = wl_array < edges[i + 1]
        throughput = np.where(cond1 & cond2, throughput_array, 0)
        areas.append(jsp.integrate.trapezoid(y=throughput, x=wl_array))

    areas = np.array(areas)
    weights = areas / areas.sum()

    wavels *= 1e-10
    return np.array([wavels, weights])


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
