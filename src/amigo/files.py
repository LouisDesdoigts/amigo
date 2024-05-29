import os
import jax.numpy as np
import jax.scipy as jsp
from astroquery.simbad import Simbad
import pyia
import pkg_resources as pkg
import numpy as onp
from jax import vmap
from .stats import check_symmetric, check_positive_semi_definite, build_cov
from .misc import convert_adjacent_to_true, fit_slope, slope_im
from .interferometry import uv_hex_mask
from webbpsf import mast_wss
from xara.core import determine_origin
from tqdm.notebook import tqdm
import amigo


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


def get_filters(files, nwavels=9):

    filters = {}
    for file in files:
        filt = file[0].header["FILTER"]
        if filt in filters.keys():
            continue

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
        filters[filt] = np.array([wavels, weights])
    return filters


def estimate_psf_and_bias(data):
    ngroups = len(data)
    ramp_bottom = data[:2]
    ramp_bottom = np.where(np.isnan(ramp_bottom), 0, ramp_bottom)
    psf, bias = slope_im(ramp_bottom)  # Estimate from the bottom of the ramp
    return psf * ngroups, bias


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
    support = np.where(supp_mask)
    data = data.at[:, ~supp_mask].set(np.nan)
    var = var.at[..., ~supp_mask].set(np.nan)

    return data, var, support


def get_wss_ops(files):
    opds = {}
    opd_files = []
    for file in files:
        date = file[0].header["DATE-BEG"]
        opd0, opd1, t0, t1 = mast_wss.mast_wss_opds_around_date_query(date, verbose=False)
        closest_fn, closest_dt = (opd1, t1) if abs(t1) < abs(t0) else (opd0, t0)
        opd_file = mast_wss.mast_retrieve_opd(closest_fn)
        if opd_file not in opds.keys():
            opd = mast_wss.import_wss_opd(opd_file)[0].data
            opds[opd_file] = np.asarray(opd, float)
        opd_files.append(opds[opd_file])
    return opd_files


def get_Teffs(files, default=4500, straight_default=False, Teff_cache="files/Teffs"):
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
        if straight_default:
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


def get_amplitudes(exposures, dc=False):
    """If dc is True, an extra value for the dc term is included"""
    if dc:
        n = 22
    else:
        n = 21
    amplitudes = {}
    for exp in exposures:
        key = "_".join([exp.star, exp.filter])
        if key not in amplitudes.keys():
            amplitudes[key] = np.ones(n)
    return amplitudes


def get_phases(exposures, dc=False):
    """If dc is True, an extra value for the dc term is included"""
    if dc:
        n = 22
    else:
        n = 21
    phases = {}
    for exp in exposures:
        key = "_".join([exp.star, exp.filter])
        if key not in phases.keys():
            phases[key] = np.zeros(n)
    return phases


def get_coherence(exposures):
    coherences = {}
    for exp in exposures:
        coherences[exp.key] = np.zeros(7)
    return coherences


def find_position(psf, pixel_scale=0.065524085):
    origin = np.array(determine_origin(psf, verbose=False))
    origin -= (np.array(psf.shape) - 1) / 2
    position = origin * pixel_scale * np.array([1, -1])
    return position


# def get_exposures(files, add_read_noise=False):
def get_exposures(files, optics, ms_thresh=None, as_psf=False):
    print("Prepping exposures...")
    opds = get_wss_ops(files)
    return [
        amigo.core.ExposureFit(file, optics, opd=opd, ms_thresh=ms_thresh)
        for file, opd in zip(files, opds)
    ]


def initialise_params(exposures):
    positions = {}
    fluxes = {}
    aberrations = {}
    one_on_fs = {}
    aberrations = {}
    for exp in exposures:
        positions[exp.key] = exp.position
        aberrations[exp.key] = exp.aberrations
        fluxes[exp.key] = exp.flux
        one_on_fs[exp.key] = exp.one_on_fs
    return {
        "positions": positions,
        "fluxes": fluxes,
        "aberrations": aberrations,
        "one_on_fs": one_on_fs,
    }


def full_to_SUB80(full_arr, npix_out=80, fill=0.0):
    """
    This is taken from the JWST pipeline, so its probably correct.

    The padding adds zeros to the edges of the array, keeping the SUB80 array centered.
    """
    xstart = 1045
    ystart = 1
    xsize = 80
    ysize = 80
    xstop = xstart + xsize - 1
    ystop = ystart + ysize - 1
    SUB80 = full_arr[ystart - 1 : ystop, xstart - 1 : xstop]
    if npix_out != 80:
        pad = (npix_out - 80) // 2
        SUB80 = np.pad(SUB80, pad, constant_values=fill)
    return SUB80


def get_uv_masks(files, optics, filters, mask_cache="files/uv_masks", verbose=False):
    """
    Note caches masks to disk for faster loading. The cache is indexed _relative_ to
    where the file is run from.
    """

    # Check whether the specified cache directory exists
    if not os.path.exists(mask_cache):
        os.makedirs(mask_cache)

    masks = {}
    for file in files:
        filt = file[0].header["FILTER"]
        if filt in masks.keys():
            continue
        wavels = filters[filt][0]

        calc_pad = 3  # 3x oversample on the mask calculation for soft edges
        uv_pad = 2  # 2x oversample on the UV transform

        _masks = []
        looper = tqdm(wavels) if verbose else wavels
        for wavelength in looper:
            wl_key = f"{int(wavelength*1e9)}"  # The nearest nm
            file_key = f"{mask_cache}/{wl_key}_{optics.oversample}"
            try:
                mask = np.load(f"{file_key}.npy")
            except FileNotFoundError:
                mask = uv_hex_mask(
                    optics.pupil_mask.holes,
                    optics.pupil_mask.f2f,
                    optics.pupil_mask.transformation,
                    wavelength,
                    optics.psf_pixel_scale,
                    optics.psf_npixels,
                    optics.oversample,
                    uv_pad,
                    calc_pad,
                )
                np.save(f"{file_key}.npy", mask)
            _masks.append(mask)

        masks[filt] = np.array(_masks)
    return masks
