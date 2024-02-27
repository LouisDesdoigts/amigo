import jax.numpy as np
import jax.scipy as jsp
from astroquery.simbad import Simbad
import pyia
import pkg_resources as pkg
import numpy as onp
from jax import vmap
from .stats import check_symmetric, check_positive_semi_definite, build_covariance_matrix
from .misc import convert_adjacent_to_true, fit_slope, slope_im
from webbpsf import mast_wss
from xara.core import determine_origin
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
    return table["SP_TYPE"].tolist()[0], table["SP_QUAL"].tolist()[0]


def get_gaia_Teff(source_id, data_dr="dr3"):
    """Returns the Teff from Gaia for a given source ID."""
    result_table = Simbad.query_objectids(source_id)
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
    spec_type, qual = get_simbad_spectral_type(targ_name)
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



def prep_data(file, read_noise, ms_thresh=0., bs_thresh=250, ngroups=None):
    ramp = np.asarray(file["SCI"].data, float)    
    err = np.asarray(file["ERR"].data, float)
    dq = np.asarray(file["PIXELDQ"].data > 0, bool)

    # only using the first ngroups
    if ngroups is not None:
        total_groups = ramp.shape[0]
        ramp = ramp[:ngroups]
        err = err[:ngroups]

    # Build the covariance matrix
    cov = build_covariance_matrix(err, read_noise=read_noise, min_value=True)

    # Check for bad pixels around dq'd pixels
    dilated_dq = convert_adjacent_to_true(dq)
    dq_edges = dilated_dq & ~dq
    dq_edge_inds = np.array(np.where(dq_edges))
    dq_edge_ramps = ramp[:, *dq_edge_inds].T
    ms, bs = vmap(fit_slope)(dq_edge_ramps)
    for i in range(len(dq_edge_ramps)):
        if ms[i] <= ms_thresh:
            dq = dq.at[*dq_edge_inds[:, i]].set(True)
        if bs[i] > bs_thresh:
            dq = dq.at[*dq_edge_inds[:, i]].set(True)

    # Set bad rows and cols
    dq = dq.at[:4].set(True)  # Bottom 4 rows are bad
    dq = dq.at[-1].set(True)  # Top row is bad
    dq = dq.at[:, -2:].set(True)  # Right 2 columns are bad
    dq = dq.at[:, 0].set(True)  # Left column is bad

    # Check for symmetry and positive semi-definite
    ngroups = ramp.shape[0]
    flat_cov = cov.reshape(ngroups, ngroups, -1)
    is_sym = vmap(check_symmetric, -1)(flat_cov).reshape(80, 80)
    is_psd = vmap(check_positive_semi_definite, -1)(flat_cov).reshape(80, 80)
    supp_mask = is_sym & is_psd & ~np.isnan(ramp.sum(0)) & ~dq

    # Nan the bad pixels
    support = np.where(supp_mask)
    ramp = ramp.at[:, ~supp_mask].set(np.nan)
    cov = cov.at[..., ~supp_mask].set(np.nan)
    return ramp, cov, support

def get_wss_ops(files):
    opds = {}
    opd_files = []
    for file in files:
        date = file[0].header["DATE-BEG"]
        opd0, opd1, t0, t1 = mast_wss.mast_wss_opds_around_date_query(
            date, verbose=False
        )
        closest_fn, closest_dt = (opd1, t1) if abs(t1) < abs(t0) else (opd0, t0)
        opd_file = mast_wss.mast_retrieve_opd(closest_fn)
        if opd_file not in opds.keys():
            opds[opd_file] = mast_wss.import_wss_opd(opd_file)[0].data
        opd_files.append(opds[opd_file])
    return opd_files


def get_Teffs(files, default=4500):
    Teffs = {}
    for file in files:
        prop_name = file[0].header["TARGPROP"]

        if prop_name in Teffs:
            continue

        Teff = get_Teff(file[0].header["TARGNAME"])

        if Teff == -1:
            print(f"No Teff found for {prop_name}, defaulting to 4500K")
            Teffs[prop_name] = default
        else:
            Teffs[prop_name] = Teff

    return Teffs

def get_amplitudes(files):
    amplitudes = {}
    for file in files:
        prop_name = file[0].header["TARGPROP"]
        filt = file[0].header["FILTER"]
        if prop_name in amplitudes:
            if filt in amplitudes[prop_name].keys():
                continue
            else:
                amplitudes[prop_name][filt] = np.ones(21)
        amplitudes[prop_name] = {filt: np.ones(21)}
    return amplitudes


def get_phases(files):
    phases = {}
    for file in files:
        prop_name = file[0].header["TARGPROP"]
        filt = file[0].header["FILTER"]
        if prop_name in phases:
            if filt in phases[prop_name].keys():
                continue
            else:
                phases[prop_name][filt] = np.zeros(21)
        phases[prop_name] = {filt: np.zeros(21)}
    return phases


def find_position(psf, pixel_scale):
    origin = np.array(determine_origin(psf, verbose=False))
    origin -= (np.array(psf.shape) - 1) / 2
    position = origin * pixel_scale * np.array([1, -1])
    return position

def get_exposures(files, ngroups=None):
    opds = get_wss_ops(files)
    return [amigo.core.Exposure(file, opd=opd, ngroups=ngroups) for file, opd in zip(files, opds)]

def initialise_params(exposures, pixel_scale=0.065524085):
    FDA_coefficients = np.load(pkg.resource_filename(__name__, "data/FDA_coeffs.npy"))
    positions = {}
    fluxes = {}
    aberrations = {}
    biases = {}
    OneOnFs = {}
    aberrations = {}
    for exp in exposures:
        psf_guess, bias = estimate_psf_and_bias(exp.data)
        biases[exp.key] = bias
        fluxes[exp.key] = psf_guess.sum() * 1.075  # Seems to be under estimated

        # TODO: PSF Pixel scale
        positions[exp.key] = find_position(psf_guess, pixel_scale)
        OneOnFs[exp.key] = np.zeros((exp.ngroups, 80, 2))
        aberrations[exp.key] = FDA_coefficients
    return {
        "positions": positions,
        "fluxes": fluxes,
        "aberrations": aberrations,
        "biases": biases,
        "OneOnFs": OneOnFs,
    }


def full_to_SUB80(full_arr, npix_out=80, fill=0.):
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