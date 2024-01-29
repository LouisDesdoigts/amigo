import jax.numpy as np
import dLux as dl
import dLux.utils as dlu
from jax.scipy.signal import convolve
import numpy as onp


def estimate_ramp(data):
    ramp = np.arange(len(data)) + 1
    mean_diff = np.diff(data, axis=0).mean(0)
    ramp = mean_diff[None, ...] * ramp[..., None, None]
    return ramp.at[np.where(np.isnan(ramp))].set(0.0)


def estimate_bias(data):
    mean_diff = np.diff(data, axis=0).mean(0)
    bias = data[0] - mean_diff
    return bias.at[np.where(np.isnan(bias))].set(0.0)


def summarise_files(files, extra_keys=[]):
    main_keys = [
        "TARGPROP",
        "FILTER",
        "OBSERVTN",
        "PATTTYPE",
    ]

    main_keys += extra_keys
    for key in main_keys:
        values = set([f[0].header[key] for f in files])
        vals_str = ", ".join([f"{val}" for val in values])
        print(f"  {key}: {vals_str}")


def get_files(dir_or_files, ext, is_dir=True, **kwargs):
    """

    data_path: Path to the data files
    ext: File extension to search for
    """
    import os
    from astropy.io import fits

    if is_dir:
        file_names = os.listdir(dir_or_files)
    else:
        file_names = dir_or_files

    files = []
    checked = False
    for name in file_names:
        if name.endswith(f"{ext}.fits"):
            if is_dir:
                file = fits.open(dir_or_files + name)
            else:
                file = fits.open(name)
            h = file[0].header

            if not checked:
                if not all([key in h.keys() for key in kwargs.keys()]):
                    raise KeyError(
                        f"Header keys {kwargs.keys()} not found in file {name}"
                    )

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


def get_webb_osys_fits(file, load_wss=True):
    import webbpsf
    import datetime

    inst = getattr(webbpsf, file[0].header["INSTRUME"])()

    # Set filter
    inst.filter = file[0].header["FILTER"]

    # Set aperture
    inst.aperturename = file[0].header["APERNAME"]

    # Set pupil mask
    if file[0].header["PUPIL"] == "NRM":
        pupil_in = "MASK_NRM"
    else:
        pupil_in = file[0].header["PUPIL"]
    inst.pupil_mask = pupil_in

    # Set WFS data
    d1 = datetime.datetime.fromisoformat(file[0].header["DATE-BEG"])
    d2 = datetime.datetime.fromisoformat(file[0].header["DATE-END"])

    # Weirdness here because you cant add datetimes
    time = (d1 + (d2 - d1) / 2).isoformat()

    # Load WFS data
    if load_wss:
        print("Loading WFS data")
        inst.load_wss_opd_by_date(time, verbose=False)

    # Calculate data to ensure things are populated correctly
    psf_fits = inst.calc_psf()

    return inst, psf_fits


# from scipy.ndimage import center_of_mass


# def get_intial_values(tel, im):
#     # Enforce correct npix
#     im = im.at[np.where(np.isnan(im))].set(0.0)

#     # Get naive model
#     tel = tel.set("psf_npixels", im.shape[-1])
#     tel = tel.set("position", np.zeros(2))
#     tel = tel.set("flux", im.sum())
#     psf = tel.model()[0][-1]  # psf, last group

#     # Get correct pixel scale
#     if not isinstance(tel.optics, (dl.AngularOpticalSystem, dl.CartesianOpticalSystem)):
#         optics = tel.optics.optics
#     else:
#         optics = tel.optics

#     if isinstance(optics, dl.CartesianOpticalSystem):
#         pscale = dlu.rad2arcsec(1e-6 * optics.psf_pixel_scale / optics.focal_length)
#     elif isinstance(optics, dl.AngularOpticalSystem):
#         pscale = optics.psf_pixel_scale
#     else:
#         raise ValueError("Optics must be Cartesian or Angular")

#     # Get position
#     conv = convolve(im, psf, mode="same")
#     max_idx = np.array(np.where(conv == np.nanmax(conv))).squeeze()

#     k = 1
#     conv_small = conv[
#         max_idx[0] - k : max_idx[0] + k + 1, max_idx[1] - k : max_idx[1] + k + 1
#     ]

#     com_small = np.array(center_of_mass(onp.array(conv_small)))
#     max_idx += com_small - k

#     parax_pix_pos = max_idx - im.shape[0] // 2
#     pos = np.roll(pscale * parax_pix_pos, 1)

#     # Y-flip to match model
#     pos *= np.array([1, -1])

#     # Get flux
#     ratio = im.sum() / psf.sum()
#     flux = ratio * im.sum()

#     return im.shape[0], pos, flux


def convert_adjacent_to_true(bool_array):
    trues = np.array(np.where(bool_array))
    trues = np.swapaxes(trues, 0, 1)
    for i in range(len(trues)):
        y, x = trues[i]
        bool_array = bool_array.at[y, x + 1].set(True)
        bool_array = bool_array.at[y, x - 1].set(True)
        bool_array = bool_array.at[y + 1, x].set(True)
        bool_array = bool_array.at[y - 1, x].set(True)
    return bool_array


def nan_brightest(array, n_mask, order=1, thresh=None):
    # Get the high flux mask
    if n_mask > 0:
        sorted = np.sort(array.flatten())
        thresh_in = sorted[~np.isnan(sorted)][-n_mask]

        if thresh is not None:
            thresh_in = np.minimum(thresh, thresh_in)

        flux_mask = convert_adjacent_to_true(array >= thresh_in)
        if order > 1:
            for i in range(order - 1):
                flux_mask = convert_adjacent_to_true(flux_mask)
        return array.at[np.where(flux_mask)].set(np.nan)
    return array


def get_nan_support(file, n_mask=1, order=1, thresh=None, edge_mask=0):
    # Get the data we need
    im = np.array(file[1].data).astype(float)
    dq = np.array(file[3].data).astype(bool)

    im = nan_brightest(im, n_mask, order, thresh)

    if edge_mask > 0:
        im = im.at[:edge_mask].set(np.nan)
        im = im.at[-edge_mask:].set(np.nan)
        im = im.at[:, :edge_mask].set(np.nan)
        im = im.at[:, -edge_mask:].set(np.nan)
    return ~np.isnan(im) & ~dq


from optical_layers import DynamicAMI
import webbpsf
from astropy.io import fits
from zodiax.experimental import serialise, deserialise
from jax import vmap
from core import (
    Modeller,
    SUB80Ramp,
    AMIOptics,
    Exposure,
    Star,
    get_read_cov,
    build_covariance_matrix,
    get_covariance_matrix,
)
from BFE import PolyBFE
from misc import full_to_SUB80
import jax.scipy as jsp


def check_symmetric(mat):
    """Checks if a matrix is symmetric"""
    return np.allclose(mat, mat.T)


def check_positive_semi_definite(mat):
    """Checks if a matrix is positive semi-definite"""
    if np.isnan(mat).any():
        return False
    return np.all(np.linalg.eigvals(mat) >= 0)


def get_filter(filter_name: str, filter_dir: str, n_wavels: int = 9):
    if filter_name not in ["F380M", "F430M", "F480M", "F277W"]:
        raise ValueError("Supported filters are F380M, F430M, F480M, F277W.")

    wl_array, throughput_array = np.array(
        onp.loadtxt(filter_dir + "JWST_NIRISS." + filter_name + ".dat", unpack=True)
    )

    edges = np.linspace(wl_array.min(), wl_array.max(), n_wavels + 1)
    wavels = np.linspace(wl_array.min(), wl_array.max(), 2 * n_wavels + 1)[1::2]

    areas = []
    for i in range(n_wavels):
        cond1 = edges[i] < wl_array
        cond2 = wl_array < edges[i + 1]
        throughput = np.where(cond1 & cond2, throughput_array, 0)
        areas.append(jsp.integrate.trapezoid(y=throughput, x=wl_array))

    areas = np.array(areas)
    weights = areas / areas.sum()

    wavels *= 1e-10
    return wavels, weights


def initialise_model(files, key_fn, nwavels=11, BFE_model=None):
    print("Constructing model")

    # key = key_fn(files[0])
    read_noise_file = fits.open(
        "/Users/louis/Data/JWST/jwst_niriss_readnoise_0005.fits"
    )
    read_noise = full_to_SUB80(np.array(read_noise_file[1].data))

    # Get webbpsf optical system
    inst = webbpsf.NIRISS()
    inst.load_wss_opd_by_date(files[0][0].header["DATE-BEG"], verbose=False)
    # TODO: Be more clever here and use hte webbpsf functions to check the before and
    # after WFS measurements and check if all data is within those bounds. Can use a
    # dictionary of the times to check for pre-loaded ones and possibly interpolate
    # between them to the correct time. This may be easier than just loading directly
    # from CRDS.

    # Load the FF
    ff_file = fits.open("/Users/louis/Data/JWST/jwst_niriss_flat_0277.fits")
    ff = full_to_SUB80(np.array(ff_file[1].data))

    # Get dLux optical system
    optics = AMIOptics(radial_orders=np.arange(6), normalise=True)
    optics = optics.set("opd", np.array(inst.get_optical_system().planes[0].opd))

    # Update to calibrated AMI mask
    optics = optics.set("pupil_mask", deserialise("files/pupil_mask.zdx"))

    # Get the detector model
    # ngroups = 2  # Dummy value
    if BFE_model is None:
        BFE_model = PolyBFE(5, order=3)
    detector = SUB80Ramp(2, BFE_model)
    # detector = SUB80Ramp(2, deserialise("files/BFE_5_cubic.zdx"))
    detector = detector.set("FF", ff)

    # Get filters
    path = "/Users/louis/Data/JWST/niriss_filters/"
    filters = {}
    for file in files:
        filter = file[0].header["FILTER"]
        if filter in filters.keys():
            continue
        filters[filter] = np.array(get_filter(filter, path, nwavels))

    # Get the stars:
    Teffs = {"HD-36805": 4814}
    stars = {}
    for file in files:
        star_name = file[0].header["TARGPROP"]
        if star_name in stars.keys():
            continue
        if star_name in Teffs.keys():
            stars[star_name] = Star(Teffs[star_name])
        else:
            stars[star_name] = Star(4500)  # Default to 4500K

    # Get the exposures:
    exposures = []
    # OneOnFs = []
    for i, file in enumerate(files):
        key = key_fn(file)
        nints = file[0].header["NINTS"]
        ngroups = file[0].header["NGROUPS"]
        filter = file[0].header["FILTER"]
        star = file[0].header["TARGPROP"]

        data = np.asarray(file[1].data, float)
        data = data.at[:, :4].set(np.nan)  # Bottom 4 rows are bad
        data = data.at[:, -1].set(np.nan)  # Top row is bad
        data = data.at[:, :, -2:].set(np.nan)  # Right 2 columns are bad
        data = data.at[:, :, 0].set(np.nan)  # Left column is bad

        # read_cov = get_read_cov(read_noise, ngroups)
        # # cov_mat_inds = build_covariance_matrix_inds(ngroups)
        # cov = build_covariance_matrix(data) + read_cov

        one_on_fs = np.zeros((ngroups, 80, 2))
        bias = estimate_bias(data)
        cov = get_covariance_matrix(data, bias, one_on_fs, read_noise)
        flat_cov = cov.reshape(ngroups, ngroups, -1)
        is_sym = vmap(check_symmetric, -1)(flat_cov).reshape(80, 80)

        is_psd = []
        for i in range(flat_cov.shape[-1]):
            is_psd.append(check_positive_semi_definite(flat_cov[..., i]))
        is_psd = np.array(is_psd).reshape(80, 80)

        supp_mask = is_sym & is_psd & ~np.isnan(data.sum(0))
        support = np.where(supp_mask)
        data = data.at[:, ~supp_mask].set(np.nan)

        exposures.append(Exposure(data, support, nints, filter, star, key))

    positions = {}
    fluxes = {}
    aberrations = {}
    biases = {}
    OneOnFs = {}

    # coeffs = np.array([ab for ab in final_model.aberrations.values()]).mean(0)
    # pupil_mask = final_model.optics.pupil_mask

    # aberrations = deserialise("files/aberration_coeffs.zdx")
    coeffs = np.load("files/aberration_coeffs.npy")

    from xara.core import determine_origin

    for exposure in exposures:
        key = exposure.key

        im = estimate_ramp(exposure.data)[-1]
        origin = np.array(determine_origin(im, verbose=False))
        origin -= (np.array(im.shape) - 1) / 2

        pos = origin * optics.psf_pixel_scale * np.array([1, -1])

        positions[key] = pos
        fluxes[key] = 1.075 * estimate_ramp(exposure.data)[-1].sum()
        aberrations[key] = coeffs
        OneOnFs[key] = np.zeros((exposure.ngroups, 80, 2))
        biases[key] = estimate_bias(exposure.data)

    # Build exposure model
    model = Modeller(
        optics,
        detector,
        stars,
        filters,
        positions,
        fluxes,
        aberrations,
        biases,
        OneOnFs,
    )

    return model, exposures


