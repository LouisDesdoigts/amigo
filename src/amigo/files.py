import jax.numpy as np
import dLux as dl
import dLux.utils as dlu
from jax.scipy.signal import convolve
import numpy as onp


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


def initialise_for_data(tel, file=None, im=None, err=None, scale_to_counts=True):
    if file is not None:
        im, err = image_from_file(file)
        psf_npix, pos, flux = get_intial_values(tel, im, err)
    else:
        if im is None or err is None:
            raise ValueError("Must provide either file or im and err")
        psf_npix, pos, flux = get_intial_values(tel, im, err)
    return tel.set(["psf_npixels", "position", "flux"], [psf_npix, pos, flux])


from scipy.ndimage import center_of_mass


def image_from_file(file, scale_to_counts=True):
    counts_per_sec = file[1].data
    err_per_sec = np.array(file[2].data).astype(float)

    if scale_to_counts:
        int_time = file[0].header["EFFINTTM"]
        counts = counts_per_sec * int_time
        err = err_per_sec * int_time
        return np.array(counts).astype(float), err

    return np.array(counts_per_sec).astype(float), err_per_sec


def get_intial_values(tel, im, err):
    # Enforce correct npix
    im = im.at[np.where(np.isnan(im))].set(0.0)

    # Get naive model
    tel = tel.set("psf_npixels", im.shape[0])
    tel = tel.set("position", np.zeros(2))
    tel = tel.set("flux", im.sum())
    # tel = tel.set("flux", 1.0)
    psf = tel.model()

    # Get correct pixel scale
    if not isinstance(tel.optics, (dl.AngularOpticalSystem, dl.CartesianOpticalSystem)):
        optics = tel.optics.optics
    else:
        optics = tel.optics

    if isinstance(optics, dl.CartesianOpticalSystem):
        pscale = dlu.rad2arcsec(1e-6 * optics.psf_pixel_scale / optics.focal_length)
    elif isinstance(optics, dl.AngularOpticalSystem):
        pscale = optics.psf_pixel_scale
    else:
        raise ValueError("Optics must be Cartesian or Angular")

    # Get position
    conv = convolve(im, psf, mode="same")
    max_idx = np.array(np.where(conv == np.nanmax(conv))).squeeze()

    k = 1
    conv_small = conv[
        max_idx[0] - k : max_idx[0] + k + 1, max_idx[1] - k : max_idx[1] + k + 1
    ]

    com_small = np.array(center_of_mass(onp.array(conv_small)))
    max_idx += com_small - k

    # parax_pos = max_idx - im.shape[0] // 2
    # print(parax_pos)

    # shift = (im.shape[-1] - 1) / 2
    # parax_pos = max_idx - shift
    # print(parax_pos)

    # parax_pos *= pscale
    # pos = np.roll(parax_pos, 1)  # (i, j) -> (x, y)
    # print(pos)

    parax_pix_pos = max_idx - im.shape[0] // 2
    pos = np.roll(pscale * parax_pix_pos, 1)
    # print(pos)

    # Y-flip to match model
    pos *= np.array([1, -1])

    # Get flux
    ratio = im.sum() / psf.sum()
    flux = ratio * im.sum()

    return im.shape[0], pos, flux


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
