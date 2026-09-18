import jax
from jax import numpy as np, scipy as jsp
import numpy as onp
from scipy.ndimage import center_of_mass
from scipy.interpolate import griddata

# import pkg_resources as pkg
from importlib import resources
import interpax as ipx
import equinox as eqx


badpix = np.load(resources.files(__package__) / "data" / "badpix.npy")


# BIG is arbitrarily large number, depends on bit precision
if jax.config.read("jax_enable_x64"):
    BIG = np.finfo(np.float64).max / 1e1
else:
    BIG = np.finfo(np.float32).max / 1e1

def calc_laplacian(arr):
    """Laplacian - ie scalar field divergence (I think)"""
    lap_mat = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return jsp.signal.convolve(arr, lap_mat, mode="same")


def planck(wav, T):
    """
    Planck's Law:
    I(W, T) = (2hc^2 / W^5) * (1 / (exp{hc/WkT} - 1))
    where
    h = Planck's constant
    c = speed of light
    k = Boltzmann's constant

    W = wavelength array
    T = effective temperature

    Here A is the first fraction and B is the second fraction.
    The calculation is (sort of) performed in log space.
    """
    logW = np.log10(wav)  # wavelength array
    logT = np.log10(T)  # effective temperature

    # -15.92... is [log2 + logh + 2*logc]
    logA = -15.92347606 - 5 * logW
    logB = -np.log10(
        np.exp(
            # -1.84...is logh + logc - logk
            np.power(10, -1.8415064 - logT - logW)
        )
        - 1.0
    )
    return np.power(10, logA + logB)


def interp_badpix(array):
    # Get the coordinates of the good pixels
    x, y = np.indices(array.shape)
    good_pixels = ~np.isnan(array)

    # Interpolate over the bad pixels
    fixed = griddata((x[good_pixels], y[good_pixels]), array[good_pixels], (x, y), method="cubic")
    return np.where(np.isnan(fixed), 0.0, fixed)


def find_position(psf, pixel_scale=0.065524085, window_size: int = 30):
    # TODO: Maybe iterate this operation to gain resilience to bad pixels?

    # Interpolate the bad pixels
    psf = onp.array(interp_badpix(psf))

    if window_size is not None:

        # grabbing index of the peak pixel
        idx = np.unravel_index(np.argmax(psf), psf.shape)

        # checking to see the window is within the subarray
        for size in [idx[0], idx[1], psf.shape[0] - idx[0], psf.shape[1] - idx[1]]:

            # if the window is too big, reduce it
            if window_size // 2 > size:
                window_size = 2 * size

        # zeroing out outside the window
        psf[: idx[0] - window_size // 2, :] = 0
        psf[idx[0] + window_size // 2 :, :] = 0
        psf[:, : idx[1] - window_size // 2] = 0
        psf[:, idx[1] + window_size // 2 :] = 0

    # Compute the center of mass
    cen = np.array(center_of_mass(psf))

    # Convert back to paraxial coordinates
    cen -= (np.array(psf.shape) - 1) / 2

    # Scale and flip the y
    y, x = cen * pixel_scale * np.array([-1, 1])

    # Return as (x, y)
    return np.array([x, y])


def get_pos(file, box=5, return_centroid=False):
    """
    Determine which AMI primary dither position (POS1-POS4) an exposure
    corresponds to, based on centroiding the source in the image data
    and matching it to the nearest known POS pixel location.

    (CRPIX1/CRPIX2 don't work for this since they're fixed by APERNAME
    and don't vary between POS1-4.)

    Parameters
    ----------
    file : astropy.io.fits.HDUList
        An already-opened FITS file, e.g. from fits.open(path).
    box : int
        Half-width of the box around the brightest pixel used for
        the refined centroid.

    Returns
    -------
    pos_label : str
        Closest matching position, e.g. "POS1"
    dist : float
        Pixel distance to that position (sanity check - should be small)
    """
    
    # Reference pixel locations for each dither position (from your image)
    POS_COORDS = {
        "POS1": (46, 40),
        "POS2": (31, 54),
        "POS3": (18, 25),
        "POS4": (59, 24),
    }
    
    data = np.array(file[1].data)
    if data.ndim == 3:
        data = data[-1]  # collapse integrations/frames if needed
    data = np.where(badpix, 0., np.array(data))

    # crude centroid: brightest pixel, refined with a local window
    flat_idx = np.nanargmax(data)
    y0, x0 = np.unravel_index(flat_idx, data.shape)

    y_lo, y_hi = max(0, y0 - box), min(data.shape[0], y0 + box + 1)
    x_lo, x_hi = max(0, x0 - box), min(data.shape[1], x0 + box + 1)
    sub = data[y_lo:y_hi, x_lo:x_hi]

    ys, xs = np.mgrid[y_lo:y_hi, x_lo:x_hi]
    total = np.sum(sub)
    y_c = np.sum(ys * sub) / total
    x_c = np.sum(xs * sub) / total

    target = np.array([x_c, y_c])
    labels = list(POS_COORDS.keys())
    coords = np.array([POS_COORDS[l] for l in labels])

    dists = np.sqrt(np.sum((coords - target) ** 2, axis=1))
    best_idx = int(np.argmin(dists))

    if return_centroid:
        return labels[best_idx], target
    else:
        return labels[best_idx]


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


def calc_throughput(filt, nwavels=9):

    if filt not in ["F380M", "F430M", "F480M", "F277W"]:
        raise ValueError("Supported filters are F380M, F430M, F480M, F277W.")

    # filter_path = os.path.join()
    # file_path = pkg.resource_filename(__name__, f"/data/filters/{filt}.dat")
    file_path = resources.files(__package__) / "data" / "filters" / f"{filt}.dat"
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


def convert_adjacent_to_true(bool_array, n=1, corners=False):
    for i in range(n):
        trues = np.array(np.where(bool_array))
        trues = np.swapaxes(trues, 0, 1)
        for i in range(len(trues)):
            y, x = trues[i]
            bool_array = bool_array.at[y, x + 1].set(True)
            bool_array = bool_array.at[y, x - 1].set(True)
            bool_array = bool_array.at[y + 1, x].set(True)
            bool_array = bool_array.at[y - 1, x].set(True)
            if corners:
                bool_array = bool_array.at[y + 1, x + 1].set(True)
                bool_array = bool_array.at[y - 1, x - 1].set(True)
                bool_array = bool_array.at[y + 1, x - 1].set(True)
                bool_array = bool_array.at[y - 1, x + 1].set(True)
    return bool_array


def nearest_fn(pt, coords):
    dist = np.hypot(*(coords - pt[:, None, None]))
    return dist == dist.min()


def nuke_brightest(file, n=0):
    im = np.array(file["SLOPE"].data.sum(0))
    badpix = np.array(file["BADPIX"].data)
    im = np.where(badpix, np.nan, im)

    for i in range(n):
        badpix = badpix.at[np.where(im == np.nanmax(im))].set(1)
        im = np.where(badpix, np.nan, im)

    file["BADPIX"].data = badpix


@eqx.filter_jit
def interp(image, knot_coords, sample_coords, method="linear", fill=0.0):
    xs, ys = knot_coords
    xpts, ypts = sample_coords.reshape(2, -1)

    return ipx.interp2d(ypts, xpts, ys[:, 0], xs[0], image, method=method, extrap=fill).reshape(
        sample_coords[0].shape
    )


@eqx.filter_jit
def interp_ramp(ramp, ngroups, method="cubic2", extrap=True):
    # Assumes that the ramp time samples are from 0 to 1.
    ts = np.linspace(0, 1, len(ramp))
    groups = (np.arange(ngroups) + 1) / ngroups

    # Build the vectorised interpolator
    interpolator = eqx.filter_vmap(
        lambda f: ipx.interp1d(groups, ts, f, method=method, extrap=extrap),
        in_axes=1,
        out_axes=1,
    )

    # Get the group sample points and interpolate the ramp
    ramp_vec = ramp.reshape(len(ramp), -1)
    return interpolator(ramp_vec).reshape(ngroups, *ramp.shape[1:])


def populate_from_state(model, state):
    if not isinstance(state, dict):
        state = state.params

    # Populate parameters
    for param_key, value in state.items():
        if isinstance(value, dict):
            for exp_key, sub_value in value.items():
                try:
                    model = model.set(f"{param_key}.{exp_key}", sub_value)

                # Catch key not existing since we might not have a certain exposure here
                except KeyError:
                    pass
        else:
            model = model.set(param_key, value)
    return model


def gen_surface(coords, coeffs, pows):
    """The only difference here and the fn in amigo is this one return the sum of the
    distortion, not the coords + distortion"""
    pow_base = np.multiply(*(coords[:, None, ...] ** pows[..., None, None]))
    surface = np.sum(coeffs[..., None, None] * pow_base[None, ...], axis=1)
    return surface.sum(0)
