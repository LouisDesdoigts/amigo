import jax.numpy as np
import jax.scipy as jsp
import numpy as onp
from scipy.ndimage import center_of_mass
from scipy.interpolate import griddata
import pkg_resources as pkg


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


def find_position(psf, pixel_scale=0.065524085):
    # TODO: Maybe iterate this operation to gain resilience to bad pixels?

    # Interpolate the bad pixels
    psf = onp.array(interp_badpix(psf))

    # Compute the center of mass
    cen = np.array(center_of_mass(psf))

    # Convert back to paraxial coordinates
    cen -= (np.array(psf.shape) - 1) / 2

    # Scale and flip the y
    y, x = cen * pixel_scale * np.array([-1, 1])

    # Return as (x, y)
    return np.array([x, y])


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
