import jax.numpy as np
from jax import vmap
import numpy as onp
import jax.scipy as jsp


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


def full_to_SUB80(full_arr):
    """This is taken from the JWST pipeline, so its probably correct"""
    xstart = 1045
    ystart = 1
    xsize = 80
    ysize = 80
    xstop = xstart + xsize - 1
    ystop = ystart + ysize - 1
    return full_arr[ystart - 1 : ystop, xstart - 1 : xstop]


def least_sq(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, b

def fit_slope(y):
    return least_sq(np.arange(len(y)) + 1, y)

def slope_im(im):
    ms, bs = vmap(fit_slope)(im.reshape(len(im), -1).T)
    return ms.reshape(im.shape[1:]), bs.reshape(im.shape[1:])

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


def estimate_psf_and_bias(data):
    ngroups = len(data)
    ramp_bottom = data[:2]
    ramp_bottom = np.where(np.isnan(ramp_bottom), 0, ramp_bottom)
    psf, bias = slope_im(ramp_bottom) # Estimate from the bottom of the ramp
    return psf * ngroups, bias

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