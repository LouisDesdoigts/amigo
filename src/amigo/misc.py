import jax.numpy as np
import numpy as onp
from scipy.ndimage import center_of_mass
from scipy.interpolate import griddata
from astropy.stats import sigma_clip


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


def calc_mean_and_std_var(data, axis=0):
    support = np.asarray(~np.isnan(data), int).sum(axis)
    mean = np.nanmean(data, axis=axis)
    std_var = np.nanvar(data, axis=axis) / support
    return mean, std_var


def apply_sigma_clip(array, sigma=5.0, axis=0):
    masked = onp.ma.masked_invalid(array, copy=True)
    clipped = sigma_clip(masked, axis=axis, sigma=sigma)
    return onp.ma.filled(clipped, fill_value=onp.nan)
