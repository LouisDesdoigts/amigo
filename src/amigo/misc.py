import jax.numpy as np


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
