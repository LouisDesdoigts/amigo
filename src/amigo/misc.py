import jax.numpy as np
from jax import vmap


# def planck(wav, T):
#     """
#     Planck's Law:
#     I(W, T) = (2hc^2 / W^5) * (1 / (exp{hc/WkT} - 1))
#     where
#     h = Planck's constant
#     c = speed of light
#     k = Boltzmann's constant

#     W = wavelength array
#     T = effective temperature

#     Here A is the first fraction and B is the second fraction.
#     The calculation is (sort of) performed in log space.
#     """
#     logW = np.log10(wav)  # wavelength array
#     logT = np.log10(T)  # effective temperature

#     # -15.92... is [log2 + logh + 2*logc]
#     logA = -15.92347606 - 5 * logW
#     logB = -np.log10(
#         np.exp(
#             # -1.84...is logh + logc - logk
#             np.power(10, -1.8415064 - logT - logW)
#         )
#         - 1.0
#     )

#     return np.power(10, logA + logB)


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
