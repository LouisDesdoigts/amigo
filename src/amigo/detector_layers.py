import jax
import jax.numpy as np
import dLux as dl
from jax import vmap


class AmplifierNoise(dl.layers.detector_layers.DetectorLayer):
    coeffs: jax.Array
    axis: int

    def __init__(self, coeffs, axis=0):
        self.coeffs = np.asarray(coeffs, float)
        self.axis = int(axis)

    def build(self, array):
        xs = np.linspace(-1, 1, array.shape[self.axis])
        vals = jax.vmap(lambda coeffs: np.polyval(coeffs, xs))(self.coeffs)
        if self.axis == 0:
            return vals.T
        return vals

    def apply(self, PSF):
        return PSF.add("data", self.build(PSF.data))


class AmplifierNoiseRamp(dl.layers.detector_layers.DetectorLayer):
    """This class has been 'un-generalised' a bit to make working with ramps easier"""

    coeffs: jax.Array
    axis: int

    def __init__(self, ngroups, npix=80, order=1, axis=0):
        self.coeffs = np.zeros((ngroups, npix, order + 1))
        self.axis = int(axis)

    @property
    def build(self):
        # return vmap(model_amplifier, (0, None))(self.coeffs, self.axis)
        xs = np.linspace(-1, 1, self.coeffs.shape[1])

        # Evaluation function
        eval_fn = lambda coeffs: np.polyval(coeffs, xs)

        # Vectorise over columns and groups in the data
        vals = vmap(vmap(eval_fn, 0), 0)(self.coeffs)

        if self.axis == 0:
            return vmap(np.rot90, 0)(vals)
        return vals

    def apply(self, PSF):
        return PSF.add("data", self.build)


def model_amplifier(coeffs, axis=0):
    """
    Models the amplifier noise as a polynomial along one axis of the detector.
    Assumes Detector is square and coeffs has shape (npix, order + 1).
    """
    # Evaluation function
    xs = np.linspace(-1, 1, coeffs.shape[1])
    eval_fn = lambda coeffs: np.polyval(coeffs, xs)

    # Vectorise over columns and groups in the data
    vals = vmap(eval_fn, 0)(coeffs)

    if axis == 0:
        return np.rot90(vals)
    return vals


# def model_amplifier(coeffs, axis=0):
#     """
#     Coeffs should have shape (ngroups, npix, order + 1)
#     """
#     # Evaluation function
#     xs = np.linspace(-1, 1, coeffs.shape[1])
#     eval_fn = lambda coeffs: np.polyval(coeffs, xs)

#     # Vectorise over columns and groups in the data
#     vals = vmap(vmap(eval_fn, 0), 0)(self.coeffs)

#     if axis == 0:
#         return vmap(np.rot90, 0)(vals)
#     return vals
