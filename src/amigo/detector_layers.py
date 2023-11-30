import jax
import jax.numpy as np
import dLux as dl


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
