"""Optical null source coordinates use the propagation offsets, not the basis."""
from unittest import TestCase
import jax
import jax.numpy as np
import zodiax as zdx
import dLux.utils as dlu
from amigo.discos import vis_jac
from amigo.misc import spectral_weights

jax.config.update("jax_enable_x64", True)


class Optics(zdx.Base):
    coefficient: jax.Array
    filters: dict

    def propagate(self, wavelengths, *, offset, weights, return_intensity):
        return np.stack((self.coefficient, offset[0], offset[1],
                         np.sum(weights * wavelengths)))


class Basis(zdx.Base):
    def intensity_to_latent(self, intensity, filt):
        return intensity


class OpticalNullTests(TestCase):
    def test_position_and_spectrum_columns(self):
        wavelengths, throughput = np.array([3., 4., 5.]), np.array([.2, .5, .3])
        optics = Optics(np.array(.4), {"test": (wavelengths, throughput)})
        jac = vis_jac(optics, Basis(), "test", ["coefficient", "positions", "spectra"], spectra=.1)
        def wavelength(slope):
            return np.sum(spectral_weights(throughput, slope) * wavelengths)
        h = 1e-5
        derivative = (wavelength(.1+h) - wavelength(.1-h)) / (2*h)
        expected = np.diag(np.array([1., dlu.arcsec2rad(1.), dlu.arcsec2rad(1.), derivative]))
        self.assertTrue(np.allclose(jac.matrix, expected, rtol=1e-8, atol=1e-12))
        self.assertTrue(bool(np.all(np.linalg.norm(jac.matrix, axis=0) > 0)))
