"""Physical sampling and derivative checks for focal-field interpolation."""

from unittest import TestCase

import dLux.utils as dlu
import jax
import jax.numpy as jnp
import numpy as np

from amigo.optical_models import resample_intensity


class TestOpticalSampling(TestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update("jax_enable_x64", True)

    def test_complex_polynomial_matches_direct_subpixel_intensities(self):
        size, factor = 9, 3
        x, y = dlu.pixel_coords(size, 2)
        field = (1 + x**2 + 2*y) + 1j*(x - y**2)
        result = resample_intensity(field, factor)

        sx, sy = dlu.pixel_coords(size*factor, 2)
        intensity = (1 + sx**2 + 2*sy)**2 + (sx - sy**2)**2
        inside = (jnp.abs(sx) <= x.max()) & (jnp.abs(sy) <= y.max())
        expected = dlu.downsample(jnp.where(inside, intensity, 0), factor, mean=True)
        np.testing.assert_allclose(result, expected, atol=2e-14, rtol=2e-14)
        self.assertGreaterEqual(float(result.min()), 0.)

    def test_field_zero_has_finite_first_and_second_intensity_derivatives(self):
        x, y = dlu.pixel_coords(8, 2)
        field = jnp.exp(-x**2-y**2) * jnp.exp(1j*(x+2*y))

        def energy(scale):
            return resample_intensity(scale*field, 3).sum()

        self.assertEqual(float(jax.grad(energy)(0.)), 0.)
        np.testing.assert_allclose(jax.hessian(energy)(0.), 2*energy(1.), rtol=2e-14)

    def test_subpixels_are_combined_in_intensity(self):
        x, y = dlu.pixel_coords(8, 2)
        field = jnp.exp(8j*x)
        result = resample_intensity(field, 3)
        # A rapidly rotating field cancels when averaged coherently; an
        # intensity integral must retain that incoherent subpixel power.
        self.assertGreater(float(result[2:-2, 2:-2].mean()), .8)

