"""Physical sampling and derivative checks for focal-field interpolation."""

from types import SimpleNamespace
from unittest import TestCase

import dLux.utils as dlu
import jax
import jax.numpy as jnp
import numpy as np

from amigo.optical_models import _fft, _ifft, intensity_to_amplitude, resample_psf


class TestOpticalSampling(TestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update("jax_enable_x64", True)

    def test_compiled_fft_then_tridiagonal_solve_preserves_values_and_derivatives(self):
        size = 24
        x, y = dlu.pixel_coords(size, 2)
        field = (1 + x**2 + 2*y) + 1j*(x - y**2)
        matrix = 4*np.eye(size) + np.eye(size, k=1) + np.eye(size, k=-1)
        for transform, reference in (
            (_fft, np.fft.fft2(np.asarray(field))/size),
            (_ifft, size*np.fft.ifft2(np.asarray(field))),
        ):
            with self.subTest(transform=transform.__name__):
                def energy(scale):
                    rhs = transform(scale*field, pad=1).real
                    solution = jax.lax.linalg.tridiagonal_solve(
                        jnp.ones(size).at[0].set(0), jnp.full(size, 4.),
                        jnp.ones(size).at[-1].set(0), rhs,
                    )
                    return jnp.sum(solution**2)

                expected = np.sum(np.linalg.solve(matrix, reference.real)**2)
                value, derivative = jax.jit(jax.value_and_grad(energy))(1.)
                second = jax.jit(jax.grad(jax.grad(energy)))(1.)
                np.testing.assert_allclose(
                    [value, derivative, second], [expected, 2*expected, 2*expected],
                    rtol=2e-13, atol=2e-13,
                )

    def test_complex_polynomial_matches_direct_subpixel_intensities(self):
        size, factor = 9, 3
        x, y = dlu.pixel_coords(size, 2)
        field = (1 + x**2 + 2*y) + 1j*(x - y**2)
        wf = SimpleNamespace(amplitude=jnp.abs(field), phasor=field)
        result = resample_psf(wf, factor)

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
            wf = SimpleNamespace(amplitude=jnp.abs(field), phasor=scale*field)
            return resample_psf(wf, 3).sum()

        self.assertEqual(float(jax.grad(energy)(0.)), 0.)
        np.testing.assert_allclose(jax.hessian(energy)(0.), 2*energy(1.), rtol=2e-14)

    def test_subpixels_are_combined_in_intensity(self):
        x, y = dlu.pixel_coords(8, 2)
        field = jnp.exp(8j*x)
        wf = SimpleNamespace(amplitude=jnp.ones_like(x), phasor=field)
        result = resample_psf(wf, 3)
        # A rapidly rotating field cancels when averaged coherently; an
        # intensity integral must retain that incoherent subpixel power.
        self.assertGreater(float(result[2:-2, 2:-2].mean()), .8)

    def test_local_amplitude_conversion_preserves_positive_values(self):
        intensity = jnp.array([0., 1e-20, .5, 4.])
        np.testing.assert_allclose(intensity_to_amplitude(intensity)**2, intensity)
        self.assertEqual(float(jax.grad(lambda x: intensity_to_amplitude(x)**2)(0.)), 0.)
        self.assertTrue(bool(jnp.isnan(intensity_to_amplitude(jnp.array(-1.)))))
