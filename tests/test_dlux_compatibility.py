"""Small API checks for dLux 0.15; full propagation checks run on the GPU."""
from unittest import TestCase

import dLux as dl
import dLux.utils as dlu
import jax
import jax.numpy as jnp
import numpy as np

from amigo.detector_models import LinearDetector, Resample
from amigo.optical_models import DynamicApertureMask, PolynomialDistortion, crop_windows, fill, window_centers
from amigo.ramp_models import Ramp
from amigo.read_models import ReadModel


class TestDLuxCompatibility(TestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update('jax_enable_x64', True)

    def test_window_coordinates_match_dense_pupil(self):
        corners = jnp.array([[3, 5], [20, 18]])
        size, npixels, diameter = 8, 32, 6.6
        coordinates = dlu.pixel_coords(npixels, diameter)
        centers = window_centers(corners, size, npixels, diameter)
        local = dlu.pixel_coords(size, pixel_scale=diameter/npixels)
        for axis in range(2):
            np.testing.assert_allclose(
                crop_windows(coordinates[axis], corners, size),
                centers[:, axis, None, None] + local[axis], atol=1e-15,
            )

    def test_dynamic_dense_and_sparse_pupils_include_reference_opd(self):
        holes = jnp.array([[-1., -1.], [1., 1.]])
        # An empty aberration basis must still give valid window corners.
        mask = DynamicApertureMask(holes=holes, npixels=64, size=16,
                                   distortion=PolynomialDistortion(order=3, n_holes=2), aberration_orders=None,
                                   amplitude_orders=None)
        windows, opd, amp, _ = mask.sparse_fields(64, 6.603464)
        np.testing.assert_allclose(mask.calc_mask(64, 6.603464), fill(windows, mask.corners, 64))
        wf = dl.Wavefront(4.3e-6, 64, 6.603464)
        dense = mask(wf)
        self.assertEqual(dense.phasor.shape, (64, 64))
        np.testing.assert_allclose(jnp.sum(dense.psf), 1., atol=1e-14)
        np.testing.assert_array_equal(opd, 0.)
        np.testing.assert_array_equal(amp, 0.)
        reference = jnp.broadcast_to(jnp.linspace(-1e-8, 1e-8, 16), (2, 16, 16))
        mask = mask.set('opds', {'reference': reference})
        _, opd, _, _ = mask.sparse_fields(64, 6.603464)
        np.testing.assert_allclose(opd, reference)
        dense = mask(wf)
        expected = windows*jnp.exp(-2j*jnp.pi*reference/4.3e-6)
        expected = expected/jnp.sqrt(jnp.sum(jnp.abs(expected)**2))
        np.testing.assert_allclose(dense.phasor, (fill(expected.real, mask.corners, 64) + 1j*fill(expected.imag, mask.corners, 64)), atol=1e-14)


    def test_detector_resampling_is_applied_and_returns_requested_type(self):
        data = jnp.exp(-((jnp.arange(21)[:, None]-8)**2 +
                         (jnp.arange(21)[None, :]-12)**2)/5)
        psf = dl.PSF(data, dlu.arcsec2rad(.022))
        detector = LinearDetector().set('layers', {'resampler': Resample(11., 1.2)})
        expected = detector.resampler(psf)
        np.testing.assert_allclose(detector(psf), expected.data)
        np.testing.assert_allclose(detector(psf, return_psf=True).data, expected.data)
        np.testing.assert_allclose(detector.apply(psf).data, expected.data)
        self.assertGreater(float(jnp.max(jnp.abs(expected.data-data))), .01)

    def test_read_layers_and_optional_ipc(self):
        ramp = Ramp(jnp.full((2, 80, 80), 100.), dlu.arcsec2rad(.065))
        read = ReadModel(ipc=False, dark_current=.25, gain=2.)
        expected = (ramp.data + .25*jnp.arange(1, 3)[:, None, None])/2
        np.testing.assert_allclose(read(ramp), expected)
        self.assertIsInstance(read(ramp, return_psf=True), Ramp)
