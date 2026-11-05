"""Checks for fitted optical references, masks, and parameter-unit invariance."""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as np
import numpy.testing as npt

from amigo.discos import null_space, image_kernel
from amigo.model_fits import Exposure
from amigo.optical_models import DynamicApertureMask, AMIOptics
from amigo.opd import PTT


class ReviewTests(TestCase):
    def test_null_is_invariant_to_parameter_units(self):
        response = np.array([[1., 0.], [0., 1.], [1., 2.], [0., 0.]])
        kernel, info = null_space(response)
        rescaled, _ = null_space(response * np.array([1e-15, 1e15]))
        self.assertEqual(int(info["rank"]), 2)
        npt.assert_allclose(kernel.T @ kernel, rescaled.T @ rescaled, atol=1e-14)
        npt.assert_allclose(info["leakage"], 0, atol=1e-14)
        a = image_kernel(np.eye(4), response)
        b = image_kernel(np.eye(4), response * np.array([1e-15, 1e15]))
        npt.assert_allclose(a.T @ a, b.T @ b, atol=1e-14)

    def test_basis_keeps_exposure_selected_optics(self):
        pupil = DynamicApertureMask(holes=np.array([[-1., -1.], [1., 1.]]),
                                   npixels=64, size=16, opds={"ptt": PTT()})
        optics = AMIOptics(pupil_mask=pupil, wf_npixels=64, filters=["F380M"],
                           nwavels=1, psf_npixels=8)
        optics = optics.set(["opd_coefficients", "opd_selected"],
                            [{"ptt_coeffs": {"001": pupil.ptt_coeffs}}, True])
        with patch.object(AMIOptics, "for_filter", side_effect=AssertionError("reference reset")), \
             patch.object(AMIOptics, "propagate", return_value=None), \
             patch("amigo.vis_models.visibility_jtj", return_value=np.eye(8)):
            basis = optics.visibility_model("F380M", n_knots=3, n_modes=2)
        self.assertEqual(basis.n_basis, 2)

    def test_support_uses_final_bad_pixel_mask(self):
        header = dict(ROLL_REF=0., NINTS=10, FILTER="F380M", TARGPROP="test", OBSERVTN="001",
                      PROGRAM="001", ACT_ID="01", VISITGRP="01", EXPOSURE="00001",
                      IS_PSF=True, FILENAME="test_nis_calslope.fits")
        file = {0: SimpleNamespace(header=header)}
        arrays = dict(SLOPE=np.ones((1,80,80)), SLOPE_COV=np.ones((1,1,80,80)),
                      BADPIX=np.zeros((80,80), bool), RAMP=np.ones((2,80,80)),
                      RAMP_COV=np.ones((2,2,80,80)), SLOPE_SUP=np.full((1,80,80),10))
        file.update({key: SimpleNamespace(data=value) for key, value in arrays.items()})
        exposure = Exposure(file)
        self.assertFalse(bool(exposure.badpix[exposure.support].any()))
