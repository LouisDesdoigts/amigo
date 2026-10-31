"""DISCO representation and explicit Virgil log-visibility projection."""
from unittest import TestCase

import equinox as eqx
import jax
import jax.numpy as np
import numpy.testing as npt

from amigo.discos import Disco

jax.config.update("jax_enable_x64", True)


class TestDisco(TestCase):
    def test_projection_is_real_linear_and_differentiable(self):
        operator = np.array([[1 + 2j, 3 - 4j]])
        disco = Disco([.4], [.2], operator, [[1., 2.], [-.5, .5]], 4.3e-6)
        vis = np.array([.8 + .3j, 1 - .1j])
        expected = operator.real @ vis.real + operator.imag @ vis.imag
        npt.assert_allclose(eqx.filter_jit(disco.project)(vis), expected)
        npt.assert_allclose(jax.jacfwd(lambda x: disco.project(x + 1j * vis.imag))(vis.real),
                            operator.real)

    def test_native_virgil_handoff(self):
        from amigo.discos import to_virgil
        from virgil.models import BinaryModelCartesian
        from virgil.likelihood import model_loglike, whitened_residuals
        from jax.scipy.stats import norm

        disco = Disco([.1], [.2], [[1 + 2j, 3 - 4j]], [[1., 2.], [-.5, .5]],
                      np.array([4.3e-6, 4.8e-6]), transform=[[0., -1.], [-1., 0.]])
        data = to_virgil(disco)
        model = BinaryModelCartesian(70., -30., .02)
        prediction = disco.model(model)
        npt.assert_allclose(data.model(model), prediction)
        npt.assert_allclose(whitened_residuals(model, data),
                            (prediction-disco.data)/disco.uncertainty)
        npt.assert_allclose(model_loglike(model, data),
                            norm.logpdf(prediction, disco.data, disco.uncertainty).sum())
        npt.assert_allclose(whitened_residuals(model, data.with_model(model)), 0, atol=1e-15)

    def test_coordinate_map_preserves_fourier_phase(self):
        uv = np.array([[1., 2.], [-.5, .5]])
        transform = np.array([[.8, .3], [.2, -1.1]])
        disco = Disco([0.], [.1], [[1j, 2.]], uv, 4.3e-6, transform)
        optical_position = np.array([.03, -.04])
        sky_position = transform @ optical_position
        npt.assert_allclose(disco.sky_uv.T @ sky_position, uv.T @ optical_position)
        # Geometry remains differentiable for calibration experiments.
        gradient = jax.grad(lambda a: disco.set("transform", a).sky_uv.sum())(transform)
        self.assertTrue(bool(np.all(np.isfinite(gradient))))
