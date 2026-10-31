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

    def test_combine_matches_stacked_generalized_least_squares(self):
        from amigo.discos import combine

        values = np.array([[1., -2.], [3., .5], [-1., 2.]])
        cov = np.array([[[2., .7], [.7, 1.]], [[1., -.4], [-.4, 3.]],
                        [[4., 1.], [1., 2.]]])
        mean, covariance = jax.jit(combine)(values, cov)
        # Independent reference: solve one stacked, whitened observation problem.
        lower = np.linalg.cholesky(cov)
        design = jax.vmap(lambda l: np.linalg.solve(l, np.eye(2)))(lower).reshape(-1, 2)
        observations = jax.vmap(np.linalg.solve)(lower, values).reshape(-1)
        npt.assert_allclose(mean, np.linalg.lstsq(design, observations, rcond=None)[0])
        npt.assert_allclose(covariance, np.linalg.inv(design.T @ design))
        one, one_cov = combine(values[:1], cov[:1])
        npt.assert_allclose(one, values[0])
        npt.assert_allclose(one_cov, cov[0])

    def test_calibration_null_and_rotation_keep_data_and_operator_aligned(self):
        from amigo.discos import calibrate
        from amigo.stats import orthogonalise, build_disco

        basis = np.array([[1+2j, 0.], [0., 2-1j], [1., 1j]])
        null = np.array([[1., -1., 0.], [0., 1., 2.]])
        vis = np.array([.1+.2j, -.3+.1j])
        cal = np.array([.3, -.2, .1])
        science = cal + np.real(basis @ np.conj(vis))
        science_cov = np.array([[2., .4, .2], [.4, 3., -.3], [.2, -.3, 1.]])
        cal_cov = np.eye(3)*.2
        c, C = calibrate(science, science_cov, cal, cal_cov)
        data, covariance, rotation, variance = orthogonalise(null @ c, null @ C @ null.T)
        operator = build_disco(basis, null, rotation)
        disco = Disco(data, np.sqrt(variance), operator, [[1., 2.], [0., 1.]], 4.3e-6)
        npt.assert_allclose(disco.project(vis), data, atol=1e-14)
        npt.assert_allclose(covariance, np.diag(variance), atol=1e-14)
        expected_chi2 = (null @ c) @ np.linalg.solve(null @ C @ null.T, null @ c)
        npt.assert_allclose(np.sum((data / disco.uncertainty)**2), expected_chi2)
