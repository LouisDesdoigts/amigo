"""Checks for the tridiagonal slope covariance (read- plus photon-noise model)."""

from types import SimpleNamespace
from unittest import TestCase

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as np
import numpy.testing as npt

from amigo.model_fits import ModelFit
from amigo.optimisation import _cholesky
from amigo.stats import tridiagonal_covariance


def toeplitz(n, diagonal, off_diagonal):
    """(n, n, pixels) covariance with per-pixel diagonal and first off-diagonal values."""
    eye, off = np.eye(n)[..., None], (np.eye(n, k=1) + np.eye(n, k=-1))[..., None]
    return eye * diagonal + off * off_diagonal


def by_pixel(cov):
    """(slopes, slopes, pixels) -> (pixels, slopes, slopes), as optimisation._cholesky takes."""
    return np.moveaxis(cov, -1, 0)


def fake_file(slope_cov):
    """A calslope-like file whose every pixel has the given (slopes, slopes) covariance."""
    n = slope_cov.shape[0]
    header = dict(
        ROLL_REF=0.0,
        NINTS=10,
        FILTER="F380M",
        TARGPROP="test",
        OBSERVTN="001",
        PROGRAM="001",
        ACT_ID="01",
        VISITGRP="01",
        EXPOSURE="00001",
        IS_PSF=True,
        FILENAME="test_nis_calslope.fits",
    )
    arrays = dict(
        SLOPE=np.ones((n, 80, 80)),
        SLOPE_COV=np.broadcast_to(slope_cov[..., None, None], (n, n, 80, 80)),
        BADPIX=np.zeros((80, 80), bool),
        RAMP=np.ones((n + 1, 80, 80)),
        RAMP_COV=np.ones((n + 1, n + 1, 80, 80)),
        SLOPE_SUP=np.full((n, 80, 80), 10),
    )
    file = {0: SimpleNamespace(header=header)}
    file.update({key: SimpleNamespace(data=value) for key, value in arrays.items()})
    return file


class TridiagonalCovarianceTests(TestCase):
    def test_strong_anticorrelation_gives_positive_definite_covariance(self):
        # Neighbouring correlation -0.55 is past the -0.5 limit for many slopes: the
        # tridiagonal cut is not positive definite, the read/photon-noise model is.
        n, variance = 10, np.linspace(1.0, 2.0, 6)
        cut = toeplitz(n, variance, -0.55 * variance)
        data = np.zeros((variance.size, n))
        with self.assertRaisesRegex(ValueError, "positive definite"):
            _cholesky(data, by_pixel(cut))
        lower = _cholesky(data, by_pixel(tridiagonal_covariance(cut)))
        self.assertTrue(bool(np.all(np.isfinite(lower))))

    def test_keeps_measured_diagonal_and_sets_off_diagonals_to_minus_read_noise(self):
        n = 6
        diagonal = np.array(
            [
                [1.0, 1.2, 1.5, 2.0, 0.3, 1.1],  # slope 4 below 2r: its photon noise clips to 0
                [0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
                [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # positive neighbours: read noise clips to 0
            ]
        ).T
        off_diagonal = np.array(
            [[-0.25, -0.2, -0.3, -0.15, -0.35], [-0.4] * 5, [0.05, 0.1, 0.0, 0.05, 0.1]]
        ).T
        cov = toeplitz(n, 0.0, 0.0)
        cov = cov + np.eye(n)[..., None] * diagonal[None]
        cov = cov + np.eye(n, k=1)[..., None] * np.vstack([off_diagonal, np.zeros(3)])[:, None]
        cov = cov + np.eye(n, k=-1)[..., None] * np.vstack([off_diagonal, np.zeros(3)])[None]
        cov = cov + 0.05 * (np.eye(n, k=2) + np.eye(n, k=-2))[..., None]  # dropped by the model

        model = tridiagonal_covariance(cov)
        read = np.maximum(0.0, -off_diagonal.mean(0))
        npt.assert_allclose(read, [0.25, 0.4, 0.0])
        npt.assert_allclose(np.diagonal(model, 0, 0, 1), np.maximum(diagonal.T, 2 * read[:, None]))
        npt.assert_allclose(
            np.diagonal(model, 1, 0, 1), -np.broadcast_to(read[:, None], (3, n - 1))
        )
        npt.assert_allclose(
            np.diagonal(model, -1, 0, 1), -np.broadcast_to(read[:, None], (3, n - 1))
        )
        npt.assert_allclose(np.diagonal(model, 2, 0, 1), 0.0)
        unclipped = diagonal.T >= 2 * read[:, None]
        self.assertEqual(int((~unclipped).sum()), 1)
        npt.assert_allclose(np.diagonal(model, 0, 0, 1)[unclipped], diagonal.T[unclipped])

    def test_single_slope_is_unchanged(self):
        cov = np.array([[[0.7, 1.3, 2.0]]])
        npt.assert_array_equal(tridiagonal_covariance(cov), cov)
        fit = ModelFit(fake_file(np.array([[0.7]])), covariance="tridiagonal")
        npt.assert_array_equal(fit.cov, np.full((1, 1, 80, 80), 0.7))

    def test_only_diag_alias_selects_tridiagonal(self):
        n = 4
        measured = toeplitz(n, np.array([1.0]), np.array([-0.45]))[..., 0]
        with self.assertWarnsRegex(FutureWarning, "only_diag are deprecated.*'tridiagonal'"):
            fit = ModelFit(fake_file(measured), only_diag=True)
        self.assertEqual(fit.covariance, "tridiagonal")
        npt.assert_allclose(fit.cov[:, :, 0, 0], tridiagonal_covariance(measured))
        with self.assertRaisesRegex(ValueError, "covariance must be one of"):
            ModelFit(fake_file(measured), covariance="banded")
