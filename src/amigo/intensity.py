"""Spectral intensity images in the optical frame."""

import dLux as dl
import jax.numpy as np
from jax import Array


class Intensity(dl.PSF):
    """A PSF carrying wavelengths, with no optical amplitude or phase.

    ``data`` contains wavelength-weighted images with shape (..., y, x).
    ``wavelength`` is in metres; ``pixel_scale`` is in radians per pixel.
    Both metadata arrays follow the leading image dimensions.
    """

    wavelength: Array

    def __init__(self, data, wavelength, pixel_scale):
        self.wavelength = np.asarray(wavelength, dtype=float)
        super().__init__(data, np.broadcast_to(pixel_scale, self.wavelength.shape))

    @property
    def psf(self):
        """Expose intensity through the existing wavefront/visibility interface."""
        return self.data

    def to_psf(self, axis=0):
        """Sum weighted images into a PSF, over wavelengths by default.

        For multiple sources, ``axis=(0, 1)`` sums sources and wavelengths.
        The images must share the same angular pixel scale.
        """
        return dl.PSF(self.data.sum(axis), self.pixel_scale.mean(axis))
