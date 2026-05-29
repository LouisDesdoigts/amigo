# import pkg_resources as pkg
import jax.numpy as np
from jax.scipy.stats import multivariate_normal
import dLux as dl
import dLux.utils as dlu
from dLux.detectors import LayeredDetector
from dLux.layers.detector_layers import DetectorLayer
from .misc import interp
import equinox as eqx


class Resample(DetectorLayer):
    rotation: float
    anisotropy: np.ndarray

    def __init__(self, rotation=0.0, anisotropy=1.00765):
        self.rotation = np.array(rotation, float)
        self.anisotropy = np.array(anisotropy, float)

    def __call__(self, psf):
        angle = dlu.deg2rad(self.rotation)
        coords = dlu.pixel_coords(psf.data.shape[0], 2)
        rot_coords = dlu.rotate_coords(coords, angle)
        sample_coords = rot_coords * np.array([1.0, self.anisotropy])[:, None, None]
        # TODO: Test different interpolation methods
        return psf.set("data", interp(psf.data, coords, sample_coords, "cubic2"))


class LinearDetector(LayeredDetector):
    def __init__(
        self,
        rot_angle=+0.56126717,
        anisotropy=1.00765,
        jitter=0.0214,  # in arcseconds
        kernel_size=11,
        kernel_osamp=5,
    ):
        # NOTE: converting jitter sigma into pixels, assuming an oversample of 3
        jitter_pixels = jitter / (0.065524085 / 3)  # arcsec / (arcsec/pixel)
        
        super().__init__(
            [
                ("jitter_model", dl.ApplyJitter(jitter_pixels, kernel_size, kernel_osamp)),
                ("resampler", Resample(rotation=rot_angle, anisotropy=anisotropy)),
            ]
        )


# def gaussian_kernel(kernel_size, cov, pixel_scale, oversample):
#     # Generate distribution
#     extent = pixel_scale * kernel_size
#     x = np.linspace(0, extent, oversample * kernel_size) - 0.5 * extent
#     xs, ys = np.meshgrid(x, x)

#     #
#     pos = np.dstack((xs, ys))
#     mean = np.array([0.0, 0.0])

#     kernel = dlu.downsample(
#         multivariate_normal.pdf(pos, mean=mean, cov=cov),
#         oversample,
#     )

#     return kernel / np.sum(kernel)


# class BaseJitter(dl.layers.detector_layers.DetectorLayer):
#     """Base jitter class, ensures units are arcseconds"""

#     kernel_size: int = eqx.field(static=True)
#     kernel_oversample: int = eqx.field(static=True)

#     def __init__(self, kernel_size=9, kernel_oversample=3):
#         if kernel_size % 2 == 0:
#             raise ValueError("kernel_size must be an odd integer")
#         self.kernel_size = int(kernel_size)
#         self.kernel_oversample = kernel_oversample

#     def apply(self, psf):
#         """Convert the pixel scale to arcseconds and convolve"""
#         kernel = self.generate_kernel(dlu.rad2arcsec(psf.pixel_scale))
#         return psf.convolve(kernel)


# class GaussianJitter(BaseJitter):
#     """Has units of arcseconds"""

#     jitter: np.ndarray

#     def __init__(self, jiiter=0.02, **kwargs):
#         super().__init__(**kwargs)
#         self.jitter = np.array(jiiter, float)

#     def generate_kernel(self, pixel_scale):
#         cov = np.square(self.jitter) * np.eye(2)
#         return gaussian_kernel(self.kernel_size, cov, pixel_scale, self.kernel_oversample)


# class AsymmetricJitter(BaseJitter):
#     """Has units of arcseconds"""

#     # TODO: Change rx, ry to stdevs
#     rx: float
#     ry: float
#     corr: float

#     def __init__(self, rx=0.02, ry=0.02, corr=0.0, **kwargs):
#         super().__init__(**kwargs)
#         self.rx = np.asarray(rx, float)
#         self.ry = np.asarray(ry, float)
#         self.corr = np.asarray(corr, float)

#     def generate_kernel(self, pixel_scale):
#         cov = np.array(
#             [
#                 [self.rx**2, self.corr],
#                 [self.corr, self.ry**2],
#             ]
#         )
#         return gaussian_kernel(self.kernel_size, cov, pixel_scale, self.kernel_oversample)
