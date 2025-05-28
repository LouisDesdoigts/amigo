# import pkg_resources as pkg
import jax.numpy as np
import dLux as dl
import dLux.utils as dlu
from jax.scipy.stats import multivariate_normal
from .misc import interp
import equinox as eqx

# from .ramp_models import NonLinearRamp


class Resample(dl.layers.detector_layers.DetectorLayer):
    rotation: float
    anisotropy: np.ndarray

    def __init__(self, rotation=0.0, anisotropy=1.00765):
        self.rotation = np.array(rotation, float)
        self.anisotropy = np.array(anisotropy, float)

    def apply(self, PSF):
        angle = dlu.deg2rad(self.rotation)
        coords = dlu.pixel_coords(PSF.data.shape[0], 2)
        rot_coords = dlu.rotate_coords(coords, angle)
        sample_coords = rot_coords * np.array([1.0, self.anisotropy])[:, None, None]
        # TODO: Test different interpolation methods
        return PSF.set("data", interp(PSF.data, coords, sample_coords, "cubic2"))


def gaussian_kernel(kernel_size, cov, pixel_scale, oversample):
    # Generate distribution
    extent = pixel_scale * kernel_size
    x = np.linspace(0, extent, oversample * kernel_size) - 0.5 * extent
    xs, ys = np.meshgrid(x, x)

    #
    pos = np.dstack((xs, ys))
    mean = np.array([0.0, 0.0])

    kernel = dlu.downsample(
        multivariate_normal.pdf(pos, mean=mean, cov=cov),
        oversample,
    )

    return kernel / np.sum(kernel)


class BaseJitter(dl.layers.detector_layers.DetectorLayer):
    """Base jitter class, ensures units are arcseconds"""

    kernel_size: int = eqx.field(static=True)
    kernel_oversample: int = eqx.field(static=True)

    def __init__(self, kernel_size=9, kernel_oversample=3):
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer")
        self.kernel_size = int(kernel_size)
        self.kernel_oversample = kernel_oversample

    def apply(self, psf):
        """Convert the pixel scale to arcseconds and convolve"""
        kernel = self.generate_kernel(dlu.rad2arcsec(psf.pixel_scale))
        return psf.convolve(kernel)


class GaussianJitter(BaseJitter):
    """Has units of arcseconds"""

    jitter: np.ndarray

    def __init__(self, jiiter=0.02, **kwargs):
        super().__init__(**kwargs)
        self.jitter = np.array(jiiter, float)

    def generate_kernel(self, pixel_scale):
        cov = np.square(self.jitter) * np.eye(2)
        return gaussian_kernel(self.kernel_size, cov, pixel_scale, self.kernel_oversample)


class AsymmetricJitter(BaseJitter):
    """Has units of arcseconds"""

    # TODO: Change rx, ry to stdevs
    rx: float
    ry: float
    corr: float

    def __init__(self, rx=0.02, ry=0.02, corr=0.0, **kwargs):
        super().__init__(**kwargs)
        self.rx = np.asarray(rx, float)
        self.ry = np.asarray(ry, float)
        self.corr = np.asarray(corr, float)

    def generate_kernel(self, pixel_scale):
        cov = np.array(
            [
                [self.rx**2, self.corr],
                [self.corr, self.ry**2],
            ]
        )
        return gaussian_kernel(self.kernel_size, cov, pixel_scale, self.kernel_oversample)


class LayeredDetector(dl.detectors.LayeredDetector):

    def __getattr__(self, key: str):
        if key in self.layers.keys():
            return self.layers[key]
        for layer in list(self.layers.values()):
            if hasattr(layer, key):
                return getattr(layer, key)
        raise AttributeError(f"{self.__class__.__name__} has no attribute " f"{key}.")

    def apply(self, psf):
        for layer in list(self.layers.values()):
            if layer is None:
                continue
            psf = layer.apply(psf)
        return psf


class LinearDetector(LayeredDetector):

    def __init__(
        self,
        rot_angle=+0.56126717,
        anisotropy=1.00765,
        jitter=0.0214,
    ):
        self.layers = dlu.list2dictionary(
            [
                ("jitter_model", GaussianJitter(jitter, kernel_size=11, kernel_oversample=5)),
                ("resampler", Resample(rotation=rot_angle, anisotropy=anisotropy)),
            ],
            ordered=True,
        )

    def __getattr__(self, key):
        # if hasattr(self.ramp_model, key):
        #     return getattr(self.ramp_model, key)
        if key in self.layers.keys():
            return self.layers[key]
        for layer in list(self.layers.values()):
            if hasattr(layer, key):
                return getattr(layer, key)

        raise AttributeError(f"LinearDetector has no attribute {key}")
