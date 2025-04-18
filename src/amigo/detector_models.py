import pkg_resources as pkg
import jax.numpy as np
import dLux as dl
import dLux.utils as dlu
import jax
from jax.scipy.stats import multivariate_normal
from .misc import interp
import equinox as eqx


def quadratic_SRF(a, oversample, norm=True):
    """
    norm will normalise the SRF to have a mean of 1
    """
    coords = dlu.pixel_coords(oversample, 2)
    quad = 1 - np.sum((a * coords) ** 2, axis=0)
    if norm:
        quad -= quad.mean() - 1
    return quad


def broadcast_subpixel(pixels, subpixel):
    npix = pixels.shape[1]
    oversample = subpixel.shape[0]
    bc_sens_map = subpixel[None, :, None, :] * pixels[:, None, :, None]
    return bc_sens_map.reshape((npix * oversample, npix * oversample))


class PixelSensitivities(dl.layers.detector_layers.DetectorLayer):
    FF: jax.Array
    SRF: jax.Array
    method: str = eqx.field(static=True)
    oversample: int = eqx.field(static=True)

    def __init__(self, FF, SRF=0.0, method="quad", oversample=4):
        if method not in ["quad", "pixel"]:
            raise ValueError(f"Method {method} not recognised")
        if method == "pixel":
            if SRF.ndim != 2:
                raise ValueError("Pixel method requires a 2D SRF array")

            i, j = SRF.shape
            if i != j:
                raise ValueError("Pixel method requires a square SRF array")

            if i != oversample:
                raise ValueError(
                    "Pixel method requires a square SRF array with the same size as"
                    " the oversample"
                )

        self.FF = np.array(FF, float)
        self.SRF = np.array(SRF, float)
        self.method = str(method)
        self.oversample = int(oversample)

    @property
    def sensitivity_map(self):
        if self.method == "quad":
            subpixel_fn = quadratic_SRF(self.SRF, self.oversample)
        else:
            subpixel_fn = self.SRF

        return broadcast_subpixel(self.FF, subpixel_fn)

    def apply(self, PSF):
        return PSF * self.sensitivity_map


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

    r: np.ndarray

    def __init__(self, r=0.02, **kwargs):
        super().__init__(**kwargs)
        self.r = np.array(r, float)

    def generate_kernel(self, pixel_scale):
        cov = np.square(self.r) * np.eye(2)
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


class SUB80Detector(LayeredDetector):
    ramp: None
    # sensitivity: PixelSensitivities
    oversample: int = eqx.field(static=True)

    def __init__(
        self,
        ramp_model=None,
        oversample=3,
        npixels_in=80,
        rot_angle=+0.56126717,
        anisotropy=1.00765,
        # SRF=0.05,
        FF=None,
    ):
        # Ramp model
        self.ramp = ramp_model
        self.oversample = int(oversample)

        # Load the FF
        if FF is None:
            file_path = pkg.resource_filename(__name__, "data/SUB80_flatfield.npy")
            FF = np.load(file_path)
            if npixels_in != 80:
                pad = (npixels_in - 80) // 2
                FF = np.pad(FF, pad, constant_values=1)

        # self.sensitivity = PixelSensitivities(FF, SRF, oversample=oversample)

        self.layers = dlu.list2dictionary(
            [
                # ("jitter", AsymmetricJitter(kernel_size=11, kernel_oversample=5)),
                ("jitter", GaussianJitter(r=0.0214, kernel_size=11, kernel_oversample=5)),
                ("resampler", Resample(rotation=rot_angle, anisotropy=anisotropy)),
            ],
            ordered=True,
        )

    def __getattr__(self, key):
        if hasattr(self.ramp, key):
            return getattr(self.ramp, key)
        # if hasattr(self.sensitivity, key):
        #     return getattr(self.sensitivity, key)
        # super().__getattr__(self)
        if key in self.layers.keys():
            return self.layers[key]
        for layer in list(self.layers.values()):
            if hasattr(layer, key):
                return getattr(layer, key)

        raise AttributeError(f"SUB80Detector.ODE has no attribute {key}")

    # def evolve_ramp(self, illuminance, ngroups, z_point, badpix):

    #     if self.ramp is None:
    #         # sensitivity = self.sensitivity_map
    #         illuminance = dlu.downsample(illuminance, self.oversample, mean=False)
    #         return model_ramp(illuminance, ngroups)

    #     elif isinstance(self.ramp, GainDiffusionRamp):
    #         sensitivity = self.sensitivity_map
    #         ramp, latent_paths = self.ramp.evolve_ramp(illuminance, ngroups, sensitivity)
    #         return ramp, latent_paths

    #     # This should be manually coded for the PolyPredictive ramp model, with the FF
    #     # and SRF included in the inputs.
    #     else:
    #         ramp, bleeds = self.ramp.evolve_ramp(illuminance, ngroups, z_point, None, badpix)
    #         return ramp, bleeds

    # else:
    #     raise NotImplementedError("No implementation for this ramp type")


# class SUB80Ramp(dl.detectors.BaseDetector):
#     ramp: None
#     sensitivity: PixelSensitivities
#     # jitter: AsymmetricJitter
#     # resampler: Resample

#     def __init__(
#         self,
#         ramp_model=None,
#         # oversample=4,
#         # npixels_in=80,
#         # rot_angle=+0.56126717,
#         # anisotropy=1.0,
#         # jitter_amplitude=0.02,  # as
#         # asymmetric_jitter=True,
#         SRF=1e-3,
#         FF=None,
#     ):

#         self.ramp_model = ramp_model
#         self.jitter.AsymmetricJitter(kernel_size=11, kernel_oversample=5)
#         self.resampler.Resample(rotation=rot_angle, anisotropy=anisotropy)

#         # Load the FF
#         if FF is None:
#             file_path = pkg.resource_filename(__name__, "data/SUB80_flatfield.npy")
#             FF = np.load(file_path)
#             if npixels_in != 80:
#                 pad = (npixels_in - 80) // 2
#                 FF = np.pad(FF, pad, constant_values=1)

#         self.sensitivity.PixelSensitivities(FF, SRF, oversample=oversample)

#     def __getattr__(self, a):
#         pass


#     def evolve_ramp(self, illuminance, ngroups):
#         if isinstance(self.ramp, GainDiffusionRamp):
#             sensitivity = self.sensitivity_map
#             ramp, latent_paths = self.evolve_ramp(illuminance, ngroups, sensitivity)

#         else:
#             raise NotImplementedError("No implementation for this ramp type")


# if asymmetric_jitter:
# jitter = AsymmetricJitter(kernel_size=11, kernel_oversample=5)
# else:
#     jitter = GaussianJitter(0.02, kernel_size=11, kernel_oversample=3)

# PixelSensitivities(FF, SRF, oversample=oversample)

# layers = [
# ("jitter", AsymmetricJitter(kernel_size=11, kernel_oversample=5)),
# ("resampler", Resample(rotation=rot_angle, anisotropy=anisotropy)),
# ("sensitivity", PixelSensitivities(FF, SRF, oversample=oversample)),
# ("pixel_anisotropy", PixelAnisotropy(anisotropy)),
# ]

# if SRF is None:
#     SRF = np.ones((oversample, oversample))

# layers.append()

# self.layers = dlu.list2dictionary(layers, ordered=True)

# def apply_linear(self, psf):
#     for layer in list(self.layers.values()):
#         if layer is None:
#             continue
#         psf = layer.apply(psf)
#     return psf

# def apply_ramp(self, psf):

#     if isinstance(self.ramp, None):
#         illuminance = dlu.downsample(illuminance, model.optics.oversample, mean=False)
#         ramp = psf.set("data", model_ramp(illuminance, self.ngroups))

#     ramp, latent_paths = self.ramp.evolve_ramp(illuminance, ngroups, sensitivity_map)
#         if layer is None:
#             continue
#         psf = layer.apply(psf)
#     return psf
