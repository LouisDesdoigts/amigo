import pkg_resources as pkg
import jax.numpy as np
import dLux as dl
import dLux.utils as dlu
import jax
from jax.scipy.stats import multivariate_normal
from .misc import interp
import equinox as eqx

# import interpax as ipx


# def interp(image, knots, sample_coords, extrap=0.0):
#     xs, ys = knots[0, 0, :], knots[1, :, 0]
#     xpts, ypts = sample_coords.reshape(2, -1)
#     return ipx.interp2d(ypts, xpts, ys, xs, image, method="cubic2", extrap=extrap).reshape(
#         sample_coords[0].shape
#     )


def quadratic_SRF(a, oversample, norm=True):
    """
    norm will normalise the SRF to have a mean of 1
    """
    coords = dlu.pixel_coords(oversample, 2)
    quad = 1 - np.sum((a * coords) ** 2, axis=0)
    if norm:
        quad -= quad.mean() - 1
    return quad


class ApplySensitivities(dl.layers.detector_layers.DetectorLayer):

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
        npix = self.FF.shape[1]

        if self.method == "quad":
            SRF = quadratic_SRF(self.SRF, self.oversample)
        else:
            SRF = self.SRF

        bc_sens_map = SRF[None, :, None, :] * self.FF[:, None, :, None]
        return bc_sens_map.reshape((npix * self.oversample, npix * self.oversample))

    def apply(self, PSF):
        return PSF * self.sensitivity_map


class Rotate(dl.layers.detector_layers.DetectorLayer):
    rotation: float

    def __init__(self, rotation):
        self.rotation = np.array(rotation, float)

    def apply(self, PSF):
        coords = dlu.pixel_coords(PSF.data.shape[0], 2)
        rot_coords = dlu.rotate_coords(coords, dlu.deg2rad(self.rotation))
        return PSF.set("data", interp(PSF.data, coords, rot_coords, "cubic2"))


class PixelAnisotropy(dl.layers.detector_layers.DetectorLayer):
    anisotropy: np.ndarray

    def __init__(self, anisotropy=1.0):
        self.anisotropy = np.array(anisotropy, float)

    def apply(self, PSF):
        npix = PSF.data.shape[0]
        coords = dlu.pixel_coords(npix, npix * PSF.pixel_scale)
        new_coords = coords * np.array([1.0, self.anisotropy])[:, None, None]
        return PSF.set("data", interp(PSF.data, coords, new_coords, "cubic2"))


class GaussianJitter(dl.layers.detector_layers.DetectorLayer):
    """r has units of arcseconds"""

    r: float
    kernel_size: int
    kernel_oversample: int

    def __init__(self, r, kernel_size=9, kernel_oversample=3):
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer")

        self.kernel_size = int(kernel_size)
        self.r = np.asarray(r, float)
        self.kernel_oversample = kernel_oversample

    def apply(self, psf):
        kernel = self.generate_kernel(dlu.rad2arcsec(psf.pixel_scale))
        return psf.convolve(kernel)

    def generate_kernel(self, pixel_scale):
        # r_mas = self.r

        # Generate distribution
        extent = pixel_scale * self.kernel_size
        x = np.linspace(0, extent, self.kernel_oversample * self.kernel_size) - 0.5 * extent
        xs, ys = np.meshgrid(x, x)

        #
        pos = np.dstack((xs, ys))
        mean = np.array([0.0, 0.0])
        cov = np.square(self.r) * np.eye(2)

        kernel = dlu.downsample(
            multivariate_normal.pdf(pos, mean=mean, cov=cov),
            self.kernel_oversample,
        )

        return kernel / np.sum(kernel)


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


class LinearDetectorModel(LayeredDetector):

    def __init__(
        self,
        oversample=4,
        npixels_in=80,
        rot_angle=+0.56126717,
        anisotropy=1.0,
        jitter_amplitude=1.12e-3,  # as
        SRF=1e-3,
        FF=None,
    ):
        layers = [
            ("rotate", Rotate(rot_angle)),
            ("pixel_anisotropy", PixelAnisotropy(anisotropy)),
            ("jitter", GaussianJitter(jitter_amplitude, kernel_size=11, kernel_oversample=3)),
        ]

        # Load the FF
        if FF is None:
            file_path = pkg.resource_filename(__name__, "data/SUB80_flatfield.npy")
            FF = np.load(file_path)
            if npixels_in != 80:
                pad = (npixels_in - 80) // 2
                FF = np.pad(FF, pad, constant_values=1)

        # if SRF is None:
        #     SRF = np.ones((oversample, oversample))

        layers.append(("sensitivity", ApplySensitivities(FF, SRF, oversample=oversample)))

        self.layers = dlu.list2dictionary(layers, ordered=True)
