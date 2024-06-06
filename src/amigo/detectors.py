import pkg_resources as pkg
import jax.numpy as np
import dLux as dl
import dLux.utils as dlu

from .jitter import GaussianJitter
from .detector_layers import (
    PixelAnisotropy,
    ApplySensitivities,
    DarkCurrent,
    IPC,
    Amplifier,
    Rotate,
)


def model_ramp(psf, ngroups):
    """Applies an 'up the ramp' model of the input 'optical' PSF. Input PSF.data
    should have shape (npix, npix) and return shape (ngroups, npix, npix)"""
    lin_ramp = (np.arange(ngroups) + 1) / ngroups
    return psf[None, ...] * lin_ramp[..., None, None]


def model_dark_current(dark_current, ngroups):
    """Models the dark current as a constant background value added cumulatively to
    each group. For now we assume that the dark current is a float."""
    return (dark_current * (np.arange(ngroups) + 1))[..., None, None]


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


# class EffectiveDetectorModel(dl.detectors.BaseDetector):
#     linear_model: LayeredDetector
#     non_linear_model: LayeredDetector
#     read_model: LayeredDetector

#     def __init__(self, linear_model, non_linear_model, read_model):
#         self.linear_model = linear_model
#         self.non_linear_model = non_linear_model
#         self.read_model = read_model

#     def __getattr__(self, key: str):
#         if hasattr(self.linear_model, key):
#             return getattr(self.linear_model, key)
#         if hasattr(self.non_linear_model, key):
#             return getattr(self.non_linear_model, key)
#         if hasattr(self.read_model, key):
#             return getattr(self.read_model, key)
#         raise AttributeError(f"{self.__class__.__name__} has no attribute " f"{key}.")


class LinearDetectorModel(LayeredDetector):

    def __init__(
        self,
        oversample=4,
        npixels_in=80,
        rot_angle=-0.56126717,
        SRF=None,
        FF=None,
        jitter=True,
        anisotropy=True,
    ):
        layers = [("rotate", Rotate(rot_angle))]

        if anisotropy:
            compression = np.array([0.99580676, 1.00343162])
            anisotropy = PixelAnisotropy(order=3).set("compression", compression)
            layers.append(("anisotropy", anisotropy))

        if jitter:
            layers.append(("jitter", GaussianJitter(2.5e-7, kernel_size=19, kernel_oversample=3)))

        # Load the FF
        if FF is None:
            file_path = pkg.resource_filename(__name__, "data/SUB80_flatfield.npy")
            FF = np.load(file_path)
            if npixels_in != 80:
                pad = (npixels_in - 80) // 2
                FF = np.pad(FF, pad, constant_values=1)

        if SRF is None:
            SRF = np.ones((oversample, oversample))

        layers.append(("sensitivity", ApplySensitivities(FF, SRF)))

        self.layers = dlu.list2dictionary(layers, ordered=True)


class SimpleRamp(dl.detectors.BaseDetector):

    def apply(self, psf, flux, exposure, oversample):
        return model_ramp(dlu.downsample(psf * flux, oversample, mean=False), exposure.ngroups)

    def model(self, psf):
        raise NotImplementedError


class ReadModel(LayeredDetector):

    def __init__(
        self,
        dark_current=0.0,
        ipc=True,
        one_on_fs=np.zeros((2, 80, 2)),
    ):
        layers = []
        layers.append(("read", DarkCurrent(dark_current)))
        if ipc:
            file_path = pkg.resource_filename(__name__, "data/SUB80_ipc.npy")
            ipc = np.load(file_path)
        else:
            ipc = np.array([[1.0]])
        layers.append(("IPC", IPC(ipc)))
        layers.append(("amplifier", Amplifier(one_on_fs)))
        self.layers = dlu.list2dictionary(layers, ordered=True)
