import pkg_resources as pkg
import jax.numpy as np
import dLux as dl
import dLux.utils as dlu
from .jitter import GaussianJitter
from .detector_layers import (
    PixelAnisotropy,
    ApplySensitivities,
    # DarkCurrent,
    # IPC,
    # Amplifier,
    Rotate,
    # model_ramp,
    # Ramp,
)


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
