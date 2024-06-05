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
    Ramp,
    model_ramp,
)


class SUB80Ramp(dl.detectors.LayeredDetector):

    def __init__(
        self,
        EDM=None,
        angle=-0.56126717,
        oversample=4,
        SRF=None,
        FF=None,
        npixels_in=80,
        anisotropy=True,
        jitter=True,
        dark_current=0.0,
        ipc=True,
        one_on_fs=np.zeros((2, 80, 2)),
    ):
        layers = [("rotate", Rotate(angle))]

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

        if EDM is None:
            EDM = NullEDM(oversample)
        layers.append(("EDM", EDM))

        if ipc:
            file_path = pkg.resource_filename(__name__, "data/SUB80_ipc.npy")
            ipc = np.load(file_path)
        else:
            ipc = np.array([[1.0]])

        layers.append(("read", DarkCurrent(dark_current)))
        layers.append(("IPC", IPC(ipc)))
        layers.append(("amplifier", Amplifier(one_on_fs)))

        self.layers = dlu.list2dictionary(layers, ordered=True)

    def __getattr__(self, key: str):
        if key in self.layers.keys():
            return self.layers[key]
        for layer in list(self.layers.values()):
            if hasattr(layer, key):
                return getattr(layer, key)
        raise AttributeError(f"{self.__class__.__name__} has no attribute " f"{key}.")


class NullEDM(dl.detector_layers.DetectorLayer):
    downsample: int
    ngroups: int
    flux: float
    filter: str

    def __init__(self, downsample=4, ngroups=2, flux=1.0, filter="F430M"):
        self.ngroups = int(ngroups)
        self.flux = np.asarray(flux, float)
        self.downsample = int(downsample)
        self.filter = str(filter)

    def apply(self, psf):
        downsampled = dlu.downsample(psf.data * self.flux, self.downsample, mean=False)
        return Ramp(model_ramp(downsampled, self.ngroups), psf.pixel_scale)
