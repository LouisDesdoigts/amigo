import jax
import jax.numpy as np
import dLux as dl
import dLuxWebbpsf as dlw
import dLux.utils as dlu
from jax import vmap
from jax.scipy.signal import convolve
from jax import Array
from dLuxWebbpsf.utils.interpolation import _map_coordinates

# from .modelling import model_dark_current, model_ramp


def arr2pix(coords, pscale=1):
    n = coords.shape[-1]
    shift = (n - 1) / 2
    return pscale * (coords - shift)


def pix2arr(coords, pscale=1):
    n = coords.shape[-1]
    shift = (n - 1) / 2
    return (coords / pscale) + shift


# Amplifier/ramp modelling
def model_amplifier(coeffs, axis=0):
    """
    Models the amplifier noise as a polynomial along one axis of the detector.
    Assumes Detector is square and coeffs has shape (npix, order + 1).
    """

    def read_fn(coeffs):
        # Evaluation function
        xs = np.linspace(-1, 1, coeffs.shape[0])
        eval_fn = lambda coeffs: np.polyval(coeffs, xs)

        # Vectorise over each column
        vals = vmap(eval_fn, 0)(coeffs)

        if axis == 0:
            return np.rot90(vals)
        return vals

    # vmap over each group
    return vmap(read_fn)(coeffs)


def model_ramp(psf, ngroups):
    """Applies an 'up the ramp' model of the input 'optical' PSF. Input PSF.data
    should have shape (npix, npix) and return shape (ngroups, npix, npix)"""
    lin_ramp = (np.arange(ngroups) + 1) / ngroups
    return psf[None, ...] * lin_ramp[..., None, None]


def model_dark_current(dark_current, ngroups):
    """Models the dark current as a constant background value added cumulatively to
    each group. For now we assume that the dark current is a float."""
    return (dark_current * (np.arange(ngroups) + 1))[..., None, None]


class ApplySensitivities(dl.layers.detector_layers.DetectorLayer):

    FF: jax.Array
    SRF: jax.Array

    def __init__(
        self,
        FF,
        SRF,
    ):
        self.FF = FF
        self.SRF = SRF

    @property
    def sensitivity_map(self):
        oversample = self.SRF.shape[0]
        npix = self.FF.shape[1]
        bc_sens_map = self.SRF[None, :, None, :] * self.FF[:, None, :, None]
        return bc_sens_map.reshape((npix * oversample, npix * oversample))

    def apply(self, PSF):
        return PSF * self.sensitivity_map


class Rotate(dl.layers.detector_layers.DetectorLayer):
    """
    Applies cubic spline interpolator for rotation of the PSF
    """

    angle: float

    def __init__(self, angle):
        self.angle = angle

    def apply(self, PSF):
        psf = dlw.utils.rotate(PSF.data, dlu.deg2rad(self.angle), order=3)
        return PSF.set("data", psf)


class PixelAnisotropy(dl.layers.detector_layers.DetectorLayer):
    transform: dl.CoordTransform
    order: int

    def __init__(self, order=3):
        self.transform = dl.CoordTransform(compression=np.ones(2))
        self.order = int(order)

    def __getattr__(self, key):
        if hasattr(self.transform, key):
            return getattr(self.transform, key)
        raise AttributeError(f"PixelAnisotropy has no attribute {key}")

    def apply(self, PSF):
        npix = PSF.data.shape[0]
        transformed = self.transform.apply(dlu.pixel_coords(npix, npix * PSF.pixel_scale))
        coords = np.roll(pix2arr(transformed, PSF.pixel_scale), 1, axis=0)
        interp_fn = lambda x: _map_coordinates(x, coords, order=3, mode="constant", cval=0.0)
        return PSF.set("data", interp_fn(PSF.data))


class Ramp(dl.PSF):
    pass


class DownsampleRamp(dl.detector_layers.Downsample):

    def apply(self, ramp):
        dsample_fn = lambda x: dlu.downsample(x, self.kernel_size, mean=False)
        return ramp.set("data", vmap(dsample_fn)(ramp))


class EmptyLayer(dl.detector_layers.Downsample):

    def apply(self, ramp):
        return ramp


class IPC(dl.detector_layers.DetectorLayer):
    ipc: Array

    def __init__(self, ipc):
        self.ipc = np.array(ipc, float)

    def apply(self, ramp):
        conv_fn = lambda x: convolve(x, self.ipc, mode="same")
        return ramp.set("data", vmap(conv_fn)(ramp.data))


class Amplifier(dl.detector_layers.DetectorLayer):
    one_on_fs: Array

    def __init__(self, one_on_fs):
        self.one_on_fs = np.array(one_on_fs, float)

    def apply(self, ramp):
        return ramp.add("data", model_amplifier(self.one_on_fs))


class DarkCurrent(dl.detector_layers.DetectorLayer):
    dark_current: Array

    def __init__(self, dark_current):
        self.dark_current = np.array(dark_current, float)

    def apply(self, ramp):
        dark_current = model_dark_current(self.dark_current, len(ramp.data))
        return ramp.add("data", dark_current)


# class BuildRamp(dl.detector_layers.DetectorLayer):
#     ngroups: int

#     def __init__(self, ngroups):
#         self.ngroups = int(ngroups)

#     def apply(self, psf):
#         return Ramp(model_ramp(psf.data, self.ngroups), psf.pixel_scale)
