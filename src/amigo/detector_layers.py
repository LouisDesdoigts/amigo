import jax
import jax.numpy as np
import dLux as dl
import dLuxWebbpsf as dlw
import dLux.utils as dlu
from dLuxWebbpsf.utils.interpolation import _map_coordinates


def arr2pix(coords, pscale=1):
    n = coords.shape[-1]
    shift = (n - 1) / 2
    return pscale * (coords - shift)


def pix2arr(coords, pscale=1):
    n = coords.shape[-1]
    shift = (n - 1) / 2
    return (coords / pscale) + shift


# def model_ramp(psf, ngroups):
#     """Applies an 'up the ramp' model of the input 'optical' PSF. Input PSF.data
#     should have shape (npix, npix) and return shape (ngroups, npix, npix)"""
#     lin_ramp = (np.arange(ngroups) + 1) / ngroups
#     return psf[None, ...] * lin_ramp[..., None, None]


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


# class Ramp(dl.PSF):
#     pass


# class DownsampleRamp(dl.detector_layers.Downsample):

#     def apply(self, ramp):
#         dsample_fn = lambda x: dlu.downsample(x, self.kernel_size, mean=False)
#         return ramp.set("data", vmap(dsample_fn)(ramp))
