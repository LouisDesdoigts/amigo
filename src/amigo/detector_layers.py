import jax
import jax.numpy as np
import dLux as dl
import dLuxWebbpsf as dlw
import dLux.utils as dlu
from jax import vmap


# Amplifier/ramp modelling
def model_amplifier(coeffs, axis=0):
    """
    Models the amplifier noise as a polynomial along one axis of the detector.
    Assumes Detector is square and coeffs has shape (npix, order + 1).
    """
    # Evaluation function
    xs = np.linspace(-1, 1, coeffs.shape[0])
    eval_fn = lambda coeffs: np.polyval(coeffs, xs)

    # Vectorise over columns and groups in the data
    vals = vmap(eval_fn, 0)(coeffs)

    if axis == 0:
        return np.rot90(vals)
    return vals


def model_ramp(psf, ngroups):
    """Applies an 'up the ramp' model of the input 'optical' PSF. Input PSF.data
    should have shape (npix, npix) and return shape (ngroups, npix, npix)"""
    lin_ramp = (np.arange(ngroups) + 1) / ngroups
    return psf[None, ...] * lin_ramp[..., None, None]


def model_dark_current(ramp, dark_current):
    """Models the dark current as a constant background value added cumulatively to
    each group. For now we assume that the dark current is a float."""
    dark_ramp = dark_current * (np.arange(len(ramp)) + 1)
    return ramp + dark_ramp[..., None, None]


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


#
