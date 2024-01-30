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


class ApplyPRF(dl.layers.detector_layers.DetectorLayer):
    """
    Note! Applies the downsample
    """

    FF: jax.Array
    PRF: jax.Array

    def __init__(self, FF, PRF):
        self.FF = FF
        self.PRF = PRF

    def apply(self, PSF):
        # Get shapes and reshape data
        osamp = self.PRF.shape[0]
        npix = PSF.data.shape[0] // osamp
        bc_psf = PSF.data.reshape((npix, osamp, npix, osamp))

        # Apply intra and inter-pixel sensitivities
        psf = (bc_psf * self.PRF[None, :, None, :]).reshape(PSF.data.shape)
        psf = self.FF * dlu.downsample(psf, osamp, mean=False)
        return PSF.set("data", psf)


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
