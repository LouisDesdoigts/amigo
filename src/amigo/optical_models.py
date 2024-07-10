import jax
import jax.numpy as np
import dLux as dl
import dLux.utils as dlu
from jax import Array
from .mask_models import DynamicAMIStaticAbb


class AMIOptics(dl.optical_systems.AngularOpticalSystem):

    def __init__(
        self,
        radial_orders=4,
        pupil_mask=None,
        opd=None,
        normalise=True,
        coherence_orders=4,
        psf_npixels=80,
        oversample=4,
        pixel_scale=0.065524085,
        diameter=6.603464,
        wf_npixels=1024,
        f2f=0.80,
        oversize=1.1,
        polike=False,
        unique_holes=False,
    ):
        """Free space locations can be 'before', 'after'"""
        self.wf_npixels = wf_npixels
        self.diameter = diameter
        self.psf_npixels = psf_npixels
        self.oversample = oversample
        self.psf_pixel_scale = pixel_scale

        layers = []
        layers += [("InvertY", dl.Flip(0))]

        if pupil_mask is None:
            pupil_mask = DynamicAMIStaticAbb(
                diameter=diameter,
                npixels=wf_npixels,
                f2f=f2f,
                normalise=normalise,
                aberration_orders=radial_orders,
                amplitude_orders=coherence_orders,
                oversize=oversize,
                polike=polike,
                unique_holes=unique_holes,
            )
        layers += [("pupil_mask", pupil_mask)]
        self.layers = dlu.list2dictionary(layers, ordered=True)


class AMIOpticsFresnel(dl.CartesianOpticalSystem):
    """
    fl = pixel_scale_m / pixel_scale_rad -> NIRISS pixel scales are 18um  and
    0.0656 arcsec respectively, so fl ~= 56.6m
    """

    defocus: jax.Array  # metres, is this actually um??

    def __init__(
        self,
        radial_orders=4,
        pupil_mask=None,
        opd=None,
        normalise=True,
        coherence_orders=4,
        psf_npixels=80,
        oversample=4,
        pixel_scale=0.065524085,  # angular pixel scale in arcsec
        diameter=6.603464,
        wf_npixels=1024,
        f2f=0.80,
        oversize=1.1,
        polike=False,
        unique_holes=False,
    ):
        """Free space locations can be 'before', 'after'"""
        self.wf_npixels = wf_npixels
        self.diameter = diameter
        self.psf_npixels = psf_npixels
        self.oversample = oversample

        # Cartesian pixel scale is 18 microns
        self.psf_pixel_scale = np.array(18.0, float)  # 18 microns
        self.focal_length = self.psf_pixel_scale * 1e-6 / dlu.arcsec2rad(pixel_scale)
        self.defocus = np.array(0.0, float)

        layers = []
        layers += [("InvertY", dl.Flip(0))]

        if pupil_mask is None:
            pupil_mask = DynamicAMIStaticAbb(
                diameter=diameter,
                npixels=wf_npixels,
                f2f=f2f,
                normalise=normalise,
                aberration_orders=radial_orders,
                amplitude_orders=coherence_orders,
                oversize=oversize,
                polike=polike,
                unique_holes=unique_holes,
            )
        layers += [("pupil_mask", pupil_mask)]
        self.layers = dlu.list2dictionary(layers, ordered=True)

    def propagate_mono(
        self: dl.optical_systems.OpticalSystem,
        wavelength: jax.Array,
        offset: jax.Array = np.zeros(2),
        return_wf: bool = False,
    ) -> jax.Array:
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : float, metres
            The wavelength of the wavefront to propagate through the optical layers.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        return_wf: bool = False
            Should the Wavefront object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront
            if `return_wf` is False, returns the psf Array.
            if `return_wf` is True, returns the Wavefront object.
        """
        # Unintuitive syntax here, this is saying call the _parent class_ of
        # CartesianOpticalSystem, ie LayeredOpticalSystem, which is what we want.
        wf = super(dl.optical_systems.CartesianOpticalSystem, self).propagate_mono(
            wavelength, offset, return_wf=True
        )

        # Propagate
        true_pixel_scale = self.psf_pixel_scale / self.oversample
        pixel_scale = 1e-6 * true_pixel_scale
        psf_npixels = self.psf_npixels * self.oversample

        wf = wf.propagate_fresnel(
            psf_npixels,
            pixel_scale,
            self.focal_length,
            focal_shift=self.defocus,
        )

        # Return PSF or Wavefront
        if return_wf:
            return wf
        return wf.psf


### Sub-propagations ###
def transfer(coords, npixels, wavelength, pscale, distance):
    """
    The optical transfer function (OTF) for the gaussian beam.
    Assumes propagation is along the axis.
    """
    scaling = npixels * pscale**2
    rho_sq = ((coords / scaling) ** 2).sum(0)
    return np.exp(-1.0j * np.pi * wavelength * distance * rho_sq)


def _fft(phasor):
    return 1 / phasor.shape[0] * np.fft.fft2(phasor)


def _ifft(phasor):
    return phasor.shape[0] * np.fft.ifft2(phasor)


def plane_to_plane(wf, distance):
    tf = transfer(wf.coordinates, wf.npixels, wf.wavelength, wf.pixel_scale, distance)
    phasor = _fft(wf.phasor)
    phasor *= np.fft.fftshift(tf)
    phasor = _ifft(phasor)
    return phasor


class FreeSpace(dl.layers.optics.OpticalLayer):
    distance: Array

    def __init__(self, dist):
        self.distance = np.asarray(dist, float)

    def apply(self, wf):
        phasor_out = plane_to_plane(wf, self.distance)
        return wf.set(["amplitude", "phase"], [np.abs(phasor_out), np.angle(phasor_out)])
