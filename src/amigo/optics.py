import pkg_resources as pkg
import jax
import jax.numpy as np
from jax import vmap
import dLux as dl
import dLux.utils as dlu
import dLuxWebbpsf as dlw
from jax import Array


class AMIOptics(dl.optical_systems.AngularOpticalSystem):
    def __init__(
        self,
        radial_orders=4,
        pupil_mask=None,
        opd=None,
        normalise=True,
        coherence=None,
        # free_space_locations=[],
        psf_npixels=80,
        oversample=4,
        pixel_scale=0.065524085,
        diameter=6.603464,
        wf_npixels=1024,
        f2f=0.80,
    ):
        """Free space locations can be 'before', 'after'"""
        self.wf_npixels = wf_npixels
        self.diameter = diameter
        self.psf_npixels = psf_npixels
        self.oversample = oversample
        self.psf_pixel_scale = pixel_scale

        # Get the primary mirror transmission
        file_path = pkg.resource_filename(__name__, "data/primary.npy")
        transmission = np.load(file_path)
        # Create the primary
        primary = dlw.JWSTAberratedPrimary(
            transmission,
            opd=np.zeros_like(transmission),
            radial_orders=np.arange(radial_orders),
            AMI=True,
        )

        layers = []

        # Load the values into the primary
        n_fda = primary.basis.shape[1]
        file_path = pkg.resource_filename(__name__, "data/FDA_coeffs.npy")
        primary = primary.set("coefficients", np.load(file_path)[:, :n_fda])

        if opd is None:
            opd = np.zeros_like(transmission)
        primary = primary.set("opd", opd)
        primary = primary.multiply("basis", 1e-9)  # Normalise to nm

        layers += [("pupil", primary), ("InvertY", dl.Flip(0))]

        # if coherence is not None:
        pupil_basis = dlw.JWSTAberratedPrimary(
            np.ones((1024, 1024)),
            np.zeros((1024, 1024)),
            radial_orders=[0],
            AMI=True,
        ).basis[:, 0]
        layers += [("coherence", PupilAmplitudes(np.flip(pupil_basis, axis=1)))]

        if pupil_mask is None:
            pupil_mask = DynamicAMI(diameter, wf_npixels, f2f=f2f, normalise=normalise)
        layers += [("pupil_mask", pupil_mask)]

        # Set the layers
        self.layers = dlu.list2dictionary(layers, ordered=True)

    # def model(self, exposure, wavels, weights, to_BFE=False, slopes=False):
    #     # Get exposure key
    #     key = exposure.key

    #     position = self.positions[key]
    #     flux = 10 ** self.fluxes[key]
    #     aberrations = self.aberrations[key]
    #     one_on_fs = self.one_on_fs[key]
    #     opd = exposure.opd

    #     optics = self.optics.set(["pupil.coefficients", "pupil.opd"], [aberrations, opd])

    #     if "coherence" in self.params.keys():
    #         coherence = self.coherence[key]
    #         optics = optics.set("holes.reflectivity", coherence)

    #     detector = self.detector.set(
    #         ["EDM.ngroups", "EDM.flux", "EDM.filter", "one_on_fs"],
    #         [exposure.ngroups, flux, exposure.filter, one_on_fs],
    #     )

    #     self = self.set(["optics", "detector"], [optics, detector])
    #     psf = self.model_psf(position, wavels, weights)


class DynamicAMI(dl.layers.optical_layers.OpticalLayer):
    holes: jax.Array
    f2f: jax.Array
    transformation: dl.CoordTransform
    normalise: bool

    def __init__(self, diameter, wf_npixels, f2f=0.80, normalise=False):
        file_path = pkg.resource_filename(__name__, "data/AMI_holes.npy")
        self.holes = np.load(file_path)
        self.f2f = np.asarray(f2f, float)
        pixel_scale = diameter / wf_npixels
        shift_pix = (+21, -13)
        shift = np.array(shift_pix) * pixel_scale
        self.transformation = dl.CoordTransform(shift, 0.0, (1.0, 1.0), (0.0, 0.0))
        self.normalise = normalise

    def gen_AMI(self, npix, diameter):
        rmax = self.f2f / np.sqrt(3)
        coords = dlu.pixel_coords(npix, diameter)

        # Rotate, shear and compress coordinates
        coords = dlu.translate_coords(coords, self.translation)
        coords = dlu.rotate_coords(coords, self.rotation)
        coords = dlu.compress_coords(coords, self.compression)
        coords = dlu.shear_coords(coords, self.shear)

        # Shift the coordinates for each hole
        coords = vmap(dlu.translate_coords, (None, 0))(coords, self.holes)

        # Generate the hexagons
        pscale = diameter / npix
        hex_fn = lambda coords: dlu.soft_reg_polygon(coords, rmax, 6, pscale)
        return vmap(hex_fn)(coords).sum(0)

    def apply(self, wavefront):
        wavefront = wavefront * self.gen_AMI(wavefront.npixels, wavefront.diameter)
        if self.normalise:
            return wavefront.normalise()
        return wavefront()

    def __getattr__(self, key):
        if hasattr(self.transformation, key):
            return getattr(self.transformation, key)
        else:
            raise AttributeError(f"Interpolator has no attribute {key}")


class PupilAmplitudes(dl.layers.optics.OpticalLayer):
    basis: Array
    reflectivity: Array

    def __init__(self, basis, reflectivity=None):
        self.basis = np.asarray(basis, float)

        if reflectivity is None:
            self.reflectivity = np.zeros(basis.shape[:-2])
        else:
            self.reflectivity = np.asarray(reflectivity, float)

    def normalise(self):
        # Normalise to mean of 1
        return self.add("reflectivity", self.reflectivity.mean())

    def apply(self, wavefront):
        # self = self.normalise()
        reflectivity = 1 + dlu.eval_basis(self.basis, self.reflectivity)
        return wavefront.multiply("amplitude", reflectivity)


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
