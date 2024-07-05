import jax
import jax.numpy as np
import dLux as dl
import dLux.utils as dlu
from jax import Array
from .ami_mask import DynamicAMIStaticAbb


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
        # free_space_before=False,
        # free_space_after=False,
    ):
        """Free space locations can be 'before', 'after'"""
        self.wf_npixels = wf_npixels
        self.diameter = diameter
        self.psf_npixels = psf_npixels
        self.oversample = oversample
        self.psf_pixel_scale = pixel_scale

        # # Get the primary mirror transmission
        # file_path = pkg.resource_filename(__name__, "data/primary.npy")
        # transmission = np.load(file_path)
        # Create the primary
        # primary = dlw.JWSTAberratedPrimary(
        #     transmission,
        #     opd=np.zeros_like(transmission),
        #     radial_orders=np.arange(radial_orders),
        #     AMI=True,
        # )
        # primary = dlw.JWSTAberratedPrimary(
        #     transmission,
        #     opd=np.zeros_like(transmission),
        #     radial_orders=np.arange(radial_orders),
        #     AMI=True,
        # )

        layers = []

        # # Load the values into the primary
        # n_fda = primary.basis.shape[1]
        # file_path = pkg.resource_filename(__name__, "data/FDA_coeffs.npy")
        # primary = primary.set("coefficients", np.load(file_path)[:, :n_fda])

        # if opd is None:
        #     opd = np.zeros_like(transmission)
        # primary = primary.set("opd", opd)
        # primary = primary.multiply("basis", 1e-9)  # Normalise to nm

        # # if coherence is not None:
        # pupil_basis = dlw.JWSTAberratedPrimary(
        #     np.ones((1024, 1024)),
        #     np.zeros((1024, 1024)),
        #     radial_orders=np.arange(coherence_orders),
        #     AMI=True,
        # ).basis
        # layers += [("coherence", PupilAmplitudes(pupil_basis))]

        # layers += [("pupil", primary), ("InvertY", dl.Flip(0))]
        layers += [("InvertY", dl.Flip(0))]

        # if free_space_before:
        #     layers += [("fresnel", FreeSpace(-1e-3))]

        if pupil_mask is None:
            # pupil_mask = DynamicAMI(diameter, wf_npixels, f2f=f2f, normalise=normalise)
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

        # if free_space_after:
        #     layers += [("fresnel2", FreeSpace(1e-3))]

        # Set the layers
        self.layers = dlu.list2dictionary(layers, ordered=True)


class FresnelOptics(dl.CartesianOpticalSystem):
    """
    fl = pixel_scale_m / pixel_scale_rad -> NIRISS pixel scales are 18um  and
    0.0656 arcsec respectively, so fl ~= 56.6m
    """

    defocus: jax.Array  # metres, is this actually um??

    def __init__(self, *args, **kwargs):
        self.defocus = np.array(0.0)
        super().__init__(*args, **kwargs)

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


# from dLux.utils.zernikes import scale_coords, eval_radial, eval_azimuthal
# from jax import Array


# # This is here to add diameter as in input
# def zernike_fast(n: int, m: int, c: Array, k: Array, coordinates: Array, diameter) -> Array:
#     """
#     Calculates the Zernike polynomial using the pre-calculated c and k parameters, such
#     that this function is jittable.

#     Parameters
#     ----------
#     n : int
#         The radial order of the Zernike polynomial.
#     m : int
#         The azimuthal order of the Zernike polynomial.
#     c : Array
#         The normalisation coefficients of the Zernike polynomial.
#     k : Array
#         The powers of the Zernike polynomial.
#     coordinates : Array
#         The Cartesian coordinates to calculate the Zernike polynomial upon.

#     Returns
#     -------
#     zernike : Array
#         The Zernike polynomial.
#     """
#     coordinates = scale_coords(coordinates, diameter / 2)
#     polar_coordinates = dlu.cart2polar(coordinates)
#     rho = polar_coordinates[0]
#     theta = polar_coordinates[1]
#     aperture = rho <= 1.0
#     return aperture * eval_radial(rho, n, c, k) * eval_azimuthal(theta, n, m)


# def polike_fast(
#     nsides: int, n: int, m: int, c: Array, k: Array, coordinates: Array, diameter
# ) -> Array:
#     """
#     Calculates the Zernike polynomial on an n-sided aperture using the pre-calculated
#     c and k parameters, such that this function is jittable.

#     Parameters
#     ----------
#     nsides : int
#         The number of sides of the aperture.
#     n : int
#         The radial order of the Zernike polynomial.
#     m : int
#         The azimuthal order of the Zernike polynomial.
#     c : Array
#         The normalisation coefficients of the Zernike polynomial.
#     k : Array
#         The powers of the Zernike polynomial.
#     coordinates : Array
#         The Cartesian coordinates to calculate the Zernike polynomial upon.

#     Returns
#     -------
#     polike : Array
#         The Zernike polynomial on an n-sided aperture.
#     """
#     if nsides < 3:
#         raise ValueError(f"nsides must be >= 3, not {nsides}.")
#     coordinates = scale_coords(coordinates, diameter / 2)
#     alpha = np.pi / nsides
#     phi = dlu.cart2polar(coordinates)[1] + alpha
#     wedge = np.floor((phi + alpha) / (2.0 * alpha))
#     u_alpha = phi - wedge * (2 * alpha)
#     r_alpha = np.cos(alpha) / np.cos(u_alpha)
#     return 1 / r_alpha * zernike_fast(n, m, c, k, coordinates / r_alpha)


# class DynamicAMI(dl.layers.optical_layers.OpticalLayer):
#     holes: jax.Array
#     f2f: jax.Array
#     transformation: dl.CoordTransform
#     normalise: bool
#     abb_factors: dict
#     abb_coeffs: jax.Array
#     amp_factors: dict
#     amp_coeffs: jax.Array

#     def __init__(
#         self,
#         diameter,
#         wf_npixels,
#         f2f=0.80,
#         normalise=False,
#         abb_radial_orders=None,
#         amp_radial_orders=None,
#     ):
#         file_path = pkg.resource_filename(__name__, "data/AMI_holes.npy")
#         self.holes = np.load(file_path)
#         self.f2f = np.asarray(f2f, float)
#         pixel_scale = diameter / wf_npixels
#         shift_pix = (+21, -13)
#         shift = np.array(shift_pix) * pixel_scale
#         self.transformation = dl.CoordTransform(shift, 0.0, (1.0, 1.0), (0.0, 0.0))
#         # make_transform = lambda x: dl.CoordTransform(shift, 0.0, (1.0, 1.0), (0.0, 0.0))
#         # self.transformation = eqx.filter_vmap(make_transform)(np.arange(7))
#         self.normalise = normalise

#         if abb_radial_orders is not None:
#             noll_inds = dlw.basis.get_noll_indices(np.arange(abb_radial_orders))
#             values = {"c": [], "k": [], "n": [], "m": []}
#             for ind in noll_inds:
#                 n, m = dlu.noll_indices(ind)
#                 c, k = dlu.zernike_factors(ind)
#                 values["n"].append(n)
#                 values["m"].append(m)
#                 values["c"].append(c)
#                 values["k"].append(k)
#             self.abb_factors = values
#             self.abb_coeffs = np.zeros((7, len(noll_inds)))
#         else:
#             self.abb_factors = None
#             self.abb_coeffs = None

#         if amp_radial_orders is not None:
#             noll_inds = dlw.basis.get_noll_indices(np.arange(amp_radial_orders))
#             values = {"c": [], "k": [], "n": [], "m": []}
#             for ind in noll_inds:
#                 n, m = dlu.noll_indices(ind)
#                 c, k = dlu.zernike_factors(ind)
#                 values["n"].append(n)
#                 values["m"].append(m)
#                 values["c"].append(c)
#                 values["k"].append(k)
#             self.amp_factors = values
#             self.amp_coeffs = np.zeros((7, len(noll_inds)))
#         else:
#             self.amp_factors = None
#             self.amp_coeffs = None

#     def gen_coords(self, npix, diameter):
#         # Generate coordinates
#         coords = dlu.pixel_coords(npix, diameter)

#         # Apply transformations
#         coords = dlu.translate_coords(coords, self.translation)
#         coords = dlu.rotate_coords(coords, self.rotation)
#         coords = dlu.compress_coords(coords, self.compression)
#         coords = dlu.shear_coords(coords, self.shear)

#         # Shift the coordinates for each hole
#         coords = vmap(dlu.translate_coords, (None, 0))(coords, self.holes)

#         return coords

#     def gen_AMI(self, npix, diameter):
#         coords = self.gen_coords(npix, diameter)

#         # Generate the hexagons
#         pscale = diameter / npix
#         rmax = self.f2f / np.sqrt(3)
#         hex_fn = lambda coords: dlu.soft_reg_polygon(coords, rmax, 6, pscale)
#         hexes = vmap(hex_fn)(coords).sum(0)

#         # Generate the zernikes - add 1% oversize, note we cant vmap this
#         # inner fn because c, and k have different shapes for different zernikes
#         full_basis_fn = lambda ns, ms, cs, ks, coords: np.array(
#             [
#                 zernike_fast(n, m, c, k, coords, 2 * rmax * 1.75)
#                 for n, m, c, k in zip(ns, ms, cs, ks)
#             ]
#         )

#         # Generate the aberration zernikes
#         if self.abb_factors is not None:
#             vals = self.abb_factors
#             basis_fn = lambda coords: full_basis_fn(
#                 vals["n"], vals["m"], vals["c"], vals["k"], coords
#             )
#             # jax.debug.print("{x}", x=self.abb_coeffs)
#             aberrations = dlu.eval_basis(
#                 1e-9 * vmap(basis_fn)(coords), self.abb_coeffs
#             )  # normalise to nm
#         else:
#             aberrations = np.zeros((npix, npix))

#         # Generate the amplitude zernikes
#         if self.amp_factors is not None:
#             vals = self.amp_factors
#             basis_fn = lambda coords: full_basis_fn(
#                 vals["n"], vals["m"], vals["c"], vals["k"], coords
#             )
#             amplitudes = 1 + dlu.eval_basis(vmap(basis_fn)(coords), self.amp_coeffs)
#         else:
#             amplitudes = np.zeros((npix, npix))
#         return hexes, aberrations, amplitudes

#         # def gen_coords(self, npix, diameter):
#         #     # Generate coordinates
#         #     make_coords = lambda x: dlu.pixel_coords(npix, diameter)
#         #     coords = eqx.filter_vmap(make_coords)(np.arange(7))

#         #     # Shift the coordinates for each hole
#         #     coords = vmap(dlu.translate_coords, (0, 0))(coords, self.holes)

#         #     # Apply transformations
#         #     coords = vmap(dlu.translate_coords, (0, 0))(coords, self.translation)
#         #     coords = vmap(dlu.rotate_coords, (0, 0))(coords, self.rotation)
#         #     coords = vmap(dlu.compress_coords, (0, 0))(coords, self.compression)
#         #     coords = vmap(dlu.shear_coords, (0, 0))(coords, self.shear)

#         #     return coords

#         # def gen_AMI(self, npix, diameter):
#         #     coords = self.gen_coords(npix, diameter)

#         #     # Generate the hexagons
#         #     pscale = diameter / npix
#         #     rmax = self.f2f / np.sqrt(3)
#         #     hex_fn = lambda coords: dlu.soft_reg_polygon(coords, rmax, 6, pscale)
#         #     hexes = vmap(hex_fn)(coords).sum(0)

#         #     # Generate the aberration zernikes - add 1% oversize, note we cant vmap this
#         #     # inner fn because c, and k have different shapes for different zernikes
#         #     full_basis_fn = lambda ns, ms, cs, ks, coords: np.array(
#         #         [
#         #             zernike_fast(n, m, c, k, coords, 2 * rmax * 1.01)
#         #             for n, m, c, k in zip(ns, ms, cs, ks)
#         #         ]
#         #     )
#         # # Generate the aberration zernikes
#         # if self.abb_factors is not None:
#         #     vals = self.abb_factors
#         #     basis_fn = lambda coords: full_basis_fn(
#         #         vals["n"], vals["m"], vals["c"], vals["k"], coords
#         #     )
#         #     aberrations = dlu.eval_basis(
#         #         1e-9 * vmap(basis_fn)(coords), self.abb_coeffs
#         #     )  # normalise to nm
#         # else:
#         #     aberrations = np.zeros((7, npix, npix))

#         # # Generate the amplitude zernikes
#         # if self.amp_factors is not None:
#         #     vals = self.amp_factors
#         #     basis_fn = lambda coords: full_basis_fn(
#         #         vals["n"], vals["m"], vals["c"], vals["k"], coords
#         #     )
#         #     amplitudes = dlu.eval_basis(vmap(basis_fn)(coords), self.amp_coeffs)
#         # else:
#         #     amplitudes = np.zeros((7, npix, npix))

#     #     return hexes, aberrations, amplitudes

#     def apply(self, wavefront):

#         # Calculate mask and aberrations
#         mask, aberrations, amplitudes = self.gen_AMI(wavefront.npixels, wavefront.diameter)

#         # Modify amplitudes
#         # wavefront *= np.maximum(mask + np.where(mask == 0.0, 0.0, amplitudes), 0.0)
#         wavefront *= mask * amplitudes

#         # Modify OPD
#         wavefront += aberrations
#         # wavefront = wavefront.add_opd(aberrations)

#         # import matplotlib.pyplot as plt

#         # plt.imshow(aberrations)
#         # plt.colorbar()
#         # plt.show()

#         # Normalise and return
#         if self.normalise:
#             return wavefront.normalise()
#         return wavefront()

#     def __getattr__(self, key):
#         if hasattr(self.transformation, key):
#             return getattr(self.transformation, key)
#         else:
#             raise AttributeError(f"Interpolator has no attribute {key}")


# class PupilAmplitudes(dl.layers.optics.OpticalLayer):
#     basis: Array
#     reflectivity: Array

#     def __init__(self, basis, reflectivity=None):
#         self.basis = np.asarray(basis, float)

#         if reflectivity is None:
#             self.reflectivity = np.zeros(basis.shape[:-2])
#         else:
#             self.reflectivity = np.asarray(reflectivity, float)

#     def normalise(self):
#         # Normalise to mean of 1
#         return self.add("reflectivity", self.reflectivity.mean())

#     def apply(self, wavefront):
#         # self = self.normalise()
#         reflectivity = 1 + dlu.eval_basis(self.basis, self.reflectivity)
#         return wavefront.multiply("amplitude", reflectivity)


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
