import pkg_resources as pkg
import zodiax as zdx
from jax import Array, vmap
import jax.numpy as np
import dLux as dl
import dLux.utils as dlu
from .misc import calc_throughput
from jax.lax import dynamic_update_slice, dynamic_slice
from dLux.utils.propagation import transfer_matrix, calc_nfringes


def gen_powers(degree):
    """
    Generates the powers required for a 2d polynomial
    """
    n = dlu.triangular_number(degree)
    vals = np.arange(n)

    # Ypows
    tris = dlu.triangular_number(np.arange(degree))
    ydiffs = np.repeat(tris, np.arange(1, degree + 1))
    ypows = vals - ydiffs

    # Xpows
    tris = dlu.triangular_number(np.arange(1, degree + 1))
    xdiffs = np.repeat(n - np.flip(tris), np.arange(degree, 0, -1))
    xpows = np.flip(vals - xdiffs)

    return xpows, ypows


def distort_coords(coords, coeffs, pows):
    pow_base = np.multiply(*(coords[:, None, ...] ** pows[..., None, None]))
    distortion = np.sum(coeffs[..., None, None] * pow_base[None, ...], axis=1)
    return coords + distortion


### Fresnel propagators ###
def transfer_fn(coords, npixels, wavelength, pscale, distance):
    scaling = npixels * pscale**2
    rho_sq = ((coords / scaling) ** 2).sum(0)
    return _fftshift(np.exp(-1.0j * np.pi * wavelength * distance * rho_sq))


def transfer(wf, distance, pad=2):
    coords = dlu.pixel_coords(pad * wf.npixels, pad * wf.diameter)
    return transfer_fn(coords, wf.npixels, wf.wavelength, pad * wf.pixel_scale, distance)


def _fft(phasor, pad=2):
    padded = dlu.resize(phasor, phasor.shape[0] * pad)
    return 1 / padded.shape[0] * np.fft.fft2(padded)


def _ifft(phasor, pad=1):
    padded = dlu.resize(phasor, phasor.shape[0] * pad)
    return phasor.shape[0] * np.fft.ifft2(padded)


def _fftshift(phasor):
    return np.fft.fftshift(phasor)


def plane_to_plane(wf, distance, pad=2):
    fft_wf = _fft(wf.phasor, pad=pad)
    tf = transfer(wf, distance, pad=pad)
    phasor = dlu.resize(_ifft(fft_wf * tf), wf.npixels)
    return wf.set(["amplitude", "phase"], [np.abs(phasor), np.angle(phasor)])


class DistortedCoords(zdx.Base):
    powers: np.ndarray
    distortion: np.ndarray

    def __init__(self, order=1, distortion=None):
        self.powers = np.array(gen_powers(order + 1))[:, 1:]

        if distortion is None:
            distortion = np.zeros_like(self.powers)
        if distortion is not None and distortion.shape != self.powers.shape:
            raise ValueError("Distortion shape must match powers shape")
        self.distortion = distortion

    def calculate(self, npix, diameter):
        coords = dlu.pixel_coords(npix, diameter)
        return distort_coords(coords, self.distortion, self.powers)

    def apply(self, coords):
        return distort_coords(coords, self.distortion, self.powers)


def get_noll_indices(radial_orders: Array | list = None, noll_indices: Array | list = None):
    if radial_orders is not None:
        radial_orders = np.array(radial_orders)

        if (radial_orders < 0).any():
            raise ValueError("Radial orders must be >= 0")

        noll_indices = []
        for order in radial_orders:
            start = dlu.triangular_number(order)
            stop = dlu.triangular_number(order + 1)
            noll_indices.append(np.arange(start, stop) + 1)
        noll_indices = np.concatenate(noll_indices)

    elif noll_indices is None:
        raise ValueError("Must specify either radial_orders or noll_indices")

    if noll_indices is not None:
        noll_indices = np.array(noll_indices, dtype=int)

    return noll_indices


def calc_mask(coords, f2f, pixel_scale):
    hex_fn = lambda coords: dlu.soft_reg_polygon(coords, f2f / np.sqrt(3), 6, pixel_scale)
    return vmap(hex_fn)(coords).sum(0)


def calc_basis(coords, f2f, radial_orders, polike=False):
    noll_inds = get_noll_indices(np.arange(radial_orders))

    if polike:
        basis_fn = lambda coords: dlu.polike_basis(6, noll_inds, coords, 2 * f2f / np.sqrt(3))
    else:
        basis_fn = lambda coords: dlu.zernike_basis(noll_inds, coords, 2 * f2f / np.sqrt(3))
    return vmap(basis_fn)(coords)


def get_initial_holes(diameter=6.603464, npixels=1024, x_shift=21, y_shift=-13):
    file_path = pkg.resource_filename(__name__, "data/AMI_holes.npy")
    shift = np.array([x_shift, y_shift]) * (diameter / npixels)
    return np.load(file_path) + shift[None, :]


# class BaseApertureMask(dl.layers.optical_layers.OpticalLayer):
#     abb_basis: Array
#     abb_coeffs: Array
#     amp_basis: Array
#     amp_coeffs: Array

#     def __init__(
#         self,
#         hole_coords,
#         f2f,
#         # npix=1024,
#         # diameter=6.603464,
#         aberration_orders=None,
#         amplitude_orders=None,
#         # oversize=1.1,
#         polike=False,
#     ):
#         # coords = dlu.pixel_coords(npix, diameter)
#         # hole_coords = vmap(dlu.translate_coords, (None, 0))(coords, holes)

#         # Calculate the aberration basis functions
#         if aberration_orders is not None:
#             self.abb_basis = 1e-9 * calc_basis(hole_coords, f2f, aberration_orders, polike)
#             self.abb_coeffs = np.zeros(self.abb_basis.shape[:-2])
#         else:
#             self.abb_basis = None
#             self.abb_coeffs = None

#         # Calculate the amplitude basis functions
#         if amplitude_orders is not None:
#             self.amp_basis = calc_basis(hole_coords, f2f, amplitude_orders, polike)
#             self.amp_coeffs = np.zeros(self.amp_basis.shape[:-2])
#         else:
#             self.amp_basis = None
#             self.amp_coeffs = None

#     def calc_transmission(self, npixels=1024):
#         if self.amp_basis is not None:
#             return 1 + dlu.eval_basis(self.amp_basis, self.amp_coeffs)
#         return np.ones((npixels, npixels))

#     def calc_aberrations(self, npixels=1024):
#         if self.abb_basis is not None:
#             return dlu.eval_basis(self.abb_basis, self.abb_coeffs)
#         return np.zeros((npixels, npixels))


def reduce_basis(basis, coords, holes, size=180):
    xs = coords[0, 0]
    npixels = len(xs)
    pixel_scale = np.diff(xs, axis=0).mean()

    # Re-scale the coordinates to pixel units
    arr_coords = coords / pixel_scale

    # Shift the coordinates to be centred at the corner (ie array indexed)
    cen_pix = npixels / 2
    if npixels % 2 == 0:
        cen_pix -= 0.5
    arr_coords = arr_coords + (npixels / 2)

    # Get the holes positions in units of pixels
    holes_pix = np.rint((holes / pixel_scale) + cen_pix).astype(int)

    # Get the corners of the hole cut outs
    hole_corners = holes_pix - size // 2

    # Cut out the sections
    small_basis = np.zeros((*basis.shape[:2], size, size))

    # Note we do (j, i) here since the coordinates are (x, y) indexed
    for idx, (j, i) in enumerate(hole_corners):
        cut = basis[idx, :, i : i + size, j : j + size]
        small_basis = small_basis.at[idx, :].set(cut)
    return small_basis, hole_corners


def eval_small_basis(small_basis, coeffs):
    return vmap(dlu.eval_basis)(small_basis, coeffs)


# Fill in the full array
def expand(arr, index, npix):
    j, i = index
    empty = np.zeros((npix, npix))
    return dynamic_update_slice(empty, arr, (i, j))


def fill(arr, indices, npix):
    return vmap(expand, (0, 0, None))(arr, indices, npix).sum(0)


class BaseApertureMask(dl.layers.optical_layers.OpticalLayer):
    abb_basis: Array
    abb_coeffs: Array
    amp_basis: Array
    amp_coeffs: Array
    corners: Array

    def __init__(
        self,
        coords,
        holes,
        hole_coords,
        f2f,
        aberration_orders=None,
        amplitude_orders=None,
        polike=False,
        small_npix=180,
    ):

        # Calculate the aberration basis functions
        if aberration_orders is not None:
            abb_basis = 1e-9 * calc_basis(hole_coords, f2f, aberration_orders, polike)
            self.abb_basis, corners = reduce_basis(abb_basis, coords, holes, size=small_npix)
            self.abb_coeffs = np.zeros(self.abb_basis.shape[:-2])
        else:
            self.abb_basis = None
            self.abb_coeffs = None

        # Calculate the amplitude basis functions
        if amplitude_orders is not None:
            amp_basis = calc_basis(hole_coords, f2f, amplitude_orders, polike)
            self.amp_basis, corners = reduce_basis(amp_basis, coords, holes, size=small_npix)
            self.amp_coeffs = np.zeros(self.amp_basis.shape[:-2])
        else:
            self.amp_basis = None
            self.amp_coeffs = None

        self.corners = corners

    def eval_basis(self, basis, coeffs, npixels=1024):
        small_eval = eval_small_basis(basis, coeffs)
        return fill(small_eval, self.corners, npixels)

    def calc_transmission(self, npixels=1024):
        if self.amp_basis is not None:
            return 1 + self.eval_basis(self.amp_basis, self.amp_coeffs, npixels)
        return np.ones((npixels, npixels))

    def calc_aberrations(self, npixels=1024):
        if self.abb_basis is not None:
            return self.eval_basis(self.abb_basis, self.abb_coeffs, npixels)
        return np.zeros((npixels, npixels))


class StaticApertureMask(BaseApertureMask, dl.layers.optical_layers.TransmissiveLayer):

    def __init__(
        self,
        holes=None,
        f2f=0.80,
        diameter=6.603464,
        npixels=1024,
        transformation=None,
        normalise=True,
        aberration_orders=None,
        amplitude_orders=None,
        oversize=1.1,
        polike=False,
    ):
        # Get distorted coordinates
        coords = dlu.pixel_coords(npixels, diameter)
        if transformation is not None:
            coords = transformation.apply(coords)

        # Get the holes coordinates
        if holes is None:
            holes = get_initial_holes(diameter, npixels)
        hole_coords = vmap(dlu.translate_coords, (None, 0))(coords, holes)

        # Calculate the transmission mask
        self.transmission = calc_mask(hole_coords, f2f, diameter / npixels)
        self.normalise = bool(normalise)

        super().__init__(
            coords=coords,
            holes=holes,
            hole_coords=hole_coords,
            f2f=f2f * oversize,  # Oversize the aberrations to avoid edge effects
            aberration_orders=aberration_orders,
            amplitude_orders=amplitude_orders,
            polike=polike,
        )

    def calc_transmission(self):
        if self.amp_basis is not None:
            return self.transmission + super().calc_transmission(
                npixels=self.transmission.shape[0]
            )
        return self.transmission

    def calc_aberrations(self):
        if self.abb_basis is not None:
            return dlu.eval_basis(self.abb_basis, self.abb_coeffs)
        return np.zeros_like(self.transmission)

    def apply(self, wavefront):
        wavefront *= self.calc_transmission()
        wavefront += self.calc_aberrations()
        if self.normalise:
            return wavefront.normalise()
        return wavefront


class DynamicApertureMask(BaseApertureMask, dl.layers.optical_layers.OpticalLayer):
    holes: Array
    f2f: Array
    normalise: bool
    transformation: None

    def __init__(
        self,
        holes=None,
        f2f=0.80,
        diameter=6.603464,
        npixels=1024,
        distortion_orders=None,
        normalise=True,
        aberration_orders=None,
        amplitude_orders=None,
        oversize=1.1,
        polike=False,
    ):
        if holes is None:
            holes = get_initial_holes(diameter, npixels)
        self.holes = holes
        self.f2f = np.asarray(f2f, float)
        self.transformation = DistortedCoords(distortion_orders)
        self.normalise = bool(normalise)

        # Get undistorted coordinates for aberrations
        coords = dlu.pixel_coords(npixels, diameter)
        hole_coords = vmap(dlu.translate_coords, (None, 0))(coords, holes)

        super().__init__(
            coords=coords,
            holes=holes,
            hole_coords=hole_coords,
            f2f=f2f * oversize,  # Oversize the aberrations to avoid edge effects
            aberration_orders=aberration_orders,
            amplitude_orders=amplitude_orders,
            polike=polike,
        )

    def calc_mask(self, npixels, diameter):
        coords = self.transformation.apply(dlu.pixel_coords(npixels, diameter))
        hole_coords = vmap(dlu.translate_coords, (None, 0))(coords, self.holes)
        return calc_mask(hole_coords, self.f2f, diameter / npixels)

    def apply(self, wavefront):
        wavefront *= self.calc_transmission(npixels=wavefront.npixels)
        wavefront *= self.calc_mask(wavefront.npixels, wavefront.diameter)
        wavefront += self.calc_aberrations(npixels=wavefront.npixels)
        if self.normalise:
            return wavefront.normalise()
        return wavefront

    def __getattr__(self, key):
        if hasattr(self.transformation, key):
            return getattr(self.transformation, key)
        raise AttributeError(f"{self.__class__.__name__} has no attribute " f"{key}.")


class AMIOptics(dl.optical_systems.AngularOpticalSystem):
    filters: dict
    defocus_type: str
    defocus: np.ndarray
    corners: np.ndarray

    def __init__(
        self,
        nwavels=9,
        filters=["F380M", "F430M", "F480M"],
        radial_orders=6,
        distortion_orders=5,
        oversample=4,
        pupil_mask=None,
        opd=None,
        normalise=True,
        coherence_orders=None,
        psf_npixels=80,
        pixel_scale=0.065524085,
        diameter=6.603464,
        wf_npixels=1024,
        f2f=0.80,
        oversize=1.1,
        defocus=0.01,
        defocus_type="fft",
        polike=False,
        unique_holes=False,
        static_opd=False,
    ):
        if defocus_type not in ["phase", "fft", None]:
            raise ValueError("defocus_type must be one of 'phase', 'fft', or None")
        self.filters = filters
        self.wf_npixels = wf_npixels
        self.diameter = diameter
        self.psf_npixels = psf_npixels
        self.oversample = oversample
        self.psf_pixel_scale = pixel_scale
        self.defocus = np.array(defocus, float)
        self.defocus_type = defocus_type
        self.filters = dict([(filt, calc_throughput(filt, nwavels=nwavels)) for filt in filters])

        layers = []

        if static_opd:
            layers += [("wfs_opd", dl.AberratedLayer(opd=np.zeros((1024, 1024))))]

        layers += [("InvertY", dl.Flip(0))]

        if pupil_mask is None:
            pupil_mask = DynamicApertureMask(
                distortion_orders=distortion_orders,
                diameter=diameter,
                npixels=wf_npixels,
                f2f=f2f,
                normalise=normalise,
                aberration_orders=radial_orders,
                amplitude_orders=coherence_orders,
                oversize=oversize,
                polike=polike,
            )
        layers += [("pupil_mask", pupil_mask)]
        self.layers = dlu.list2dictionary(layers, ordered=True)

        # Get the corners of the arrays for sparse propagation
        basis = np.zeros((1, 1, self.wf_npixels, self.wf_npixels))
        coords = dlu.pixel_coords(self.wf_npixels, self.diameter)
        _, corners = reduce_basis(basis, coords, self.holes, size=180)
        self.corners = corners

    def propagate_mono(self, wavelength, offset=np.zeros(2), return_wf=False):
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
        # Initialise wavefront
        wf = dl.Wavefront(self.wf_npixels, self.diameter, wavelength)
        wf = wf.tilt(offset)

        # Apply layers
        for layer in list(self.layers.values()):
            wf *= layer

        # Propagate
        true_pixel_scale = self.psf_pixel_scale / self.oversample
        pixel_scale = dlu.arcsec2rad(true_pixel_scale)
        psf_npixels = self.psf_npixels * self.oversample

        if self.defocus_type == "phase":
            first, second = dlu.propagation.fresnel_phase_factors(
                wavelength=wf.wavelength * 1e6,
                npixels_in=wf.npixels,
                pixel_scale_in=wf.pixel_scale * 1e6,
                focal_shift=self.defocus * 1e6,
                # In theory these do not matter since the second factor only modifies
                # the phase and so the PSF is unaffected. The 18 (microns) is the
                # pixel scale but should cancel out.
                npixels_out=psf_npixels,
                pixel_scale_out=18 / self.oversample,
                focal_length=18 / dlu.arcsec2rad(pixel_scale),
            )

            wf *= first
            wf = wf.propagate(psf_npixels, pixel_scale)
            wf *= second

        if self.defocus_type == "fft":
            # Default to um defocus
            # wf = wf.propagate(psf_npixels, pixel_scale)
            wf = propagate_sparse(wf, psf_npixels, pixel_scale, corners=self.corners, size=180)
            wf = plane_to_plane(wf, 1e-6 * self.defocus, pad=2)

        if self.defocus_type is None:
            wf = wf.propagate(psf_npixels, pixel_scale)

        # Return PSF or Wavefront
        if return_wf:
            return wf
        return wf.psf


def SparseMFT(
    phasor,
    wavelength: float,
    pixel_scale_in: float,
    npixels_out: int,
    pixel_scale_out: float,
    focal_length: float = None,
    shift=np.zeros(2),
    pixel: bool = True,
    inverse: bool = False,
    corner=None,
    size=None,
):
    # Get parameters
    npixels_in = phasor.shape[-1]
    if not pixel:
        shift /= pixel_scale_out

    # Alias the transfer matrix function
    get_tf_mat = lambda s: transfer_matrix(
        wavelength,
        npixels_in,
        pixel_scale_in,
        npixels_out,
        pixel_scale_out,
        s,
        focal_length,
        0.0,
        inverse,
    )

    # Get transfer matrices and propagate
    x_mat, y_mat = vmap(get_tf_mat)(shift)

    # Cut the bits out
    if corner is not None:
        x, y = corner
        x_mat = dynamic_slice(x_mat, (x, 0), (size, x_mat.shape[1]))
        y_mat = dynamic_slice(y_mat, (y, 0), (size, y_mat.shape[1]))
        phasor = dynamic_slice(phasor, (y, x), (size, size))

    # Propagate
    phasor = (y_mat.T @ phasor) @ x_mat

    # Normalise
    nfringes = calc_nfringes(
        wavelength,
        npixels_in,
        pixel_scale_in,
        npixels_out,
        pixel_scale_out,
        focal_length,
    )
    phasor *= np.exp(np.log(nfringes) - (np.log(npixels_in) + np.log(npixels_out)))

    return phasor


def propagate_sparse(
    wavefront,
    npixels: int,
    pixel_scale: float,
    focal_length: float = None,
    shift: Array = np.zeros(2),
    pixel: bool = True,
    corners=None,
    size=None,
):

    inverse, plane, units = wavefront._prep_prop(focal_length)

    # Enforce array so output can be vectorised by vmap
    pixel_scale = np.asarray(pixel_scale, float)

    # Calculate
    phasor = SparseMFT(
        wavefront.phasor,
        wavefront.wavelength,
        wavefront.pixel_scale,
        npixels,
        pixel_scale,
        focal_length,
        shift,
        pixel,
        inverse,
    )

    # Update
    return wavefront.set(
        ["amplitude", "phase", "pixel_scale", "plane", "units"],
        [np.abs(phasor), np.angle(phasor), pixel_scale, plane, units],
    )
