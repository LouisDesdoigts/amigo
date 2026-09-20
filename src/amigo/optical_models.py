# import pkg_resources as pkg
from importlib import resources
import equinox as eqx
import zodiax as zdx
from jax import Array, vmap
import jax.numpy as np
import dLux as dl
import dLux.utils as dlu
from abcdLux import lct
from .misc import calc_throughput, interp
from jax.lax import dynamic_update_slice, dynamic_slice


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
def _fft(phasor, pad=2):
    padded = dlu.resize(phasor, phasor.shape[0] * pad)
    return 1 / padded.shape[0] * np.fft.fft2(padded)


def _ifft(phasor, pad=1):
    padded = dlu.resize(phasor, phasor.shape[0] * pad)
    return phasor.shape[0] * np.fft.ifft2(padded)


def _fftshift(phasor):
    return np.fft.fftshift(phasor)


def transfer_fn(coords, npixels, wavelength, pscale, distance):
    rho_sq = (coords**2).sum(0)
    return _fftshift(np.exp(-1.0j * np.pi * wavelength * distance * rho_sq))


def transfer(wf, distance, pad=2):
    npix = pad * wf.npixels
    diam = pad * wf.diameter
    freqs = np.fft.fftshift(np.fft.fftfreq(npix, diam / npix))
    coords = np.array(np.meshgrid(freqs, freqs))
    return transfer_fn(coords, wf.npixels, wf.wavelength, pad * wf.pixel_scale, distance)


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
    # file_path = pkg.resource_filename(__name__, "data/AMI_holes.npy")
    file_path = resources.files(__package__) / "data" / "AMI_holes.npy"
    shift = np.array([x_shift, y_shift]) * (diameter / npixels)
    return np.load(file_path) + shift[None, :]


def reduce_basis(basis, coords, holes, size=180):
    xs = coords[0, 0]
    npixels = len(xs)
    pixel_scale = np.diff(xs, axis=0).mean()

    # # Re-scale the coordinates to pixel units
    # arr_coords = coords / pixel_scale

    # Shift the coordinates to be centred at the corner (ie array indexed)
    cen_pix = npixels / 2
    if npixels % 2 == 0:
        cen_pix -= 0.5
    # arr_coords = arr_coords + (npixels / 2)

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


def crop_windows(arr, corners, size):
    """
    Crops a single dense 2d array into per-hole (size, size) windows at the
    given corners -- the single-array equivalent of `reduce_basis`'s crop,
    used to pull per-hole content back out of an already-computed dense
    array (e.g. StaticApertureMask.transmission) for sparse propagation.

    Uses `dynamic_slice` (rather than plain array slicing) since `corners`
    is a regular (non-static) pytree leaf and so may be a traced value under
    jit/grad.
    """

    def crop_one(corner):
        j, i = corner
        return dynamic_slice(arr, (i, j), (size, size))

    return vmap(crop_one)(corners)


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
        # self.corners = np.array(corners, dtype=float)

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
    size: int = eqx.field(static=True)

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
        small_npix=180,
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
        self.size = small_npix

        super().__init__(
            coords=coords,
            holes=holes,
            hole_coords=hole_coords,
            f2f=f2f * oversize,  # Oversize the aberrations to avoid edge effects
            aberration_orders=aberration_orders,
            amplitude_orders=amplitude_orders,
            polike=polike,
            small_npix=small_npix,
        )

    def calc_transmission(self):
        if self.amp_basis is not None:
            return self.transmission * super().calc_transmission(
                npixels=self.transmission.shape[0]
            )
        return self.transmission

    def sparse_fields(self, npixels, diameter):
        """
        Per-hole mask, OPD (nm), and amplitude-perturbation arrays -- each
        shape (n_holes, size, size) -- plus the physical (x, y) centre of
        each hole's local window. Sparse-propagation equivalent of
        `calc_transmission` + `calc_aberrations`, without ever assembling a
        dense (npixels, npixels) array.

        Note: `self.transmission` can be overwritten wholesale after
        construction (e.g. `optics.set("transmission", ...)` when restoring a
        fitted/measured aperture), so the per-hole windows are cropped fresh
        from it on every call rather than cached at `__init__` time.
        """
        mask = crop_windows(self.transmission, self.corners, self.size)

        if self.abb_basis is not None:
            opd = eval_small_basis(self.abb_basis, self.abb_coeffs)
        else:
            opd = np.zeros_like(mask)

        if self.amp_basis is not None:
            amp = eval_small_basis(self.amp_basis, self.amp_coeffs)
        else:
            amp = np.zeros_like(mask)

        centers = window_centers(self.corners, self.size, npixels, diameter)
        return mask, opd, amp, centers

    def __call__(self, wavefront):
        wavefront *= self.calc_transmission()
        wavefront = wavefront.add_opd(self.calc_aberrations())
        if self.normalise:
            return wavefront.normalise()
        return wavefront


def _cen_pix(n):
    cen_pix = n / 2
    if n % 2 == 0:
        cen_pix -= 0.5
    return cen_pix


def calc_corners(holes, npixels, diameter, size):
    # Shift the coordinates to be centred at the corner (ie array indexed)
    cen_pix = _cen_pix(npixels)

    # Get the corners of the hole cut outs
    pixel_scale = diameter / npixels
    holes_pix = np.rint((holes / pixel_scale) + cen_pix).astype(int)
    corners = holes_pix - size // 2
    return corners


def window_centers(corners, size, npixels, diameter):
    """
    Physical (x, y) coordinates of the centre of each hole's local cut-out
    window (as pasted by `calc_corners`/`dynamic_update_slice`), for use as
    the per-hole coordinate offset in sparse propagation.
    """
    pixel_scale = diameter / npixels
    pix_offset = _cen_pix(size) - _cen_pix(npixels)
    return (corners + pix_offset) * pixel_scale


def calc_mask_hole(coeffs, coords, hole_cen, powers, ap_fn, oversample=3):
    coords += hole_cen[:, None, None]
    coords = distort_coords(coords, coeffs, powers)
    return dlu.downsample(ap_fn(coords), oversample, mean=True)


class DynamicApertureMask(BaseApertureMask, dl.layers.optical_layers.OpticalLayer):
    holes: Array
    f2f: Array
    normalise: bool
    transformation: None
    primary_beam: Array
    primary_powers: Array
    corners: Array
    size: int = eqx.field(static=True)

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
        oversize=1.2,
        polike=False,
        size=180,
    ):
        if holes is None:
            holes = get_initial_holes(diameter, npixels)
        self.holes = holes
        self.f2f = np.asarray(f2f, float)
        self.transformation = DistortedCoords(distortion_orders)
        self.normalise = bool(normalise)

        #
        # self.softness = np.array(softness, float)
        self.primary_powers = np.array(gen_powers(4))[:, 1:]
        self.primary_beam = np.zeros((7, *self.primary_powers.shape))
        self.corners = calc_corners(holes, npixels, diameter, size)
        self.size = size

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

    def calc_mask(self, npixels, diameter, oversample=3):
        pixel_scale = diameter / npixels

        # Get the oversample sub-array coordinates
        npix = npixels * oversample
        full_size = self.size * oversample
        full_pixel_scale = diameter / npix
        small_diam = full_size * full_pixel_scale
        coords = dlu.pixel_coords(full_size, small_diam)

        # Calculate the offset
        distort_fn = lambda coords: distort_coords(coords, self.distortion, self.primary_powers)
        holes = distort_fn(self.holes.T[..., None])[..., 0].T
        offset = holes - (pixel_scale * (2 * self.corners + self.size) - diameter) / 2

        # Calculate the individual apertures
        ap_fn = lambda coords: dlu.soft_reg_polygon(
            coords, self.f2f / np.sqrt(3), 6, 0.25 * pixel_scale
        )
        mask_fn = lambda coeffs, cen: calc_mask_hole(
            coeffs, coords, cen, self.primary_powers, ap_fn
        )
        apertures = vmap(mask_fn)(self.primary_beam, offset)

        # Paste into the full array
        full = np.zeros((1024, 1024))
        for ind, (j, i) in enumerate(self.corners):
            full = dynamic_update_slice(full, apertures[ind], (i, j))
        return full

    def sparse_apertures(self, npixels, diameter, oversample=3):
        """
        Per-hole hexagonal transmission masks, shape (n_holes, size, size),
        each evaluated on its own local coordinate window -- the same
        per-hole content computed by `calc_mask`, without pasting it into a
        dense (npixels, npixels) array.
        """
        pixel_scale = diameter / npixels

        # Get the oversample sub-array coordinates
        npix = npixels * oversample
        full_size = self.size * oversample
        full_pixel_scale = diameter / npix
        small_diam = full_size * full_pixel_scale
        coords = dlu.pixel_coords(full_size, small_diam)

        # Calculate the offset
        distort_fn = lambda coords: distort_coords(coords, self.distortion, self.primary_powers)
        holes = distort_fn(self.holes.T[..., None])[..., 0].T
        offset = holes - (pixel_scale * (2 * self.corners + self.size) - diameter) / 2

        # Calculate the individual apertures
        ap_fn = lambda coords: dlu.soft_reg_polygon(
            coords, self.f2f / np.sqrt(3), 6, 0.25 * pixel_scale
        )
        mask_fn = lambda coeffs, cen: calc_mask_hole(
            coeffs, coords, cen, self.primary_powers, ap_fn, oversample
        )
        return vmap(mask_fn)(self.primary_beam, offset)

    def sparse_fields(self, npixels, diameter, oversample=3):
        """
        Per-hole mask, OPD (nm), and amplitude-perturbation arrays -- each
        shape (n_holes, size, size) -- plus the physical (x, y) centre of
        each hole's local window. This is the sparse-propagation equivalent
        of `calc_mask` + `calc_transmission` + `calc_aberrations`, without
        ever pasting per-hole content into a dense (npixels, npixels) array.
        """
        mask = self.sparse_apertures(npixels, diameter, oversample)

        if self.abb_basis is not None:
            opd = eval_small_basis(self.abb_basis, self.abb_coeffs)
        else:
            opd = np.zeros_like(mask)

        if self.amp_basis is not None:
            amp = eval_small_basis(self.amp_basis, self.amp_coeffs)
        else:
            amp = np.zeros_like(mask)

        centers = window_centers(self.corners, self.size, npixels, diameter)
        return mask, opd, amp, centers

    # def calc_mask(self, npixels, diameter, oversample=3):
    #     # npix = npixels * oversample
    #     # coords = self.transformation.apply(dlu.pixel_coords(npixels, diameter))

    #     # Distort the hole coordinates
    # distort_fn = lambda coords: distort_coords(coords, self.distortion, self.primary_powers)
    # holes = distort_fn(self.holes.T[..., None])[..., 0].T
    # holes = distort_coords(self.holes.T[..., None], self.distortion, powers)[..., 0].T

    #     # Apply the per-hole primary-beam distortion
    #     npix = oversample * npixels
    #     coords = dlu.pixel_coords(npix, diameter)
    #     distort_fn = lambda coeffs: distort_coords(coords, coeffs, self.primary_powers)
    #     coords = vmap(distort_fn)(self.primary_beam)
    #     # coords = distort_coords(coords, self.primary_beam, self.primary_powers)

    #     # Shift to the hole positions
    #     hole_coords = vmap(dlu.translate_coords)(coords, holes)
    #     # hole_coords = vmap(dlu.translate_coords, (None, 0))(coords, self.holes)
    #     # mask = calc_mask(hole_coords, self.f2f, self.softness * diameter / npixels)
    #     mask = calc_mask(hole_coords, self.f2f, 0.5 * diameter / npix)
    #     return dlu.downsample(mask, oversample, mean=True)

    def __call__(self, wavefront):
        wavefront *= self.calc_transmission(npixels=wavefront.npixels)
        wavefront *= self.calc_mask(wavefront.npixels, wavefront.diameter)
        wavefront = wavefront.add_opd(self.calc_aberrations())
        if self.normalise:
            return wavefront.normalise()
        return wavefront

    def __getattr__(self, key):
        if hasattr(self.transformation, key):
            return getattr(self.transformation, key)
        raise AttributeError(f"{self.__class__.__name__} has no attribute " f"{key}.")


class AMIOptics(dl.AngularOpticalSystem):
    filters: dict
    defocus: np.ndarray
    corners: np.ndarray
    psf_upsample: int
    sparse: bool = eqx.field(static=True)

    def __init__(
        self,
        nwavels=9,
        filters=["F380M", "F430M", "F480M"],
        radial_orders=4,
        distortion_orders=3,
        coherence_orders=4,
        oversample=3,
        psf_upsample=3,
        pupil_mask=None,
        normalise=True,
        psf_npixels=80,
        psf_pixel_scale=0.065524085,  # mas/pixel?
        diameter=6.603464,
        wf_npixels=1024,
        f2f=0.80,
        oversize=1.2,
        defocus=0.01,
        polike=False,
        static=True,
        sparse=False,
    ):
        self.sparse = bool(sparse)

        # Instantiate pupil mask layer
        if pupil_mask is None:
            if not static:
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
            else:
                pupil_mask = StaticApertureMask(
                    diameter=diameter,
                    npixels=wf_npixels,
                    f2f=f2f,
                    normalise=normalise,
                    aberration_orders=radial_orders,
                    amplitude_orders=coherence_orders,
                    oversize=oversize,
                    polike=polike,
                )

        # optical layers
        layers = [("InvertY", dl.Flip(0)), ("pupil_mask", pupil_mask)]

        super().__init__(
            wf_npixels,
            diameter,
            layers,
            psf_npixels,
            np.array(psf_pixel_scale),
            oversample,
        )

        self.psf_upsample = psf_upsample
        self.defocus = np.array(defocus, float)
        self.filters = dict([(filt, calc_throughput(filt, nwavels=nwavels)) for filt in filters])

        # Get the corners of the arrays for sparse propagation
        if not hasattr(self, "holes"):
            holes = get_initial_holes(diameter, wf_npixels)
        else:
            holes = self.holes
        basis = np.zeros((1, 1, self.wf_npixels, self.wf_npixels))
        coords = dlu.pixel_coords(self.wf_npixels, self.diameter)
        _, corners = reduce_basis(basis, coords, holes, size=180)
        self.corners = corners
        # self.corners = np.array(corners, dtype=float)

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
        # Get pixel scale in radians
        pixel_scale = dlu.arcsec2rad(self.psf_pixel_scale / self.oversample)
        psf_npixels = self.psf_npixels * self.oversample

        # Getting the focal length from the pixel scale and pixel size.
        # NIRISS Pixel size quoted as 18 micron x 18 micron in JDOX
        pixel_scale_metres = 18e-6 / self.oversample
        focal_length = pixel_scale_metres / pixel_scale  # derived focal length

        # defining the propagator
        # defocus stored in microns, converted to metres in the propagator
        to_focal = dl.MFTPropagator(
            [
                ("ThinLens", dl.ABCDConjugatePlane(focal_length)),
                ("FreeSpace", dl.ABCDFreeSpace(+1e-6 * self.defocus)),
            ],
            dl.CoordSpec(n=psf_npixels, d=pixel_scale_metres),
        )

        if self.sparse:
            wf = self._propagate_pupil_sparse(wavelength, offset, to_focal)
        else:
            # Initialise wavefront
            wf = self.initialise_wavefront(wavelength, offset)

            # Apply layers
            for layer in list(self.layers.values()):
                wf = layer(wf)

            wf = to_focal(wf)

        # Upsample and then downsample to get more PSF precision
        knots = dlu.pixel_coords(psf_npixels, diameter=2)
        sample_coords = dlu.pixel_coords(psf_npixels * self.psf_upsample, diameter=2)
        psf = interp(wf.psf, knots, sample_coords, "cubic2")  # Upsampling with interp
        psf = dlu.downsample(psf, self.psf_upsample, mean=True)
        psf = np.where(psf < 0, 0.0, psf)  # clipping

        # resetting amplitude while not affecting phase
        amplitude = np.sqrt(psf)
        phase = np.angle(wf.phasor)
        wf = wf.set("phasor", amplitude * np.exp(1j * phase))

        # Return PSF or Wavefront
        if return_wf:
            return wf
        return wf.psf

    def _propagate_pupil_sparse(self, wavelength, offset, to_focal):
        """
        Sparse-propagation equivalent of building the pupil-plane wavefront
        (`initialise_wavefront` + the `InvertY`/`pupil_mask` layers) and
        propagating it to the focal plane with `to_focal`. Instead of
        assembling a dense (wf_npixels, wf_npixels) pupil array, each hole's
        small local field is propagated individually and the results are
        coherently summed at the output.

        Parameters
        ----------
        wavelength : float, metres
        offset : Array, radians
            The (x, y) offset from the optical axis of the source.
        to_focal : dl.MFTPropagator
            The (uninvoked) focal-plane propagator, used only for its
            composed ABCD matrix and output coordinate specification -- this
            guarantees the sparse path uses exactly the same defocus/lens
            physics and output sampling as the dense path.

        Returns
        -------
        wf : dl.Wavefront
            The focal-plane wavefront, equivalent to `to_focal(wf)` in the
            dense path.
        """
        pupil_mask = self.layers["pupil_mask"]
        if not hasattr(pupil_mask, "sparse_fields"):
            raise NotImplementedError(
                f"Sparse propagation is not implemented for " f"{type(pupil_mask).__name__}."
            )

        ABCD = to_focal.abcd
        spec_out = to_focal.spec.xs

        # Per-hole mask, OPD, amplitude-perturbation, and window centres
        mask, opd, amp, centers = pupil_mask.sparse_fields(self.wf_npixels, self.diameter)

        # Per-hole local (x, y) coordinates, in the same frame as `centers`
        pixel_scale = self.diameter / self.wf_npixels
        local = dlu.nd_coords(pupil_mask.size, pixel_scale)
        x = centers[:, 0:1] + local[None, :]
        y = centers[:, 1:2] + local[None, :]

        # Entrance-pupil complex field per hole
        amplitude = mask * (1 + amp)
        phase = dlu.opd2phase(opd, wavelength)

        # `InvertY` is applied to the wavefront before the pupil mask in the
        # dense path; on a flat tilted wavefront that only negates the
        # y-component of the source-offset tilt, so replicate that here
        # rather than flipping the already spatially-varying per-hole fields.
        tilt = (2 * np.pi / wavelength) * (offset[0] * x[:, None, :] - offset[1] * y[:, :, None])

        field = amplitude * np.exp(1j * (phase + tilt))
        if pupil_mask.normalise:
            field = field / np.sqrt(np.sum(np.abs(field) ** 2))

        prop_fn = vmap(
            lambda f, xi, yi: lct.lct_prop(
                u_in=f, spec_in=(xi, yi), spec_out=spec_out, lam=wavelength, ABCD=ABCD
            )
        )
        focal_field = prop_fn(field, x, y).sum(0)
        return dl.Wavefront.from_phasor(focal_field, wavelength, pixel_scale=to_focal.spec.d)


# class Wavefront(dl.Wavefront):

#     def downsample(self, factor=2):
#         """
#         Downsample the wavefront by a factor of 2.
#         """
#         phasor = self.phasor
#         real = dlu.downsample(phasor.real, factor, mean=True)
#         imag = dlu.downsample(phasor.imag, factor, mean=True)
#         phasor = real + 1j * imag
#         amplitude = np.abs(phasor)
#         phase = np.angle(phasor)

#         pixel_scale = self.pixel_scale * factor
#         return self.set(
#             ["pixel_scale", "amplitude", "phase"],
#             [pixel_scale, amplitude, phase],
#         )
