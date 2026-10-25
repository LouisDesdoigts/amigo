# import pkg_resources as pkg
from importlib import resources
import equinox as eqx
import zodiax as zdx
from jax import Array, vmap
import jax.numpy as np
import dLux as dl
import dLux.utils as dlu
from .misc import calc_throughput, interp, spectral_weights
from .intensity import Intensity
from jax.lax import dynamic_update_slice, dynamic_slice
from abcdLux import lct


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


def resample_intensity(field, factor):
    """Interpolate a complex field, then average its subpixel intensities.

    The output keeps the input shape and pixel scale. Real and imaginary
    components are interpolated separately; no optical phase is reconstructed.
    """
    npixels = field.shape[-1]
    knots = dlu.pixel_coords(npixels, 2)
    samples = dlu.pixel_coords(npixels * factor, 2)
    real = interp(field.real, knots, samples, "cubic2")
    imag = interp(field.imag, knots, samples, "cubic2")
    return dlu.downsample(real**2 + imag**2, factor, mean=True)


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


# Fill in the full array
def expand(arr, index, npix):
    j, i = index
    empty = np.zeros((npix, npix))
    return dynamic_update_slice(empty, arr, (i, j))


def fill(arr, indices, npix):
    return vmap(expand, (0, 0, None))(arr, indices, npix).sum(0)


class BaseApertureMask(dl.layers.optical_layers.OpticalLayer):
    """AMIGO pupil coefficients retain their dLux 0.14 OPD sign convention.

    dLux 0.15 reverses the Fourier sign, so negate OPD when applying it to
    the complex field. The fitted coefficients stay unchanged.
    """
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

        pixel_scale = np.diff(coords[0, 0]).mean()
        corners = calc_corners(holes, coords.shape[-1],
                               pixel_scale * coords.shape[-1], small_npix)

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
    ):
        self.size = 180
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
            return self.transmission * super().calc_transmission(
                npixels=self.transmission.shape[0]
            )
        return self.transmission

    def sparse_fields(self, npixels, diameter):
        """
        Per-hole mask, OPD (metres), and amplitude-perturbation arrays -- each
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
        wavefront = wavefront.add_opd(-self.calc_aberrations(wavefront.npixels))
        if self.normalise:
            return wavefront.normalise()
        return wavefront


def calc_corners(holes, npixels, diameter, size):
    # Shift the coordinates to be centred at the corner (ie array indexed)
    cen_pix = npixels / 2
    if npixels % 2 == 0:
        cen_pix -= 0.5

    # Get the corners of the hole cut outs
    pixel_scale = diameter / npixels
    holes_pix = np.rint((holes / pixel_scale) + cen_pix).astype(int)
    corners = holes_pix - size // 2
    return corners


def crop_windows(arr, corners, size):
    """Crop the pupil windows using (x, y) corners and array (y, x) indexing."""
    return vmap(lambda corner: dynamic_slice(arr, (corner[1], corner[0]), (size, size)))(corners)


def window_centers(corners, size, npixels, diameter):
    """Physical (x, y) centres of the windows pasted into the full pupil."""
    return (corners + (size - npixels) / 2) * (diameter / npixels)


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
            small_npix=size,
        )

    def calc_mask(self, npixels, diameter, oversample=3):
        return fill(self.sparse_apertures(npixels, diameter, oversample), self.corners, npixels)

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
        Per-hole mask, OPD (metres), and amplitude-perturbation arrays -- each
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

    def __call__(self, wavefront):
        wavefront *= self.calc_transmission(npixels=wavefront.npixels)
        wavefront *= self.calc_mask(wavefront.npixels, wavefront.diameter)
        wavefront = wavefront.add_opd(-self.calc_aberrations(wavefront.npixels))
        if self.normalise:
            return wavefront.normalise()
        return wavefront

    def __getattr__(self, key):
        if hasattr(self.transformation, key):
            return getattr(self.transformation, key)
        raise AttributeError(f"{self.__class__.__name__} has no attribute " f"{key}.")


class AMIOptics(dl.optical_systems.AngularOpticalSystem):
    """NIRISS optics using the retrain branch's ABCD dense/sparse propagators.

    ``defocus`` is the physical axial displacement in micrometres; it is not
    numerically interchangeable with the former angular-grid FFT parameter.
    ``sparse=True`` propagates the aperture windows separately, then adds their
    complex fields. The dense path is the default, matching retrain.
    """
    filters: dict
    sparse: bool
    defocus: np.ndarray
    corners: np.ndarray
    psf_upsample: int
    psf_interpolation: str

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
        psf_pixel_scale=0.065524085,
        diameter=6.603464,
        wf_npixels=1024,
        f2f=0.80,
        oversize=1.2,
        defocus=0.01,
        polike=False,
        static=True,
        psf_interpolation="phasor",
        sparse=False,
    ):
        self.sparse = bool(sparse)
        self.psf_upsample = psf_upsample
        if psf_interpolation not in ("intensity", "phasor"):
            raise ValueError("psf_interpolation must be 'intensity' or 'phasor'")
        self.psf_interpolation = psf_interpolation
        self.defocus = np.array(defocus, float)
        self.filters = dict([(filt, calc_throughput(filt, nwavels=nwavels)) for filt in filters])

        layers = []

        layers += [("InvertY", dl.Flip(0))]

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
        layers += [("pupil_mask", pupil_mask)]

        super().__init__(
            wf_npixels, diameter, layers, psf_npixels,
            np.asarray(psf_pixel_scale), oversample,
        )
        self.corners = pupil_mask.corners

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
            Return the propagated field before subpixel intensity averaging?

        Returns
        -------
        object : Array, Wavefront
            if `return_wf` is False, returns the psf Array.
            if `return_wf` is True, returns the field before intensity averaging.
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

        # The propagator samples in metres; downstream detector/visibility
        # models use radians per pixel, as before the dLux upgrade.
        wf = wf.set("pixel_scale", pixel_scale)
        if return_wf:
            return wf
        if self.psf_interpolation == "intensity":
            # Retain the explicit legacy option for comparisons with saved fits.
            npixels = wf.phasor.shape[-1]
            knots = dlu.pixel_coords(npixels, 2)
            samples = dlu.pixel_coords(npixels * self.psf_upsample, 2)
            intensity = interp(wf.psf, knots, samples, "cubic2")
            intensity = dlu.downsample(intensity, self.psf_upsample, mean=True)
            return np.where(intensity < 0, 0.0, intensity)
        return resample_intensity(wf.phasor, self.psf_upsample)

    def propagate(self, wavelengths, offset=None, weights=None,
                  return_wf=False, return_psf=False, return_intensity=False):
        """Return a broadband array, PSF, spectral Intensity, or optical fields.

        Set at most one return flag. Fields precede subpixel intensity averaging;
        Intensity retains the wavelength axis for visibility modelling.
        """
        if sum((return_wf, return_psf, return_intensity)) > 1:
            raise ValueError("Set at most one of return_wf, return_psf, return_intensity")
        wavelengths = np.atleast_1d(wavelengths)
        weights = np.ones_like(wavelengths) / wavelengths.size if weights is None else weights
        weights = np.atleast_1d(weights)
        offset = np.zeros(2) if offset is None else np.asarray(offset)
        if weights.shape != wavelengths.shape:
            raise ValueError("Weights must have the same shape as wavelengths")
        if offset.shape != (2,):
            raise ValueError("Offset must have shape (2,) in radians")

        result = eqx.filter_vmap(
            lambda wavelength: self.propagate_mono(wavelength, offset, return_wf=return_wf)
        )(wavelengths)
        if return_wf:
            return result.multiply("phasor", np.sqrt(weights)[:, None, None])
        intensity = Intensity(
            result * weights[:, None, None], wavelengths,
            dlu.arcsec2rad(self.psf_pixel_scale / self.oversample),
        )
        if return_intensity:
            return intensity
        psf = intensity.to_psf()
        return psf if return_psf else psf.data

    def visibility_model(
        self, filter_name, *, n_knots, spectral_slope=None, pad=2,
        downsample=1, n_modes=None, retained_trace=0.99,
        batch_size=8, checkpoint=True,
    ):
        """Build a LogVisModel from this optics' centred, normalized-spectrum PSF.

        Uses the configured optical field, pixel scale and interpolation, with
        uniform image-pixel weighting and no source flux or detector response.
        ``n_knots`` is an odd UV grid size spanning twice the pupil diameter.
        ``spectral_slope=None`` uses throughput alone; otherwise apply the same
        linear spectral weighting as the forward fit.

        ``pad`` sets the FFT size factor for generation and later injection.
        ``downsample`` sums optical pixels for basis selection only. Choose
        ``n_modes`` explicitly or retain ``retained_trace`` of the JTJ spectrum.
        ``batch_size`` limits derivative columns per batch; ``checkpoint``
        trades recomputation for memory. The result stores complex modes,
        the full eigenvalue spectrum and sampling choices.
        """
        from .vis_models import LogVisModel
        from .visibility_basis import visibility_jtj, select_visibility_modes

        for name, value in (("n_knots", n_knots), ("pad", pad),
                            ("downsample", downsample), ("batch_size", batch_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if n_knots < 3 or n_knots % 2 == 0:
            raise ValueError("n_knots must be odd and at least three")
        if (self.psf_npixels * self.oversample) % downsample:
            raise ValueError("downsample must divide the optical intensity image size")
        if spectral_slope is not None:
            if np.ndim(spectral_slope) != 0 or not bool(np.isfinite(spectral_slope)):
                raise ValueError("spectral_slope must be a finite scalar")

        # Propagate the reference PSF using the forward model's spectral weights.
        wavelengths, throughput = self.filters[filter_name]
        weights = spectral_weights(throughput, spectral_slope)
        if not bool(np.all(np.isfinite(weights))):
            raise ValueError("spectral weights must have a finite, nonzero total")
        intensity = self.propagate(
            wavelengths, offset=np.zeros(2), weights=weights, return_intensity=True,
        )

        # Each independent complex knot has two real degrees of freedom.
        coords = dlu.pixel_coords(n_knots, 2 * self.diameter)
        identity = np.eye((n_knots**2 - 1) // 2, dtype=complex)
        knot_modes = np.concatenate([identity, 1j * identity])

        # Rank combinations of knots by their unweighted image response.
        jtj = visibility_jtj(
            intensity, knot_modes, coords, pad=pad, downsample=downsample,
            batch_size=batch_size, checkpoint=checkpoint,
        )
        modes, eigenvalues = select_visibility_modes(
            jtj, knot_modes, n_modes=n_modes, retained_trace=retained_trace,
        )
        return LogVisModel({
            "otf_coords": coords,
            "modes": {filter_name: modes},
            "eigen_values": {filter_name: eigenvalues},
            "fft_pad": pad,
            "basis_downsample": downsample,
        }, filter_name)

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
        phase = -dlu.opd2phase(opd, wavelength)

        # `InvertY` is applied to the wavefront before the pupil mask in the
        # dense path; on a flat tilted wavefront that only negates the
        # y-component of the source-offset tilt, so replicate that here
        # rather than flipping the already spatially-varying per-hole fields.
        tilt = (2 * np.pi / wavelength) * (offset[0] * x[:, None, :] - offset[1] * y[:, :, None])

        field = amplitude * np.exp(1j * (phase + tilt))
        if pupil_mask.normalise:
            field = field / np.sqrt(np.sum(np.abs(field) ** 2))
        else:
            # Match dLux's initial wavefront amplitude when normalization is off.
            field = field / self.wf_npixels**2

        prop_fn = vmap(
            lambda f, xi, yi: lct.lct_prop(
                u_in=f, spec_in=(xi, yi), spec_out=spec_out, lam=wavelength, ABCD=ABCD
            )
        )
        focal_field = prop_fn(field, x, y).sum(0)
        return dl.Wavefront.from_phasor(focal_field, wavelength, pixel_scale=to_focal.spec.d)
