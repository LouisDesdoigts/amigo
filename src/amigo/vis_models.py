import zodiax as zdx
import dLux as dl
import jax.numpy as np
import equinox as eqx
import dLux.utils as dlu
from .misc import interp
from jax import vmap


def fft_coords(wavelength, npix, pscale, pad=2):
    x = np.fft.fftshift(np.fft.fftfreq(pad * npix, d=pscale / wavelength))
    return np.array(np.meshgrid(x, x))


def intensity_fft_coords(intensity, pad=2):
    """Fourier coordinates per wavelength, in metres of baseline."""
    return vmap(
        lambda wavelength: fft_coords(
            wavelength, intensity.npixels, intensity.pixel_scale[0], pad=pad
        )
    )(intensity.wavelength)


def vis_to_uvgrid(log_visibility, shape):
    """Complete a complex log-visibility half-grid by conjugate symmetry.

    Zero at the central knot fixes the DC multiplier to one.
    """
    return np.concatenate(
        [
            log_visibility,
            np.zeros(1, dtype=log_visibility.dtype),
            np.conj(log_visibility[::-1]),
        ]
    ).reshape(shape)


def inject_vis(intensity, log_visibility, otf_coords, *, pad=2):
    """Apply complex log visibility to wavelength-weighted intensity.

    Interpolate in baseline coordinates, multiply each image's Fourier
    transform, then crop and sum the corrected images over wavelength.
    """
    log_visibility = vis_to_uvgrid(log_visibility, otf_coords.shape[1:])
    uv_coords = intensity_fft_coords(intensity, pad=pad)
    sampled = vmap(lambda uv: interp(log_visibility, otf_coords, uv, method="linear", fill=0.0))(
        uv_coords
    )

    size = intensity.npixels
    extra = (pad - 1) * size
    before, after = extra // 2, extra - extra // 2
    to_uv = vmap(
        lambda image: np.fft.fftshift(
            np.fft.fft2(np.pad(image, ((before, after), (before, after))))
        )
    )
    from_uv = vmap(
        lambda spectrum: np.fft.ifft2(np.fft.ifftshift(spectrum))[
            before : before + size, before : before + size
        ]
    )
    spectrum = to_uv(intensity.data) * np.exp(sampled)
    # Recover intensity without folding negative predictions positive.
    return np.real(from_uv(spectrum)).sum(0)


class LogVisModel(zdx.Base):
    """Real coefficients on complex, orthonormal log-visibility modes.

    Rows of each half-grid basis are orthonormal under Re(B @ B.conj().T).
    """

    modes: dict
    eigenvalues: dict
    fft_pad: int = eqx.field(static=True)
    basis_downsample: int = eqx.field(static=True)
    otf_coords: np.ndarray
    n_modes: int = eqx.field(static=True)
    n_knots: int = eqx.field(static=True)
    active_filter: str | None = eqx.field(static=True)

    def __init__(self, basis, filt=None, *, n_modes=None):
        if "modes" not in basis:
            raise ValueError("Expected complex 'modes' per filter; regenerate the old split basis")
        if "eigen_values" in basis:
            raise ValueError("Rename the basis key 'eigen_values' to 'eigenvalues'")
        self.fft_pad = basis.get("fft_pad", 2)
        self.basis_downsample = basis.get("basis_downsample", 1)
        for name in ("fft_pad", "basis_downsample"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.eigenvalues = {
            name: np.asarray(values) for name, values in basis.get("eigenvalues", {}).items()
        }
        self.otf_coords = np.asarray(basis["otf_coords"], dtype=float)
        self.n_knots = int(self.otf_coords.shape[-1])
        self.active_filter = filt
        self.modes = {
            name: np.asarray(rows[:n_modes], dtype=complex)
            for name, rows in basis["modes"].items()
        }
        if filt is None:
            counts = {len(rows) for rows in self.modes.values()}
            if len(counts) != 1:
                raise ValueError("Specify filt when filters have different mode counts")
            self.n_modes = counts.pop()
        else:
            self.n_modes = len(self.modes[filt])

    def from_latent(self, coefficients, filt):
        if self.active_filter is not None and filt != self.active_filter:
            raise ValueError(f"Model configured for {self.active_filter}, received {filt}")
        return coefficients @ self.modes[filt]

    def to_latent(self, log_visibility, filt):
        return np.real(log_visibility @ self.modes[filt].conj().T)

    def model_vis(self, intensity, coefficients, filt):
        log_visibility = self.from_latent(coefficients, filt)
        image = inject_vis(intensity, log_visibility, self.otf_coords, pad=self.fft_pad)
        return dl.PSF(image, intensity.pixel_scale.mean(0))

    def intensity_to_otf(self, intensity, oversample=2):
        wavelengths = intensity.wavelength
        psf_pscale = intensity.pixel_scale[0]
        npix = oversample * self.otf_coords.shape[-1]
        pscale = np.diff(self.otf_coords[0, 0]).mean() / oversample
        to_uv = vmap(lambda arr, wavelength: dlu.MFT(arr, wavelength, psf_pscale, npix, pscale))
        downsample = vmap(lambda arr: dlu.downsample(arr, oversample, mean=True))
        return np.mean(downsample(to_uv(intensity.data, wavelengths)), axis=0)

    def intensity_to_latent(self, intensity, filt):
        vis = self.intensity_to_otf(intensity)
        return self.to_latent(np.log(vis.flatten()[: vis.size // 2]), filt)


class PlaceholderVisModel(zdx.Base):
    """
    No visibility modulation: placeholder for initialising the AmigoModel with
    a vis_model but if no visibility modulation is (yet) required. For example,
    use in amigo_pipeline for 1.instrument and 2.reference stages, before the
    relevant vis_basis is calculated.
    """

    n_modes = 0

    def model_vis(self, intensity, coefficients, filt):
        return dl.PSF(intensity.data.sum(0), intensity.pixel_scale.mean(0))


def visibility_jtj(
    intensity,
    knot_modes,
    coords,
    *,
    pad=2,
    downsample=1,
    batch_size=8,
    checkpoint=True,
):
    """Calculate unweighted image J.T @ J at zero visibility coefficients."""

    def image(coefficients):
        result = inject_vis(intensity, coefficients @ knot_modes, coords, pad=pad)
        return dlu.downsample(result, downsample, mean=False).reshape(-1)

    # Differentiate the image directly, without noise or PSF weighting.
    dimension = len(knot_modes)
    jtj = zdx.gauss_newton(
        image,
        np.zeros(dimension),
        nbatches=(dimension + batch_size - 1) // batch_size,
        checkpoint=checkpoint,
    ).matrix
    if not bool(np.all(np.isfinite(jtj))):
        raise FloatingPointError("visibility JTJ contains non-finite values")
    return jtj


def select_visibility_modes(jtj, knot_modes, *, n_modes=None, retained_trace=0.99):
    """Return leading complex modes and the full descending JTJ spectrum."""
    dimension = len(knot_modes)
    if n_modes is not None:
        if (
            isinstance(n_modes, bool)
            or not isinstance(n_modes, int)
            or not 1 <= n_modes <= dimension
        ):
            raise ValueError(f"n_modes must be between 1 and {dimension}")
    elif not 0 < retained_trace <= 1:
        raise ValueError("retained_trace must lie in (0, 1]")

    # Symmetrize roundoff, then order modes by image response.
    eigenvalues, vectors = np.linalg.eigh((jtj + jtj.T) / 2)
    eigenvalues, rows = eigenvalues[::-1], vectors[:, ::-1].T
    positive = np.maximum(eigenvalues, 0)
    if not bool(positive.sum() > 0):
        raise ValueError("optical model has no visibility response")
    if n_modes is None:
        cumulative = np.cumsum(positive) / positive.sum()
        n_modes = min(int(np.searchsorted(cumulative, retained_trace)) + 1, dimension)
    if not bool(np.all(eigenvalues[:n_modes] > 0)):
        raise ValueError("requested modes include a zero or negative JTJ eigenvalue")

    # Fix the arbitrary eigenvector sign before expressing modes on the UV grid.
    rows = rows[:n_modes]
    pivots = np.argmax(np.abs(rows), axis=1)
    rows = rows * np.sign(rows[np.arange(n_modes), pivots])[:, None]
    return rows @ knot_modes, eigenvalues
