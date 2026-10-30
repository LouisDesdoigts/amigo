import zodiax as zdx
import dLux as dl
import jax.numpy as np
import equinox as eqx
import dLux.utils as dlu
from .misc import interp
from jax import vmap


def fft_coords(wl, npix, pscale, pad=2):
    x = np.fft.fftshift(np.fft.fftfreq(pad * npix, d=pscale / wl))
    return np.array(np.meshgrid(x, x))


def intensity_fft_coords(intensity, pad=2):
    """Fourier coordinates per wavelength, in metres of baseline."""
    return vmap(lambda wl: fft_coords(
        wl, intensity.npixels, intensity.pixel_scale[0], pad=pad
    ))(intensity.wavelength)


def vis_to_im(log_visibility, shape):
    """Complete a complex log-visibility half-grid by conjugate symmetry.

    Zero at the central knot fixes the DC multiplier to one.
    """
    return np.concatenate([
        log_visibility, np.zeros(1, dtype=log_visibility.dtype),
        np.conj(log_visibility[::-1]),
    ]).reshape(shape)


def inject_vis(intensity, log_visibility, otf_coords, *, pad=2):
    """Apply complex log visibility to wavelength-weighted intensity.

    Interpolate in baseline coordinates, multiply each image's Fourier
    transform, then crop and sum the corrected images over wavelength.
    """
    log_visibility = vis_to_im(log_visibility, otf_coords.shape[1:])
    uv_coords = intensity_fft_coords(intensity, pad=pad)
    sampled = vmap(lambda uv: interp(
        log_visibility, otf_coords, uv, method="linear", fill=0.0
    ))(uv_coords)

    size = intensity.npixels
    extra = (pad - 1) * size
    before, after = extra // 2, extra - extra // 2
    to_uv = vmap(lambda image: np.fft.fftshift(
        np.fft.fft2(np.pad(image, ((before, after), (before, after))))
    ))
    from_uv = vmap(lambda spectrum: np.fft.ifft2(
        np.fft.ifftshift(spectrum)
    )[before:before + size, before:before + size])
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
    n_basis: int = eqx.field(static=True)
    n_knots: int = eqx.field(static=True)
    active_filter: str | None = eqx.field(static=True)

    def __init__(self, basis, filter_name=None, *, n_basis=None):
        if "modes" not in basis:
            raise ValueError("Expected complex 'modes' per filter; regenerate the old split basis")
        self.fft_pad = basis.get("fft_pad", 2)
        self.basis_downsample = basis.get("basis_downsample", 1)
        for name in ("fft_pad", "basis_downsample"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.eigenvalues = {
            name: np.asarray(values) for name, values in basis.get("eigen_values", {}).items()
        }
        self.otf_coords = np.asarray(basis["otf_coords"], dtype=float)
        self.n_knots = int(self.otf_coords.shape[-1])
        self.active_filter = filter_name
        self.modes = {
            name: np.asarray(rows[:n_basis], dtype=complex)
            for name, rows in basis["modes"].items()
        }
        if filter_name is None:
            counts = {len(rows) for rows in self.modes.values()}
            if len(counts) != 1:
                raise ValueError("Specify filter_name when filters have different mode counts")
            self.n_basis = counts.pop()
        else:
            self.n_basis = len(self.modes[filter_name])

    def from_latent(self, coefficients, filter):
        if self.active_filter is not None and filter != self.active_filter:
            raise ValueError(f"Model configured for {self.active_filter}, received {filter}")
        return coefficients @ self.modes[filter]

    def to_latent(self, log_visibility, filter):
        return np.real(log_visibility @ self.modes[filter].conj().T)

    def model_vis(self, intensity, coefficients, filter):
        log_visibility = self.from_latent(coefficients, filter)
        image = inject_vis(intensity, log_visibility, self.otf_coords, pad=self.fft_pad)
        return dl.PSF(image, intensity.pixel_scale.mean(0))

    def intensity_to_otf(self, intensity, oversample=2):
        wls = intensity.wavelength
        psf_pscale = intensity.pixel_scale[0]
        npix = oversample * self.otf_coords.shape[-1]
        pscale = np.diff(self.otf_coords[0, 0]).mean() / oversample
        to_uv = vmap(lambda arr, wl: dlu.MFT(arr, wl, psf_pscale, npix, pscale))
        downsample = vmap(lambda arr: dlu.downsample(arr, 2, mean=True))
        return np.mean(downsample(to_uv(intensity.data, wls)), axis=0)

    def intensity_to_latent(self, intensity, filter):
        vis = self.intensity_to_otf(intensity)
        return self.to_latent(np.log(vis.flatten()[:vis.size // 2]), filter)
