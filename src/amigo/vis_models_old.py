import zodiax as zdx
import dLux as dl
import jax.tree as jtu
import jax.numpy as np
import equinox as eqx
import dLux.utils as dlu
from .misc import interp
from jax import vmap


def fft_coords(wl, npix, pscale, pad=2):
    x = np.fft.fftshift(np.fft.fftfreq(pad * npix, d=pscale / wl))
    return np.array(np.meshgrid(x, x))


def wf_fft_coords(wfs, pad=2):
    wls = wfs.wavelength
    psf_pscale = wfs.pixel_scale[0]
    psf_npix = wfs.npixels
    return vmap(lambda wl: fft_coords(wl, psf_npix, psf_pscale, pad=pad))(wls)


def vis_to_im(vis, shape):
    # Convert from vis vector to amplitudes and phases
    log_amps, phases = np.split(vis, 2, axis=0)
    amps = np.exp(log_amps)

    # Conjugate the amplitudes and phases
    amp = np.concatenate([amps, np.array([1.0]), amps[::-1]], axis=0)
    phase = np.concatenate([phases, np.array([0.0]), -phases[::-1]], axis=0)

    # Reshape to the correct shape
    return amp.reshape(shape), phase.reshape(shape)


def create_vis(vis, otf_coords, uv_coords):
    # Convert from vis vector to amplitude and phase images
    amp, phase = vis_to_im(vis, otf_coords.shape[1:])

    # Build the complex visibility maps
    interp_fn = lambda im, uv, fill: interp(im, otf_coords, uv, method="linear", fill=fill)
    amps = vmap(lambda uv: interp_fn(amp, uv, fill=1.0))(uv_coords)
    phases = vmap(lambda uv: interp_fn(phase, uv, fill=0.0))(uv_coords)
    return amps * np.exp(1j * phases)


def inject_vis(wfs, vis, otf_coords):
    # Get the visibility array
    uv_coords = wf_fft_coords(wfs, pad=2)
    vis = create_vis(vis, otf_coords, uv_coords)

    # Fourier Functions
    n = uv_coords.shape[-1] // 4
    pad_fn = lambda x: np.pad(x, n, mode="constant")
    crop_fn = lambda x: x[n:-n, n:-n]
    to_uv = vmap(lambda x: np.fft.fftshift(np.fft.fft2(pad_fn(x))))
    from_uv = vmap(lambda x: crop_fn(np.fft.ifft2(np.fft.ifftshift(x))))

    # Apply the visibility maps
    splodges = to_uv(wfs.psf)
    applied = vis * splodges
    return np.abs(from_uv(applied)).sum(0)


def model_vis_psf(optics, vis, filter, aberrations, defocus, n_knots=51):
    # Update the optics to the correct aberrations and defocus
    optics = optics.set(
        ["pupil_mask.abb_coeffs", "defocus"],
        [aberrations, defocus],
    )

    # Model the wavefront
    wavels, weights = optics.filters[filter]
    wfs = eqx.filter_jit(optics.propagate)(wavels, weights=weights, return_wf=True)

    # Inject the visibility maps
    otf_coords = dlu.pixel_coords(n_knots, 2 * optics.diameter)
    return inject_vis(wfs, vis, otf_coords)


class LogVisModel(zdx.Base):
    V: dict
    otf_coords: np.ndarray
    n_knots: int = eqx.field(static=True)
    n_terms: int = eqx.field(static=True)

    def __init__(self, V, otf_coords, n_terms=100):
        self.V = jtu.map(lambda x: x[:n_terms], V)
        self.otf_coords = np.array(otf_coords, float)
        self.n_terms = int(n_terms)
        self.n_knots = int(otf_coords.shape[-1])

    # def model_vis(self, wfs, amps, phases, filter):
    def model_vis(self, wfs, latent_vis, filter):
        vis = np.dot(latent_vis, self.V[filter])
        psf = inject_vis(wfs, vis, self.otf_coords)
        return dl.PSF(psf, wfs.pixel_scale.mean(0))


def vis_jac_fn(model, exp):
    wfs = exp.model_wfs(model)
    wls = wfs.wavelength
    psf_pscale = wfs.pixel_scale[0]

    npix = 2 * model.vis_model.otf_coords.shape[-1]
    pscale = 0.5 * np.diff(model.vis_model.otf_coords[0, 0]).mean()

    # Fourier Functions
    to_uv = vmap(lambda arr, wl: dlu.MFT(arr, wl, psf_pscale, npix, pscale))
    downsample = vmap(lambda arr: dlu.downsample(arr, 2, mean=True))

    # Calculate the visibility maps
    vis = downsample(to_uv(wfs.psf, wls))
    vis = np.mean(vis, axis=0)
    vis = vis.flatten()[: vis.size // 2]

    # Project the resulting visibilities to the latent space
    cvis = np.concatenate([np.log10(np.abs(vis)), np.angle(vis)])
    V_inv = np.linalg.pinv(model.vis_model.V[exp.filter])
    return np.dot(cvis, V_inv)
