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


def vis_to_im(log_amps, phases, shape):
    # Conjugate the amplitudes and phases
    log_amps = np.concatenate([log_amps, np.array([0.0]), log_amps[::-1]], axis=0)
    phases = np.concatenate([phases, np.array([0.0]), -phases[::-1]], axis=0)
    return log_amps.reshape(shape), phases.reshape(shape)


def inject_vis(wfs, log_amps, phases, otf_coords):
    # Get the amplitudes and phases
    log_amps, phase = vis_to_im(log_amps, phases, otf_coords.shape[1:])

    # Interpolate the visibility maps to uv coordinates
    uv_coords = wf_fft_coords(wfs, pad=2)
    interp_fn = lambda im, uv: interp(im, otf_coords, uv, method="linear", fill=0.0)
    log_amps = vmap(lambda uv: interp_fn(log_amps, uv))(uv_coords)
    phases = vmap(lambda uv: interp_fn(phase, uv))(uv_coords)

    # Fourier Functions (use 2x pad)
    n = uv_coords.shape[-1] // 4
    crop_fn = lambda x: x[n:-n, n:-n]
    pad_fn = lambda x: np.pad(x, n, mode="constant")
    to_uv = vmap(lambda x: np.fft.fftshift(np.fft.fft2(pad_fn(x))))
    from_uv = vmap(lambda x: crop_fn(np.fft.ifft2(np.fft.ifftshift(x))))

    # Apply the visibility maps
    splodges = to_uv(wfs.psf) * np.exp(log_amps + 1j * phases)
    return np.abs(from_uv(splodges)).sum(0)


def project(latent_vis, V):
    vis_vec = np.dot(latent_vis, V)
    log_amp, phase = np.array_split(vis_vec, 2, axis=0)
    return log_amp, phase


# class LogVisModel(zdx.Base):
#     V: dict
#     otf_coords: np.ndarray
#     n_knots: int = eqx.field(static=True)
#     n_terms: int = eqx.field(static=True)

#     def __init__(self, V, otf_coords, n_terms=100):
#         self.n_terms = int(n_terms)
#         self.n_knots = int(otf_coords.shape[-1])
#         self.V = jtu.map(lambda x: x[:n_terms], V)
#         self.otf_coords = np.array(otf_coords, float)

#     def model_vis(self, wfs, latent_vis, filter):
#         log_amps, phases = project(latent_vis, self.V[filter])
#         psf = inject_vis(wfs, log_amps, phases, self.otf_coords)
#         return dl.PSF(psf, wfs.pixel_scale.mean(0))


class LogVisModel(zdx.Base):
    V_amp: dict
    V_phase: dict
    otf_coords: np.ndarray
    n_knots: int = eqx.field(static=True)
    n_terms: int = eqx.field(static=True)

    def __init__(self, V_amp, V_phase, otf_coords, n_terms=100):
        self.n_terms = int(n_terms)
        self.n_knots = int(otf_coords.shape[-1])
        self.otf_coords = np.array(otf_coords, float)
        self.V_amp = jtu.map(lambda x: x[:n_terms], V_amp)
        self.V_phase = jtu.map(lambda x: x[:n_terms], V_phase)

    def model_vis(self, wfs, log_amps, phases, filter):
        # log_amps, phases = project(latent_vis, self.V[filter])
        log_amps = np.dot(log_amps, self.V_amp[filter])
        phases = np.dot(phases, self.V_phase[filter])
        psf = inject_vis(wfs, log_amps, phases, self.otf_coords)
        return dl.PSF(psf, wfs.pixel_scale.mean(0))


# class LogVisModel(zdx.Base):
#     V: dict
#     otf_coords: np.ndarray
#     n_knots: int = eqx.field(static=True)
#     n_terms: int = eqx.field(static=True)

#     def __init__(self, V, otf_coords, n_terms=100):
#         self.n_terms = int(n_terms)
#         self.n_knots = int(otf_coords.shape[-1])
#         self.V = jtu.map(lambda x: x[:n_terms], V)
#         self.otf_coords = np.array(otf_coords, float)

#     def model_vis(self, wfs, latent_vis, filter):
#         log_amps, phases = project(latent_vis, self.V[filter])
#         psf = inject_vis(wfs, log_amps, phases, self.otf_coords)
#         return dl.PSF(psf, wfs.pixel_scale.mean(0))


def model_vis_psf(optics, latent_vis, filter, aberrations, defocus, n_knots=51):
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
    log_amps, phases = np.array_split(latent_vis, 2, axis=0)
    return inject_vis(wfs, log_amps, phases, otf_coords)


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

    # # Project the resulting visibilities to the latent space
    log_vis = np.log(vis)
    vis_vec = np.concatenate([log_vis.real, log_vis.imag])
    V_inv = np.linalg.pinv(model.vis_model.V[exp.filter])
    return np.dot(vis_vec, V_inv)

    # # Project the resulting visibilities to the latent space
    # vis_vec = np.concatenate([np.log10(np.abs(vis)), np.angle(vis)])
    # V_inv = np.linalg.pinv(model.vis_model.V[exp.filter])
    # return np.dot(vis_vec, V_inv)
