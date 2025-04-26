import zodiax as zdx
import dLux as dl
import jax.tree as jtu
import jax.numpy as np
import equinox as eqx
import dLux.utils as dlu
from amigo.misc import interp
from jax import vmap


def fft_coords(wl, npix, pscale, pad=2):
    x = np.fft.fftshift(np.fft.fftfreq(pad * npix, d=pscale / wl))
    return np.array(np.meshgrid(x, x))


def wf_fft_coords(wfs, pad=2):
    wls = wfs.wavelength
    psf_pscale = wfs.pixel_scale[0]
    psf_npix = wfs.npixels
    return vmap(lambda wl: fft_coords(wl, psf_npix, psf_pscale, pad=pad))(wls)


def vis_to_im(amps, phases, shape):
    # Conjugate the amplitudes and phases
    amp = 1.0 + np.concatenate([amps, np.array([0.0]), amps[::-1]], axis=0)
    phase = np.concatenate([phases, np.array([0.0]), -phases[::-1]], axis=0)
    return amp.reshape(shape), phase.reshape(shape)


def inject_vis(psfs, amps, phases, otf_coords, uv_coords):
    # Get the amplitudes and phases
    amp, phase = vis_to_im(amps, phases, otf_coords.shape[1:])

    # Build the visibility maps
    interp_fn = lambda im, uv, fill: interp(im, otf_coords, uv, method="linear", fill=fill)
    amps = vmap(lambda uv: interp_fn(amp, uv, fill=1.0))(uv_coords)
    phases = vmap(lambda uv: interp_fn(phase, uv, fill=0.0))(uv_coords)
    cplx = amps * np.exp(1j * phases)

    # Fourier Functions
    n = uv_coords.shape[-1] // 4
    pad_fn = lambda x: np.pad(x, n, mode="constant")
    crop_fn = lambda x: x[n:-n, n:-n]
    to_uv = vmap(lambda x: np.fft.fftshift(np.fft.fft2(pad_fn(x))))
    from_uv = vmap(lambda x: crop_fn(np.fft.ifft2(np.fft.ifftshift(x))))

    # Apply the visibility maps
    splodges = to_uv(psfs)
    applied = cplx * splodges
    return np.abs(from_uv(applied)).sum(0)


def model_vis_psf(optics, amp, phase, filter, aberrations, defocus, n_knots=51):
    #
    optics = optics.set(
        ["pupil_mask.abb_coeffs", "defocus"],
        [aberrations, defocus],
    )

    #
    wavels, weights = optics.filters[filter]
    wfs = eqx.filter_jit(optics.propagate)(wavels, weights=weights, return_wf=True)

    #
    otf_coords = dlu.pixel_coords(n_knots, 2 * optics.diameter)
    uv_coords = wf_fft_coords(wfs, pad=2)
    return inject_vis(wfs.psf, amp, phase, otf_coords, uv_coords)


def project(amps, phases, vis_model, filt):
    amps = np.dot(amps, vis_model.amp_matrix[filt])
    phases = np.dot(phases, vis_model.phase_matrix[filt])
    return amps, phases


def model_vis(model, exp):
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
    return vis


class VisModel(zdx.Base):
    amp_matrix: dict
    phase_matrix: dict
    otf_coords: np.ndarray
    n_knots: int = eqx.field(static=True)
    n_terms: int = eqx.field(static=True)

    def __init__(self, amp_matrix, phase_matrix, otf_coords, n_terms=100):
        self.amp_matrix = jtu.map(lambda x: x[:n_terms], amp_matrix)
        self.phase_matrix = jtu.map(lambda x: x[:n_terms], phase_matrix)
        self.otf_coords = np.array(otf_coords, float)
        self.n_terms = int(n_terms)
        self.n_knots = int(otf_coords.shape[-1])

    def model_vis(self, wfs, amps, phases, filter):
        uv_coords = wf_fft_coords(wfs, pad=2)
        amp = np.dot(amps, self.amp_matrix[filter])
        phase = np.dot(phases, self.phase_matrix[filter])
        psf = inject_vis(wfs.psf, amp, phase, self.otf_coords, uv_coords)
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

    #
    amps = np.abs(vis)
    phases = np.angle(vis)

    #
    amp_mat = np.linalg.pinv(model.vis_model.amp_matrix[exp.filter])
    phase_mat = np.linalg.pinv(model.vis_model.phase_matrix[exp.filter])
    return np.dot(amps, amp_mat), np.dot(phases, phase_mat)
