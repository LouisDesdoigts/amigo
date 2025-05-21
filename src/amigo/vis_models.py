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


class BaseLogVisModel(zdx.Base):
    V_amp: np.ndarray
    V_phase: np.ndarray
    otf_coords: np.ndarray
    n_knots: int = eqx.field(static=True)
    n_basis: int = eqx.field(static=True)

    def __init__(self, otf_coords, V_amp, V_phase, n_basis=500):
        # Populate the model
        self.n_basis = int(n_basis)
        self.n_knots = int(otf_coords.shape[-1])
        self.otf_coords = np.array(otf_coords, float)
        self.V_amp = np.array(V_amp, float)
        self.V_phase = np.array(V_phase, float)

    def upsample_uv(self, coords, log_amps, phases):
        interp_fn = lambda im, uv: interp(im, self.otf_coords, uv, method="linear", fill=0.0)
        log_amps = vmap(lambda uv: interp_fn(log_amps, uv))(coords)
        phases = vmap(lambda uv: interp_fn(phases, uv))(coords)
        return log_amps, phases

    def inject_to_wfs(self, wfs, log_amps, phases):
        # Get the bits for mapping to UV plane
        uv_coords = wf_fft_coords(wfs, pad=2)
        n = uv_coords.shape[-1] // 4  # Per edge - 2x pad
        crop_fn = lambda x: x[n:-n, n:-n]
        pad_fn = lambda x: np.pad(x, n, mode="constant")

        # Inject the visibility maps
        log_amps, phases = self.upsample_uv(uv_coords, log_amps, phases)
        cplx_vis = np.exp(log_amps + 1j * phases)

        # Apply through FFT
        to_uv = vmap(lambda x: np.fft.fftshift(np.fft.fft2(pad_fn(x))))
        from_uv = vmap(lambda x: crop_fn(np.fft.ifft2(np.fft.ifftshift(x))))
        vis_splodges = to_uv(wfs.psf) * cplx_vis
        return np.abs(from_uv(vis_splodges)).sum(0)

    def wfs_to_otf(self, wfs, oversample=2):
        # Get the bits for mapping to UV plane
        wls = wfs.wavelength
        psf_pscale = wfs.pixel_scale[0]
        npix = oversample * self.otf_coords.shape[-1]
        pscale = np.diff(self.otf_coords[0, 0]).mean() / oversample

        # Project to otf_coords (oversampled)
        to_uv = vmap(lambda arr, wl: dlu.MFT(arr, wl, psf_pscale, npix, pscale))
        downsample = vmap(lambda arr: dlu.downsample(arr, 2, mean=True))
        vis = downsample(to_uv(wfs.psf, wls))
        return np.mean(vis, axis=0)


class LogVisModel(BaseLogVisModel):
    V_amp: dict
    V_phase: dict
    otf_coords: np.ndarray
    n_knots: int = eqx.field(static=True)
    n_basis: int = eqx.field(static=True)

    def __init__(self, basis_dict, n_basis=500):
        # Load the values from the basis dictionary
        otf_coords = basis_dict["otf_coords"]
        V_amp = basis_dict["eigen_vectors"]["log_amp"]
        V_phase = basis_dict["eigen_vectors"]["phase"]

        # Populate the model
        self.n_basis = int(n_basis)
        self.n_knots = int(otf_coords.shape[-1])
        self.otf_coords = np.array(otf_coords, float)
        self.V_amp = jtu.map(lambda x: x[:n_basis], V_amp)
        self.V_phase = jtu.map(lambda x: x[:n_basis], V_phase)

    def latent_to_im(self, latent_amps, latent_phases, filter):
        # Project the latent amplitudes and phases to the the pixel space
        log_amps, phases = self.from_latent(latent_amps, latent_phases, filter)

        # Conjugate the amplitudes and phases
        log_amps = np.concatenate([log_amps, np.array([0.0]), log_amps[::-1]], axis=0)
        phases = np.concatenate([phases, np.array([0.0]), -phases[::-1]], axis=0)

        # Reshape to an image
        shape = self.otf_coords.shape[1:]
        return log_amps.reshape(shape), phases.reshape(shape)

    def latent_to_wf(self, wfs, latent_amps, latent_phases, filter):
        # log_amps, phases = self.project(latent_amps, latent_phases, filter)
        log_amps, phases = self.latent_to_im(latent_amps, latent_phases, filter)
        return self.inject_to_wfs(wfs, log_amps, phases)

    def from_latent(self, latent_amps, latent_phases, filter):
        log_amps = np.dot(latent_amps, self.V_amp[filter])
        log_phases = np.dot(latent_phases, self.V_phase[filter])
        return log_amps, log_phases

    def to_latent(self, log_amps, log_phases, filter):
        log_amps = np.dot(log_amps, np.linalg.pinv(self.V_amp[filter]))
        log_phases = np.dot(log_phases, np.linalg.pinv(self.V_phase[filter]))
        return log_amps, log_phases

    def model_vis(self, wfs, latent_amps, latent_phases, filter):
        psf = self.latent_to_wf(wfs, latent_amps, latent_phases, filter)
        return dl.PSF(psf, wfs.pixel_scale.mean(0))

    def wfs_to_latent(self, wfs, filter):
        vis = self.wfs_to_otf(wfs)
        vis = vis.flatten()[: vis.size // 2]

        # # Project the resulting visibilities to the latent space
        log_vis = np.log(vis)
        return self.to_latent(log_vis.real, log_vis.imag, filter)


def vis_jac_fn(model_params, args):
    optics, vis_model, filter = args

    # Populate the optics with any bits we want
    for key, value in model_params.items():
        if hasattr(optics, key):
            optics = optics.set(key, value)

    # Get the spectrally weighted wavelengths
    wavels, filt_weights = optics.filters[filter]
    xs = np.linspace(-1, 1, len(wavels), endpoint=True)
    spectra_slopes = 1 + model_params.spectra * xs
    weights = filt_weights * spectra_slopes
    weights /= weights.sum()

    # Apply flux if in there
    if "flux" in model_params.keys():
        weights *= 10**model_params.flux

    # Propagate the wavefront and project to the latent space
    if "positions" in model_params.keys():
        wavels, weights = optics.filters[filter]
        offset = dlu.arcsec2rad(model_params.positions)
        wfs = optics.propagate(wavels, offset, weights, return_wf=True)
    else:
        wfs = optics.propagate(wavels, weights=weights, return_wf=True)
    return vis_model.wfs_to_latent(wfs, filter)
