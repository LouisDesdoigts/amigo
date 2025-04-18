from jax import vmap
import jax.numpy as np
import equinox as eqx
import dLux.utils as dlu
from amigo.misc import interp


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


# def project_vis_map(vis, proj_matrix):
#     # Get the first half of the visibilities, since the second half is conjugate
#     vis = vis.flatten()
#     vis = vis[: len(vis) // 2]

#     # Build the amplitudes and phases into a vector to project to the latest space
#     amps, phases = np.abs(vis), np.angle(vis)
#     cvis = np.concatenate([amps, phases], axis=0)

#     # Project amplitudes and phases to latent visibilities
#     # proj_matrix = model.vis_model.proj_matrix[exp.filter]
#     inv_proj_mat = np.linalg.pinv(proj_matrix)
#     latent_vis = np.dot(cvis, inv_proj_mat)
#     return latent_vis


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


# def model_latent_vis_phases(model, exp):
#     # Get the model visibilities
#     vis = model_vis(model, exp)

#     # Get the projected visibility phases
#     proj_matrix = model.vis_model.proj_matrix[exp.filter]
#     latent_vis = project_vis_map(vis, proj_matrix)
#     amps, phases = np.array_split(latent_vis, 2)
#     return phases  # vis_vals


import zodiax as zdx
import dLux as dl
import jax.tree as jtu


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


# import equinox as eqx
# import zodiax as zdx
# import jax.numpy as np
# import jax.tree as jtu
# import dLux as dl
# import dLux.utils as dlu
# from jax import vmap
# from jax.scipy.signal import correlate
# from .misc import interp, nearest_fn


# def get_hole_mask(pt, mask, coords, k=100):
#     inds = np.where(nearest_fn(pt, coords))
#     i, j = inds[0][0], inds[1][0]
#     sy, ey, sx, ex = i - k, i + k, j - k, j + k
#     return mask.at[sy:ey, sx:ex].set(True)


# # def calc_splodge_masks(optics, thresh=1e-10, downsample=8):
# #     holes = optics.holes
# #     mask = optics.calc_mask(optics.wf_npixels, optics.diameter)
# #     coords = dlu.pixel_coords(optics.wf_npixels, optics.diameter)

# #     splodge_masks = {}
# #     for i in range(len(holes)):
# #         for j in range(len(holes)):
# #             if i == j or (j, i) in splodge_masks.keys():
# #                 continue

# #             pt1, pt2 = holes[i], holes[j]
# #             hole_mask = np.zeros_like(mask, bool)
# #             hole_mask = get_hole_mask(pt1, hole_mask, coords)
# #             hole_mask = get_hole_mask(pt2, hole_mask, coords)

# #             reduced_mask = dlu.downsample(mask * hole_mask, downsample)
# #             corr = correlate(reduced_mask, reduced_mask, mode="full", method="fft")
# #             corr /= corr.max()
# #             splodge_masks[(i, j)] = (corr > thresh).astype(float)
# #     return splodge_masks


# def build_vis_pts(amp_vec, pha_vec, shape):
#     vis_vec = amp_vec * np.exp(1j * pha_vec)
#     dc = np.array([np.exp(0j)])
#     return np.concatenate([vis_vec, dc, vis_vec.conj()[::-1]]).reshape(shape)


# def find_knot_map(full_knot_map):
#     # Set the second half of the array to False (since visibilities conjugate)
#     full_flat_knot_map = np.array(full_knot_map.flatten())
#     flat_knots_map = full_flat_knot_map.at[: (len(full_flat_knot_map) // 2) + 1].set(False)
#     return flat_knots_map.reshape(full_knot_map.shape)


# def find_knot_inds(full_knot_map):
#     flat_knot_map = full_knot_map.flatten()
#     n = len(flat_knot_map) // 2
#     return np.where(flat_knot_map[:n])


# def calc_splodge_masks(optics, thresh=1e-3):
#     from amigo.vis_models_old import get_hole_mask

#     holes = optics.holes
#     mask = optics.calc_mask(optics.wf_npixels, optics.diameter)
#     mask = dlu.downsample(mask, 8)
#     coords = dlu.pixel_coords(len(mask), optics.diameter)

#     splodge_masks = {}
#     for i in range(len(holes)):
#         for j in range(len(holes)):
#             if i == j or (j, i) in splodge_masks.keys():
#                 continue

#             pt1, pt2 = holes[i], holes[j]
#             hole_mask = np.zeros_like(mask, bool)
#             hole_mask = get_hole_mask(pt1, hole_mask, coords, k=12)
#             hole_mask = get_hole_mask(pt2, hole_mask, coords, k=12)

#             reduced_mask = mask * hole_mask
#             corr = correlate(reduced_mask, reduced_mask, method="fft")
#             corr /= corr.max()
#             if thresh is not None:
#                 corr = (corr > thresh).astype(float)
#             splodge_masks[(i, j)] = corr
#     return splodge_masks


# class MFTVis(zdx.Base):
#     method: str = eqx.field(static=True)
#     otf_coords: np.ndarray
#     knot_coords: np.ndarray
#     otf_mask: np.ndarray
#     knot_map: np.ndarray
#     conj_map: np.ndarray
#     knot_inds: np.ndarray
#     uv_pscale: np.ndarray

#     def __init__(self, optics, method="linear", sample_factor=5):
#         if method not in ["linear"]:
#             raise ValueError("Presently only 'linear' is supported.")

#         if sample_factor not in [3, 5]:
#             raise ValueError("Under sample factor must be 3 or 5")

#         self.method = method

#         # Calculate the Aperture mask
#         downsample = 8  # 16 is the minimal downsample that doesn't alias
#         mask = optics.calc_mask(optics.wf_npixels, optics.diameter)
#         mask = dlu.downsample(mask, downsample)

#         # Get the fourier pixel scale
#         self.uv_pscale = np.array(downsample * optics.diameter / optics.wf_npixels)

#         # Calculate the OTF mask
#         thresh = 1e-10
#         corr = correlate(mask, mask, mode="full", method="fft")
#         self.otf_mask = corr > (thresh * corr.max())

#         n = len(self.otf_mask)
#         n_knots = n // sample_factor
#         otf_diam = 2 * optics.diameter

#         self.otf_coords = dlu.pixel_coords(n, otf_diam)
#         self.knot_coords = dlu.pixel_coords(n_knots, otf_diam)

#         otf_support = dlu.downsample(self.otf_mask, sample_factor)
#         full_knot_map = otf_support > 0.55
#         # self.knot_inds = np.array(find_knot_inds(full_knot_map))

#         # Get the splodge masks
#         masks = np.stack(jtu.leaves(calc_splodge_masks(optics)))
#         mean_otf_support = dlu.downsample(masks.sum(0), 5, mean=True)
#         otf_mask = (mean_otf_support >= 1.0).astype(int)

#         # Get the splodge masks
#         masks = np.stack(jtu.leaves(calc_splodge_masks(optics, thresh=None)))
#         otf_support = dlu.downsample(masks.sum(0), 5, mean=True)
#         corr_mask = otf_support > 0.02

#         #
#         knot_map = find_knot_map(full_knot_map).astype(int)
#         true_knot_map = (knot_map + otf_mask + corr_mask) == 3
#         self.knot_map = true_knot_map
#         self.conj_map = np.flip(true_knot_map.flatten()).reshape(true_knot_map.shape)
#         self.knot_inds = np.array(find_knot_inds(self.knot_map + self.conj_map))

#     @property
#     def valid_knot_coords(self):
#         return self.knot_coords[:, self.knot_map]

#     @property
#     def conj_knot_coords(self):
#         return self.knot_coords[:, self.conj_map]

#     def interp_fn(self, im):
#         return interp(im, self.knot_coords, self.otf_coords, method=self.method)

#     def get_vis_map(self, amplitudes, phases):
#         # Full the full array with the input amplitudes and phases
#         n_pts = self.knot_map.size // 2
#         amps = np.ones(n_pts).at[*self.knot_inds].set(amplitudes)
#         phases = np.zeros(n_pts).at[*self.knot_inds].set(phases)

#         # Get the amplitude and phase visibility maps
#         vis_vals = build_vis_pts(amps, phases, self.knot_map.shape)
#         amplitude = self.interp_fn(np.abs(vis_vals))
#         phase = self.interp_fn(np.angle(vis_vals))

#         # Return as a complex visibility surface
#         return amplitude * np.exp(1j * phase)

#     def model_vis(self, wfs, amps, phases):
#         vis_maps = self.get_vis_map(amps, phases)

#         # Get the bits we need
#         wls = wfs.wavelength
#         psf_pscale = wfs.pixel_scale[0]
#         psf_npix = wfs.npixels
#         uv_npix = len(self.otf_mask)

#         # Fourier Functions
#         to_uv = vmap(lambda arr, wl: dlu.MFT(arr, wl, psf_pscale, uv_npix, self.uv_pscale))
#         from_uv = vmap(
#             lambda arr, wl: dlu.MFT(arr, wl, self.uv_pscale, psf_npix, psf_pscale, inverse=True)
#         )

#         # Masking fn
#         mask_fn = vmap(lambda x, y: np.where(self.otf_mask, x, y))

#         # Apply the visibility maps
#         splodges = to_uv(wfs.psf, wls)
#         applied = mask_fn(splodges * vis_maps, splodges)
#         psfs = np.abs(from_uv(applied, wls))

#         # Return the PSF object
#         return dl.PSF(psfs.sum(0), wfs.pixel_scale.mean(0))
