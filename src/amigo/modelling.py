import equinox as eqx
import dLux.utils as dlu
from jax import vmap
from amigo.misc import planck
from amigo.stats import total_read_noise, total_amplifier_noise
from amigo.interferometry import visibilities, uv_model
from amigo.detector_layers import model_ramp, model_amplifier, model_dark_current
from jax.scipy.signal import convolve
import jax.numpy as np
import jax.tree_util as jtu


def model_fn(model, exposure, with_BFE=True, to_BFE=False, zero_idx=-1, noise=True):
    # Get exposure key
    key = exposure.key

    # Get wavelengths and weights
    wavels, filt_weights = model.filters[exposure.filter]
    weights = filt_weights * planck(wavels, model.Teffs[exposure.star])
    weights *= (10 ** model.fluxes[key]) / weights.sum()
    # weights *= model.fluxes[key] / weights.sum()

    # Apply correct aberrations
    aberrations = model.aberrations[key]
    if zero_idx != -1:
        aberrations = aberrations.at[zero_idx, 0].set(0.0)  # Pin piston to zero

    optics = model.optics.set(
        ["coefficients", "pupil.opd"],
        [aberrations, exposure.opd],
    )

    # Per exposure mirror coherence
    if hasattr(model, "coherence"):
        optics = optics.set("holes.reflectivity", model.coherence[key])

    # Make sure this has correct position units and get wavefronts
    pos_rad = dlu.arcsec2rad(model.positions[key])
    # PSF = optics.propagate(wavels, pos_rad, weights, return_psf=True)
    wfs = optics.propagate(wavels, pos_rad, weights, return_wf=True)

    if hasattr(model, "masks"):
        # Get visibilities and final psfs
        vis_key = "_".join([exposure.star, exposure.filter])
        mask = model.masks[exposure.filter]
        amplitudes = model.amplitudes[vis_key]
        phases = model.phases[vis_key]
        vis = visibilities(amplitudes, phases)
        psf = uv_model(vis, wfs.psf, mask).sum(0)
        PSF = dl.PSF(psf, wfs.pixel_scale.mean(0))
    else:
        PSF = dl.PSF(wfs.psf.sum(0), wfs.pixel_scale.mean(0))

    # Apply the detector model and turn it into a ramp
    psf = model.detector.model(PSF)

    # Model the ramp
    ramp = model_ramp(psf, exposure.ngroups)

    # # Add the bias - Method 1, estimate from model
    # first_group_est = ramp[0]
    # bias_est = exposure.zero_point - dlu.downsample(first_group_est, 4, mean=False)
    # bias_est = np.repeat(np.repeat(bias_est, 4, axis=0), 4, axis=1)
    # bias_est /= 16 # Normalise the subsampling
    # bias_est = np.where(np.isnan(bias_est), 0.0, bias_est)

    # # Add the bias - Method 2, estimate from data
    # bias_est = exposure.zero_point - exposure.data[0]
    # bias_est = np.repeat(np.repeat(bias_est, 4, axis=0), 4, axis=1) / 16
    # bias_est = np.where(np.isnan(bias_est), 0.0, bias_est)

    # # Add the estimated bias to the ramp
    # ramp += bias_est[None, ...]

    if to_BFE:
        return ramp

    # Now apply the CNN BFE and downsample
    if with_BFE:
        # if test_BFE:
        if hasattr(model.BFE, "gru_cell"):
            flux = 10 ** model.fluxes[key]
            norm_psf = psf / flux
            ramp = model.BFE.model(norm_psf, flux, exposure.ngroups)
            dsample_fn = lambda x: dlu.downsample(x, 4, mean=False)
            ramp = vmap(dsample_fn)(ramp)
        else:
            ramp = eqx.filter_vmap(model.BFE.apply_array)(ramp)
    else:
        dsample_fn = lambda x: dlu.downsample(x, 4, mean=False)
        ramp = vmap(dsample_fn)(ramp)

    # Ensure we always have 80x80 images
    ramp = vmap(dlu.resize, (0, None))(ramp, 80)

    # Apply IPC
    if hasattr(model.detector, "ipc"):
        ipc_fn = lambda x: convolve(x, model.detector.ipc, mode="same")
        ramp = vmap(ipc_fn)(ramp)

    # Model the dark current
    ramp = model_dark_current(ramp, model.detector.dark_current)

    # Apply one of F model
    if noise:
        ramp += total_amplifier_noise(model.one_on_fs[key])

    # return ramp
    return np.diff(ramp, axis=0)


def build_optical_inputs(model, exposures, zero_idx=-1):

    wavels_in = []
    weights_in = []
    aberrations_in = []
    opds_in = []
    positions_in = []
    for exposure in exposures:
        # Get exposure key
        key = exposure.key

        # Get wavelengths and weights
        wavels, filt_weights = model.filters[exposure.filter]
        weights = filt_weights * planck(wavels, model.Teffs[exposure.star])
        weights *= (10 ** model.fluxes[key]) / weights.sum()

        wavels_in.append(wavels)
        weights_in.append(weights)

        # Apply correct aberrations
        aberrations = model.aberrations[key]
        if zero_idx != -1:
            aberrations = aberrations.at[zero_idx, 0].set(0.0)  # Pin piston to zero

        aberrations_in.append(aberrations)
        opds_in.append(exposure.opd)

        # Make sure this has correct position units and get wavefronts
        pos_rad = dlu.arcsec2rad(model.positions[key])

        positions_in.append(pos_rad)

    return wavels_in, weights_in, aberrations_in, opds_in, positions_in


def model_optics(model, wavels, weights, aberrations, opd, position):
    optics = model.optics.set(["coefficients", "pupil.opd"], [aberrations, opd])
    return optics.propagate(wavels, position, weights, return_psf=True)


def rebuild_ramps(ramps, bled_images):
    lengths = [len(ramp) for ramp in ramps]

    n = 0
    ramps = []
    for length in lengths:
        ramps.append(bled_images[n : n + length])
        n += length
    return ramps


import dLux as dl


# TODO: can probably be improved here to calc all the one on fs ones, using
# 'rebuild_ramps' function
def add_noise(model, exposures, ramps):
    coeffs = [model.one_on_fs[exp.key] for exp in exposures]
    add_noise = lambda ramp, coeff: ramp + vmap(model_amplifier)(coeff)
    return [add_noise(ramp, coeff) for ramp, coeff in zip(ramps, coeffs)]


def fast_model_fn(model, exposures, with_BFE=True, to_BFE=False, zero_idx=-1, noise=True):
    # Model optics
    optical_inputs = build_optical_inputs(model, exposures)
    is_leaf = lambda x: isinstance(x, list)
    optical_model_fn = lambda *args: model_optics(model, *args)
    PSFs = jtu.tree_map(optical_model_fn, optical_inputs, is_leaf=is_leaf)

    # Model detector
    is_leaf = lambda x: isinstance(x, dl.PSF)
    # psfs = jtu.tree_map(model.detector.model, PSFs, is_leaf=is_leaf)
    psfs = [model.detector.model(psf) for psf in PSFs]

    # Model ramp
    is_leaf = lambda x: isinstance(x, list)
    ramp_fn = lambda x, ngroup: model_ramp(x, ngroup)
    ngroups = [exp.ngroups for exp in exposures]
    # ramps = jtu.tree_map(ramp_fn, psfs, ngroups, is_leaf=is_leaf)
    ramps = [ramp_fn(psf, ngroup) for psf, ngroup in zip(psfs, ngroups)]

    # Return non-BFEd
    if to_BFE:
        return ramps

    # Apply BFE (or not)
    images = np.concatenate(ramps, axis=0)
    if with_BFE:
        images = eqx.filter_vmap(model.BFE.apply_array)(images)
    else:
        dsample_fn = lambda x: dlu.downsample(x, 4, mean=False)
        images = vmap(dsample_fn)(images)
    images = vmap(dlu.resize, (0, None))(images, 80)

    # Rebuild into per-exposure ramps
    ramps = rebuild_ramps(ramps, images)

    # Apply bias and one of F correction
    if noise:
        ramps = add_noise(model, exposures, ramps)

    # return ramp
    return jtu.tree_map(lambda x: np.diff(x, axis=0), ramps)
