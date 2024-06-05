import pkg_resources as pkg
import jax.numpy as np
import jax.tree_util as jtu
import equinox as eqx
import dLux as dl
import dLux.utils as dlu
from jax import vmap
from .interferometry import visibilities, uv_model
from .detector_layers import model_ramp, model_amplifier


def planck(wav, T):
    """
    Planck's Law:
    I(W, T) = (2hc^2 / W^5) * (1 / (exp{hc/WkT} - 1))
    where
    h = Planck's constant
    c = speed of light
    k = Boltzmann's constant

    W = wavelength array
    T = effective temperature

    Here A is the first fraction and B is the second fraction.
    The calculation is (sort of) performed in log space.
    """
    logW = np.log10(wav)  # wavelength array
    logT = np.log10(T)  # effective temperature

    # -15.92... is [log2 + logh + 2*logc]
    logA = -15.92347606 - 5 * logW
    logB = -np.log10(
        np.exp(
            # -1.84...is logh + logc - logk
            np.power(10, -1.8415064 - logT - logW)
        )
        - 1.0
    )

    return np.power(10, logA + logB)


def model_optics(
    optics,
    pos,
    wavels,
    weights,
    aberrations=None,
    opd=None,
    amplitudes=None,
    phases=None,
    mask=None,
    coherence=None,
):
    if opd is not None:
        optics = optics.set("pupil.opd", opd)

    if aberrations is not None:
        optics = optics.set("coefficients", aberrations)

    if coherence is not None:
        optics = optics.set("holes.reflectivity", coherence)

    wfs = optics.propagate(wavels, dlu.arcsec2rad(pos), weights, return_wf=True)
    psfs = wfs.psf

    if amplitudes is not None:
        vis = visibilities(amplitudes, phases)
        psf = uv_model(vis, psfs, mask).sum(0)
    else:
        psf = psfs.sum(0)
    return dl.PSF(psf, wfs.pixel_scale.mean(0))


def model_detector(
    detector, psf, flux, ngroups, filter, dark_current=None, one_on_fs=None, to_BFE=False
):

    detector = detector.set(["EDM.ngroups", "EDM.flux", "EDM.filter"], [ngroups, flux, filter])

    if one_on_fs is not None:
        detector = detector.set("amplifier.one_on_fs", one_on_fs)

    if dark_current is not None:
        detector = detector.set("dark_current", dark_current)

    for key, layer in detector.layers.items():
        if key == "EDM" and to_BFE:
            return psf.data
        psf = layer.apply(psf)
    return psf.data


def variance_model(model, exposure, true_read_noise=False, read_noise=10):
    """
    True read noise will use the CRDS read noise array, else it will use a constant
    value as determined by the input. true_read_noise therefore supersedes read_noise.
    Using a flat value of 10 seems to be more accurate that the CRDS array.

    That said I think the data has overly ambitious variances as a consequence of the
    sigma clipping that is performed. We could determine the variance analytically from
    the variance of the individual pixel values, but we will look at this later.
    """

    nan_mask = np.isnan(exposure.data)

    # Estimate the photon covariance
    # psf = model_fn(model, exposure)
    psf = model.model(exposure, slopes=True)

    psf = psf.at[np.where(nan_mask)].set(np.nan)
    variance = psf / exposure.nints

    # Read noise covariance
    if true_read_noise:
        rn = np.load(pkg.resource_filename(__name__, "data/SUB80_readnoise.npy"))
    else:
        rn = read_noise
    read_variance = (rn**2) * np.ones((80, 80)) / exposure.nints
    variance += read_variance[None, ...]

    return psf, variance


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


def rebuild_ramps(ramps, bled_images):
    lengths = [len(ramp) for ramp in ramps]

    n = 0
    ramps = []
    for length in lengths:
        ramps.append(bled_images[n : n + length])
        n += length
    return ramps


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
