import pkg_resources as pkg
import jax.numpy as np
import dLux as dl
import dLux.utils as dlu


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


def model_ramp(psf, ngroups):
    """Applies an 'up the ramp' model of the input 'optical' PSF. Input PSF.data
    should have shape (npix, npix) and return shape (ngroups, npix, npix)"""
    lin_ramp = (np.arange(ngroups) + 1) / ngroups
    return psf[None, ...] * lin_ramp[..., None, None]


def model_dark_current(dark_current, ngroups):
    """Models the dark current as a constant background value added cumulatively to
    each group. For now we assume that the dark current is a float."""
    return (dark_current * (np.arange(ngroups) + 1))[..., None, None]


def model_exposure(model, exposure, to_BFE=False, slopes=False):
    # Get wavelengths and weights
    wavels, filt_weights = model.filters[exposure.filter]
    weights = filt_weights * planck(wavels, model.Teffs[exposure.star])
    weights = weights / weights.sum()

    optics = model.optics.set(
        ["pupil.coefficients", "pupil.opd"], [model.aberrations[exposure.key], exposure.opd]
    )

    if "coherence" in model.params.keys():
        coherence = model.coherence[exposure.key]
        optics = optics.set("holes.reflectivity", coherence)

    # Model the optics
    pos = dlu.arcsec2rad(model.positions[exposure.key])
    wfs = optics.propagate(wavels, pos, weights, return_wf=True)

    psfs = wfs.psf
    if model.visibilities is not None:
        psf = model.visibilities(psfs, exposure)
    else:
        psf = psfs.sum(0)

    # PSF is still unitary here
    psf = model.detector.apply(dl.PSF(psf, wfs.pixel_scale.mean(0)))

    # Get the hyper-parameters for the non-linear model
    flux = 10 ** model.fluxes[exposure.key]
    oversample = optics.oversample

    # Return the BFE and required meta-data
    if to_BFE:
        return psf, flux, oversample

    # Non linear model always goes from unit psf, flux, oversample to an 80x80 ramp
    if model.ramp is not None:
        ramp = model.ramp.apply(psf, flux, exposure, oversample)
    else:
        psf_data = dlu.downsample(psf.data * flux, oversample, mean=False)
        ramp = psf.set("data", model_ramp(psf_data, exposure.ngroups))

    # Model the read effects
    if "one_on_fs" in model.params.keys():
        one_on_fs = model.one_on_fs[exposure.key]
        ramp = model.read.set("one_on_fs", one_on_fs).apply(ramp)

    # Return the slopes if required
    if slopes:
        return np.diff(ramp.data, axis=0)

    # Return the ramp
    return ramp.data


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
