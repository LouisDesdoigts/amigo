import pkg_resources as pkg
import jax.numpy as np


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


# def model_ramp(psf, ngroups):
#     """Applies an 'up the ramp' model of the input 'optical' PSF. Input PSF.data
#     should have shape (npix, npix) and return shape (ngroups, npix, npix)"""
#     lin_ramp = (np.arange(ngroups) + 1) / ngroups
#     return psf[None, ...] * lin_ramp[..., None, None]


# def model_dark_current(dark_current, ngroups):
#     """Models the dark current as a constant background value added cumulatively to
#     each group. For now we assume that the dark current is a float."""
#     return (dark_current * (np.arange(ngroups) + 1))[..., None, None]


# def model_exposure(model, exposure, to_BFE=False, slopes=False):
#     # Get wavelengths and weights
#     wavels, filt_weights = model.filters[exposure.filter]
#     weights = filt_weights * planck(wavels, model.Teffs[exposure.star])
#     weights = weights / weights.sum()

#     # optics = model.optics.set(
#     #     ["pupil.coefficients", "pupil.opd"],
#           [model.aberrations[exposure.abb_key], exposure.opd]
#     # )

#     # optics = model.optics.set(
#     #     ["pupil_mask.abb_coeffs", "pupil.opd"],
#     #     [model.aberrations[exposure.abb_key], exposure.opd],
#     # )

#     optics = model.optics.set("pupil_mask.abb_coeffs", model.aberrations[exposure.abb_key])

#     # print(optics.pupil_mask.abb_coeffs)

#     if "reflectivity" in model.params.keys():
#         optics = optics.set(["pupil_mask.amp_coeffs"], [model.reflectivity[exposure.amp_key]])

#     # if "coherence" in model.params.keys():
#     #     coherence = model.coherence[exposure.key]
#     #     optics = optics.set("holes.reflectivity", coherence)

#     if exposure.type == "binary":
#         # Convert binary parameters to positions parameters
#         position = model.positions[exposure.key]

#         #
#         pos_angle = dlu.deg2rad(model.position_angles[exposure.star])
#         r = model.separations[exposure.star] / 2
#         sep_vec = np.array([r * np.sin(pos_angle), r * np.cos(pos_angle)])
#         positions = np.array([position + sep_vec, position - sep_vec])

#         # Convert to radians
#         poses = vmap(dlu.arcsec2rad)(positions)

#         prop_fn = lambda pos: optics.propagate(wavels, pos, weights, return_wf=True)
#         wfs = eqx.filter_jit(eqx.filter_vmap(prop_fn))(poses)

#         # Get the fluxes - use unit flux here to simply get normalised relative fluxes
#         contrast = 10 ** model.contrasts[exposure.star]
#         flux_weights = np.array([contrast * 1, 1]) / (1 + contrast)

#         # Wfs is vectorised now, so all attributes will have an extra dimension
#         # Also need to divide by 2 to keep it a unit psf
#         psf = (flux_weights[:, None, None] * wfs.psf.sum(1)).sum(0)
#         pixel_scale = wfs.pixel_scale.mean((0, 1))

#     else:

#         # Model the optics
#         pos = dlu.arcsec2rad(model.positions[exposure.key])

#         # NOTE: Dispersion only implemented for point source right now
#         if model.dispersion is not None:
#             # Dispersion contrast
#             contrast = 10**model.contrast
#             flux_weights = np.array([contrast * 1, 1]) / (1 + contrast)

#             # Model the proper psf
#             wfs = eqx.filter_jit(optics.propagate)(wavels, pos, weights, return_wf=True).psf
#             primary_psfs = flux_weights[0] * wfs.psf

#             wf_prop = lambda *args: optics.propagate_mono(*args, return_wf=True)
#             prop_fn = lambda wav, disp: wf_prop(wav, pos + disp)

#             # # This one does free-floating (x, y)
#             # dispersion = dlu.arcsec2rad(model.dispersion[exposure.filter])

#             # This one does furthest point (x, y)
#             xmax, ymax = dlu.arcsec2rad(model.dispersion[exposure.filter])
#             xs = np.linspace(-xmax, xmax, len(wavels))
#             ys = np.linspace(-ymax, ymax, len(wavels))
#             dispersion = np.array([xs, ys]).T

#             # Apply it
#             wfs = eqx.filter_jit(eqx.filter_vmap(prop_fn))(wavels, dispersion)
#             wfs = wfs.multiply("amplitude", weights[:, None, None] ** 0.5)
#             secondary_psfs = flux_weights[1] * wfs.psf
#             psfs = primary_psfs + secondary_psfs

#         elif hasattr(model, "is_polarised") and model.is_polarised:

#             polarisation_keys = [
#                 "pupil.coefficients",
#                 "coherence.reflectivity",
#                 "pupil_mask.holes",
#                 "pupil_mask.f2f",
#                 "pupil_mask.transformation",
#             ]

#             # Partition the optics - assumed the model is already partition-vectorised
#             filter_spec = zdx.boolean_filter(optics, polarisation_keys)
#             polarised, unpolarised = eqx.partition(optics, filter_spec)

#             # Ensemble
#             @eqx.filter_jit
#             @eqx.filter_vmap(in_axes=(0, None))
#             def eval_polarised(polar_optics, null_optics):
#                 optics = eqx.combine(polar_optics, null_optics)
#                 wfs = eqx.filter_jit(optics.propagate)(wavels, pos, weights, return_wf=True)
#                 return wfs

#             wfs = eval_polarised(polarised, unpolarised)

#             from jax.nn import sigmoid

#             contrast = sigmoid(model.contrasts[exposure.filter])
#             contrasts = np.array([contrast, 1 - contrast])[:, None, None, None]
#             psfs = (contrasts * wfs.psf).sum(0)
#             pixel_scale = wfs.pixel_scale.mean((0, 1))

#         else:
#             wfs = eqx.filter_jit(optics.propagate)(wavels, pos, weights, return_wf=True)
#             pixel_scale = wfs.pixel_scale.mean(0)
#             psfs = wfs.psf

#         # Deal with potential unit issues from cartesian optical system required for fresnel
#         if wfs.units == "Cartesian":
#             pixel_scale /= optics.focal_length

#         # Convolutions done here
#         # psfs = wfs.psf
#         # pixel_scale = wfs.pixel_scale.mean(0)
#         if model.visibilities is not None:
#             vis_key = "_".join([exposure.star, exposure.filter])
#             amplitudes = model.amplitudes[vis_key]
#             phases = model.phases[vis_key]
#             jit_fn = eqx.filter_jit(model.visibilities.uv_model)
#             psf = jit_fn(psfs, amplitudes, phases, exposure.filter).sum(0)
#         else:
#             psf = psfs.sum(0)

#     # PSF is still unitary here
#     # psf = model.detector.apply(dl.PSF(psf, wfs.pixel_scale.mean(0)))
#     psf = eqx.filter_jit(model.detector.apply)(dl.PSF(psf, pixel_scale))

#     # Get the hyper-parameters for the non-linear model
#     flux = 10 ** model.fluxes[exposure.flux_key]
#     oversample = optics.oversample

#     # Return the BFE and required meta-data
#     if to_BFE:
#         return psf, flux, oversample

#     # Non linear model always goes from unit psf, flux, oversample to an 80x80 ramp
#     if model.ramp is not None:
#         # ramp = model.ramp.apply(psf, flux, exposure, oversample)
#         ramp = eqx.filter_jit(model.ramp.apply)(psf, flux, exposure, oversample)
#     else:
#         psf_data = dlu.downsample(psf.data * flux, oversample, mean=False)
#         ramp = psf.set("data", model_ramp(psf_data, exposure.ngroups))

#     # Model the read effects
#     if "one_on_fs" in model.params.keys():
#         model = model.set("read.one_on_fs", model.one_on_fs[exposure.key])
#     ramp = model.read.apply(ramp)

#     # Return the slopes if required
#     if slopes:
#         return np.diff(ramp.data, axis=0)

#     # Return the ramp
#     return ramp.data


def variance_model(model, exposure, true_read_noise=False, read_noise=10):
    """
    True read noise will use the CRDS read noise array, else it will use a constant
    value as determined by the input. true_read_noise therefore supersedes read_noise.
    Using a flat value of 10 seems to be more accurate that the CRDS array.

    That said I think the data has overly ambitious variances as a consequence of the
    sigma clipping that is performed. We could determine the variance analytically from
    the variance of the individual pixel values, but we will look at this later.
    """

    nan_mask = np.isnan(exposure.slopes)

    # Estimate the photon covariance
    # psf = model_fn(model, exposure)
    slopes = model.model(exposure)  # , slopes=True)

    slopes = slopes.at[np.where(nan_mask)].set(np.nan)
    variance = slopes / exposure.nints

    # Read noise covariance
    if true_read_noise:
        rn = np.load(pkg.resource_filename(__name__, "data/SUB80_readnoise.npy"))
    else:
        rn = read_noise
    read_variance = (rn**2) * np.ones((80, 80)) / exposure.nints
    variance += read_variance[None, ...]

    return slopes, variance
